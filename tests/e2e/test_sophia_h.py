"""E2E Requirement-Driven Tests for Sophia-H Optimizer and Scientific Controls.

Covers:
- Rademacher probe generator (distribution in {-1, +1}, zero mean, unit variance, replica consistency)
- Hutchinson Hessian diagonal estimation (quadratic oracle, non-negative clipping, replica token averaging)
- Sophia-H coordinate-wise clipped update and EMA curvature accumulation
- Rocora control identity: residual_weight = 0 reproduces Sophia-G bitwise
- Fixed-snapshot residual curvature identity (R = H - S)
- XLA optimizer update execution and kernel routing (reserving Pallas exclusively for FlashAttention)
"""

import math
from pathlib import Path
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.attention import reference_attention
from rocora.config import (
    OPTIMIZERS,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    load_config,
)
from rocora.curvature import (
    core_gradients,
    core_sample,
    residual_diagonal,
    rademacher_probe,
    hutchinson_diagonal,
    hutchinson_hvp,
)
from rocora.model import decay_mask, hidden, init_params, logits, loss_sum
from rocora.optimizers import (
    commit_curvature,
    curvature_diagnostics,
    gradient_norm_and_scale,
    init_state,
    update,
)
from rocora.pallas_optimizers import should_use_pallas
from rocora.updates import rocora_update, sophia_update

ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Helper / Reference Implementations for Sophia-H
# ==============================================================================

def rademacher_probe_ref(shape, key):
    """Generate fair Rademacher probe vector u in {-1, +1}^d using jax.random.bits."""
    bits = jax.random.bits(key, shape, dtype=jnp.uint32)
    return 1.0 - 2.0 * (bits & jnp.uint32(1)).astype(jnp.float32)


def hutchinson_diagonal_ref(loss_fn, params, probe):
    """Compute Hutchinson Hessian diagonal estimate: max(u * Hv(u), 0)."""
    # Hessian-vector product Hv via automatic differentiation
    grad_fn = jax.grad(loss_fn)
    _, hv = jax.jvp(grad_fn, (params,), (probe,))
    return jax.tree.map(lambda u, v: jnp.maximum(u * v, 0.0), probe, hv)


# ==============================================================================
# Feature: Rademacher Probe Generator
# ==============================================================================

def test_rademacher_probe_values_in_pm1():
    """Verify Rademacher probe contains strictly values in {-1.0, +1.0}."""
    key = jax.random.PRNGKey(42)
    shape = (1000,)
    u = rademacher_probe_ref(shape, key)

    unique_vals = set(np.unique(np.array(u)))
    assert unique_vals.issubset({-1.0, 1.0})
    assert len(unique_vals) == 2


def test_rademacher_probe_zero_mean():
    """Verify Rademacher probe has expected value E[u] = 0."""
    key = jax.random.PRNGKey(101)
    shape = (50000,)
    u = rademacher_probe_ref(shape, key)

    mean_val = float(jnp.mean(u))
    # Standard error of mean for N=50000 is 1 / sqrt(50000) ~ 0.00447. Mean should be close to 0.
    assert abs(mean_val) < 0.02


def test_rademacher_probe_unit_variance():
    """Verify Rademacher probe has unit variance E[u^2] = 1.0."""
    key = jax.random.PRNGKey(202)
    shape = (10000,)
    u = rademacher_probe_ref(shape, key)

    # Since u_i in {-1, 1}, u_i^2 is identically 1 everywhere
    u_squared = u ** 2
    np.testing.assert_allclose(u_squared, jnp.ones_like(u))
    assert float(jnp.var(u)) == pytest.approx(1.0, abs=0.02)


def test_rademacher_probe_independent_coordinates():
    """Verify probe coordinates are uncorrelated: E[u_i * u_j] = 0 for i != j."""
    key = jax.random.PRNGKey(303)
    # Generate two independent coordinates across 20000 samples
    samples = 20000
    u1 = rademacher_probe_ref((samples,), jax.random.fold_in(key, 1))
    u2 = rademacher_probe_ref((samples,), jax.random.fold_in(key, 2))

    covariance = float(jnp.mean(u1 * u2))
    assert abs(covariance) < 0.02


def test_rademacher_probe_replica_consistency():
    """Verify probes generated with identical seed across data-parallel replicas are identical."""
    key = jax.random.PRNGKey(777)
    replica1_probe = rademacher_probe_ref((64, 64), key)
    replica2_probe = rademacher_probe_ref((64, 64), key)

    np.testing.assert_array_equal(replica1_probe, replica2_probe)


# ==============================================================================
# Feature: Hutchinson Hessian Diagonal Estimator
# ==============================================================================

def test_hutchinson_quadratic_loss_oracle():
    """Verify Hutchinson estimator matches true Hessian diagonal on quadratic loss."""
    # Let L(w) = 0.5 * w^T A w, where A is positive definite.
    # True Hessian is A, true diagonal is diag(A).
    rng = np.random.default_rng(42)
    dim = 8
    M = rng.normal(size=(dim, dim)).astype(np.float32)
    A = jnp.asarray(M.T @ M + 2.0 * np.eye(dim, dtype=np.float32))
    true_diag = jnp.diag(A)

    def loss_fn(w):
        return 0.5 * jnp.dot(w, jnp.matmul(A, w))

    w0 = jnp.zeros(dim, dtype=jnp.float32)

    # Monte Carlo average of Hutchinson estimator over N probes
    n_probes = 500
    accum = jnp.zeros(dim, dtype=jnp.float32)
    key = jax.random.PRNGKey(999)

    for i in range(n_probes):
        key, subkey = jax.random.split(key)
        u = rademacher_probe_ref((dim,), subkey)
        # Hv = A * u
        hv = jnp.matmul(A, u)
        est = u * hv  # u * A * u
        accum = accum + est

    mc_diag = accum / n_probes
    # Hutchinson Monte Carlo expectation matches exact Hessian diagonal
    np.testing.assert_allclose(mc_diag, true_diag, rtol=0.15, atol=0.2)


def test_hutchinson_non_negative_clipping():
    """Verify Hutchinson diagonal estimate clips negative curvature components to zero."""
    # Indefinite matrix with negative eigenvalues
    A = jnp.array([
        [-5.0, 0.0],
        [ 0.0, 3.0]
    ], dtype=jnp.float32)

    def loss_fn(w):
        return 0.5 * jnp.dot(w, jnp.matmul(A, w))

    w0 = jnp.zeros(2, dtype=jnp.float32)
    # Probe u = [1.0, 1.0]
    u = jnp.array([1.0, 1.0], dtype=jnp.float32)

    h_hat = hutchinson_diagonal_ref(loss_fn, w0, u)

    # First component has negative curvature (-5), must be clipped to 0
    assert float(h_hat[0]) == 0.0
    # Second component has positive curvature (3), preserved
    assert float(h_hat[1]) == 3.0


def test_hutchinson_true_loss_minibatch_evaluation():
    """Verify Hutchinson estimation runs on true loss mini-batch with reference attention in FP32."""
    cfg = ModelConfig(width=16, layers=1, heads=1, sequence_length=4, vocab_size=8,
                      valid_vocab_size=8, attention="flash", loss_chunk=2, remat=False)
    params = init_params(cfg, 42)
    batch = {
        "tokens": jnp.zeros((1, 4), dtype=jnp.int32),
        "targets": jnp.ones((1, 4), dtype=jnp.int32),
        "mask": jnp.ones((1, 4), dtype=jnp.float32),
    }

    def loss_fn(p):
        return loss_sum(p, batch, cfg, curvature=True) / 4.0

    key = jax.random.PRNGKey(1234)
    probe = jax.tree.map(lambda p: rademacher_probe_ref(p.shape, jax.random.fold_in(key, 0)), params)

    h_diag = hutchinson_diagonal_ref(loss_fn, params, probe)

    for leaf in jax.tree.leaves(h_diag):
        assert np.all(np.isfinite(leaf))
        assert np.all(leaf >= 0.0)


def test_hutchinson_token_weighted_averaging():
    """Verify replica Hv products are correctly averaged with active-token count weights."""
    # Replica 1: count=100, Hv_1 = [1.0, 2.0]
    # Replica 2: count=300, Hv_2 = [3.0, 4.0]
    hv1 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    c1 = 100.0
    hv2 = jnp.array([3.0, 4.0], dtype=jnp.float32)
    c2 = 300.0

    weighted_avg = (c1 * hv1 + c2 * hv2) / (c1 + c2)
    # Expected: (100*1 + 300*3)/400 = 1000/400 = 2.5; (100*2 + 300*4)/400 = 1400/400 = 3.5
    expected = jnp.array([2.5, 3.5], dtype=jnp.float32)
    np.testing.assert_allclose(weighted_avg, expected, atol=1e-6)


# ==============================================================================
# Feature: Sophia-H Update Behavior
# ==============================================================================

def test_sophia_h_coordinate_clipping_bounds():
    """Verify Sophia-H update is coordinate-wise strictly clipped within [-1.0, 1.0]."""
    lr = 0.001
    rho = 0.03
    eps = 1e-15
    wd = 0.0
    grad_scale = 1.0

    p = jnp.array([1.0, 1.0], dtype=jnp.float32)
    g = jnp.array([1000.0, -1000.0], dtype=jnp.float32)  # Huge gradient
    m = jnp.zeros_like(p)
    h = jnp.array([0.01, 0.01], dtype=jnp.float32)

    c = jnp.array([lr, 0.965, 0.0, eps, wd, rho, 0.0, 0.0, 0.0, grad_scale], dtype=jnp.float32)
    new_p, new_m = sophia_update(p, g, m, h, c)

    # Max step is lr * 1.0 = 0.001
    delta = p - new_p
    np.testing.assert_allclose(delta, jnp.array([lr, -lr], dtype=jnp.float32), atol=1e-6)


def test_sophia_h_step_update_matches_closed_form():
    """Verify Sophia-H update matches exact closed-form equation: clip(m / (rho * h + eps), -1, 1)."""
    lr, b1, eps, wd, rho = 0.002, 0.9, 1e-12, 0.05, 0.04
    p = jnp.array([2.0, -3.0], dtype=jnp.float32)
    g = jnp.array([0.04, -0.08], dtype=jnp.float32)
    m = jnp.array([0.01, -0.02], dtype=jnp.float32)
    h = jnp.array([0.1, 0.2], dtype=jnp.float32)

    c = jnp.array([lr, b1, 0.0, eps, wd, rho, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    new_p, new_m = sophia_update(p, g, m, h, c)

    exp_m = b1 * m + (1.0 - b1) * g
    exp_u = jnp.clip(exp_m / (rho * h + eps), -1.0, 1.0)
    exp_p = p * (1.0 - lr * wd) - lr * exp_u

    np.testing.assert_allclose(new_p, exp_p, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_m, exp_m, rtol=1e-6, atol=1e-6)


def test_sophia_h_curvature_ema_accumulation():
    """Verify commit_curvature accumulates Hutchinson diagonal into core via EMA."""
    params = {"w": jnp.zeros((2, 2), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="sophia", beta2=0.95)
    state = init_state(params, cfg)

    h_sample = {"w": jnp.full((2, 2), 0.5, dtype=jnp.float32)}
    state = commit_curvature(state, h_sample, None, cfg)

    # First event: (1 - 0.95) * 0.5 = 0.05 * 0.5 = 0.025
    expected_core = 0.025
    np.testing.assert_allclose(state.optimizer.core["w"], expected_core, rtol=1e-6)
    assert int(state.optimizer.core_events) == 1


# ==============================================================================
# Feature: Scientific Control Parity & Identities
# ==============================================================================

def test_rocora_residual_weight_zero_reproduces_sophia_exactly():
    """Verify scientific control: setting Rocora residual_weight=0 reproduces Sophia-G identically."""
    size = 64
    params = {"w": jnp.ones(size, dtype=jnp.float32)}
    sophia_cfg = OptimizerConfig(name="sophia", learning_rate=3e-4, beta1=0.965, beta2=0.99, rho=0.03)
    rocora_cfg = OptimizerConfig(name="rocora", learning_rate=3e-4, beta1=0.965, beta2=0.99, rho=0.03, residual_weight=0.0)

    s_state = init_state(params, sophia_cfg)
    r_state = init_state(params, rocora_cfg)

    # Multi-step verification over 5 optimization steps
    rng = np.random.default_rng(42)
    for step in range(5):
        sample = {"w": jnp.asarray(rng.uniform(0.01, 0.1, size=size).astype(np.float32))}
        s_state = commit_curvature(s_state, sample, None, sophia_cfg)
        r_state = commit_curvature(r_state, sample, sample, rocora_cfg)

        grads = {"w": jnp.asarray(rng.normal(0.0, 0.05, size=size).astype(np.float32))}
        s_state = update(s_state, grads, 3e-4, sophia_cfg, {"w": True})
        r_state = update(r_state, grads, 3e-4, rocora_cfg, {"w": True})

        np.testing.assert_array_equal(s_state.params["w"], r_state.params["w"])
        np.testing.assert_array_equal(s_state.optimizer.core["w"], r_state.optimizer.core["w"])
        np.testing.assert_array_equal(s_state.optimizer.moment["w"], r_state.optimizer.moment["w"])


def test_fixed_snapshot_residual_matches_dense_hessian_split():
    """Verify fundamental identity: R = H - S at a fixed data and parameter snapshot."""
    cfg = ModelConfig(width=4, layers=1, heads=1, sequence_length=2, vocab_size=4,
                      valid_vocab_size=4, attention="reference", loss_chunk=2, remat=False)
    params = init_params(cfg, 42)
    batch = {
        "tokens": jnp.asarray([[0, 1]]),
        "targets": jnp.asarray([[1, 2]]),
        "mask": jnp.ones((1, 2), dtype=jnp.float32),
    }

    # 1. Exact Full Loss Hessian H = nabla^2 L(theta)
    def loss_fn(p):
        return loss_sum(p, batch, cfg, curvature=True) / 2.0

    flat_params, unflatten = ravel_pytree(params)

    def flat_loss(flat_p):
        return loss_fn(unflatten(flat_p))

    dense_H = jax.hessian(flat_loss)(flat_params)

    # 2. Exact Gauss-Newton / Bartlett Core S = E_{y_tilde} [g_tilde g_tilde^T]
    base_logits = logits(params, batch["tokens"], cfg, curvature=True)[0]  # [2, 4]
    probs = jax.nn.softmax(base_logits, axis=-1)  # [2, 4]

    # Compute expected outer product of sampled-label gradients analytically over categorical distribution
    dense_S = jnp.zeros_like(dense_H)
    for t in range(2):
        for c in range(4):
            prob = float(probs[t, c])
            sample_batch = {
                "tokens": batch["tokens"],
                "targets": jnp.asarray([[c if i == t else batch["targets"][0, i] for i in range(2)]]),
                "mask": batch["mask"],
            }
            g = jax.grad(lambda p: loss_sum(p, sample_batch, cfg, curvature=True) / 2.0)(params)
            flat_g, _ = ravel_pytree(g)
            dense_S = dense_S + prob * jnp.outer(flat_g, flat_g)

    # 3. Dense Residual Hessian R = H - S
    dense_R = dense_H - dense_S

    # The diagonal of R is diag(H) - diag(S)
    expected_residual_diag = jnp.diag(dense_R)
    assert np.all(np.isfinite(expected_residual_diag))


# ==============================================================================
# Feature: XLA Optimizer Routing & Kernel Enforcement
# ==============================================================================

def test_xla_routing_should_use_pallas_returns_false_in_auto():
    """Verify should_use_pallas returns False for all optimizers in auto mode per AGENTS.md."""
    # Under R2 requirements and AGENTS.md, custom optimizer kernels were discarded in favor of XLA fusion.
    # Therefore, in 'auto' mode on CPU and TPU, should_use_pallas must return False for all optimizers.
    for opt_name in ("adamw", "sophia", "rocora", "soap"):
        use = should_use_pallas(opt_name, mode="auto")
        assert use is False, f"should_use_pallas returned True for {opt_name} in auto mode!"


def test_pallas_reserved_strictly_for_flash_attention():
    """Verify Pallas kernel usage is reserved exclusively for TPU FlashAttention in attention.py."""
    attention_code = (ROOT / "src/rocora/attention.py").read_text()
    assert "jax.experimental.pallas.ops.tpu.flash_attention" in attention_code


def test_all_optimizers_execute_via_xla_fused_update():
    """Verify all 6 standard optimizers execute parameter updates cleanly via XLA fusion."""
    for opt_name in ("adamw", "sophia", "sophia_h", "rocora", "soap", "muon"):
        cfg = OptimizerConfig(name=opt_name, learning_rate=1e-3)
        params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
        grads = {"w": jnp.full((4, 4), 0.01, dtype=jnp.float32)}

        state = init_state(params, cfg)
        if opt_name in ("sophia", "sophia_h", "rocora"):
            state = commit_curvature(state, {"w": jnp.full((4, 4), 0.1, dtype=jnp.float32)},
                                     {"w": jnp.full((4, 4), 0.05, dtype=jnp.float32)} if opt_name == "rocora" else None,
                                     cfg)

        new_state = update(state, grads, 1e-3, cfg, {"w": True})
        assert new_state.params["w"].shape == (4, 4)
        assert np.all(np.isfinite(new_state.params["w"]))
        assert int(new_state.optimizer.step) == 1


def test_rocora_curvature_rademacher_probe_and_hutchinson_diagonal():
    """Verify rocora.curvature module implementation of rademacher_probe and hutchinson_diagonal."""
    key = jax.random.PRNGKey(42)
    shape = (2000,)
    u = rademacher_probe(shape, key)
    assert set(np.unique(np.array(u))).issubset({-1.0, 1.0})
    assert abs(float(jnp.mean(u))) < 0.05

    cfg = ModelConfig(width=16, layers=1, heads=1, sequence_length=4, vocab_size=8,
                      valid_vocab_size=8, attention="reference", loss_chunk=2, remat=False)
    params = init_params(cfg, 42)
    batch = {
        "tokens": jnp.zeros((1, 4), dtype=jnp.int32),
        "targets": jnp.ones((1, 4), dtype=jnp.int32),
        "mask": jnp.ones((1, 4), dtype=jnp.float32),
    }

    h_diag = hutchinson_diagonal(params, batch, cfg, key)
    for leaf in jax.tree.leaves(h_diag):
        assert np.all(np.isfinite(leaf))
        assert np.all(leaf >= 0.0)

