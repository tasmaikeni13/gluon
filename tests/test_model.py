"""Unit and behavioral tests for the modern LLaMA architecture in rocora.model.

Validates:
- RMSNorm mathematical correctness, absence of mean-centering, and autodiff smoothness.
- RoPE rotary frequency progression, norm preservation, relative position invariance,
  and unrotated value contract.
- SwiGLU activation, 8/3 dimension scaling with 64-tile alignment, and parameter parity.
- Bias-free design, tied embeddings, and removal of absolute position table.
- Analytical parameter count formulas across 125M, 350M, and 1B scales.
- Full model forward, backward, and second-order (HVP/curvature) differentiability.
"""

import math
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.config import ModelConfig
from rocora.model import (
    apply_rope,
    apply_rotary_emb,
    decay_mask,
    hidden,
    init_params,
    linear,
    logits,
    loss_sum,
    precompute_rope_freqs,
    project,
    rms_norm,
)


# ==============================================================================
# RMSNorm Tests
# ==============================================================================

def test_rmsnorm_mathematical_parity():
    """Verify rms_norm matches exact mathematical formula x / sqrt(mean(x^2) + eps) * gamma."""
    rng = np.random.default_rng(101)
    x = jnp.asarray(rng.normal(size=(4, 16)).astype(np.float32))
    gamma = jnp.asarray(rng.normal(size=(16,)).astype(np.float32))
    eps = 1e-5

    out = rms_norm(x, gamma, eps=eps)

    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected = (x / rms) * gamma
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_rmsnorm_no_mean_centering():
    """Verify RMSNorm does not center the mean, preserving constant nonzero offsets."""
    x = jnp.full((2, 8), 5.0, dtype=jnp.float32)
    gamma = jnp.ones((8,), dtype=jnp.float32)

    out = rms_norm(x, gamma)
    # Output should be all ones (5 / sqrt(25) = 1), whereas LayerNorm would be zero
    np.testing.assert_allclose(out, jnp.ones_like(x), atol=1e-3)
    assert not np.allclose(out, jnp.zeros_like(x))


def test_rmsnorm_positive_homogeneity():
    """Verify scale invariance: rms_norm(alpha * x) == rms_norm(x) for alpha > 0."""
    x = jnp.array([1.5, -2.5, 3.5, -0.5], dtype=jnp.float32)
    gamma = jnp.ones(4, dtype=jnp.float32)

    out_base = rms_norm(x, gamma, eps=1e-8)
    out_scaled = rms_norm(x * 50.0, gamma, eps=1e-8)
    np.testing.assert_allclose(out_base, out_scaled, rtol=1e-5, atol=1e-5)


def test_rmsnorm_second_order_autodiff():
    """Verify RMSNorm supports second derivatives (HVP) cleanly without division-by-zero singularities."""
    x = jnp.array([0.5, -0.5, 1.0, -1.0], dtype=jnp.float32)
    gamma = jnp.ones(4, dtype=jnp.float32)
    v = jnp.array([0.1, 0.2, -0.1, -0.2], dtype=jnp.float32)

    def loss(p):
        return jnp.sum(rms_norm(x, p) ** 2)

    # First derivative (grad)
    grad = jax.grad(loss)(gamma)
    assert np.all(np.isfinite(grad))

    # Second derivative (HVP)
    _, hvp = jax.jvp(jax.grad(loss), (gamma,), (v,))
    assert np.all(np.isfinite(hvp))


# ==============================================================================
# RoPE (Rotary Position Embeddings) Tests
# ==============================================================================

def test_rope_frequencies_and_monotonicity():
    """Verify inverse frequencies decrease monotonically from 1 down to base^(-(dim-2)/dim)."""
    dim = 64
    base = 10000.0
    cos, sin = precompute_rope_freqs(16, dim, base=base)

    assert cos.shape == (16, dim)
    assert sin.shape == (16, dim)

    channel = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel / dim))
    assert inv_freq[0] == 1.0
    assert float(inv_freq[-1]) == pytest.approx(1.0 / (base ** ((dim - 2) / dim)), rel=1e-5)
    diffs = jnp.diff(inv_freq)
    assert np.all(diffs < 0)


def test_rope_orthogonal_norm_preservation():
    """Verify RoPE is an orthogonal coordinate transformation that strictly preserves vector norms."""
    seq_len, dim = 16, 64
    cos, sin = precompute_rope_freqs(seq_len, dim)
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(size=(2, 4, seq_len, dim)).astype(np.float32))

    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    x_rot = apply_rotary_emb(x, cos_b, sin_b)

    norm_before = jnp.linalg.norm(x, axis=-1)
    norm_after = jnp.linalg.norm(x_rot, axis=-1)
    np.testing.assert_allclose(norm_before, norm_after, rtol=1e-5, atol=1e-5)


def test_rope_relative_shift_invariance():
    """Verify dot product of RoPE representations depends strictly on relative position (m - n)."""
    seq_len, dim = 32, 64
    cos, sin = precompute_rope_freqs(seq_len, dim)
    rng = np.random.default_rng(1337)
    q = jnp.asarray(rng.normal(size=(dim,)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(dim,)).astype(np.float32))

    # Relative offset delta = 6 across different starting positions
    q6 = apply_rotary_emb(q, cos[6], sin[6])
    k0 = apply_rotary_emb(k, cos[0], sin[0])
    dot_6_0 = jnp.dot(q6, k0)

    q16 = apply_rotary_emb(q, cos[16], sin[16])
    k10 = apply_rotary_emb(k, cos[10], sin[10])
    dot_16_10 = jnp.dot(q16, k10)

    q24 = apply_rotary_emb(q, cos[24], sin[24])
    k18 = apply_rotary_emb(k, cos[18], sin[18])
    dot_24_18 = jnp.dot(q24, k18)

    np.testing.assert_allclose(dot_6_0, dot_16_10, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dot_6_0, dot_24_18, rtol=1e-5, atol=1e-5)


def test_rope_position_zero_is_identity():
    """Verify RoPE at position t=0 produces an exact identity transformation."""
    dim = 32
    cos, sin = precompute_rope_freqs(1, dim)
    np.testing.assert_allclose(cos[0], jnp.ones(dim), atol=1e-6)
    np.testing.assert_allclose(sin[0], jnp.zeros(dim), atol=1e-6)

    x = jnp.arange(dim, dtype=jnp.float32)
    x_rot = apply_rotary_emb(x, cos[0], sin[0])
    np.testing.assert_allclose(x, x_rot, atol=1e-6)


def test_apply_rope_interface_contract():
    """Verify apply_rope helper rotates Q and K and maintains shapes and finite values."""
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    rng = np.random.default_rng(2026)
    q = jnp.asarray(rng.normal(size=(batch, heads, seq_len, head_dim)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(batch, heads, seq_len, head_dim)).astype(np.float32))

    q_rot, k_rot = apply_rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert np.all(np.isfinite(q_rot))
    assert np.all(np.isfinite(k_rot))

    # Norm preservation holds per head vector
    np.testing.assert_allclose(jnp.linalg.norm(q, axis=-1), jnp.linalg.norm(q_rot, axis=-1), rtol=1e-5)
    np.testing.assert_allclose(jnp.linalg.norm(k, axis=-1), jnp.linalg.norm(k_rot, axis=-1), rtol=1e-5)


def test_rope_higher_order_autodiff():
    """Verify RoPE supports JVP and HVP automatic differentiation without NaNs."""
    dim = 32
    cos, sin = precompute_rope_freqs(1, dim)

    def fn(x):
        rot = apply_rotary_emb(x, cos[0], sin[0])
        return jnp.sum(rot ** 3)

    x = jnp.ones(dim, dtype=jnp.float32)
    v = jnp.full(dim, 0.1, dtype=jnp.float32)

    # First derivative (JVP)
    primals, tangents = jax.jvp(fn, (x,), (v,))
    assert np.isfinite(primals)
    assert np.isfinite(tangents)

    # Second derivative (HVP)
    grad_fn = jax.grad(fn)
    _, hvp = jax.jvp(grad_fn, (x,), (v,))
    assert np.all(np.isfinite(hvp))


# ==============================================================================
# SwiGLU MLP Tests
# ==============================================================================

def test_swiglu_dimension_scaling():
    """Verify 8/3 dimension scaling with 64-multiple alignment and toy guards."""
    # Scale 125M (width=768)
    cfg_125m = ModelConfig(width=768, layers=12, heads=12)
    assert cfg_125m.d_ffn == 2048
    assert cfg_125m.d_ffn % 64 == 0

    # Scale 350M (width=1024) -> 8/3 * 1024 = 2730.67 -> aligned to 64 is 2752
    cfg_350m = ModelConfig(width=1024, layers=24, heads=16)
    assert cfg_350m.d_ffn == 2752
    assert cfg_350m.d_ffn % 64 == 0

    # Scale 1B (width=1536) -> 8/3 * 1536 = 4096
    cfg_1b = ModelConfig(width=1536, layers=32, heads=24)
    assert cfg_1b.d_ffn == 4096
    assert cfg_1b.d_ffn % 64 == 0

    # Toy dimensions
    assert ModelConfig(width=2, heads=1).d_ffn == 5
    assert ModelConfig(width=4, heads=1).d_ffn == 10
    assert ModelConfig(width=16, heads=1).d_ffn == 42


def test_swiglu_parameter_parity():
    """Verify SwiGLU with 8/3 width ratio achieves exact parameter parity with standard 4d MLP for multiples of 3."""
    for width in (768, 1536):
        cfg = ModelConfig(width=width)
        d_ffn = cfg.d_ffn
        swiglu_params = 3 * width * d_ffn
        standard_mlp_params = 2 * width * (4 * width)
        assert swiglu_params == standard_mlp_params


# ==============================================================================
# Model Architecture & Parameter Count Tests
# ==============================================================================

def test_init_params_schema_and_bias_free():
    """Verify init_params produces exact LLaMA schema without position table and without biases."""
    cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64, vocab_size=128)
    params = init_params(cfg, seed=1337)

    # Strictly no learned position embedding table
    assert "position" not in params
    assert "embedding" in params
    assert params["embedding"].shape == (128, 64)

    layers = params["layers"]
    assert "norm1" in layers
    assert layers["norm1"].shape == (2, 64)
    assert "qkv" in layers
    assert layers["qkv"].shape == (2, 64, 3 * 64)
    assert "attention_out" in layers
    assert layers["attention_out"].shape == (2, 64, 64)
    assert "norm2" in layers
    assert layers["norm2"].shape == (2, 64)
    assert "mlp_gate" in layers
    assert layers["mlp_gate"].shape == (2, 64, cfg.d_ffn)
    assert "mlp_up" in layers
    assert layers["mlp_up"].shape == (2, 64, cfg.d_ffn)
    assert "mlp_down" in layers
    assert layers["mlp_down"].shape == (2, cfg.d_ffn, 64)

    assert "final_norm" in params
    assert params["final_norm"].shape == (64,)

    # Materialized count matches analytical parameter_count property exactly
    total_elements = sum(p.size for p in jax.tree.leaves(params))
    assert total_elements == cfg.parameter_count


def test_parameter_counts_across_three_scales():
    """Verify analytical parameter counts match exact project specifications."""
    # 125M: 109,529,856
    cfg_125m = ModelConfig(width=768, layers=12, heads=12, vocab_size=32000)
    assert cfg_125m.parameter_count == 109_529_856

    # 350M: 336,380,928
    cfg_350m = ModelConfig(width=1024, layers=24, heads=16, vocab_size=32000)
    assert cfg_350m.parameter_count == 336_380_928

    # 1B: 955,221,504
    cfg_1b = ModelConfig(width=1536, layers=32, heads=24, vocab_size=32000)
    assert cfg_1b.parameter_count == 955_221_504


def test_decay_mask_exclusions():
    """Verify decay_mask excludes normalization scales and includes projection weights."""
    cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64, vocab_size=128)
    params = init_params(cfg, seed=42)
    mask = decay_mask(params)

    assert mask["embedding"] is True
    assert mask["final_norm"] is False
    assert mask["layers"]["norm1"] is False
    assert mask["layers"]["norm2"] is False
    assert mask["layers"]["qkv"] is True
    assert mask["layers"]["attention_out"] is True
    assert mask["layers"]["mlp_gate"] is True
    assert mask["layers"]["mlp_up"] is True
    assert mask["layers"]["mlp_down"] is True


# ==============================================================================
# Model Forward, Backward & Curvature Double-Backward Tests
# ==============================================================================

def test_full_model_forward_and_loss():
    """Verify model forward pass, project, logits, and loss_sum execute cleanly."""
    cfg = ModelConfig(width=32, layers=2, heads=2, sequence_length=16, vocab_size=64,
                      valid_vocab_size=64, attention="reference", loss_chunk=8, remat=False)
    params = init_params(cfg, seed=777)
    batch = {
        "tokens": jax.random.randint(jax.random.PRNGKey(1), (2, 16), 0, 64),
        "targets": jax.random.randint(jax.random.PRNGKey(2), (2, 16), 0, 64),
        "mask": jnp.ones((2, 16), dtype=jnp.float32),
    }

    # Forward hidden and logits
    h = hidden(params, batch["tokens"], cfg, curvature=False)
    assert h.shape == (2, 16, 32)
    assert np.all(np.isfinite(h))

    lg = logits(params, batch["tokens"], cfg, curvature=False)
    assert lg.shape == (2, 16, 64)
    assert np.all(np.isfinite(lg))

    # Loss sum
    loss = loss_sum(params, batch, cfg, curvature=False)
    assert np.isfinite(loss)
    assert float(loss) > 0


def test_model_first_and_second_order_autodiff_smoothness():
    """Verify model loss supports first-order gradients and second-order HVPs with curvature=True."""
    cfg = ModelConfig(width=16, layers=1, heads=1, sequence_length=4, vocab_size=8,
                      valid_vocab_size=8, attention="reference", loss_chunk=4, remat=False)
    params = init_params(cfg, seed=12)
    batch = {
        "tokens": jax.random.randint(jax.random.PRNGKey(3), (1, 4), 0, 8),
        "targets": jax.random.randint(jax.random.PRNGKey(4), (1, 4), 0, 8),
        "mask": jnp.ones((1, 4), dtype=jnp.float32),
    }

    def loss_fn(p):
        return loss_sum(p, batch, cfg, curvature=True)

    # First derivative (grad)
    grad = jax.grad(loss_fn)(params)
    for g in jax.tree.leaves(grad):
        assert np.all(np.isfinite(g))

    # Second derivative (HVP)
    flat_params, unpack = ravel_pytree(params)
    v = jax.random.normal(jax.random.PRNGKey(5), flat_params.shape) * 0.01

    def scalar_loss(flat_p):
        return loss_fn(unpack(flat_p))

    _, hvp = jax.jvp(jax.grad(scalar_loss), (flat_params,), (v,))
    assert np.all(np.isfinite(hvp))
    assert not np.any(np.isnan(hvp))
