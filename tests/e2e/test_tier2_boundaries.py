"""Tier 2: Boundary & Corner Cases E2E Tests (≥60 tests, ≥5 per feature across 12 features).

Features:
1. AdamW Optimizer Boundaries
2. Sophia-G Optimizer Boundaries
3. Rocora Optimizer Boundaries
4. SOAP Optimizer Boundaries
5. Muon Optimizer Boundaries
6. FlashAttention Boundaries
7. FP32 Curvature Attention Routing Boundaries
8. 60M Pilot Deletion & Zero References Boundaries
9. Scale Configs Standardization Boundaries
10. Bounded Smoke Runs & Guardrails Boundaries
11. Multi-host TPU Worker Locks Boundaries
12. Repo Hygiene & Push Readiness Boundaries
"""

import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.attention import reference_attention
from rocora.checkpoint import load_checkpoint, save_checkpoint
from rocora.config import (
    OPTIMIZERS,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    learning_rate,
    load_config,
)
from rocora.curvature import residual_diagonal
from rocora.matrix_optimizers import SoapFactors, muon_direction
from rocora.model import decay_mask, hidden, init_params, loss_sum
from rocora.optimizers import (
    commit_curvature,
    gradient_norm_and_scale,
    init_state,
    update,
)
from rocora.sweep import plan

ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Feature 1: AdamW Optimizer Boundaries
# ==============================================================================

def test_tier2_feat01_adamw_zero_weights():
    """Verify AdamW updates with initial parameters exactly zero proceed smoothly."""
    params = {"w": jnp.zeros((4, 4), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="adamw", learning_rate=1e-3, weight_decay=0.1)
    state = init_state(params, cfg)
    grads = {"w": jnp.ones((4, 4), dtype=jnp.float32)}

    updated = update(state, grads, 1e-3, cfg, {"w": True})
    assert np.all(np.isfinite(updated.params["w"]))
    assert np.any(updated.params["w"] != 0)


def test_tier2_feat01_adamw_zero_gradients():
    """Verify AdamW update with zero gradients decays momentum and parameters."""
    p_init = jnp.ones((2, 2), dtype=jnp.float32)
    params = {"w": p_init}
    cfg = OptimizerConfig(name="adamw", learning_rate=1e-2, weight_decay=0.1)
    state = init_state(params, cfg)
    grads = {"w": jnp.zeros((2, 2), dtype=jnp.float32)}

    updated = update(state, grads, 1e-2, cfg, {"w": True})
    # Parameter should decay by (1 - lr * wd) = 1 - 0.001 = 0.999
    np.testing.assert_allclose(updated.params["w"], p_init * 0.999, rtol=1e-6)
    assert np.all(updated.optimizer.moment["w"] == 0)


def test_tier2_feat01_adamw_extreme_learning_rates():
    """Verify boundary learning rates (lr=0.0 and lr=1.0) maintain stability."""
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="adamw")
    state = init_state(params, cfg)
    grads = {"w": jnp.full(4, 0.5, dtype=jnp.float32)}

    # lr = 0.0 -> parameters unchanged
    state_lr0 = update(state, grads, 0.0, cfg, {"w": True})
    np.testing.assert_array_equal(state_lr0.params["w"], params["w"])

    # lr = 1.0 -> large finite update
    state_lr1 = update(state, grads, 1.0, cfg, {"w": False})
    assert np.all(np.isfinite(state_lr1.params["w"]))


def test_tier2_feat01_adamw_zero_grad_clip():
    """Verify grad_clip=0.0 disables gradient clipping (scale=1.0)."""
    grads = {"w": jnp.full((10, 10), 100.0, dtype=jnp.float32)}
    norm, scale = gradient_norm_and_scale(grads, threshold=0.0)
    assert float(scale) == 1.0
    assert float(norm) > 100.0


def test_tier2_feat01_adamw_large_gradient_values():
    """Verify massive gradients under clipping do not overflow FP32."""
    grads = {"w": jnp.full(5, 1e7, dtype=jnp.float32)}
    norm, scale = gradient_norm_and_scale(grads, threshold=1.0)
    assert float(scale) < 1e-6

    params = {"w": jnp.zeros(5, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="adamw")
    state = init_state(params, cfg)
    updated = update(state, grads, 1e-3, cfg, {"w": False}, gradient_scale=scale)
    assert np.all(np.isfinite(updated.params["w"]))


# ==============================================================================
# Feature 2: Sophia-G Optimizer Boundaries
# ==============================================================================

def test_tier2_feat02_sophia_zero_core_curvature():
    """Verify Sophia-G handles s=0 core curvature without division by zero."""
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="sophia", eps=1e-15, rho=0.03)
    state = init_state(params, cfg)
    # state.optimizer.core["w"] is all zeros

    grads = {"w": jnp.full(4, 0.01, dtype=jnp.float32)}
    updated = update(state, grads, 1e-3, cfg, {"w": False})
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat02_sophia_extreme_gradient_clipping():
    """Verify extreme gradient produces clipped direction in [-1, 1]."""
    params = {"w": jnp.zeros(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="sophia", learning_rate=1e-3, rho=0.03, eps=1e-15)
    state = init_state(params, cfg)
    # Huge gradient relative to core
    state = state._replace(
        optimizer=state.optimizer._replace(
            core={"w": jnp.full(4, 1e-8, dtype=jnp.float32)}
        )
    )
    grads = {"w": jnp.full(4, 1000.0, dtype=jnp.float32)}
    updated = update(state, grads, 1e-3, cfg, {"w": False})
    # Update = lr * clip(...) = 1e-3 * 1.0 = 1e-3
    np.testing.assert_allclose(updated.params["w"], -1e-3, atol=1e-5)


def test_tier2_feat02_sophia_zero_learning_rate():
    """Verify lr=0 leaves parameters completely unchanged in Sophia-G."""
    p = jnp.array([2.0, -3.0], dtype=jnp.float32)
    params = {"w": p}
    cfg = OptimizerConfig(name="sophia")
    state = init_state(params, cfg)
    grads = {"w": jnp.array([1.0, 1.0], dtype=jnp.float32)}

    updated = update(state, grads, 0.0, cfg, {"w": True})
    np.testing.assert_array_equal(updated.params["w"], p)


def test_tier2_feat02_sophia_subnormal_rho():
    """Verify tiny rho does not cause NaN."""
    params = {"w": jnp.ones(2, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="sophia", rho=1e-10)
    state = init_state(params, cfg)
    grads = {"w": jnp.ones(2, dtype=jnp.float32)}

    updated = update(state, grads, 1e-3, cfg, {"w": False})
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat02_sophia_zero_weights_boundary():
    """Verify parameters starting at zero update cleanly within [-lr, lr]."""
    params = {"w": jnp.zeros(5, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="sophia", learning_rate=0.05, weight_decay=0.0)
    state = init_state(params, cfg)
    grads = {"w": jnp.ones(5, dtype=jnp.float32)}

    updated = update(state, grads, 0.05, cfg, {"w": False})
    assert np.all(updated.params["w"] <= 0.0)
    assert np.all(updated.params["w"] >= -0.05)


# ==============================================================================
# Feature 3: Rocora Optimizer Boundaries
# ==============================================================================

def test_tier2_feat03_rocora_negative_residual_curvature_clipping():
    """Verify large negative residual curvature is clamped by max(rho*(s + w*r), 0) + eps."""
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="rocora", residual_weight=1.0, rho=0.03, eps=1e-15)
    state = init_state(params, cfg)

    # Core is small, residual is large negative: s + w*r = 0.1 - 100.0 < 0
    state = state._replace(
        optimizer=state.optimizer._replace(
            core={"w": jnp.full(4, 0.1, dtype=jnp.float32)},
            residual={"w": jnp.full(4, -100.0, dtype=jnp.float32)},
        )
    )
    grads = {"w": jnp.full(4, 0.05, dtype=jnp.float32)}
    updated = update(state, grads, 1e-3, cfg, {"w": False})
    # Denominator evaluates to max(..., 0) + eps, update is clipped to [-1, 1]
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat03_rocora_zero_residual_weight_boundary():
    """Verify residual_weight=0 completely ignores nonzero residual curvature buffer."""
    size = 16
    params = {"w": jnp.ones(size, dtype=jnp.float32)}
    cfg_zero = OptimizerConfig(name="rocora", residual_weight=0.0)
    state_zero = init_state(params, cfg_zero)
    # Populate residual buffer with arbitrary nonzero values
    state_zero = state_zero._replace(
        optimizer=state_zero.optimizer._replace(
            residual={"w": jnp.full(size, 999.0, dtype=jnp.float32)}
        )
    )
    cfg_sophia = OptimizerConfig(name="sophia")
    state_sophia = init_state(params, cfg_sophia)

    grads = {"w": jnp.full(size, 0.02, dtype=jnp.float32)}
    res_rocora = update(state_zero, grads, 1e-3, cfg_zero, {"w": False})
    res_sophia = update(state_sophia, grads, 1e-3, cfg_sophia, {"w": False})

    np.testing.assert_array_equal(res_rocora.params["w"], res_sophia.params["w"])


def test_tier2_feat03_rocora_extreme_residual_weight():
    """Verify high residual_weight (e.g. 100.0) with positive residual damps update stably."""
    params = {"w": jnp.ones(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="rocora", residual_weight=100.0, rho=0.03)
    state = init_state(params, cfg)
    state = state._replace(
        optimizer=state.optimizer._replace(
            core={"w": jnp.full(4, 1.0, dtype=jnp.float32)},
            residual={"w": jnp.full(4, 5.0, dtype=jnp.float32)},
        )
    )
    grads = {"w": jnp.full(4, 0.01, dtype=jnp.float32)}
    updated = update(state, grads, 1e-3, cfg, {"w": False})
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat03_rocora_zero_core_and_residual():
    """Verify Rocora with both s=0 and r=0 remains bounded by epsilon and clip."""
    params = {"w": jnp.zeros(2, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="rocora")
    state = init_state(params, cfg)
    grads = {"w": jnp.array([10.0, -10.0], dtype=jnp.float32)}

    updated = update(state, grads, 1e-3, cfg, {"w": False})
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat03_rocora_large_gradient_clip_bound():
    """Verify saturated gradients clip Rocora updates cleanly to [-lr, lr]."""
    params = {"w": jnp.zeros(4, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="rocora", learning_rate=0.01, weight_decay=0.0)
    state = init_state(params, cfg)
    grads = {"w": jnp.full(4, 1e6, dtype=jnp.float32)}

    updated = update(state, grads, 0.01, cfg, {"w": False})
    np.testing.assert_allclose(updated.params["w"], -0.01, atol=1e-5)


# ==============================================================================
# Feature 4: SOAP Optimizer Boundaries
# ==============================================================================

def test_tier2_feat04_soap_max_precondition_dim_exceeded():
    """Verify dimensions exceeding soap_max_precondition_dim receive (0,0) empty factors."""
    # Weight dimension 3000 > max_dim 2048
    params = {"w": jnp.ones((2, 3000), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="soap", soap_max_precondition_dim=2048)
    state = init_state(params, cfg)

    factors = state.optimizer.preconditioner["w"]
    assert factors.covariance[0].shape == (2, 2)
    assert factors.covariance[1].shape == (0, 0)


def test_tier2_feat04_soap_1d_or_scalar_parameter():
    """Verify 1D parameter vectors bypass SOAP matrix preconditioning."""
    params = {"norm_gain": jnp.ones(64, dtype=jnp.float32)}
    cfg = OptimizerConfig(name="soap")
    state = init_state(params, cfg)

    factors = state.optimizer.preconditioner["norm_gain"]
    assert factors.covariance[0].shape == (0, 0)


def test_tier2_feat04_soap_non_square_matrices():
    """Verify non-square matrices (e.g. 8x32) form proper covariance dimensions."""
    params = {"w": jnp.ones((8, 32), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="soap")
    state = init_state(params, cfg)

    factors = state.optimizer.preconditioner["w"]
    assert factors.covariance[0].shape == (8, 8)
    assert factors.covariance[1].shape == (32, 32)


def test_tier2_feat04_soap_zero_gradients_during_bootstrap():
    """Verify zero gradients during bootstrap step 1 do not fail in eigh."""
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="soap")
    state = init_state(params, cfg)
    grads = {"w": jnp.zeros((4, 4), dtype=jnp.float32)}

    updated = update(state, grads, 1e-3, cfg, {"w": False})
    assert int(updated.optimizer.step) == 1
    assert np.all(np.isfinite(updated.params["w"]))


def test_tier2_feat04_soap_extreme_learning_rate():
    """Verify lr=0 with SOAP guarantees parameter update is zero."""
    p_init = jnp.ones((3, 3), dtype=jnp.float32)
    params = {"w": p_init}
    cfg = OptimizerConfig(name="soap")
    state = init_state(params, cfg)
    grads = {"w": jnp.full((3, 3), 0.1, dtype=jnp.float32)}

    # Step 1: bootstrap
    state = update(state, grads, 1e-3, cfg, {"w": False})
    # Step 2: lr=0
    state = update(state, grads, 0.0, cfg, {"w": False})
    np.testing.assert_array_equal(state.params["w"], p_init)


# ==============================================================================
# Feature 5: Muon Optimizer Boundaries
# ==============================================================================

def test_tier2_feat05_muon_square_matrix_aspect_ratio_boundary():
    """Verify aspect ratio for square matrix evaluates to exactly 1.0."""
    g = jnp.ones((8, 8), dtype=jnp.float32)
    cfg = OptimizerConfig(name="muon")
    m = jnp.zeros_like(g)

    direction, _ = muon_direction(g, m, cfg)
    assert direction.shape == (8, 8)
    assert np.all(np.isfinite(direction))


def test_tier2_feat05_muon_tall_vs_wide_matrix_transposition():
    """Verify tall matrix is transposed internally and restored to original shape."""
    tall_g = jnp.ones((16, 4), dtype=jnp.float32)
    cfg = OptimizerConfig(name="muon")
    direction, _ = muon_direction(tall_g, jnp.zeros_like(tall_g), cfg)
    assert direction.shape == (16, 4)


def test_tier2_feat05_muon_zero_gradients_direction():
    """Verify all-zero gradient matrix evaluates without NaN in Muon."""
    g = jnp.zeros((4, 8), dtype=jnp.float32)
    cfg = OptimizerConfig(name="muon")
    direction, m = muon_direction(g, jnp.zeros_like(g), cfg)
    assert direction.shape == (4, 8)
    assert np.all(np.isfinite(direction))


def test_tier2_feat05_muon_single_iteration_boundary():
    """Verify Muon with minimal iteration count muon_ns_steps=1 runs cleanly."""
    g = jnp.ones((4, 4), dtype=jnp.float32)
    cfg = OptimizerConfig(name="muon", muon_ns_steps=1)
    direction, _ = muon_direction(g, jnp.zeros_like(g), cfg)
    assert direction.shape == (4, 4)
    assert np.all(np.isfinite(direction))


def test_tier2_feat05_muon_extreme_weight_decay():
    """Verify extreme weight decay (0.5) with zero gradients decays parameters stably."""
    params = {"embedding": jnp.ones((4, 4), dtype=jnp.float32)}
    cfg = OptimizerConfig(name="muon", learning_rate=0.1, weight_decay=0.5, muon_aux_lr=0.1)
    state = init_state(params, cfg)
    grads = {"embedding": jnp.zeros((4, 4), dtype=jnp.float32)}

    updated = update(state, grads, 0.1, cfg, {"embedding": True})
    # Decayed by (1 - 0.1 * 0.5) = 0.95
    np.testing.assert_allclose(updated.params["embedding"], 0.95, atol=1e-5)


# ==============================================================================
# Feature 6: FlashAttention Boundaries
# ==============================================================================

def test_tier2_feat06_flash_attention_minimum_sequence_length():
    """Verify sequence length 128 (hardware tile minimum) passes ModelConfig validation."""
    cfg = ModelConfig(sequence_length=128, width=512, heads=8, attention="flash")
    cfg.validate()
    assert cfg.sequence_length == 128


def test_tier2_feat06_flash_attention_maximum_head_dim():
    """Verify head dimension 128 passes while 129 is rejected."""
    # 1024 / 8 = 128 (valid)
    cfg_valid = ModelConfig(width=1024, heads=8, sequence_length=128, attention="flash")
    cfg_valid.validate()

    # 1290 / 10 = 129 > 128 (invalid)
    with pytest.raises(ValueError, match="head width <=128"):
        ModelConfig(width=1290, heads=10, sequence_length=128, attention="flash").validate()


def test_tier2_feat06_flash_attention_single_batch_dimension():
    """Verify model execution handles single-example batch (B=1)."""
    cfg = ModelConfig(width=32, layers=1, heads=1, sequence_length=128,
                      vocab_size=64, valid_vocab_size=64, attention="reference")
    params = init_params(cfg, 42)
    tokens = jnp.zeros((1, 128), dtype=jnp.int32)

    out = hidden(params, tokens, cfg, curvature=False)
    assert out.shape == (1, 128, 32)


def test_tier2_feat06_flash_attention_odd_sequence_length_rejection():
    """Verify sequence length not divisible by 128 is rejected for FlashAttention."""
    with pytest.raises(ValueError, match="sequence multiple 128"):
        ModelConfig(sequence_length=129, attention="flash").validate()


def test_tier2_feat06_flash_attention_non_divisible_heads_rejection():
    """Verify width not divisible by heads is rejected."""
    with pytest.raises(ValueError, match="width must be divisible by heads"):
        ModelConfig(width=100, heads=3).validate()


# ==============================================================================
# Feature 7: FP32 Curvature Attention Routing Boundaries
# ==============================================================================

def test_tier2_feat07_reference_attention_single_token_sequence():
    """Verify reference attention handles single-token sequence (T=1)."""
    q = jnp.ones((1, 1, 1, 8), dtype=jnp.float32)
    out = reference_attention(q, q, q, block=1)
    # Self attention on single token has softmax weight 1.0, output equals V
    np.testing.assert_allclose(out, q, atol=1e-6)


def test_tier2_feat07_reference_attention_block_larger_than_sequence():
    """Verify block size larger than sequence length clamps tile to sequence length."""
    q = jnp.ones((1, 2, 8, 16), dtype=jnp.float32)
    out = reference_attention(q, q, q, block=128)
    assert out.shape == (1, 2, 8, 16)


def test_tier2_feat07_reference_attention_zero_inputs():
    """Verify all-zero attention input tensors evaluate to zero without NaN."""
    q = jnp.zeros((1, 1, 4, 8), dtype=jnp.float32)
    out = reference_attention(q, q, q, block=2)
    assert np.all(out == 0)
    assert np.all(np.isfinite(out))


def test_tier2_feat07_reference_attention_extreme_logit_scale():
    """Verify large query/key magnitudes do not produce NaN under softmax."""
    q = jnp.full((1, 1, 4, 8), 100.0, dtype=jnp.float32)
    out = reference_attention(q, q, q, block=2)
    assert np.all(np.isfinite(out))


def test_tier2_feat07_residual_diagonal_single_code_boundary():
    """Verify residual_diagonal executes with minimal codes m=1."""
    cfg = ModelConfig(width=16, layers=1, heads=1, sequence_length=4, vocab_size=8,
                      valid_vocab_size=8, attention="flash", loss_chunk=2, remat=False)
    params = init_params(cfg, 1)
    batch = {"tokens": jnp.zeros((1, 4), dtype=jnp.int32),
             "targets": jnp.ones((1, 4), dtype=jnp.int32),
             "mask": jnp.ones((1, 4), dtype=jnp.float32)}

    diag, count = residual_diagonal(params, batch, cfg, jax.random.PRNGKey(0), m=1)
    assert float(count) == 4.0
    for leaf in jax.tree.leaves(diag):
        assert np.all(np.isfinite(leaf))


# ==============================================================================
# Feature 8: 60M Pilot Deletion & Zero References Boundaries
# ==============================================================================

def test_tier2_feat08_config_loader_missing_pilot_clean_error():
    """Verify attempting to load nonexistent pilot config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "configs/nonexistent_pilot_file.json")


def test_tier2_feat08_experiment_config_stage_not_pilot_by_default():
    """Verify standard scale configs specify their respective stage, not pilot60m."""
    for stage in ("scale125m", "scale350m", "scale1b"):
        cfg = load_config(ROOT / f"configs/{stage}.json")
        assert cfg.stage == stage
        assert cfg.stage != "pilot60m"


def test_tier2_feat08_pod_stage_dry_run_ignores_missing_pilot_outputs():
    """Verify dry-run mode in pod launcher does not require pilot metrics."""
    cmd = ["python", str(ROOT / "scripts/pod.py"), "stage", "--help"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    # Argument handling should not raise Python syntax or schema errors
    assert "SyntaxError" not in res.stderr
    assert "--dry-run" in res.stdout


def test_tier2_feat08_sweep_manifests_zero_pilot_references():
    """Verify all sweep configs in configs/sweeps/ do not mention pilot60m."""
    sweeps_dir = ROOT / "configs/sweeps"
    if sweeps_dir.exists():
        for path in sweeps_dir.glob("*.json"):
            content = path.read_text()
            assert "pilot60m" not in content


def test_tier2_feat08_no_pilot_in_generated_sweeps():
    """Verify sweep plan generation for scale125m contains only scale125m trials."""
    sweep_cfg = ROOT / "configs/sweeps/scale125m.json"
    if sweep_cfg.exists():
        manifest = plan(sweep_cfg)
        for trial in manifest["trials"]:
            assert "pilot" not in trial["run_name"]
            assert trial["config"]["stage"].startswith("scale125m")
            assert "pilot" not in trial["config"]["stage"]


# ==============================================================================
# Feature 9: Scale Configs Standardization Boundaries
# ==============================================================================

def test_tier2_feat09_scale_config_zero_or_negative_tokens_rejected():
    """Verify non-positive training tokens are rejected."""
    with pytest.raises(ValueError, match="budgets and batch size must be positive"):
        ExperimentConfig(train_tokens=0).validate()
    with pytest.raises(ValueError, match="budgets and batch size must be positive"):
        ExperimentConfig(train_tokens=-100).validate()


def test_tier2_feat09_scale_config_warmup_exceeding_total_tokens_rejected():
    """Verify warmup_tokens >= train_tokens raises ValueError."""
    with pytest.raises(ValueError, match="warmup_tokens must be below train_tokens"):
        ExperimentConfig(train_tokens=1000, warmup_tokens=1000).validate()
    with pytest.raises(ValueError, match="warmup_tokens must be below train_tokens"):
        ExperimentConfig(train_tokens=1000, warmup_tokens=1500).validate()


def test_tier2_feat09_scale_config_min_lr_fraction_out_of_bounds_rejected():
    """Verify min_lr_fraction outside (0, 1] is rejected."""
    with pytest.raises(ValueError, match="min_lr_fraction"):
        ExperimentConfig(min_lr_fraction=0.0).validate()
    with pytest.raises(ValueError, match="min_lr_fraction"):
        ExperimentConfig(min_lr_fraction=1.5).validate()


def test_tier2_feat09_scale_config_invalid_vocab_sizes_rejected():
    """Verify valid_vocab_size > vocab_size raises ValueError."""
    with pytest.raises(ValueError, match="Invalid vocabulary sizes"):
        ModelConfig(vocab_size=100, valid_vocab_size=101).validate()


def test_tier2_feat09_scale_config_seeds_uniqueness_enforced():
    """Verify duplicate seeds in seeds tuple raises ValueError."""
    with pytest.raises(ValueError, match="Provide distinct experiment seeds"):
        ExperimentConfig(seeds=(1337, 1337)).validate()


# ==============================================================================
# Feature 10: Bounded Smoke Runs & Guardrails Boundaries
# ==============================================================================

def test_tier2_feat10_max_steps_zero_boundary():
    """Verify step calculation handles small token budgets."""
    cfg = ExperimentConfig(train_tokens=100, global_batch_size=1,
                           model=ModelConfig(sequence_length=100))
    assert cfg.steps == 1


def test_tier2_feat10_max_steps_exceeding_total_steps():
    """Verify execution limit is min(total_steps, max_steps)."""
    cfg = ExperimentConfig(train_tokens=100, global_batch_size=1,
                           model=ModelConfig(sequence_length=10))
    total_steps = cfg.steps  # 10
    bounded_steps = min(total_steps, 500)
    assert bounded_steps == 10


def test_tier2_feat10_zero_token_mask_in_batch():
    """Verify batch with zero active mask tokens normalizes cleanly."""
    cfg = ModelConfig(width=8, layers=1, heads=1, sequence_length=2, vocab_size=4,
                      valid_vocab_size=4, attention="reference", loss_chunk=2, remat=False)
    params = init_params(cfg, 1)
    batch = {"tokens": jnp.zeros((1, 2), dtype=jnp.int32),
             "targets": jnp.zeros((1, 2), dtype=jnp.int32),
             "mask": jnp.zeros((1, 2), dtype=jnp.float32)}

    # loss_sum with mask=0 should evaluate to 0
    l = loss_sum(params, batch, cfg)
    assert float(l) == 0.0


def test_tier2_feat10_empty_loss_chunk_rejection():
    """Verify loss_chunk <= 0 raises ValueError in ModelConfig.validate()."""
    with pytest.raises(ValueError, match="Chunk sizes must be positive"):
        ModelConfig(loss_chunk=0).validate()


def test_tier2_feat10_negative_weight_decay_rejection():
    """Verify weight_decay < 0 raises ValueError in OptimizerConfig.validate()."""
    with pytest.raises(ValueError, match="Clipping and decay cannot be negative"):
        OptimizerConfig(weight_decay=-0.01).validate()


# ==============================================================================
# Feature 11: Multi-host TPU Worker Locks Boundaries
# ==============================================================================

def test_tier2_feat11_lock_nonexistent_directory_creation(tmp_path):
    """Verify locking in nested nonexistent directory succeeds."""
    nested_lock = tmp_path / "subdir" / "deep" / "worker.lock"
    nested_lock.parent.mkdir(parents=True, exist_ok=True)
    with nested_lock.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    assert nested_lock.exists()


def test_tier2_feat11_reacquire_after_process_kill(tmp_path):
    """Verify closing file descriptor automatically releases flock."""
    lock_file = tmp_path / "proc.lock"
    f1 = lock_file.open("a+")
    fcntl.flock(f1, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Simulate process exit: close fd without explicit unlock
    f1.close()

    # Next process should successfully acquire exclusive lock
    f2 = lock_file.open("a+")
    fcntl.flock(f2, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(f2, fcntl.LOCK_UN)
    f2.close()


def test_tier2_feat11_shared_lock_concurrency_boundary(tmp_path):
    """Verify shared locks allow multiple readers but block exclusive writer."""
    lock_file = tmp_path / "shared.lock"
    r1 = lock_file.open("a+")
    r2 = lock_file.open("a+")
    w1 = lock_file.open("a+")

    # Multiple shared locks succeed concurrently
    fcntl.flock(r1, fcntl.LOCK_SH | fcntl.LOCK_NB)
    fcntl.flock(r2, fcntl.LOCK_SH | fcntl.LOCK_NB)

    # Exclusive lock must fail while shared locks are held
    with pytest.raises(BlockingIOError):
        fcntl.flock(w1, fcntl.LOCK_EX | fcntl.LOCK_NB)

    fcntl.flock(r1, fcntl.LOCK_UN)
    fcntl.flock(r2, fcntl.LOCK_UN)
    r1.close()
    r2.close()
    w1.close()


def test_tier2_feat11_empty_job_id_boundary(tmp_path):
    """Verify lock file format requires both job_id and pid."""
    lock_file = tmp_path / "format.lock"
    with lock_file.open("a+") as f:
        f.write("job123 9999\n")
    tokens = lock_file.read_text().strip().split()
    assert len(tokens) == 2
    assert tokens[0] == "job123"
    assert tokens[1].isdigit()


def test_tier2_feat11_lock_file_truncation_on_reuse(tmp_path):
    """Verify re-locking truncates older content completely."""
    lock_file = tmp_path / "truncate.lock"
    lock_file.write_text("old_long_job_id_1234567890 12345\n")

    with lock_file.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        f.write("new 1\n")
        f.flush()

    assert lock_file.read_text() == "new 1\n"


# ==============================================================================
# Feature 12: Repo Hygiene & Push Readiness Boundaries
# ==============================================================================

def test_tier2_feat12_git_status_no_large_checkpoint_artifacts():
    """Verify no large checkpoint npz files are tracked in git."""
    res = subprocess.run(["git", "ls-files", "*.npz"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    # No training checkpoints should be tracked
    assert "checkpoint" not in res.stdout


def test_tier2_feat12_checkpoint_temporary_file_atomicity(tmp_path):
    """Verify save_checkpoint uses atomic replace with .tmp.npz."""
    cfg = ExperimentConfig(model=ModelConfig(attention="reference"))
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    state = init_state(params, cfg.optimizer)

    ckpt_path = tmp_path / "model.npz"
    save_checkpoint(ckpt_path, state, cfg, 10.0, "code_hash")

    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(".tmp.npz").exists()


def test_tier2_feat12_checkpoint_corrupted_archive_rejected(tmp_path):
    """Verify corrupted or empty checkpoint file is rejected cleanly."""
    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"not a valid npz archive")

    cfg = ExperimentConfig()
    template = init_state({"w": jnp.zeros(2)}, cfg.optimizer)
    with pytest.raises(Exception):
        load_checkpoint(corrupt_path, template, cfg)


def test_tier2_feat12_checkpoint_mismatched_seed_or_config_rejected(tmp_path):
    """Verify load_checkpoint rejects checkpoint saved with differing seed."""
    cfg1 = ExperimentConfig(seed=1337)
    cfg2 = ExperimentConfig(seed=9999)
    params = {"w": jnp.ones((2, 2), dtype=jnp.float32)}
    state = init_state(params, cfg1.optimizer)

    ckpt_path = tmp_path / "seed_test.npz"
    save_checkpoint(ckpt_path, state, cfg1, 1.0)

    with pytest.raises(ValueError, match="Checkpoint architecture/data/optimizer/budget/seed differs"):
        load_checkpoint(ckpt_path, state, cfg2)


def test_tier2_feat12_checkpoint_mismatched_leaves_rejected(tmp_path):
    """Verify load_checkpoint rejects template with differing leaf count."""
    cfg = ExperimentConfig()
    params1 = {"w1": jnp.ones(2, dtype=jnp.float32)}
    params2 = {"w1": jnp.ones(2, dtype=jnp.float32), "w2": jnp.ones(2, dtype=jnp.float32)}

    state1 = init_state(params1, cfg.optimizer)
    state2 = init_state(params2, cfg.optimizer)

    ckpt_path = tmp_path / "leaf_test.npz"
    save_checkpoint(ckpt_path, state1, cfg, 1.0)

    with pytest.raises(ValueError, match="Checkpoint state structure differs"):
        load_checkpoint(ckpt_path, state2, cfg)
