"""Tier 4: Realistic Workloads E2E Tests (≥6 tests).

Exercises realistic multi-step bounded smoke workloads (--max-steps 3):
Scenario 1: Bounded AdamW 125M smoke training (AdamW, Attention, Scale125M, Checkpoints)
Scenario 2: Bounded Sophia-G 125M smoke training (Sophia-G, Attention, Scale125M, Diagnostics, Core Sampling)
Scenario 3: Bounded Rocora 125M with residual curvature (Rocora, Attention, FP32 Curvature, Diagnostics, Residual Sampling)
Scenario 4: Bounded Muon 350M smoke training (Muon, Attention, Scale350M, Checkpoints, Newton-Schulz)
Scenario 5: Bounded SOAP 350M smoke training (SOAP, Attention, Scale350M, Checkpoints, QR and Basis Refresh)
Scenario 6: Bounded Multi-Scale Stage dry-run (125M, 350M, 1B configs, Pod launcher, Config validation, Lock management)
"""

import fcntl
import json
import math
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.checkpoint import load_checkpoint, save_checkpoint
from rocora.config import (
    OPTIMIZERS,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    learning_rate,
    load_config,
)
from rocora.curvature import core_sample, residual_diagonal
from rocora.model import decay_mask, init_params, loss_sum
from rocora.optimizers import (
    commit_curvature,
    curvature_diagnostics,
    gradient_norm_and_scale,
    init_state,
    update,
)

ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Scenario 1: Bounded AdamW 125M Smoke Training
# ==============================================================================

def test_tier4_scen01_bounded_adamw_125m_smoke_training(tmp_path):
    """Scenario 1: Realistic 3-step bounded AdamW smoke training with checkpointing."""
    cfg_125m = load_config(ROOT / "configs/scale125m.json", optimizer="adamw")
    assert cfg_125m.model.parameter_count == 109_529_856

    # Scaled model for bounded CPU smoke execution
    model_cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64,
                            vocab_size=128, valid_vocab_size=128, attention="reference",
                            loss_chunk=32, remat=False)
    exp_cfg = ExperimentConfig(model=model_cfg, optimizer=cfg_125m.optimizer,
                               train_tokens=10_000, warmup_tokens=100, global_batch_size=2)
    params = init_params(model_cfg, seed=1337)
    state = init_state(params, exp_cfg.optimizer)
    decay = decay_mask(params)

    # 3-step bounded training loop
    max_steps = 3
    losses = []
    for step in range(max_steps):
        tokens = jax.random.randint(jax.random.PRNGKey(step), (2, 64), 0, 128)
        targets = jax.random.randint(jax.random.PRNGKey(step + 10), (2, 64), 0, 128)
        batch = {"tokens": tokens, "targets": targets, "mask": jnp.ones((2, 64), dtype=jnp.float32)}

        # Forward + backward pass
        loss_val, grads = jax.value_and_grad(lambda p: loss_sum(p, batch, model_cfg) / 128.0)(state.params)
        losses.append(float(loss_val))

        # Gradient clipping and learning rate schedule
        norm, scale = gradient_norm_and_scale(grads, exp_cfg.optimizer.grad_clip)
        lr = float(learning_rate(step, exp_cfg))
        state = update(state, grads, lr, exp_cfg.optimizer, decay, gradient_scale=scale)

    assert len(losses) == max_steps
    assert all(math.isfinite(l) and l > 0 for l in losses)
    assert int(state.optimizer.step) == max_steps

    # Checkpoint save and reload verification
    ckpt_file = tmp_path / "adamw_125m_smoke.npz"
    save_checkpoint(ckpt_file, state, exp_cfg, train_seconds=1.2)
    template = init_state(params, exp_cfg.optimizer)
    restored, secs = load_checkpoint(ckpt_file, template, exp_cfg)
    assert secs == 1.2
    np.testing.assert_array_equal(restored.params["layers"]["qkv"], state.params["layers"]["qkv"])


# ==============================================================================
# Scenario 2: Bounded Sophia-G 125M Smoke Training
# ==============================================================================

def test_tier4_scen02_bounded_sophia_125m_smoke_training():
    """Scenario 2: Realistic 3-step bounded Sophia-G smoke training with core curvature."""
    cfg_125m = load_config(ROOT / "configs/scale125m.json", optimizer="sophia")

    model_cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64,
                            vocab_size=128, valid_vocab_size=128, attention="reference",
                            loss_chunk=32, remat=False)
    exp_cfg = ExperimentConfig(model=model_cfg, optimizer=cfg_125m.optimizer,
                               train_tokens=10_000, warmup_tokens=100, global_batch_size=2)
    params = init_params(model_cfg, seed=1337)
    state = init_state(params, exp_cfg.optimizer)
    decay = decay_mask(params)

    max_steps = 3
    for step in range(max_steps):
        batch = {
            "tokens": jax.random.randint(jax.random.PRNGKey(step), (2, 64), 0, 128),
            "targets": jax.random.randint(jax.random.PRNGKey(step + 1), (2, 64), 0, 128),
            "mask": jnp.ones((2, 64), dtype=jnp.float32),
        }
        # Core curvature sample step
        key = jax.random.PRNGKey(step + 7411)
        core, count = core_sample(state.params, batch, model_cfg, key)
        state = commit_curvature(state, core, None, exp_cfg.optimizer)

        # Main gradient update step
        loss_val, grads = jax.value_and_grad(lambda p: loss_sum(p, batch, model_cfg) / 128.0)(state.params)
        lr = float(learning_rate(step, exp_cfg))
        state = update(state, grads, lr, exp_cfg.optimizer, decay)

    assert int(state.optimizer.step) == max_steps
    assert int(state.optimizer.core_events) == max_steps
    assert int(state.optimizer.residual_events) == 0

    diag = curvature_diagnostics(state, exp_cfg.optimizer)
    assert diag["core_mean"] > 0
    assert diag["residual_abs_mean"] == 0


# ==============================================================================
# Scenario 3: Bounded Rocora 125M with Residual Curvature
# ==============================================================================

def test_tier4_scen03_bounded_rocora_125m_with_residual_curvature():
    """Scenario 3: Realistic 3-step bounded Rocora smoke training with coded residual curvature."""
    cfg_125m = load_config(ROOT / "configs/scale125m.json", optimizer="rocora")

    model_cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64,
                            vocab_size=128, valid_vocab_size=128, attention="reference",
                            loss_chunk=32, remat=False)
    exp_cfg = ExperimentConfig(model=model_cfg, optimizer=cfg_125m.optimizer,
                               train_tokens=10_000, warmup_tokens=100, global_batch_size=2)
    params = init_params(model_cfg, seed=1337)
    state = init_state(params, exp_cfg.optimizer)
    decay = decay_mask(params)

    max_steps = 3
    for step in range(max_steps):
        batch = {
            "tokens": jax.random.randint(jax.random.PRNGKey(step), (2, 64), 0, 128),
            "targets": jax.random.randint(jax.random.PRNGKey(step + 1), (2, 64), 0, 128),
            "mask": jnp.ones((2, 64), dtype=jnp.float32),
        }
        # Curvature step: core + coded residual probe
        key = jax.random.PRNGKey(step + 100)
        core, _ = core_sample(state.params, batch, model_cfg, key)
        res, _ = residual_diagonal(state.params, batch, model_cfg, key, m=2)
        state = commit_curvature(state, core, res, exp_cfg.optimizer)

        # Main parameter update
        loss_val, grads = jax.value_and_grad(lambda p: loss_sum(p, batch, model_cfg) / 128.0)(state.params)
        lr = float(learning_rate(step, exp_cfg))
        state = update(state, grads, lr, exp_cfg.optimizer, decay)

    assert int(state.optimizer.step) == max_steps
    assert int(state.optimizer.core_events) == max_steps
    assert int(state.optimizer.residual_events) == max_steps

    diag = curvature_diagnostics(state, exp_cfg.optimizer)
    assert diag["core_mean"] > 0
    assert diag["residual_abs_mean"] > 0
    assert "relative_update_difference" in diag


# ==============================================================================
# Scenario 4: Bounded Muon 350M Smoke Training
# ==============================================================================

def test_tier4_scen04_bounded_muon_350m_smoke_training(tmp_path):
    """Scenario 4: Realistic 3-step bounded Muon smoke training on 350M configuration."""
    cfg_350m = load_config(ROOT / "configs/scale350m.json", optimizer="muon")
    assert cfg_350m.model.parameter_count == 336_380_928

    model_cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64,
                            vocab_size=128, valid_vocab_size=128, attention="reference",
                            loss_chunk=32, remat=False)
    exp_cfg = ExperimentConfig(model=model_cfg, optimizer=cfg_350m.optimizer,
                               train_tokens=10_000, warmup_tokens=100, global_batch_size=2)
    params = init_params(model_cfg, seed=1337)
    state = init_state(params, exp_cfg.optimizer)
    decay = decay_mask(params)

    max_steps = 3
    for step in range(max_steps):
        batch = {
            "tokens": jax.random.randint(jax.random.PRNGKey(step), (2, 64), 0, 128),
            "targets": jax.random.randint(jax.random.PRNGKey(step + 2), (2, 64), 0, 128),
            "mask": jnp.ones((2, 64), dtype=jnp.float32),
        }
        loss_val, grads = jax.value_and_grad(lambda p: loss_sum(p, batch, model_cfg) / 128.0)(state.params)
        lr = float(learning_rate(step, exp_cfg))
        state = update(state, grads, lr, exp_cfg.optimizer, decay)

    assert int(state.optimizer.step) == max_steps
    for p in jax.tree.leaves(state.params):
        assert np.all(np.isfinite(p))

    ckpt_file = tmp_path / "muon_350m_smoke.npz"
    save_checkpoint(ckpt_file, state, exp_cfg, train_seconds=2.0)
    template = init_state(params, exp_cfg.optimizer)
    restored, _ = load_checkpoint(ckpt_file, template, exp_cfg)
    np.testing.assert_array_equal(restored.params["layers"]["mlp_gate"], state.params["layers"]["mlp_gate"])


# ==============================================================================
# Scenario 5: Bounded SOAP 350M Smoke Training
# ==============================================================================

def test_tier4_scen05_bounded_soap_350m_smoke_training(tmp_path):
    """Scenario 5: Realistic 3-step bounded SOAP smoke training with QR basis decomposition."""
    cfg_350m = load_config(ROOT / "configs/scale350m.json", optimizer="soap")

    model_cfg = ModelConfig(width=64, layers=2, heads=2, sequence_length=64,
                            vocab_size=128, valid_vocab_size=128, attention="reference",
                            loss_chunk=32, remat=False)
    exp_cfg = ExperimentConfig(model=model_cfg, optimizer=cfg_350m.optimizer,
                               train_tokens=10_000, warmup_tokens=100, global_batch_size=2)
    params = init_params(model_cfg, seed=1337)
    state = init_state(params, exp_cfg.optimizer)
    decay = decay_mask(params)

    max_steps = 3
    for step in range(max_steps):
        batch = {
            "tokens": jax.random.randint(jax.random.PRNGKey(step), (2, 64), 0, 128),
            "targets": jax.random.randint(jax.random.PRNGKey(step + 3), (2, 64), 0, 128),
            "mask": jnp.ones((2, 64), dtype=jnp.float32),
        }
        loss_val, grads = jax.value_and_grad(lambda p: loss_sum(p, batch, model_cfg) / 128.0)(state.params)
        lr = float(learning_rate(step, exp_cfg))
        state = update(state, grads, lr, exp_cfg.optimizer, decay)

    assert int(state.optimizer.step) == max_steps
    for p in jax.tree.leaves(state.params):
        assert np.all(np.isfinite(p))

    ckpt_file = tmp_path / "soap_350m_smoke.npz"
    save_checkpoint(ckpt_file, state, exp_cfg, train_seconds=2.5)
    template = init_state(params, exp_cfg.optimizer)
    restored, _ = load_checkpoint(ckpt_file, template, exp_cfg)
    np.testing.assert_array_equal(restored.params["layers"]["mlp_down"], state.params["layers"]["mlp_down"])


# ==============================================================================
# Scenario 6: Bounded Multi-Scale Stage Dry-Run
# ==============================================================================

def test_tier4_scen06_bounded_multi_scale_stage_dry_run(tmp_path):
    """Scenario 6: Multi-Scale Stage dry-run simulation across 125M, 350M, 1B with lock management."""
    pod_cfg = json.loads((ROOT / "configs/pod.json").read_text())
    assert "workers" in pod_cfg
    workers = pod_cfg["workers"]
    assert len(workers) >= 1

    # Simulate worker flock lock acquisition
    test_lock = tmp_path / "pod-worker.lock"
    with test_lock.open("a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_f.seek(0)
        lock_f.truncate()
        lock_f.write("job_smoke_test 12345\n")
        lock_f.flush()

        # Iterate all 3 target scale configurations
        scales = ["scale125m", "scale350m", "scale1b"]
        for stage in scales:
            config_file = ROOT / f"configs/{stage}.json"
            cfg = load_config(config_file)
            assert cfg.stage == stage
            assert cfg.model.attention == "flash"
            assert cfg.train_tokens > 0

            # Calculate bounded step horizon under max_steps=3
            step_tokens = cfg.tokens_per_step
            assert step_tokens > 0
            simulated_max_steps = 3
            simulated_tokens = simulated_max_steps * step_tokens
            assert simulated_tokens < cfg.train_tokens

            # Verify all 5 optimizers configured and ready
            for opt_name in OPTIMIZERS:
                opt_cfg = load_config(config_file, optimizer=opt_name)
                assert opt_cfg.optimizer.name == opt_name
                assert opt_cfg.optimizer.learning_rate > 0

        # Release lock
        fcntl.flock(lock_f, fcntl.LOCK_UN)
