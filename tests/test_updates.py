"""Tests for XLA-fused optimizer updates, kernel routing, and scientific parity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.config import OptimizerConfig
from rocora.optimizers import init_state, update
from rocora.pallas_optimizers import should_use_pallas, pallas_optimizer_update
from rocora.updates import adamw_update, sophia_update, sophia_h_update, rocora_update
from rocora.matrix_optimizers import matrix_update


class MockTPUDevice:
    platform = "tpu"


ALL_OPTIMIZERS = ("adamw", "sophia", "sophia_h", "rocora", "muon", "soap")


def test_should_use_pallas_auto_mode_unconditional_false():
    """Verify that should_use_pallas unconditionally returns False in auto mode for all optimizers and backends."""
    cpu_device = jax.devices("cpu")[0]
    tpu_device = MockTPUDevice()

    for opt in ALL_OPTIMIZERS:
        # Default device / platform check
        assert should_use_pallas(opt, mode="auto") is False, f"Failed for {opt} with default device"
        # Explicit CPU device
        assert should_use_pallas(opt, device=cpu_device, mode="auto") is False, f"Failed for {opt} on CPU"
        # Explicit TPU device (must be False per AGENTS.md rule reserving Pallas for FlashAttention)
        assert should_use_pallas(opt, device=tpu_device, mode="auto") is False, f"Failed for {opt} on TPU"


def test_should_use_pallas_unconditional_false_across_all_modes():
    """Verify routing modes unconditionally return False since custom Pallas optimizer kernels are removed."""
    for opt in ALL_OPTIMIZERS:
        for mode in ("auto", "xla", "pallas", "unknown_mode"):
            assert should_use_pallas(opt, mode=mode) is False
            assert should_use_pallas(opt, tensor_shape=(256, 256), mode=mode) is False
            assert should_use_pallas(opt, tensor_shape=(4096, 4096), mode=mode) is False


def test_adamw_xla_update():
    """Verify XLA AdamW elementwise update formulas and decay properties."""
    p = jnp.array([1.0, 2.0], dtype=jnp.float32)
    g = jnp.array([0.1, -0.2], dtype=jnp.float32)
    m = jnp.array([0.0, 0.0], dtype=jnp.float32)
    v = jnp.array([0.0, 0.0], dtype=jnp.float32)

    # c = [lr, b1, b2, eps, wd, rho, corr1, corr2, residual_weight, grad_scale]
    c = jnp.array([0.01, 0.9, 0.999, 1e-8, 0.1, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    p_next, m_next, v_next = adamw_update(p, g, m, v, c)

    # Momentum and variance updates
    expected_m = 0.9 * m + 0.1 * g
    expected_v = 0.999 * v + 0.001 * jnp.square(g)
    np.testing.assert_allclose(m_next, expected_m, atol=1e-6)
    np.testing.assert_allclose(v_next, expected_v, atol=1e-6)

    # Direction and weight decay
    u = expected_m / (jnp.sqrt(expected_v) + 1e-8)
    expected_p = p * (1.0 - 0.01 * 0.1) - 0.01 * u
    np.testing.assert_allclose(p_next, expected_p, atol=1e-6)


def test_sophia_and_sophia_h_xla_update_and_clipping():
    """Verify Sophia-G and Sophia-H coordinate-wise clipped updates and scale identity."""
    p = jnp.array([1.0, -1.0], dtype=jnp.float32)
    g = jnp.array([10.0, -10.0], dtype=jnp.float32)  # Large gradient to trigger clipping
    m = jnp.array([0.0, 0.0], dtype=jnp.float32)
    s = jnp.array([0.01, 0.01], dtype=jnp.float32)

    # c = [lr, b1, b2, eps, wd, rho, corr1, corr2, residual_weight, grad_scale]
    c = jnp.array([0.001, 0.965, 0.99, 1e-15, 0.01, 0.03, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)

    p_sophia, m_sophia = sophia_update(p, g, m, s, c)
    p_sophia_h, m_sophia_h = sophia_h_update(p, g, m, s, c)

    # Parity check: sophia_h_update exactly reproduces sophia_update given the same Hessian diagonal
    np.testing.assert_array_equal(p_sophia, p_sophia_h)
    np.testing.assert_array_equal(m_sophia, m_sophia_h)

    # Clipping check: m / (rho * s + eps) = 0.35 / (0.03 * 0.01 + 1e-15) = 0.35 / 0.0003 = 1166.67 >> 1
    # Clipped to 1.0
    expected_m = 0.965 * m + (1.0 - 0.965) * g
    u = jnp.clip(expected_m / (0.03 * s + 1e-15), -1.0, 1.0)
    np.testing.assert_allclose(u, jnp.array([1.0, -1.0]))
    expected_p = p * (1.0 - 0.001 * 0.01) - 0.001 * u
    np.testing.assert_allclose(p_sophia, expected_p, atol=1e-6)


def test_rocora_scientific_control_identity():
    """Verify scientific control: Rocora at residual_weight=0.0 reproduces Sophia-G bitwise."""
    p = jnp.array([1.5, -2.5, 0.5], dtype=jnp.float32)
    g = jnp.array([0.2, -0.4, 0.1], dtype=jnp.float32)
    m = jnp.array([0.05, -0.02, 0.01], dtype=jnp.float32)
    s = jnp.array([0.04, 0.08, 0.02], dtype=jnp.float32)
    r = jnp.array([0.1, -0.05, 0.03], dtype=jnp.float32)

    c_zero_residual = jnp.array([0.001, 0.965, 0.99, 1e-15, 0.02, 0.03, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    c_sophia = jnp.array([0.001, 0.965, 0.99, 1e-15, 0.02, 0.03, 1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)

    p_rocora, m_rocora = rocora_update(p, g, m, s, r, c_zero_residual)
    p_sophia, m_sophia = sophia_update(p, g, m, s, c_sophia)

    # Must match bitwise
    np.testing.assert_array_equal(p_rocora, p_sophia)
    np.testing.assert_array_equal(m_rocora, m_sophia)


def test_update_routing_auto_mode_uses_xla():
    """Verify that update() routes to XLA fusion in auto mode and matches default XLA execution."""
    params = {"w": jnp.ones((4, 4), dtype=jnp.float32)}
    grads = {"w": jnp.full((4, 4), 0.05, dtype=jnp.float32)}
    decay = {"w": True}

    for opt_name in ("adamw", "sophia", "sophia_h", "rocora"):
        cfg_xla = OptimizerConfig(name=opt_name, learning_rate=1e-3, use_pallas=False, pallas_mode="xla")
        cfg_auto = OptimizerConfig(name=opt_name, learning_rate=1e-3, use_pallas=True, pallas_mode="auto")

        state_xla = init_state(params, cfg_xla)
        state_auto = init_state(params, cfg_auto)

        next_xla = update(state_xla, grads, 1e-3, cfg_xla, decay)
        next_auto = update(state_auto, grads, 1e-3, cfg_auto, decay)

        # Both must execute via XLA-fused paths and yield bitwise identical output
        np.testing.assert_array_equal(next_xla.params["w"], next_auto.params["w"])
        np.testing.assert_array_equal(next_xla.optimizer.moment["w"], next_auto.optimizer.moment["w"])


def test_matrix_optimizers_xla_execution():
    """Verify that Muon and SOAP update passes execute cleanly through XLA fusion in auto mode."""
    params = {"weight": jnp.eye(8, dtype=jnp.float32)}
    grads = {"weight": jnp.full((8, 8), 0.02, dtype=jnp.float32)}
    decay = {"weight": True}

    # Muon update
    cfg_muon = OptimizerConfig(name="muon", learning_rate=1e-3, pallas_mode="auto")
    state_muon = init_state(params, cfg_muon)
    next_muon = update(state_muon, grads, 1e-3, cfg_muon, decay)
    assert next_muon.params["weight"].shape == (8, 8)
    assert not jnp.isnan(next_muon.params["weight"]).any()

    # SOAP update
    cfg_soap = OptimizerConfig(name="soap", learning_rate=1e-3, pallas_mode="auto")
    state_soap = init_state(params, cfg_soap)
    next_soap = update(state_soap, grads, 1e-3, cfg_soap, decay)
    assert next_soap.params["weight"].shape == (8, 8)
    assert not jnp.isnan(next_soap.params["weight"]).any()
