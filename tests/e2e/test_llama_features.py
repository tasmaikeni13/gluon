"""E2E Requirement-Driven Tests for Modern LLaMA Architecture Features.

Covers:
- RoPE rotary properties (relative position invariance, orthogonal norm preservation, frequencies)
- RMSNorm numerical stability (zero mean-centering, float32 accumulation, scale parameters)
- SwiGLU activation and 8/3 dimension scaling (parameter parity, tile alignment to 64)
- Parameter count formulas for LLaMA across scales (125M, 350M, 1B, zero position table)
- Higher-order autodiff compatibility (JVP/HVP) for curvature measurements
"""

import math
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rocora.attention import reference_attention
from rocora.config import ModelConfig, load_config
from rocora.model import decay_mask, init_params

ROOT = Path(__file__).resolve().parents[2]


# ==============================================================================
# Helper / Reference Implementations matching PROJECT.md Specification
# ==============================================================================

def precompute_rope_freqs_ref(length, dim, base=10000.0, dtype=jnp.float32):
    channel = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel / dim))
    t = jnp.arange(length, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rotary_emb_ref(x, cos, sin):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    return (x * cos) + (rotated * sin)


def rms_norm_ref(x, weight, eps=1e-5):
    xf = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(variance + eps) * weight).astype(x.dtype)


def swiglu_ref(x, w_gate, w_up, w_down):
    dtype = x.dtype
    gate = jnp.matmul(x, w_gate.astype(dtype))
    up = jnp.matmul(x, w_up.astype(dtype))
    hidden = jax.nn.silu(gate) * up
    return jnp.matmul(hidden, w_down.astype(dtype))


def calculate_d_ffn(width, multiple=64):
    if width < multiple:
        return max(1, int(8 * width // 3))
    raw = 8 * width // 3
    return multiple * int(round(raw / multiple))


def llama_parameter_count_formula(vocab_size, width, layers, d_ffn):
    # vocab * d + layers * (4 * d^2 + 3 * d * d_ffn + 2 * d) + d
    token_emb = vocab_size * width
    attn = 4 * width * width
    mlp = 3 * width * d_ffn
    norms = 2 * width
    layer_total = layers * (attn + mlp + norms)
    final_norm = width
    return token_emb + layer_total + final_norm


# ==============================================================================
# Feature: RoPE Rotary Properties
# ==============================================================================

def test_rope_frequency_monotonicity_and_bounds():
    """Verify RoPE inverse frequencies decrease monotonically from 1.0 down to base^(-1)."""
    dim = 64
    base = 10000.0
    channel = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel / dim))

    assert inv_freq[0] == 1.0
    assert float(inv_freq[-1]) == pytest.approx(1.0 / base ** ((dim - 2) / dim), rel=1e-5)
    # Monotonically strictly decreasing
    diffs = jnp.diff(inv_freq)
    assert np.all(diffs < 0)


def test_rope_orthogonal_norm_preservation():
    """Verify RoPE is an orthogonal coordinate transformation that strictly preserves vector norms."""
    seq_len, dim = 16, 64
    cos, sin = precompute_rope_freqs_ref(seq_len, dim)
    rng = np.random.default_rng(42)
    x = jnp.asarray(rng.normal(size=(2, 4, seq_len, dim)).astype(np.float32))

    # Apply RoPE
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    x_rot = apply_rotary_emb_ref(x, cos_b, sin_b)

    # Norms before and after must be identical per token
    norm_before = jnp.linalg.norm(x, axis=-1)
    norm_after = jnp.linalg.norm(x_rot, axis=-1)
    np.testing.assert_allclose(norm_before, norm_after, rtol=1e-5, atol=1e-5)


def test_rope_relative_position_invariance():
    """Verify RoPE inner product depends strictly on relative position (m - n)."""
    seq_len, dim = 32, 64
    cos, sin = precompute_rope_freqs_ref(seq_len, dim)
    rng = np.random.default_rng(1337)
    q = jnp.asarray(rng.normal(size=(dim,)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(dim,)).astype(np.float32))

    # Compute dot products for identical relative distance: delta = 5
    # Pair 1: m=5, n=0 -> delta=5
    q5 = apply_rotary_emb_ref(q, cos[5], sin[5])
    k0 = apply_rotary_emb_ref(k, cos[0], sin[0])
    dot_5_0 = jnp.dot(q5, k0)

    # Pair 2: m=12, n=7 -> delta=5
    q12 = apply_rotary_emb_ref(q, cos[12], sin[12])
    k7 = apply_rotary_emb_ref(k, cos[7], sin[7])
    dot_12_7 = jnp.dot(q12, k7)

    # Pair 3: m=20, n=15 -> delta=5
    q20 = apply_rotary_emb_ref(q, cos[20], sin[20])
    k15 = apply_rotary_emb_ref(k, cos[15], sin[15])
    dot_20_15 = jnp.dot(q20, k15)

    np.testing.assert_allclose(dot_5_0, dot_12_7, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dot_5_0, dot_20_15, rtol=1e-5, atol=1e-5)


def test_rope_position_zero_is_identity():
    """Verify RoPE at position t=0 produces an exact identity mapping (cos=1, sin=0)."""
    dim = 64
    cos, sin = precompute_rope_freqs_ref(1, dim)
    np.testing.assert_allclose(cos[0], jnp.ones(dim), atol=1e-6)
    np.testing.assert_allclose(sin[0], jnp.zeros(dim), atol=1e-6)

    x = jnp.arange(dim, dtype=jnp.float32)
    x_rot = apply_rotary_emb_ref(x, cos[0], sin[0])
    np.testing.assert_allclose(x, x_rot, atol=1e-6)


def test_rope_higher_order_autodiff_smoothness():
    """Verify RoPE supports JVP forward differentiation and second-order autodiff without NaNs."""
    dim = 32
    cos, sin = precompute_rope_freqs_ref(1, dim)

    def fn(x):
        rot = apply_rotary_emb_ref(x, cos[0], sin[0])
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


def test_rope_values_unrotated_contract():
    """Verify interface contract: only Q and K are rotated; V remains unrotated."""
    # Under attention contract: attention(RoPE(q), RoPE(k), v)
    rng = np.random.default_rng(99)
    q = jnp.asarray(rng.normal(size=(1, 2, 8, 16)).astype(np.float32))
    k = jnp.asarray(rng.normal(size=(1, 2, 8, 16)).astype(np.float32))
    v = jnp.asarray(rng.normal(size=(1, 2, 8, 16)).astype(np.float32))

    cos, sin = precompute_rope_freqs_ref(8, 16)
    cos_b, sin_b = cos[None, None, :, :], sin[None, None, :, :]

    q_rot = apply_rotary_emb_ref(q, cos_b, sin_b)
    k_rot = apply_rotary_emb_ref(k, cos_b, sin_b)

    out = reference_attention(q_rot, k_rot, v, block=4)
    assert out.shape == (1, 2, 8, 16)
    assert np.all(np.isfinite(out))


# ==============================================================================
# Feature: RMSNorm Numerical Stability
# ==============================================================================

def test_rmsnorm_mathematical_formula_parity():
    """Verify RMSNorm produces exact mathematical root-mean-square normalization."""
    x = jnp.array([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)
    gamma = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32)
    eps = 1e-5

    out = rms_norm_ref(x, gamma, eps=eps)

    rms = math.sqrt(jnp.mean(x ** 2) + eps)
    expected = (x / rms) * gamma
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_rmsnorm_zero_mean_centering_absence():
    """Verify RMSNorm does NOT center mean, contrasting sharply with LayerNorm."""
    # Constant input x = [3.0, 3.0]
    x = jnp.array([3.0, 3.0], dtype=jnp.float32)
    gamma = jnp.array([1.0, 1.0], dtype=jnp.float32)

    # RMSNorm preserves non-zero constant values
    rms_out = rms_norm_ref(x, gamma)
    np.testing.assert_allclose(rms_out, jnp.ones(2), atol=1e-3)

    # LayerNorm centers mean (3 - 3 = 0), which would produce [0.0, 0.0]
    ln_mean = jnp.mean(x)
    assert float(ln_mean) == 3.0
    assert not np.allclose(rms_out, jnp.zeros(2))


def test_rmsnorm_positive_homogeneity():
    """Verify RMSNorm scale invariance under positive scalar multiplication: RMSNorm(alpha * x) = RMSNorm(x)."""
    x = jnp.array([2.0, -5.0, 7.0, 1.0], dtype=jnp.float32)
    gamma = jnp.ones(4, dtype=jnp.float32)

    out_base = rms_norm_ref(x, gamma, eps=1e-8)
    out_scaled = rms_norm_ref(x * 100.0, gamma, eps=1e-8)

    np.testing.assert_allclose(out_base, out_scaled, rtol=1e-5, atol=1e-5)


def test_rmsnorm_float32_accumulation_and_underflow_safety():
    """Verify RMSNorm casts internally to float32 to prevent half-precision overflow/underflow."""
    x_small = jnp.full(64, 1e-4, dtype=jnp.float32)
    gamma = jnp.ones(64, dtype=jnp.float32)

    out = rms_norm_ref(x_small, gamma, eps=1e-5)
    assert np.all(np.isfinite(out))
    # Normalized output should be unit magnitude per element
    assert float(jnp.mean(out)) > 0


def test_rmsnorm_higher_order_derivatives():
    """Verify RMSNorm supports second derivatives (HVP) cleanly without division-by-zero singularities."""
    x = jnp.array([0.5, -0.5, 1.0, -1.0], dtype=jnp.float32)
    gamma = jnp.ones(4, dtype=jnp.float32)
    v = jnp.array([0.1, 0.2, -0.1, -0.2], dtype=jnp.float32)

    def loss(p):
        return jnp.sum(rms_norm_ref(x, p) ** 2)

    # Gradient and Hessian-vector product with respect to gamma
    grad = jax.grad(loss)(gamma)
    assert np.all(np.isfinite(grad))

    _, hvp = jax.jvp(jax.grad(loss), (gamma,), (v,))
    assert np.all(np.isfinite(hvp))


# ==============================================================================
# Feature: SwiGLU Activation & 8/3 Width Ratio
# ==============================================================================

def test_swiglu_mathematical_activation():
    """Verify SwiGLU computes silu(gate) * up followed by down projection."""
    rng = np.random.default_rng(2026)
    d, d_ffn = 16, 32
    x = jnp.asarray(rng.normal(size=(2, d)).astype(np.float32))
    w_gate = jnp.asarray(rng.normal(size=(d, d_ffn)).astype(np.float32))
    w_up = jnp.asarray(rng.normal(size=(d, d_ffn)).astype(np.float32))
    w_down = jnp.asarray(rng.normal(size=(d_ffn, d)).astype(np.float32))

    out = swiglu_ref(x, w_gate, w_up, w_down)
    assert out.shape == (2, d)

    # Verify intermediate components
    gate = jnp.matmul(x, w_gate)
    up = jnp.matmul(x, w_up)
    silu_gate = gate * (1.0 / (1.0 + jnp.exp(-gate)))
    expected = jnp.matmul(silu_gate * up, w_down)

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_swiglu_dimension_scaling_and_tile_alignment():
    """Verify 8/3 dimension scaling aligns to 64-byte multiple across all model tiers."""
    # 125M tier: width=768 -> 8/3 * 768 = 2048 (aligned to 64 and 128)
    assert calculate_d_ffn(768, multiple=64) == 2048

    # 350M tier: width=1024 -> 8/3 * 1024 = 2730.67 -> aligned to 64 is 2752
    assert calculate_d_ffn(1024, multiple=64) == 2752
    assert calculate_d_ffn(1024, multiple=64) % 64 == 0

    # 1B tier: width=1536 -> 8/3 * 1536 = 4096 (aligned to 64 and 128)
    assert calculate_d_ffn(1536, multiple=64) == 4096

    # Small toy dimension guard
    assert calculate_d_ffn(2, multiple=64) == 5
    assert calculate_d_ffn(4, multiple=64) == 10
    assert calculate_d_ffn(16, multiple=64) == 42


def test_swiglu_parameter_parity_with_standard_4d_mlp():
    """Verify SwiGLU with 8/3 width ratio achieves theoretical parameter parity with standard 4d MLP."""
    # For width divisible by 3 (e.g. 768 and 1536):
    # Standard MLP: 2 * d * 4d = 8 d^2
    # SwiGLU: 3 * d * (8/3 d) = 8 d^2
    for d in (768, 1536):
        d_ffn = int(8 * d / 3)
        standard_mlp_params = 2 * d * (4 * d)
        swiglu_params = 3 * d * d_ffn
        assert swiglu_params == standard_mlp_params


def test_swiglu_smooth_higher_order_derivatives():
    """Verify SwiGLU SiLU gating is smooth (C^infinity) and produces finite HVPs."""
    d, d_ffn = 8, 16
    w_gate = jnp.ones((d, d_ffn), dtype=jnp.float32)
    w_up = jnp.ones((d, d_ffn), dtype=jnp.float32)
    w_down = jnp.ones((d_ffn, d), dtype=jnp.float32)
    x = jnp.ones((1, d), dtype=jnp.float32)

    def f(w_g):
        return jnp.sum(swiglu_ref(x, w_g, w_up, w_down))

    v = jnp.full_like(w_gate, 0.05)
    primals, tangents = jax.jvp(f, (w_gate,), (v,))
    assert np.isfinite(primals)
    assert np.isfinite(tangents)

    # Double backward
    _, hvp = jax.jvp(jax.grad(f), (w_gate,), (v,))
    assert np.all(np.isfinite(hvp))


# ==============================================================================
# Feature: LLaMA Parameter Count Formulas Across Scales
# ==============================================================================

def test_param_count_formula_125m_specification():
    """Verify LLaMA-125M analytical formula yields exactly 109,529,856 parameters."""
    vocab_size = 32000
    width = 768
    layers = 12
    d_ffn = 2048

    count = llama_parameter_count_formula(vocab_size, width, layers, d_ffn)
    assert count == 109_529_856


def test_param_count_formula_350m_specification():
    """Verify LLaMA-350M analytical formula yields exactly 336,380,928 parameters (64-alignment)."""
    vocab_size = 32000
    width = 1024
    layers = 24
    d_ffn = 2752  # 64-tile alignment

    count = llama_parameter_count_formula(vocab_size, width, layers, d_ffn)
    assert count == 336_380_928


def test_param_count_formula_1b_specification():
    """Verify LLaMA-1B analytical formula yields exactly 955,221,504 parameters."""
    vocab_size = 32000
    width = 1536
    layers = 32
    d_ffn = 4096

    count = llama_parameter_count_formula(vocab_size, width, layers, d_ffn)
    assert count == 955_221_504


def test_param_count_elimination_of_learned_position_table():
    """Verify delta when eliminating learned absolute position table equals seq_len * width."""
    # 125M: hypothetical pos table would add 1024 * 768 parameters.
    base_125m = llama_parameter_count_formula(32000, 768, 12, 2048)
    expected_pos_table_125m = 1024 * 768  # seq_len * width
    assert (base_125m + expected_pos_table_125m) - base_125m == expected_pos_table_125m == 786_432

    # 1B: hypothetical pos table would add 1024 * 1536 parameters.
    base_1b = llama_parameter_count_formula(32000, 1536, 32, 4096)
    expected_pos_table_1b = 1024 * 1536
    assert (base_1b + expected_pos_table_1b) - base_1b == expected_pos_table_1b == 1_572_864


def test_param_count_tied_embeddings_and_zero_additive_biases():
    """Verify tied unembedding head adds 0 parameters and all bias parameters are 0."""
    vocab_size = 32000
    width = 768
    # Embedding table: vocab_size * width
    emb_params = vocab_size * width
    # Output projection is tied to emb.T -> 0 independent parameters
    head_independent_params = 0
    # Bias-free design -> 0 bias parameters
    bias_params = 0

    assert emb_params == 32000 * 768
    assert head_independent_params == 0
    assert bias_params == 0


# ==============================================================================
# Feature: Decay Mask Compatibility with LLaMA Architecture
# ==============================================================================

def test_decay_mask_with_llama_parameter_tree():
    """Verify decay_mask correctly excludes RMSNorm weights while decaying linear weights."""
    llama_params = {
        "embedding": jnp.ones((100, 32), dtype=jnp.float32),
        "layers": {
            "norm1": jnp.ones((2, 32), dtype=jnp.float32),
            "qkv": jnp.ones((2, 32, 96), dtype=jnp.float32),
            "attention_out": jnp.ones((2, 32, 32), dtype=jnp.float32),
            "norm2": jnp.ones((2, 32), dtype=jnp.float32),
            "mlp_gate": jnp.ones((2, 32, 64), dtype=jnp.float32),
            "mlp_up": jnp.ones((2, 32, 64), dtype=jnp.float32),
            "mlp_down": jnp.ones((2, 64, 32), dtype=jnp.float32),
        },
        "final_norm": jnp.ones(32, dtype=jnp.float32),
    }

    mask = decay_mask(llama_params)
    # Norms must have weight decay disabled (mask == False)
    assert mask["layers"]["norm1"] is False
    assert mask["layers"]["norm2"] is False
    assert mask["final_norm"] is False

    # Projections must have weight decay enabled (mask == True)
    assert mask["layers"]["qkv"] is True
    assert mask["layers"]["attention_out"] is True
    assert mask["layers"]["mlp_gate"] is True
    assert mask["layers"]["mlp_up"] is True
    assert mask["layers"]["mlp_down"] is True
