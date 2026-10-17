"""Bias-free modern LLaMA architecture with RMSNorm, RoPE, and SwiGLU."""

import math
import jax
import jax.numpy as jnp
import numpy as np

from .attention import flash_attention, reference_attention


def rms_norm(x, weight, eps=1e-5):
    """Root-mean-square normalization without mean-centering."""
    xf = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(variance + eps) * weight.astype(jnp.float32)).astype(x.dtype)


# Alias for backward compatibility
layer_norm = rms_norm


def precompute_rope_freqs(length, dim, base=10000.0, dtype=jnp.float32):
    """Precompute complex rotary frequency embeddings."""
    channel = jnp.arange(0, dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel / dim))
    t = jnp.arange(length, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to the given tensor."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = jnp.concatenate((-x2, x1), axis=-1)
    return (x * cos) + (rotated * sin)


def apply_rope(q, k, base=10000.0):
    """Apply rotary position embeddings to query and key projections per head."""
    length = q.shape[-2]
    head_dim = q.shape[-1]
    cos, sin = precompute_rope_freqs(length, head_dim, base=base, dtype=q.dtype)
    leading_dims = (1,) * (q.ndim - 2)
    cos = cos.reshape(*leading_dims, length, head_dim)
    sin = sin.reshape(*leading_dims, length, head_dim)
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def linear(x, weight, dtype):
    precision = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    return jnp.matmul(x, weight.astype(dtype), precision=precision)


def init_params(cfg, seed=1337):
    """FP32 master parameters. Stacking layers keeps optimizer kernel count small."""
    if isinstance(seed, (int, np.integer)):
        key = jax.random.PRNGKey(seed)
    else:
        key = seed
    d, l = cfg.width, cfg.layers
    d_ffn = cfg.d_ffn

    def normal(shape, scale=0.02):
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.normal(sub, shape, dtype=jnp.float32) * scale

    residual_scale = 0.02 / math.sqrt(2 * l)
    return {
        "embedding": normal((cfg.vocab_size, d)),
        "layers": {
            "norm1": jnp.ones((l, d), jnp.float32),
            "qkv": normal((l, d, 3 * d)),
            "attention_out": normal((l, d, d), residual_scale),
            "norm2": jnp.ones((l, d), jnp.float32),
            "mlp_gate": normal((l, d, d_ffn)),
            "mlp_up": normal((l, d, d_ffn)),
            "mlp_down": normal((l, d_ffn, d), residual_scale),
        },
        "final_norm": jnp.ones((d,), jnp.float32),
    }


def hidden(params, tokens, cfg, curvature=False):
    dtype = jnp.float32 if curvature else jnp.bfloat16
    x = params["embedding"][tokens].astype(dtype)
    batch, length = tokens.shape
    head_dim = cfg.width // cfg.heads
    cos, sin = precompute_rope_freqs(length, head_dim, dtype=dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    def block(x, p):
        qkv = linear(rms_norm(x, p["norm1"]), p["qkv"], dtype)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(a):
            return a.reshape(batch, length, cfg.heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = map(split_heads, (q, k, v))
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        if cfg.attention == "flash" and not curvature:
            y = flash_attention(q, k, v)
        else:
            y = reference_attention(q, k, v, cfg.attention_block)
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, cfg.width)
        x = x + linear(y, p["attention_out"], dtype)

        # SwiGLU MLP
        h = rms_norm(x, p["norm2"])
        gate = linear(h, p["mlp_gate"], dtype)
        up = linear(h, p["mlp_up"], dtype)
        act = jax.nn.silu(gate) * up
        y = linear(act, p["mlp_down"], dtype)
        return x + y, None

    scan_block = jax.checkpoint(block) if cfg.remat else block
    x, _ = jax.lax.scan(scan_block, x, params["layers"])
    return rms_norm(x, params["final_norm"])


def project(params, features, cfg):
    logits = linear(features, params["embedding"].T, features.dtype).astype(jnp.float32)
    return jnp.where(jnp.arange(cfg.vocab_size) < cfg.valid_vocab_size, logits, -1e30)


def logits(params, tokens, cfg, curvature=False):
    return project(params, hidden(params, tokens, cfg, curvature), cfg)


def loss_sum(params, batch, cfg, sample_key=None, curvature=False):
    """Chunk the tied output projection and CE; never retain full training logits.

    With sample_key, labels are sampled independently from the current model's
    categorical distribution, giving the Gauss-Newton-Bartlett core sample.
    """
    features = hidden(params, batch["tokens"], cfg, curvature)
    length = features.shape[1]
    chunk = min(cfg.loss_chunk, length)
    if length % chunk:
        chunk = length
    count = length // chunk
    h = jnp.moveaxis(features.reshape(features.shape[0], count, chunk, cfg.width), 1, 0)
    targets = batch["targets"].reshape(features.shape[0], count, chunk).transpose(1, 0, 2)
    masks = batch["mask"].reshape(features.shape[0], count, chunk).transpose(1, 0, 2)

    def compute(inputs):
        index, hc, tc, mask = inputs
        lg = project(params, hc, cfg)
        if sample_key is not None:
            key = jax.random.fold_in(sample_key, index)
            tc = jax.random.categorical(key, jax.lax.stop_gradient(lg), axis=-1).astype(jnp.int32)
        ce = jax.nn.logsumexp(lg, axis=-1) - jnp.take_along_axis(lg, tc[..., None], -1)[..., 0]
        return jnp.sum(ce * mask)

    pieces = jax.lax.map(jax.checkpoint(compute), (jnp.arange(count), h, targets, masks))
    return jnp.sum(pieces)


def decay_mask(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: not any("norm" in str(k) for k in path), params
    )
