"""Bias-free GPT with tied embeddings and stacked layers for efficient TPU scans."""

import math
import jax
import jax.numpy as jnp

from .attention import flash_attention, reference_attention


def init_params(cfg, seed=1337):
    """FP32 master parameters. Stacking layers keeps optimizer kernel count small."""
    key = jax.random.PRNGKey(seed)
    d, l = cfg.width, cfg.layers

    def normal(shape, scale=0.02):
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.normal(sub, shape, dtype=jnp.float32) * scale

    residual_scale = 0.02 / math.sqrt(2 * l)
    return {
        "embedding": normal((cfg.vocab_size, d)),
        "position": normal((cfg.sequence_length, d)),
        "layers": {
            "norm1": jnp.ones((l, d), jnp.float32),
            "qkv": normal((l, d, 3 * d)),
            "attention_out": normal((l, d, d), residual_scale),
            "norm2": jnp.ones((l, d), jnp.float32),
            "mlp_in": normal((l, d, cfg.mlp_ratio * d)),
            "mlp_out": normal((l, cfg.mlp_ratio * d, d), residual_scale),
        },
        "final_norm": jnp.ones((d,), jnp.float32),
    }


def layer_norm(x, weight):
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    return ((xf - mean) * jax.lax.rsqrt(variance + 1e-5) * weight).astype(x.dtype)


def linear(x, weight, dtype):
    precision = jax.lax.Precision.HIGHEST if dtype == jnp.float32 else jax.lax.Precision.DEFAULT
    return jnp.matmul(x, weight.astype(dtype), precision=precision)


def hidden(params, tokens, cfg, curvature=False):
    dtype = jnp.float32 if curvature else jnp.bfloat16
    x = (params["embedding"][tokens] + params["position"][:tokens.shape[1]]).astype(dtype)
    batch, length = tokens.shape

    def block(x, p):
        qkv = linear(layer_norm(x, p["norm1"]), p["qkv"], dtype)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        def split_heads(a):
            return a.reshape(batch, length, cfg.heads, cfg.width // cfg.heads).transpose(0, 2, 1, 3)
        q, k, v = map(split_heads, (q, k, v))
        if cfg.attention == "flash" and not curvature:
            y = flash_attention(q, k, v)
        else:
            y = reference_attention(q, k, v, cfg.attention_block)
        y = y.transpose(0, 2, 1, 3).reshape(batch, length, cfg.width)
        x = x + linear(y, p["attention_out"], dtype)
        y = linear(layer_norm(x, p["norm2"]), p["mlp_in"], dtype)
        y = jax.nn.gelu(y, approximate=False)
        return x + linear(y, p["mlp_out"], dtype), None

    scan_block = jax.checkpoint(block) if cfg.remat else block
    x, _ = jax.lax.scan(scan_block, x, params["layers"])
    return layer_norm(x, params["final_norm"])


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
