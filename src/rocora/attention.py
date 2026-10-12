"""TPU FlashAttention for first derivatives; tiled JAX attention for curvature."""

import math
import jax
import jax.numpy as jnp


def reference_attention(q, k, v, block=128):
    """Causal [B,H,T,D] attention with rematerialized query tiles.

    This path supports JVPs and second derivatives. A complete T-by-T attention
    matrix need not be retained in HBM for the residual measurements.
    """
    length = q.shape[-2]
    tile = min(block, length)
    if length % tile:
        tile = length
    qs = q.reshape(*q.shape[:2], length // tile, tile, q.shape[-1])
    qs = jnp.moveaxis(qs, 2, 0)
    keys = jnp.arange(length)
    scale = 1 / math.sqrt(q.shape[-1])

    def attend(inputs):
        index, query = inputs
        scores = jnp.einsum("bhqd,bhkd->bhqk", query, k,
                            precision=jax.lax.Precision.HIGHEST,
                            preferred_element_type=jnp.float32) * scale
        positions = index * tile + jnp.arange(tile)
        scores = jnp.where(positions[:, None] >= keys[None, :], scores, -1e30)
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        return jnp.einsum("bhqk,bhkd->bhqd", weights, v,
                          precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32).astype(v.dtype)

    out = jax.lax.map(jax.checkpoint(attend), (jnp.arange(length // tile), qs))
    return jnp.moveaxis(out, 0, 2).reshape(q.shape)


def flash_attention(q, k, v):
    """The JAX team's tiled TPU kernel and custom backward (JAX 0.6.2)."""
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention as flash, BlockSizes
    # Measured on v4 with B=8, H=8, T=1024, D=64: the wider KV tile reduces
    # kernel loop/DMA overhead substantially versus the stock 128-token tile.
    kv = min(1024, k.shape[-2])
    blocks = BlockSizes(
        block_q=128, block_k_major=kv, block_k=kv, block_b=1,
        block_q_major_dkv=128, block_k_major_dkv=kv, block_k_dkv=kv,
        block_q_dkv=128, block_k_major_dq=kv, block_k_dq=kv, block_q_dq=128,
    )
    return flash(q, k, v, causal=True, sm_scale=1 / math.sqrt(q.shape[-1]), block_sizes=blocks)
