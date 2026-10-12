"""Positive core samples and fixed-snapshot residual-only coded products."""

import math
import jax
import jax.numpy as jnp

from .model import logits, loss_sum


def latin_codes(shape, m):
    if len(shape) < 2:
        return (jnp.arange(math.prod(shape), dtype=jnp.uint32) % m).reshape(shape)
    row = jnp.arange(shape[-2], dtype=jnp.uint32)[:, None]
    col = jnp.arange(shape[-1], dtype=jnp.uint32)[None, :]
    leading = math.prod(shape[:-2])
    codes = (row + col + jnp.arange(leading, dtype=jnp.uint32)[:, None, None]) % m
    return codes.reshape(shape)


def coded_probe(params, key, row, m):
    leaves, tree = jax.tree.flatten(params)
    probes = []
    for i, p in enumerate(leaves):
        flips = jax.random.rademacher(jax.random.fold_in(key, i), p.shape, dtype=p.dtype)
        code = latin_codes(p.shape, m)
        parity = jax.lax.population_count(code & jnp.uint32(row)) & jnp.uint32(1)
        signs = 1 - 2 * parity.astype(p.dtype)
        probes.append(flips * signs)
    return tree.unflatten(probes)


def core_gradients(params, batch, cfg, key):
    """Sampled-label mean gradient. Distributed callers average before squaring."""
    count = jnp.sum(batch["mask"])
    grads = jax.grad(lambda p: loss_sum(p, batch, cfg, sample_key=key)
                     / jnp.maximum(count, 1))(params)
    return grads, count


def core_sample(params, batch, cfg, key):
    """Single-device N*g_tilde^2 (never a real-label squared gradient)."""
    grads, count = core_gradients(params,batch,cfg,key)
    return jax.tree.map(lambda g: count*jnp.square(g),grads),count


def residual_diagonal(params, batch, cfg, key, m=4):
    """One complete coded group at a fixed parameter/data snapshot.

    Freeze lambda=d(loss)/d(logits) and differentiate <lambda,f(theta)> twice:
    this directly measures R, avoiding cancellation in Hv-Sv and the need to
    differentiate the loss's softmax. No product is taken at a later iterate.
    """
    count = jnp.sum(batch["mask"])
    base_logits = logits(params, batch["tokens"], cfg, curvature=True)
    lam = jax.nn.softmax(base_logits, axis=-1)
    lam = lam - jax.nn.one_hot(batch["targets"], cfg.vocab_size, dtype=jnp.float32)
    lam = jax.lax.stop_gradient(lam * batch["mask"][..., None] / jnp.maximum(count, 1))

    def linear_loss(p):
        # project() masks padded logits to a constant; their lambda is exactly zero.
        return jnp.sum(lam * logits(p, batch["tokens"], cfg, curvature=True))

    # A shared linearization reuses the same derivative tape for all coded rows.
    _, product = jax.linearize(jax.grad(linear_loss), params)
    initial = jax.tree.map(jnp.zeros_like, params)

    def accumulate(total, row):
        probe = coded_probe(params, key, row, m)
        rv = product(probe)
        total = jax.tree.map(lambda a, v, r: a + v * r / m, total, probe, rv)
        return total, None

    estimate, _ = jax.lax.scan(accumulate, initial, jnp.arange(m, dtype=jnp.uint32))
    return estimate, count
