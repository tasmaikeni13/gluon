"""Optimizer expressions fused by XLA inside the compiled training step."""

import jax.numpy as jnp


def adamw_update(p, g, m, v, c):
    lr, b1, b2, eps, wd, _, corr1, corr2, _, grad_scale = c
    g = g * grad_scale
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * jnp.square(g)
    update = (m * corr1) / (jnp.sqrt(v * corr2) + eps)
    return p * (1 - lr * wd) - lr * update, m, v


def sophia_update(p, g, m, s, c):
    lr, b1, _, eps, wd, rho, _, _, _, grad_scale = c
    g = g * grad_scale
    m = b1 * m + (1 - b1) * g
    update = jnp.clip(m / (rho * s + eps), -1, 1)
    return p * (1 - lr * wd) - lr * update, m


def sophia_h_update(p, g, m, h, c):
    """Sophia-H coordinate-wise clipped update using Hutchinson Hessian diagonal."""
    return sophia_update(p, g, m, h, c)


def rocora_update(p, g, m, s, r, c):
    lr, b1, _, eps, wd, rho, _, _, residual_weight, grad_scale = c
    g = g * grad_scale
    m = b1 * m + (1 - b1) * g
    denom = jnp.maximum(rho * (s + residual_weight * r), 0) + eps
    update = jnp.clip(m / denom, -1, 1)
    return p * (1 - lr * wd) - lr * update, m
