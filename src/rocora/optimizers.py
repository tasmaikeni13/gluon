"""Canonical AdamW/Sophia-G and Sophia-G plus Rocora's residual correction."""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from .updates import adamw_update, sophia_update, rocora_update


class OptimizerState(NamedTuple):
    moment: dict
    core: dict
    residual: dict
    step: jax.Array
    core_events: jax.Array
    residual_events: jax.Array


class TrainState(NamedTuple):
    params: dict
    optimizer: OptimizerState


def init_state(params, cfg):
    zeros = lambda: jax.tree.map(jnp.zeros_like, params)
    return TrainState(params, OptimizerState(
        zeros(), zeros(), zeros() if cfg.name == "rocora" else {},
        jnp.int32(0), jnp.int32(0), jnp.int32(0),
    ))


def commit_curvature(state, core_sample, residual_sample, cfg):
    opt = state.optimizer
    core, residual = opt.core, opt.residual
    nc, nr = opt.core_events, opt.residual_events
    if core_sample is not None:
        core = jax.tree.map(lambda old, new: cfg.beta2 * old + (1 - cfg.beta2) * new,
                            core, core_sample)
        nc += 1
    if residual_sample is not None:
        residual = jax.tree.map(
            lambda old, new: cfg.beta_residual * old + (1 - cfg.beta_residual) * new,
            residual, residual_sample,
        )
        nr += 1
    return TrainState(state.params, opt._replace(core=core, residual=residual,
                                                core_events=nc, residual_events=nr))


def update(state, grads, lr, cfg, decay, gradient_scale=1.0):
    opt = state.optimizer
    step = opt.step + 1
    leaves, structure = jax.tree.flatten(state.params)
    gs, ms, ss = map(jax.tree.leaves, (grads, opt.moment, opt.core))
    rs = jax.tree.leaves(opt.residual) if opt.residual else [None] * len(leaves)
    masks = jax.tree.leaves(decay)
    out_p, out_m, out_s = [], [], []
    for p, g, m, s, r, use_decay in zip(leaves, gs, ms, ss, rs, masks):
        coefficients = jnp.asarray([
            lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay if use_decay else 0,
            cfg.rho, 1 / (1 - cfg.beta1 ** step) if cfg.name == "adamw" else 1,
            1 / (1 - cfg.beta2 ** step) if cfg.name == "adamw" else 1,
            cfg.residual_weight, gradient_scale,
        ], dtype=jnp.float32)
        args = (p, g, m, s) if cfg.name != "rocora" else (p, g, m, s, r)
        fn = {"adamw": adamw_update, "sophia": sophia_update,
              "rocora": rocora_update}[cfg.name]
        values = fn(*args, coefficients)
        out_p.append(values[0])
        out_m.append(values[1])
        out_s.append(values[2] if cfg.name == "adamw" else s)
    unpack = structure.unflatten
    return TrainState(unpack(out_p), opt._replace(
        moment=unpack(out_m), core=unpack(out_s), step=step,
    ))


def gradient_norm_and_scale(grads, threshold):
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1, threshold / (norm + 1e-12)) if threshold > 0 else 1
    return norm, scale


def curvature_diagnostics(state, cfg):
    """Observe whether the residual actually changes the clipped update."""
    moments, cores = map(jax.tree.leaves, (state.optimizer.moment, state.optimizer.core))
    residuals = jax.tree.leaves(state.optimizer.residual) if state.optimizer.residual else [None] * len(cores)
    core_sum = residual_sum = negative = clipped = changed = difference = baseline_norm = 0.0
    count = sum(s.size for s in cores)
    for m,s,r in zip(moments,cores,residuals):
        h = s if r is None else s + cfg.residual_weight*r
        core_update = jnp.clip(m/(cfg.rho*s+cfg.eps),-1,1)
        actual = jnp.clip(m/(jnp.maximum(cfg.rho*h,0)+cfg.eps),-1,1)
        delta = actual-core_update
        core_sum += jnp.sum(s)
        if r is not None:
            residual_sum += jnp.sum(jnp.abs(cfg.residual_weight*r))
        negative += jnp.sum(h<0)
        clipped += jnp.sum(jnp.abs(actual)>=1)
        changed += jnp.sum(jnp.abs(delta)>1e-6)
        difference += jnp.sum(jnp.square(delta))
        baseline_norm += jnp.sum(jnp.square(core_update))
    return {"core_mean":core_sum/count,"residual_abs_mean":residual_sum/count,
            "residual_share":residual_sum/(core_sum+residual_sum+1e-30),
            "negative_curvature_fraction":negative/count,"clipped_fraction":clipped/count,
            "changed_update_fraction":changed/count,
            "relative_update_difference":jnp.sqrt(difference/(baseline_norm+1e-30)),
            "core_events":state.optimizer.core_events,"residual_events":state.optimizer.residual_events}
