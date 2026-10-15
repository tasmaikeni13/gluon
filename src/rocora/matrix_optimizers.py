"""JAX ports of the authors' SOAP and Muon algorithms (see docs/optimizers.md).

Layer stacks are batches of independent matrices, never tensor dimensions of
one preconditioner. All state is FP32. Muon's quintic products use BF16.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class SoapFactors(NamedTuple):
    covariance: tuple
    basis: tuple


def is_layer_stack(path):
    return bool(path) and getattr(path[0], "key", None) == "layers"


def init_soap_factors(params, cfg):
    def init(path, p):
        stacked = is_layer_stack(path)
        shape = p.shape[1:] if stacked else p.shape
        prefix = p.shape[:1] if stacked else ()
        # Match precondition_1d=False and merge_dims=False in official SOAP.
        cov = tuple(jnp.zeros(prefix + (n, n), jnp.float32)
                    if len(shape) > 1 and n <= cfg.soap_max_precondition_dim
                    else jnp.zeros(prefix + (0, 0), jnp.float32) for n in shape)
        return SoapFactors(cov, tuple(jnp.zeros_like(x) for x in cov))
    return jax.tree_util.tree_map_with_path(init, params)


def _project(x, bases, back=False):
    # Each contraction moves the first tensor axis to the end. A full cycle
    # restores the original axis order, including axes without preconditioning.
    for q in bases:
        if q.shape[-1]:
            x = jnp.tensordot(x, q, axes=((0,), (1 if back else 0,)),
                              precision=jax.lax.Precision.HIGHEST)
        else:
            x = jnp.moveaxis(x, 0, -1)
    return x


def _covariances(g, old, beta):
    result = []
    for axis, c in enumerate(old):
        if c.shape[-1]:
            x = jnp.moveaxis(g, axis, 0).reshape(g.shape[axis], -1)
            sample = jnp.matmul(x, x.T, precision=jax.lax.Precision.HIGHEST)
            c = beta * c + (1 - beta) * sample
        result.append(c)
    return tuple(result)


def _initial_basis(cov):
    return tuple(jnp.linalg.eigh(c + 1e-30 * jnp.eye(c.shape[-1]))[1][:, ::-1]
                 if c.shape[-1] else c for c in cov)


def _refresh_basis(cov, bases, variance):
    result = []
    for axis, (c, q) in enumerate(zip(cov, bases)):
        if q.shape[-1]:
            product = jnp.matmul(c, q, precision=jax.lax.Precision.HIGHEST)
            order = jnp.argsort(jnp.sum(q * product, axis=0), descending=True, stable=True)
            variance = jnp.take(variance, order, axis=axis)
            q = jnp.linalg.qr(jnp.matmul(c, q[:, order], precision=jax.lax.Precision.HIGHEST))[0]
        result.append(q)
    return tuple(result), variance


def soap_leaf(p, g, m, v, factors, step, lr, wd, cfg):
    """Official bootstrap, rotated Adam, then covariance/basis refresh order."""
    def bootstrap(_):
        cov = _covariances(g, factors.covariance, cfg.soap_shampoo_beta)
        return p, m, v, SoapFactors(cov, _initial_basis(cov))

    def advance(_):
        projected = _project(g, factors.basis)
        moment = cfg.beta1 * m + (1 - cfg.beta1) * projected
        variance = cfg.beta2 * v + (1 - cfg.beta2) * jnp.square(projected)
        # SOAP's epsilon is inside the uncorrected denominator, as upstream.
        t = step - 1
        rate = lr * jnp.sqrt(1 - cfg.beta2 ** t) / (1 - cfg.beta1 ** t)
        direction = _project(moment / (jnp.sqrt(variance) + cfg.eps), factors.basis, back=True)
        param = (p - rate * direction) * (1 - lr * wd)
        cov = _covariances(g, factors.covariance, cfg.soap_shampoo_beta)
        bases, variance = jax.lax.cond(
            t % cfg.soap_precondition_frequency == 0,
            lambda _: _refresh_basis(cov, factors.basis, variance),
            lambda _: (factors.basis, variance), None)
        # Transport the first moment exactly through parameter coordinates.
        # Upstream only permutes v, an approximation documented in theory.md.
        moment = _project(_project(moment, factors.basis, back=True), bases)
        return param, moment, variance, SoapFactors(cov, bases)

    return jax.lax.cond(step == 1, bootstrap, advance, None)


def muon_direction(g, momentum, cfg, dtype=jnp.bfloat16):
    """Authors' EMA/Nesterov and original aspect-ratio adjustment.

    Model weights are stored [input, output], the transpose of torch Linear.
    Orthogonalize in [output, input] coordinates before applying their scale.
    """
    def lerp(start, end, weight):
        # Match the reference's numerically stable torch.lerp convention.
        delta = end - start
        return start + weight * delta if weight < .5 else end - (1 - weight) * delta

    def rounded(x):
        # XLA may retain extra precision through BF16 elementwise operations.
        # Explicit reduction preserves the eager reference's rounding boundaries
        # while allowing fusion, without forcing separate optimizer kernels.
        if dtype == jnp.bfloat16:
            x = jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
        return x.astype(dtype)

    def scale(x, coefficient):
        return rounded(x.astype(jnp.float32) * coefficient)

    def matmul(x, y):
        return rounded(jnp.matmul(x, y, preferred_element_type=jnp.float32))

    def add(x, y):
        return rounded(x.astype(jnp.float32) + y.astype(jnp.float32))

    momentum = lerp(momentum, g, 1 - cfg.muon_momentum)
    x = (lerp(g, momentum, cfg.muon_momentum) if cfg.muon_nesterov else momentum).T.astype(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    # Match upstream BF16 normalization; do not replace the finite polynomial
    # with an exact polar decomposition.
    norm = rounded(jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)))))
    denominator = rounded(norm.astype(jnp.float32) + 1e-7)
    x = rounded(x.astype(jnp.float32) / denominator.astype(jnp.float32))
    def iteration(_, x):
        a = matmul(x, x.T)
        # Upstream's c*A@A evaluates (c*A)@A; reassociation changes BF16 results.
        b = add(scale(a, -4.7750), matmul(scale(a, 2.0315), a))
        return add(scale(x, 3.4445), matmul(b, x))
    x = jax.lax.fori_loop(0, cfg.muon_ns_steps, iteration, x)
    if transposed:
        x = x.T
    aspect = max(1, g.shape[1] / g.shape[0]) ** 0.5
    # Upstream scales the BF16 result before converting back to master dtype.
    return rounded(x.astype(jnp.float32) * aspect).T.astype(g.dtype), momentum


def matrix_update(state, grads, lr, cfg, decay, gradient_scale):
    opt, step = state.optimizer, state.optimizer.step + 1
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(state.params)
    outputs, moments, variances, factors = [], [], [], []
    old_factors = (jax.tree.leaves(opt.preconditioner, is_leaf=lambda x: isinstance(x, SoapFactors))
                   if cfg.name == "soap" else [None] * len(path_leaves))
    for (path, p), g, m, v, f, use_decay in zip(
            path_leaves, jax.tree.leaves(grads), jax.tree.leaves(opt.moment),
            jax.tree.leaves(opt.core), old_factors, jax.tree.leaves(decay)):
        g = g * gradient_scale
        wd = cfg.weight_decay if use_decay else 0
        stacked = is_layer_stack(path)
        if cfg.name == "soap":
            fn = lambda arrays: soap_leaf(*arrays, step, lr, wd, cfg)
            values = jax.lax.map(fn, (p, g, m, v, f)) if stacked else fn((p, g, m, v, f))
            param, moment, variance, factor = values
            factors.append(factor)
        elif stacked and p.ndim == 3:
            # Scan layers to bound temporary Gram-matrix memory at 1B scale.
            direction, moment = jax.lax.map(lambda arrays: muon_direction(*arrays, cfg), (g, m))
            param, variance = p * (1 - lr * wd) - lr * direction, v
        else:
            # Tied embedding/output, positions, and norm gains use AdamW.
            moment = cfg.beta1 * m + (1 - cfg.beta1) * g
            variance = cfg.beta2 * v + (1 - cfg.beta2) * jnp.square(g)
            direction = (moment / (1 - cfg.beta1 ** step)
                         / (jnp.sqrt(variance / (1 - cfg.beta2 ** step)) + cfg.eps))
            aux_rate = lr * (cfg.muon_aux_lr / cfg.learning_rate)
            param = p * (1 - aux_rate * wd) - aux_rate * direction
        outputs.append(param)
        moments.append(moment)
        variances.append(variance)
    return state._replace(params=tree.unflatten(outputs), optimizer=opt._replace(
        moment=tree.unflatten(moments), core=tree.unflatten(variances), step=step,
        preconditioner=tree.unflatten(factors) if cfg.name == "soap" else {}))
