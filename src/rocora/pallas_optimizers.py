"""High-performance Pallas TPU kernels and intelligent XLA fallbacks for Rocora optimizers.

This module provides custom Pallas TPU kernels and intelligent fallback routing for:
- Pointwise optimizers: AdamW, Sophia-G, and Rocora (memory-bandwidth bound, VMEM block tiled).
- Matrix optimizers: Muon (Newton-Schulz iterations in VMEM) and SOAP (Kronecker eigenspace projection).
- Fallback engine: dynamically routes to XLA fusion on CPU or when profiling indicates XLA outperforms Pallas,
  satisfying both Requirement R1 and AGENTS.md scientific rules.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .updates import adamw_update, sophia_update, rocora_update
from .matrix_optimizers import (
    SoapFactors,
    _covariances,
    _initial_basis,
    _project,
    _refresh_basis,
    is_layer_stack,
    muon_direction,
    soap_leaf,
)


def is_tpu_platform() -> bool:
    """Check if Cloud TPU is available and active."""
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


def should_use_pallas(
    optimizer_name: str,
    tensor_shape: Tuple[int, ...] = (),
    device: Optional[jax.Device] = None,
    mode: str = "auto",
) -> bool:
    """Determine whether Pallas should be executed for an update pass.

    Routing logic per Requirement R1 and AGENTS.md:
    1. If mode == "xla": return False.
    2. VMEM Capacity Guard for TPU v4 (16.0 MiB hardware limit, 12.0 MiB safe threshold):
       If tensor_shape is provided and required memory exceeds 12.0 MiB VMEM,
       fall back to XLA to prevent hardware OOM.
    3. If mode == "pallas": return True (unless tensor exceeds VMEM capacity).
    4. If mode == "auto":
       a. On CPU backend: return False (fall back to XLA fusion to avoid Python interpret overhead).
       b. On Cloud TPU v4:
          - Pointwise optimizers (AdamW, Sophia, Rocora) are memory-bandwidth bound (~0.5 FLOP/byte).
            Historical profiling in `docs/tpu-kernels.json` and `docs/validation.md` shows XLA fusion
            matches or slightly outperforms Pallas (0.401 ms vs 0.403 ms) by eliminating kernel launch
            overhead and fusing directly into the backward pass graph. Per AGENTS.md, return False.
          - Matrix optimizers (Muon) on 2D hidden layer matrices: Muon is compute-bound (~8660 FLOP/byte)
            and fits entirely in 16 MiB VMEM on TPU v4. Can return True on TPU if within VMEM limits.
    """
    if mode == "xla":
        return False

    # VMEM Capacity Guard for TPU v4 (16.0 MiB hardware limit, 12.0 MiB safe threshold)
    MAX_VMEM_BYTES = 12 * 1024 * 1024
    if tensor_shape:
        matrix_shape = tensor_shape[-2:] if len(tensor_shape) >= 2 else tensor_shape
        num_elements = 1
        for dim in matrix_shape:
            num_elements *= dim
        tensor_bytes = num_elements * 2
        if optimizer_name == "muon" and len(matrix_shape) == 2:
            k = min(matrix_shape[0], matrix_shape[1])
            vmem_required = tensor_bytes + 2 * (k * k * 4)
        elif optimizer_name == "soap" and len(matrix_shape) == 2:
            d1, d2 = matrix_shape
            vmem_required = (d1 * d2 * 4) * 2 + (d1 * d1 * 4) + (d2 * d2 * 4)
        else:
            vmem_required = tensor_bytes

        if vmem_required > MAX_VMEM_BYTES or tensor_bytes > MAX_VMEM_BYTES:
            return False

    if mode == "pallas":
        return True

    # mode == "auto"
    if device is not None:
        if device.platform != "tpu":
            return False
    elif not is_tpu_platform():
        return False

    # On TPU in auto mode:
    # Pointwise updates default to XLA per AGENTS.md rule
    if optimizer_name in ("adamw", "sophia", "rocora"):
        return False
    if optimizer_name == "muon":
        # Muon compute-bound Newton-Schulz benefits from VMEM residency on TPU
        return True
    if optimizer_name == "soap":
        return False
    return False


# ==============================================================================
# 1. Pointwise Optimizer Kernels (AdamW, Sophia-G, Rocora)
# ==============================================================================

def pallas_adamw_update(
    param: Union[jax.Array, Dict[str, Any]],
    grad: Union[jax.Array, Dict[str, Any]],
    moment: Union[jax.Array, Dict[str, Any]],
    variance: Union[jax.Array, Dict[str, Any]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int = 1,
    grad_scale: float = 1.0,
    block_size: int = 4096,
    interpret: Optional[bool] = None,
    use_fallback: bool = False,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Fused AdamW update pass using Pallas TPU kernel with XLA fallback.

    Formula:
        g = grad * grad_scale
        m = beta1 * moment + (1 - beta1) * g
        v = beta2 * variance + (1 - beta2) * (g ** 2)
        corr1 = 1 / (1 - beta1 ** step)
        corr2 = 1 / (1 - beta2 ** step)
        u = (m * corr1) / (sqrt(v * corr2) + eps)
        param_new = param * (1 - lr * weight_decay) - lr * u

    Returns:
        (new_param, new_moment, new_variance)
    """
    if use_fallback:
        corr1 = 1.0 / (1.0 - beta1 ** step)
        corr2 = 1.0 / (1.0 - beta2 ** step)
        c = jnp.asarray([lr, beta1, beta2, eps, weight_decay, 0.0, corr1, corr2, 0.0, grad_scale], dtype=jnp.float32)
        if isinstance(param, (jax.Array, np.ndarray)):
            return adamw_update(param, grad, moment, variance, c)
        p_leaves, treedef = jax.tree.flatten(param)
        g_leaves = treedef.flatten_up_to(grad)
        m_leaves = treedef.flatten_up_to(moment)
        v_leaves = treedef.flatten_up_to(variance)
        results = [adamw_update(p, g, m, v, c) for p, g, m, v in zip(p_leaves, g_leaves, m_leaves, v_leaves)]
        return (
            treedef.unflatten([r[0] for r in results]),
            treedef.unflatten([r[1] for r in results]),
            treedef.unflatten([r[2] for r in results]),
        )

    if interpret is None:
        interpret = (jax.default_backend() == "cpu" or not is_tpu_platform())

    def _update_array(p, g, m, v):
        orig_shape = p.shape
        size = p.size
        # Block size aligned to TPU sublane width (128 elements)
        bs = min(max(128, block_size), max(128, ((size + 127) // 128) * 128)) if size >= 128 else 128
        pad_len = (bs - (size % bs)) % bs

        p_flat = p.ravel()
        g_flat = g.ravel()
        m_flat = m.ravel()
        v_flat = v.ravel()
        if pad_len > 0:
            p_flat = jnp.pad(p_flat, (0, pad_len))
            g_flat = jnp.pad(g_flat, (0, pad_len))
            m_flat = jnp.pad(m_flat, (0, pad_len))
            v_flat = jnp.pad(v_flat, (0, pad_len))

        corr1 = 1.0 / (1.0 - beta1 ** step)
        corr2 = 1.0 / (1.0 - beta2 ** step)

        def kernel(pr, gr, mr, vr, opr, omr, ovr):
            g_scaled = gr[...] * jnp.float32(grad_scale)
            m_new = jnp.float32(beta1) * mr[...] + (jnp.float32(1.0) - jnp.float32(beta1)) * g_scaled
            v_new = jnp.float32(beta2) * vr[...] + (jnp.float32(1.0) - jnp.float32(beta2)) * (g_scaled * g_scaled)
            u = (m_new * jnp.float32(corr1)) / (jnp.sqrt(v_new * jnp.float32(corr2)) + jnp.float32(eps))
            p_new = pr[...] * (jnp.float32(1.0) - jnp.float32(lr) * jnp.float32(weight_decay)) - jnp.float32(lr) * u
            opr[...] = p_new.astype(pr.dtype)
            omr[...] = m_new.astype(mr.dtype)
            ovr[...] = v_new.astype(vr.dtype)

        grid = (p_flat.size // bs,)
        out_p, out_m, out_v = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(p_flat.shape, p_flat.dtype),
                jax.ShapeDtypeStruct(m_flat.shape, m_flat.dtype),
                jax.ShapeDtypeStruct(v_flat.shape, v_flat.dtype),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                out_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                grid=grid,
            ),
            interpret=interpret,
        )(p_flat, g_flat, m_flat, v_flat)

        return (
            out_p[:size].reshape(orig_shape),
            out_m[:size].reshape(orig_shape),
            out_v[:size].reshape(orig_shape),
        )

    if isinstance(param, (jax.Array, np.ndarray)):
        return _update_array(param, grad, moment, variance)
    p_leaves, treedef = jax.tree.flatten(param)
    g_leaves = treedef.flatten_up_to(grad)
    m_leaves = treedef.flatten_up_to(moment)
    v_leaves = treedef.flatten_up_to(variance)
    results = [_update_array(p, g, m, v) for p, g, m, v in zip(p_leaves, g_leaves, m_leaves, v_leaves)]
    return (
        treedef.unflatten([r[0] for r in results]),
        treedef.unflatten([r[1] for r in results]),
        treedef.unflatten([r[2] for r in results]),
    )


def pallas_sophia_update(
    param: Union[jax.Array, Dict[str, Any]],
    grad: Union[jax.Array, Dict[str, Any]],
    moment: Union[jax.Array, Dict[str, Any]],
    hess_diag: Union[jax.Array, Dict[str, Any]],
    lr: float,
    beta1: float,
    beta2: float = 0.99,
    eps: float = 1e-15,
    rho: float = 0.03,
    weight_decay: float = 0.0,
    grad_scale: float = 1.0,
    block_size: int = 4096,
    interpret: Optional[bool] = None,
    use_fallback: bool = False,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Fused Sophia-G update pass using Pallas TPU kernel with XLA fallback.

    Formula:
        g = grad * grad_scale
        m = beta1 * moment + (1 - beta1) * g
        u = clip(m / (rho * hess_diag + eps), -1, 1)
        param_new = param * (1 - lr * weight_decay) - lr * u

    Returns:
        (new_param, new_moment)
    """
    if use_fallback:
        c = jnp.asarray([lr, beta1, beta2, eps, weight_decay, rho, 0.0, 0.0, 0.0, grad_scale], dtype=jnp.float32)
        if isinstance(param, (jax.Array, np.ndarray)):
            return sophia_update(param, grad, moment, hess_diag, c)
        p_leaves, treedef = jax.tree.flatten(param)
        g_leaves = treedef.flatten_up_to(grad)
        m_leaves = treedef.flatten_up_to(moment)
        s_leaves = treedef.flatten_up_to(hess_diag)
        results = [sophia_update(p, g, m, s, c) for p, g, m, s in zip(p_leaves, g_leaves, m_leaves, s_leaves)]
        return (
            treedef.unflatten([r[0] for r in results]),
            treedef.unflatten([r[1] for r in results]),
        )

    if interpret is None:
        interpret = (jax.default_backend() == "cpu" or not is_tpu_platform())

    def _update_array(p, g, m, s):
        orig_shape = p.shape
        size = p.size
        bs = min(max(128, block_size), max(128, ((size + 127) // 128) * 128)) if size >= 128 else 128
        pad_len = (bs - (size % bs)) % bs

        p_flat = p.ravel()
        g_flat = g.ravel()
        m_flat = m.ravel()
        s_flat = s.ravel()
        if pad_len > 0:
            p_flat = jnp.pad(p_flat, (0, pad_len))
            g_flat = jnp.pad(g_flat, (0, pad_len))
            m_flat = jnp.pad(m_flat, (0, pad_len))
            s_flat = jnp.pad(s_flat, (0, pad_len))

        def kernel(pr, gr, mr, sr, opr, omr):
            g_scaled = gr[...] * jnp.float32(grad_scale)
            m_new = jnp.float32(beta1) * mr[...] + (jnp.float32(1.0) - jnp.float32(beta1)) * g_scaled
            u = jnp.clip(m_new / (jnp.float32(rho) * sr[...] + jnp.float32(eps)), -1.0, 1.0)
            p_new = pr[...] * (jnp.float32(1.0) - jnp.float32(lr) * jnp.float32(weight_decay)) - jnp.float32(lr) * u
            opr[...] = p_new.astype(pr.dtype)
            omr[...] = m_new.astype(mr.dtype)

        grid = (p_flat.size // bs,)
        out_p, out_m = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(p_flat.shape, p_flat.dtype),
                jax.ShapeDtypeStruct(m_flat.shape, m_flat.dtype),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                out_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                grid=grid,
            ),
            interpret=interpret,
        )(p_flat, g_flat, m_flat, s_flat)

        return (
            out_p[:size].reshape(orig_shape),
            out_m[:size].reshape(orig_shape),
        )

    if isinstance(param, (jax.Array, np.ndarray)):
        return _update_array(param, grad, moment, hess_diag)
    p_leaves, treedef = jax.tree.flatten(param)
    g_leaves = treedef.flatten_up_to(grad)
    m_leaves = treedef.flatten_up_to(moment)
    s_leaves = treedef.flatten_up_to(hess_diag)
    results = [_update_array(p, g, m, s) for p, g, m, s in zip(p_leaves, g_leaves, m_leaves, s_leaves)]
    return (
        treedef.unflatten([r[0] for r in results]),
        treedef.unflatten([r[1] for r in results]),
    )


def pallas_rocora_update(
    param: Union[jax.Array, Dict[str, Any]],
    grad: Union[jax.Array, Dict[str, Any]],
    moment: Union[jax.Array, Dict[str, Any]],
    hess_diag: Union[jax.Array, Dict[str, Any]],
    residual: Union[jax.Array, Dict[str, Any]],
    lr: float,
    beta1: float,
    beta2: float = 0.99,
    eps: float = 1e-15,
    rho: float = 0.03,
    weight_decay: float = 0.0,
    residual_weight: float = 1.0,
    grad_scale: float = 1.0,
    block_size: int = 4096,
    interpret: Optional[bool] = None,
    use_fallback: bool = False,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Fused Rocora update pass using Pallas TPU kernel with XLA fallback.

    Formula:
        g = grad * grad_scale
        m = beta1 * moment + (1 - beta1) * g
        h = hess_diag + residual_weight * residual
        denom = max(rho * h, 0) + eps
        u = clip(m / denom, -1, 1)
        param_new = param * (1 - lr * weight_decay) - lr * u

    Scientific Control:
        When residual_weight == 0.0, denom = max(rho * hess_diag, 0) + eps = rho * hess_diag + eps,
        reproducing Sophia-G identically.

    Returns:
        (new_param, new_moment)
    """
    if use_fallback:
        c = jnp.asarray([lr, beta1, beta2, eps, weight_decay, rho, 0.0, 0.0, residual_weight, grad_scale], dtype=jnp.float32)
        if isinstance(param, (jax.Array, np.ndarray)):
            return rocora_update(param, grad, moment, hess_diag, residual, c)
        p_leaves, treedef = jax.tree.flatten(param)
        g_leaves = treedef.flatten_up_to(grad)
        m_leaves = treedef.flatten_up_to(moment)
        s_leaves = treedef.flatten_up_to(hess_diag)
        r_leaves = treedef.flatten_up_to(residual)
        results = [rocora_update(p, g, m, s, r, c) for p, g, m, s, r in zip(p_leaves, g_leaves, m_leaves, s_leaves, r_leaves)]
        return (
            treedef.unflatten([r[0] for r in results]),
            treedef.unflatten([r[1] for r in results]),
        )

    if interpret is None:
        interpret = (jax.default_backend() == "cpu" or not is_tpu_platform())

    def _update_array(p, g, m, s, r):
        orig_shape = p.shape
        size = p.size
        bs = min(max(128, block_size), max(128, ((size + 127) // 128) * 128)) if size >= 128 else 128
        pad_len = (bs - (size % bs)) % bs

        p_flat = p.ravel()
        g_flat = g.ravel()
        m_flat = m.ravel()
        s_flat = s.ravel()
        r_flat = r.ravel()
        if pad_len > 0:
            p_flat = jnp.pad(p_flat, (0, pad_len))
            g_flat = jnp.pad(g_flat, (0, pad_len))
            m_flat = jnp.pad(m_flat, (0, pad_len))
            s_flat = jnp.pad(s_flat, (0, pad_len))
            r_flat = jnp.pad(r_flat, (0, pad_len))

        def kernel(pr, gr, mr, sr, rr, opr, omr):
            g_scaled = gr[...] * jnp.float32(grad_scale)
            m_new = jnp.float32(beta1) * mr[...] + (jnp.float32(1.0) - jnp.float32(beta1)) * g_scaled
            h = sr[...] + jnp.float32(residual_weight) * rr[...]
            denom = jnp.maximum(jnp.float32(rho) * h, 0.0) + jnp.float32(eps)
            u = jnp.clip(m_new / denom, -1.0, 1.0)
            p_new = pr[...] * (jnp.float32(1.0) - jnp.float32(lr) * jnp.float32(weight_decay)) - jnp.float32(lr) * u
            opr[...] = p_new.astype(pr.dtype)
            omr[...] = m_new.astype(mr.dtype)

        grid = (p_flat.size // bs,)
        out_p, out_m = pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(p_flat.shape, p_flat.dtype),
                jax.ShapeDtypeStruct(m_flat.shape, m_flat.dtype),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                out_specs=[
                    pl.BlockSpec((bs,), lambda i: (i,)),
                    pl.BlockSpec((bs,), lambda i: (i,)),
                ],
                grid=grid,
            ),
            interpret=interpret,
        )(p_flat, g_flat, m_flat, s_flat, r_flat)

        return (
            out_p[:size].reshape(orig_shape),
            out_m[:size].reshape(orig_shape),
        )

    if isinstance(param, (jax.Array, np.ndarray)):
        return _update_array(param, grad, moment, hess_diag, residual)
    p_leaves, treedef = jax.tree.flatten(param)
    g_leaves = treedef.flatten_up_to(grad)
    m_leaves = treedef.flatten_up_to(moment)
    s_leaves = treedef.flatten_up_to(hess_diag)
    r_leaves = treedef.flatten_up_to(residual)
    results = [_update_array(p, g, m, s, r) for p, g, m, s, r in zip(p_leaves, g_leaves, m_leaves, s_leaves, r_leaves)]
    return (
        treedef.unflatten([r[0] for r in results]),
        treedef.unflatten([r[1] for r in results]),
    )


# ==============================================================================
# 2. Matrix Optimizer Kernels (Muon and SOAP)
# ==============================================================================

# Newton-Schulz quintic coefficients
NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315


def pallas_muon_ns_step_kernel(x_ref: Any, out_ref: Any):
    """Pallas TPU kernel executing a single Newton-Schulz step in VMEM.

    Gram matrix A = X @ X^T has shape [M, M] (where M <= N).
    TPU v4 16 MiB VMEM holds [M, M] for M <= 1536 across 125M, 350M, 1B scales.
    """
    x = x_ref[:, :]
    xf = x.astype(jnp.float32)

    # Gram matrix: A = X @ X.T
    a = jnp.matmul(x, x.T, preferred_element_type=jnp.float32)

    # Polynomial: B = -4.7750 * A + 2.0315 * A^2
    # Upstream strictly evaluates (2.0315 * A) @ A for BF16 associativity
    c_a = (a * NS_C).astype(jnp.bfloat16)
    a2 = jnp.matmul(c_a, a, preferred_element_type=jnp.float32)
    b = (a * NS_B + a2).astype(jnp.bfloat16)

    # Next iterate: X_next = 3.4445 * X + B @ X
    bx = jnp.matmul(b, x, preferred_element_type=jnp.float32)
    x_next = (xf * NS_A + bx).astype(jnp.bfloat16)

    out_ref[:, :] = x_next


def pallas_muon_direction(
    g: jax.Array,
    momentum: jax.Array,
    cfg: Any = None,
    momentum_val: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    eps: float = 1e-7,
    dtype: jnp.dtype = jnp.bfloat16,
    use_pallas: bool = True,
    interpret: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Muon direction using Pallas TPU kernel on TPU, falling back to XLA on CPU/unsupported."""
    if cfg is not None:
        momentum_val = getattr(cfg, "muon_momentum", momentum_val)
        ns_steps = getattr(cfg, "muon_ns_steps", ns_steps)
        nesterov = getattr(cfg, "muon_nesterov", nesterov)

    # Fallback to XLA if requested or not on TPU (unless explicit interpret mode) or if tensor exceeds VMEM
    is_tpu = is_tpu_platform()
    if interpret is None:
        interpret = False

    vmem_ok = should_use_pallas("muon", tensor_shape=g.shape, mode="pallas" if (is_tpu or interpret) else "xla")
    if not (use_pallas and (is_tpu or interpret) and vmem_ok):
        # Call canonical reference muon_direction
        class DummyCfg:
            muon_momentum = momentum_val
            muon_ns_steps = ns_steps
            muon_nesterov = nesterov
        return muon_direction(g, momentum, cfg or DummyCfg(), dtype=dtype)

    def lerp(start, end, weight):
        delta = end - start
        return start + weight * delta if weight < 0.5 else end - (1 - weight) * delta

    def rounded(x):
        if dtype == jnp.bfloat16:
            x = jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=7)
        return x.astype(dtype)

    momentum = lerp(momentum, g, 1 - momentum_val)
    x = (lerp(g, momentum, momentum_val) if nesterov else momentum).T.astype(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    norm = rounded(jnp.sqrt(jnp.sum(jnp.square(x.astype(jnp.float32)))))
    denominator = rounded(norm.astype(jnp.float32) + 1e-7)
    x = rounded(x.astype(jnp.float32) / denominator.astype(jnp.float32))

    # Newton-Schulz iterations via Pallas
    for _ in range(ns_steps):
        x = pl.pallas_call(
            pallas_muon_ns_step_kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, dtype),
            interpret=interpret,
        )(x)

    if transposed:
        x = x.T

    aspect = max(1, g.shape[1] / g.shape[0]) ** 0.5
    direction = rounded(x.astype(jnp.float32) * aspect).T.astype(g.dtype)
    return direction, momentum


def pallas_muon_update(
    params: Any,
    grads: Any,
    exp_avg: Any,
    lr: float,
    momentum: float = 0.95,
    n_steps: int = 5,
    eps: float = 1e-7,
    wd: float = 0.0,
    cfg: Any = None,
    decay: Any = None,
    grad_scale: float = 1.0,
    use_pallas: bool = True,
    interpret: Optional[bool] = None,
    exp_avg_sq: Any = None,
    step: int = 1,
) -> Any:
    """Muon update supporting both single array (2D/3D) and full parameter PyTree."""
    if cfg is not None:
        momentum = getattr(cfg, "muon_momentum", momentum)
        n_steps = getattr(cfg, "muon_ns_steps", n_steps)
        wd = getattr(cfg, "weight_decay", wd)

    # Single array input handling (contract with unit/e2e tests)
    if isinstance(params, (jax.Array, np.ndarray)):
        p = params
        g = grads * grad_scale
        m = exp_avg
        if p.ndim == 3:
            # Stacked 3D layers: map over layers
            direction, new_m = jax.lax.map(
                lambda arrays: pallas_muon_direction(
                    arrays[0], arrays[1], cfg=cfg, momentum_val=momentum,
                    ns_steps=n_steps, eps=eps, use_pallas=use_pallas, interpret=interpret
                ),
                (g, m),
            )
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m
        elif p.ndim == 2:
            direction, new_m = pallas_muon_direction(
                g, m, cfg=cfg, momentum_val=momentum, ns_steps=n_steps, eps=eps,
                use_pallas=use_pallas, interpret=interpret
            )
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m
        else:
            # 1D auxiliary update
            new_m = momentum * m + (1.0 - momentum) * g
            direction = new_m / (1.0 + eps)
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m

    # PyTree input handling
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(params)
    out_p, out_m, out_v = [], [], []
    masks = jax.tree.leaves(decay) if decay is not None else [True] * len(path_leaves)
    vs = jax.tree.leaves(exp_avg_sq) if exp_avg_sq is not None else [None] * len(path_leaves)

    beta1 = getattr(cfg, "beta1", 0.9) if cfg else 0.9
    beta2 = getattr(cfg, "beta2", 0.95) if cfg else 0.95
    muon_aux_lr = getattr(cfg, "muon_aux_lr", 0.0006) if cfg else 0.0006
    cfg_lr = getattr(cfg, "learning_rate", lr) if cfg else lr
    aux_rate = lr * (muon_aux_lr / cfg_lr) if cfg_lr > 0 else lr
    step_val = int(step) if step is not None else 1
    corr1 = 1.0 / (1.0 - beta1 ** step_val) if step_val > 0 else 1.0
    corr2 = 1.0 / (1.0 - beta2 ** step_val) if step_val > 0 else 1.0
    eps_val = getattr(cfg, "eps", eps) if cfg else eps

    for (path, p), g, m, v, use_decay in zip(
        path_leaves, jax.tree.leaves(grads), jax.tree.leaves(exp_avg), vs, masks
    ):
        current_wd = wd if use_decay else 0.0
        g_scaled = g * grad_scale
        stacked = is_layer_stack(path)

        if stacked and p.ndim == 3:
            direction, new_m = jax.lax.map(
                lambda arrays: pallas_muon_direction(
                    arrays[0], arrays[1], cfg=cfg, momentum_val=momentum,
                    ns_steps=n_steps, eps=eps, use_pallas=use_pallas, interpret=interpret
                ),
                (g_scaled, m),
            )
            new_p = p * (1.0 - lr * current_wd) - lr * direction
            new_v = v if v is not None else jnp.zeros((p.shape[0], 0), jnp.float32)
        else:
            # Auxiliary AdamW matching matrix_optimizers.py:175-181
            v_val = v if v is not None else jnp.zeros_like(p)
            new_m = beta1 * m + (1.0 - beta1) * g_scaled
            new_v = beta2 * v_val + (1.0 - beta2) * jnp.square(g_scaled)
            direction = (new_m * corr1) / (jnp.sqrt(new_v * corr2) + eps_val)
            new_p = p * (1.0 - aux_rate * current_wd) - aux_rate * direction

        out_p.append(new_p)
        out_m.append(new_m)
        out_v.append(new_v)

    if exp_avg_sq is not None:
        return tree.unflatten(out_p), tree.unflatten(out_m), tree.unflatten(out_v)
    return tree.unflatten(out_p), tree.unflatten(out_m)


def pallas_soap_project_kernel(g_ref: Any, q1_ref: Any, q2_ref: Any, out_ref: Any):
    """Projects gradient into Kronecker eigenspace: Z = Q1^T @ G @ Q2."""
    g = g_ref[:, :]
    q1 = q1_ref[:, :]
    q2 = q2_ref[:, :]
    out_ref[:, :] = jnp.matmul(
        jnp.matmul(q1.T, g, preferred_element_type=jnp.float32),
        q2,
        preferred_element_type=jnp.float32,
    )


def pallas_soap_backproject_kernel(u_ref: Any, q1_ref: Any, q2_ref: Any, out_ref: Any):
    """Back-projects update into parameter space: Delta = Q1 @ U @ Q2^T."""
    u = u_ref[:, :]
    q1 = q1_ref[:, :]
    q2 = q2_ref[:, :]
    out_ref[:, :] = jnp.matmul(
        jnp.matmul(q1, u, preferred_element_type=jnp.float32),
        q2.T,
        preferred_element_type=jnp.float32,
    )


def pallas_soap_leaf(
    p: jax.Array,
    g: jax.Array,
    m: jax.Array,
    v: jax.Array,
    factors: SoapFactors,
    step: int,
    lr: float,
    wd: float,
    cfg: Any,
    use_pallas: bool = True,
    interpret: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, SoapFactors]:
    """SOAP leaf update with Pallas projection/Adam and XLA QR refresh fallback."""
    # When not on TPU (or fallback requested without explicit interpret), route to XLA soap_leaf
    is_tpu = is_tpu_platform()
    if interpret is None:
        interpret = False

    vmem_ok = should_use_pallas("soap", tensor_shape=g.shape, mode="pallas" if (is_tpu or interpret) else "xla")
    if not (use_pallas and (is_tpu or interpret) and vmem_ok):
        return soap_leaf(p, g, m, v, factors, step, lr, wd, cfg)

    # Pallas path for eigenspace projection and rotated Adam
    def bootstrap(_):
        cov = _covariances(g, factors.covariance, cfg.soap_shampoo_beta)
        return p, m, v, SoapFactors(cov, _initial_basis(cov))

    def advance(_):
        # 1. Project gradient into eigenspace
        if len(factors.basis) == 2 and factors.basis[0].shape[-1] and factors.basis[1].shape[-1]:
            q1, q2 = factors.basis
            projected = pl.pallas_call(
                pallas_soap_project_kernel,
                out_shape=jax.ShapeDtypeStruct(g.shape, g.dtype),
                interpret=interpret,
            )(g, q1, q2)
        else:
            projected = _project(g, factors.basis)

        # 2. Rotated Adam update
        moment = cfg.beta1 * m + (1 - cfg.beta1) * projected
        variance = cfg.beta2 * v + (1 - cfg.beta2) * jnp.square(projected)
        t = step - 1
        rate = lr * jnp.sqrt(1 - cfg.beta2 ** t) / (1 - cfg.beta1 ** t)
        u = moment / (jnp.sqrt(variance) + cfg.eps)

        # 3. Back-project into parameter space
        if len(factors.basis) == 2 and factors.basis[0].shape[-1] and factors.basis[1].shape[-1]:
            q1, q2 = factors.basis
            direction = pl.pallas_call(
                pallas_soap_backproject_kernel,
                out_shape=jax.ShapeDtypeStruct(u.shape, u.dtype),
                interpret=interpret,
            )(u, q1, q2)
        else:
            direction = _project(u, factors.basis, back=True)

        param = (p - rate * direction) * (1 - lr * wd)

        # 4. Covariance update & periodic basis refresh via XLA LAPACK QR
        cov = _covariances(g, factors.covariance, cfg.soap_shampoo_beta)
        bases, variance = jax.lax.cond(
            t % cfg.soap_precondition_frequency == 0,
            lambda _: _refresh_basis(cov, factors.basis, variance),
            lambda _: (factors.basis, variance),
            None,
        )
        moment = _project(_project(moment, factors.basis, back=True), bases)
        return param, moment, variance, SoapFactors(cov, bases)

    return jax.lax.cond(step == 1, bootstrap, advance, None)


def pallas_soap_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    wd: float = 0.0,
    cfg: Any = None,
    decay: Any = None,
    grad_scale: float = 1.0,
    use_pallas: bool = True,
    interpret: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """SOAP update supporting single array call and full TrainState/OptimizerState PyTree."""
    # Handle single array call (contract with unit/e2e tests)
    if isinstance(params, (jax.Array, np.ndarray)):
        p = params
        g = grads * grad_scale
        opt = getattr(state, "optimizer", state)

        # Extract or initialize factors and moment buffers
        if hasattr(opt, "preconditioner") and isinstance(opt.preconditioner, dict) and "w" in opt.preconditioner:
            f = opt.preconditioner["w"]
            m = opt.moment["w"]
            v = opt.core["w"]
            step = int(opt.step) if hasattr(opt, "step") else 1
        elif hasattr(opt, "preconditioner") and isinstance(opt.preconditioner, SoapFactors):
            f = opt.preconditioner
            m = getattr(opt, "moment", jnp.zeros_like(p))
            v = getattr(opt, "core", jnp.zeros_like(p))
            step = int(getattr(opt, "step", 1))
        elif isinstance(opt, SoapFactors):
            f = opt
            m = jnp.zeros_like(p)
            v = jnp.zeros_like(p)
            step = 1
        else:
            cov = _covariances(g, tuple(jnp.zeros((d, d), jnp.float32) for d in p.shape), 0.95)
            f = SoapFactors(cov, _initial_basis(cov))
            m = jnp.zeros_like(p)
            v = jnp.zeros_like(p)
            step = 1

        class DefaultSoapConfig:
            beta1 = 0.965
            beta2 = 0.99
            eps = 1e-15
            soap_shampoo_beta = 0.95
            soap_precondition_frequency = 10
            soap_max_precondition_dim = 2048
            weight_decay = wd

        config_obj = cfg or DefaultSoapConfig()
        new_p, new_m, new_v, new_f = pallas_soap_leaf(
            p, g, m, v, f, step, lr, wd, config_obj, use_pallas=use_pallas, interpret=interpret
        )
        return new_p, new_f

    # PyTree TrainState call
    opt = state.optimizer
    step = opt.step + 1
    path_leaves, tree = jax.tree_util.tree_flatten_with_path(state.params)
    outputs, moments, variances, factors = [], [], [], []
    old_factors = jax.tree.leaves(opt.preconditioner, is_leaf=lambda x: isinstance(x, SoapFactors))
    masks = jax.tree.leaves(decay) if decay is not None else [False] * len(path_leaves)

    class DefaultSoapConfig:
        beta1 = 0.965
        beta2 = 0.99
        eps = 1e-15
        soap_shampoo_beta = 0.95
        soap_precondition_frequency = 10
        soap_max_precondition_dim = 2048
        weight_decay = wd

    config_obj = cfg or DefaultSoapConfig()

    for (path, p), g, m, v, f, use_decay in zip(
        path_leaves,
        jax.tree.leaves(grads),
        jax.tree.leaves(opt.moment),
        jax.tree.leaves(opt.core),
        old_factors,
        masks,
    ):
        current_wd = wd if use_decay else 0.0
        g_scaled = g * grad_scale
        stacked = is_layer_stack(path)
        fn = lambda arrays: pallas_soap_leaf(
            *arrays, step, lr, current_wd, config_obj, use_pallas=use_pallas, interpret=interpret
        )
        values = jax.lax.map(fn, (p, g_scaled, m, v, f)) if stacked else fn((p, g_scaled, m, v, f))
        param, moment, variance, factor = values
        outputs.append(param)
        moments.append(moment)
        variances.append(variance)
        factors.append(factor)

    new_state = state._replace(
        params=tree.unflatten(outputs),
        optimizer=opt._replace(
            moment=tree.unflatten(moments),
            core=tree.unflatten(variances),
            step=step,
            preconditioner=tree.unflatten(factors),
        ),
    )
    return new_state.params, new_state


# ==============================================================================
# 3. High-Level Pallas Optimizer Dispatcher
# ==============================================================================

def pallas_optimizer_update(
    state: Any,
    grads: Any,
    lr: float,
    cfg: Any,
    decay: Any,
    gradient_scale: float = 1.0,
) -> Any:
    """High-level optimizer update dispatcher routing between Pallas and XLA."""
    mode = getattr(cfg, "pallas_mode", "auto")
    use_pallas_flag = getattr(cfg, "use_pallas", False) or (mode == "pallas")

    # If should_use_pallas returns False, directly fall back to standard XLA update
    if not (use_pallas_flag and should_use_pallas(cfg.name, mode=mode)):
        from .optimizers import update as xla_update
        return xla_update(state, grads, lr, cfg, decay, gradient_scale)

    opt = state.optimizer
    step = opt.step + 1

    if cfg.name == "soap":
        _, new_state = pallas_soap_update(
            state.params, grads, state, lr, wd=cfg.weight_decay,
            cfg=cfg, decay=decay, grad_scale=gradient_scale, use_pallas=True
        )
        return new_state

    if cfg.name == "muon":
        new_params, new_moment, new_core = pallas_muon_update(
            state.params, grads, opt.moment, lr, cfg=cfg, decay=decay,
            grad_scale=gradient_scale, use_pallas=True,
            exp_avg_sq=opt.core, step=step,
        )
        return state._replace(
            params=new_params,
            optimizer=opt._replace(moment=new_moment, core=new_core, step=step)
        )

    # Pointwise optimizers: AdamW, Sophia, Rocora
    leaves, structure = jax.tree.flatten(state.params)
    gs, ms, ss = map(jax.tree.leaves, (grads, opt.moment, opt.core))
    rs = jax.tree.leaves(opt.residual) if opt.residual else [None] * len(leaves)
    masks = jax.tree.leaves(decay)
    out_p, out_m, out_s = [], [], []

    for p, g, m, s, r, use_decay in zip(leaves, gs, ms, ss, rs, masks):
        wd = cfg.weight_decay if use_decay else 0.0
        if cfg.name == "adamw":
            np_, nm_, nv_ = pallas_adamw_update(
                p, g, m, s, lr, cfg.beta1, cfg.beta2, cfg.eps, wd,
                step=int(step), grad_scale=gradient_scale, use_fallback=False
            )
            out_p.append(np_)
            out_m.append(nm_)
            out_s.append(nv_)
        elif cfg.name == "sophia":
            np_, nm_ = pallas_sophia_update(
                p, g, m, s, lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.rho, wd,
                grad_scale=gradient_scale, use_fallback=False
            )
            out_p.append(np_)
            out_m.append(nm_)
            out_s.append(s)
        elif cfg.name == "rocora":
            np_, nm_ = pallas_rocora_update(
                p, g, m, s, r, lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.rho, wd,
                residual_weight=cfg.residual_weight, grad_scale=gradient_scale, use_fallback=False
            )
            out_p.append(np_)
            out_m.append(nm_)
            out_s.append(s)

    unpack = structure.unflatten
    return state._replace(
        params=unpack(out_p),
        optimizer=opt._replace(moment=unpack(out_m), core=unpack(out_s), step=step)
    )
