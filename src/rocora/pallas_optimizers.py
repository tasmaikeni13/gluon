"""Stub module confirming complete removal of custom Pallas optimizer kernels.

Per AGENTS.md and explicit user directive:
Custom Pallas optimizer kernels have been completely removed from Rocora.
All optimizer updates (AdamW, Sophia-G, Sophia-H, Rocora, Muon, SOAP) execute
strictly and solely via pure XLA-fused paths in `updates.py` and `matrix_optimizers.py`.
Pallas is reserved exclusively for TPU FlashAttention in `src/rocora/attention.py`.
"""

from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import numpy as np

from .updates import adamw_update, sophia_update, rocora_update
from .matrix_optimizers import (
    SoapFactors,
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
    """Determine whether Pallas should be executed for an optimizer update pass.

    Returns False unconditionally. Custom Pallas optimizer kernels are completely
    removed in favor of pure XLA fusion.
    """
    return False


# ==============================================================================
# Legacy Shims (Forwarding strictly to pure XLA implementations)
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
    use_fallback: bool = True,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Pure XLA AdamW update shim (custom Pallas optimizer kernels removed)."""
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


def pallas_sophia_update(
    param: Union[jax.Array, Dict[str, Any]],
    grad: Union[jax.Array, Dict[str, Any]],
    moment: Union[jax.Array, Dict[str, Any]],
    hess_diag: Union[jax.Array, Dict[str, Any]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    rho: float,
    weight_decay: float,
    grad_scale: float = 1.0,
    block_size: int = 4096,
    interpret: Optional[bool] = None,
    use_fallback: bool = True,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Pure XLA Sophia-G update shim (custom Pallas optimizer kernels removed)."""
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


def pallas_rocora_update(
    param: Union[jax.Array, Dict[str, Any]],
    grad: Union[jax.Array, Dict[str, Any]],
    moment: Union[jax.Array, Dict[str, Any]],
    hess_diag: Union[jax.Array, Dict[str, Any]],
    residual: Union[jax.Array, Dict[str, Any]],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    rho: float,
    weight_decay: float,
    residual_weight: float = 1.0,
    grad_scale: float = 1.0,
    block_size: int = 4096,
    interpret: Optional[bool] = None,
    use_fallback: bool = True,
) -> Tuple[Union[jax.Array, Dict[str, Any]], Union[jax.Array, Dict[str, Any]]]:
    """Pure XLA Rocora update shim (custom Pallas optimizer kernels removed)."""
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


def pallas_muon_direction(
    g: jax.Array,
    momentum: jax.Array,
    cfg: Optional[Any] = None,
    momentum_val: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    eps: float = 1e-7,
    dtype: jnp.dtype = jnp.bfloat16,
    use_pallas: bool = False,
    interpret: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Pure XLA Muon direction shim (custom Pallas optimizer kernels removed)."""
    if cfg is None:
        class DummyCfg:
            pass
        cfg = DummyCfg()
        cfg.muon_momentum = momentum_val
        cfg.muon_ns_steps = ns_steps
        cfg.muon_nesterov = nesterov
        cfg.eps = eps
    return muon_direction(g, momentum, cfg, dtype=dtype)


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
    use_pallas: bool = False,
    interpret: Optional[bool] = None,
    exp_avg_sq: Any = None,
    step: int = 1,
) -> Any:
    """Pure XLA Muon parameter update shim (custom Pallas optimizer kernels removed)."""
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
                    ns_steps=n_steps, eps=eps, use_pallas=False, interpret=interpret
                ),
                (g, m),
            )
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m
        elif p.ndim == 2:
            direction, new_m = pallas_muon_direction(
                g, m, cfg=cfg, momentum_val=momentum, ns_steps=n_steps, eps=eps,
                use_pallas=False, interpret=interpret
            )
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m
        else:
            new_m = momentum * m + (1.0 - momentum) * g
            direction = new_m / (1.0 + eps)
            new_p = p * (1.0 - lr * wd) - lr * direction
            return new_p, new_m

    from .matrix_optimizers import matrix_update
    new_state = matrix_update(params, grads, lr, cfg, decay, grad_scale)
    return new_state.params, new_state.optimizer.moment, new_state.optimizer.core


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
    use_pallas: bool = False,
    interpret: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, SoapFactors]:
    """Pure XLA SOAP leaf update shim (custom Pallas optimizer kernels removed)."""
    return soap_leaf(p, g, m, v, factors, step, lr, wd, cfg)


def pallas_soap_update(
    params: Any,
    grads: Any,
    state: Any,
    lr: float,
    wd: float = 0.0,
    cfg: Any = None,
    decay: Any = None,
    grad_scale: float = 1.0,
    use_pallas: bool = False,
    interpret: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """Pure XLA SOAP update shim (custom Pallas optimizer kernels removed)."""
    if isinstance(params, (jax.Array, np.ndarray)):
        p = params
        g = grads * grad_scale
        opt = getattr(state, "optimizer", state)

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
            from .matrix_optimizers import _covariances, _initial_basis
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
        new_p, new_m, new_v, new_f = soap_leaf(
            p, g, m, v, f, step, lr, wd, config_obj
        )
        return new_p, new_f

    from .matrix_optimizers import matrix_update
    new_state = matrix_update(state, grads, lr, cfg, decay, grad_scale)
    return new_state.params, new_state


def pallas_optimizer_update(
    state: Any,
    grads: Any,
    lr: float,
    cfg: Any,
    decay: Any,
    gradient_scale: float = 1.0,
) -> Any:
    """Pure XLA optimizer update dispatcher shim (custom Pallas optimizer kernels removed)."""
    from .optimizers import update
    return update(state, grads, lr, cfg, decay, gradient_scale)
