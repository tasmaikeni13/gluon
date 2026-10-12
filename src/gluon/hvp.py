"""Hessian-vector products via double backward, Turing-safe.

Two paths:

* `hvp_probe`   — standalone reference probe (own forward), fp32 or fp16-autocast,
                  MATH SDPA backend (flash/mem-efficient have no double backward on
                  sm_75). Used by tests and as the fused path's fallback.
* `fused_step_probe` — production path: runs the TRAINING forward/backward of a probe
                  step with create_graph=True (fp16 autocast + GradScaler scale), hands
                  the (scaled) gradient to the optimizer as usual, then pays only ONE
                  extra backward for Hv on the full batch. The scaler's loss scale S
                  protects both backward passes end-to-end; Hv = second-grad / S.
                  fp16 probe accuracy vs fp32: rel err ~1e-3 (validated).

Probe products v ⊙ Hv are returned in fp32.
"""

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def hvp_probe(model, params, X, Y, probes, amp: bool = False, init_scale: float = 2.0**15):
    """Standalone probe: [(param, v ⊙ Hv)] for the loss on (X, Y). Reference path."""
    was_training = model.training
    model.eval()
    try:
        scale = init_scale if amp else 1.0
        for _ in range(4):
            with sdpa_kernel(SDPBackend.MATH):
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = model(X, Y)
                grads = torch.autograd.grad(loss * scale, params, create_graph=True)
                gv = sum((g * v).sum() for g, v in zip(grads, probes))
                hvs = torch.autograd.grad(gv, params)
            if not amp or all(torch.isfinite(h).all() for h in hvs):
                break
            scale /= 16.0
        hvs = [h / scale for h in hvs]
    finally:
        if was_training:
            model.train()
    return [(p, (hv * v).to(p.dtype)) for p, hv, v in zip(params, hvs, probes)]


def directional_probe(model, params, X, Y, vs):
    """Exact batch scalars along direction v: (g·v, vᵀHv) from one fp32 HVP.

    This is the noise-free regime of the machinery: a scalar quadratic form has
    no estimator variance (Lean: `directional_newton_optimal` consumes these).
    """
    was_training = model.training
    model.eval()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            with torch.amp.autocast("cuda", enabled=False):
                loss = model(X, Y)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                gv = sum((g * v).sum() for g, v in zip(grads, vs))
                hvs = torch.autograd.grad(gv, params)
        vHv = sum((hv * v).sum() for hv, v in zip(hvs, vs)).item()
    finally:
        if was_training:
            model.train()
    return gv.item(), vHv


def fused_step_probe(model, params, X, Y, probes, scaler):
    """Probe-step fused path.

    Runs forward + first backward (create_graph) under fp16 autocast with the scaler's
    current scale, installs scaled grads into p.grad (so scaler.unscale_/step work as
    on any step), then one extra backward yields Hv.

    Returns (loss_detached, probe_products or None). probe_products is None if the
    second backward overflowed — caller should fall back to `hvp_probe(amp=False)`.
    """
    S = scaler.get_scale()
    with sdpa_kernel(SDPBackend.MATH):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = model(X, Y)
        grads = torch.autograd.grad(loss.float() * S, params, create_graph=True)
        for p, g in zip(params, grads):
            p.grad = g.detach()  # scaled, as scaler expects; unscale_ divides by S
        gv = sum((g * v).sum() for g, v in zip(grads, probes))
        hvs = torch.autograd.grad(gv, params)
    if not all(torch.isfinite(h).all() for h in hvs):
        return loss.detach(), None
    return loss.detach(), [(p, (hv * v).float() / S)
                           for p, hv, v in zip(params, hvs, probes)]
