"""The split channels of GLUON (theory.md §3, §5-§6).

The loss Hessian decomposes exactly as H = S + R (Lean: `split_exact`):
  S = Jᵀ (∇²_logits ℓ) J   — structural core: PSD, known without any H product
  R = ℋ(∇ℓ)                — residual: linear in the loss gradient

Two channels estimate the two diagonals:

* CORE (`fisher_diag_sample`): for the cross-entropy family the logit Hessian is
  Λ = diag(p) − ppᵀ regardless of targets (targets enter CE linearly), so sampled-label
  gradients give an UNBIASED, ELEMENTWISE-NONNEGATIVE estimate of diag S:
      E[N · g̃ ⊙ g̃] = diag S,   g̃ = ∇θ mean-CE(logits, ŷ),  ŷ ~ softmax(logits).
  One forward + one backward; no double backward. Nonnegativity mirrors
  Lean `core_posCore` / `posCore_diag_nonneg`: the preconditioner base cannot flip sign.

* RESIDUAL (`residual_probe`): coded sign probes applied to R alone (Lean
  `split_probe_unbiased`, `split_variance_residual_only`): R·v = H·v − S·v with
      H·v  by double backward,
      S·v  = Jᵀ Λ (J v):  Jv by the dummy double-VJP trick, Λ applied in closed form,
             one final VJP.
  Probing R instead of H is the point: the estimator's variance scales with ‖∇ℓ‖²
  (Lean `split_variance_scaling`) and disappears at critical residual
  (`split_exact_at_critical`).

All probe math runs in fp32 under the MATH SDPA backend (double backward support).
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _up(t):
    """Lift half precisions to fp32; leave fp32/fp64 untouched (gate runs fp64)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _logit_lambda_apply(p, u, n_pos):
    """Apply the mean-CE logit Hessian in closed form: Λu = (p⊙u − p(p·u)) / N."""
    inner = (p * u).sum(dim=-1, keepdim=True)
    return (p * u - p * inner) / n_pos


@torch.enable_grad()
def fisher_diag_sample(model, params, X):
    """One unbiased nonnegative sample of diag S. Returns [(param, tensor fp32)].

    Works for any loss in the CE family (plain, label-smoothed, mixup): the logit
    Hessian is target-independent, so sampling labels from the model's own softmax
    reproduces exactly the structural core of the actual training loss.
    """
    was_training = model.training
    model.eval()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(X)
            V = logits.size(-1)
            flat = logits.reshape(-1, V)
            n_pos = flat.size(0)
            with torch.no_grad():
                probs = F.softmax(_up(flat), dim=-1)
                yhat = torch.multinomial(probs, 1).squeeze(-1)
            loss_s = F.cross_entropy(flat, yhat)
            grads = torch.autograd.grad(loss_s, params)
    finally:
        if was_training:
            model.train()
    return [(p, (_up(g) ** 2) * n_pos) for p, g in zip(params, grads)]


@torch.enable_grad()
def residual_probe(model, params, X, loss_from_logits, probes):
    """[(param, (Rv) ⊙ v)] for R = H − S of the loss at the current point. fp32.

    `loss_from_logits(logits) -> scalar` must be the exact training loss (mean
    reduction) as a function of the logits; its logit Hessian must be the CE family's
    Λ/N (true for plain/smoothed/mixup CE — targets are linear in the logits).
    """
    was_training = model.training
    model.eval()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(X)
            n_pos = logits.reshape(-1, logits.size(-1)).size(0)
            loss = loss_from_logits(logits)

            # H·v : double backward through the loss
            g = torch.autograd.grad(loss, params, create_graph=True)
            gv = sum((gi * vi).sum() for gi, vi in zip(g, probes))
            Hv = torch.autograd.grad(gv, params, retain_graph=True)

            # J·v : dummy double-VJP (gθ = Jᵀ·dummy is linear in dummy;
            #        d⟨gθ, v⟩/d dummy = J v)
            dummy = torch.zeros_like(logits, requires_grad=True)
            gtheta = torch.autograd.grad(logits, params, grad_outputs=dummy,
                                         create_graph=True, retain_graph=True)
            sv_lin = sum((gi * vi).sum() for gi, vi in zip(gtheta, probes))
            (Jv,) = torch.autograd.grad(sv_lin, dummy, retain_graph=True)

            # S·v = Jᵀ Λ (J v) with Λ applied in closed form at the softmax point
            p_soft = F.softmax(_up(logits).detach(), dim=-1)
            t = _logit_lambda_apply(p_soft, _up(Jv), n_pos).to(logits.dtype)
            Sv = torch.autograd.grad(logits, params, grad_outputs=t)
    finally:
        if was_training:
            model.train()
    return [(p, ((_up(hv) - _up(sv)) * v))
            for p, hv, sv, v in zip(params, Hv, Sv, probes)]


@torch.enable_grad()
def sv_product(model, params, X, probes):
    """S·v alone (structural-core product) — used by gates/tests."""
    was_training = model.training
    model.eval()
    try:
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(X)
            n_pos = logits.reshape(-1, logits.size(-1)).size(0)
            dummy = torch.zeros_like(logits, requires_grad=True)
            gtheta = torch.autograd.grad(logits, params, grad_outputs=dummy,
                                         create_graph=True, retain_graph=True)
            sv_lin = sum((gi * vi).sum() for gi, vi in zip(gtheta, probes))
            (Jv,) = torch.autograd.grad(sv_lin, dummy, retain_graph=True)
            p_soft = F.softmax(_up(logits).detach(), dim=-1)
            t = _logit_lambda_apply(p_soft, _up(Jv), n_pos).to(logits.dtype)
            Sv = torch.autograd.grad(logits, params, grad_outputs=t)
    finally:
        if was_training:
            model.train()
    return [_up(sv) for sv in Sv]
