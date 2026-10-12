"""Fused Triton kernel for the GLUON parameter update.

One memory pass per tensor:
    mu    ← β₁·mu + (1−β₁)·g                        (stored)
    μ̂     = mu / bias1                               bias1 = 1 − β₁^t
    ĥ     = S·sscale + Rsrc·rscale                   (two-channel curvature:
                                                      nonneg core + residual)
    u     = clamp( μ̂ / max(γ·ĥ, ε), −1, 1 )
    p     ← p·(1 − lr·λ) − lr·u                      (stored)

The Python loop in optimizer.py is the reference; parity enforced by tests.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _gluon_step(P, G, MU, S, RSRC, n_elem,
                    lr, beta1, bias1, sscale, rscale, gamma, eps, decay,
                    BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elem
        g = tl.load(G + offs, mask=mask, other=0.0)
        mu = tl.load(MU + offs, mask=mask, other=0.0)
        p = tl.load(P + offs, mask=mask, other=0.0)
        s = tl.load(S + offs, mask=mask, other=0.0)
        r = tl.load(RSRC + offs, mask=mask, other=0.0)

        mu = beta1 * mu + (1.0 - beta1) * g
        mu_hat = mu / bias1
        h = s * sscale + r * rscale
        denom = tl.maximum(gamma * h, eps)
        u = tl.minimum(tl.maximum(mu_hat / denom, -1.0), 1.0)
        p = p * (1.0 - lr * decay) - lr * u

        tl.store(MU + offs, mu, mask=mask)
        tl.store(P + offs, p, mask=mask)

    def gluon_step_fused(p, g, mu, s, rsrc, lr, beta1, bias1, sscale, rscale,
                         gamma, eps, decay):
        n = p.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gluon_step[grid](p.view(-1), g.reshape(-1), mu.view(-1), s.view(-1),
                          rsrc.view(-1), n, lr, beta1, bias1, sscale, rscale,
                          gamma, eps, decay, BLOCK=BLOCK)
