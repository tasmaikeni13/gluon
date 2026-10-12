"""GLUON optimizer — clipped preconditioned momentum on a split-exact curvature
diagonal (theory.md §6).

Curvature state has two channels mirroring the proved split H = S + R:

* s : EMA of unbiased NONNEGATIVE core samples (sampled-label Fisher diagonals),
      committed once per core event. The base of the preconditioner cannot flip
      sign (Lean `posCore_diag_nonneg`).
* r : EMA over COMPLETED residual-probe cycles of coded products v ⊙ (Rv); within a
      cycle the m products are equally weighted (exact interference cancellation for
      orthogonal codes, Lean `diag_recovery_exact` / `split_probe_exact`); before the
      first commit the partial-cycle mean is used (unbiased for any prefix, Lean
      `split_probe_unbiased`).

ĥ = ŝ + r̂  estimates diag H = diag S + diag R exactly-in-expectation, with variance
carried by the residual only (`split_variance_residual_only`) — shrinking as ‖∇ℓ‖²
(`split_variance_scaling`).

Update:  θ ← θ − lr·clamp( μ̂ / max(γ·ĥ, ε), −1, 1 )  − lr·λ·θ   (decoupled decay)
"""

import torch

from .fused_step import HAS_TRITON

if HAS_TRITON:
    from .fused_step import gluon_step_fused


class Gluon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.96, 0.95), beta_core=0.95,
                 gamma=0.02, eps=1e-12, weight_decay=0.0, cycle_m=16, fused=None,
                 r_weight=1.0):
        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, cycle_m=cycle_m)
        super().__init__(params, defaults)
        self.cycle_m = cycle_m
        self.beta_core = beta_core
        self.r_weight = r_weight  # weight on the residual channel in ĥ = ŝ + w·r̂
        self.total_probes = 0   # residual probes accumulated (across cycles)
        self.n_commits = 0      # completed residual cycles committed into r
        self.n_core = 0         # core samples committed into s
        self.fused = HAS_TRITON if fused is None else (fused and HAS_TRITON)

    # ---- curvature side ----------------------------------------------------

    def _state(self, p):
        st = self.state[p]
        if len(st) == 0:
            st["mu"] = torch.zeros_like(p)
            st["s"] = torch.zeros_like(p)
            st["r"] = torch.zeros_like(p)
            st["racc"] = torch.zeros_like(p)
            st["t"] = 0
        return st

    @torch.no_grad()
    def accumulate_core(self, core_samples):
        """Commit one nonnegative core sample per parameter into the s-EMA."""
        for p, cs in core_samples:
            st = self._state(p)
            st["s"].mul_(self.beta_core).add_(cs, alpha=1 - self.beta_core)
        self.n_core += 1

    @torch.no_grad()
    def accumulate_residual(self, probe_products):
        """Add one residual probe product v ⊙ (Rv) per parameter; commit at cycle end."""
        for p, pp in probe_products:
            st = self._state(p)
            st["racc"].add_(pp)
        self.total_probes += 1
        if self.total_probes % self.cycle_m == 0:
            beta2 = self.param_groups[0]["betas"][1]
            for group in self.param_groups:
                for p in group["params"]:
                    st = self._state(p)
                    st["r"].mul_(beta2).add_(st["racc"], alpha=(1 - beta2) / self.cycle_m)
                    st["racc"].zero_()
            self.n_commits += 1

    def _channel_scales(self):
        """(sscale, use_acc, rscale): bias corrections for the two channels."""
        beta2 = self.param_groups[0]["betas"][1]
        sscale = 0.0 if self.n_core == 0 else 1.0 / (1 - self.beta_core ** self.n_core)
        w = self.r_weight
        if self.n_commits == 0:
            rpart = self.total_probes % self.cycle_m
            if rpart == 0:
                return sscale, False, 0.0
            return sscale, True, w / rpart
        return sscale, False, w / (1 - beta2 ** self.n_commits)

    def _h_hat(self, st):
        """Bias-corrected split curvature ĥ = ŝ + r̂ (None if no info at all)."""
        sscale, use_acc, rscale = self._channel_scales()
        if sscale == 0.0 and rscale == 0.0:
            return None
        rsrc = st["racc"] if use_acc else st["r"]
        return st["s"] * sscale + rsrc * rscale

    # ---- parameter update --------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        sscale, use_acc, rscale = self._channel_scales()
        no_info = (sscale == 0.0 and rscale == 0.0)
        for group in self.param_groups:
            lr = group["lr"]
            beta1, _ = group["betas"]
            gamma, eps, wd = group["gamma"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._state(p)
                st["t"] += 1
                bias1 = 1 - beta1 ** st["t"]
                if self.fused and p.is_cuda and p.dtype == torch.float32:
                    rsrc = st["racc"] if use_acc else st["r"]
                    gluon_step_fused(p, p.grad, st["mu"], st["s"], rsrc, lr,
                                     beta1, bias1, sscale, rscale, gamma, eps, wd)
                    continue
                st["mu"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                mu_hat = st["mu"] / bias1
                if no_info:
                    denom = torch.full_like(p, eps)
                else:
                    h_hat = self._h_hat(st)
                    denom = (gamma * h_hat).clamp_min(eps)
                u = (mu_hat / denom).clamp_(-1.0, 1.0)
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss

    # ---- checkpointing -------------------------------------------------------

    def state_dict(self):
        sd = super().state_dict()
        sd["gluon_extra"] = {"total_probes": self.total_probes,
                             "n_commits": self.n_commits,
                             "n_core": self.n_core}
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        extra = state_dict.pop("gluon_extra", None)
        super().load_state_dict(state_dict)
        if extra is not None:
            self.total_probes = extra["total_probes"]
            self.n_commits = extra["n_commits"]
            self.n_core = extra["n_core"]

    # ---- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def stats(self):
        """Curvature/update diagnostics (sparse calls; does device sync)."""
        hs, ss, rs, clip_n, tot = [], [], [], 0, 0
        for group in self.param_groups:
            beta1, _ = group["betas"]
            gamma, eps = group["gamma"], group["eps"]
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                h_hat = self._h_hat(st)
                if h_hat is None:
                    continue
                sscale, use_acc, rscale = self._channel_scales()
                hs.append(h_hat.flatten())
                ss.append((st["s"] * sscale).flatten())
                rs.append(((st["racc"] if use_acc else st["r"]) * rscale).flatten())
                mu_hat = st["mu"] / (1 - beta1 ** max(st["t"], 1))
                u = mu_hat / (gamma * h_hat).clamp_min(eps)
                clip_n += (u.abs() >= 1).sum().item()
                tot += u.numel()
        if not hs:
            return {}
        h = torch.cat(hs).float()
        s = torch.cat(ss).float()
        r = torch.cat(rs).float()
        idx = torch.randint(0, h.numel(), (min(200_000, h.numel()),), device=h.device)
        q = torch.quantile(h[idx], torch.tensor([0.05, 0.5, 0.95], device=h.device))
        return {
            "h_p05": q[0].item(), "h_p50": q[1].item(), "h_p95": q[2].item(),
            "h_neg_frac": (h < 0).float().mean().item(),
            "r_share": (r.abs().sum() / (s.sum() + r.abs().sum() + 1e-30)).item(),
            "clip_frac": clip_n / max(tot, 1),
            "n_commits": self.n_commits, "n_core": self.n_core,
        }
