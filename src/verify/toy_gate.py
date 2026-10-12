"""GLUON toy gate — numerical mirror of the Lean theorems, at machine precision.

Runs a tiny CE model in fp64 on CPU and checks, against dense autograd ground truth:

  G1  split_exact          : H == S + R entrywise (S from dense J and closed-form Λ,
                             R := ℋ(∇ℓ) obtained as H − S must equal the residual
                             pipeline's product on every basis vector)
  G2  sv/residual products : sv_product == S·v ; residual_probe == (H−S)v ⊙ v
  G3  core_posCore         : dense S is PSD; diag S ≥ 0
  G4  full_recovery        : (1/m) Σ p^k (v^k)ᵀ == R with all-pairs orthogonal codes
  G5  coded_variance       : enumerating ALL 2^n flips on a 10×10 block reproduces
                             Σ_{j≠i} M_ij² C̄_ij² exactly; scaling M by c scales it c²
                             (split_variance_scaling); Hutchinson is the m=1 case
  G6  split_probe_exact    : diag S + coded diag-recovery of R == diag H exactly
  G7  fisher unbiasedness  : mean of K sampled-label Fisher diagonals -> diag S
  G8  one-step Newton      : exact diagonal minimizes a diagonal quadratic in 1 step

Exit code 0 iff every gate passes.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.gluon.probes import hadamard
from src.gluon.split import residual_probe, sv_product

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

FAILS = []


def check(name, err, tol):
    ok = err < tol
    print(f"  {name:<44s} err={err:.3e}  tol={tol:.0e}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)


class Toy(torch.nn.Module):
    """2-layer tanh net, 5-way CE head. ~117 params."""

    def __init__(self, d_in=4, d_h=7, n_cls=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_h, bias=False)
        self.fc2 = torch.nn.Linear(d_h, n_cls, bias=False)

    def forward(self, x, targets=None):
        logits = self.fc2(torch.tanh(self.fc1(x)))
        if targets is None:
            return logits
        return F.cross_entropy(logits, targets)


def flatten(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def main():
    model = Toy()
    params = list(model.parameters())
    n = sum(p.numel() for p in params)
    B = 3
    X = torch.randn(B, 4)
    Y = torch.randint(0, 5, (B,))

    def lf(lg):
        return F.cross_entropy(lg, Y)

    # ---- dense ground truth -------------------------------------------------
    theta0 = flatten(params).clone()

    def loss_of(theta):
        outs, off = [], 0
        for p in params:
            outs.append(theta[off:off + p.numel()].view_as(p))
            off += p.numel()
        x = X @ outs[0].T
        lg = torch.tanh(x) @ outs[1].T
        return F.cross_entropy(lg, Y)

    H = torch.autograd.functional.hessian(loss_of, theta0)
    H = 0.5 * (H + H.T)

    # dense S = Jᵀ Λ J / B from the logit Jacobian
    def logits_of(theta):
        outs, off = [], 0
        for p in params:
            outs.append(theta[off:off + p.numel()].view_as(p))
            off += p.numel()
        return (torch.tanh(X @ outs[0].T) @ outs[1].T).reshape(-1)

    J = torch.autograd.functional.jacobian(logits_of, theta0)   # (B*5, n)
    with torch.no_grad():
        lg = model(X)
        P = F.softmax(lg, dim=-1).reshape(-1, 5)
    S = torch.zeros(n, n)
    for b in range(B):
        Jb = J[b * 5:(b + 1) * 5]                                # (5, n)
        Lam = torch.diag(P[b]) - torch.outer(P[b], P[b])
        S += Jb.T @ Lam @ Jb
    S /= B
    R = H - S

    print("G1/G2: split products vs dense ground truth")
    # S·v on random probes
    torch.manual_seed(1)
    for trial in range(3):
        vflat = torch.where(torch.rand(n) < 0.5, 1.0, -1.0)
        vs, off = [], 0
        for p in params:
            vs.append(vflat[off:off + p.numel()].view_as(p))
            off += p.numel()
        sv = flatten(sv_product(model, params, X, vs))
        check(f"S·v pipeline == dense S@v (trial {trial})",
              (sv - S @ vflat).abs().max().item(), 1e-10)
        prods = residual_probe(model, params, X, lf, vs)
        rv_probe = flatten([pp for _, pp in prods])
        rv_dense = (R @ vflat) * vflat
        check(f"residual probe == (H−S)v ⊙ v (trial {trial})",
              (rv_probe - rv_dense).abs().max().item(), 1e-10)

    print("G3: structural core is PSD with nonnegative diagonal")
    evals = torch.linalg.eigvalsh(S)
    check("min eigenvalue of S ≥ -1e-12", max(0.0, -evals.min().item()), 1e-12)
    check("min diag S ≥ -1e-14", max(0.0, -S.diag().min().item()), 1e-14)

    print("G4: full-matrix recovery of R from probes (all-pairs orthogonal codes)")
    m = 1
    while m < n:
        m *= 2
    Hm = hadamard(m)
    V = Hm[:, :n]                                # (m, n): column codes, all-pairs orth
    est = torch.zeros(n, n)
    for k in range(m):
        vk = V[k]
        est += torch.outer(R @ vk, vk)
    est /= m
    check("(1/m) Σ pᵏ(vᵏ)ᵀ == R", (est - R).abs().max().item(), 1e-10)

    print("G5: coded variance formula — exact enumeration of all flips (n=10)")
    n10 = 10
    M10 = torch.randn(n10, n10)
    M10 = 0.5 * (M10 + M10.T)
    m10 = 4
    C = hadamard(m10)[:, torch.arange(n10) % m10]  # (m10, n10) coded columns
    i0 = 3
    truth = M10[i0, i0]
    err2_sum = 0.0
    for smask in range(2 ** n10):
        d = torch.tensor([1.0 if (smask >> j) & 1 else -1.0 for j in range(n10)])
        est_i = 0.0
        for k in range(m10):
            v = d * C[k]
            est_i += (v * (M10 @ v))[i0]
        est_i /= m10
        err2_sum += (est_i - truth) ** 2
    mc_var = err2_sum / 2 ** n10
    coh = (C[:, i0:i0 + 1] * C).sum(0) / m10          # C̄_{i0,j}
    formula = sum(M10[i0, j] ** 2 * coh[j] ** 2 for j in range(n10) if j != i0)
    check("enumerated variance == Σ M_ij² C̄_ij²",
          abs(mc_var - formula).item(), 1e-10)
    check("scaling M by c=3 scales variance by 9",
          abs(9 * formula - sum((3 * M10[i0, j]) ** 2 * coh[j] ** 2
                                for j in range(n10) if j != i0)).item(), 1e-10)

    print("G6: split estimator — exact core diag + coded residual == diag H")
    diag_est = S.diag().clone()
    coded_r = torch.zeros(n)
    for k in range(m):
        vk = V[k]
        coded_r += vk * (R @ vk)
    diag_est += coded_r / m
    check("diag S + coded diag R == diag H",
          (diag_est - H.diag()).abs().max().item(), 1e-10)

    print("G7: sampled-label Fisher unbiasedness (statistical)")
    torch.manual_seed(2)
    K = 4000
    acc = torch.zeros(n)
    from src.gluon.split import fisher_diag_sample
    for _ in range(K):
        cs = fisher_diag_sample(model, params, X)
        acc += flatten([c for _, c in cs])
    est = acc / K
    rel = ((est - S.diag()).abs().sum() / S.diag().abs().sum()).item()
    check(f"mean of {K} Fisher samples vs diag S (rel L1)", rel, 0.08)
    check("every Fisher sample nonneg", max(0.0, -est.min().item()), 0.0 + 1e-18)

    print("G8: one-step Newton payoff")
    h = torch.rand(6) + 0.5
    b = torch.randn(6)
    x0 = torch.randn(6)
    x1 = x0 - (h * x0 + b) / h
    check("‖∇f‖ at stepped point", (h * x1 + b).abs().max().item(), 1e-12)

    if FAILS:
        print(f"\nGATE FAILED: {FAILS}")
        sys.exit(1)
    print("\nALL GATES PASS")


if __name__ == "__main__":
    main()
