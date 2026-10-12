"""GLUON test suite: probes/codes, optimizer mechanics, kernel parity, models."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.gluon.optimizer import Gluon
from src.gluon.probes import ProbeGenerator, code_tensor, hadamard
from src.model.gpt import GPT, GPTConfig
from src.model.vit import ViT, ViTConfig

CUDA = torch.cuda.is_available()


# ---------------- probes / codes ----------------

def test_hadamard_orthogonal():
    H = hadamard(16)
    assert torch.allclose(H @ H.T, 16 * torch.eye(16))
    assert set(H.unique().tolist()) == {-1.0, 1.0}


def test_latin_coloring_row_col_orthogonal():
    """Same-row / same-col coordinate pairs get distinct codes (offset < m)."""
    m = 8
    codes = code_tensor(torch.Size([5, 12]), m, "cpu")
    for a in range(5):
        row = codes[a]
        for j in range(12):
            for l in range(j + 1, min(j + m, 12)):
                assert row[j] != row[l]
    for b in range(12):
        col = codes[:, b]
        for i in range(5):
            for k in range(i + 1, min(i + m, 5)):
                assert col[i] != col[k]


def test_cycle_coherence_cancels():
    """Σ_r v^r_i v^r_j over a full cycle = m·[code_i = code_j]·d_i d_j."""
    torch.manual_seed(0)
    p = torch.zeros(6, 10)
    gen = ProbeGenerator([p], m=8, seed=7)
    acc = None
    probes = [gen.probe(0, r)[0] for r in range(8)]
    flat = [v.reshape(-1) for v in probes]
    n = flat[0].numel()
    coh = sum(v.outer(v) for v in flat)
    codes = gen.codes[0].reshape(-1)
    for i in range(0, n, 7):
        for j in range(0, n, 11):
            if codes[i] != codes[j]:
                assert abs(coh[i, j].item()) < 1e-6, (i, j)
            elif i == j:
                assert coh[i, j].item() == 8.0


# ---------------- optimizer mechanics ----------------

def _tiny_problem():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4, 3))
    return w


def test_gluon_matches_reference_step():
    """Fused=False python path: one step reproduces the formula by hand."""
    w = _tiny_problem()
    opt = Gluon([w], lr=0.1, betas=(0.9, 0.95), gamma=0.5, weight_decay=0.0,
                fused=False)
    s_sample = torch.rand_like(w) + 0.1
    opt.accumulate_core([(w, s_sample)])
    r_prod = 0.05 * torch.randn_like(w)
    for _ in range(opt.cycle_m):
        opt.accumulate_residual([(w, r_prod)])
    g = torch.randn_like(w)
    w.grad = g.clone()
    w0 = w.detach().clone()
    opt.step()
    s_hat = s_sample  # one core commit, bias corr (1-0.95)/(1-0.95)=... n_core=1
    s_hat = (1 - 0.95) * s_sample / (1 - 0.95 ** 1)
    r_hat = ((1 - 0.95) * r_prod) / (1 - 0.95 ** 1)  # one commit of mean r_prod
    h_hat = s_hat + r_hat
    mu_hat = ((1 - 0.9) * g) / (1 - 0.9 ** 1)
    u = (mu_hat / (0.5 * h_hat).clamp_min(1e-12)).clamp(-1, 1)
    expect = w0 - 0.1 * u
    assert torch.allclose(w.detach(), expect, atol=1e-6), \
        (w.detach() - expect).abs().max()


def test_gluon_no_info_step_is_tiny_clip():
    w = _tiny_problem()
    opt = Gluon([w], lr=0.01, fused=False)
    w.grad = torch.randn_like(w)
    w0 = w.detach().clone()
    opt.step()
    # denom = eps ⇒ u = ±1 ⇒ |Δw| = lr exactly
    assert torch.allclose((w.detach() - w0).abs(), torch.full_like(w0, 0.01),
                          atol=1e-7)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fused_kernel_parity():
    torch.manual_seed(0)
    shapes = [(64, 33), (127,), (8, 4, 5)]
    for shape in shapes:
        p1 = torch.nn.Parameter(torch.randn(*shape, device="cuda"))
        p2 = torch.nn.Parameter(p1.detach().clone())
        g = torch.randn_like(p1)
        o1 = Gluon([p1], lr=0.07, betas=(0.9, 0.95), gamma=0.3,
                   weight_decay=0.02, fused=False)
        o2 = Gluon([p2], lr=0.07, betas=(0.9, 0.95), gamma=0.3,
                   weight_decay=0.02, fused=True)
        s = torch.rand_like(p1) + 0.05
        r = 0.02 * torch.randn_like(p1)
        for o, pp in ((o1, p1), (o2, p2)):
            o.accumulate_core([(pp, s)])
            for _ in range(o.cycle_m):
                o.accumulate_residual([(pp, r)])
        for step in range(3):
            for o, pp in ((o1, p1), (o2, p2)):
                pp.grad = g.clone()
                o.step()
        assert torch.allclose(p1.detach(), p2.detach(), atol=5e-6), \
            (p1.detach() - p2.detach()).abs().max()


# ---------------- models ----------------

def test_vit_params_and_shapes():
    m = ViT(ViTConfig())
    n = m.num_params()
    assert 38.5e6 < n < 39.6e6, n
    x = torch.randn(2, 3, 32, 32)
    lg = m(x)
    assert lg.shape == (2, 100)
    loss = m(x, torch.tensor([3, 7]))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_gpt_params():
    m = GPT(GPTConfig())
    n = m.num_params()
    assert 38.5e6 < n < 39.5e6, n
    x = torch.randint(0, 100, (2, 16))
    lg = m(x)
    assert lg.shape == (2, 16, 50304)


# ---------------- split channels on a GPU model ----------------

@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_split_channels_on_vit_gpu():
    """Residual probe pipeline runs on the real ViT (fp32) and Hv−Sv == probe
    against a finite-difference Hv sanity bound."""
    from src.gluon.split import fisher_diag_sample, residual_probe, sv_product
    torch.manual_seed(0)
    cfg = ViTConfig(n_layer=2, n_embd=64, n_head=4)
    m = ViT(cfg).cuda().double()
    params = list(m.parameters())
    X = torch.randn(4, 3, 32, 32, device="cuda", dtype=torch.float64)
    Y = torch.randint(0, 100, (4,), device="cuda")

    def lf(lg):
        return F.cross_entropy(lg, Y)

    gen = ProbeGenerator(params, m=8, seed=3)
    probes = [v.double() for v in gen.probe(0, 0)]
    prods = residual_probe(m, params, X, lf, probes)
    # cross-check: Hv by autograd here, Sv by pipeline; R v ⊙ v must match
    loss = lf(m(X))
    g = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((gi * vi).sum() for gi, vi in zip(g, probes))
    Hv = torch.autograd.grad(gv, params)
    Sv = sv_product(m, params, X, probes)
    for (p, pp), hv, sv, v in zip(prods, Hv, Sv, probes):
        ref = (hv - sv) * v
        assert (pp - ref).abs().max().item() < 1e-9

    cs = fisher_diag_sample(m, params, X)
    for p, c in cs:
        assert c.min().item() >= 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
