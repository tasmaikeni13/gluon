"""Probe machinery for HADRON: Hadamard codes + Latin coloring + flip randomization.

Implements theory.md §3/§6 (Lean: cycle_master_identity, exact_recovery,
coded_probe_unbiased, coded_estimator_variance):

* Each coordinate gets a CODE = a column index of a Sylvester Hadamard matrix H_m.
* Weight matrices use Latin coloring  code(a, b) = (a + b) mod m  — coordinates in the
  same row or same column (the K-FAC-dominant interference pairs) receive distinct,
  hence orthogonal, codes for offsets < m.
* Vectors (norms) use band coloring  code(i) = i mod m.
* Probe r of a cycle: v = d ⊙ H_m[r, code]  with a global sign flip d fixed for the
  whole cycle (re-derived deterministically from the cycle seed, so no O(n) storage).
Over a full cycle r = 0..m−1:  Σ_r v^r_i v^r_j = d_i d_j · m·[code_i = code_j]  — exact
cancellation for distinct-code pairs, unbiased randomization for collisions.
"""

import torch


def hadamard(m: int) -> torch.Tensor:
    """Sylvester Hadamard matrix of order m (m must be a power of 2). ±1 float32."""
    assert m > 0 and (m & (m - 1)) == 0, "m must be a power of 2"
    H = torch.ones(1, 1)
    while H.shape[0] < m:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def code_tensor(shape: torch.Size, m: int, device) -> torch.Tensor:
    """Code assignment for one parameter tensor (int64 in [0, m))."""
    if len(shape) >= 2:
        rows = shape[0]
        cols = int(torch.tensor(shape[1:]).prod().item())
        a = torch.arange(rows, device=device).unsqueeze(1)
        b = torch.arange(cols, device=device).unsqueeze(0)
        return ((a + b) % m).reshape(shape)
    idx = torch.arange(shape[0] if len(shape) == 1 else 1, device=device)
    return (idx % m).reshape(shape)


def flip_signs(shape: torch.Size, seed: int, device, dtype) -> torch.Tensor:
    """Deterministic ±1 flip tensor for a given cycle seed (no persistent storage)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return (torch.randint(0, 2, shape, generator=g, device=device,
                          dtype=torch.int8).to(dtype) * 2 - 1)


class ProbeGenerator:
    """Generates probe tensors for a list of parameters, cycling Hadamard rows."""

    def __init__(self, params, m: int = 16, seed: int = 1234):
        self.m = m
        self.seed = seed
        self.params = list(params)
        self.had = {}
        self.codes = {}
        for i, p in enumerate(self.params):
            dev = p.device
            if dev not in self.had:
                self.had[dev] = hadamard(m).to(dev)
            self.codes[i] = code_tensor(p.shape, m, dev)

    def probe(self, cycle_idx: int, r: int) -> list[torch.Tensor]:
        """Probe r ∈ [0, m) of cycle `cycle_idx`, one ±1 tensor per parameter."""
        assert 0 <= r < self.m
        out = []
        for i, p in enumerate(self.params):
            row = self.had[p.device][r]  # (m,) of ±1
            d = flip_signs(p.shape,
                           self.seed + 1_000_003 * cycle_idx + 7919 * i,
                           p.device, torch.float32)
            out.append(d * row[self.codes[i]])
        return out
