# GLUON — Split-Exact Curvature

**A second-order optimizer built on a proved decomposition: the loss Hessian is an
exactly-known positive core plus a gradient-sized measurable residual.**
Every mathematical claim below is machine-verified in Lean 4 + Mathlib
(`lean/Gluon.lean`, zero `sorry`, axioms = Lean's standard three; theorem names in
parentheses) and mirrored numerically by the gates in `src/verify/` and `tests/`.

---

## 1. The problem, delimited from both sides

The Hessian `H ∈ ℝ^{N×N}` of a deep loss cannot be stored: every symmetric matrix *is*
an exact Hessian (of its half-quadratic — `matrix_quad_witness`, `range_hess`), so
unstructured exact curvature costs `dim Sym(N) = N(N+1)/2` degrees of freedom. And it
cannot be read out through a single product `p = Hv` either: for every probe `v` there
is a symmetric perturbation invisible to it yet nonzero on the diagonal
(`one_probe_insufficient`). What remains affordable is a handful of Hessian-vector
products per step. GLUON is the theory of extracting the *most curvature per product* —
exactly where structure permits, at provably minimal variance where it does not.

## 2. Carriers: curvature as a composable operator

The curvature of a block `f : ℝⁿ → ℝᵐ` at a point is carried by the pair

```
𝒞(f;x) = (J, ℋ),   J = Df(x),   ℋ(λ) = ∇²⟨λ, f⟩(x)   (linear in λ, symmetric-valued)
```

with the **pullback composition law** for `h = g ∘ f`:

```
J_h = J_g J_f          ℋ_h(λ) = J_fᵀ ℋ_g(λ) J_f + ℋ_f(J_gᵀ λ)          (⋆)
```

Machine-checked: carriers form a category (`comp_assoc`, `id_comp`, `comp_id`) with
symmetry preserved; the law (⋆) is the true second derivative of the true composition —
proved through a coordinate-free second-order chain rule at Banach level and a bridge
theorem down to matrices (`chain_snd`, `ExactAt.comp`, consistency `H_consistent`).
Hessian-vector products of arbitrarily deep chains are computed by vector sweeps alone —
no `n×n` intermediate ever exists (`chainHv_eq`, `chain_entry_recovery`), and a chain's
curvature is a sum of pulled-back layer-local cores (`compChain_H`) — affine layers
contribute zero, pointwise nonlinearities a diagonal (`affineCarrier`, `diagCarrier`).

## 3. The split: what is known exactly vs. what must be measured

For a loss `ℓ ∘ f` (loss layer over features), (⋆) evaluated at the loss layer gives an
exact — not approximate — decomposition (`split_exact`):

```
H  =  S + R,      S = J_fᵀ (∇²ℓ) J_f      the structural core
                  R = ℋ_f(∇ℓ)             the residual
```

The two parts have opposite characters, and this asymmetry is the whole design:

- **S is structurally known.** For convex `ℓ` it is positive semidefinite
  (`core_posCore`) with nonnegative diagonal (`posCore_diag_nonneg`) and rank at most
  the feature width (`rank_core_le`). Its diagonal admits exact or unbiased-nonnegative
  evaluation without any Hessian product (for softmax/cross-entropy, `S` is the Fisher;
  sampled-label gradients give `E[g̃ ⊙ g̃] = diag S`, each sample `≥ 0`).
- **R is linear in the loss gradient** (`residual_smul`, `residual_zero`): it *shrinks
  as training progresses* and vanishes at critical points of the loss.

## 4. Tomographic probing: measurement with designed interference

What structure does not give, sign-probe products give. For a `±1` probe, dividing the
response by the probe *is* multiplying by it (`probe_div_eq_mul`), and

```
v_i (Mv)_i = M_ii + Σ_{j≠i} M_ij v_i v_j          (master_identity)
```

— the diagonal plus an interference term controlled, over a cycle of `m` coded probes,
by the **code coherence** `Ĉ_ij = Σ_k v^k_i v^k_j`, a *design quantity* (`cycle_identity`).
Machine-checked consequences:

- Codes orthogonal across interacting pairs ⇒ **exact** diagonal, deterministically
  (`diag_recovery_exact`); all-pairs orthogonal codes reconstruct the **full matrix**
  as `M = (1/m) Σ_k p^k (v^k)ᵀ` (`full_recovery`) at the provably necessary cost
  `m ≥ n` (`full_recovery_cost`); banded interference needs only `m = 2`
  (`banded_two_probe`) — probe complexity tracks the interference graph, not the size.
- A single uniform random sign flip de-biases *any* code cycle (`coded_unbiased`), with
  variance **exactly**

  ```
  Var_i = Σ_{j≠i} M_ij² · C̄_ij²,   C̄_ij = Ĉ_ij / m     (coded_variance)
  ```

  Hutchinson estimation is the degenerate one-probe case (`hutchinson_unbiased`,
  `hutchinson_variance`); orthogonal pairs contribute exactly zero
  (`orthogonal_zero_variance`); coded never exceeds Hutchinson (`coded_le_hutchinson`).

## 5. The fusion: probe the residual, not the Hessian

GLUON's central move: **take `S` structurally, spend every probe on `R` alone.**
The bound theorems (all machine-checked, §8 of the Lean file):

| Claim | Theorem |
|---|---|
| exact core diag + residual probes orthogonal on `R`'s (sparser) graph ⇒ exact `diag H` | `split_probe_exact` |
| unbiased for `diag H` under any codes, any flip | `split_probe_unbiased` |
| estimator variance = the **residual's** variance only — `S` contributes zero | `split_variance_residual_only` |
| variance scales as the **squared loss gradient**: `Var(c·∇ℓ) = c² Var(∇ℓ)` | `split_variance_scaling` |
| at critical residual: deterministic exactness for every flip, any codes, even one probe | `split_exact_at_critical` |
| measured ray curvature splits `vᵀHv = vᵀSv + vᵀRv` with `vᵀSv ≥ 0` guaranteed | `directional_split` |

Probing `H` directly has no such mechanism: its variance retains the full structural
term `Σ_j S_ij² C̄_ij²` *forever*, gradient-independent. The split estimator's error
budget contracts along the very trajectory the optimizer produces — an estimator that
**converges to exactness as training converges**, with the noise floor tied to `‖∇ℓ‖²`.

Payoffs (why the recovered quantities matter): the exact diagonal of a diagonal PD
quadratic minimizes it in one preconditioned step from any start
(`diag_newton_one_step`); the measured directional step `η* = (g·v)/(vᵀHv)` — both
scalars exact from one product — is optimal along the ray (`directional_newton_optimal`).

## 6. The GLUON optimizer

State: momentum `μ` (EMA β₁), curvature diagonal `ĥ` fused from the two channels.

1. **Core channel** (every probe event; cheap — one backward, no double-backward):
   sampled-label gradient `g̃` on a small sub-batch; commit `ŝ ← EMA(g̃ ⊙ g̃)`.
   Always `≥ 0` — the preconditioner base cannot flip sign (`posCore_diag_nonneg`).
2. **Residual channel** (every `k` steps): one fp32 residual-vector product
   `Rv = Hv − Sv` (`Hv` by double backward; `Sv` by JVP→`∇²ℓ`→VJP) with probe
   `v = d ⊙ c^(r)`, `r` cycling through `m = 16` Hadamard codes; codes assigned
   per-tensor by Latin coloring `code(a,b) = (a+b) mod m` so same-row/same-column
   neighbours — the dominant interference — get orthogonal codes; global flip `d`
   re-derived per cycle. Accumulate `r̂ ← EMA of cycle-mean (Rv ⊙ v)`.
3. **Update** (clipped preconditioned momentum, decoupled weight decay), fused kernel:

   ```
   ĥ = ŝ + r̂          θ ← θ − η·clamp( μ̂ / max(γ·ĥ, ε), −1, 1 ) − η·λ_wd·θ
   ```

The clip is the trust region: coordinates whose curvature is still noise-dominated take
bounded sign-momentum steps; the confident high-curvature tail gets true Newton braking.
The split moves coordinates from the first regime to the second as `‖∇ℓ‖` falls.

## 7. Scope and honesty

- Lean covers the estimator algebra and probability (finite, exact), the split, the
  chain rule tower, and the quadratic payoffs — **not** convergence of nonconvex
  training; that is what the benchmarks measure (`REPORT.md`).
- Smoothness (`C²`) is assumed at the probed point: GELU/tanh/SiLU networks qualify;
  ReLU kinks do not (measure-zero, but stated).
- The `Sv` product is exact in exact arithmetic; in fp32 its implementation error is
  measured by the toy gate, not assumed away.
- Cost claims (sweep counts, wall-clock overhead) are engineering measurements, not
  theorems.

## 8. Artifacts

- `lean/Gluon.lean` — the verified development (≈1150 lines, 40+ theorems;
  compile: `lake env lean lean/Gluon.lean` against Mathlib v4.31).
- `src/` — optimizer, models (39M decoder LM; 39M ViT), training, gates, plots.
- `tests/` — numerical mirror of every theorem group + kernel parity + gates.
- `REPORT.md` — WikiText-103 and CIFAR-100 benchmarks vs tuned AdamW, logs in `runs/`,
  figures in `figures/`.
