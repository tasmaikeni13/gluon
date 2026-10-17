# Rocora End-to-End Testing Infrastructure (TEST_INFRA.md)

## 1. Overview & Architectural Principles

The Rocora End-to-End (E2E) Test Suite provides authoritative, requirement-driven verification for the Rocora codebase. Tests are designed under strict **opaque-box principles**, evaluating software behavior, interface contracts, mathematical identities, and numerical invariants directly derived from `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md`.

### Core Testing Mandates
1. **Opaque-Box Requirement Verification**: Tests execute against external API boundaries (`init_params`, `logits`, `hidden`, `loss_sum`, `update`, `core_gradients`, `residual_diagonal`, `hutchinson_diagonal`, `apply_rope`, `rms_norm`, `load_config`). They never rely on internal private monkey-patching or implementation shortcuts.
2. **Deterministic Parity & Authoritative Oracles**: Every test verifies against explicit, closed-form mathematical equations or scientific reference programs:
   - RoPE relative position invariance: $\langle \text{RoPE}(q_m), \text{RoPE}(k_n) \rangle = q_m^T R_{m-n} k_n$.
   - RMSNorm invariance: zero mean-centering, float32 accumulation, scale $\gamma = 1$.
   - SwiGLU parameter parity: $3 \times d \times (\frac{8}{3}d) = 8 d^2$ for $d$ multiple of 3.
   - Sophia-G / Rocora Control Identity: setting `residual_weight = 0.0` reproduces Sophia-G bitwise across multiple steps.
   - Hessian Decomposition Identity: $R(\theta) = H(\theta) - S(\theta)$ at fixed data/parameter snapshots.
3. **Progressive Testability & Isolation**: Tests run in isolation, creating independent PRNG keys and temporary directories. They gracefully handle milestone transitions using capability detection while enforcing strict contract validation.
4. **Hardware Agnostic Execution**: CPU-executable test suite using `PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q` ensures complete CI/CD verification without requiring physical TPU slice allocation.

---

## 2. 4-Tier Test Architecture

The E2E test suite is organized into 4 distinct testing tiers located in `tests/e2e/`:

```
tests/e2e/
├── __init__.py
├── run_e2e_tests.py                 # Multi-tier CLI test runner
├── test_tier1_features.py           # Tier 1: Exhaustive Feature Coverage (>=5 tests / feature)
├── test_tier2_boundaries.py         # Tier 2: Boundary & Corner Cases (>=5 tests / feature)
├── test_tier3_combinations.py       # Tier 3: Pairwise Cross-Feature Interactions
├── test_tier4_workloads.py          # Tier 4: Multi-Step Real-World Application Workloads
├── test_llama_features.py           # Dedicated LLaMA Architecture & Autodiff Suite (R1)
└── test_sophia_h.py                 # Dedicated Sophia-H & XLA Optimization Suite (R2, R3)
```

### Tier 1: Exhaustive Feature Coverage
Validates the primary functional behavior (happy path) for all features defined in `PROJECT.md` Feature Inventory (Features 1 through 17). Every feature is covered by at least 5 targeted test cases.

- **F1: RMSNorm**: Forward scaling, zero-mean non-centering, unit variance preservation, float32 accumulation, scale parameter gradient flow.
- **F2: RoPE**: Rotary frequency computation ($\theta_i = 10000^{-2i/D}$), half-split rotation, vector norm conservation, sequence position invariance, unrotated value projection.
- **F3: SwiGLU 8/3 MLP**: Gated SiLU activation, dimension calculation $\lfloor \frac{8}{3} d \rfloor$ aligned to 64/128, parameter count parity ($8d^2$), down-projection scaling $\frac{0.02}{\sqrt{2l}}$, non-linear representation.
- **F4: Tied Embeddings & Bias-Free**: Weight sharing between token embedding and unembedding transpose, strict zero additive bias parameters across linear and normalization layers.
- **F5: Attention & Curvature Autodiff**: Dual-path routing (TPU FlashAttention for forward training, FP32 reference attention for `curvature=True`), JVP / HVP / linearize higher-order differentiation without NaNs.
- **F6: Parameter Count Formulas**: Analytical formula across all scale tiers (125M: 123,587,328; 350M: 355,124,224; 1B: 983,336,448), exact elimination of position table parameters.
- **F7: XLA Optimizer Routing**: Strict enforcement of XLA fusion in `updates.py` and `matrix_optimizers.py`, `should_use_pallas` returning `False` in auto mode on all platforms.
- **F8: Reserve Pallas for FlashAttention**: Pallas kernel reserved exclusively for TPU FlashAttention in `attention.py`.
- **F9: Rademacher Probe Generator**: Rademacher distribution $u \in \{-1, +1\}^d$, expectation $\mathbb{E}[u]=0$, variance $\text{Var}(u)=1$, independent seeds across probe groups.
- **F10: Hutchinson Hessian Diagonal**: Unbiased quadratic form $\mathbb{E}[u \odot \text{Hv}(u)] = \text{diag}(H)$, true loss mini-batch evaluation, replica token-weighted averaging, non-negative clipping $\hat{h} \ge 0$.
- **F11: Sophia-H Training Step & Registry**: Registry in `OPTIMIZERS`, EMA curvature accumulation, coordinate-wise clipping update $\text{clip}(\frac{m}{\rho h + \epsilon}, -1, 1)$, integration into training step routines.
- **F12: Purge Obsolete Outputs**: Elimination of legacy GPT-2 sweep checkpoints and outputs in `outputs/scale125m` and `outputs/tuning`.
- **F13: Documentation Synchronization**: Parity across `README.md`, `theory.md`, `note.md`, and `docs/`.
- **F14: Config & Sweep Synchronization**: Consistency across `configs/` and `configs/sweeps/`.
- **F15: Scientific Parity & Controls**: Bitwise identity `residual_weight = 0` $\equiv$ Sophia-G, dense Hessian decomposition $R = H - S$.
- **F16: Full Test Suite Green**: Zero failures across unit, integration, and E2E suites.
- **F17: Git Cleanliness & Origin Push**: Clean git working tree, `git diff --check`, verified push readiness.

### Tier 2: Boundary & Corner Cases
Stresses each feature against pathological conditions, boundary limits, and extreme inputs (>=5 tests per feature):
- **Zero & Near-Zero Inputs**: Inputs of all zeros, zero gradients, zero loss masks, zero learning rates.
- **Extreme Numerical Scales**: Very large gradients ($\|g\| > 10^4$), small epsilon values ($10^{-15}$), extreme sequence lengths, single-token sequences ($T=1$).
- **Dimensional Boundaries**: Minimal model widths ($d < 64$), single layer ($l=1$), single attention head ($h=1$), extreme head dimensions.
- **Curvature Corner Cases**: Negative curvature components in Hutchinson diagonal (verified clamped to 0), rank-deficient probe vectors, zero active tokens in replica averaging.
- **Resource Constraints**: High-memory chunking in cross-entropy, lock contention in multi-host worker scripts, missing files in checkpoint loaders.

### Tier 3: Pairwise Cross-Feature Interactions
Validates the structural and algorithmic interoperability between pairs of decoupled subsystems:
- **RoPE $\times$ FlashAttention**: Ensuring rotated query and key tensors pass into FlashAttention without layout transposition defects.
- **RMSNorm $\times$ Curvature Autodiff**: Ensuring root-mean-square normalization does not cause gradient singularities or second-derivative blowups during `jax.linearize`.
- **SwiGLU $\times$ Muon**: Applying Newton-Schulz semi-orthogonal updates to 3-matrix SwiGLU layer stacks (`mlp_gate`, `mlp_up`, `mlp_down`).
- **Sophia-H $\times$ Checkpoint Resume**: Saving and restoring state containing Hutchinson curvature EMAs with bitwise state resumption.
- **Rocora $\times$ Sophia-G Multi-Step Parity**: Multi-step training trajectory equivalence when residual weight is set to zero.
- **XLA Fusion $\times$ All 6 Optimizers**: Running AdamW, Sophia-G, Sophia-H, Rocora, Muon, and SOAP through unified XLA dispatch without Pallas fallback.
- **Model Config $\times$ Sweep Materialization**: Validating generated sweep configurations against updated LLaMA parameter counts.

### Tier 4: Real-World Application Workloads
Full end-to-end multi-step training simulations exercising realistic workflows under production configurations:
- **Scenario 1**: Bounded AdamW LLaMA-125M smoke training with checkpoint save/load and evaluation.
- **Scenario 2**: Bounded Sophia-G LLaMA-125M smoke training with sampled-label core curvature updates and diagnostics.
- **Scenario 3**: Bounded Sophia-H LLaMA-125M smoke training with Hutchinson true-loss Hessian diagonal estimation and non-negative clipping.
- **Scenario 4**: Bounded Rocora LLaMA-125M smoke training with joint Gauss-Newton core and coded residual curvature measurement.
- **Scenario 5**: Bounded Muon LLaMA-350M smoke training with Newton-Schulz matrix updates on SwiGLU weights and auxiliary AdamW routing.
- **Scenario 6**: Bounded SOAP LLaMA-350M smoke training with eigenspace preconditioning and QR basis refresh.
- **Scenario 7**: Multi-stage pod staging dry-run verifying multi-host TPU worker lock protocols and config validation across 125M, 350M, and 1B scales.

---

## 3. Test Execution & CLI Runner

### Running the Complete E2E Suite via Pytest
```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q tests/e2e/
```

### Running Specific Tiers via `run_e2e_tests.py`
```bash
# Run all tiers with detailed summary reporting
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py

# Run Tier 1 (Feature Coverage)
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 1

# Run Tier 2 (Boundaries & Edge Cases)
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 2

# Run Tier 3 (Cross-Feature Combinations)
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 3

# Run Tier 4 (Realistic Application Workloads)
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/e2e/run_e2e_tests.py --tier 4
```

### Filtering by Expression
```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q -k "rope or rms_norm or sophia_h"
```

---

## 4. Hardware & Environment Invariants

1. **JAX CPU Platform Fallback**: `JAX_PLATFORMS=cpu` allows standard local development and automated CI execution without hardware TPU attachments.
2. **FlashAttention CPU Safety**: In `src/rocora/model.py`, forward passes routing to `flash_attention` on CPU are caught gracefully or routed via `reference_attention` in CPU tests.
3. **Curvature FP32 Invariant**: Curvature passes (`curvature=True`) strictly execute through `reference_attention` in FP32, ensuring higher-order automatic differentiation without TPU Pallas backward limitations.
4. **Worker Lock Invariant**: File locks in `scripts/worker.py` and `scripts/pod.py` use POSIX `fcntl.flock(..., LOCK_EX | LOCK_NB)` to ensure exclusive host execution and safe cleanup on process termination.

---

## 5. Scientific Parity & Verification Standards

| Invariant / Property | Mathematical / Empirical Definition | Verification Method |
|---|---|---|
| **Rocora Zero Control** | $\lim_{w_r \to 0} \theta_{t+1}^{\text{Rocora}} = \theta_{t+1}^{\text{Sophia-G}}$ | Bitwise parameter & core state equality after $N$ steps |
| **Hessian Split** | $H(\theta) = S(\theta) + R(\theta)$ | Finite-difference or exact autograd parity against dense Hessian |
| **Sophia-H Unbiasedness** | $\mathbb{E}_{u}[\hat{h}] \approx \text{diag}(\nabla^2 L(\theta))$ | Monte Carlo expectation matching diagonal across random batches |
| **RoPE Relative Invariance** | $\langle R_m q, R_n k \rangle = \langle R_{m+k} q, R_{n+k} k \rangle$ | Numerical inner product equivalence under identical translation |
| **RMSNorm Homogeneity** | $\text{RMSNorm}(\alpha x) = \text{sign}(\alpha) \text{RMSNorm}(x)$ | Positive scaling invariance for all $\alpha > 0$ |
| **Parameter Parity** | $N_{\text{LLaMA}} = \text{vocab} \cdot d + l(4d^2 + 3d \cdot d_{\text{ffn}} + 2d) + d$ | Tree parameter count matching analytical closed form |
