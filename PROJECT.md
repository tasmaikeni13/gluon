# Project: Rocora Modernization & Refactoring

## Architecture
- **Model Layer (`src/rocora/model.py`)**:
  - RMSNorm: $\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$, bias-free, scale parameter initialized to ones.
  - RoPE (Rotary Position Embeddings): Dynamic rotation applied per-head to query and key projections in attention. Complete removal of absolute learned position embedding table (`position`).
  - SwiGLU MLP: Gated feed-forward network computing $(\text{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$. Hidden dimension $d_{\text{ffn}} = \lfloor \frac{8}{3} d \rfloor$ rounded to nearest multiple of 64 (or $\max(1, \lfloor \frac{8}{3}d \rfloor)$ for $d < 64$).
  - Tied Embeddings: Unembedding head strictly tied to transpose of input embedding table (`params["embedding"].T`), with zero additive biases.
  - Attention Routing: TPU FlashAttention (`attention.py`) during standard forward training passes; differentiable FP32 causal Reference Attention during curvature measurement passes (`curvature=True`).
- **Optimization Layer (`src/rocora/`)**:
  - Pure XLA Fusion: All optimizer updates (`adamw`, `sophia`, `sophia_h`, `rocora`, `muon`, `soap`) execute via pure XLA-fused paths in `src/rocora/updates.py` and `src/rocora/matrix_optimizers.py`. Custom Pallas optimizer kernels are completely removed and deleted; Pallas is reserved exclusively for TPU FlashAttention.
  - Sophia-H Optimizer: Hutchinson's unbiased Hessian diagonal estimator $\hat{h} = \max(u \odot \text{Hv}(u), 0)$ using Rademacher probe vectors $u \in \{-1, +1\}^d$ on the true loss mini-batch with FP32 reference attention. Token-weighted collective averaging across replicas (`average_tree`) before coordinate-wise clipping $\text{clip}\left(\frac{m}{\rho \cdot h + \epsilon}, -1, 1\right)$. Registered in `OPTIMIZERS` across `config.py`, `train.py`, `checkpoint.py`, `sweep.py`, and configs.
- **Repository Cleanliness & Parity**:
  - Purge obsolete GPT-2 sweep directories (`outputs/scale125m`, `outputs/tuning`).
  - Synchronize documentation (`README.md`, `theory.md`, `note.md`, `docs/`) and configs with modern LLaMA architecture and 6 optimizer arms.
  - Full CPU test verification (`PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q`), `git diff --check`, and clean commit/push to `origin/main`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | RMSNorm | Root-mean-square normalization without mean-centering, bias-free | M1 | Survey 1 (R1) |
| 2 | RoPE | Rotary position embeddings applied to Q and K per head; remove learned position table | M1 | Survey 1 (R1) |
| 3 | SwiGLU 8/3 MLP | Gated SiLU activation $(\text{silu}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$ with $d_{\text{ffn}} = \lfloor \frac{8}{3}d \rfloor$ (aligned to 64) | M1 | Survey 1 (R1) |
| 4 | Tied Embeddings & Bias-Free | Head weights tied to embedding table; zero additive biases in linear/norm layers | M1 | Survey 1 (R1) |
| 5 | Attention & Curvature Autodiff | Reference attention & FlashAttention compatibility; smooth double-diff (JVP/HVP) | M1 | Survey 1 (R1) |
| 6 | Parameter Count Formula | Accurate formula for LLaMA 32k vocab: 125M (109,529,856), 350M (336,380,928), 1B (955,221,504) | M1 | Survey 1, 3 (R1, R4) |
| 7 | Pure XLA Optimizer Updates | Completely remove custom Pallas optimizer kernels; all optimizers strictly use XLA fusion | M2 | Survey 2 (R2) |
| 8 | Reserve Pallas for FlashAttention | Pallas strictly reserved for TPU FlashAttention kernel in `attention.py` | M2 | Survey 2 (R2) |
| 9 | Rademacher Probe Generator | Generate $u \in \{-1, +1\}^d$ probe vectors using `jax.random.bits` | M3 | Survey 2 (R3) |
| 10 | Hutchinson Hessian Diagonal | Compute $\text{Hv}(u)$ on true loss with FP32 reference attention, average across replicas, $u \odot \text{Hv}$, clip $\ge 0$ | M3 | Survey 2 (R3) |
| 11 | Sophia-H Training Step & Registry | Register `sophia_h` in `OPTIMIZERS`, `train.py` `do_core`, `checkpoint.py`, `sweep.py`, `report.py` | M3 | Survey 2 (R3) |
| 12 | Purge Obsolete Outputs | Safely delete historical GPT-2 runs in `outputs/scale125m` and `outputs/tuning` | M4 | Survey 3 (R4) |
| 13 | Documentation Synchronization | Update `README.md`, `theory.md`, `note.md`, `docs/` to reflect LLaMA & Sophia-H | M4 | Survey 3 (R4) |
| 14 | Config & Sweep Synchronization | Add `sophia_h` to `configs/*.json` and `configs/sweeps/*.json`, update parameter counts | M4 | Survey 3 (R4) |
| 15 | Scientific Parity & Controls | Verify `residual_weight=0` reproduces Sophia-G; verify $R=H-S$ dense Hessian split | M5 | Survey 1, 2 (R5) |
| 16 | Full Test Suite Green | 100% test pass on CPU (`PYTHONPATH=src JAX_PLATFORMS=cpu pytest -q`) | M5 | Survey 1, 2, 3 (R5) |
| 17 | Git Cleanliness & Origin Push | Pass `git diff --check`, stage/commit authorized work, push cleanly to `origin/main` | M5 | Survey 3 (R5) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Modern LLaMA Architecture & Attention | RMSNorm, RoPE, SwiGLU 8/3, tied embeddings, bias-free, parameter count formula in `src/rocora/model.py`, `src/rocora/config.py`, model tests | none | COMPLETE (`20655b28-0779-470c-9583-49e241e7c4cc`) |
| M2 | XLA Optimizer Updates & Kernel Routing | Enforce pure XLA fusion across all optimizer updates, delete Pallas optimizer kernels, reserve Pallas for FlashAttention | none | COMPLETE (`de02df20-f943-4c96-8573-b1cac963aa13`) |
| M3 | Sophia-H Optimizer Implementation | Hutchinson Hessian diagonal estimator, Rademacher probes, registry in `config.py`, `train.py`, `checkpoint.py`, `sweep.py`, configs, tests | M1, M2 | COMPLETE (`df53a818-4f45-408b-b3f4-be449e186ef4`) |
| M4 | Purge Obsolete Results & Sync Docs/Configs | Remove `outputs/scale125m`, `outputs/tuning`; update `README.md`, `theory.md`, `note.md`, `docs/`, configs, sweep manifests | M1, M3 | COMPLETE (`92d2d067-1cf4-4c55-98e2-0353d3f75dd2`) |
| M5 | Comprehensive E2E Verification & GitHub Push | Full E2E test suite pass (Tiers 1-4), scientific control checks, `git diff --check`, commit and push to `origin/main` | M1, M2, M3, M4 | IN_PROGRESS |

## Interface Contracts
### `model.py` ↔ `train.py` & `curvature.py`
- `init_params(cfg: ModelConfig, key)`: Returns dict containing:
  - `"embedding"`: shape `[vocab_size, width]`
  - `"layers"`: dict with:
    - `"norm_1"`: shape `[layers, width]`
    - `"qkv"`: shape `[layers, width, 3, width]`
    - `"proj"`: shape `[layers, width, width]`
    - `"norm_2"`: shape `[layers, width]`
    - `"mlp_gate"`: shape `[layers, width, d_ffn]`
    - `"mlp_up"`: shape `[layers, width, d_ffn]`
    - `"mlp_down"`: shape `[layers, d_ffn, width]`
  - `"norm_final"`: shape `[width]`
  (No `"position"`, no additive biases).
- `logits(params, tokens, cfg, curvature=False)`:
  - Takes `tokens: [batch, length]` integer array.
  - Returns `logits: [batch, length, vocab_size]` in FP32.
  - When `curvature=True`, calls `reference_attention(q, k, v, block)` to allow smooth double-differentiation.
- `apply_rope(q, k)`:
  - Takes $q, k \in \mathbb{R}^{B \times H \times T \times D}$.
  - Computes rotary position encoding using $\theta_i = 10000^{-2i/D}$.
  - Returns rotated $q_{\text{rot}}, k_{\text{rot}}$ with identical shapes.

### `curvature.py` ↔ `train.py` (Sophia-H Estimator)
- `rademacher_probe(shape, key)`:
  - Returns array of random values in $\{-1, +1\}$ with given shape using `jax.random.bits`.
- `hutchinson_diagonal(params, batch, model_cfg, key)`:
  - Generates Rademacher probe $u$ for each parameter tensor.
  - Computes $\text{Hv}(u)$ using `jax.jvp(jax.grad(loss_fn), (params,), (u,))` on the true loss mini-batch with `curvature=True`.
  - Averages local Hv products across replicas with token weights (`average_tree(hv, count)`).
  - Returns $\hat{h} = \max(u \odot \text{Hv}(u), 0)$.

### `pallas_optimizers.py` ↔ `optimizers.py`
- `should_use_pallas(optimizer_name, mode="auto")`:
  - Returns `False` unconditionally when `mode == "auto"`.
  - All standard training routes to XLA-fused paths in `updates.py` and `matrix_optimizers.py`.

## Code Layout
- `src/rocora/model.py`: Owned exclusively by Milestone M1 Workers.
- `src/rocora/config.py`: Owned by M1 Worker (model config) and M3 Worker (optimizer config).
- `src/rocora/pallas_optimizers.py`: Owned exclusively by Milestone M2 Worker.
- `src/rocora/curvature.py`: Owned exclusively by Milestone M3 Worker.
- `src/rocora/train.py`, `src/rocora/checkpoint.py`, `src/rocora/sweep.py`: Owned by Milestone M3 Worker.
- `outputs/`, `configs/`, `*.md`, `docs/`: Owned by Milestone M4 Worker.
- `tests/`: Owned by E2E Testing Track and Milestone Workers for unit tests.
