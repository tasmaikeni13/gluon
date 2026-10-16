# Optimizer recipes and budget sweeps

These are implementations and launch recipes. No sweep or main experiment was
run to choose the checked-in learning rates. Bounded checks establish numerical
and runtime behavior, not training quality.

## Reference algorithms

The JAX ports were checked against the authors' sources on 2026-10-15:

- [SOAP paper](https://arxiv.org/abs/2409.11321) and
  [official SOAP code](https://github.com/nikhilvyas/SOAP/blob/f42d296cb4146a67fbe811371e6badb9a39cc54d/soap.py).
  SOAP applies Adam in the eigenbasis of exponentially averaged per-axis gradient
  Gram matrices. The first batch initializes the basis and performs no update or
  decay. Each subsequent batch updates rotated moments, takes an update, updates
  the Gram matrices, and periodically sorts estimated eigenvalues and performs a
  QR power iteration. The first moment is transported through parameter space;
  the second moment is permuted as in the reference. Bias correction counts actual
  updates after initialization. Epsilon is added before the correction factor.
  The reference's post-update decoupled decay is preserved.
- [Official Muon code](https://github.com/KellerJordan/Muon/blob/64560829f70a216f8263ea8ae0fefeac8b451d36/muon.py)
  and [author's explanation](https://kellerjordan.github.io/posts/muon/).
  Muon uses an EMA momentum, optional Nesterov combination, BF16 quintic
  Newton–Schulz iterations with coefficients (3.4445, -4.7750, 2.0315), Frobenius
  normalization with 1e-7, and the original sqrt(max(1, output/input)) adjustment.
  Matrix orientation accounts for JAX's [input, output] storage. Five iterations
  approximate orthogonalization; they are not an exact polar decomposition.
  Tied embedding/output weights, positions and norm gains use bias-corrected
  AdamW with a separate auxiliary learning rate and the same schedule multiplier.

CPU tests compare SOAP against independent NumPy calculations and author-generated
vectors. Nine Muon BF16 momentum/direction outputs match vectors generated from
the pinned upstream implementation. These checks need no PyTorch runtime
dependency in the Rocora package.

Every transformer layer is an independent matrix for both methods. Stacked
storage never mixes layers or treats norm gains as matrices. SOAP uses
merge_dims=false, precondition_1d=false, normalize_grads=false, correct_bias=true,
and max_precond_dim=2048. Larger axes are skipped, as in the reference; allowed
axes are still preconditioned. This is especially relevant to the vocabulary
axis, QKV output axis and MLP expansion at the largest scale. Layer operations
are scanned to limit temporary memory. State/covariances/bases are FP32; normal
forward/backward matrix products and Muon polynomial products use BF16 on TPU.
SOAP is an additional practical baseline, as is Muon. The scientific residual
control remains Sophia-G versus Rocora.

Microbatches accumulate **loss sums and gradient sums** before division by the
active-token count and the device all-reduce. Optimizers update once per global
batch. This preserves masked final batches and a single global clipping norm.
Sampled-label core events still average token-weighted gradients across devices
before squaring. SOAP/Muon never run Sophia curvature events.

## Recipes

| Config | Parameters | Main tokens per arm/seed | Main seeds | Trial tokens | Trials per arm | Tuning tokens per arm |
|---|---:|---:|---|---:|---:|---:|
| configs/scale125m.json | 124,373,760 | 2,500,000,000 | 1337, 1338, 1339 | 25,000,000 | 6 | 150,000,000 |
| configs/scale350m.json | 354,599,936 | 7,000,000,000 | 1337 | 70,000,000 | 6 | 420,000,000 |
| configs/scale1b.json | 984,909,312 | 20,000,000,000 | 1337 | 200,000,000 | 6 | 1,200,000,000 |

All three stages support AdamW, Sophia-G, Rocora, SOAP and Muon. The 350M and 1B
labels are nominal; the exact counts above include tied/padded embeddings and
positions. All use context 1024, global batch 128, one sequence per device per
microbatch, Pallas FlashAttention, and exact final loss masks. One 1B seed is a
screening experiment, not a repeatability study.

Sweeps use seed 1337, a short standalone cosine schedule with 5% warmup, three
learning-rate multipliers (0.5, 1, 2), and two values of one additional setting:
AdamW beta2 (0.95, 0.99), Sophia/Rocora rho (0.02, 0.03), SOAP basis frequency
(10, 20), Muon momentum (0.9, 0.95). Muon's auxiliary rate receives the same
multiplier. Each trial starts from the same initialization and data order.
Every candidate receives the full trial token budget; there is no early pruning.
An evaluation occurs roughly every quarter of the trial and at its exact end.

The tuning budgets are equal **tokens**, not equal wall time or FLOPs; extra
optimizer work is measured. Totals across all five arms are 750M, 2.1B, and 6B
tokens respectively. Short-budget winners may change at the main budget. Rates
are provisional until a sweep is actually completed. Validation is used for
selection and main reporting; these metrics are not an independent test set.

## Prepare the larger data once

The 350M/1B recipes require new archives; they reject the existing 2.5B archive
as too short. Do not run these preparers during implementation validation.
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) provides
the sample-10BT and sample-100BT subsets. The preparer pins the resolved dataset
revision, uses GPT-2 BPE, partitions by document-content hash, and records counts
and hashes. No heldout document enters training. On each worker, install/copy
the same archives at the configured paths:

```bash
PYTHONPATH=src python -m rocora.prepare --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --train-limit 7000000000 --output /home/tasma/data/fw7b
PYTHONPATH=src python -m rocora.prepare --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-100BT --train-limit 20000000000 --output /home/tasma/data/fw20b
PYTHONPATH=src python -m rocora.preflight --config configs/scale350m.json
PYTHONPATH=src python -m rocora.preflight --config configs/scale1b.json
```

## Plan, tune, select, then launch

Planning writes reviewable trial configs and a manifest without contacting TPU
workers. Generated configs and selected recipes are ignored by Git. Run from
the repository root. These commands are recipes; implementation work does not
execute unbounded sweeps or main runs.

```bash
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale125m.json --plan-only
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale350m.json --plan-only
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale1b.json --plan-only
```

For any scale, replace scale350m below. A sweep syncs the generated configs under
the normal locks, then runs each candidate sequentially on the entire pod.
--dry-run validates all trials without updating parameters; --max-steps bounds
each trial and writes under outputs/validation/tuning/. Those outputs cannot be
selected as winners. With --resume, existing checkpoints (including completed
trials) are restored, and unstarted trials start fresh. An existing trial
directory without a checkpoint fails explicitly and is preserved. Never
overwrite a trial or change its source before resume.

```bash
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale350m.json --dry-run
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale350m.json
PYTHONPATH=src /home/tasma/venv/bin/python -m rocora.sweep select --manifest configs/generated/scale350m-budget-v1/manifest.json --root outputs --output configs/selected/scale350m.json
python scripts/pod.py sync
python scripts/pod.py stage --config configs/selected/scale350m.json --dry-run
python scripts/pod.py stage --config configs/selected/scale350m.json
```

Selection rejects missing/incomplete/bounded trials, configuration mismatches,
and mixed code, hardware or data. The default matched selection chooses Sophia's
best validation setting and applies that same setting to Rocora; AdamW, SOAP
and Muon choose their individual best trials. This preserves the residual
control. --mode independent selects each arm separately for a practical tuned
comparison; use rocora.report --comparison tuned, and do not attribute its
Sophia/Rocora difference solely to the residual. Neither selection changes the
main data, schedule, token budgets or seeds. Main runs restart from initialization
and never resume a tuning checkpoint.

Reports require every configured arm for every observed seed:

```bash
PYTHONPATH=src python -m rocora.report --root outputs/scale350m
PYTHONPATH=src python -m rocora.report --root outputs/scale1b
```

## Bounded validation

Use a distinct run name and --max-steps. SOAP needs at least two batches to test
an actual parameter update, because its first batch only initializes the basis.
Exercise a later basis refresh and checkpoint resume too. Large shape checks
may use --skip-checkpoint together with --max-steps to avoid storing multi-GB
validation checkpoints; this is forbidden for unbounded/resumed training.
Keep those checks separate from the checkpoint-resume checks. The validation
record states any corpus/evaluation substitutions and the exact scope tested.
