# Rocora

**Residual Only Coded Optimization with a Regularized Anchor.**

This is a JAX/XLA research codebase for a Cloud TPU v4-32 pod. It asks:

> Does adding residual-only coded curvature measurements to a positive
> Gauss–Newton/Fisher core improve language-model updates enough to pay for the
> extra computation?

The experiment compares **AdamW, Sophia-G, Rocora, SOAP, and Muon** at the larger
scales. Sophia-G is the core-only
control: it has the same sampled-label core, momentum, clipping, decay, and
hyperparameters as Rocora. Rocora adds an EMA of the signed residual diagonal.
Setting Rocora's residual weight to zero reproduces Sophia-G exactly.

The Hessian decomposition is established mathematics. A useful contribution
would be evidence about this residual estimator and its training efficiency;
novelty or improved convergence is not assumed. See [theory.md](theory.md) for the
estimator, its limits, and the experimental decision criteria.

## Experiments

| Stage | Actual parameters | Loss tokens processed | Seeds | Optimizers |
|---|---:|---:|---|---|
| 125M | 124,373,760 | 2,500,000,000 each | 1337, 1338, 1339 | All five |
| 350M | 354,599,936 | 7,000,000,000 each | 1337 | All five |
| 1B | 984,909,312 | 20,000,000,000 each | 1337 | All five |

All GPT models use a 1024-token context, GPT-2 BPE, tied embeddings, smooth GELU,
and no dropout. The global batch is 128 sequences. The final batch is masked so
the loss-token budget is exact. Initial learning rates are configurable baseline
settings, rather than claimed optimal values. Sophia-G and Rocora deliberately
share their settings to isolate the residual's effect.

The 125M stage uses 2.5B training tokens from
[FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
The existing 2.5B archive divides one document across the validation/training
boundary. The 125M configuration excludes its remaining 241-token training
prefix to prevent that overlap. This exclusion is recorded; the exact 2.5B
processed-token budget can repeat a few remaining tokens. Fresh content-hash
preparation avoids this boundary split and does not need the exclusion.

The 350M and 1B recipes require newly prepared 7B/20B-token archives and reject
shorter corpora. Budget sweeps provide six candidates per optimizer at 125M,
350M and 1B, with 25M/70M/200M tokens per trial respectively. Rates in the main
configs are provisional. [docs/optimizers.md](docs/optimizers.md) describes the
reference SOAP/Muon algorithms, exact parameter counts, tuning budgets, data
preparation, validation limitations, and recipe selection.

Plan a sweep without starting any training:

```bash
PYTHONPATH=src /home/tasma/venv/bin/python scripts/pod.py sweep --spec configs/sweeps/scale350m.json --plan-only
```

## TPU setup and launch

The current workers are w0–w3. A v4-32 slice exposes **16 JAX devices across four
hosts**, with four local devices per host. The launcher starts every worker for
one coordinated job; it runs optimizer arms sequentially across the whole pod.
Edit [configs/pod.json](configs/pod.json) for another pod or Python environment.
Worker locks prevent overlapping launches and source sync during a job. If a
worker fails or SSH disconnects, the launcher cleans up that job's remote process
groups before releasing its locks.

The tested runtime is Python 3.10, JAX/jaxlib 0.6.2 and libtpu 0.0.17. An existing
environment with these versions can use PYTHONPATH=src. To install a fresh one:

```bash
python -m pip install -e '.[tpu,data,test]'
```

From the checkout, sync the source and verify data/device/configuration on all
workers. This preserves remote token archives and outputs:

```bash
python scripts/pod.py sync
python scripts/pod.py stage --config configs/scale125m.json --dry-run
```

The following commands launch full training and are provided for later use.
After preparing the needed data, launch a stage
(prefer selected sweep recipes as described in docs/optimizers.md):

```bash
python scripts/pod.py stage --config configs/scale125m.json
python scripts/pod.py stage --config configs/scale350m.json
python scripts/pod.py stage --config configs/scale1b.json
```

To resume a stage, append --resume. For one arm or a bounded hardware smoke run:

```bash
python scripts/pod.py run --config configs/scale125m.json --optimizer rocora --seed 1337
python scripts/pod.py run --config configs/scale125m.json --optimizer rocora --max-steps 3 --run-name smoke125m-rocora
```

A bounded smoke run retains the experiment's schedule, records its actual token
count, and is marked incomplete. It is never accepted as a full benchmark result.
Checkpoints include parameters, every optimizer channel/counter, the seed and
experiment identity. Stateless batch/probe selection makes a resumed run
reproducible. Outputs are saved on worker 0 under outputs/; --run-name prevents
accidentally reusing another run's directory.

## TPU implementation

- TPU Pallas FlashAttention with the JAX team's custom backward is used for
  normal training and sampled-label core measurements.
- FP32 master weights/state, BF16 training matrix multiplications, stacked-layer
  scans, activation rematerialization, chunked output projection/loss, and host
  batch prefetch reduce memory traffic and launch overhead.
- XLA compiles and fuses the complete training step, including gradient scaling,
  momentum, preconditioning, clipping and decoupled decay for all optimizers.
  Parameter/state buffers are donated. Custom optimizer kernels were removed
  after measurements showed no meaningful advantage over XLA fusion.
- SOAP keeps per-axis Gram matrices and bases and transports rotated Adam
  moments at basis refreshes. Muon applies BF16 Newton–Schulz to hidden matrices
  and AdamW to tied embeddings, positions and norm gains. Each stacked layer
  is processed independently. One-sequence microbatches bound activation memory
  while accumulating one token-weighted gradient for each global update.
- Residual measurements use FP32, tiled differentiable attention. The installed
  FlashAttention backward explicitly does not support higher-order autodiff.
  Each complete coded group measures one fixed set of weights and data.
- The residual operator is computed directly by freezing the logit loss gradient
  and differentiating its inner product with model logits. This avoids subtracting
  two large products Hv and Sv. One derivative linearization is reused for the
  full coded group.

The FlashAttention KV tile was tuned on v4, and compiled
executables are cached on each worker for later launches.

The default group contains four Hadamard codes, measured every 40 updates; the
core is refreshed every 20. These intervals and batch sizes are experiment
settings. TPU timings include the measured curvature cost.

## Data and evaluation

The checked-in configurations use the existing /home/tasma/data token archives.
Their source/tokenizer/counts are checked at startup. Check every token and the
file hashes with:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m rocora.preflight --config configs/scale125m.json
```

For a new archive, use the streaming preparer, which resolves a dataset revision, inserts document
separators, separates documents by content hash, and records token/file hashes:

```bash
PYTHONPATH=src python -m rocora.prepare --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --train-limit 2500000000 --output data/scale
```

Point each configuration's train_file, validation_file and data_metadata at the
prepared files and copy them to every worker. Keep validation documents out of
training. For a freshly prepared 125M archive, set
train_skip_initial_document to false; that option repairs the existing archive's
boundary split. Every arm sees the same sequence permutation for a given seed.

Metrics include heldout loss/perplexity, exact processed tokens, total training
time, curvature-event time, residual size, clipping, and the fraction of updates
changed by the residual. Compilation is recorded separately and excluded
from steady-state training time; evaluation and checkpoints are also separate.
After complete matched runs, compare equal-token losses and interpolated
equal-time losses:

```bash
PYTHONPATH=src python -m rocora.report --root outputs/scale125m
```

The report rejects bounded checks, incomplete budgets, missing arms, changed
code/data, and Sophia/Rocora settings that would confound the residual comparison.
Independently tuned comparisons require --comparison tuned and do not isolate
the residual contribution. Multi-seed studies measure repeatability; the
one-seed 350M/1B recipes screen larger scales.

## Validation and references

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q
```

Tests cover the actual fixed-snapshot residual against a dense Hessian split,
compiled update formulas, Sophia/core-only equivalence, token partitioning, exact
budgets and checkpoint identity. See [docs/validation.md](docs/validation.md) for
hardware measurements and the tested launch scope.
See [AGENTS.md](AGENTS.md) for repository changes, scientific controls, and pod
validation rules.

[Sophia's official implementation](https://github.com/Liuhong99/Sophia) defines
the Sophia-G baseline. Its stored squared sampled gradients are normalized here
by their token count before entering the core EMA, algebraically matching the
batch factor in the original denominator. On the pod, sampled gradients are
all-reduced before squaring and scaled by the global active-token count.
Momentum/core EMAs are not bias
corrected, as in that implementation.

[JAX TPU Pallas documentation](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
describes the kernel backend. The retained [Lean development](lean/Rocora.lean)
records the algebraic split/probing results; it does not prove better neural
network training or validate the numerical TPU implementation.
