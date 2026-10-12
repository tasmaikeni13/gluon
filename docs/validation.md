# Implementation validation

These checks validate the implementation and launch path. They do not establish
that Rocora improves convergence or pays for its residual measurements. The
1B/2.5B-token experiments remain to be run.

## Environment and numerical checks

The hardware is a Cloud TPU v4-32 slice: four hosts, four local JAX devices per
host, 16 devices in each distributed model run. The tested runtime is Python
3.10, JAX/jaxlib 0.6.2 and libtpu 0.0.17.

The CPU suite passes 13 tests. It covers:

- Direct residual products against a dense H−S calculation at one fixed snapshot.
- Exact Sophia-G equivalence when Rocora's residual weight is zero, including
  compiled XLA updates, the published AdamW/Sophia update formulas, and Rocora's
  signed-curvature/clipping rule with gradient scaling.
- Sampled-label core normalization, disjoint host batches, exact masked final
  budgets, and exclusion of the existing corpus's shared boundary document.
- Checkpoint state/seed/code identity, report rejection of incomplete or
  confounded comparisons, and the 2.5B-token schedule beyond the int32 limit.
- Exclusive worker locks, clean successful exit, cleanup on launcher disconnect,
  and lock release after the child has been reaped.

The editable package builds and all four console entry points display help.
The retained Lean development was renamed and scoped in its documentation;
it was not rebuilt as part of this TPU validation.

## TPU kernel measurements

[tpu-kernels.json](tpu-kernels.json) preserves the prototype comparison that led
to removing the custom optimizer kernels. Measurements use 8,388,608 FP32 elements, 65,536-element tiles,
donated parameter/moment buffers, three warmup calls and 25 timed calls.
Compilation is excluded; calls are blocked until completion.

| Update | XLA median (ms) | Pallas median (ms) | XLA/Pallas |
|---|---:|---:|---:|
| AdamW | 0.4595 | 0.4505 | 1.020 |
| Sophia-G | 0.4010 | 0.4026 | 0.996 |
| Rocora | 0.4394 | 0.4356 | 1.009 |

The maximum observed update error was zero for these inputs. The custom updates
were approximately as fast as XLA fusion on this microbenchmark and did not show
a meaningful advantage. Their implementations, configuration toggles and tile
settings have been deleted. All three experiment arms now use XLA-fused updates.

FlashAttention parity uses BF16 tensors with batch 8, heads 8, sequence 1024 and
head width 64. Maximum observed forward error is 0.001953125 and backward error
is 4.47e−8 against tiled JAX attention. The tuned FlashAttention forward/backward
medians are 0.5295/2.1180 ms, versus 0.6261/2.1345 ms for tiled JAX.
[tpu-tuning.json](tpu-tuning.json) separately records the tile sweep: moving the
KV tile from 128 to 1024 tokens reduced FlashAttention forward/backward medians
from 1.3130/4.2024 ms to 0.5623/2.1905 ms in that measurement session. Small timing
differences between sessions are expected. This sweep used the pilot head shape;
end-to-end model checks also cover the follow-up configuration.

## Data and model checks

[corpus-checksums.json](corpus-checksums.json) records the full token-file hashes.
Every worker was checked against these hashes and a full vocabulary scan. Existing
training counts are 990,964,987 for the pilot and 2,500,000,000 for the follow-up;
both reserve 10,000,000 validation tokens. The follow-up loader excludes 241
training-prefix tokens from a document shared with validation. These are exact
processed-token experiments; the pilot repeats a small portion of its corpus.

[tpu-models.json](tpu-models.json) records the bounded distributed model checks,
source fingerprint, optimizer settings, finite training/evaluation metrics,
curvature diagnostics and checkpoint counters. Each fresh check uses seed 1337,
the checked-in global batch of 128, context 1024, FlashAttention, and XLA updates.
The two actual parameter counts are 60,894,720 and 124,373,760. Rocora checks include
a complete four-code residual group; Sophia-G checks include a sampled-label core
event. Outputs live under validation names and have budget_complete=false.

Recorded training time includes input/device transfer, update computation and
curvature events. Compilation, evaluation and checkpoint writes are separate.
Three-step timings include a curvature event at the first step and are not
estimates of amortized full-run overhead or final model quality.

The 60M Rocora checkpoint was resumed for two updates after its initial three.
An independent uninterrupted five-update run produced **bitwise identical values
for all 39 saved parameter/state/counter arrays** on the full 16-device pod.
The resumed checkpoint therefore records step 5; the other fresh model checks
record step 3. The state digest and comparison scope are in tpu-models.json.

The launcher also rejects a second job while the entire pod is locked. Source
sync uses those same locks. No full-budget training was launched during this
refactor, and no legacy training figures or results are carried forward.

## Recheck and launch

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q
git diff --check
python scripts/pod.py sync
python scripts/pod.py stage --config configs/pilot60m.json --dry-run
python scripts/pod.py stage --config configs/pilot60m.json --max-steps 3
```

Bounded runs use outputs/validation/ by default. Reuse their names only with
--resume, or choose a new --run-name for one arm. Fresh production runs use
outputs/pilot60m/ and outputs/scale125m/. See README.md for full-stage launches
and complete-run reports; a smoke run cannot satisfy the follow-up pilot gate.
