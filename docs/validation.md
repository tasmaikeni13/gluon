# Implementation validation

These checks validate the implementation and launch path. They do not establish
that Rocora improves convergence or pays for its residual measurements. The
complete-budget 125M, 350M and 1B experiments remain to be run.

## Environment and numerical checks

The hardware is a Cloud TPU v4-32 slice: four hosts, four local JAX devices per
host, 16 devices in each distributed model run. The tested runtime is Python
3.10, JAX/jaxlib 0.6.2 and libtpu 0.0.17.

The current CPU suite passes 33 tests. In addition to the checks recorded in
[tpu-models.json](tpu-models.json), it covers:

- Direct residual products against a dense H−S calculation at one fixed snapshot.
- Exact Sophia-G equivalence when Rocora's residual weight is zero, including
  compiled XLA updates, the published AdamW/Sophia update formulas, and Rocora's
  signed-curvature/clipping rule with gradient scaling.
- Sampled-label core normalization, disjoint host batches, exact masked final
  budgets, and exclusion of the existing corpus's shared boundary document.
- Checkpoint state/seed/code identity, report rejection of incomplete or
  confounded comparisons, and the 2.5B-token schedule beyond the int32 limit.
- SOAP updates against an independent NumPy reference and fixed author vectors;
  Muon momentum, BF16 outputs, matrix orientation and FP32 polynomial behavior.
- Independent per-layer matrix state, exact optimizer checkpoint resume, masked
  microbatch gradient accumulation, all five stage recipes and equal-token sweep
  selection/accounting.
- Exclusive worker locks, clean successful exit, cleanup on launcher disconnect,
  and lock release after the child has been reaped.

The wheel builds with five registered console entry points, and each target
module displays help.
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

The maximum observed update error was zero or within numerical tolerance (<= 1e-5).
Extensive microbenchmarking across 125M, 350M, and 1B scales on Cloud TPU v4 (recorded in
[note.md](../note.md) and [docs/benchmarks/](benchmarks/)) demonstrated that pointwise updates
(AdamW, Sophia-G, Rocora) and Newton–Schulz iterations (Muon) do not beat XLA compiler graph
fusion, which fuses elementwise updates directly into backward passes and eliminates HBM round-trips
for memory-bandwidth bound operations (~0.5 FLOP/byte). While custom Pallas kernels achieved a
+15.5% speedup on SOAP at 125M scale, enabling custom kernels solely for SOAP would introduce an
asymmetric implementation advantage confounding algorithmic evaluation. Consequently, all five
optimizers default to XLA-fused execution (`pallas_mode: "xla"` / `use_pallas: false`), while
custom Pallas kernels are preserved in `src/rocora/pallas_optimizers.py` as supplementary material
and can be toggled on demand via `pallas_mode: "pallas"`.

FlashAttention parity uses BF16 tensors with batch 8, heads 8, sequence 1024 and
head width 64. Maximum observed forward error is 0.001953125 and backward error
is 4.47e−8 against tiled JAX attention. The tuned FlashAttention forward/backward
medians are 0.5295/2.1180 ms, versus 0.6261/2.1345 ms for tiled JAX.
[tpu-tuning.json](tpu-tuning.json) separately records the tile sweep: moving the
KV tile from 128 to 1024 tokens reduced FlashAttention forward/backward medians
from 1.3130/4.2024 ms to 0.5623/2.1905 ms in that measurement session. Small timing
differences between sessions are expected. This sweep used head width 64;
end-to-end model checks cover the 125M, 350M, and 1B configurations.

## Data and model checks

[corpus-checksums.json](corpus-checksums.json) records the full token-file hashes.
Every worker was checked against these hashes and a full vocabulary scan. Existing
training count is 2,500,000,000 for the 125M stage, reserving 10,000,000 validation tokens.
The 125M loader excludes 241 training-prefix tokens from a document shared with validation.
These are exact processed-token experiments.

[tpu-models.json](tpu-models.json) records the bounded distributed model checks,
source fingerprint, optimizer settings, finite training/evaluation metrics,
curvature diagnostics and checkpoint counters. Each fresh check uses seed 1337,
the checked-in global batch of 128, context 1024, FlashAttention, and XLA updates.
The actual parameter count for 125M is 124,373,760 (with 350M at 354,599,936 and 1B at 984,909,312). Rocora checks include
a complete four-code residual group; Sophia-G checks include a sampled-label core
event. Outputs live under validation names and have budget_complete=false.

Recorded training time includes input/device transfer, update computation and
curvature events. Compilation, evaluation and checkpoint writes are separate.
Three-step timings include a curvature event at the first step and are not
estimates of amortized full-run overhead or final model quality.

Checkpoint resume validation is confirmed on the full 16-device pod: a 125M checkpoint
resumed and matched an uninterrupted run bitwise across all saved arrays (see optimizer-validation.json).
The fresh 125M model checks in tpu-models.json record step 3, with bitwise-identical checkpoint resumption
verified on the 16-device pod.

## SOAP, Muon and 1B memory checks

[optimizer-validation.json](optimizer-validation.json) records the bounded checks
added for the five-arm recipes. A 125M SOAP checkpoint resumed from step 3 to step
12 and matched an uninterrupted 12-step run bitwise across all 60 saved arrays;
the compared state digest and both source fingerprints are recorded there. A
separate 125M Muon smoke test completed three updates. At 1B, Rocora's sampled
core and coded residual curvature events compiled separately in 429.5 and 450.4
seconds; the bounded run then completed three updates and evaluation. It processed
393,216 training tokens and used no checkpoint. This confirms that the split
curvature path fits on the pod for this implementation check.

The new smoke checks used the existing 2.5B-token archive, its 10M heldout set,
and the recorded 241-token shared-document prefix exclusion. The 1B smoke recipe
temporarily permits that shorter archive; it does not validate the required 20B
corpus or the 20B-token schedule. No full sweep or main training run was launched.
None of these bounded losses are convergence evidence.

The launcher also rejects a second job while the entire pod is locked. Source
sync uses those same locks. No full-budget training was launched during this
refactor, and no legacy training figures or results are carried forward.

## Recheck and launch

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q
git diff --check
python scripts/pod.py sync
python scripts/pod.py stage --config configs/scale125m.json --dry-run
python scripts/pod.py stage --config configs/scale125m.json --max-steps 3
```

Bounded runs use outputs/validation/ by default. Reuse their names only with
--resume, or choose a new --run-name for one arm. Fresh production runs use
outputs/scale125m/. See README.md for full-stage launches and complete-run reports.
