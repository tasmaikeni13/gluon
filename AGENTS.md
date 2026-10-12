# Working on Rocora

Rocora means **Residual Only Coded Optimization with a Regularized Anchor**.
This repository tests whether adding residual-only coded curvature measurements
to a positive Gauss–Newton/Fisher core improves updates enough to pay their
measured computation cost. Do not describe the Hessian split as new or claim
training gains from algebra, toy tests, or smoke runs.

## Scientific controls

- AdamW, Sophia-G, and Rocora are the three experiment arms. Sophia-G and Rocora
  share the core estimator, momentum, clipping, decay, schedule, and core settings.
  Setting Rocora's residual weight to zero must reproduce Sophia-G exactly.
- For Sophia-G's sampled-label core, average gradients across devices with token
  weights **before squaring**, then multiply by the global active-token count.
  Squared real-label gradients are not this estimator.
- Measure every row of a coded residual group at one fixed parameter/data
  snapshot. Freeze the logit loss gradient when differentiating the residual
  operator. Use fresh random flips between groups and identical probes across
  replicas. Do not spread a group across optimizer updates.
- Preserve exact masked token budgets, deterministic data order within seed,
  heldout document separation, and checkpointed optimizer channels/counters.
  Record numerical approximation and estimator limitations in theory.md.
- Keep complete-budget, equal-token and equal-time comparisons separate from
  bounded implementation checks. Never report a smoke run as convergence evidence.

## Runtime and hardware

- The supported implementation is JAX/XLA on Cloud TPU, with JAX's Pallas
  FlashAttention. Tested versions are
  Python 3.10, JAX/jaxlib 0.6.2, libtpu 0.0.17. Do not restore CUDA/PyTorch kernels
  or legacy runs/, figures/, or visualize/ artifacts.
- configs/pod.json describes the current four workers and their Python path.
  A v4-32 slice exposes 16 JAX devices: four per host. Launch through
  `python scripts/pod.py`; each arm uses the entire pod.
- Run only one coordinated TPU job at a time. The launcher claims all worker
  locks before starting and reaps remote process groups on failure/disconnect.
  Do not bypass locks, sync source during a job, or kill unrelated processes.
- Normal training/core events use TPU FlashAttention and XLA-fused optimizer updates.
  Custom optimizer kernels were discarded after measurements showed no meaningful
  advantage over XLA. Do not restore them without a repeatable measured benefit.
  Residual products use the differentiable FP32 attention path because the pinned
  FlashAttention backward does not support higher-order differentiation.
- Source/configuration fingerprints must match on every worker. Production
  checkpoints reject changed source. Keep validation outputs under distinct
  run names and preserve token archives and existing user outputs.

## Checks and repository workflow

Use rg for searches. Read the relevant implementation and configuration before
editing. Keep tests focused on scientific identities, observable behavior,
numerical parity, and failure handling.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /home/tasma/venv/bin/python -m pytest -q
git diff --check
python scripts/pod.py sync
python scripts/pod.py stage --config configs/pilot60m.json --dry-run
```

For changes to shapes, attention, curvature, or optimizer updates, run bounded
hardware checks on the affected configured model/arms and exercise checkpoint
resume where relevant. Use --max-steps with a distinct validation run name;
launch full training only when the user requests it. Record the actual source,
hardware, tokens, checks, and timing scope in docs/validation.md.

Keep README.md, theory.md, configurations, and commands consistent. Do not commit
datasets, checkpoints, compiler caches, logs, or generated training artifacts.
When the user asks to push, inspect the staged diff, commit the authorized work,
push normally without force, and verify the remote commit. Do not request
confirmation again for work already authorized in the conversation.
