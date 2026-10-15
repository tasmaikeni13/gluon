"""Data-parallel JAX training with FlashAttention and XLA-fused updates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from .checkpoint import identity, load_checkpoint, save_checkpoint
from .config import OPTIMIZERS, load_config, learning_rate
from .curvature import core_gradients, residual_diagonal
from .data import TokenBatches, shard_batch, validate_corpus
from .model import decay_mask, init_params, loss_sum
from .optimizers import init_state, update, commit_curvature, gradient_norm_and_scale, curvature_diagnostics


def initialize_distributed(coordinator=None, processes=None, process_id=None):
    if processes is not None and processes > 1:
        if coordinator is None or process_id is None:
            raise ValueError("Multi-host jobs require coordinator, processes, and process-id")
        jax.distributed.initialize(coordinator_address=coordinator,
                                   num_processes=processes, process_id=process_id)


def build_steps(cfg, params):
    opt_cfg, model_cfg = cfg.optimizer, cfg.model
    decay = decay_mask(params)

    def average_tree(tree, local_count):
        total = jax.lax.psum(local_count, "data")
        return jax.tree.map(lambda g: jax.lax.psum(g * local_count, "data") / jnp.maximum(total, 1), tree)

    def local_gradient(params, batch):
        size = batch["tokens"].shape[0]
        micro = min(cfg.microbatch_per_device, size)
        if size % micro:
            raise ValueError("Per-device batch must be divisible by microbatch_per_device")
        batches = jax.tree.map(lambda x: x.reshape((-1, micro) + x.shape[1:]), batch)
        initial = (jnp.float32(0), jax.tree.map(jnp.zeros_like, params))
        def accumulate(carry, piece):
            value, gradient = jax.value_and_grad(lambda p: loss_sum(p, piece, model_cfg))(params)
            return (carry[0] + value, jax.tree.map(jnp.add, carry[1], gradient)), None
        (total_loss, grads), _ = jax.lax.scan(accumulate, initial, batches)
        count = jnp.sum(batch["mask"])
        denominator = jnp.maximum(count, 1)
        return total_loss / denominator, jax.tree.map(lambda g: g / denominator, grads), count

    def train_step(state, batch):
        loss, grads, count = local_gradient(state.params, batch)
        grads = average_tree(grads, count)
        norm, scale = gradient_norm_and_scale(grads, opt_cfg.grad_clip)
        lr = learning_rate(state.optimizer.step, cfg)
        state = update(state, grads, lr, opt_cfg, decay, gradient_scale=scale)
        total = jax.lax.psum(count, "data")
        loss = jax.lax.psum(loss * count, "data") / jnp.maximum(total, 1)
        return state, {"loss": loss, "grad_norm": norm, "learning_rate": lr,
                       "tokens": total}

    def evaluation(params, batch):
        size = batch["tokens"].shape[0]
        micro = min(cfg.microbatch_per_device, size)
        batches = jax.tree.map(lambda x: x.reshape((-1, micro) + x.shape[1:]), batch)
        losses = jax.lax.map(lambda b: loss_sum(params, b, model_cfg), batches)
        return {"loss_sum": jax.lax.psum(jnp.sum(losses), "data"),
                "tokens": jax.lax.psum(jnp.sum(batch["mask"]), "data")}

    def curvature_step(state, batch, do_core, do_residual):
        batch = jax.tree.map(lambda a: a[:opt_cfg.curvature_batch_per_device], batch)
        key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 7411), state.optimizer.step)
        core, residual = None, None
        if do_core:
            core_key = jax.random.fold_in(jax.random.fold_in(key, 0), jax.lax.axis_index("data"))
            grads, count = core_gradients(state.params, batch, model_cfg, core_key)
            grads = average_tree(grads, count)
            total = jax.lax.psum(count,"data")
            core = jax.tree.map(lambda g: total*jnp.square(g),grads)
        if do_residual:
            # Every replica uses the SAME probe vector; products are averaged over data.
            # Carry the residual EMA through the complete group instead of
            # retaining both an old EMA and a parameter-sized sample buffer.
            initial = jax.tree.map(lambda r: opt_cfg.beta_residual * r, state.optimizer.residual)
            residual, count = residual_diagonal(state.params, batch, model_cfg,
                                                 jax.random.fold_in(key, 1), opt_cfg.probe_codes,
                                                 initial=initial,
                                                 sample_weight=1-opt_cfg.beta_residual)
            residual = average_tree(residual, count)
        state = commit_curvature(state, core, None, opt_cfg)
        if do_residual:
            state = state._replace(optimizer=state.optimizer._replace(
                residual=residual, residual_events=state.optimizer.residual_events+1))
        return state

    return (
        jax.pmap(train_step, axis_name="data", donate_argnums=(0,)),
        jax.pmap(evaluation, axis_name="data"),
        jax.pmap(curvature_step, axis_name="data", static_broadcasted_argnums=(2, 3), donate_argnums=(0,)),
        jax.pmap(lambda state: curvature_diagnostics(state,opt_cfg), axis_name="data"),
    )


def _first(tree):
    return jax.tree.map(lambda x: np.asarray(x[0]), tree)


def _code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--coordinator")
    parser.add_argument("--processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--max-steps", type=int, help="Bounded smoke run; does not change the experiment budget")
    parser.add_argument("--skip-checkpoint", action="store_true",
                        help="Bounded validation only: avoid writing a multi-GB checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference", action="store_true", help="Differentiable attention for CPU validation")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.optimizer, args.seed)
    if args.reference:
        cfg = replace(cfg, model=replace(cfg.model, attention="reference"))
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if args.skip_checkpoint and (args.max_steps is None or args.resume):
        parser.error("--skip-checkpoint requires --max-steps and cannot be used with --resume")
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_COMPILATION_CACHE_DIR", str(Path(".cache/jax").resolve())))
    initialize_distributed(args.coordinator, args.processes, args.process_id)
    host, hosts = jax.process_index(), jax.process_count()
    devices = jax.local_devices()
    count = jax.device_count()
    if cfg.model.attention == "flash" and any(d.platform != "tpu" for d in devices):
        raise RuntimeError("FlashAttention requires TPU devices. Use --reference for CPU checks.")
    if cfg.global_batch_size % count:
        raise ValueError(f"Global batch {cfg.global_batch_size} is not divisible by {count} devices")
    if cfg.optimizer.curvature_batch_per_device > cfg.global_batch_size // count:
        raise ValueError("Curvature batch cannot exceed the training batch per device")
    data_metadata = validate_corpus(cfg)
    fingerprint = bytes.fromhex(identity(cfg) + _code_digest())
    fingerprints = np.asarray(multihost_utils.process_allgather(np.frombuffer(fingerprint, np.uint8)))
    if not np.all(fingerprints.reshape(hosts, -1) == fingerprints.reshape(hosts, -1)[0]):
        raise RuntimeError("Workers have different configuration or code")
    summary = {"optimizer": cfg.optimizer.name, "seed": cfg.seed,
               "parameters": cfg.model.parameter_count, "train_tokens": cfg.train_tokens,
               "steps": cfg.steps, "processes": hosts, "devices": count,
               "local_devices": len(devices), "jax": jax.__version__, "data": data_metadata}
    summary["device_kind"] = devices[0].device_kind
    if host == 0:
        print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return
    run_name = args.run_name or (("validation/" if args.max_steps else "")
                               + f"{cfg.stage}/{args.optimizer}-s{cfg.seed}")
    run_dir = Path(cfg.output_dir) / run_name
    checkpoint = run_dir / "checkpoints" / "latest.npz"
    # All hosts receive the same decision; fail before entering collective training.
    allowed = np.int32(not (host == 0 and run_dir.exists() and not args.resume))
    allowed = multihost_utils.broadcast_one_to_all(allowed)
    if not allowed:
        raise FileExistsError(f"{run_dir} already exists. Resume it or choose another run-name.")
    if args.resume:
        exists = multihost_utils.broadcast_one_to_all(np.int32(checkpoint.exists() if host == 0 else 0))
        if not exists:
            raise FileNotFoundError(checkpoint)
    params = init_params(cfg.model, cfg.seed)
    if sum(p.size for p in jax.tree.leaves(params)) != cfg.model.parameter_count:
        raise AssertionError("Architecture parameter count mismatch")
    state = init_state(params, cfg.optimizer)
    train_seconds = 0.0
    if args.resume:
        if host == 0:
            state, train_seconds = load_checkpoint(checkpoint, state, cfg, _code_digest())
        state = multihost_utils.broadcast_one_to_all(jax.tree.map(np.asarray, state))
        train_seconds = float(multihost_utils.broadcast_one_to_all(np.asarray(train_seconds)))
    start_step = int(state.optimizer.step)
    state = jax.device_put_replicated(state, devices)
    train_step, evaluate, curvature_step, diagnostics = build_steps(cfg, params)
    # The replicated state owns the live weights. Keeping the initialization
    # arrays too would retain another ~3.7 GiB on device 0 at nominal 1B scale.
    del params
    loader = TokenBatches(cfg.train_file, cfg.model.sequence_length, cfg.global_batch_size,
                         hosts, host, cfg.seed, True, cfg.train_tokens,
                         cfg.train_skip_initial_document)
    val_loader = TokenBatches(cfg.validation_file, cfg.model.sequence_length, cfg.global_batch_size,
                             hosts, host, cfg.seed, False, cfg.eval_tokens)
    log_file = None
    if host == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = (run_dir / "metrics.jsonl").open("a", buffering=1)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    def log(record):
        if log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")

    log({"type": "resume" if args.resume else "meta", **summary,
         "config": cfg.to_dict(), "code_sha256": _code_digest(),
         "start_step": start_step, "reference": args.reference,
         "bounded": args.max_steps is not None, "checkpoint_enabled": not args.skip_checkpoint,
         "train_skipped_prefix_tokens": loader.skipped_tokens,
         "usable_corpus_tokens": int(loader.tokens.size)})
    stop = min(cfg.steps, start_step + args.max_steps) if args.max_steps else cfg.steps
    if stop <= start_step:
        if log_file:
            log_file.close()
        return
    elapsed, curvature_elapsed, loss_total, n, window_tokens = 0.0, 0.0, 0.0, 0, 0
    compiled = {}
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        next_batch = prefetch.submit(loader.batch, start_step)
        for step in range(start_step, stop):
            step_started = time.perf_counter()
            compile_seconds = 0.0
            batch = next_batch.result()
            if step + 1 < stop:
                next_batch = prefetch.submit(loader.batch, step + 1)
            batch = shard_batch(batch, devices)
            do_core = (cfg.optimizer.name in ("sophia", "rocora")
                       and step % cfg.optimizer.core_interval == 0)
            do_residual = (cfg.optimizer.name == "rocora" and cfg.optimizer.residual_weight > 0
                           and step % cfg.optimizer.residual_interval == 0)
            event_seconds = 0.0
            # Separate events share the exact parameter/data snapshot and keys.
            # This bounds compile/temporary memory without overlapping jobs or
            # carrying any coded row across an optimizer update.
            events = ([(True, False)] if do_core else []) + ([(False, True)] if do_residual else [])
            for event in events:
                t0 = time.perf_counter()
                if event not in compiled:
                    compiled[event] = curvature_step.lower(state, batch, *event).compile()
                    multihost_utils.sync_global_devices(f"compile-curvature-{event}")
                    compile_seconds += time.perf_counter() - t0
                    log({"type": "compile", "kind": "curvature", "core": event[0],
                         "residual": event[1], "seconds": time.perf_counter() - t0})
                t0 = time.perf_counter()
                state = compiled[event](state, batch)
                jax.block_until_ready(state)
                event_seconds += time.perf_counter() - t0
            if "train" not in compiled:
                t0 = time.perf_counter()
                compiled["train"] = train_step.lower(state, batch).compile()
                multihost_utils.sync_global_devices("compile-train")
                compile_seconds += time.perf_counter() - t0
                log({"type": "compile", "kind": "train", "seconds": time.perf_counter() - t0})
            t0 = time.perf_counter()
            state, metrics = compiled["train"](state, batch)
            jax.block_until_ready((state, metrics))
            metrics = _first(metrics)
            duration = time.perf_counter() - step_started - compile_seconds
            if not np.isfinite(metrics["loss"]) or not np.isfinite(metrics["grad_norm"]):
                raise FloatingPointError(f"Nonfinite training value at step {step + 1}")
            elapsed += duration
            curvature_elapsed += event_seconds
            train_seconds += duration
            loss_total += float(metrics["loss"])
            n += 1
            window_tokens += int(metrics["tokens"])
            tokens = min((step + 1) * cfg.tokens_per_step, cfg.train_tokens)
            if (step + 1) % cfg.log_interval == 0 or step + 1 == stop:
                record = {"type": "train", "step": step + 1, "tokens": tokens,
                          "loss": loss_total / n, "grad_norm": float(metrics["grad_norm"]),
                          "learning_rate": float(metrics["learning_rate"]),
                          "seconds_per_step": elapsed / n, "curvature_seconds_per_step": curvature_elapsed / n,
                          "train_seconds": train_seconds, "tokens_per_second":
                          window_tokens / max(elapsed, 1e-9)}
                log(record)
                if host == 0:
                    print(f"step {step+1}/{cfg.steps} loss={record['loss']:.4f} "
                          f"tokens={tokens:,} time={train_seconds:.1f}s", flush=True)
                elapsed, curvature_elapsed, loss_total, n, window_tokens = 0.0, 0.0, 0.0, 0, 0
            if (step + 1) % cfg.eval_interval == 0 or step + 1 == stop:
                if cfg.optimizer.name in ("sophia", "rocora"):
                    if "diagnostics" not in compiled:
                        compiled["diagnostics"] = diagnostics.lower(state).compile()
                        multihost_utils.sync_global_devices("compile-diagnostics")
                    stats = _first(compiled["diagnostics"](state))
                    if not all(np.isfinite(x) for x in stats.values()):
                        raise FloatingPointError("Nonfinite curvature diagnostics")
                    log({"type":"diagnostics","step":step+1,"tokens":tokens,
                         **{k:float(v) for k,v in stats.items()}})
                t0 = time.perf_counter()
                ev_loss, ev_tokens = 0.0, 0
                for ev_step in range(math.ceil(cfg.eval_tokens / cfg.tokens_per_step)):
                    ev_batch = shard_batch(val_loader.batch(ev_step), devices)
                    result = _first(evaluate(state.params, ev_batch))
                    ev_loss += float(result["loss_sum"])
                    ev_tokens += int(result["tokens"])
                val_loss = ev_loss / ev_tokens
                if not math.isfinite(val_loss):
                    raise FloatingPointError(f"Nonfinite validation loss at step {step + 1}")
                log({"type": "eval", "step": step + 1, "tokens": tokens,
                     "validation_tokens": ev_tokens, "loss": val_loss,
                     "perplexity": math.exp(min(val_loss, 30)), "train_seconds": train_seconds,
                     "eval_seconds": time.perf_counter() - t0})
                if host == 0:
                    print(f"validation loss={val_loss:.4f}", flush=True)
            if not args.skip_checkpoint and ((step + 1) % cfg.checkpoint_interval == 0 or step + 1 == stop):
                multihost_utils.sync_global_devices(f"checkpoint-{step}")
                if host == 0:
                    save_checkpoint(checkpoint, _first(state), cfg, train_seconds, _code_digest())
                multihost_utils.sync_global_devices(f"checkpoint-done-{step}")
    log({"type": "complete", "step": stop, "tokens": min(stop * cfg.tokens_per_step, cfg.train_tokens),
         "budget_complete": stop == cfg.steps, "train_seconds": train_seconds})
    if log_file:
        log_file.close()


def entrypoint():
    try:
        main()
    except Exception:
        if jax.distributed.is_initialized():
            # A failed worker can otherwise hang at interpreter shutdown while
            # JAX's coordination threads wait for peers. Let the pod launcher
            # observe failure promptly and reap the remaining process groups.
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        raise


if __name__ == "__main__":
    entrypoint()
