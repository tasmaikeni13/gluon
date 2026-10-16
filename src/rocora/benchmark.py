"""Measure XLA optimizer updates and FlashAttention correctness/speed on TPU."""

import argparse
import json
import statistics
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from .attention import flash_attention, reference_attention
from .updates import adamw_update, sophia_update, rocora_update
from .matrix_optimizers import SoapFactors, _covariances, _initial_basis, muon_direction, soap_leaf
from .pallas_optimizers import (
    pallas_adamw_update,
    pallas_sophia_update,
    pallas_rocora_update,
    pallas_muon_direction,
    pallas_soap_leaf,
    is_tpu_platform,
)


def measure(fn, args, repetitions):
    compiled = fn.lower(*args).compile()
    jax.block_until_ready(compiled(*args))
    durations = []
    for _ in range(repetitions):
        t = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        durations.append((time.perf_counter() - t) * 1000)
    return {"median_ms": statistics.median(durations), "min_ms": min(durations)}


def measure_update(fn, arrays, repetitions, kind):
    # Match training's donation and carry updated state between calls.
    arrays = tuple(a.copy() for a in arrays)
    compiled = fn.lower(*arrays).compile()
    durations = []
    for i in range(repetitions + 3):
        t = time.perf_counter()
        result = compiled(*arrays)
        jax.block_until_ready(result)
        if i >= 3:
            durations.append((time.perf_counter() - t) * 1000)
        if kind == "adamw":
            arrays = (result[0], arrays[1], result[1], result[2])
        else:
            arrays = (result[0], arrays[1], result[1], *arrays[3:])
    return {"median_ms": statistics.median(durations), "min_ms": min(durations)}


def update_bench(size, repetitions):
    c = jnp.asarray([4e-4, 0.965, 0.99, 1e-8, 0.1, 0.03, 1.5, 2.0, 1.0, .7], jnp.float32)
    p, g, m, s, r = (jnp.full((size,), x, jnp.float32) for x in (1.0, .01, .1, .2, -.01))
    results = {}
    for kind, fn in (("adamw", adamw_update), ("sophia", sophia_update), ("rocora", rocora_update)):
        arrays = (p, g, m, s) if kind != "rocora" else (p, g, m, s, r)
        donated = (0, 2, 3) if kind == "adamw" else (0, 2)
        compiled_update = jax.jit(lambda *a: fn(*a, c), donate_argnums=donated)
        results[kind] = {"xla": measure_update(compiled_update, arrays, repetitions, kind)}
    return results


SCALE_CONFIGS = {
    "125m": {"heads": 12, "head_dim": 64, "sequence": 1024, "width": 768, "layers": 12},
    "scale125m": {"heads": 12, "head_dim": 64, "sequence": 1024, "width": 768, "layers": 12},
    "350m": {"heads": 16, "head_dim": 64, "sequence": 1024, "width": 1024, "layers": 24},
    "scale350m": {"heads": 16, "head_dim": 64, "sequence": 1024, "width": 1024, "layers": 24},
    "1b": {"heads": 24, "head_dim": 64, "sequence": 1024, "width": 1536, "layers": 32},
    "scale1b": {"heads": 24, "head_dim": 64, "sequence": 1024, "width": 1536, "layers": 32},
}


def optimizer_bench(size=8_388_608, repetitions=20, optimizer=None, scale=None):
    """Benchmark XLA vs Pallas wall-clock timings, speedups, and numerical error across all 5 optimizers."""
    if scale is not None:
        key = str(scale).lower()
        if key in SCALE_CONFIGS:
            width = SCALE_CONFIGS[key].get("width", 768)
        else:
            width = 768
    else:
        width = 768

    optimizers_to_run = (
        ("adamw", "sophia", "rocora", "soap", "muon")
        if (optimizer is None or optimizer == "all")
        else (optimizer.lower(),)
    )

    step = 1
    corr1 = 1.0 / (1.0 - 0.965 ** step)
    corr2 = 1.0 / (1.0 - 0.99 ** step)
    c = jnp.asarray([4e-4, 0.965, 0.99, 1e-8, 0.1, 0.03, corr1, corr2, 1.0, 0.7], jnp.float32)
    p, g, m, s, r = (jnp.full((size,), x, jnp.float32) for x in (1.0, 0.01, 0.1, 0.2, -0.01))

    mat_shape = (width, width)
    mp = jnp.ones(mat_shape, dtype=jnp.float32)
    mg = jnp.full(mat_shape, 0.01, dtype=jnp.float32)
    mm = jnp.zeros_like(mp)
    mv = jnp.zeros_like(mp)

    results = {}

    for opt_name in optimizers_to_run:
        if opt_name == "adamw":
            adamw_lr = 1e-3
            adamw_b1 = 0.9
            adamw_b2 = 0.999
            adamw_eps = 1e-8
            adamw_wd = 0.01
            adamw_grad_scale = 1.0
            corr1_w = 1.0 / (1.0 - adamw_b1 ** step)
            corr2_w = 1.0 / (1.0 - adamw_b2 ** step)
            c_adamw = jnp.asarray([adamw_lr, adamw_b1, adamw_b2, adamw_eps, adamw_wd, 0.0, corr1_w, corr2_w, 0.0, adamw_grad_scale], jnp.float32)
            arrays = (p, g, m, s)
            compiled_xla = jax.jit(lambda *a: adamw_update(*a, c_adamw), donate_argnums=(0, 2, 3))
            compiled_pallas = jax.jit(
                lambda *a: pallas_adamw_update(a[0], a[1], a[2], a[3], adamw_lr, adamw_b1, adamw_b2, adamw_eps, adamw_wd, step=1, grad_scale=adamw_grad_scale),
                donate_argnums=(0, 2, 3),
            )
            xla_out = compiled_xla(*(a.copy() for a in arrays))
            pallas_out = compiled_pallas(*(a.copy() for a in arrays))
            timing_xla = measure_update(compiled_xla, tuple(a.copy() for a in arrays), repetitions, "adamw")
            timing_pallas = measure_update(compiled_pallas, tuple(a.copy() for a in arrays), repetitions, "adamw")
            max_error = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(xla_out, pallas_out))
            speedup = timing_xla["median_ms"] / timing_pallas["median_ms"] if timing_pallas["median_ms"] > 0 else 1.0
            results["adamw"] = {
                "xla": timing_xla,
                "pallas": timing_pallas,
                "speedup": speedup,
                "max_error": max_error,
            }

        elif opt_name == "sophia":
            arrays = (p, g, m, s)
            compiled_xla = jax.jit(lambda *a: sophia_update(*a, c), donate_argnums=(0, 2))
            compiled_pallas = jax.jit(
                lambda *a: pallas_sophia_update(a[0], a[1], a[2], a[3], 4e-4, 0.965, 0.99, 1e-8, 0.03, 0.1, grad_scale=0.7),
                donate_argnums=(0, 2),
            )
            xla_out = compiled_xla(*(a.copy() for a in arrays))
            pallas_out = compiled_pallas(*(a.copy() for a in arrays))
            timing_xla = measure_update(compiled_xla, tuple(a.copy() for a in arrays), repetitions, "sophia")
            timing_pallas = measure_update(compiled_pallas, tuple(a.copy() for a in arrays), repetitions, "sophia")
            max_error = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(xla_out, pallas_out))
            speedup = timing_xla["median_ms"] / timing_pallas["median_ms"] if timing_pallas["median_ms"] > 0 else 1.0
            results["sophia"] = {
                "xla": timing_xla,
                "pallas": timing_pallas,
                "speedup": speedup,
                "max_error": max_error,
            }

        elif opt_name == "rocora":
            arrays = (p, g, m, s, r)
            compiled_xla = jax.jit(lambda *a: rocora_update(*a, c), donate_argnums=(0, 2))
            compiled_pallas = jax.jit(
                lambda *a: pallas_rocora_update(a[0], a[1], a[2], a[3], a[4], 4e-4, 0.965, 0.99, 1e-8, 0.03, 0.1, residual_weight=1.0, grad_scale=0.7),
                donate_argnums=(0, 2),
            )
            xla_out = compiled_xla(*(a.copy() for a in arrays))
            pallas_out = compiled_pallas(*(a.copy() for a in arrays))
            timing_xla = measure_update(compiled_xla, tuple(a.copy() for a in arrays), repetitions, "rocora")
            timing_pallas = measure_update(compiled_pallas, tuple(a.copy() for a in arrays), repetitions, "rocora")
            max_error = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(xla_out, pallas_out))
            speedup = timing_xla["median_ms"] / timing_pallas["median_ms"] if timing_pallas["median_ms"] > 0 else 1.0
            results["rocora"] = {
                "xla": timing_xla,
                "pallas": timing_pallas,
                "speedup": speedup,
                "max_error": max_error,
            }

        elif opt_name == "muon":
            class MuonDummyCfg:
                muon_momentum = 0.95
                muon_ns_steps = 5
                muon_nesterov = True
                learning_rate = 0.02
                weight_decay = 0.1
                eps = 1e-7

            cfg_m = MuonDummyCfg()
            xla_fn = jax.jit(lambda g_, m_: muon_direction(g_, m_, cfg_m))
            pallas_fn = jax.jit(lambda g_, m_: pallas_muon_direction(g_, m_, cfg=cfg_m, use_pallas=True))
            dir_xla, m_xla = xla_fn(mg.copy(), mm.copy())
            dir_pallas, m_pallas = pallas_fn(mg.copy(), mm.copy())
            t_xla = measure(xla_fn, (mg.copy(), mm.copy()), repetitions)
            t_pallas = measure(pallas_fn, (mg.copy(), mm.copy()), repetitions)
            max_error = max(
                float(jnp.max(jnp.abs(dir_xla.astype(jnp.float32) - dir_pallas.astype(jnp.float32)))),
                float(jnp.max(jnp.abs(m_xla.astype(jnp.float32) - m_pallas.astype(jnp.float32)))),
            )
            speedup = t_xla["median_ms"] / t_pallas["median_ms"] if t_pallas["median_ms"] > 0 else 1.0
            results["muon"] = {
                "shape": list(mat_shape),
                "xla": t_xla,
                "pallas": t_pallas,
                "speedup": speedup,
                "max_error": max_error,
            }

        elif opt_name == "soap":
            cov = _covariances(mg, (jnp.zeros(mat_shape, jnp.float32), jnp.zeros(mat_shape, jnp.float32)), 0.95)
            factors = SoapFactors(cov, _initial_basis(cov))

            class SoapDummyCfg:
                beta1 = 0.965
                beta2 = 0.99
                eps = 1e-15
                soap_shampoo_beta = 0.95
                soap_precondition_frequency = 10
                soap_max_precondition_dim = 2048

            cfg_s = SoapDummyCfg()
            xla_fn = jax.jit(lambda p_, g_, m_, v_: soap_leaf(p_, g_, m_, v_, factors, 2, 1e-3, 0.1, cfg_s))
            pallas_fn = jax.jit(lambda p_, g_, m_, v_: pallas_soap_leaf(p_, g_, m_, v_, factors, 2, 1e-3, 0.1, cfg_s, use_pallas=True))
            out_xla = xla_fn(mp.copy(), mg.copy(), mm.copy(), mv.copy())
            out_pallas = pallas_fn(mp.copy(), mg.copy(), mm.copy(), mv.copy())
            t_xla = measure(xla_fn, (mp.copy(), mg.copy(), mm.copy(), mv.copy()), repetitions)
            t_pallas = measure(pallas_fn, (mp.copy(), mg.copy(), mm.copy(), mv.copy()), repetitions)
            max_error = max(float(jnp.max(jnp.abs(out_xla[i] - out_pallas[i]))) for i in range(3))
            speedup = t_xla["median_ms"] / t_pallas["median_ms"] if t_pallas["median_ms"] > 0 else 1.0
            results["soap"] = {
                "shape": list(mat_shape),
                "xla": t_xla,
                "pallas": t_pallas,
                "speedup": speedup,
                "max_error": max_error,
            }

    return {
        "scale": scale,
        "size": size,
        "width": width,
        "repetitions": repetitions,
        "results": results,
    }



def compute_attention_flops(batch, heads, sequence, head_dim):
    """Compute theoretical FLOPs for causal and dense attention forward and backward."""
    causal_forward = 2 * batch * heads * (sequence ** 2) * head_dim
    causal_backward = int(2.5 * causal_forward)
    dense_forward = 4 * batch * heads * (sequence ** 2) * head_dim
    dense_backward = int(2.5 * dense_forward)
    return {
        "forward": dense_forward,
        "forward_causal": causal_forward,
        "backward": dense_backward,
        "backward_causal": causal_backward,
    }


def attention_bench(sequence=1024, repetitions=20, batch=8, heads=None, head_dim=64, scale=None):
    if isinstance(sequence, str):
        scale = sequence
        sequence = 1024
    if scale is not None:
        key = str(scale).lower()
        if key in SCALE_CONFIGS:
            heads = SCALE_CONFIGS[key]["heads"]
            head_dim = SCALE_CONFIGS[key]["head_dim"]
            if sequence == 1024:
                sequence = SCALE_CONFIGS[key]["sequence"]
    if heads is None:
        heads = 8

    shape = (batch, heads, sequence, head_dim)
    q, k, v = [jax.random.normal(jax.random.PRNGKey(i), shape, dtype=jnp.bfloat16) for i in range(3)]
    ref, flash = jax.jit(reference_attention), jax.jit(flash_attention)
    actual, expected = flash(q, k, v), ref(q, k, v)
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=.04, atol=.02)
    def scalar(fn, a, b, c):
        return jnp.mean(jnp.square(fn(a, b, c).astype(jnp.float32)))
    rf = jax.jit(jax.grad(lambda a, b, c: scalar(reference_attention, a, b, c), argnums=(0, 1, 2)))
    ff = jax.jit(jax.grad(lambda a, b, c: scalar(flash_attention, a, b, c), argnums=(0, 1, 2)))
    gradients, reference = ff(q, k, v), rf(q, k, v)
    for x, y in zip(gradients, reference):
        np.testing.assert_allclose(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), rtol=.08, atol=3e-5)

    flops = compute_attention_flops(batch, heads, sequence, head_dim)
    causal_forward_flops = flops["forward_causal"]
    causal_backward_flops = flops["backward_causal"]
    dense_forward_flops = flops["forward"]
    dense_backward_flops = flops["backward"]

    def attach_tflops(timing, causal_flops, dense_flops):
        min_sec = timing["min_ms"] * 1e-3
        med_sec = timing["median_ms"] * 1e-3
        return {
            **timing,
            "tflops_per_sec": (causal_flops / min_sec) / 1e12 if min_sec > 0 else 0.0,
            "tflops_per_sec_median": (causal_flops / med_sec) / 1e12 if med_sec > 0 else 0.0,
            "dense_tflops_per_sec": (dense_flops / min_sec) / 1e12 if min_sec > 0 else 0.0,
            "dense_tflops_per_sec_median": (dense_flops / med_sec) / 1e12 if med_sec > 0 else 0.0,
        }

    return {
        "scale": scale,
        "batch": batch,
        "heads": heads,
        "sequence": sequence,
        "head_dim": head_dim,
        "theoretical_flops": flops,
        "max_forward_error": float(jnp.max(jnp.abs(actual.astype(jnp.float32) - expected.astype(jnp.float32)))),
        "max_backward_error": max(float(jnp.max(jnp.abs(x.astype(jnp.float32) - y.astype(jnp.float32)))) for x, y in zip(gradients, reference)),
        "xla_forward": attach_tflops(measure(ref, (q, k, v), repetitions), causal_forward_flops, dense_forward_flops),
        "flash_forward": attach_tflops(measure(flash, (q, k, v), repetitions), causal_forward_flops, dense_forward_flops),
        "xla_backward": attach_tflops(measure(rf, (q, k, v), repetitions), causal_backward_flops, dense_backward_flops),
        "flash_backward": attach_tflops(measure(ff, (q, k, v), repetitions), causal_backward_flops, dense_backward_flops),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8_388_608)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")
    parser.add_argument("--scale", type=str, default=None,
                        choices=["125m", "scale125m", "350m", "scale350m", "1b", "scale1b"],
                        help="Model scale configuration")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adamw", "sophia", "rocora", "soap", "muon", "all"],
                        help="Optimizer to benchmark (default: all)")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if min(args.size, args.sequence, args.batch, args.repetitions) <= 0:
        parser.error("Sizes and repetitions must be positive")
    if any(d.platform != "tpu" for d in jax.devices()):
        raise RuntimeError("Hardware benchmark requires TPU devices")
    result = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
              "size": args.size, "sequence": args.sequence, "batch": args.batch,
              "scale": args.scale, "heads": args.heads,
              "optimizer": args.optimizer,
              "updates": update_bench(args.size, args.repetitions),
              "optimizer_bench": optimizer_bench(args.size, args.repetitions,
                                                 optimizer=args.optimizer, scale=args.scale),
              "attention": attention_bench(args.sequence, args.repetitions, args.batch,
                                           heads=args.heads, head_dim=args.head_dim, scale=args.scale)}
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
