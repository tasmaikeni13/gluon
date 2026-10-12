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


def attention_bench(sequence, repetitions, batch=8):
    shape = (batch, 8, sequence, 64)
    q, k, v = [jax.random.normal(jax.random.PRNGKey(i), shape, dtype=jnp.bfloat16) for i in range(3)]
    ref, flash = jax.jit(reference_attention), jax.jit(flash_attention)
    actual, expected = flash(q,k,v), ref(q,k,v)
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=.04, atol=.02)
    def scalar(fn, a,b,c):
        return jnp.mean(jnp.square(fn(a,b,c).astype(jnp.float32)))
    rf = jax.jit(jax.grad(lambda a,b,c: scalar(reference_attention,a,b,c), argnums=(0,1,2)))
    ff = jax.jit(jax.grad(lambda a,b,c: scalar(flash_attention,a,b,c), argnums=(0,1,2)))
    gradients, reference = ff(q,k,v), rf(q,k,v)
    for x,y in zip(gradients,reference):
        np.testing.assert_allclose(np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.float32), rtol=.08, atol=3e-5)
    return {"max_forward_error": float(jnp.max(jnp.abs(actual.astype(jnp.float32)-expected.astype(jnp.float32)))),
            "max_backward_error": max(float(jnp.max(jnp.abs(x.astype(jnp.float32)-y.astype(jnp.float32)))) for x,y in zip(gradients,reference)),
            "xla_forward": measure(ref,(q,k,v),repetitions),
            "flash_forward": measure(flash,(q,k,v),repetitions),
            "xla_backward": measure(rf,(q,k,v),repetitions),
            "flash_backward": measure(ff,(q,k,v),repetitions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8_388_608)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if min(args.size,args.sequence,args.batch,args.repetitions) <= 0:
        parser.error("Sizes and repetitions must be positive")
    if any(d.platform != "tpu" for d in jax.devices()):
        raise RuntimeError("Hardware benchmark requires TPU devices")
    result = {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
              "size": args.size, "sequence": args.sequence, "batch": args.batch,
              "updates": update_bench(args.size,args.repetitions),
              "attention": attention_bench(args.sequence,args.repetitions,args.batch)}
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
