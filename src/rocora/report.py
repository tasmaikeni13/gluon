"""Compare complete matched runs by loss, tokens and measured training time."""

import argparse
import json
from pathlib import Path
import statistics


def load_run(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    meta = next(r for r in rows if r["type"] == "meta")
    for record in rows:
        if record["type"] == "resume":
            if record["code_sha256"] != meta["code_sha256"]:
                raise ValueError(f"{path} changed code during training")
            observer = ("output_dir","log_interval","eval_interval","checkpoint_interval")
            original = {k:v for k,v in meta["config"].items() if k not in observer}
            resumed = {k:v for k,v in record["config"].items() if k not in observer}
            if original != resumed:
                raise ValueError(f"{path} changed experiment configuration during training")
    complete = [r for r in rows if r["type"] == "complete"]
    if not complete or not complete[-1]["budget_complete"]:
        raise ValueError(f"{path} has not completed its token budget")
    latest = {r["step"]: r for r in rows if r["type"] == "eval"}
    evaluations = [latest[s] for s in sorted(latest)]
    if not evaluations or evaluations[-1]["tokens"] != meta["config"]["train_tokens"]:
        raise ValueError(f"{path} lacks evaluation at the final token budget")
    return {"meta": meta, "eval": evaluations, "final": evaluations[-1]}


def interpolate(evals, seconds):
    if not evals[0]["train_seconds"] <= seconds <= evals[-1]["train_seconds"]:
        raise ValueError("Equal-time comparison lies outside the evaluation range")
    for left, right in zip(evals, evals[1:]):
        if left["train_seconds"] <= seconds <= right["train_seconds"]:
            weight = (seconds - left["train_seconds"]) / (right["train_seconds"] - left["train_seconds"])
            return left["loss"] * (1 - weight) + right["loss"] * weight
    return evals[-1]["loss"]


def summarize(paths):
    runs = {}
    for path in paths:
        run = load_run(path)
        key = (run["meta"]["seed"], run["meta"]["optimizer"])
        if key in runs:
            raise ValueError(f"Duplicate arm/seed {key}")
        runs[key] = run
    seeds = sorted({s for s, _ in runs})
    if not seeds:
        raise ValueError("No runs found")
    arms = ("adamw", "sophia", "rocora")
    for seed in seeds:
        if any((seed, arm) not in runs for arm in arms):
            raise ValueError(f"Seed {seed} is missing an optimizer arm")
    shared = ("stage", "model", "train_tokens", "global_batch_size", "warmup_tokens",
              "min_lr_fraction", "train_file", "validation_file", "data_metadata", "dataset", "eval_tokens")
    shared += ("train_skip_initial_document",)
    first = next(iter(runs.values()))["meta"]
    for run in runs.values():
        if any(run["meta"]["config"][k] != first["config"][k] for k in shared):
            raise ValueError("Runs differ in architecture, data, schedule, or budget")
        if run["meta"]["code_sha256"] != first["code_sha256"]:
            raise ValueError("Runs were produced by different code")
        for key in ("devices","processes","jax","device_kind"):
            if key not in run["meta"] or run["meta"][key] != first.get(key):
                raise ValueError("Runs differ in hardware or JAX runtime")
    for seed in seeds:
        s, r = [runs[(seed,arm)]["meta"]["config"]["optimizer"] for arm in ("sophia","rocora")]
        for key in ("learning_rate","beta1","beta2","rho","eps","weight_decay","grad_clip",
                    "core_interval","curvature_batch_per_device"):
            if s[key] != r[key]:
                raise ValueError(f"Sophia/Rocora differ in {key}; residual contribution is confounded")
    summary = {"seeds": seeds, "train_tokens": first["config"]["train_tokens"], "arms": {}, "paired": []}
    for arm in arms:
        losses = [runs[(seed,arm)]["final"]["loss"] for seed in seeds]
        times = [runs[(seed,arm)]["final"]["train_seconds"] for seed in seeds]
        summary["arms"][arm] = {"mean_loss": statistics.mean(losses),
                                "std_loss": statistics.stdev(losses) if len(losses)>1 else None,
                                "mean_train_seconds": statistics.mean(times)}
    for seed in seeds:
        core, residual = runs[(seed,"sophia")], runs[(seed,"rocora")]
        common = min(core["final"]["train_seconds"], residual["final"]["train_seconds"])
        summary["paired"].append({"seed": seed,
            "core_minus_residual_loss_equal_tokens": core["final"]["loss"]-residual["final"]["loss"],
            "core_minus_residual_loss_equal_time_interpolated": interpolate(core["eval"],common)-interpolate(residual["eval"],common),
            "equal_time_seconds": common,
            "residual_time_overhead_fraction": residual["final"]["train_seconds"]/core["final"]["train_seconds"]-1})
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output")
    args=parser.parse_args(argv)
    result=summarize(sorted(Path(args.root).glob("*/metrics.jsonl")))
    print(json.dumps(result,indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
