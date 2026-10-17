"""Deterministic, equal-token budget sweeps and explicit main-recipe selection.

Planning and selection perform no training. The pod launcher executes trials
sequentially using the same worker locks as main training.
"""

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path

from .checkpoint import identity
from .config import OPTIMIZERS, load_config
from .report import load_run


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    base = Path(spec["base_config"])
    if not spec["name"] or Path(spec["name"]).name != spec["name"]:
        raise ValueError("Sweep name must be a single path component")
    axes = spec["axes"]
    if set(axes) != set(OPTIMIZERS):
        raise ValueError(f"A budget sweep must specify all optimizers in {OPTIMIZERS}")
    if axes["sophia"] != axes["rocora"] or ("sophia_h" in axes and axes["sophia_h"] != axes["sophia"]):
        raise ValueError("Sophia/Sophia-H/Rocora must have identical candidate axes")
    tokens, warmup = spec["trial_tokens"], spec["warmup_fraction"]
    if not isinstance(tokens, int) or tokens <= 0 or not 0 <= warmup < 1:
        raise ValueError("Invalid tuning token budget or warmup fraction")
    trials, counts = [], []
    for arm in OPTIMIZERS:
        cfg = load_config(base, arm, spec["seed"])
        if tokens >= cfg.train_tokens:
            raise ValueError("Tuning trial must be smaller than the main budget")
        keys = sorted(axes[arm])
        count = 0
        for multiplier, values in itertools.product(spec["learning_rate_multipliers"],
                                                   itertools.product(*(axes[arm][k] for k in keys))):
            if multiplier <= 0:
                raise ValueError("Learning-rate multipliers must be positive")
            settings = dict(zip(keys, values))
            settings["learning_rate"] = cfg.optimizer.learning_rate * multiplier
            if arm == "muon":
                settings["muon_aux_lr"] = cfg.optimizer.muon_aux_lr * multiplier
            optimizer = replace(cfg.optimizer, **settings)
            trial = replace(cfg, stage=spec["name"], optimizer=optimizer,
                            purpose="tuning", train_tokens=tokens,
                            warmup_tokens=int(tokens * warmup), seeds=(spec["seed"],),
                            arms=(arm,), eval_interval=max(1, (tokens + cfg.tokens_per_step - 1)
                                                          // cfg.tokens_per_step // 4))
            trial.validate()
            trials.append({"id": f"{arm}-t{count:02d}", "optimizer": arm,
                           "run_name": f"tuning/{spec['name']}/{arm}-t{count:02d}",
                           "config": json.loads(json.dumps(trial.to_dict())), "identity": identity(trial)})
            count += 1
        counts.append(count)
    if not counts[0] or len(set(counts)) != 1:
        raise ValueError("Each optimizer must receive the same number of trials")
    return {"name": spec["name"], "spec_sha256": _digest(spec),
            "base_config": str(base), "base_sha256": _digest(json.loads(base.read_text())),
            "trial_tokens": tokens, "trials_per_arm": counts[0],
            "tokens_per_arm": tokens * counts[0], "total_tokens": tokens * len(trials),
            "budget_kind": "equal processed loss tokens; measured compute may differ",
            "trials": trials}


def materialize(spec_path, directory=None):
    manifest = plan(spec_path)
    root = Path(directory or f"configs/generated/{manifest['name']}")
    root.mkdir(parents=True, exist_ok=True)
    for trial in manifest["trials"]:
        path = root / (trial["id"] + ".json")
        if path.exists() and json.loads(path.read_text()) != trial["config"]:
            raise FileExistsError(f"Changed trial at {path}; choose a new sweep name")
        path.write_text(json.dumps(trial["config"], indent=2) + "\n")
        trial["config_path"] = str(path)
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise FileExistsError(f"Changed sweep at {path}; choose a new sweep name")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def select(manifest, output_root, destination, mode="matched"):
    if mode not in ("matched", "independent"):
        raise ValueError("Selection mode must be matched or independent")
    base = json.loads(Path(manifest["base_config"]).read_text())
    if _digest(base) != manifest["base_sha256"]:
        raise ValueError("Base recipe changed since planning")
    candidates = {arm: [] for arm in OPTIMIZERS}
    code = hardware = data = None
    for trial in manifest["trials"]:
        run = load_run(Path(output_root) / trial["run_name"] / "metrics.jsonl")
        meta, final = run["meta"], run["final"]
        cfg = load_config(trial["config_path"], trial["optimizer"])
        # The logged configuration is authoritative, not a trial name or file.
        if _digest(meta["config"]) != _digest(cfg.to_dict()) or identity(cfg) != trial["identity"]:
            raise ValueError(f"Trial configuration differs: {trial['id']}")
        if meta.get("reference") or meta["config"]["purpose"] != "tuning":
            raise ValueError("Reference/smoke outputs cannot select a production recipe")
        this_hardware = tuple(meta[k] for k in ("devices", "processes", "jax", "device_kind"))
        this_data = _digest(meta["data"])
        if code is not None and (code != meta["code_sha256"] or hardware != this_hardware or data != this_data):
            raise ValueError("Sweep trials differ in code, data, or hardware")
        code, hardware, data = meta["code_sha256"], this_hardware, this_data
        candidates[trial["optimizer"]].append((float(final["loss"]), trial))
    winners = {arm: min(rows, key=lambda x: (x[0], x[1]["id"]))[1]
               for arm, rows in candidates.items()}
    if mode == "matched":
        # Choose shared settings by Sophia's validation loss, then compare the
        # residual at those exact settings. Do not optimize the residual arm
        # separately and claim that difference measures the residual channel.
        index = candidates["sophia"].index(next(x for x in candidates["sophia"]
                                                if x[1]["id"] == winners["sophia"]["id"]))
        winners["rocora"] = candidates["rocora"][index][1]
        if "sophia_h" in candidates:
            winners["sophia_h"] = candidates["sophia_h"][index][1]
    base["optimizers"] = {arm: trial["config"]["optimizer"] for arm, trial in winners.items()}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(base, indent=2) + "\n")
    provenance = {"selection": mode, "sweep": manifest["name"], "code_sha256": code,
                  "winners": {a: t["id"] for a, t in winners.items()},
                  "warning": "Validation-selected recipes; evaluation is not an independent test set"}
    destination.with_suffix(".selection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--spec", required=True)
    p.add_argument("--output")
    p = sub.add_parser("select")
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", default="outputs")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("matched", "independent"), default="matched")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = materialize(args.spec, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))
    else:
        result = select(json.loads(Path(args.manifest).read_text()), args.root, args.output, args.mode)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
