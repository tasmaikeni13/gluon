#!/usr/bin/env python3
"""Run targeted low-learning-rate sweep for Sophia-G and Rocora.

Tests learning rates [5e-5, 1e-4, 1.5e-4] at rho=0.02 with 600M tokens and 15M warmup.
Automatically cleans up checkpoints after each trial to guarantee light disk usage.
"""

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.pod import run_workers, train_commands, sync

def main():
    pod_cfg = json.loads((ROOT / "configs/pod.json").read_text())
    base_template = json.loads((ROOT / "configs/generated/scale125m-budget-600m/sophia-t00.json").read_text())

    out_dir = ROOT / "configs/generated/scale125m-lowlr"
    out_dir.mkdir(parents=True, exist_ok=True)

    lrs = [
        ("05e-4", 0.00005),
        ("10e-4", 0.00010),
        ("15e-4", 0.00015),
    ]

    trials = []
    for opt in ["sophia", "rocora"]:
        for label, lr in lrs:
            trial_id = f"{opt}-lowlr-{label}"
            cfg = json.loads(json.dumps(base_template))
            cfg["stage"] = "scale125m-lowlr"
            cfg["arms"] = [opt]
            cfg["optimizer"]["name"] = opt
            cfg["optimizer"]["learning_rate"] = lr
            cfg["optimizer"]["rho"] = 0.02
            cfg["optimizer"]["residual_weight"] = 1.0
            cfg["optimizer"]["pallas_mode"] = "xla"
            cfg["optimizer"]["use_pallas"] = False
            cfg["model"]["attention"] = "flash"

            cfg_path = out_dir / f"{trial_id}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

            run_name = f"tuning/scale125m-lowlr/{trial_id}"
            trials.append({
                "id": trial_id,
                "optimizer": opt,
                "lr": lr,
                "config_path": str(cfg_path),
                "run_name": run_name,
            })

    print(f"Generated {len(trials)} low-LR trial configs in {out_dir}")
    sync(pod_cfg)

    for i, t in enumerate(trials, 1):
        print(f"\n=======================================================", flush=True)
        print(f"[{i}/{len(trials)}] Starting {t['id']} (LR={t['lr']})", flush=True)
        print(f"=======================================================", flush=True)

        extra = ["--run-name", t["run_name"]]
        run_workers(pod_cfg, train_commands(pod_cfg, t["config_path"], t["optimizer"], 1337, extra))

        # Checkpoint cleanup immediately after trial completion to preserve storage
        ckpt_dir = ROOT / "outputs" / t["run_name"] / "checkpoints"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            print(f"Cleaned {ckpt_dir} to maintain free storage headroom.", flush=True)

    print("\n=======================================================")
    print("ALL LOW-LR TRIALS COMPLETED")
    print("=======================================================")

if __name__ == "__main__":
    main()
