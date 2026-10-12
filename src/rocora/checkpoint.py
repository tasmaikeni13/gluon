"""Atomic NPZ checkpoints with strict experiment identity and exact resume."""

import hashlib
import json
import os
from pathlib import Path
import jax
import numpy as np


def identity(cfg):
    raw = cfg.to_dict()
    for key in ("output_dir", "log_interval", "eval_interval", "checkpoint_interval"):
        raw.pop(key)
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def save_checkpoint(path, state, cfg, train_seconds, code_sha256=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree.leaves(state)
    payload = {f"array_{i}": np.asarray(x) for i, x in enumerate(leaves)}
    metadata = {"config": cfg.to_dict(), "identity": identity(cfg),
                "train_seconds": float(train_seconds), "arrays": len(leaves), "code_sha256":code_sha256}
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def load_checkpoint(path, template, cfg, code_sha256=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["identity"] != identity(cfg):
            raise ValueError("Checkpoint architecture/data/optimizer/budget/seed differs from this run")
        if code_sha256 is not None and metadata.get("code_sha256") != code_sha256:
            raise ValueError("Checkpoint was written by different or unrecorded code")
        leaves, structure = jax.tree.flatten(template)
        if len(leaves) != metadata["arrays"]:
            raise ValueError("Checkpoint state structure differs")
        loaded = []
        for i, expected in enumerate(leaves):
            value = archive[f"array_{i}"]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Checkpoint array {i} has the wrong shape or dtype")
            loaded.append(value)
    return structure.unflatten(loaded), metadata["train_seconds"]
