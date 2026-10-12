"""Stream FineWeb-Edu, split by document content, and write uint16 NPY streams."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class TokenWriter:
    """Append without holding the corpus in memory; finalize with a valid NPY header."""
    def __init__(self, path):
        self.path = Path(path)
        self.raw_path = self.path.with_suffix(".partial.bin")
        self.stream = self.raw_path.open("wb")
        self.count = 0

    def append(self, tokens, limit=None):
        data = np.asarray(tokens, dtype=np.uint16)
        if limit is not None:
            data = data[:max(limit - self.count, 0)]
        data.tofile(self.stream)
        self.count += len(data)

    def finish(self):
        self.stream.close()
        if self.count == 0:
            raise ValueError(f"No tokens for {self.path}")
        temporary = self.path.with_suffix(".partial.npy")
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint16, shape=(self.count,))
        source = np.memmap(self.raw_path, mode="r", dtype=np.uint16)
        for start in range(0, self.count, 8_000_000):
            output[start:start + 8_000_000] = source[start:start + 8_000_000]
        output.flush()
        del output, source
        os.replace(temporary, self.path)
        self.raw_path.unlink()


def is_validation_document(text):
    # Identical documents always land on the same side, even if their URLs differ.
    digest = hashlib.blake2b(text.encode(), digest_size=8, person=b"rocora-split").digest()
    return int.from_bytes(digest, "little") % 1000 < 10


def prepare(dataset, output, dataset_config=None, revision=None, train_limit=None,
            validation_tokens=10_000_000, batch_size=512):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import tiktoken
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / f).exists() for f in ("train.npy", "valid.npy", "metadata.json")):
        raise FileExistsError(f"Prepared data already exists in {root}; use a new output directory")
    resolved_revision = HfApi().dataset_info(dataset, revision=revision).sha
    ds = load_dataset(dataset, name=dataset_config, revision=resolved_revision,
                      split="train", streaming=True)
    enc = tiktoken.get_encoding("gpt2")
    writers = {"train": TokenWriter(root / "train.npy"), "valid": TokenWriter(root / "valid.npy")}
    documents = {"train": 0, "valid": 0}
    pending = []
    start = time.perf_counter()

    def flush():
        nonlocal pending
        encoded = enc.encode_ordinary_batch(pending, num_threads=16)
        for text, tokens in zip(pending, encoded):
            name = "valid" if is_validation_document(text) else "train"
            writer = writers[name]
            limit = validation_tokens if name == "valid" else train_limit
            if limit is None or writer.count < limit:
                writer.append(tokens + [enc.eot_token], limit)
                documents[name] += 1
        pending = []

    for document in ds:
        text = document["text"]
        if text:
            pending.append(text)
        if len(pending) >= batch_size:
            flush()
            if (train_limit is not None and writers["train"].count >= train_limit
                    and writers["valid"].count >= validation_tokens):
                break
    if pending:
        flush()
    for writer in writers.values():
        writer.finish()
    if writers["valid"].count < validation_tokens:
        raise ValueError("Dataset exhausted before reaching the validation budget")
    if train_limit is not None and writers["train"].count < train_limit:
        raise ValueError("Dataset exhausted before reaching train-limit. Use a larger FineWeb-Edu sample.")
    metadata = {"dataset": dataset, "dataset_config": dataset_config, "revision": resolved_revision,
                "tokenizer": "gpt2 (tiktoken)", "document_separator": enc.eot_token,
                "split": "content hash modulo 1000; validation <10; no heldout documents enter training",
                "train_tokens": writers["train"].count, "valid_tokens": writers["valid"].count,
                "train_documents": documents["train"], "valid_documents": documents["valid"],
                "train_sha256": file_sha256(root / "train.npy"),
                "valid_sha256": file_sha256(root / "valid.npy"),
                "seconds": time.perf_counter() - start}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="codelion/fineweb-edu-1B")
    parser.add_argument("--dataset-config")
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-tokens", type=int, default=10_000_000)
    args = parser.parse_args(argv)
    if args.validation_tokens <= 0 or (args.train_limit is not None and args.train_limit <= 0):
        parser.error("Token limits must be positive")
    prepare(args.dataset, args.output, args.dataset_config, args.revision,
            args.train_limit, args.validation_tokens)


if __name__ == "__main__":
    main()
