"""Deterministic packed-token loading with disjoint hosts and exact budgets."""

import json
from pathlib import Path
import numpy as np


class CorpusPathError(ValueError, AssertionError):
    """Raised when training and validation paths are not distinct."""
    pass


def validate_corpus(cfg, allow_missing: bool = False):
    train_path, val_path = Path(cfg.train_file), Path(cfg.validation_file)
    if train_path.resolve() == val_path.resolve():
        raise CorpusPathError("Training and validation files must be different")
    assert train_path != val_path, "Training and validation files must be different"
    meta_path = Path(cfg.data_metadata)
    if allow_missing and (not meta_path.exists() or not train_path.exists() or not val_path.exists()):
        return {
            "dataset": cfg.dataset,
            "status": "unmaterialized",
            "train_file": str(train_path),
            "validation_file": str(val_path),
            "data_metadata": str(meta_path),
        }
    metadata = json.loads(meta_path.read_text())
    if metadata.get("dataset") != cfg.dataset:
        raise ValueError(f"Wrong dataset provenance: {metadata.get('dataset')} != {cfg.dataset}")
    tokenizer_name = str(metadata.get("tokenizer", "")).lower()
    if "gpt2" in tokenizer_name or tokenizer_name.startswith("gpt-2"):
        raise ValueError("GPT-2 tokenization is deprecated and rejected; modern LLaMA/Mistral tokenization is required")
    if not any(k in tokenizer_name for k in ("llama", "mistral", "sentencepiece", "fast")):
        raise ValueError(f"These configurations require modern LLaMA/Mistral tokenization (got '{metadata.get('tokenizer')}')")
    for name, path in (("train", train_path), ("valid", val_path)):
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        if tokens.ndim != 1 or tokens.dtype != np.uint16:
            raise ValueError(f"{path} must be a flat uint16 .npy token stream")
        if tokens.size <= cfg.model.sequence_length:
            raise ValueError(f"{path} is too short")
        recorded = metadata.get(name + "_tokens")
        if recorded is not None and int(recorded) != tokens.size:
            raise ValueError(f"Token count in {path} disagrees with metadata")
        if name == "train" and tokens.size < cfg.minimum_train_corpus_tokens:
            raise ValueError(f"{path} has {tokens.size} tokens; this recipe requires at least "
                             f"{cfg.minimum_train_corpus_tokens}. Prepare the larger corpus first.")
        if int(tokens[:min(tokens.size, 65536)].max()) >= cfg.model.valid_vocab_size:
            raise ValueError(f"Out-of-vocabulary token in {path}")
    return metadata


class TokenBatches:
    """All hosts use the same permutation; each reads a disjoint slice of it.

    A training budget is tokens processed, rather than unique corpus size. On
    the final step a binary mask removes surplus loss tokens without changing
    compiled shapes. Epoch reshuffling handles training budgets exceeding unique corpus size.
    """

    def __init__(self, path, sequence, global_batch, process_count=1,
                 process_index=0, seed=1337, shuffle=True, token_budget=None,
                 skip_initial_document=False, document_separator=2):
        if global_batch % process_count:
            raise ValueError("Global batch must be divisible by process count")
        self.tokens = np.load(path, mmap_mode="r", allow_pickle=False)
        self.skipped_tokens = 0
        if skip_initial_document:
            # Exclude boundary document remaining prefix from training.
            # Modern LLaMA / Mistral document separator is 2 (</s>).
            for start in range(0, len(self.tokens), 65536):
                endings = np.flatnonzero(self.tokens[start:start+65536] == document_separator)
                if len(endings):
                    self.skipped_tokens = start + int(endings[0]) + 1
                    break
            if not self.skipped_tokens:
                raise ValueError("No document separator in training stream")
            self.tokens = self.tokens[self.skipped_tokens:]
        self.sequence, self.global_batch = sequence, global_batch
        self.local_batch = global_batch // process_count
        self.offset = process_index * self.local_batch
        self.seed, self.shuffle, self.token_budget = seed, shuffle, token_budget
        self.blocks = (len(self.tokens) - 1) // sequence
        if self.blocks == 0:
            raise ValueError("Not enough tokens for one sequence")
        self._epoch, self._permutation = None, None

    def _indices(self, sample_numbers):
        indices = np.empty(sample_numbers.shape, dtype=np.int64)
        epochs = sample_numbers // self.blocks
        for epoch in np.unique(epochs):
            if self._epoch != int(epoch):
                self._epoch = int(epoch)
                self._permutation = (np.random.default_rng(self.seed + int(epoch)).permutation(self.blocks)
                                     if self.shuffle else np.arange(self.blocks))
            selected = epochs == epoch
            indices[selected] = self._permutation[sample_numbers[selected] % self.blocks]
        return indices

    def batch(self, step):
        samples = step * self.global_batch + self.offset + np.arange(self.local_batch)
        indices = self._indices(samples)
        windows = np.stack([self.tokens[i * self.sequence:i * self.sequence + self.sequence + 1]
                            for i in indices]).astype(np.int32)
        positions = samples[:, None] * self.sequence + np.arange(self.sequence)[None, :]
        mask = (positions < self.token_budget).astype(np.float32) if self.token_budget else np.ones_like(positions, np.float32)
        return {"tokens": windows[:, :-1], "targets": windows[:, 1:], "mask": mask}


def shard_batch(batch, devices):
    import jax
    count = len(devices)
    if len(batch["tokens"]) % count:
        raise ValueError("Local batch must be divisible by local device count")
    return jax.tree.map(lambda a: jax.device_put_sharded(list(a.reshape(count, -1, a.shape[-1])), devices), batch)
