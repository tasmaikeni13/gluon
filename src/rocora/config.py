"""Strict, serializable experiment settings. Token budgets count loss tokens."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    width: int = 512
    layers: int = 11
    heads: int = 8
    sequence_length: int = 1024
    vocab_size: int = 50304
    valid_vocab_size: int = 50257
    mlp_ratio: int = 4
    attention: str = "flash"
    attention_block: int = 128
    loss_chunk: int = 128
    remat: bool = True

    def validate(self):
        if min(self.width, self.layers, self.heads, self.sequence_length) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 1 < self.valid_vocab_size <= self.vocab_size:
            raise ValueError("Invalid vocabulary sizes")
        if self.attention not in ("flash", "reference"):
            raise ValueError("attention must be flash or reference")
        if self.attention == "flash" and (
            self.sequence_length % 128 or self.width // self.heads > 128
        ):
            raise ValueError("TPU FlashAttention requires sequence multiple 128 and head width <=128")
        if self.loss_chunk <= 0 or self.attention_block <= 0:
            raise ValueError("Chunk sizes must be positive")

    @property
    def parameter_count(self):
        d = self.width
        return (self.vocab_size + self.sequence_length) * d + self.layers * (
            (4 + 2 * self.mlp_ratio) * d * d + 2 * d
        ) + d


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "rocora"
    learning_rate: float = 4e-4
    beta1: float = 0.965
    beta2: float = 0.99
    beta_residual: float = 0.95
    rho: float = 0.03
    eps: float = 1e-15
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    core_interval: int = 20
    residual_interval: int = 40
    curvature_batch_per_device: int = 1
    probe_codes: int = 4
    residual_weight: float = 1.0

    def validate(self):
        if self.name not in ("adamw", "sophia", "rocora"):
            raise ValueError("optimizer must be adamw, sophia, or rocora")
        if not all(0 <= b < 1 for b in (self.beta1, self.beta2, self.beta_residual)):
            raise ValueError("EMA betas must be in [0, 1)")
        if min(self.learning_rate, self.eps, self.rho) <= 0:
            raise ValueError("learning_rate, eps, and rho must be positive")
        if self.grad_clip < 0 or self.weight_decay < 0:
            raise ValueError("Clipping and decay cannot be negative")
        if min(self.core_interval, self.residual_interval, self.curvature_batch_per_device) <= 0:
            raise ValueError("Curvature intervals and batch size must be positive")
        if self.probe_codes <= 0 or self.probe_codes & (self.probe_codes - 1):
            raise ValueError("probe_codes must be a power of two")
        if self.residual_weight < 0:
            raise ValueError("residual_weight cannot be negative")


@dataclass(frozen=True)
class ExperimentConfig:
    stage: str = "pilot60m"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 1337
    train_tokens: int = 1_000_000_000
    global_batch_size: int = 128
    warmup_tokens: int = 50_000_000
    min_lr_fraction: float = 0.1
    train_file: str = "/home/tasma/data/tok/fineweb1b_train.npy"
    validation_file: str = "/home/tasma/data/tok/fineweb1b_valid.npy"
    data_metadata: str = "/home/tasma/data/tok/fineweb1b_metadata.json"
    dataset: str = "codelion/fineweb-edu-1B"
    train_skip_initial_document: bool = False
    output_dir: str = "outputs"
    eval_tokens: int = 1_048_576
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    log_interval: int = 20

    def validate(self):
        self.model.validate()
        self.optimizer.validate()
        if min(self.train_tokens, self.global_batch_size, self.eval_tokens) <= 0:
            raise ValueError("Training/evaluation budgets and batch size must be positive")
        if not 0 <= self.warmup_tokens < self.train_tokens:
            raise ValueError("warmup_tokens must be below train_tokens")
        if not 0 < self.min_lr_fraction <= 1:
            raise ValueError("min_lr_fraction must be in (0,1]")
        if min(self.eval_interval, self.checkpoint_interval, self.log_interval) <= 0:
            raise ValueError("Logging/evaluation/checkpoint intervals must be positive")

    @property
    def tokens_per_step(self):
        return self.global_batch_size * self.model.sequence_length

    @property
    def steps(self):
        return math.ceil(self.train_tokens / self.tokens_per_step)

    def to_dict(self):
        return asdict(self)


def load_config(path, optimizer=None, seed=None):
    raw = json.loads(Path(path).read_text())
    model = ModelConfig(**raw.pop("model", {}))
    shared = raw.pop("optimizer", {})
    arms = raw.pop("optimizers", {})
    name = optimizer or shared.get("name", "rocora")
    shared.update(arms.get(name, {}))
    shared["name"] = name
    if seed is not None:
        raw["seed"] = seed
    for k in ("train_file", "validation_file", "data_metadata", "output_dir"):
        if k in raw:
            raw[k] = os.path.expandvars(os.path.expanduser(raw[k]))
    cfg = ExperimentConfig(model=model, optimizer=OptimizerConfig(**shared), **raw)
    cfg.validate()
    return cfg


def learning_rate(step, cfg):
    """Token-based warmup/cosine, using the same schedule on all arms."""
    import jax.numpy as jnp
    tokens = jnp.minimum((step + 1) * float(cfg.tokens_per_step), float(cfg.train_tokens))
    warm = max(float(cfg.warmup_tokens), 1.0)
    # Python integer budgets above 2**31 must not be lowered as JAX int32 args.
    denominator = max(float(cfg.train_tokens) - warm, 1.0)
    progress = jnp.clip((tokens - warm) / denominator, 0, 1)
    cosine = cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * (1 + jnp.cos(jnp.pi * progress)) / 2
    multiplier = jnp.where(tokens < warm, tokens / warm, cosine)
    return cfg.optimizer.learning_rate * multiplier
