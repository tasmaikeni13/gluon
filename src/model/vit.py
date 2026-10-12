"""~39M Vision Transformer for CIFAR-100 (bias-free, pre-norm, GELU, SDPA).

d=520, L=12, heads=8 (head dim 65), patch 4 on 32x32 -> 64 tokens + CLS.
Parameter count ~39.0M — matched to the 39M language model for a paired benchmark.
GELU keeps the network C² (the split/probe theory assumes twice-differentiability).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    img_size: int = 32
    patch: int = 4
    n_classes: int = 100
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 520
    dropout: float = 0.0


class LayerNorm(nn.Module):
    """LayerNorm with weight only (no bias)."""

    def __init__(self, ndim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        n_tok = (cfg.img_size // cfg.patch) ** 2
        in_dim = 3 * cfg.patch * cfg.patch
        self.patch_embed = nn.Linear(in_dim, cfg.n_embd, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.n_classes, bias=False)

        nn.init.normal_(self.pos, 0.0, 0.02)
        nn.init.normal_(self.cls, 0.0, 0.02)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, targets=None):
        """x: (B, 3, 32, 32) float (already normalized). Returns logits or CE loss."""
        B = x.shape[0]
        P = self.cfg.patch
        # (B,3,H,W) -> (B, n_tok, 3*P*P)
        x = x.unfold(2, P, P).unfold(3, P, P)          # B,3,h,w,P,P
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * P * P)
        x = self.patch_embed(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        x = self.drop(x + self.pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x[:, 0])
        if targets is None:
            return logits
        return F.cross_entropy(logits, targets)

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
