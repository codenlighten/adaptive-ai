"""Tiny ternary transformer for physics sequence prediction.

A causal self-attention transformer where every weight matrix inside the
block — Q, K, V, output projection, and the two MLP projections — uses
BitLinear (weights in {-1, 0, +1}). Only the input/output projections to
and from the model dim stay full-precision.

This is the architecture that BitNet b1.58 demonstrated for LLMs;
we apply it to a physics task: given a partial trajectory of a damped
harmonic oscillator, predict the rest.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import BitLinear


class BitSelfAttention(nn.Module):
    """Causal self-attention with ternary weights on Q/K/V/O."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = BitLinear(d_model, 3 * d_model)
        self.proj = BitLinear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class BitMLPBlock(nn.Module):
    """Two-layer feed-forward block with ternary weights, GELU between."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = BitLinear(d_model, d_ff)
        self.fc2 = BitLinear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BitTransformerBlock(nn.Module):
    """Pre-norm transformer block — attention + MLP, both ternary."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BitSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = BitMLPBlock(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BitTrajectoryTransformer(nn.Module):
    """Ternary transformer for oscillator trajectory prediction.

    Input: (B, T, 1) — sequence of x(t) values at uniform dt.
    Output: (B, T, 1) — predicted next-step x(t+dt).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 d_ff: int | None = None, max_len: int = 128):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        # Boundary layers stay full-precision (BitNet convention).
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            BitTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.embed(x) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm_out(h))

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively extend prefix by n_steps."""
        seq = prefix.clone()
        for _ in range(n_steps):
            pred = self(seq)
            seq = torch.cat([seq, pred[:, -1:, :]], dim=1)
        return seq


class FPTrajectoryTransformer(nn.Module):
    """Identical shape but with full-precision Linear everywhere — baseline."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 d_ff: int | None = None, max_len: int = 128):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.embed(x) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm_out(h))

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, n_steps: int) -> torch.Tensor:
        seq = prefix.clone()
        for _ in range(n_steps):
            pred = self(seq)
            seq = torch.cat([seq, pred[:, -1:, :]], dim=1)
        return seq
