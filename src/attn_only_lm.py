"""Inverse hybrid: ternary attention, fp32 FFN.

Phase 17's hybrid kept attention fp32 and ternarized the FFN. This is
the opposite — the FFN is the cheap path, attention is where we
concentrate the discrete weights. Tests where the harder-to-quantize
part of a transformer actually is.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import BitLinear


class _BitAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = BitLinear(d_model, 3 * d_model)
        self.proj = BitLinear(d_model, d_model)

    def forward(self, x):
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


class _AttnOnlyBlock(nn.Module):
    """Ternary attention + fp32 FFN."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _BitAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class AttnOnlyBitLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=64):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            _AttnOnlyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok_embed(idx) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

    def ternary_param_count(self) -> int:
        return sum(
            blk.attn.qkv.weight.numel() + blk.attn.proj.weight.numel()
            for blk in self.blocks
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx
