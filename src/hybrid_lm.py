"""Hybrid char LM: ternary FFN + fp32 attention.

This is the deploy-realistic config: FFN holds the bulk of the parameters
(typically ~2/3 of a transformer), so ternarizing only the FFN captures
most of the storage and compute savings while leaving the harder-to-quantize
attention path in float. Some quantized-LLM systems (e.g. the original
BitNet variants and many community quantization recipes) follow this.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import BitLinear


class HybridBlock(nn.Module):
    """fp32 multi-head attention + ternary FFN."""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = BitLinear(d_model, d_ff)
        self.fc2 = BitLinear(d_ff, d_model)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class HybridCharLM(nn.Module):
    """Causal char-level LM. Attention QKV + proj: fp32. FFN: ternary."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=64):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            HybridBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def ternary_param_count(self) -> int:
        """How many weights are stored as trits?"""
        return sum(
            blk.fc1.weight.numel() + blk.fc2.weight.numel()
            for blk in self.blocks
        )
