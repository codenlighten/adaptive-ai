"""Causal char-level LM where every BitLinear is a DeltaSigmaLinear.

Same architecture as BitCharLM (`src/char_lm.py`) but with delta-sigma
weights in Q/K/V/proj and FFN. Embeddings and final unembedding stay
fp32, matching standard BitNet practice.

This tests whether the precision–compute knob of delta-sigma weights
works for autoregressive sequence modeling, not just MLP regression.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsigma_linear import DeltaSigmaLinear


class _DSAttention(nn.Module):
    def __init__(self, d_model, n_heads, T):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = DeltaSigmaLinear(d_model, 3 * d_model, T=T)
        self.proj = DeltaSigmaLinear(d_model, d_model, T=T)

    def forward(self, x):
        B, T_seq, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T_seq, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T_seq, T_seq, device=x.device, dtype=torch.bool),
                          diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T_seq, D)
        return self.proj(out)


class _DSBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, T):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _DSAttention(d_model, n_heads, T=T)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = DeltaSigmaLinear(d_model, d_ff, T=T)
        self.fc2 = DeltaSigmaLinear(d_ff, d_model, T=T)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class DSigmaCharLM(nn.Module):
    """Causal char-level LM with delta-sigma weights everywhere."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=64, T=8):
        super().__init__()
        d_ff = d_ff or 2 * d_model
        self.T = T
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            _DSBlock(d_model, n_heads, d_ff, T=T) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx):
        B, T_seq = idx.shape
        h = self.tok_embed(idx) + self.pos[:, :T_seq]
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
