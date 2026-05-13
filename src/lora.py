"""Ternary LoRA: low-rank ternary adapter on top of a frozen fp32 layer.

Given a frozen fp32 weight W (out, in), we add a learned delta:
    W_eff = W + (alpha / r) * (B_t @ A_t)
where A_t in {-1,0,+1}^(r, in) and B_t in {-1,0,+1}^(out, r) are ternary
selectors with their own scale, and r is the LoRA rank (small).

Storage: the base W stays fp32 (frozen, untouched). The delta uses
(r * in + out * r) trits, which at r=8 and dim=128 is 2 * 1024 = 2048
trits = 410 bytes — tiny.

This is the natural extension of LoRA to ternary networks: train cheap,
deploy cheap, leave the original network intact.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import ste_ternarize


class TernaryLoRA(nn.Module):
    """Wraps a frozen fp32 Linear; adds a trainable rank-r ternary delta.

    Forward:  y = x @ W^T + b   +   (alpha / r) * x @ A_t^T @ B_t^T
    where A_t = ternarize(A), B_t = ternarize(B), both with STE.

    A and B are the underlying float "shadow" parameters; the forward
    pass uses their ternarization, the backward pass uses STE to update
    them with Adam.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 alpha: float = 16.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # Frozen base weight (fp32). Caller will copy values in.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features),
                                        requires_grad=False)
        if bias:
            self.base_bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.base_bias = None

        # Trainable ternary LoRA factors.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # zero-init so init delta is 0
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.T
        if self.base_bias is not None:
            base = base + self.base_bias
        A_t = ste_ternarize(self.lora_A)
        B_t = ste_ternarize(self.lora_B)
        # x @ A_t.T : (..., rank); then @ B_t.T : (..., out)
        delta = (x @ A_t.T) @ B_t.T
        return base + (self.alpha / self.rank) * delta

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8,
                    alpha: float = 16.0) -> "TernaryLoRA":
        """Create a LoRA layer wrapping an existing fp32 Linear."""
        m = cls(linear.in_features, linear.out_features, rank=rank, alpha=alpha,
                bias=linear.bias is not None)
        m.base_weight.data.copy_(linear.weight.data)
        if linear.bias is not None and m.base_bias is not None:
            m.base_bias.data.copy_(linear.bias.data)
        return m

    def trainable_param_count(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()
