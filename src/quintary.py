"""Quintary {-2, -1, 0, +1, +2} weight quantization — ~2.32 bits/weight.

A natural extension of ternary. The quantization rule:
    alpha = mean(|W|)
    levels = {-2, -1, 0, +1, +2} * alpha
    each weight snaps to its nearest level

Matmul is still multiply-light: each weight is sign * shift-1 (one
add or sub for ±1, two for ±2). On hardware this is one shifter +
one signed adder per accumulation — still no full multiplier.

Quintary uses ceil(log2(5)) = 3 bits naively, or log2(5) = 2.32 bits with
arithmetic coding. We store 3 quints per byte (5^3 = 125 < 256).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quintize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to {-2, -1, 0, +1, +2} * alpha.

    Threshold rule (analogous to BitNet b1.58 for ternary):
      alpha = mean(|W|)
      |w| < 0.5 * alpha          -> 0
      0.5 * alpha <= |w| < 1.5*alpha -> ±1 * sign(w)
      |w| >= 1.5 * alpha         -> ±2 * sign(w)
    """
    alpha = w.abs().mean().clamp_min(1e-5)
    abs_w = w.abs() / alpha
    sign = torch.sign(w)
    mag = torch.where(
        abs_w < 0.5, torch.zeros_like(w),
        torch.where(abs_w < 1.5, torch.ones_like(w), 2.0 * torch.ones_like(w)),
    )
    return sign * mag, alpha


class _STEQuintize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w_q, alpha = quintize(w)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_quintize(w):
    return _STEQuintize.apply(w)


class QuintLinear(nn.Module):
    """Linear layer with weights in {-2, -1, 0, +1, +2} (per-tensor scale)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = self.norm(x)
        w = ste_quintize(self.weight)
        return F.linear(x, w, self.bias)

    def quint_stats(self):
        with torch.no_grad():
            w_q, alpha = quintize(self.weight)
            total = w_q.numel()
            return {
                "neg2": (w_q == -2).sum().item() / total,
                "neg1": (w_q == -1).sum().item() / total,
                "zero": (w_q == 0).sum().item() / total,
                "pos1": (w_q == 1).sum().item() / total,
                "pos2": (w_q == 2).sum().item() / total,
                "alpha": alpha.item(),
            }


# ---------------------------------------------------------------------------
# Binary {-1, +1} for an even simpler comparison point.
# ---------------------------------------------------------------------------

def binarize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = w.abs().mean().clamp_min(1e-5)
    return torch.sign(w), alpha


class _STEBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w_q, alpha = binarize(w)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_binarize(w):
    return _STEBinarize.apply(w)


class BinaryLinear(nn.Module):
    """Linear layer with weights in {-1, +1} (no zeros — pure binary)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = self.norm(x)
        w = ste_binarize(self.weight)
        return F.linear(x, w, self.bias)
