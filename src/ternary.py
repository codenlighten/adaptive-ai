"""BitNet b1.58-style ternary quantization.

Weights are quantized to {-1, 0, +1} with a per-tensor scale alpha = mean(|W|).
A straight-through estimator passes gradients through the quantization step
so the underlying float weights can still be trained with normal SGD/Adam.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternarize(w: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to {-1, 0, +1} using the BitNet b1.58 rule.

    Returns (w_ternary, alpha) where alpha is the per-tensor scale.
    Threshold = 0.75 * mean(|W|) — values below this snap to 0, others to sign(w).
    """
    alpha = w.abs().mean().clamp_min(eps)
    threshold = 0.75 * alpha
    w_q = torch.where(
        w.abs() < threshold,
        torch.zeros_like(w),
        torch.sign(w),
    )
    return w_q, alpha


class _STETernarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        w_q, alpha = ternarize(w)
        return w_q * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def ste_ternarize(w: torch.Tensor) -> torch.Tensor:
    """Forward: ternary weights * alpha. Backward: identity (STE)."""
    return _STETernarize.apply(w)


class BitLinear(nn.Module):
    """Linear layer with ternary {-1, 0, +1} weights (BitNet b1.58 style).

    Activations stay in float; only weights are quantized. A LayerNorm on the
    input stabilizes training (this matches the BitNet recipe).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        w = ste_ternarize(self.weight)
        return F.linear(x, w, self.bias)

    def ternary_stats(self) -> dict[str, float]:
        """Diagnostics: fraction of weights at -1, 0, +1 after quantization."""
        with torch.no_grad():
            w_q, alpha = ternarize(self.weight)
            total = w_q.numel()
            return {
                "neg": (w_q == -1).sum().item() / total,
                "zero": (w_q == 0).sum().item() / total,
                "pos": (w_q == 1).sum().item() / total,
                "alpha": alpha.item(),
            }
