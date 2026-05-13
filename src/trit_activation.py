"""Ternary activation function — outputs in {-1, 0, +1}.

Forward:  sign(x) with a deadzone — values inside [-tau, +tau] snap to 0.
Backward: straight-through estimator clipped to |x|<=1 (Hardtanh STE).

The deadzone threshold tau is a per-tensor scale that adapts to the
input distribution (defaults to 2/3 of the standard deviation). This
keeps roughly 25%-50% of activations at zero — exactly the natural
ternary distribution we want, and it tracks the data scale across
layers without requiring a separate LayerNorm before the activation.

When combined with BitLinear, every intermediate value in the network
flows as a trit. The only fp computations are LayerNorm (inside
BitLinear), the per-layer alpha rescale, and the small boundary
projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _STETritAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # Sign with deadzone — output in {-1, 0, +1}
        y = torch.where(
            x.abs() < tau,
            torch.zeros_like(x),
            torch.sign(x),
        )
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        # Hardtanh-style STE: gradient passes through where |x| <= 1, zero outside.
        # This is the standard trick for binary/ternary nets — it prevents huge
        # gradients from blowing up the underlying float pre-activations.
        mask = (x.abs() <= 1.0).to(grad_output.dtype)
        return grad_output * mask, None


class TernaryActivation(nn.Module):
    """Forward: outputs trits {-1, 0, +1}. Backward: clipped STE.

    The deadzone threshold is tau = scale * std(x). With scale=2/3 we get
    a roughly even three-way split for ~Gaussian inputs (this is the
    optimal threshold for a balanced ternary quantizer under L2 loss
    for Gaussian inputs — analogous to the BitNet b1.58 weight rule).
    """

    def __init__(self, scale: float = 2.0 / 3.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-tensor adaptive threshold. detach() so tau doesn't take gradient.
        tau = (self.scale * x.detach().std()).clamp_min(1e-6)
        return _STETritAct.apply(x, tau)

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor) -> dict[str, float]:
        y = self.forward(x)
        total = y.numel()
        return {
            "neg": (y == -1).sum().item() / total,
            "zero": (y == 0).sum().item() / total,
            "pos": (y == 1).sum().item() / total,
        }
