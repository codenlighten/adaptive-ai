"""Learned mixed-precision assignment via Gumbel-softmax.

Each hidden layer has a learnable categorical distribution over
{binary, ternary, quaternary} precisions. At forward time we sample
(via Gumbel-softmax) which precision to use; at backward we get
gradients to all three branches plus the assignment logits. After
training, we lock the assignment to its argmax and evaluate.

Inspired by HAQ / DNAS-style differentiable architecture search, but
narrowed to weight precision (not full architecture).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quaternary import ste_quaternize
from .quantize_levels import binarize_levels
from .ternary import ste_ternarize


_LEVEL_NAMES = ["Binary", "Ternary", "Quaternary"]


def _ste_binarize(w: torch.Tensor) -> torch.Tensor:
    """STE for binary {-1, +1}: forward sign(w) * alpha, backward identity."""
    class _STE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            alpha = x.abs().mean().clamp_min(1e-5)
            return torch.sign(x) * alpha

        @staticmethod
        def backward(ctx, grad):
            return grad
    return _STE.apply(w)


class MixedPrecisionLinear(nn.Module):
    """Linear with a learnable choice of weight precision.

    At forward, we compute three quantized versions of the weight
    (binary, ternary, quaternary) and combine them by Gumbel-softmax
    weights. Tau anneals from 5.0 (smooth blend) toward 0.5 (near-hard
    discrete) during training.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        self.assignment_logits = nn.Parameter(torch.zeros(3))   # binary / ternary / quaternary
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.tau = 5.0

    def forward(self, x):
        x = self.norm(x)
        # Compute three quantizations.
        w_bin  = _ste_binarize(self.weight)
        w_ter  = ste_ternarize(self.weight)
        w_quat = ste_quaternize(self.weight)
        if self.training:
            # Gumbel-softmax over precisions.
            choice = F.gumbel_softmax(self.assignment_logits, tau=self.tau, hard=False)
            w = choice[0] * w_bin + choice[1] * w_ter + choice[2] * w_quat
        else:
            # Eval: hard argmax.
            idx = self.assignment_logits.argmax().item()
            w = [w_bin, w_ter, w_quat][idx]
        return F.linear(x, w, self.bias)

    def selected_precision(self) -> str:
        return _LEVEL_NAMES[int(self.assignment_logits.argmax().item())]

    def assignment_probs(self) -> dict[str, float]:
        with torch.no_grad():
            p = F.softmax(self.assignment_logits, dim=0).tolist()
        return dict(zip(_LEVEL_NAMES, p))


class MixedPrecisionMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [MixedPrecisionLinear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def set_tau(self, tau: float):
        for m in self.net:
            if isinstance(m, MixedPrecisionLinear):
                m.tau = tau

    def assignment(self) -> list[str]:
        return [m.selected_precision() for m in self.net if isinstance(m, MixedPrecisionLinear)]

    def assignment_probs(self) -> list[dict[str, float]]:
        return [m.assignment_probs() for m in self.net if isinstance(m, MixedPrecisionLinear)]
