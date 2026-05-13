"""Quaternary weight quantization: levels {-3, -1, +1, +3} * alpha.

Four-state weights, exactly 2 bits per weight. Even cardinality (no zero
state) — the opposite design from ternary's natural sparsity.

We use levels {-3, -1, +1, +3} rather than {-1.5, -0.5, +0.5, +1.5} so the
levels are integers (compatible with the int-level storage scheme used
elsewhere). The actual effective values are alpha * level, so the absolute
magnitudes are just a rescaling of any other choice.

This is interesting as a contrast to ternary because:
  - Same bit width as quintary (~2 bits effective, 3 bits naive)
  - But no zero state — every weight carries information, no sparsity
  - Matmul costs: each weight needs an add-and-shift (still no multiplier)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_QUAT_LEVELS = (-3, -1, 1, 3)


def quaternize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap each weight to the nearest of {-3, -1, +1, +3} * alpha.

    Strategy: alpha is chosen so the unnormalized weights' empirical
    distribution roughly matches the level grid. We use alpha =
    mean(|w|) / 2 so that the levels { -3, -1, +1, +3 } * alpha span
    approximately ±3 * mean(|w|)/2, comparable to other 2-bit schemes.
    """
    alpha = (w.abs().mean() / 2.0).clamp_min(1e-5)
    levels = torch.tensor(_QUAT_LEVELS, dtype=w.dtype, device=w.device) * alpha
    # Find nearest level for each weight via L1 distance.
    diffs = (w.unsqueeze(-1) - levels).abs()
    idx = diffs.argmin(dim=-1)
    w_q = levels[idx]
    return w_q, alpha


class _STEQuaternize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        w_q, _ = quaternize(w)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_quaternize(w):
    return _STEQuaternize.apply(w)


class QuatLinear(nn.Module):
    """Linear layer with weights snapped to {-3, -1, +1, +3} * alpha."""

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
        w = ste_quaternize(self.weight)
        return F.linear(x, w, self.bias)

    def quat_stats(self):
        with torch.no_grad():
            w_q, alpha = quaternize(self.weight)
            # Convert effective values back to level indices.
            levels = torch.tensor(_QUAT_LEVELS, dtype=w_q.dtype, device=w_q.device) * alpha
            total = w_q.numel()
            counts = {}
            for lvl_idx, lvl in zip(_QUAT_LEVELS, levels):
                counts[f"level_{lvl_idx:+d}"] = (
                    torch.isclose(w_q, lvl).sum().item() / total
                )
            counts["alpha"] = alpha.item()
            return counts


class QuatMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [QuatLinear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def quat_stats(self):
        return [m.quat_stats() for m in self.net if isinstance(m, QuatLinear)]
