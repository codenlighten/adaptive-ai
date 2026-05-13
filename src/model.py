"""Ternary MLP and a full-precision baseline for comparison."""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import BitLinear


class BitMLP(nn.Module):
    """MLP with BitLinear hidden layers — weights are ternary {-1, 0, +1}.

    Following BitNet practice, the input projection and output head are kept
    full-precision; only the hidden transforms are ternarized. This is where
    the parameter count is concentrated, so it's where ternary pays off.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def ternary_stats(self) -> list[dict[str, float]]:
        return [m.ternary_stats() for m in self.net if isinstance(m, BitLinear)]


class FPMLP(nn.Module):
    """Full-precision MLP baseline with the same shape."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
