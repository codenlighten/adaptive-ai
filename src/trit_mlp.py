"""TritMLP — MLP with ternary weights AND ternary activations.

Pure trit-flow: every value passing between hidden layers is in {-1, 0, +1}.
Input and output projections stay fp32 (BitNet convention) so the model
can ingest and emit real-valued physics quantities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import BitLinear
from .trit_activation import TernaryActivation


class TritMLP(nn.Module):
    """Hidden layers: BitLinear -> TernaryActivation -> BitLinear -> ...

    The hidden representation between every pair of BitLinear layers is a
    vector of trits. This is the asymptote of the ternary idea: not just
    weights, but activations too.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        assert depth >= 3, "depth must be >= 3 to have ternary middle layers"
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), TernaryActivation()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden_dim, hidden_dim), TernaryActivation()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def collect_activation_stats(self, x: torch.Tensor) -> list[dict[str, float]]:
        """Run x through and report the {-1, 0, +1} distribution after each TernaryActivation."""
        stats = []
        h = x
        for m in self.net:
            h_in = h
            h = m(h)
            if isinstance(m, TernaryActivation):
                total = h.numel()
                stats.append({
                    "neg": (h == -1).sum().item() / total,
                    "zero": (h == 0).sum().item() / total,
                    "pos": (h == 1).sum().item() / total,
                })
        return stats
