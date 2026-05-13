"""Hybrid Quaternary + Ternary MLP.

Heuristic: layers with higher pre-quantization weight std get quaternary
(more levels needed to represent the wider distribution); the remaining
layers get ternary (cheaper, naturally sparse). The model is constructed
in two passes: first a temporary fp32 forward pass with random init to
measure per-layer std, then we assign each hidden layer to its tier.

This is the simplest "mixed-precision" strategy: don't try to learn the
assignment, use a static heuristic on initialization statistics. More
sophisticated methods would learn the assignment during training, but
the point here is to show the principle.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quaternary import QuatLinear
from .ternary import BitLinear


class HybridQTMLP(nn.Module):
    """Hidden layers are a mix of QuatLinear and BitLinear, chosen by std."""

    def __init__(self, in_dim, hidden_dim, out_dim, depth=5,
                 quaternary_fraction: float = 0.4):
        super().__init__()
        n_hidden = depth - 2  # excluding input/output projections
        n_quat = max(1, int(round(n_hidden * quaternary_fraction)))

        # Build the layers with all the same fp32 weights, then look at
        # initialization-time std to pick assignments.
        candidate_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
        stds = [float(l.weight.detach().std()) for l in candidate_layers]
        # Indices sorted by descending std — those get quaternary.
        order = sorted(range(n_hidden), key=lambda i: -stds[i])
        quat_idx = set(order[:n_quat])

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for k in range(n_hidden):
            if k in quat_idx:
                ql = QuatLinear(hidden_dim, hidden_dim)
                ql.weight.data.copy_(candidate_layers[k].weight.data)
                ql.bias.data.copy_(candidate_layers[k].bias.data)
                layers.append(ql)
            else:
                bl = BitLinear(hidden_dim, hidden_dim)
                bl.weight.data.copy_(candidate_layers[k].weight.data)
                bl.bias.data.copy_(candidate_layers[k].bias.data)
                layers.append(bl)
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._assignment = ["Q" if k in quat_idx else "T" for k in range(n_hidden)]

    def forward(self, x):
        return self.net(x)

    def assignment(self) -> list[str]:
        return self._assignment
