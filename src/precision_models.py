"""MLP variants at different weight precisions for the comparison study.

All four share the same shape; only the hidden-layer flavor changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quintary import BinaryLinear, QuintLinear
from .ternary import BitLinear


def _build(in_dim, hidden_dim, out_dim, depth, hidden_cls):
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(depth - 2):
        layers += [hidden_cls(hidden_dim, hidden_dim), nn.GELU()]
    layers += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*layers)


class BinaryMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3):
        super().__init__()
        self.net = _build(in_dim, hidden_dim, out_dim, depth, BinaryLinear)

    def forward(self, x):
        return self.net(x)


class QuintMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3):
        super().__init__()
        self.net = _build(in_dim, hidden_dim, out_dim, depth, QuintLinear)

    def forward(self, x):
        return self.net(x)

    def quint_stats(self):
        return [m.quint_stats() for m in self.net if isinstance(m, QuintLinear)]
