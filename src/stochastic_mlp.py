"""MLP using stochastic ternary activations (and ternary weights)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .stochastic_trit import StochasticTernaryActivation
from .ternary import BitLinear


class StochasticTritMLP(nn.Module):
    """Hidden activations sampled as {-1, 0, +1} during training.

    Ensemble inference (`forward_ensemble`) averages N stochastic forward
    passes — produces an analog-valued output and removes the staircase
    artifact from the deterministic TritMLP (Phase 1).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 4):
        super().__init__()
        assert depth >= 3
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), StochasticTernaryActivation()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden_dim, hidden_dim), StochasticTernaryActivation()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def forward_ensemble(self, x: torch.Tensor, n_samples: int = 16) -> torch.Tensor:
        """Average n_samples stochastic forward passes."""
        was_training = self.training
        self.train()  # turn sampling on
        try:
            acc = torch.zeros_like(self.forward(x))
            for _ in range(n_samples):
                acc = acc + self.forward(x)
            return acc / n_samples
        finally:
            self.train(was_training)
