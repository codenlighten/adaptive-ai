"""Conditional 1-D DDPM denoiser, ternary and fp variants.

Same target family as diffusion.py:  p(x | T) ∝ exp(-(x²-1)² / T)
but now T is an input — one model covers a continuous family of
double-well distributions. At low T the wells are sharp; at high T
they merge into a single broad mode.

The denoiser sees [x, time_step, T] as input. Train: sample T uniformly,
then sample x ~ p(·|T), then standard DDPM denoising loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import BitLinear


def double_well_logp(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return -((x ** 2 - 1.0) ** 2) / T


def sample_double_well_conditional(n: int, T: float, seed: int = 0) -> torch.Tensor:
    """Rejection sample from p(x|T) for one T value."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    while len(samples) < n:
        prop = (torch.rand(4 * n, generator=g) * 6.0 - 3.0)
        u = torch.rand(4 * n, generator=g)
        accept = u < torch.exp(double_well_logp(prop, torch.tensor(T)))
        samples.extend(prop[accept].tolist())
    return torch.tensor(samples[:n], dtype=torch.float32).unsqueeze(1)


def make_dataset(n_per_T: int, T_grid: list[float], seed: int = 0):
    """Build a labeled dataset across multiple temperatures.

    Returns (X, T) where X has shape (N, 1) and T has shape (N, 1).
    """
    Xs, Ts = [], []
    for i, T in enumerate(T_grid):
        Xs.append(sample_double_well_conditional(n_per_T, T, seed=seed + i))
        Ts.append(torch.full((n_per_T, 1), T))
    return torch.cat(Xs, dim=0), torch.cat(Ts, dim=0)


class _BitCondDenoiser(nn.Module):
    """Conditional denoiser: input [x, t_step, T_log] -> noise estimate."""

    def __init__(self, hidden=96, depth=4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, T):
        T_log = torch.log(T)  # log-scale conditioning helps when T spans orders of magnitude
        return self.net(torch.cat([x, t, T_log], dim=-1))


class _FPCondDenoiser(nn.Module):
    def __init__(self, hidden=96, depth=4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, T):
        T_log = torch.log(T)
        return self.net(torch.cat([x, t, T_log], dim=-1))


def make_conditional_ternary(hidden=96, depth=4):
    return _BitCondDenoiser(hidden=hidden, depth=depth)


def make_conditional_fp(hidden=96, depth=4):
    return _FPCondDenoiser(hidden=hidden, depth=depth)
