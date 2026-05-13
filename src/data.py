"""Damped harmonic oscillator dataset.

Learn the map (t, omega, zeta) -> x(t) for an underdamped oscillator
starting at x(0)=1, v(0)=0:

    x(t) = exp(-zeta*omega*t) * (cos(wd*t) + (zeta*omega/wd)*sin(wd*t))

with wd = omega*sqrt(1 - zeta^2). This is a smooth, low-dim physics
regression problem — a good stress test for a ternary-weight MLP.
"""

from __future__ import annotations

import math

import torch


def damped_oscillator(t: torch.Tensor, omega: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
    wd = omega * torch.sqrt(torch.clamp(1.0 - zeta**2, min=1e-6))
    envelope = torch.exp(-zeta * omega * t)
    return envelope * (torch.cos(wd * t) + (zeta * omega / wd) * torch.sin(wd * t))


def make_dataset(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 10.0           # t in [0, 10]
    omega = 0.5 + torch.rand(n, generator=g) * 2.5  # omega in [0.5, 3.0]
    zeta = 0.05 + torch.rand(n, generator=g) * 0.5  # zeta in [0.05, 0.55] (underdamped)
    x = damped_oscillator(t, omega, zeta)
    X = torch.stack([t, omega, zeta], dim=1)
    y = x.unsqueeze(1)
    return X, y


def normalize(X: torch.Tensor, stats: dict | None = None) -> tuple[torch.Tensor, dict]:
    if stats is None:
        stats = {"mean": X.mean(0), "std": X.std(0).clamp_min(1e-6)}
    return (X - stats["mean"]) / stats["std"], stats
