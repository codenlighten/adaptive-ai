"""Tiny 1-D diffusion model that learns to sample from a physics distribution.

Target: a double-well Boltzmann distribution
    p(x) ∝ exp(-V(x) / T),   V(x) = (x^2 - 1)^2

This has two peaks at x = ±1, separated by a barrier at x = 0. It's the
classical bistable system — molecular bonds, magnetization, neural
attractors all look like this. Capturing both modes (not collapsing onto
one) is the hard part of generative models.

We implement DDPM denoising at the level of a small MLP (ternary or fp).
At inference we sample from the learned reverse process and report:
  - both modes coverage (do we get both peaks?)
  - approximate KL divergence to the analytic target
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .ternary import BitLinear


def double_well_logp(x: torch.Tensor, T: float = 0.4) -> torch.Tensor:
    return -((x ** 2 - 1.0) ** 2) / T


def sample_double_well(n: int, T: float = 0.4, seed: int = 0) -> torch.Tensor:
    """Rejection sample from the double-well distribution.

    Bound the unnormalized density by exp(0)=1 (it's max at x=±1) and
    propose from a wide uniform.
    """
    g = torch.Generator().manual_seed(seed)
    samples = []
    while len(samples) < n:
        prop = (torch.rand(2 * n, generator=g) * 6.0 - 3.0)  # uniform on [-3, 3]
        u = torch.rand(2 * n, generator=g)
        accept = u < torch.exp(double_well_logp(prop, T))
        samples.extend(prop[accept].tolist())
    return torch.tensor(samples[:n], dtype=torch.float32).unsqueeze(1)


class _BitDenoiser(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class _FPDenoiser(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class DDPM1D:
    """Minimal DDPM with linear beta schedule for 1-D data."""

    def __init__(self, n_steps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Add noise: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps."""
        a_bar = self.alpha_bar[t].unsqueeze(-1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n: int, device="cpu") -> torch.Tensor:
        x = torch.randn(n, 1, device=device)
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((n, 1), t / self.n_steps, device=device)
            eps_pred = model(x, t_tensor)
            a = self.alphas[t]
            ab = self.alpha_bar[t]
            mean = (1.0 / a.sqrt()) * (x - (1.0 - a) / (1.0 - ab).sqrt() * eps_pred)
            if t > 0:
                noise = torch.randn_like(x)
                sigma = self.betas[t].sqrt()
                x = mean + sigma * noise
            else:
                x = mean
        return x


def make_ternary_denoiser(hidden=64, depth=4):
    return _BitDenoiser(hidden=hidden, depth=depth)


def make_fp_denoiser(hidden=64, depth=4):
    return _FPDenoiser(hidden=hidden, depth=depth)
