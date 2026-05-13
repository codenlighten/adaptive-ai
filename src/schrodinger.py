"""1D time-independent Schrödinger eigenvalue problem.

Hamiltonian (ℏ = m = 1):
    H = -1/2 d^2/dx^2 + V(x)
    V(x) = a*x^2 + b*x^4    (double-well when a<0, b>0)

We discretize on a uniform grid using a 3-point stencil for the second
derivative, then diagonalize with numpy.linalg.eigh to get the ground-state
energy E0(a, b). The network's job is to predict E0 from (a, b).

This is genuinely nonlinear and non-smooth in regions where the well
shape changes character (single well -> shallow double -> deep double),
so it's a real test of whether ternary weights can capture interesting
physics — not just a smooth decaying sinusoid.
"""

from __future__ import annotations

import numpy as np
import torch


def build_hamiltonian(a: float, b: float, n: int = 256, x_max: float = 6.0) -> np.ndarray:
    """Construct the discrete Hamiltonian matrix on [-x_max, x_max] with n points."""
    x = np.linspace(-x_max, x_max, n)
    dx = x[1] - x[0]
    # second-derivative stencil: T = -1/(2 dx^2) * tridiag(1, -2, 1)
    kin_diag = np.full(n, 1.0 / (dx * dx))
    kin_off = np.full(n - 1, -0.5 / (dx * dx))
    V = a * x**2 + b * x**4
    H = np.diag(kin_diag + V) + np.diag(kin_off, 1) + np.diag(kin_off, -1)
    return H


def ground_state_energy(a: float, b: float, n: int = 256, x_max: float = 6.0) -> float:
    """Smallest eigenvalue of the discretized Hamiltonian."""
    H = build_hamiltonian(a, b, n=n, x_max=x_max)
    # eigh is for symmetric matrices — much faster and more stable than eig.
    # We only need the smallest eigenvalue, but eigh returns all sorted; cheap at n=256.
    evals = np.linalg.eigvalsh(H)
    return float(evals[0])


def make_dataset(n_samples: int, seed: int = 0,
                 a_range: tuple[float, float] = (-4.0, 2.0),
                 b_range: tuple[float, float] = (0.05, 1.0),
                 grid_n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (a, b) and compute E0 for each. Returns (X, y) as torch tensors."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(*a_range, size=n_samples)
    b = rng.uniform(*b_range, size=n_samples)
    E0 = np.array([ground_state_energy(ai, bi, n=grid_n) for ai, bi in zip(a, b)])
    X = torch.from_numpy(np.stack([a, b], axis=1)).float()
    y = torch.from_numpy(E0).float().unsqueeze(1)
    return X, y


def normalize(X: torch.Tensor, stats: dict | None = None) -> tuple[torch.Tensor, dict]:
    if stats is None:
        stats = {"mean": X.mean(0), "std": X.std(0).clamp_min(1e-6)}
    return (X - stats["mean"]) / stats["std"], stats
