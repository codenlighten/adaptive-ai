"""Cross-modal ternary autoencoder for the 1-D Schrödinger problem.

Input:  V(x) sampled on a fixed grid (vector of length N=64).
Output: ψ_0(x) on the same grid (ground-state wavefunction) and E_0 (scalar).

The network is a ternary encoder/decoder. Latent dim is small (16-32) so
the network must compress the potential to its essential physical
features (well shape, asymmetry, anharmonicity) and decode the
corresponding wavefunction.

This is genuinely cross-modal in the physics sense: the input lives in
function space (potential energy as a function of position) and the
output also lives in function space (the ground-state eigenfunction).
A ternary network correctly capturing this mapping is doing real
functional learning.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ternary import BitLinear


N_GRID = 64
X_MAX = 5.0


def make_grid() -> np.ndarray:
    return np.linspace(-X_MAX, X_MAX, N_GRID)


def random_potential(rng, x_grid):
    """Sample a smooth random V(x) — combination of quadratic, quartic, and gaussian bumps."""
    a = rng.uniform(-2.0, 2.0)
    b = rng.uniform(0.0, 1.0)
    bump_x = rng.uniform(-3.0, 3.0)
    bump_h = rng.uniform(-2.0, 2.0)
    bump_w = rng.uniform(0.3, 1.0)
    V = a * x_grid ** 2 + b * x_grid ** 4 + bump_h * np.exp(-((x_grid - bump_x) ** 2) / (2 * bump_w ** 2))
    return V


def solve_schrodinger(V: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (E0, psi0) on the grid (psi0 normalized)."""
    n = V.shape[0]
    dx = (2 * X_MAX) / (n - 1)
    # Finite-difference Hamiltonian: T = -1/(2 dx^2) tridiag(1, -2, 1)
    kin_diag = np.full(n, 1.0 / (dx * dx))
    kin_off = np.full(n - 1, -0.5 / (dx * dx))
    H = np.diag(kin_diag + V) + np.diag(kin_off, 1) + np.diag(kin_off, -1)
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]
    psi0 = psi0 / np.sqrt((psi0 ** 2).sum() * dx)  # L2-normalize
    if psi0.sum() < 0:
        psi0 = -psi0  # canonical sign
    return float(evals[0]), psi0


def make_dataset(n_samples: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    grid = make_grid()
    Vs, psis, Es = [], [], []
    for _ in range(n_samples):
        V = random_potential(rng, grid)
        E0, psi0 = solve_schrodinger(V)
        Vs.append(V)
        psis.append(psi0)
        Es.append([E0])
    V_t = torch.from_numpy(np.array(Vs)).float()
    psi_t = torch.from_numpy(np.array(psis)).float()
    E_t = torch.from_numpy(np.array(Es)).float()
    return V_t, psi_t, E_t


class TernaryAE(nn.Module):
    """Encoder + decoder, hidden layers ternary."""

    def __init__(self, n_grid: int = N_GRID, hidden: int = 128, latent: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_grid, hidden),
            nn.GELU(),
            BitLinear(hidden, hidden),
            nn.GELU(),
            BitLinear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.decoder_psi = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            BitLinear(hidden, hidden),
            nn.GELU(),
            BitLinear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_grid),
        )
        self.decoder_E = nn.Sequential(
            nn.Linear(latent, hidden // 2),
            nn.GELU(),
            BitLinear(hidden // 2, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, V):
        z = self.encoder(V)
        psi = self.decoder_psi(z)
        E = self.decoder_E(z)
        return psi, E
