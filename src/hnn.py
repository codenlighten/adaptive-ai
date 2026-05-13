"""Hamiltonian Neural Network — learn a scalar H(q, p) such that
    dq/dt =  dH/dp
    dp/dt = -dH/dq

Then the network *automatically* respects energy conservation (because
its predicted vector field is symplectic by construction). This is a
much stronger inductive bias than learning the dynamics directly.

Reference: Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks"
(NeurIPS 2019). We add a ternary variant: the same H-network, but its
hidden layers use BitLinear weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import BitLinear


class HNNCore(nn.Module):
    """Maps (q, p) -> scalar H. Subclasses pick the layer flavor."""

    def __init__(self, dim: int = 1, hidden: int = 64, depth: int = 3, ternary: bool = False):
        super().__init__()
        in_dim = 2 * dim  # (q, p) concatenated
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            if ternary:
                layers += [BitLinear(hidden, hidden), nn.Tanh()]
            else:
                layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, p], dim=-1)).squeeze(-1)


class HamiltonianNet(nn.Module):
    """Wraps a scalar H-network and exposes the predicted vector field."""

    def __init__(self, dim: int = 1, hidden: int = 64, depth: int = 3, ternary: bool = False):
        super().__init__()
        self.dim = dim
        self.H = HNNCore(dim=dim, hidden=hidden, depth=depth, ternary=ternary)

    def vector_field(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (dq/dt, dp/dt) using autograd on the learned H."""
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)
        H = self.H(q, p).sum()
        # We need partials wrt each input independently.
        dHdq, dHdp = torch.autograd.grad(H, (q, p), create_graph=True)
        return dHdp, -dHdq

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vector_field(q, p)


def pendulum_hamiltonian(q: torch.Tensor, p: torch.Tensor, g_over_l: float = 1.0) -> torch.Tensor:
    """True H for a unit-mass pendulum (l=1 implicit): H = p^2/2 + g/l * (1 - cos q)."""
    return 0.5 * p**2 + g_over_l * (1.0 - torch.cos(q))


def pendulum_vector_field(q: torch.Tensor, p: torch.Tensor,
                          g_over_l: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """True dq/dt = p, dp/dt = -g/l * sin(q)."""
    return p, -g_over_l * torch.sin(q)


def leapfrog_step(q: torch.Tensor, p: torch.Tensor,
                  vf_fn, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One leapfrog (symplectic) integration step using a (q,p)->(dq,dp) field.

    Half-kick, drift, half-kick. This is the standard symplectic 2nd-order
    integrator — exactly conserves a "shadow Hamiltonian" close to the true one.
    """
    _, dp1 = vf_fn(q, p)
    p_half = p + 0.5 * dt * dp1
    dq, _ = vf_fn(q, p_half)
    q_new = q + dt * dq
    _, dp2 = vf_fn(q_new, p_half)
    p_new = p_half + 0.5 * dt * dp2
    return q_new, p_new


def make_pendulum_data(n_traj: int = 200, steps_per_traj: int = 64,
                       dt: float = 0.1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor,
                                                                  torch.Tensor, torch.Tensor]:
    """Roll out the true pendulum from random initial conditions.

    Returns flat arrays of (q, p) states and their true (dq, dp) time derivatives,
    sampled along trajectories. Shapes: each is (n_traj * steps_per_traj, 1).
    """
    g = torch.Generator().manual_seed(seed)
    q0 = (torch.rand(n_traj, 1, generator=g) - 0.5) * 2.0 * 2.5  # roughly [-2.5, 2.5]
    p0 = (torch.rand(n_traj, 1, generator=g) - 0.5) * 2.0 * 2.0  # roughly [-2.0, 2.0]

    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(steps_per_traj - 1):
        q, p = leapfrog_step(q, p, pendulum_vector_field, dt)
        qs.append(q)
        ps.append(p)

    q_all = torch.stack(qs, dim=0).reshape(-1, 1)  # (n_traj * steps, 1)
    p_all = torch.stack(ps, dim=0).reshape(-1, 1)
    dq, dp = pendulum_vector_field(q_all, p_all)
    return q_all, p_all, dq, dp
