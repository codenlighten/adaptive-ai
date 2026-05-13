"""3-mass linear spring chain.

H(q1, q2, q3, p1, p2, p3) = sum_i p_i^2 / 2 + sum_i (q_i - q_{i+1})^2 / 2

with fixed boundary conditions (q_0 = q_4 = 0), so:
    H = p1^2/2 + p2^2/2 + p3^2/2
      + q1^2/2  + (q1 - q2)^2/2  + (q2 - q3)^2/2  + q3^2/2

Expand the cross terms:
    = 0.5 q1^2 + 0.5 q2^2 + 0.5 q3^2
      + 0.5 q1^2 - q1*q2 + 0.5 q2^2
      + 0.5 q2^2 - q2*q3 + 0.5 q3^2
      + 0.5 q1^2 + 0.5 q3^2  (boundary already absorbed above)
... actually the cleanest expanded form is:
    H = 0.5*(p1^2 + p2^2 + p3^2)
      + q1^2 - q1*q2 + q2^2 - q2*q3 + q3^2

Verifiable: dq_i/dt = p_i, dp_1/dt = -2 q1 + q2, dp_2/dt = q1 - 2 q2 + q3,
dp_3/dt = q2 - 2 q3 (discrete Laplacian — the classical "phonons" equation).
"""

from __future__ import annotations

import torch


def chain_hamiltonian(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """q, p have shape (..., 3). Returns scalar H."""
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    kinetic = 0.5 * (p ** 2).sum(dim=-1)
    potential = q1 ** 2 - q1 * q2 + q2 ** 2 - q2 * q3 + q3 ** 2
    return kinetic + potential


def chain_vector_field(q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (dq/dt, dp/dt). dq = p; dp = -dV/dq via the chain Laplacian."""
    dq = p
    q1, q2, q3 = q[..., 0:1], q[..., 1:2], q[..., 2:3]
    dp = torch.cat([
        -(2 * q1 - q2),
        -(2 * q2 - q1 - q3),
        -(2 * q3 - q2),
    ], dim=-1)
    return dq, dp


def leapfrog_chain(q: torch.Tensor, p: torch.Tensor,
                   vf_fn, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    _, dp1 = vf_fn(q, p)
    p_half = p + 0.5 * dt * dp1
    dq, _ = vf_fn(q, p_half)
    q_new = q + dt * dq
    _, dp2 = vf_fn(q_new, p_half)
    p_new = p_half + 0.5 * dt * dp2
    return q_new, p_new


def make_chain_data(n_traj: int = 200, steps_per_traj: int = 64,
                    dt: float = 0.05, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q0 = (torch.rand(n_traj, 3, generator=g) - 0.5) * 2.0 * 1.0  # ~[-1, 1]
    p0 = (torch.rand(n_traj, 3, generator=g) - 0.5) * 2.0 * 0.5

    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(steps_per_traj - 1):
        q, p = leapfrog_chain(q, p, chain_vector_field, dt)
        qs.append(q)
        ps.append(p)

    q_all = torch.stack(qs, dim=0).reshape(-1, 3)
    p_all = torch.stack(ps, dim=0).reshape(-1, 3)
    dq, dp = chain_vector_field(q_all, p_all)
    return q_all, p_all, dq, dp
