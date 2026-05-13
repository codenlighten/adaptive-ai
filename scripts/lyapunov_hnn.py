"""Empirical Lyapunov stability analysis for the ternary Hamiltonian NN.

For a symplectic integrator on the true Hamiltonian H, energy is
conserved to O(dt^p) where p is the integrator order. Leapfrog is 2nd
order so we expect drift ~ dt^2 over a fixed time T.

For an HNN that learns H, we additionally have a learning error
contribution that doesn't depend on dt. The total drift is roughly
    |dE| ~ C_int * dt^2 + C_learn

We sweep dt and fit the scaling. If the ternary HNN preserves the
symplectic property (its learned H is a smooth function with a
well-defined gradient field), we should see clean dt^2 scaling at
small dt and a learning-error floor at the smallest dt.

Run: venv/bin/python -m scripts.lyapunov_hnn
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.hnn import (
    HamiltonianNet,
    leapfrog_step,
    make_pendulum_data,
    pendulum_hamiltonian,
    pendulum_vector_field,
)


def train(model, q, p, dq, dp, epochs, lr, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = q.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            dq_p, dp_p = model(q[idx], p[idx])
            loss = loss_fn(dq_p, dq[idx]) + loss_fn(dp_p, dp[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()


def rollout_drift(model, q0, p0, dt, T):
    """Run leapfrog for total time T with step dt and return max relative drift."""
    n_steps = int(round(T / dt))
    def vf(q, p):
        return model(q, p)
    qs, ps = [q0], [p0]
    q, p = q0, p0
    with torch.no_grad() if False else torch.enable_grad():
        for _ in range(n_steps - 1):
            q, p = leapfrog_step(q, p, vf, dt)
            qs.append(q.detach())
            ps.append(p.detach())
    qs = torch.stack(qs)
    ps = torch.stack(ps)
    H = pendulum_hamiltonian(qs, ps).squeeze()
    H0 = H[0]
    return (H - H0).abs().max().item() / abs(H0.item() + 1e-9)


def rollout_drift_truth(q0, p0, dt, T):
    n_steps = int(round(T / dt))
    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(n_steps - 1):
        q, p = leapfrog_step(q, p, pendulum_vector_field, dt)
        qs.append(q)
        ps.append(p)
    qs = torch.stack(qs); ps = torch.stack(ps)
    H = pendulum_hamiltonian(qs, ps).squeeze()
    return (H - H[0]).abs().max().item() / abs(H[0].item() + 1e-9)


def main():
    torch.manual_seed(0)
    print("Training fp and ternary HNNs on pendulum...")
    q, p, dq, dp = make_pendulum_data(n_traj=200, steps_per_traj=64, dt=0.05)
    print(f"  train data: {q.shape[0]:,} samples")

    fp = HamiltonianNet(dim=1, hidden=64, depth=4, ternary=False)
    bit = HamiltonianNet(dim=1, hidden=64, depth=4, ternary=True)

    t0 = time.time()
    train(fp, q, p, dq, dp, epochs=300, lr=1e-3, batch_size=256)
    train(bit, q, p, dq, dp, epochs=300, lr=1e-3, batch_size=256)
    print(f"  trained in {time.time()-t0:.1f}s\n")

    # Energy drift sweep
    q0 = torch.tensor([[1.5]])
    p0 = torch.tensor([[0.0]])
    T = 20.0  # 20-second rollout

    dts = [0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"Rolling out for T={T}s at various step sizes; max relative energy drift:\n")
    print(f"{'dt':>8}  {'truth':>12}  {'FP HNN':>12}  {'Bit HNN':>12}")
    drifts = {"dt": [], "truth": [], "fp": [], "bit": []}
    for dt in dts:
        d_t = rollout_drift_truth(q0, p0, dt, T)
        d_f = rollout_drift(fp, q0, p0, dt, T)
        d_b = rollout_drift(bit, q0, p0, dt, T)
        drifts["dt"].append(dt); drifts["truth"].append(d_t)
        drifts["fp"].append(d_f); drifts["bit"].append(d_b)
        print(f"{dt:>8.3f}  {d_t:>12.3e}  {d_f:>12.3e}  {d_b:>12.3e}")

    # Fit power law for each (log-log slope between the two smallest dts)
    def slope(xs, ys):
        x = np.log(np.array(xs))
        y = np.log(np.array(ys))
        p = np.polyfit(x, y, 1)
        return p[0]

    print("\nPower-law scaling (log-log slope across all dts):")
    print(f"  truth (leapfrog on exact H):  {slope(drifts['dt'], drifts['truth']):.2f}  "
          f"(expected ~2.00 for 2nd-order symplectic)")
    print(f"  FP HNN:                       {slope(drifts['dt'], drifts['fp']):.2f}")
    print(f"  Bit HNN:                      {slope(drifts['dt'], drifts['bit']):.2f}")

    print("\nInterpretation:")
    print("  - If the learned-H drift floor dominates, the slope is ~0 (drift ~ constant).")
    print("  - If symplectic-order error dominates, slope ~ 2.")
    print("  - Ternary HNNs preserving the symplectic-integrator scaling means")
    print("    their learned H is smooth enough to inherit the integrator's order.")


if __name__ == "__main__":
    main()
