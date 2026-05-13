"""Train ternary and fp32 Hamiltonian Neural Networks on pendulum data.

The network learns a scalar H(q, p); supervision is on the predicted
vector field dq/dt, dp/dt against the true pendulum's. At inference we
integrate forward with the learned H using a symplectic leapfrog scheme
and measure energy drift — the litmus test for whether the network
captured the conservation law.

Run: venv/bin/python -m src.train_hnn [--plot]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .hnn import (
    HamiltonianNet,
    leapfrog_step,
    make_pendulum_data,
    pendulum_hamiltonian,
    pendulum_vector_field,
)


def train(model, q, p, dq, dp, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = q.shape[0]
    history = {"train": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            dq_pred, dp_pred = model(q[idx], p[idx])
            loss = loss_fn(dq_pred, dq[idx]) + loss_fn(dp_pred, dp[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        history["train"].append(total / n)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:3d}/{epochs}  loss={total/n:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return history


def rollout_with_model(model, q0, p0, n_steps, dt):
    """Integrate forward with the learned H using leapfrog."""
    def vf(q, p):
        return model(q, p)
    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(n_steps - 1):
        q, p = leapfrog_step(q, p, vf, dt)
        qs.append(q.detach())
        ps.append(p.detach())
    return torch.stack(qs, dim=0), torch.stack(ps, dim=0)


def true_rollout(q0, p0, n_steps, dt):
    qs, ps = [q0], [p0]
    q, p = q0, p0
    for _ in range(n_steps - 1):
        q, p = leapfrog_step(q, p, pendulum_vector_field, dt)
        qs.append(q)
        ps.append(p)
    return torch.stack(qs, dim=0), torch.stack(ps, dim=0)


def energy_drift(qs, ps, g_over_l=1.0):
    """Compute |H(t) - H(0)| / |H(0)| at each step."""
    H = pendulum_hamiltonian(qs, ps, g_over_l)
    H0 = H[0:1]
    return (H - H0).abs() / H0.abs().clamp_min(1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-traj", type=int, default=200)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--rollout-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    q, p, dq, dp = make_pendulum_data(args.n_traj, args.steps, args.dt, seed=args.seed)
    print(f"Training set: {q.shape[0]:,} (q, p) samples")

    bit_hnn = HamiltonianNet(dim=1, hidden=args.hidden, depth=args.depth, ternary=True)
    fp_hnn = HamiltonianNet(dim=1, hidden=args.hidden, depth=args.depth, ternary=False)

    print(f"\nBit-HNN params: {sum(x.numel() for x in bit_hnn.parameters()):,}")
    print(f"FP-HNN  params: {sum(x.numel() for x in fp_hnn.parameters()):,}\n")

    h_bit = train(bit_hnn, q, p, dq, dp, args.epochs, args.lr, args.batch_size, "BitHNN")
    print()
    h_fp = train(fp_hnn, q, p, dq, dp, args.epochs, args.lr, args.batch_size, " FPHNN")

    # Long rollout from a held-out initial condition
    torch.manual_seed(args.seed + 100)
    q0 = torch.tensor([[1.5]])
    p0 = torch.tensor([[0.0]])

    print(f"\nRolling out for {args.rollout_steps} steps (dt={args.dt})...")
    qs_true, ps_true = true_rollout(q0, p0, args.rollout_steps, args.dt)
    qs_bit, ps_bit = rollout_with_model(bit_hnn, q0, p0, args.rollout_steps, args.dt)
    qs_fp, ps_fp = rollout_with_model(fp_hnn, q0, p0, args.rollout_steps, args.dt)

    drift_bit = energy_drift(qs_bit, ps_bit).squeeze()
    drift_fp = energy_drift(qs_fp, ps_fp).squeeze()
    drift_true = energy_drift(qs_true, ps_true).squeeze()

    print(f"\n  True  energy drift (leapfrog on exact H):  "
          f"max={drift_true.max():.2e}  final={drift_true[-1]:.2e}")
    print(f"  BitHNN energy drift (leapfrog on learned H): "
          f"max={drift_bit.max():.2e}  final={drift_bit[-1]:.2e}")
    print(f"  FPHNN  energy drift (leapfrog on learned H): "
          f"max={drift_fp.max():.2e}  final={drift_fp[-1]:.2e}")

    # MSE of vector field on test grid
    qg = torch.linspace(-3, 3, 50).unsqueeze(-1)
    pg = torch.linspace(-3, 3, 50).unsqueeze(-1)
    Q, P = torch.meshgrid(qg.squeeze(), pg.squeeze(), indexing="ij")
    q_flat = Q.reshape(-1, 1)
    p_flat = P.reshape(-1, 1)
    dq_t, dp_t = pendulum_vector_field(q_flat, p_flat)
    dq_b, dp_b = bit_hnn(q_flat, p_flat)
    dq_f, dp_f = fp_hnn(q_flat, p_flat)
    bit_vf_mse = ((dq_b - dq_t) ** 2 + (dp_b - dp_t) ** 2).mean().item()
    fp_vf_mse = ((dq_f - dq_t) ** 2 + (dp_f - dp_t) ** 2).mean().item()
    print(f"\n  BitHNN vector field MSE on grid: {bit_vf_mse:.5f}")
    print(f"  FPHNN  vector field MSE on grid: {fp_vf_mse:.5f}")

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

        ax[0].plot(h_bit["train"], label="BitHNN")
        ax[0].plot(h_fp["train"], label="FPHNN")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("train loss")
        ax[0].set_title("Vector field training loss")
        ax[0].legend()

        t = np.arange(args.rollout_steps) * args.dt
        ax[1].plot(t, qs_true.squeeze().numpy(), "k-", label="true", linewidth=2)
        ax[1].plot(t, qs_bit.squeeze().numpy(), "b--", label="BitHNN")
        ax[1].plot(t, qs_fp.squeeze().numpy(), "r:", label="FPHNN")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("q(t)")
        ax[1].set_title("Pendulum rollout (long horizon)")
        ax[1].legend()

        ax[2].semilogy(t, drift_bit.numpy(), "b-", label="BitHNN")
        ax[2].semilogy(t, drift_fp.numpy(), "r-", label="FPHNN")
        ax[2].semilogy(t, drift_true.numpy(), "k-", label="leapfrog (true H)")
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("|H(t) - H(0)| / |H(0)|")
        ax[2].set_title("Energy drift (lower = more conservative)")
        ax[2].legend()

        plt.tight_layout()
        plt.savefig("hnn_results.png", dpi=120)
        print("\nSaved plot to hnn_results.png")


if __name__ == "__main__":
    main()
