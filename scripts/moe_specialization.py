"""Analyze which experts in a trained ternary MoE specialize on what.

We train a ternary MoE on damped oscillator data (input dims: t, omega, zeta),
then sweep through input space and record which expert each region picks.
The resulting heatmaps show the *learned partition* of input space —
a glimpse into how routing carves the problem.

Hypothesis: the router will cluster inputs by physical regime (low/high
omega, low/high damping) without being told to.

Run: venv/bin/python -m scripts.moe_specialization
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.moe import TernaryMoE


def train(model, X, y, X_val, y_val, epochs, lr, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return loss_fn(model(X_val), y_val).item()


def expert_assignment(model, X, top_k=1):
    """For each input, return the index of its top-routed expert."""
    with torch.no_grad():
        logits = model.router(X)
        _, idx = logits.topk(top_k, dim=-1)
    return idx[:, 0].numpy()  # take #1 expert


def main():
    torch.manual_seed(0)
    print("Training ternary MoE (8 experts, top-2) on damped oscillator...")
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    moe = TernaryMoE(3, hidden=64, out_dim=1, n_experts=8, top_k=2, depth=3)
    t0 = time.time()
    val_mse = train(moe, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256)
    print(f"  trained in {time.time()-t0:.1f}s — val MSE = {val_mse:.6f}\n")

    # Build a dense grid over (omega, zeta) at fixed t = mid-range
    t_fixed = 5.0
    omega_grid = np.linspace(0.5, 3.0, 40)
    zeta_grid = np.linspace(0.05, 0.5, 40)
    O, Z = np.meshgrid(omega_grid, zeta_grid, indexing="ij")
    n = O.size
    T = np.full(n, t_fixed)
    X_grid = np.stack([T, O.ravel(), Z.ravel()], axis=1)
    X_grid_n = (torch.from_numpy(X_grid).float() - stats["mean"]) / stats["std"]

    expert_ids = expert_assignment(moe, X_grid_n)
    heat = expert_ids.reshape(O.shape)

    print("Routing decisions on a 40x40 grid over (omega, zeta) at t=5.0:\n")
    print("Each cell shows the top-routed expert id (0..7):")
    # Print a small ASCII visualization
    for i in range(O.shape[0]):
        row = "".join(str(e) for e in heat[i])
        print(f"  omega={omega_grid[i]:.2f}: {row}")
    print()

    # How balanced is the partition?
    unique, counts = np.unique(expert_ids, return_counts=True)
    print("Expert utilization across the grid (top-1 routing):")
    for u, c in zip(unique, counts):
        bar = "█" * int(40 * c / expert_ids.size)
        print(f"  expert {u}: {c:>5} cells  ({100*c/expert_ids.size:.1f}%)  {bar}")


if __name__ == "__main__":
    main()
