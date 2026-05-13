"""Train the Equation Learner (ternary + fp) on pendulum data.

The model learns H(q, p) directly via supervision on the predicted
vector field dq/dt = dH/dp, dp/dt = -dH/dq (same loss as the regular HNN).
Then we read the discovered formula off the ternary weights.

Ground truth: H = p^2/2 + (1 - cos q) = 1 - cos(q) + 0.5*p^2.

Run: venv/bin/python -m src.train_eql
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eql import TernaryEQL
from .hnn import make_pendulum_data, pendulum_hamiltonian


def train(model, q, p, dq_t, dp_t, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = q.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            dq_p, dp_p = model.vector_field(q[idx], p[idx])
            loss = loss_fn(dq_p, dq_t[idx]) + loss_fn(dp_p, dp_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  loss={total/n:.5f}")


def main():
    torch.manual_seed(0)
    q, p, dq, dp = make_pendulum_data(n_traj=300, steps_per_traj=64, dt=0.05, seed=0)
    print(f"Training data: {q.shape[0]:,} (q, p, dq, dp) samples")

    print("\n--- Training fp32 EQL (dense continuous coefficients) ---")
    fp_eql = TernaryEQL(ternary=False)
    train(fp_eql, q, p, dq, dp, epochs=400, lr=2e-2, batch_size=512, label=" fp")
    print(f"\n  active basis functions kept: {fp_eql.active_basis_count()} / {len(fp_eql.basis.names)}")
    print("  Discovered H (fp):")
    print("    H(q,p) ≈ " + fp_eql.readable_formula())

    print("\n--- Training ternary EQL (sparse selector + per-feature scale) ---")
    bit_eql = TernaryEQL(ternary=True)
    train(bit_eql, q, p, dq, dp, epochs=400, lr=2e-2, batch_size=512, label="bit")
    print(f"\n  active basis functions kept: {bit_eql.active_basis_count()} / {len(bit_eql.basis.names)}")
    print("  Discovered H (ternary):")
    print("    H(q,p) ≈ " + bit_eql.readable_formula())

    # Ground truth for comparison: H = p^2/2 + 1 - cos q
    # = 1*const - 1*cos(q) + 0.5*p^2
    print("\n  Ground truth:")
    print("    H(q,p) = +1.0000 * 1  -1.0000 * cos(q)  +0.5000 * p^2")

    # Evaluate vector-field MSE on a grid
    qg = torch.linspace(-2.5, 2.5, 40)
    pg = torch.linspace(-2.0, 2.0, 40)
    Q, P = torch.meshgrid(qg, pg, indexing="ij")
    q_flat = Q.reshape(-1, 1)
    p_flat = P.reshape(-1, 1)
    from .hnn import pendulum_vector_field
    dq_true, dp_true = pendulum_vector_field(q_flat, p_flat)

    def grid_mse(model):
        dq_p, dp_p = model.vector_field(q_flat, p_flat)
        return (((dq_p - dq_true) ** 2 + (dp_p - dp_true) ** 2).mean()).item()

    print(f"\n  fp EQL  grid VF MSE: {grid_mse(fp_eql):.6f}")
    print(f"  bit EQL grid VF MSE: {grid_mse(bit_eql):.6f}")


if __name__ == "__main__":
    main()
