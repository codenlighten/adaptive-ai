"""Train multi-body Equation Learner on 3-mass spring chain.

Truth:  H = 0.5*(p1^2 + p2^2 + p3^2) + q1^2 + q2^2 + q3^2 - q1*q2 - q2*q3
Expected sparse subset: 8 active basis terms out of 24.

Run: venv/bin/python -m src.train_chain_eql
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eql_multibody import ChainEQL
from .spring_chain import chain_vector_field, make_chain_data


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
    q, p, dq, dp = make_chain_data(n_traj=300, steps_per_traj=80, dt=0.05)
    print(f"Training data: {q.shape[0]:,} samples; dims=3 (3-mass chain)")

    print("\n--- fp32 EQL ---")
    fp = ChainEQL(ternary=False)
    train(fp, q, p, dq, dp, epochs=400, lr=2e-2, batch_size=512, label=" fp")
    print(f"  active basis: {fp.active_basis_count()} / {len(fp.basis.names)}")
    print("  Discovered H:")
    print("    " + fp.readable_formula())

    print("\n--- Ternary EQL (sparse selector) ---")
    bit = ChainEQL(ternary=True)
    train(bit, q, p, dq, dp, epochs=400, lr=2e-2, batch_size=512, label="bit")
    print(f"  active basis: {bit.active_basis_count()} / {len(bit.basis.names)}")
    print("  Discovered H:")
    print("    " + bit.readable_formula())

    print("\n  Ground truth:")
    print("    H = +0.5000 * p1^2  +0.5000 * p2^2  +0.5000 * p3^2")
    print("        +1.0000 * q1^2  +1.0000 * q2^2  +1.0000 * q3^2")
    print("        -1.0000 * q1*q2  -1.0000 * q2*q3")
    print("    (8 active basis functions)")


if __name__ == "__main__":
    main()
