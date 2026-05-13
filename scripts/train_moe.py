"""Train ternary MoE on damped oscillator. Compare to dense ternary baseline.

Run: venv/bin/python -m scripts.train_moe
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP
from src.moe import TernaryMoE


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            print(f"  [{label}] ep {epoch+1:4d}/{epochs}  val={vl:.6f}")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    # Dense ternary baseline
    print("=== Dense BitMLP (baseline) ===")
    bit = BitMLP(3, 128, 1, depth=5)
    bit_p = sum(p.numel() for p in bit.parameters())
    print(f"  params: {bit_p:,}")
    train(bit, X_train_n, y_train, X_val_n, y_val, epochs=200, lr=2e-3,
          batch_size=256, label="bit")
    bit.eval()
    with torch.no_grad():
        bit_mse = ((bit(X_val_n) - y_val) ** 2).mean().item()

    # Ternary MoE
    print("\n=== Ternary MoE (4 experts, top-2 routing) ===")
    moe = TernaryMoE(3, hidden=64, out_dim=1, n_experts=4, top_k=2, depth=3)
    moe_p = sum(p.numel() for p in moe.parameters())
    one_expert_p = sum(p.numel() for p in moe.experts[0].parameters())
    print(f"  total params: {moe_p:,}")
    print(f"  one expert  : {one_expert_p:,}")
    print(f"  router      : {sum(p.numel() for p in moe.router.parameters()):,}")
    print(f"  active params per forward pass (top-2 of 4 experts):")
    print(f"     ~{moe.top_k * one_expert_p + sum(p.numel() for p in moe.router.parameters()):,}")
    train(moe, X_train_n, y_train, X_val_n, y_val, epochs=200, lr=2e-3,
          batch_size=256, label="moe")
    moe.eval()
    with torch.no_grad():
        moe_mse = ((moe(X_val_n) - y_val) ** 2).mean().item()

    print("\n=== Comparison ===")
    print(f"  BitMLP (dense)    val MSE = {bit_mse:.6f}  params={bit_p:,}")
    print(f"  Ternary MoE       val MSE = {moe_mse:.6f}  params={moe_p:,}")

    print("\n--- Routing statistics on validation set ---")
    stats = moe.routing_stats(X_val_n)
    for k, v in stats.items():
        bar = "█" * int(v * 40)
        print(f"  {k}: {v:.2%}  {bar}")


if __name__ == "__main__":
    main()
