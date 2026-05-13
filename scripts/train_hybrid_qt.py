"""4-way comparison on damped oscillator:
   all-ternary, all-quaternary, hybrid Q+T, fp32.

Run: venv/bin/python -m scripts.train_hybrid_qt
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.hybrid_qt import HybridQTMLP
from src.model import BitMLP, FPMLP
from src.quaternary import QuatMLP


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


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    common = dict(in_dim=3, hidden_dim=128, out_dim=1, depth=5)

    print("Training four configs on damped oscillator...\n")

    configs = []
    configs.append(("BitMLP (all ternary)", BitMLP(**common)))
    configs.append(("QuatMLP (all quaternary)", QuatMLP(**common)))
    hyb = HybridQTMLP(**common, quaternary_fraction=0.4)
    configs.append((f"Hybrid Q+T  (assignment: {' '.join(hyb.assignment())})", hyb))
    configs.append(("FPMLP (fp32)", FPMLP(**common)))

    results = []
    for label, model in configs:
        n_p = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        mse = train(model, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256)
        dt = time.time() - t0
        # average bits/weight estimate
        # (ternary ≈ 1.58, quaternary = 2, fp32 = 32)
        results.append((label, n_p, mse, dt))
        print(f"  {label:<48s} params={n_p:>7,}  val_mse={mse:.6f}  ({dt:.1f}s)")

    print("\n=== Summary ===")
    base_mse = next(r[2] for r in results if "fp32" in r[0])
    for label, n_p, mse, _ in results:
        print(f"  {label:<48s}  val MSE={mse:.6f}  ratio={mse/base_mse:.2f}x vs fp32")


if __name__ == "__main__":
    main()
