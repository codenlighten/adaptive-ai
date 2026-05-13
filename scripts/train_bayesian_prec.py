"""Train Bayesian mixed-precision MLP. Compare:
 - point-estimate learned mixed (phase 32)
 - Bayesian Dirichlet (this script): sample pi at train, average at inference
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.bayesian_prec import BayesianPrecMLP
from src.data import make_dataset, normalize


def train(model, X, y, X_val, y_val, epochs, lr, batch_size):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            print(f"  ep {epoch+1:4d}/{epochs}  val (expected pi)={vl:.6f}  "
                  f"alphas={[a for a in model.alphas()]}")
    print(f"done in {time.time()-t0:.1f}s")


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    model = BayesianPrecMLP(3, 128, 1, depth=5)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256)

    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        point_mse = loss_fn(model(Xvn), y_val).item()
    ens = model.ensemble_predict(Xvn, n_samples=32)
    ens_mse = loss_fn(ens, y_val).item()

    print("\n--- Posterior over precisions per layer ---")
    for i, a in enumerate(model.alphas()):
        total = sum(a.values())
        prefs = {k: v / total for k, v in a.items()}
        print(f"  layer {i}: alpha={a}  -> expected pi={prefs}")

    print(f"\nPoint estimate (expected pi) val MSE: {point_mse:.6f}")
    print(f"Ensemble of 32 posterior samples val MSE: {ens_mse:.6f}")


if __name__ == "__main__":
    main()
