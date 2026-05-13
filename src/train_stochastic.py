"""Train the stochastic-trit-flow MLP and show ensemble averaging removes
the staircase artifact from Phase 1's deterministic TritMLP.

Run: venv/bin/python -m src.train_stochastic [--plot]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .data import damped_oscillator, make_dataset, normalize
from .stochastic_mlp import StochasticTritMLP
from .trit_mlp import TritMLP


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
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  val={val:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-ensemble", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X_train, y_train = make_dataset(args.n_train, seed=args.seed)
    X_val, y_val = make_dataset(args.n_val, seed=args.seed + 1)
    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    stoch = StochasticTritMLP(3, args.hidden, 1, depth=args.depth)
    det = TritMLP(3, args.hidden, 1, depth=args.depth)

    print(f"Stochastic-trit MLP params:  {sum(p.numel() for p in stoch.parameters()):,}")
    print(f"Deterministic-trit MLP params: {sum(p.numel() for p in det.parameters()):,}\n")

    train(stoch, X_train_n, y_train, X_val_n, y_val,
          args.epochs, 2e-3, 256, "Stoch")
    print()
    train(det, X_train_n, y_train, X_val_n, y_val,
          args.epochs, 2e-3, 256, " Det ")

    # Compare single-pass vs ensemble inference.
    loss_fn = nn.MSELoss()
    stoch.eval()
    det.eval()
    with torch.no_grad():
        single = stoch(X_val_n)
        det_out = det(X_val_n)
    ensemble = stoch.forward_ensemble(X_val_n, n_samples=args.n_ensemble)

    print("\n--- Inference comparison on validation set ---")
    print(f"  Stochastic TritMLP — single deterministic pass MSE: "
          f"{loss_fn(single, y_val).item():.6f}")
    print(f"  Stochastic TritMLP — ensemble of {args.n_ensemble} samples MSE: "
          f"{loss_fn(ensemble, y_val).item():.6f}")
    print(f"  Deterministic  TritMLP (Phase 1) MSE:                "
          f"{loss_fn(det_out, y_val).item():.6f}")

    if args.plot:
        import matplotlib.pyplot as plt
        with torch.no_grad():
            t = torch.linspace(0, 10, 400)
            omega = torch.full_like(t, 1.5)
            zeta = torch.full_like(t, 0.15)
            X_demo = torch.stack([t, omega, zeta], dim=1)
            X_demo_n, _ = normalize(X_demo, stats)
            y_true = damped_oscillator(t, omega, zeta).numpy()
            y_det = det(X_demo_n).squeeze().numpy()
            y_stoch_single = stoch(X_demo_n).squeeze().numpy()
        y_stoch_ens = stoch.forward_ensemble(X_demo_n, n_samples=args.n_ensemble).squeeze().numpy()

        plt.figure(figsize=(11, 5))
        plt.plot(t, y_true, "k-", label="true", linewidth=2)
        plt.plot(t, y_det, "g:", label="TritMLP (deterministic)", alpha=0.7)
        plt.plot(t, y_stoch_single, "c-", alpha=0.4, label="Stoch single pass")
        plt.plot(t, y_stoch_ens, "b--", label=f"Stoch ensemble (N={args.n_ensemble})")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.title("Damped oscillator — stochastic trit-flow with ensemble averaging")
        plt.legend()
        plt.tight_layout()
        plt.savefig("stochastic_results.png", dpi=120)
        print("\nSaved plot to stochastic_results.png")


if __name__ == "__main__":
    main()
