"""Compare TritMLP (full ternary) vs BitMLP (weights only) vs FPMLP (fp32).

All three share the same shape so the comparison is apples-to-apples on
parameter count and depth. Task: damped harmonic oscillator regression.

Run: venv/bin/python -m src.train_tritmlp [--plot]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .data import damped_oscillator, make_dataset, normalize
from .model import BitMLP, FPMLP
from .trit_mlp import TritMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    history = {"val": []}
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
            val_loss = loss_fn(model(X_val), y_val).item()
        history["val"].append(val_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  val={val_loss:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s — final val MSE = {history['val'][-1]:.6f}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    X_train, y_train = make_dataset(args.n_train, seed=args.seed)
    X_val, y_val = make_dataset(args.n_val, seed=args.seed + 1)
    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    models = {
        "TritMLP": TritMLP(3, args.hidden, 1, depth=args.depth),
        "BitMLP":  BitMLP(3, args.hidden, 1, depth=args.depth),
        " FPMLP":  FPMLP(3, args.hidden, 1, depth=args.depth),
    }

    histories = {}
    for label, m in models.items():
        print(f"\n[{label}] params: {sum(p.numel() for p in m.parameters()):,}")
        histories[label] = train(m, X_train_n, y_train, X_val_n, y_val,
                                 args.epochs, args.lr, args.batch_size, label)

    print("\n--- TritMLP activation distribution (after training) ---")
    stats = models["TritMLP"].collect_activation_stats(X_val_n[:64])
    for i, s in enumerate(stats):
        print(f"  hidden layer {i}: -1={s['neg']:.2%}  0={s['zero']:.2%}  +1={s['pos']:.2%}")

    print("\n--- Final comparison ---")
    for label in models:
        print(f"  {label}: val MSE = {histories[label]['val'][-1]:.6f}")

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for label, h in histories.items():
            ax[0].plot(h["val"], label=label.strip())
        ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("val MSE")
        ax[0].set_title("All-ternary vs partial-ternary vs fp32")
        ax[0].legend()

        with torch.no_grad():
            t = torch.linspace(0, 10, 400)
            omega = torch.full_like(t, 1.5)
            zeta = torch.full_like(t, 0.15)
            X_demo = torch.stack([t, omega, zeta], dim=1)
            X_demo_n, _ = normalize(X_demo, stats={"mean": X_train.mean(0),
                                                    "std": X_train.std(0).clamp_min(1e-6)})
            y_true = damped_oscillator(t, omega, zeta)
            preds = {label: m(X_demo_n).squeeze() for label, m in models.items()}

        ax[1].plot(t, y_true, "k-", label="true", linewidth=2)
        ax[1].plot(t, preds["TritMLP"], "b--", label="TritMLP (full ternary)")
        ax[1].plot(t, preds["BitMLP"], "g:", label="BitMLP (ternary weights)")
        ax[1].plot(t, preds[" FPMLP"], "r-.", label="FPMLP (fp32)")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("x(t)")
        ax[1].set_title("omega=1.5, zeta=0.15")
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("tritmlp_results.png", dpi=120)
        print("\nSaved plot to tritmlp_results.png")


if __name__ == "__main__":
    main()
