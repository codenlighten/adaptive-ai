"""Train BitMLP and FPMLP to predict Schrodinger ground-state energy E0(a, b).

Run: venv/bin/python -m src.train_schrodinger [--plot]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .model import BitMLP, FPMLP
from .schrodinger import make_dataset, normalize


def train_one(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    history = {"train": [], "val": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        train_loss = total / n
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--grid-n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Building datasets (this is the slow part — solving {args.n_train + args.n_val} "
          f"eigenvalue problems with grid_n={args.grid_n})...")
    t0 = time.time()
    X_train, y_train = make_dataset(args.n_train, seed=args.seed, grid_n=args.grid_n)
    X_val, y_val = make_dataset(args.n_val, seed=args.seed + 1, grid_n=args.grid_n)
    print(f"  ...took {time.time()-t0:.1f}s")
    print(f"  E0 range: [{y_train.min().item():.3f}, {y_train.max().item():.3f}]")

    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    bit = BitMLP(2, args.hidden, 1, depth=args.depth)
    fp = FPMLP(2, args.hidden, 1, depth=args.depth)
    print(f"\nBitMLP params: {sum(p.numel() for p in bit.parameters()):,}")
    print(f" FPMLP params: {sum(p.numel() for p in fp.parameters()):,}\n")

    hist_bit = train_one(bit, X_train_n, y_train, X_val_n, y_val,
                         args.epochs, args.lr, args.batch_size, "BitMLP")
    print()
    hist_fp = train_one(fp, X_train_n, y_train, X_val_n, y_val,
                        args.epochs, args.lr, args.batch_size, " FPMLP")

    # Mean absolute error in physical units (Hartree-like, since hbar=m=1)
    with torch.no_grad():
        bit_mae = (bit(X_val_n) - y_val).abs().mean().item()
        fp_mae = (fp(X_val_n) - y_val).abs().mean().item()
        y_std = y_val.std().item()
    print("\n--- Final comparison ---")
    print(f"BitMLP val MSE: {hist_bit['val'][-1]:.5f}   MAE: {bit_mae:.5f}   MAE/std(E0): {bit_mae/y_std:.4f}")
    print(f" FPMLP val MSE: {hist_fp['val'][-1]:.5f}   MAE: {fp_mae:.5f}   MAE/std(E0): {fp_mae/y_std:.4f}")

    print("\n--- Ternary weight distribution (BitMLP hidden layers) ---")
    for i, s in enumerate(bit.ternary_stats()):
        print(f"  layer {i}: -1={s['neg']:.2%}  0={s['zero']:.2%}  +1={s['pos']:.2%}  alpha={s['alpha']:.4f}")

    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        ax[0].plot(hist_bit["val"], label="BitMLP (ternary)")
        ax[0].plot(hist_fp["val"], label="FPMLP (fp32)")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("val MSE")
        ax[0].set_title("Schrodinger E0 — validation loss")
        ax[0].legend()

        # 2D heatmap of true E0 over (a, b)
        a_grid = np.linspace(-4, 2, 50)
        b_grid = np.linspace(0.05, 1.0, 50)
        A, B = np.meshgrid(a_grid, b_grid)
        X_grid = torch.from_numpy(np.stack([A.ravel(), B.ravel()], axis=1)).float()
        X_grid_n, _ = normalize(X_grid, stats)
        with torch.no_grad():
            E_bit = bit(X_grid_n).numpy().reshape(A.shape)
            E_fp = fp(X_grid_n).numpy().reshape(A.shape)
        im1 = ax[1].pcolormesh(A, B, E_bit, shading="auto", cmap="viridis")
        ax[1].set_xlabel("a (x^2 coefficient)")
        ax[1].set_ylabel("b (x^4 coefficient)")
        ax[1].set_title("BitMLP prediction of E0(a, b)")
        plt.colorbar(im1, ax=ax[1])

        im2 = ax[2].pcolormesh(A, B, E_fp, shading="auto", cmap="viridis")
        ax[2].set_xlabel("a")
        ax[2].set_ylabel("b")
        ax[2].set_title("FPMLP prediction of E0(a, b)")
        plt.colorbar(im2, ax=ax[2])

        plt.tight_layout()
        plt.savefig("schrodinger_results.png", dpi=120)
        print("\nSaved plot to schrodinger_results.png")


if __name__ == "__main__":
    main()
