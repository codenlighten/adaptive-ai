"""Train BitMLP and FPMLP on the damped oscillator task and compare.

Usage: python -m src.train
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from .data import make_dataset, normalize
from .model import BitMLP, FPMLP


def train_one(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
              X_val: torch.Tensor, y_val: torch.Tensor,
              epochs: int, lr: float, batch_size: int, label: str) -> dict:
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
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
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

    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s — final val MSE = {history['val'][-1]:.5f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    X_train, y_train = make_dataset(args.n_train, seed=args.seed)
    X_val, y_val = make_dataset(args.n_val, seed=args.seed + 1)
    X_train, stats = normalize(X_train)
    X_val, _ = normalize(X_val, stats)

    bit = BitMLP(3, args.hidden, 1, depth=args.depth)
    fp = FPMLP(3, args.hidden, 1, depth=args.depth)

    n_params = lambda m: sum(p.numel() for p in m.parameters())
    print(f"BitMLP params: {n_params(bit):,}")
    print(f" FPMLP params: {n_params(fp):,}")
    print()

    hist_bit = train_one(bit, X_train, y_train, X_val, y_val,
                         args.epochs, args.lr, args.batch_size, "BitMLP")
    print()
    hist_fp = train_one(fp, X_train, y_train, X_val, y_val,
                        args.epochs, args.lr, args.batch_size, " FPMLP")

    print("\n--- Ternary weight distribution (BitMLP) ---")
    for i, s in enumerate(bit.ternary_stats()):
        print(f"  layer {i}: -1={s['neg']:.2%}  0={s['zero']:.2%}  +1={s['pos']:.2%}  alpha={s['alpha']:.4f}")

    print("\n--- Final comparison ---")
    print(f"BitMLP val MSE: {hist_bit['val'][-1]:.5f}")
    print(f" FPMLP val MSE: {hist_fp['val'][-1]:.5f}")
    print(f"Ratio (bit/fp): {hist_bit['val'][-1] / hist_fp['val'][-1]:.2f}x")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(hist_bit["val"], label="BitMLP (ternary)")
        ax[0].plot(hist_fp["val"], label="FPMLP (fp32)")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("val MSE")
        ax[0].legend()
        ax[0].set_title("Damped oscillator — validation loss")

        with torch.no_grad():
            t = torch.linspace(0, 10, 400)
            omega = torch.full_like(t, 1.5)
            zeta = torch.full_like(t, 0.15)
            X_demo = torch.stack([t, omega, zeta], dim=1)
            X_demo_n, _ = normalize(X_demo, stats)
            from .data import damped_oscillator
            y_true = damped_oscillator(t, omega, zeta)
            y_bit = bit(X_demo_n).squeeze()
            y_fp = fp(X_demo_n).squeeze()

        ax[1].plot(t, y_true, "k-", label="true", linewidth=2)
        ax[1].plot(t, y_bit, "--", label="BitMLP")
        ax[1].plot(t, y_fp, ":", label="FPMLP")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("x(t)")
        ax[1].legend()
        ax[1].set_title(f"omega=1.5, zeta=0.15")

        plt.tight_layout()
        plt.savefig("results.png", dpi=120)
        print("\nSaved plot to results.png")


if __name__ == "__main__":
    main()
