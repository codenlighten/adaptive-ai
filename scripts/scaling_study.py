"""Scaling study: BitMLP vs FPMLP at varying hidden widths.

BitNet b1.58's central claim is that the ternary/fp32 gap *closes* with scale.
We test this on the damped oscillator at hidden=[16,32,64,128,256,512].

Run: venv/bin/python -m scripts.scaling_study
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP, FPMLP


def train_quick(model, X, y, X_val, y_val, epochs, lr, batch_size):
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
    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    widths = [16, 32, 64, 128, 256, 512]
    depth = 5
    epochs = 150
    results = []

    for w in widths:
        bit = BitMLP(3, w, 1, depth=depth)
        fp = FPMLP(3, w, 1, depth=depth)
        n_bit = sum(p.numel() for p in bit.parameters())
        n_fp = sum(p.numel() for p in fp.parameters())

        t0 = time.time()
        bit_mse = train_quick(bit, X_train_n, y_train, X_val_n, y_val,
                              epochs=epochs, lr=2e-3, batch_size=256)
        t_bit = time.time() - t0

        t0 = time.time()
        fp_mse = train_quick(fp, X_train_n, y_train, X_val_n, y_val,
                             epochs=epochs, lr=2e-3, batch_size=256)
        t_fp = time.time() - t0

        ratio = bit_mse / fp_mse
        results.append({
            "width": w,
            "params_bit": n_bit,
            "params_fp": n_fp,
            "bit_mse": bit_mse,
            "fp_mse": fp_mse,
            "ratio": ratio,
            "time_bit_s": t_bit,
            "time_fp_s": t_fp,
        })
        print(f"width={w:4d}  bit_params={n_bit:>7,}  bit_mse={bit_mse:.6f}  "
              f"fp_mse={fp_mse:.6f}  ratio={ratio:6.2f}x  "
              f"(bit {t_bit:.1f}s, fp {t_fp:.1f}s)")

    print()
    print("=" * 60)
    print("SCALING SUMMARY (ratio = bit_mse / fp_mse — closer to 1.0 is better)")
    print("=" * 60)
    print(f"{'width':>6}  {'bit_mse':>12}  {'fp_mse':>12}  {'ratio':>8}")
    for r in results:
        print(f"{r['width']:>6}  {r['bit_mse']:>12.6f}  {r['fp_mse']:>12.6f}  {r['ratio']:>7.2f}x")

    with open("scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved scaling_results.json")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        widths_a = [r["width"] for r in results]
        bit_mses = [r["bit_mse"] for r in results]
        fp_mses = [r["fp_mse"] for r in results]
        ratios = [r["ratio"] for r in results]

        ax[0].plot(widths_a, bit_mses, "b-o", label="BitMLP (ternary)")
        ax[0].plot(widths_a, fp_mses, "r-s", label="FPMLP (fp32)")
        ax[0].set_xscale("log", base=2)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("hidden width")
        ax[0].set_ylabel("val MSE")
        ax[0].set_title("Val MSE vs model width")
        ax[0].legend()
        ax[0].grid(True, which="both", alpha=0.3)

        ax[1].plot(widths_a, ratios, "k-o")
        ax[1].axhline(1.0, color="gray", linestyle=":", label="parity")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("hidden width")
        ax[1].set_ylabel("bit_mse / fp_mse")
        ax[1].set_title("Ternary/fp gap (lower = ternary closer to fp)")
        ax[1].legend()
        ax[1].grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plt.savefig("scaling_results.png", dpi=120)
        print("Saved scaling_results.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
