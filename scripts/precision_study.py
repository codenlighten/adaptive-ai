"""Weight-precision study: binary, ternary, quintary, fp32.

Same task, same shape, four weight precisions. Plots accuracy vs bits/weight.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP, FPMLP
from src.precision_models import BinaryMLP, QuintMLP
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
    X_train_n, _ = normalize(X_train)
    X_val_n, _ = normalize(X_val)

    common = dict(in_dim=3, hidden_dim=128, out_dim=1, depth=5)
    configs = [
        # (label, model, bits_per_weight)
        ("Binary  {-1,+1}",    BinaryMLP(**common), 1.00),
        ("Ternary {-1,0,+1}",  BitMLP(**common),    1.58),
        ("Quaternary 4-state", QuatMLP(**common),   2.00),
        ("Quintary{-2..+2}",   QuintMLP(**common),  2.32),
        ("fp32",               FPMLP(**common),     32.0),
    ]

    results = []
    for label, model, bits in configs:
        n_p = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        mse = train(model, X_train_n, y_train, X_val_n, y_val,
                    epochs=200, lr=2e-3, batch_size=256)
        dt = time.time() - t0
        print(f"  {label:24s}  params={n_p:>7,}  bits/w={bits:>5.2f}  "
              f"val_mse={mse:.6f}  ({dt:.1f}s)")
        results.append({"label": label, "bits": bits, "params": n_p, "mse": mse})

    print()
    print("=" * 70)
    print("Precision-accuracy tradeoff")
    print("=" * 70)
    print(f"{'precision':>24}  {'bits/weight':>12}  {'val MSE':>12}  {'vs fp32':>8}")
    fp_mse = next(r["mse"] for r in results if r["label"] == "fp32")
    for r in results:
        print(f"{r['label']:>24}  {r['bits']:>12.2f}  {r['mse']:>12.6f}  "
              f"{r['mse']/fp_mse:>7.2f}x")

    # QuintMLP weight distribution
    print("\nQuintary weight distribution (first hidden layer):")
    quint_model = next(m for label, m, _ in configs if label.startswith("Quintary"))
    s = quint_model.quint_stats()[0]
    print(f"  -2: {s['neg2']:.2%}  -1: {s['neg1']:.2%}  0: {s['zero']:.2%}  "
          f"+1: {s['pos1']:.2%}  +2: {s['pos2']:.2%}  alpha={s['alpha']:.4f}")

    with open("precision_results.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        bits = [r["bits"] for r in results]
        mses = [r["mse"] for r in results]
        labels = [r["label"] for r in results]
        ax.plot(bits, mses, "ko-", linewidth=2, markersize=10)
        for b, m, l in zip(bits, mses, labels):
            ax.annotate(l, (b, m), textcoords="offset points", xytext=(10, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("bits per weight (information content)")
        ax.set_ylabel("val MSE (damped oscillator)")
        ax.set_title("Weight precision vs accuracy")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("precision_results.png", dpi=120)
        print("\nSaved precision_results.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
