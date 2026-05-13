"""Train DeltaSigmaMLP at various T values on damped oscillator.

Compare to pure ternary (T=1 effectively) and fp32. Plot val MSE vs T.

Run: venv/bin/python -m scripts.train_dsigma --plot
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from delta_sigma_nn.dsigma_linear import DeltaSigmaMLP
from src.model import BitMLP, FPMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        v = loss_fn(model(X_val), y_val).item()
    print(f"  [{label}] {time.time()-t0:.1f}s  val MSE = {v:.6f}")
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    common = dict(in_dim=3, hidden_dim=128, out_dim=1, depth=5)
    results = []

    print("=== Baselines ===")
    bit_mse = train(BitMLP(**common), Xtn, y_train, Xvn, y_val,
                    epochs=150, lr=2e-3, batch_size=256, label="BitMLP")
    fp_mse = train(FPMLP(**common), Xtn, y_train, Xvn, y_val,
                   epochs=150, lr=2e-3, batch_size=256, label="FPMLP ")
    results.append({"label": "BitMLP (ternary)", "T": 1, "mse": bit_mse})
    results.append({"label": "FPMLP (fp32)", "T": 0, "mse": fp_mse})

    print("\n=== Delta-Sigma MLP at various T ===")
    for T in [2, 4, 8, 16, 32]:
        model = DeltaSigmaMLP(**common, T=T, order=1)
        mse = train(model, Xtn, y_train, Xvn, y_val,
                    epochs=150, lr=2e-3, batch_size=256, label=f"DSigma T={T:>2d}")
        results.append({"label": f"DSigma T={T}", "T": T, "mse": mse})

    print("\n=== Second-order delta-sigma (better noise shaping) ===")
    for T in [4, 8, 16]:
        model = DeltaSigmaMLP(**common, T=T, order=2)
        mse = train(model, Xtn, y_train, Xvn, y_val,
                    epochs=150, lr=2e-3, batch_size=256, label=f"DSigma2 T={T:>2d}")
        results.append({"label": f"DSigma2 T={T}", "T": T, "mse": mse, "order": 2})

    print("\n=== Summary ===")
    print(f"{'config':>22}  {'val MSE':>12}  {'vs fp32':>10}")
    for r in results:
        print(f"{r['label']:>22}  {r['mse']:>12.6f}  {r['mse']/fp_mse:>9.2f}x")

    with open("dsigma_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        Ts = [r["T"] for r in results if r["T"] > 1 and r.get("order") != 2]
        mses = [r["mse"] for r in results if r["T"] > 1 and r.get("order") != 2]
        Ts2 = [r["T"] for r in results if r.get("order") == 2]
        mses2 = [r["mse"] for r in results if r.get("order") == 2]
        ax.plot(Ts, mses, "bo-", linewidth=2, markersize=10, label="Delta-Sigma (1st order)")
        if Ts2:
            ax.plot(Ts2, mses2, "g^--", linewidth=2, markersize=10, label="Delta-Sigma (2nd order)")
        ax.axhline(bit_mse, color="orange", linestyle=":", label=f"Ternary baseline ({bit_mse:.5f})")
        ax.axhline(fp_mse, color="red", linestyle=":", label=f"FP32 baseline ({fp_mse:.5f})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("T (delta-sigma time steps)")
        ax.set_ylabel("val MSE")
        ax.set_title("Delta-Sigma weights: val MSE vs time-step count T")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("dsigma_results.png", dpi=120)
        print("Saved dsigma_results.png")


if __name__ == "__main__":
    main()
