"""Empirical rate-distortion curve for weight quantization.

Rate = empirical bits per weight, computed from the entropy of the
quantized weight distribution (Shannon entropy of the per-level
frequencies). This is the *actual* information-theoretic cost, not the
naive bit-width — a ternary network whose weights are 80% zero has a
much lower rate than 1.58 bits/weight because zeros are common.

Distortion = validation MSE on damped oscillator.

We train identical MLPs at five weight precisions and plot the R(D)
points. Lower R for the same D = better compression. This is the proper
way to compare quantization schemes — the actual entropy, not the
bit-width "budget".

Run: venv/bin/python -m scripts.rate_distortion
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP, FPMLP
from src.precision_models import BinaryMLP, QuintMLP
from src.quantize_helpers import collect_weights_quantized
from src.quaternary import QuatMLP


def shannon_entropy(values: np.ndarray) -> float:
    """Empirical entropy in bits."""
    counts = Counter(values.tolist())
    total = len(values)
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H


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

    configs = [
        ("Binary",    BinaryMLP(**common), "binary"),
        ("Ternary",   BitMLP(**common),    "ternary"),
        ("Quaternary",QuatMLP(**common),   "quaternary"),
        ("Quintary",  QuintMLP(**common),  "quintary"),
        ("fp32",      FPMLP(**common),     "fp32"),
    ]
    results = []
    for label, model, scheme in configs:
        t0 = time.time()
        mse = train(model, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256)
        dt = time.time() - t0
        if scheme != "fp32":
            quantized = collect_weights_quantized(model, scheme)
            H = shannon_entropy(quantized)
            naive_bits = math.log2(len(set(quantized.tolist())))
        else:
            H = 32.0
            naive_bits = 32.0
        results.append({
            "label": label, "mse": mse, "H_bits": H, "naive_bits": naive_bits,
            "n_quantized": int(0 if scheme == "fp32" else len(collect_weights_quantized(model, scheme))),
            "scheme": scheme,
        })
        print(f"  {label:<11s}  MSE={mse:.6f}  H={H:.4f} bits/w  naive={naive_bits:.2f} bits/w "
              f"({dt:.1f}s)")

    print("\n=== Rate-distortion summary ===")
    print(f"{'scheme':>12}  {'rate H (bits/w)':>18}  {'distortion MSE':>16}  {'naive bits':>11}")
    for r in results:
        print(f"{r['label']:>12}  {r['H_bits']:>18.4f}  {r['mse']:>16.6f}  {r['naive_bits']:>11.2f}")

    with open("rate_distortion.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        Hs = [r["H_bits"] for r in results]
        Ds = [r["mse"] for r in results]
        labels = [r["label"] for r in results]
        ax.plot(Hs, Ds, "ko-", linewidth=2, markersize=10)
        for h, d, l in zip(Hs, Ds, labels):
            ax.annotate(l, (h, d), textcoords="offset points", xytext=(8, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rate (empirical bits per weight, Shannon)")
        ax.set_ylabel("Distortion (val MSE)")
        ax.set_title("Empirical R(D) curve — damped oscillator regression")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("rate_distortion.png", dpi=120)
        print("\nSaved rate_distortion.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
