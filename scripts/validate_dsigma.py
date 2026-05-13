"""Validate delta-sigma weights across three different tasks.

Goal: check whether the T=8 sweet spot from the damped oscillator
generalizes to (1) Schrödinger eigenvalue regression and (2) sklearn
digits classification.

Run: venv/bin/python -m scripts.validate_dsigma
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from delta_sigma_nn.dsigma_linear import DeltaSigmaMLP
from src.model import BitMLP, FPMLP
from src.schrodinger import make_dataset as schrodinger_dataset
from src.schrodinger import normalize as schrodinger_normalize


# ---------------------------------------------------------------------------
# Generic train loops
# ---------------------------------------------------------------------------

def train_regression(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
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


def train_classification(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        loss = F.cross_entropy(logits, y_val).item()
        acc = (logits.argmax(dim=-1) == y_val).float().mean().item()
    print(f"  [{label}] {time.time()-t0:.1f}s  loss = {loss:.4f}  acc = {acc:.4f}")
    return loss, acc


# ---------------------------------------------------------------------------
# Task 1: Schrödinger eigenvalue regression
# ---------------------------------------------------------------------------

def run_schrodinger():
    print("\n" + "=" * 70)
    print("Schrödinger ground-state energy regression")
    print("=" * 70)
    torch.manual_seed(0)
    X_train, y_train = schrodinger_dataset(3000, seed=0)
    X_val, y_val = schrodinger_dataset(500, seed=1)
    Xtn, stats = schrodinger_normalize(X_train)
    Xvn, _ = schrodinger_normalize(X_val, stats)

    common = dict(in_dim=2, hidden_dim=128, out_dim=1, depth=5)
    print(f"\nBaselines:")
    bit = train_regression(BitMLP(**common), Xtn, y_train, Xvn, y_val,
                            epochs=200, lr=2e-3, batch_size=128, label="BitMLP")
    fp = train_regression(FPMLP(**common), Xtn, y_train, Xvn, y_val,
                           epochs=200, lr=2e-3, batch_size=128, label="FPMLP ")

    print(f"\nDelta-Sigma sweep:")
    rows = [("BitMLP", 1, bit), ("FPMLP", 0, fp)]
    for T in [4, 8, 16]:
        m = DeltaSigmaMLP(**common, T=T, order=1)
        mse = train_regression(m, Xtn, y_train, Xvn, y_val,
                                epochs=200, lr=2e-3, batch_size=128,
                                label=f"DSigma T={T:>2d}")
        rows.append((f"DSigma T={T}", T, mse))
    return rows


# ---------------------------------------------------------------------------
# Task 2: digits classification
# ---------------------------------------------------------------------------

def run_digits():
    print("\n" + "=" * 70)
    print("sklearn digits classification (8x8, 10 classes)")
    print("=" * 70)
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr = torch.from_numpy(X_tr); X_te = torch.from_numpy(X_te)
    y_tr = torch.from_numpy(y_tr); y_te = torch.from_numpy(y_te)

    common = dict(in_dim=64, hidden_dim=128, out_dim=10, depth=4)
    print(f"\nBaselines:")
    bit_loss, bit_acc = train_classification(BitMLP(**common), X_tr, y_tr, X_te, y_te,
                                              epochs=200, lr=2e-3, batch_size=64, label="BitMLP")
    fp_loss, fp_acc = train_classification(FPMLP(**common), X_tr, y_tr, X_te, y_te,
                                            epochs=200, lr=2e-3, batch_size=64, label="FPMLP ")

    print(f"\nDelta-Sigma sweep:")
    rows = [("BitMLP", 1, bit_loss, bit_acc), ("FPMLP", 0, fp_loss, fp_acc)]
    for T in [4, 8, 16]:
        m = DeltaSigmaMLP(**common, T=T, order=1)
        loss, acc = train_classification(m, X_tr, y_tr, X_te, y_te,
                                          epochs=200, lr=2e-3, batch_size=64,
                                          label=f"DSigma T={T:>2d}")
        rows.append((f"DSigma T={T}", T, loss, acc))
    return rows


def main():
    schrod = run_schrodinger()
    digits = run_digits()

    print("\n" + "=" * 70)
    print("CROSS-TASK SUMMARY")
    print("=" * 70)
    print("\nSchrödinger (val MSE, lower = better):")
    for label, T, mse in schrod:
        print(f"  {label:>12s}  {mse:.6f}")
    print("\nDigits (val acc, higher = better):")
    for label, T, loss, acc in digits:
        print(f"  {label:>12s}  loss={loss:.4f}  acc={acc:.4f}")

    with open("validate_dsigma.json", "w") as f:
        json.dump({
            "schrodinger": [list(r) for r in schrod],
            "digits": [list(r) for r in digits],
        }, f, indent=2)


if __name__ == "__main__":
    main()
