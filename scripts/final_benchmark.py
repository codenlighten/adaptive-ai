"""End-to-end benchmark suite for delta-sigma neural networks.

Produces a single results table covering:
  - storage (bytes per layer)
  - inference compute (multiplies, adds, signed adds)
  - accuracy on three tasks
  - anytime inference characteristics
  - hardware (FPGA LCs and timing)

Run: venv/bin/python -m scripts.final_benchmark
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.data import make_dataset as oscillator_dataset
from src.data import normalize as oscillator_normalize
from delta_sigma_nn.dsigma_linear import DeltaSigmaLinear, DeltaSigmaMLP
from delta_sigma_nn.dsigma_pack import save_dsigma_mlp
from src.model import BitMLP, FPMLP
from src.schrodinger import make_dataset as schrodinger_dataset
from src.schrodinger import normalize as schrodinger_normalize


def quick_train_regression(model, X, y, X_val, y_val, epochs=150):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return loss_fn(model(X_val), y_val).item()


def quick_train_classification(model, X, y, X_val, y_val, epochs=200):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
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
    return loss, acc


def count_storage_per_weight(precision_label: str, n_total_params: int,
                              fp32_state_bytes: int) -> dict:
    """Return storage breakdown."""
    return {
        "label": precision_label,
        "fp32_state_bytes": fp32_state_bytes,
        "fp32_bytes_per_weight": fp32_state_bytes / max(n_total_params, 1),
    }


def main():
    print("=" * 70)
    print("FINAL BENCHMARK SUITE: delta-sigma vs ternary vs fp32")
    print("=" * 70)

    torch.manual_seed(0)
    results = {"tasks": {}, "hardware": {}}

    # ---- Task 1: damped oscillator ----
    print("\n[1/3] Damped harmonic oscillator regression")
    X_train, y_train = oscillator_dataset(8000, seed=0)
    X_val, y_val = oscillator_dataset(2000, seed=1)
    Xtn, stats = oscillator_normalize(X_train)
    Xvn, _ = oscillator_normalize(X_val, stats)
    common = dict(in_dim=3, hidden_dim=128, out_dim=1, depth=5)
    task1 = {}
    for label, make in [("BitMLP", lambda: BitMLP(**common)),
                         ("FPMLP",  lambda: FPMLP(**common)),
                         ("DSigma T=8",  lambda: DeltaSigmaMLP(**common, T=8)),
                         ("DSigma T=16", lambda: DeltaSigmaMLP(**common, T=16))]:
        m = make()
        mse = quick_train_regression(m, Xtn, y_train, Xvn, y_val, epochs=150)
        task1[label] = mse
        print(f"  {label:>12}: val MSE = {mse:.6f}")
    results["tasks"]["oscillator"] = task1

    # ---- Task 2: Schrödinger ----
    print("\n[2/3] Schrödinger ground-state energy regression")
    X_train, y_train = schrodinger_dataset(3000, seed=0)
    X_val, y_val = schrodinger_dataset(500, seed=1)
    Xtn, stats = schrodinger_normalize(X_train)
    Xvn, _ = schrodinger_normalize(X_val, stats)
    common = dict(in_dim=2, hidden_dim=128, out_dim=1, depth=5)
    task2 = {}
    for label, make in [("BitMLP", lambda: BitMLP(**common)),
                         ("FPMLP",  lambda: FPMLP(**common)),
                         ("DSigma T=8",  lambda: DeltaSigmaMLP(**common, T=8)),
                         ("DSigma T=16", lambda: DeltaSigmaMLP(**common, T=16))]:
        m = make()
        mse = quick_train_regression(m, Xtn, y_train, Xvn, y_val, epochs=150)
        task2[label] = mse
        print(f"  {label:>12}: val MSE = {mse:.6f}")
    results["tasks"]["schrodinger"] = task2

    # ---- Task 3: digits classification ----
    print("\n[3/3] sklearn digits classification (10-way)")
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr = torch.from_numpy(X_tr); X_te = torch.from_numpy(X_te)
    y_tr = torch.from_numpy(y_tr); y_te = torch.from_numpy(y_te)
    common = dict(in_dim=64, hidden_dim=128, out_dim=10, depth=4)
    task3 = {}
    for label, make in [("BitMLP", lambda: BitMLP(**common)),
                         ("FPMLP",  lambda: FPMLP(**common)),
                         ("DSigma T=8",  lambda: DeltaSigmaMLP(**common, T=8)),
                         ("DSigma T=16", lambda: DeltaSigmaMLP(**common, T=16))]:
        m = make()
        loss, acc = quick_train_classification(m, X_tr, y_tr, X_te, y_te, epochs=200)
        task3[label] = {"loss": loss, "acc": acc}
        print(f"  {label:>12}: val acc = {acc:.4f}  (loss = {loss:.4f})")
    results["tasks"]["digits"] = task3

    # ---- Storage analysis ----
    print("\n--- Storage comparison (oscillator-shaped model) ---")
    torch.manual_seed(0)
    common_osc = dict(in_dim=3, hidden_dim=128, out_dim=1, depth=5)
    ds_model = DeltaSigmaMLP(**common_osc, T=8)
    quick_train_regression(ds_model,
                            *oscillator_normalize(oscillator_dataset(2000, seed=0)[0])[:1] +
                            (oscillator_dataset(2000, seed=0)[1],) +
                            tuple(oscillator_normalize(oscillator_dataset(500, seed=1)[0])[:1]) +
                            (oscillator_dataset(500, seed=1)[1],), epochs=20)
    # (small training just to populate weights)
    path = Path("/tmp/dsigma_bench.npz")
    breakdown = save_dsigma_mlp(ds_model, path)
    n_params = sum(p.numel() for p in ds_model.parameters())
    fp32_total = n_params * 4
    storage = {
        "n_params": n_params,
        "fp32_state_bytes": fp32_total,
        "ds_packed_bytes": breakdown["file_bytes_on_disk"],
        "compression": fp32_total / breakdown["file_bytes_on_disk"],
    }
    print(f"  fp32 state_dict bytes:    {fp32_total:>10,}")
    print(f"  DSigma T=8 packed bytes:  {breakdown['file_bytes_on_disk']:>10,}")
    print(f"  Compression:              {storage['compression']:>10.2f}x")
    os.unlink(path)
    results["storage"] = storage

    # ---- Hardware (already measured) ----
    results["hardware"] = {
        "ternary_pe": {"LCs": 283, "max_MHz": 16.83, "device": "iCE40 HX1K"},
        "dsigma_pe":  {"LCs": 206, "max_MHz": 75.08, "device": "iCE40 HX1K"},
        "dsigma_systolic_N8": {"LCs": 1862, "device": "iCE40 HX8K",
                               "LCs_per_PE": 1862 // 8},
    }
    print("\n--- Hardware (synthesized) ---")
    for name, h in results["hardware"].items():
        extras = ", ".join(f"{k}={v}" for k, v in h.items() if k not in ("LCs", "max_MHz"))
        freq = f"{h['max_MHz']} MHz" if "max_MHz" in h else "n/a"
        print(f"  {name:>25}: {h.get('LCs', 'n/a'):>5} LCs at {freq} ({extras})")

    # ---- Final cross-task summary ----
    print("\n" + "=" * 70)
    print("CROSS-TASK SUMMARY")
    print("=" * 70)
    print(f"\n{'task':>25}  {'BitMLP':>10}  {'FPMLP':>10}  {'DSigma 8':>10}  {'DSigma 16':>10}")
    for tname in ["oscillator", "schrodinger"]:
        t = results["tasks"][tname]
        print(f"{tname + ' val MSE':>25}  "
              f"{t['BitMLP']:>10.6f}  {t['FPMLP']:>10.6f}  "
              f"{t['DSigma T=8']:>10.6f}  {t['DSigma T=16']:>10.6f}")
    t = results["tasks"]["digits"]
    print(f"{'digits val acc':>25}  "
          f"{t['BitMLP']['acc']:>10.4f}  {t['FPMLP']['acc']:>10.4f}  "
          f"{t['DSigma T=8']['acc']:>10.4f}  {t['DSigma T=16']['acc']:>10.4f}")

    with open("final_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to final_benchmark.json")


if __name__ == "__main__":
    main()
