"""End-to-end demo: train DSigma, pack the streams, run inference
with the packed format and anytime-mode k truncation.

Run: venv/bin/python -m scripts.run_dsigma_packed
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.dsigma_linear import DeltaSigmaMLP
from src.dsigma_pack import dsigma_inference, load_dsigma_arrays, save_dsigma_mlp


def train_quick(model, X, y, X_val, y_val, epochs=150, lr=2e-3, batch_size=256):
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
    print("=" * 70)
    print("1. Train DeltaSigmaMLP (T=8) on damped oscillator")
    print("=" * 70)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)
    model = DeltaSigmaMLP(3, 128, 1, depth=5, T=8, order=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    t0 = time.time()
    val = train_quick(model, Xtn, y_train, Xvn, y_val, epochs=150)
    print(f"  trained in {time.time()-t0:.1f}s, val MSE = {val:.6f}")

    print()
    print("=" * 70)
    print("2. Save with packed-trit streams")
    print("=" * 70)
    path = Path("dsigma_checkpoint.npz")
    breakdown = save_dsigma_mlp(model, path)
    for k, v in breakdown.items():
        if k.endswith("_bytes"):
            print(f"  {k:30s} {v:>10,} bytes ({v/1024:.2f} KB)")

    # Equivalent fp32 size (only counting the parameters)
    fp32_eq = n_params * 4
    print(f"\n  fp32 state_dict equivalent (all params as fp32): {fp32_eq:,} bytes "
          f"({fp32_eq/1024:.2f} KB)")
    print(f"  compression vs fp32: {fp32_eq / breakdown['file_bytes_on_disk']:.2f}x")

    print()
    print("=" * 70)
    print("3. Load via pure-NumPy packed-stream inference and verify")
    print("=" * 70)
    arrays = load_dsigma_arrays(path)
    np_out = dsigma_inference(arrays, Xvn.numpy(), k=None)
    with torch.no_grad():
        torch_out = model(Xvn).numpy()
    diff = np.abs(np_out - torch_out).max()
    print(f"  max abs diff vs torch model: {diff:.6e}")

    np_mse = ((np_out - y_val.numpy()) ** 2).mean()
    print(f"  torch val MSE:       {val:.6f}")
    print(f"  packed-NumPy val MSE: {float(np_mse):.6f}  (should match)")

    print()
    print("=" * 70)
    print("4. Anytime inference: vary k from 1 to T using the same packed file")
    print("=" * 70)
    for k in [1, 2, 4, 6, 8]:
        np_out_k = dsigma_inference(arrays, Xvn.numpy(), k=k)
        mse_k = ((np_out_k - y_val.numpy()) ** 2).mean()
        print(f"  k = {k}:  val MSE = {float(mse_k):.6f}")
    print()
    print("Every k is computed from the same packed file. The user chooses")
    print("k at runtime — small k for fast/coarse, larger k for accurate.")


if __name__ == "__main__":
    main()
