"""Compare STE-based training to discrete coordinate descent on a
problem where the truth is exactly representable as a ternary combo
of features.

Setup: we generate 16 features x_i ~ N(0, 1), define a ground-truth weight
vector w* in {-1, 0, +1}^16 with exactly 6 nonzero entries, and set
y = alpha* * x . w*  + small noise. The honest test for ternary training
is: can it recover w* exactly?
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.discrete_search import discrete_coordinate_descent
from src.ternary import ste_ternarize


def main():
    rng = np.random.default_rng(0)
    n, d = 600, 16
    X = rng.standard_normal((n, d))

    # Sparse ternary ground truth: 6 nonzero out of 16.
    w_true = np.zeros(d, dtype=np.int64)
    nz_idx = rng.choice(d, 6, replace=False)
    w_true[nz_idx] = rng.choice([-1, 1], 6)
    alpha_true = 0.7
    noise = 0.01 * rng.standard_normal(n)
    y = alpha_true * (X @ w_true.astype(float)) + noise

    print(f"Ground-truth w (alpha={alpha_true}):  {w_true.tolist()}")
    print(f"Nonzero indices: {sorted(nz_idx.tolist())}\n")

    # --- (A) STE-based training (PyTorch) ---
    print("=" * 70)
    print("A. STE-based ternary linear regression")
    print("=" * 70)
    torch.manual_seed(0)
    w_param = nn.Parameter(torch.randn(d) * 0.1)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()
    opt = torch.optim.AdamW([w_param], lr=2e-2, weight_decay=0.0)
    t0 = time.time()
    for step in range(2000):
        opt.zero_grad()
        w_q = ste_ternarize(w_param)
        pred = Xt @ w_q
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    ste_time = time.time() - t0

    # Extract final ternary weights
    from src.ternary import ternarize
    w_q_final, alpha_ste = ternarize(w_param.detach())
    w_ste = w_q_final.int().tolist()
    print(f"  STE recovered w:    {w_ste}")
    print(f"  STE alpha:          {float(alpha_ste):.4f}")
    print(f"  Matches truth:      {w_ste == w_true.tolist()}")
    print(f"  Hamming distance:   {sum(int(a != b) for a, b in zip(w_ste, w_true.tolist()))}")
    print(f"  training time:      {ste_time:.2f}s")

    # --- (B) Discrete coordinate descent ---
    print("\n" + "=" * 70)
    print("B. Discrete coordinate descent (no STE)")
    print("=" * 70)
    rng2 = np.random.default_rng(0)
    np.random.seed(0)
    t0 = time.time()
    w_dcd, alpha_dcd, losses = discrete_coordinate_descent(X, y, max_passes=50)
    dcd_time = time.time() - t0
    print(f"  DCD recovered w:    {w_dcd.tolist()}")
    print(f"  DCD alpha:          {alpha_dcd:.4f}")
    print(f"  Matches truth:      {w_dcd.tolist() == w_true.tolist()}")
    print(f"  Hamming distance:   {sum(int(a != b) for a, b in zip(w_dcd.tolist(), w_true.tolist()))}")
    print(f"  passes used:        {len(losses) - 1}")
    print(f"  training time:      {dcd_time:.2f}s")

    # --- compare final losses on the data
    pred_ste = float(alpha_ste) * (X @ np.array(w_ste, dtype=float))
    pred_dcd = alpha_dcd * (X @ w_dcd.astype(float))
    print("\n=== Final fit quality (data MSE) ===")
    print(f"  STE: {((pred_ste - y) ** 2).mean():.6f}")
    print(f"  DCD: {((pred_dcd - y) ** 2).mean():.6f}")
    print(f"  best possible (truth, noise-only): {(noise ** 2).mean():.6f}")


if __name__ == "__main__":
    main()
