"""Anytime inference with delta-sigma weights.

The bit-stream representation has a useful property: the cumulative
average over the first k of T steps is a progressively better estimate
of the true effective weight. So at inference we can:

  1. Compute the matmul using just bits 1..2 of each weight.
  2. Check if the output is "stable enough".
  3. If yes, return; if no, accumulate bits 3..4 (only the new ones).
  4. Repeat with k doubling each time.

Easy inputs converge in 2-4 bits; hard inputs need 16-32. The
*average* compute is therefore well below T, while worst-case is T.

This is the genuinely novel mechanism: a runtime accuracy/compute knob,
controlled per-example, that uses no multiplications at any time step.

Run: venv/bin/python -m scripts.anytime_inference
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.dsigma_linear import DeltaSigmaMLP


def train(model, X, y, X_val, y_val, epochs=150, lr=2e-3, batch_size=256):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    print("Training DeltaSigmaMLP at T=32...")
    model = DeltaSigmaMLP(3, 128, 1, depth=5, T=32, order=1)
    t0 = time.time()
    train(model, Xtn, y_train, Xvn, y_val)
    print(f"  trained in {time.time()-t0:.1f}s")
    model.eval()

    # Full-T baseline
    with torch.no_grad():
        full_pred = model(Xvn)
        full_mse = ((full_pred - y_val) ** 2).mean().item()
    print(f"\nFull-T (T=32) val MSE: {full_mse:.6f}")

    print("\n=== Anytime inference per example ===")
    print("Run with progressively-doubled k until output change < stop_eps.")
    print()

    # Sweep stop_eps
    for stop_eps in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        ks_used = []
        preds = []
        for i in range(Xvn.shape[0]):
            x = Xvn[i:i+1]
            y_pred, k = model.anytime_inference(x, T_max=32, stop_eps=stop_eps)
            ks_used.append(k)
            preds.append(y_pred)
        preds = torch.cat(preds, dim=0)
        mse = ((preds - y_val) ** 2).mean().item()
        avg_k = sum(ks_used) / len(ks_used)
        max_k = max(ks_used)
        print(f"  stop_eps = {stop_eps:.3f}  | avg k = {avg_k:5.2f}  "
              f"| max k = {max_k:2d}  | val MSE = {mse:.6f}  "
              f"| compute reduction = {32 / avg_k:.2f}x")

    print()
    print("Bottom row reads: 'on average we used ~k of 32 time steps per input,")
    print("at the cost of stop_eps precision loss in the output.'")
    print()
    print("This is anytime inference: faster on easy examples, full T on hard ones,")
    print("controlled by the user at runtime via a single parameter.")


if __name__ == "__main__":
    main()
