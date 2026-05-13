"""Train spiking ternary MLP on damped oscillator. Compare to stochastic
ternary MLP from phase 12 (both feed-forward trit-flow models).

Run: venv/bin/python -m scripts.train_spiking
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.spiking import SpikingTernaryMLP
from src.stochastic_mlp import StochasticTritMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            print(f"  [{label}] ep {epoch+1:4d}/{epochs}  val={vl:.5f}")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    print("Spiking ternary MLP (T=8 timesteps, LIF dynamics)")
    spike = SpikingTernaryMLP(3, 128, 1, depth=5, T=8, decay=0.9, theta=0.5)
    print(f"  params: {sum(p.numel() for p in spike.parameters()):,}")
    train(spike, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=128, label="spike")

    print("\nStochastic ternary MLP (baseline from phase 12)")
    stoch = StochasticTritMLP(3, 128, 1, depth=5)
    print(f"  params: {sum(p.numel() for p in stoch.parameters()):,}")
    train(stoch, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256, label="stoch")

    loss_fn = nn.MSELoss()
    spike.eval()
    stoch.eval()
    with torch.no_grad():
        spike_mse = loss_fn(spike(Xvn), y_val).item()
        stoch_mse = loss_fn(stoch(Xvn), y_val).item()

    print("\n=== Final comparison ===")
    print(f"  Spiking ternary  (T=8 LIF):       val MSE = {spike_mse:.6f}")
    print(f"  Stochastic ternary (single pass): val MSE = {stoch_mse:.6f}")


if __name__ == "__main__":
    main()
