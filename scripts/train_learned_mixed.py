"""Train a mixed-precision MLP that learns which layer should be binary/ternary/quaternary.

Run: venv/bin/python -m scripts.train_learned_mixed
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.learned_mixed import MixedPrecisionMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    history = []
    for epoch in range(epochs):
        model.train()
        # Anneal tau from 5.0 down to 0.5
        frac = epoch / max(1, epochs - 1)
        model.set_tau(5.0 * (1.0 - frac) + 0.5 * frac)

        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            assignment = model.assignment()
            print(f"  ep {epoch+1:4d}/{epochs}  val={vl:.6f}  tau={model.net[2].tau:.2f}  "
                  f"assignment={assignment}")
            history.append((epoch + 1, vl, assignment))
    return history


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    model = MixedPrecisionMLP(3, 128, 1, depth=5)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}\n")

    print("Training with annealed Gumbel-softmax tau...")
    t0 = time.time()
    history = train(model, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256)
    print(f"done in {time.time()-t0:.1f}s")

    print("\n--- Final assignment probabilities per layer ---")
    for i, probs in enumerate(model.assignment_probs()):
        bar = ""
        for name, p in probs.items():
            bar += f"{name[0]}={p:.2f} "
        print(f"  layer {i}: {bar}")

    print(f"\nFinal hard assignment: {model.assignment()}")
    model.eval()
    with torch.no_grad():
        final_mse = ((model(Xvn) - y_val) ** 2).mean().item()
    print(f"Hard-assignment val MSE: {final_mse:.6f}")


if __name__ == "__main__":
    main()
