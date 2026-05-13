"""Train a BitMLP variant with learnable activation precision.

Activations are uniformly quantized to n levels per layer, where n is
learned. We add a small regularizer pushing n DOWN (lower bit-width is
preferred) so the model only spends activation precision where it
actually helps.

Run: venv/bin/python -m scripts.train_act_prec
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.learned_act_prec import ActQuantizedMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, prec_reg_weight=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            data_loss = loss_fn(pred, y[idx])
            # Regularizer: prefer fewer activation levels (log to make the cost grow slowly)
            prec_penalty = sum(
                torch.log(m.n_levels())
                for m in model.net
                if hasattr(m, "n_levels")
            )
            (data_loss + prec_reg_weight * prec_penalty).backward()
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            levels = model.n_levels_per_layer()
            print(f"  ep {epoch+1:4d}/{epochs}  val={vl:.6f}  "
                  f"act levels = {[f'{l:.1f}' for l in levels]}")
            history.append((epoch + 1, vl, levels))
    print(f"done in {time.time()-t0:.1f}s")
    return history


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    print("Activation-precision-learned BitMLP, init = 16 levels (~4 bits)")
    model = ActQuantizedMLP(3, 128, 1, depth=5, init_levels=16.0)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256,
          prec_reg_weight=5e-5)

    print("\n--- Final activation level counts per layer ---")
    for i, lvl in enumerate(model.n_levels_per_layer()):
        bits = math.log2(lvl)
        print(f"  layer {i}: {lvl:.2f} levels  (~{bits:.2f} bits)")

    model.eval()
    with torch.no_grad():
        mse = ((model(Xvn) - y_val) ** 2).mean().item()
    print(f"\nFinal val MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
