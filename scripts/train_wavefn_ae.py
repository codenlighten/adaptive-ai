"""Train the ternary cross-modal autoencoder on 1-D Schrödinger.

Run: venv/bin/python -m scripts.train_wavefn_ae --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from src.wavefn_ae import N_GRID, TernaryAE, make_dataset, make_grid


def train(model, V, psi, E, V_val, psi_val, E_val, epochs, lr, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()
    n = V.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            psi_p, E_p = model(V[idx])
            loss = mse(psi_p, psi[idx]) + 0.1 * mse(E_p, E[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                psi_v, E_v = model(V_val)
                psi_mse = mse(psi_v, psi_val).item()
                E_mae = (E_v - E_val).abs().mean().item()
            print(f"  ep {epoch+1:4d}/{epochs}  total={total/n:.5f}  "
                  f"psi_val_mse={psi_mse:.5f}  E_val_mae={E_mae:.5f}")
    print(f"done in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"Building dataset: {args.n_train} train + {args.n_val} val Schrodinger problems...")
    V_train, psi_train, E_train = make_dataset(args.n_train, seed=0)
    V_val, psi_val, E_val = make_dataset(args.n_val, seed=1)
    print(f"  V shape: {tuple(V_train.shape)}")
    print(f"  E range: [{E_train.min().item():.2f}, {E_train.max().item():.2f}]")
    print()

    model = TernaryAE(n_grid=N_GRID, hidden=128, latent=args.latent)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    train(model, V_train, psi_train, E_train,
          V_val, psi_val, E_val,
          args.epochs, args.lr, args.batch_size)

    model.eval()
    with torch.no_grad():
        psi_pred, E_pred = model(V_val)
        psi_mse = ((psi_pred - psi_val) ** 2).mean().item()
        psi_norms = (psi_pred ** 2).sum(dim=1) * ((2 * 5.0) / (N_GRID - 1))
        E_mae = (E_pred - E_val).abs().mean().item()

    print("\n=== Held-out reconstruction quality ===")
    print(f"  psi MSE:   {psi_mse:.6f}")
    print(f"  E0 MAE:    {E_mae:.4f}")
    print(f"  mean |psi|^2 normalization: {psi_norms.mean().item():.3f}  (1.0 = perfect)")

    if args.plot:
        import matplotlib.pyplot as plt
        x_grid = make_grid()
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for i, ax_pair in enumerate(zip(axes[0], axes[1])):
            ax_v, ax_psi = ax_pair
            idx = i
            ax_v.plot(x_grid, V_val[idx].numpy(), "k-", label="V(x)")
            ax_v.set_ylim(-3, 5)
            ax_v.set_title(f"V(x) (sample {idx})")
            ax_v.set_xlabel("x")
            ax_psi.plot(x_grid, psi_val[idx].numpy(), "k-", label="true")
            ax_psi.plot(x_grid, psi_pred[idx].numpy(), "b--", label="ternary AE")
            ax_psi.set_title(f"psi_0(x)  E0 true={E_val[idx].item():.2f}, "
                              f"pred={E_pred[idx].item():.2f}")
            ax_psi.set_xlabel("x")
            ax_psi.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("wavefn_ae_results.png", dpi=120)
        print("\nSaved wavefn_ae_results.png")


if __name__ == "__main__":
    main()
