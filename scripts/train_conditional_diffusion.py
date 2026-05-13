"""Train conditional ternary diffusion across temperatures.

We train on T in {0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0} and then sample
the model at *held-out* temperatures (e.g., T=0.5, 1.2) to test
interpolation in the conditioning space.

Run: venv/bin/python -m scripts.train_conditional_diffusion --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from src.conditional_diffusion import (
    double_well_logp,
    make_conditional_fp,
    make_conditional_ternary,
    make_dataset,
)
from src.diffusion import DDPM1D


def train(model, ddpm, X, T, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x0 = X[idx]
            T_b = T[idx]
            t = torch.randint(0, ddpm.n_steps, (idx.shape[0],))
            noise = torch.randn_like(x0)
            xt = ddpm.q_sample(x0, t, noise)
            t_in = t.unsqueeze(-1).float() / ddpm.n_steps
            pred = model(xt, t_in, T_b)
            loss = loss_fn(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  loss={total/n:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")


@torch.no_grad()
def sample_conditional(model, ddpm, n: int, T_val: float) -> torch.Tensor:
    x = torch.randn(n, 1)
    T_t = torch.full((n, 1), T_val)
    for t in reversed(range(ddpm.n_steps)):
        t_tensor = torch.full((n, 1), t / ddpm.n_steps)
        eps_pred = model(x, t_tensor, T_t)
        a = ddpm.alphas[t]
        ab = ddpm.alpha_bar[t]
        mean = (1.0 / a.sqrt()) * (x - (1.0 - a) / (1.0 - ab).sqrt() * eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = ddpm.betas[t].sqrt()
            x = mean + sigma * noise
        else:
            x = mean
    return x


def empirical_kl(samples: np.ndarray, T: float, n_bins: int = 50,
                 x_range: tuple[float, float] = (-3.0, 3.0)) -> float:
    edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    hist, _ = np.histogram(samples, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    logp = -((centers ** 2 - 1.0) ** 2) / T
    p_target = np.exp(logp)
    p_target /= (p_target * (edges[1] - edges[0])).sum()
    eps = 1e-6
    p_samples = hist + eps
    p_samples /= (p_samples * (edges[1] - edges[0])).sum()
    p_target = p_target + eps
    p_target /= (p_target * (edges[1] - edges[0])).sum()
    return float((p_samples * np.log(p_samples / p_target) * (edges[1] - edges[0])).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--n-per-T", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    train_Ts = [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    held_out_Ts = [0.5, 1.2]  # interpolation test

    print(f"Building dataset across T = {train_Ts}...")
    X, T = make_dataset(args.n_per_T, train_Ts, seed=0)
    print(f"  total samples: {X.shape[0]:,}\n")

    ddpm = DDPM1D(n_steps=100)
    bit = make_conditional_ternary(hidden=96, depth=4)
    fp = make_conditional_fp(hidden=96, depth=4)
    print(f"Bit denoiser: {sum(p.numel() for p in bit.parameters()):,} params")
    print(f"FP denoiser:  {sum(p.numel() for p in fp.parameters()):,} params\n")

    train(bit, ddpm, X, T, args.epochs, args.lr, args.batch_size, "BitCD")
    print()
    train(fp, ddpm, X, T, args.epochs, args.lr, args.batch_size, " FPCD")

    print("\n--- Evaluation: KL to target at each T (lower better) ---")
    all_Ts = train_Ts + held_out_Ts
    rows = []
    for T_val in sorted(all_Ts):
        bit_samples = sample_conditional(bit, ddpm, args.n_samples, T_val).numpy().ravel()
        fp_samples = sample_conditional(fp, ddpm, args.n_samples, T_val).numpy().ravel()
        kl_bit = empirical_kl(bit_samples, T_val)
        kl_fp = empirical_kl(fp_samples, T_val)
        held = " (HELD OUT)" if T_val in held_out_Ts else ""
        print(f"  T={T_val:5.2f}  Bit KL={kl_bit:.4f}  FP KL={kl_fp:.4f}{held}")
        rows.append((T_val, kl_bit, kl_fp, bit_samples, fp_samples))

    if args.plot:
        import matplotlib.pyplot as plt
        n_show = min(6, len(rows))
        rows_plot = rows[:n_show]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, (T_val, kl_bit, kl_fp, bit_samples, fp_samples) in enumerate(rows_plot):
            ax = axes[i // 3][i % 3]
            x = np.linspace(-3, 3, 400)
            p_target = np.exp(-((x ** 2 - 1) ** 2) / T_val)
            p_target /= np.trapezoid(p_target, x)
            ax.hist(bit_samples, bins=50, density=True, alpha=0.4, color="blue", label="Bit")
            ax.hist(fp_samples, bins=50, density=True, alpha=0.4, color="orange", label="FP")
            ax.plot(x, p_target, "k-", linewidth=2, label="target")
            held = "  (HELD OUT)" if T_val in held_out_Ts else ""
            ax.set_title(f"T = {T_val:.2f}{held}\nBit KL={kl_bit:.3f}  FP KL={kl_fp:.3f}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("conditional_diffusion_results.png", dpi=120)
        print("\nSaved conditional_diffusion_results.png")


if __name__ == "__main__":
    main()
