"""Train ternary vs fp32 1-D diffusion on the double-well distribution.

We measure:
  - sample histograms (do both modes get covered?)
  - approximate KL divergence to the analytic target

The hard part: avoid mode collapse. Diffusion handles this well in
general; we test whether ternary preserves that property.

Run: venv/bin/python -m scripts.train_diffusion
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from src.diffusion import (
    DDPM1D,
    double_well_logp,
    make_fp_denoiser,
    make_ternary_denoiser,
    sample_double_well,
)


def train(model, ddpm: DDPM1D, data, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = data.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x0 = data[idx]
            t = torch.randint(0, ddpm.n_steps, (idx.shape[0],))
            noise = torch.randn_like(x0)
            xt = ddpm.q_sample(x0, t, noise)
            t_in = (t.unsqueeze(-1).float() / ddpm.n_steps)
            pred = model(xt, t_in)
            loss = loss_fn(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:4d}/{epochs}  loss={total/n:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")


def empirical_kl(samples: np.ndarray, T: float = 0.4, n_bins: int = 60,
                 x_range: tuple[float, float] = (-3.0, 3.0)) -> float:
    """Approximate KL(p_samples || p_target) via histograms.

    Lower is better. Adds tiny epsilon for stability.
    """
    edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    hist, _ = np.histogram(samples, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Analytic target density (normalized empirically)
    logp = -((centers ** 2 - 1.0) ** 2) / T
    p_target = np.exp(logp)
    p_target /= (p_target * (edges[1] - edges[0])).sum()

    eps = 1e-6
    p_samples = hist + eps
    p_samples /= (p_samples * (edges[1] - edges[0])).sum()
    p_target = p_target + eps
    p_target /= (p_target * (edges[1] - edges[0])).sum()

    kl = (p_samples * np.log(p_samples / p_target) * (edges[1] - edges[0])).sum()
    return float(kl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-data", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    print("Sampling target data (rejection)...")
    data = sample_double_well(args.n_data, T=0.4, seed=0)
    print(f"  data shape: {tuple(data.shape)}, mean={data.mean().item():.3f}, "
          f"std={data.std().item():.3f}\n")

    ddpm = DDPM1D(n_steps=100)

    bit = make_ternary_denoiser(hidden=args.hidden, depth=args.depth)
    fp = make_fp_denoiser(hidden=args.hidden, depth=args.depth)
    print(f"Ternary denoiser params: {sum(p.numel() for p in bit.parameters()):,}")
    print(f" FP denoiser params:     {sum(p.numel() for p in fp.parameters()):,}\n")

    train(bit, ddpm, data, args.epochs, args.lr, args.batch_size, "BitDiff")
    print()
    train(fp, ddpm, data, args.epochs, args.lr, args.batch_size, " FPDiff")

    print("\n--- Sampling from learned distributions ---")
    bit_samples = ddpm.sample(bit, args.n_samples).numpy().ravel()
    fp_samples = ddpm.sample(fp, args.n_samples).numpy().ravel()
    data_np = data.numpy().ravel()

    print(f"  Bit samples  : mean={bit_samples.mean():.3f}  std={bit_samples.std():.3f}  "
          f"frac(<0)={(bit_samples < 0).mean():.2%}")
    print(f"  FP  samples  : mean={fp_samples.mean():.3f}  std={fp_samples.std():.3f}  "
          f"frac(<0)={(fp_samples < 0).mean():.2%}")
    print(f"  truth        : mean={data_np.mean():.3f}  std={data_np.std():.3f}  "
          f"frac(<0)={(data_np < 0).mean():.2%}")
    print(f"  (for double-well target frac(<0) should be ~50%)")

    kl_bit = empirical_kl(bit_samples)
    kl_fp = empirical_kl(fp_samples)
    kl_data = empirical_kl(data_np)
    print(f"\n  KL(samples || target) — lower is better")
    print(f"    Bit:  {kl_bit:.4f}")
    print(f"    FP:   {kl_fp:.4f}")
    print(f"    data: {kl_data:.4f}  (the empirical-vs-target gap)")

    if args.plot:
        import matplotlib.pyplot as plt
        x = np.linspace(-3, 3, 400)
        p_target = np.exp(-((x ** 2 - 1.0) ** 2) / 0.4)
        p_target /= np.trapezoid(p_target, x)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(bit_samples, bins=60, density=True, alpha=0.4, label="Bit-Diff samples")
        ax.hist(fp_samples, bins=60, density=True, alpha=0.4, label="FP-Diff samples")
        ax.plot(x, p_target, "k-", linewidth=2, label="target (double-well)")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title("1-D ternary diffusion model — double-well distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig("diffusion_results.png", dpi=120)
        print("\nSaved diffusion_results.png")


if __name__ == "__main__":
    main()
