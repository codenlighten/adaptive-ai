"""End-to-end demo: train a BitMLP, save it with packed-trit weights,
load it via the pure-NumPy multiply-free inference engine, verify outputs
match the torch model, and report storage + op counts.

Run: venv/bin/python -m scripts.run_inference
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import save_bitmlp
from src.data import make_dataset, normalize
from src.inference import TernaryNet
from src.model import BitMLP


def train_quick(model, X, y, X_val, y_val, epochs=200, lr=3e-3, batch_size=256):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.MSELoss()
    n = X.shape[0]
    for epoch in range(epochs):
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
    print("1. Train a BitMLP on the damped oscillator (quick run)")
    print("=" * 70)
    X_train, y_train = make_dataset(4000, seed=0)
    X_val, y_val = make_dataset(1000, seed=1)
    X_train_n, stats = normalize(X_train)
    X_val_n, _ = normalize(X_val, stats)

    model = BitMLP(3, 128, 1, depth=5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    t0 = time.time()
    val_mse = train_quick(model, X_train_n, y_train, X_val_n, y_val, epochs=200)
    print(f"  trained in {time.time()-t0:.1f}s — torch val MSE = {val_mse:.6f}")

    print()
    print("=" * 70)
    print("2. Save with trit-packed hidden weights")
    print("=" * 70)
    ckpt_path = Path("checkpoint_packed.npz")
    breakdown = save_bitmlp(model, ckpt_path, norm_stats=stats)
    for k, v in breakdown.items():
        if k.endswith("_bytes"):
            print(f"  {k:30s} {v:>10,} bytes ({v/1024:.2f} KB)")

    # Equivalent fp32 checkpoint size for comparison
    fp32_total = 0
    for p in model.parameters():
        fp32_total += p.numel() * 4
    print(f"\n  Equivalent fp32 state_dict would be ~{fp32_total:,} bytes "
          f"({fp32_total/1024:.2f} KB)")
    print(f"  Packed file is {fp32_total / breakdown['file_bytes_on_disk']:.2f}x smaller.")

    print()
    print("=" * 70)
    print("3. Load via pure-NumPy multiply-free inference engine")
    print("=" * 70)
    net = TernaryNet(ckpt_path)
    print(f"  Loaded {len(net.arrays['layers'])} ternary layers + 2 fp boundary layers.")

    print()
    print("=" * 70)
    print("4. Verify NumPy inference matches torch BitMLP")
    print("=" * 70)
    model.eval()
    with torch.no_grad():
        torch_out = model(X_val_n).numpy()
    np_out = net.forward(X_val_n.numpy())
    diff = np.abs(torch_out - np_out)
    print(f"  max abs diff:  {diff.max():.6e}")
    print(f"  mean abs diff: {diff.mean():.6e}")
    rel = diff.max() / max(abs(torch_out).max(), 1e-9)
    print(f"  relative max diff: {rel:.6e}")
    if diff.max() < 1e-4:
        print("  -> outputs match within fp tolerance.")
    else:
        print("  -> WARNING: divergence larger than expected.")

    # Also confirm MSE against ground truth still holds
    np_mse = ((np_out - y_val.numpy()) ** 2).mean()
    print(f"\n  Torch BitMLP val MSE: {val_mse:.6f}")
    print(f"  NumPy multfree  val MSE: {float(np_mse):.6f}")

    print()
    print("=" * 70)
    print("5. Operation count from one full validation pass")
    print("=" * 70)
    net.reset_counter()
    _ = net.forward(X_val_n.numpy())
    c = net.counter
    print(f"  Matmul fp multiplies AVOIDED:    {c.matmul_multiplies_avoided:>14,}")
    print(f"  Matmul signed adds performed:    {c.matmul_signed_adds:>14,}")
    print(f"    -> matmul multiplies actually performed: 0")
    print()
    print(f"  fp boundary projection multiplies: {c.fp_boundary_multiplies:>12,}")
    print(f"  scalar rescale + layernorm multiplies: {c.scalar_multiplies:>8,}")
    print(f"  GELU multiplies:                   {c.other_multiplies:>12,}")
    print(f"  TOTAL fp multiplies actually done: {c.total_real_multiplies():>12,}")
    print()
    equivalent_fp32_matmul_mults = c.matmul_multiplies_avoided
    print(f"  An equivalent fp32 model would do "
          f"{c.matmul_multiplies_avoided + c.fp_boundary_multiplies:,} matmul multiplies.")
    print(f"  We did {c.fp_boundary_multiplies:,} (only the boundary projections).")
    print(f"  Reduction in matmul multiplies: "
          f"{1 - c.fp_boundary_multiplies/(c.matmul_multiplies_avoided + c.fp_boundary_multiplies):.1%}")


if __name__ == "__main__":
    main()
