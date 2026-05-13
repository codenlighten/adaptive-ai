"""Multitask physics regression: one ternary model, three tasks.

Tasks:
  task_id=0: damped oscillator   (t, omega, zeta)              -> x(t)
  task_id=1: pendulum H          (q, p, 0)                     -> H(q, p)
  task_id=2: relativistic E      (m, p, 0)                     -> sqrt(p^2 + m^2)

Each row of training data is [task_id, arg_0, arg_1, arg_2] -> y.
We train one shared model on all three at once and then test
generalization to held-out parameter ranges per task.

Does ternary's sparsity prior produce sub-circuits specialized to each
task (the way sparse activations would in a real brain)? Or does it
average across tasks?

Run: venv/bin/python -m scripts.multitask
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from src.data import damped_oscillator
from src.hnn import pendulum_hamiltonian
from src.model import BitMLP, FPMLP


def _osc(t, omega, zeta):
    return damped_oscillator(torch.tensor([t]), torch.tensor([omega]),
                             torch.tensor([zeta])).item()


def _pend(q, p):
    return pendulum_hamiltonian(torch.tensor([q]), torch.tensor([p])).item()


def _rel(m, p):
    return math.sqrt(p ** 2 + m ** 2)


def make_multitask_dataset(n_per_task: int, seed: int = 0,
                           ranges: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a labeled dataset of three tasks.

    `ranges` overrides the default parameter ranges per task — used for
    held-out test sets that lie outside the training ranges.
    """
    g = torch.Generator().manual_seed(seed)
    defaults = {
        "osc": {"omega": (0.5, 3.0), "zeta": (0.05, 0.5), "t": (0.0, 10.0)},
        "pend": {"q": (-2.0, 2.0), "p": (-2.0, 2.0)},
        "rel": {"m": (0.5, 5.0), "p": (-5.0, 5.0)},
    }
    if ranges:
        for k, v in ranges.items():
            defaults[k].update(v)
    rng = defaults

    rows_x, rows_y = [], []
    # Task 0: oscillator
    for _ in range(n_per_task):
        omega = (torch.rand(1, generator=g).item()
                 * (rng["osc"]["omega"][1] - rng["osc"]["omega"][0]) + rng["osc"]["omega"][0])
        zeta = (torch.rand(1, generator=g).item()
                * (rng["osc"]["zeta"][1] - rng["osc"]["zeta"][0]) + rng["osc"]["zeta"][0])
        t = (torch.rand(1, generator=g).item()
             * (rng["osc"]["t"][1] - rng["osc"]["t"][0]) + rng["osc"]["t"][0])
        y = _osc(t, omega, zeta)
        rows_x.append([0.0, t, omega, zeta])
        rows_y.append([y])

    # Task 1: pendulum H
    for _ in range(n_per_task):
        q = (torch.rand(1, generator=g).item()
             * (rng["pend"]["q"][1] - rng["pend"]["q"][0]) + rng["pend"]["q"][0])
        p = (torch.rand(1, generator=g).item()
             * (rng["pend"]["p"][1] - rng["pend"]["p"][0]) + rng["pend"]["p"][0])
        y = _pend(q, p)
        rows_x.append([1.0, q, p, 0.0])
        rows_y.append([y])

    # Task 2: relativistic E
    for _ in range(n_per_task):
        m = (torch.rand(1, generator=g).item()
             * (rng["rel"]["m"][1] - rng["rel"]["m"][0]) + rng["rel"]["m"][0])
        p = (torch.rand(1, generator=g).item()
             * (rng["rel"]["p"][1] - rng["rel"]["p"][0]) + rng["rel"]["p"][0])
        y = _rel(m, p)
        rows_x.append([2.0, m, p, 0.0])
        rows_y.append([y])

    return torch.tensor(rows_x), torch.tensor(rows_y)


def per_task_mse(model, X, y):
    """MSE broken down by task id (col 0)."""
    model.eval()
    out = {}
    with torch.no_grad():
        pred = model(X)
        for tid in [0, 1, 2]:
            mask = X[:, 0] == tid
            if mask.sum() > 0:
                m = ((pred[mask] - y[mask]) ** 2).mean().item()
                out[tid] = m
    return out


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
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 8) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            print(f"  [{label}] ep {epoch+1:4d}/{epochs}  val={vl:.5f}")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")


def main():
    torch.manual_seed(0)
    print("Building 3-task multitask dataset (oscillator + pendulum + relativistic E)")
    X_train, y_train = make_multitask_dataset(2500, seed=0)
    X_val, y_val = make_multitask_dataset(500, seed=1)
    # Held-out OOD ranges per task
    X_ood, y_ood = make_multitask_dataset(500, seed=2, ranges={
        "osc": {"omega": (3.0, 4.0)},          # higher omega
        "pend": {"q": (-3.0, -2.0)},           # outside training q
        "rel": {"m": (5.0, 8.0)},              # heavier m
    })
    print(f"  train: {X_train.shape}  val: {X_val.shape}  ood: {X_ood.shape}\n")

    # Normalize using train stats (per-feature except task id which is categorical)
    mean = X_train[:, 1:].mean(0)
    std = X_train[:, 1:].std(0).clamp_min(1e-6)
    def normalize(X):
        Xn = X.clone()
        Xn[:, 1:] = (Xn[:, 1:] - mean) / std
        return Xn
    Xt_n = normalize(X_train); Xv_n = normalize(X_val); Xo_n = normalize(X_ood)

    bit = BitMLP(4, 128, 1, depth=5)
    fp = FPMLP(4, 128, 1, depth=5)
    print(f"BitMLP params: {sum(p.numel() for p in bit.parameters()):,}")
    print(f" FPMLP params: {sum(p.numel() for p in fp.parameters()):,}\n")

    print("=== Training BitMLP ===")
    train(bit, Xt_n, y_train, Xv_n, y_val, epochs=200, lr=2e-3, batch_size=256, label="bit")
    print("\n=== Training FPMLP ===")
    train(fp, Xt_n, y_train, Xv_n, y_val, epochs=200, lr=2e-3, batch_size=256, label=" fp")

    print("\n--- Validation MSE by task ---")
    bit_val = per_task_mse(bit, Xv_n, y_val)
    fp_val = per_task_mse(fp, Xv_n, y_val)
    print(f"  {'task':>15}  {'Bit':>10}  {'FP':>10}  {'Bit/FP':>8}")
    task_names = {0: "oscillator", 1: "pendulum H", 2: "relativistic E"}
    for tid in [0, 1, 2]:
        b = bit_val[tid]
        f = fp_val[tid]
        print(f"  {task_names[tid]:>15}  {b:>10.6f}  {f:>10.6f}  {b/max(f,1e-12):>7.2f}x")

    print("\n--- OOD MSE by task (held-out parameter ranges) ---")
    bit_ood = per_task_mse(bit, Xo_n, y_ood)
    fp_ood = per_task_mse(fp, Xo_n, y_ood)
    for tid in [0, 1, 2]:
        b = bit_ood[tid]
        f = fp_ood[tid]
        print(f"  {task_names[tid]:>15}  {b:>10.6f}  {f:>10.6f}  {b/max(f,1e-12):>7.2f}x")


if __name__ == "__main__":
    main()
