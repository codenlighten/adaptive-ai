"""Ternary LoRA demo.

Setup: train a small fp32 MLP on the 'general' physics task (damped
oscillator). Then domain-shift: freeze the base, attach ternary LoRA
adapters at each hidden layer, train only the LoRA adapters to fit a
shifted task (e.g., a different parameter range).

We show that:
  1. The base model alone does badly on the new distribution.
  2. Adding ternary LoRA adapters recovers most of the gap.
  3. The number of trainable parameters added is tiny (rank=4-8).

Run: venv/bin/python -m scripts.lora_demo
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import damped_oscillator, normalize
from src.lora import TernaryLoRA


def _make_distribution(n, seed, omega_range, zeta_range, t_range):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * (t_range[1] - t_range[0]) + t_range[0]
    omega = torch.rand(n, generator=g) * (omega_range[1] - omega_range[0]) + omega_range[0]
    zeta = torch.rand(n, generator=g) * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
    x = damped_oscillator(t, omega, zeta)
    return torch.stack([t, omega, zeta], dim=1), x.unsqueeze(1)


def make_base_data(n, seed):
    # "general" distribution
    return _make_distribution(n, seed,
                              omega_range=(0.5, 3.0),
                              zeta_range=(0.05, 0.50),
                              t_range=(0.0, 10.0))


def make_shifted_data(n, seed):
    # domain-shift: smaller omega, longer times
    return _make_distribution(n, seed,
                              omega_range=(0.2, 0.7),
                              zeta_range=(0.01, 0.10),
                              t_range=(8.0, 20.0))


class FPMLPSequential(nn.Module):
    """Simple fp32 MLP — explicit Linear layers so we can attach LoRA."""

    def __init__(self, in_dim, hidden_dim, out_dim, depth=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 2)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc in self.fcs:
            x = self.act(fc(x))
        return self.fc_out(x)


class LoRAWrapped(nn.Module):
    """Same as FPMLPSequential but with TernaryLoRA on every hidden Linear."""

    def __init__(self, base: FPMLPSequential, rank=8, alpha=16.0):
        super().__init__()
        self.fc_in = base.fc_in  # leave entry projection frozen
        for p in self.fc_in.parameters():
            p.requires_grad = False
        self.fcs = nn.ModuleList([
            TernaryLoRA.from_linear(fc, rank=rank, alpha=alpha)
            for fc in base.fcs
        ])
        self.fc_out = base.fc_out
        for p in self.fc_out.parameters():
            p.requires_grad = False
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc in self.fcs:
            x = self.act(fc(x))
        return self.fc_out(x)


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label,
          parameters=None):
    if parameters is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
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
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(X_val), y_val).item()
            print(f"  [{label}] ep {epoch+1:4d}/{epochs}  val={vl:.5f}")
    print(f"  [{label}] done in {time.time()-t0:.1f}s")


def main():
    torch.manual_seed(0)

    # === Phase 1: pretrain base on "general" distribution ===
    X_base, y_base = make_base_data(8000, seed=0)
    X_base_n, stats = normalize(X_base)
    Xv_base, yv_base = make_base_data(1000, seed=1)
    Xv_base_n, _ = normalize(Xv_base, stats)

    base = FPMLPSequential(3, 128, 1, depth=5)
    print("Phase 1: pretrain fp32 base on 'general' physics distribution")
    train(base, X_base_n, y_base, Xv_base_n, yv_base,
          epochs=150, lr=3e-3, batch_size=256, label="pre")

    base.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        base_on_base = loss_fn(base(Xv_base_n), yv_base).item()

    # === Phase 2: evaluate base on shifted distribution ===
    X_sh, y_sh = make_shifted_data(4000, seed=2)
    X_sh_n, _ = normalize(X_sh, stats)  # use base's normalization stats
    Xv_sh, yv_sh = make_shifted_data(500, seed=3)
    Xv_sh_n, _ = normalize(Xv_sh, stats)
    with torch.no_grad():
        base_on_shift = loss_fn(base(Xv_sh_n), yv_sh).item()
    print(f"\nBase model:")
    print(f"  val MSE on 'general' distribution: {base_on_base:.5f}")
    print(f"  val MSE on shifted distribution:   {base_on_shift:.5f}")

    # === Phase 3: attach ternary LoRA and adapt on shifted data ===
    print("\nPhase 2: attach ternary LoRA adapters, fit only LoRA params on shifted data")
    lora = LoRAWrapped(base, rank=8, alpha=16.0)
    base_n_params = sum(p.numel() for p in base.parameters())
    lora_trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    print(f"  base params: {base_n_params:,}")
    print(f"  LoRA trainable params: {lora_trainable:,} "
          f"({100 * lora_trainable / base_n_params:.2f}% of base)")

    train(lora, X_sh_n, y_sh, Xv_sh_n, yv_sh,
          epochs=200, lr=3e-3, batch_size=256, label="lora")

    with torch.no_grad():
        lora_on_shift = loss_fn(lora(Xv_sh_n), yv_sh).item()
        lora_on_base = loss_fn(lora(Xv_base_n), yv_base).item()

    print(f"\nAfter ternary-LoRA adaptation:")
    print(f"  val MSE on shifted distribution:   {lora_on_shift:.5f}  "
          f"(was {base_on_shift:.5f}; improvement {100*(1 - lora_on_shift/base_on_shift):.1f}%)")
    print(f"  val MSE on 'general' distribution: {lora_on_base:.5f}  "
          f"(base was {base_on_base:.5f})")

    print(f"\nLoRA adapter storage as packed trits: ~{lora_trainable // 5} bytes "
          f"({lora_trainable // 5 / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
