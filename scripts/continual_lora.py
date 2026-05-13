"""Continual learning with sequential ternary LoRA adapters.

We pretrain a fp32 base model on a 'general' damped oscillator
distribution. Then we sequentially train three ternary LoRA adapters
on three distinct domain shifts:

  Adapter A: low omega range
  Adapter B: high zeta range
  Adapter C: long time range

For each adapter, we freeze the base + earlier adapters and train only
the current one. After all three are trained we evaluate on each domain
to measure catastrophic forgetting: does the network still do well on
A and B after fitting C?

Comparison:
  - Sequential LoRA (one adapter per task, switched in at eval time)
  - Single shared LoRA trained jointly on all three tasks
  - Base model alone (no adaptation)

Run: venv/bin/python -m scripts.continual_lora
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from src.data import damped_oscillator, normalize
from src.lora import TernaryLoRA


def make_dist(n, seed, omega_range, zeta_range, t_range):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * (t_range[1] - t_range[0]) + t_range[0]
    omega = torch.rand(n, generator=g) * (omega_range[1] - omega_range[0]) + omega_range[0]
    zeta = torch.rand(n, generator=g) * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
    x = damped_oscillator(t, omega, zeta)
    return torch.stack([t, omega, zeta], dim=1), x.unsqueeze(1)


def make_general(n, seed):
    return make_dist(n, seed, (0.5, 3.0), (0.05, 0.5), (0.0, 10.0))


def make_task_A(n, seed):
    return make_dist(n, seed, (0.2, 0.7), (0.05, 0.5), (0.0, 10.0))


def make_task_B(n, seed):
    return make_dist(n, seed, (0.5, 3.0), (0.4, 0.8), (0.0, 10.0))


def make_task_C(n, seed):
    return make_dist(n, seed, (0.5, 3.0), (0.05, 0.5), (8.0, 20.0))


class BaseMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=5):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden)
        self.fcs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth - 2)])
        self.fc_out = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for fc in self.fcs:
            x = self.act(fc(x))
        return self.fc_out(x)


class WrappedWithLoRA(nn.Module):
    """Base with LoRA on the hidden Linears. Lora may be 'off' (passthrough)."""

    def __init__(self, base: BaseMLP, rank=8, alpha=16.0):
        super().__init__()
        self.fc_in = base.fc_in
        for p in self.fc_in.parameters():
            p.requires_grad = False
        self.fc_out = base.fc_out
        for p in self.fc_out.parameters():
            p.requires_grad = False
        self.loras = nn.ModuleList([
            TernaryLoRA.from_linear(fc, rank=rank, alpha=alpha)
            for fc in base.fcs
        ])
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc_in(x))
        for lora in self.loras:
            x = self.act(lora(x))
        return self.fc_out(x)


def train_one(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for _ in range(epochs):
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
        v = loss_fn(model(X_val), y_val).item()
    print(f"  [{label}] done in {time.time()-t0:.1f}s  val MSE = {v:.5f}")
    return v


def evaluate_all_domains(model, datasets):
    """Evaluate model on each test set. Returns dict name -> MSE."""
    model.eval()
    out = {}
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for name, (X, y) in datasets.items():
            out[name] = loss_fn(model(X), y).item()
    return out


def lora_state_dict(model):
    return {f"loras.{i}.lora_A": l.lora_A.detach().clone()
            for i, l in enumerate(model.loras)} | \
           {f"loras.{i}.lora_B": l.lora_B.detach().clone()
            for i, l in enumerate(model.loras)}


def install_lora(model, state):
    for i, l in enumerate(model.loras):
        l.lora_A.data.copy_(state[f"loras.{i}.lora_A"])
        l.lora_B.data.copy_(state[f"loras.{i}.lora_B"])


def reset_lora(model):
    for l in model.loras:
        torch.nn.init.kaiming_uniform_(l.lora_A, a=5**0.5)
        l.lora_B.data.zero_()


def main():
    torch.manual_seed(0)

    # === 1. pretrain base on general distribution ===
    X_g, y_g = make_general(8000, 0)
    Xv_g, yv_g = make_general(500, 1)
    Xg_n, stats = normalize(X_g)
    Xv_g_n, _ = normalize(Xv_g, stats)

    base = BaseMLP(3, 128, 1, depth=5)
    print("Pretrain base (fp32) on general distribution")
    train_one(base, Xg_n, y_g, Xv_g_n, yv_g, epochs=150, lr=3e-3, batch_size=256, label="base")

    # Build datasets
    domains = {}
    for name, fn in [("A_lowOmega", make_task_A),
                     ("B_highZeta", make_task_B),
                     ("C_longTime", make_task_C),
                     ("G_general",  make_general)]:
        X, y = fn(500, seed=42 + hash(name) % 100)
        X_n = (X - stats["mean"]) / stats["std"]
        domains[name] = (X_n, y)

    # Evaluate base on each
    print("\nBase model performance on each domain:")
    base_scores = evaluate_all_domains(base, domains)
    for k, v in base_scores.items():
        print(f"  {k}: MSE = {v:.5f}")

    # === 2. Sequential LoRA adapters (continual) ===
    print("\n=== Sequential continual LoRA ===")
    lora_model = WrappedWithLoRA(base, rank=8, alpha=16.0)
    adapter_states = {}
    for tname, fn in [("A_lowOmega", make_task_A),
                      ("B_highZeta", make_task_B),
                      ("C_longTime", make_task_C)]:
        reset_lora(lora_model)
        X_t, y_t = fn(4000, seed=hash(tname) % 1000)
        Xt_n = (X_t - stats["mean"]) / stats["std"]
        Xv_t, yv_t = fn(500, seed=42 + hash(tname) % 100)
        Xv_t_n = (Xv_t - stats["mean"]) / stats["std"]
        print(f"\n  Train adapter {tname}")
        train_one(lora_model, Xt_n, y_t, Xv_t_n, yv_t,
                  epochs=120, lr=3e-3, batch_size=256, label=tname)
        adapter_states[tname] = lora_state_dict(lora_model)

    # Evaluate each adapter on every domain (switch adapter in, then eval)
    print("\nPer-adapter evaluation (switching adapters in at test time):")
    print(f"{'adapter':>14}  {'A':>10}  {'B':>10}  {'C':>10}  {'G':>10}")
    for tname in ["A_lowOmega", "B_highZeta", "C_longTime"]:
        install_lora(lora_model, adapter_states[tname])
        scores = evaluate_all_domains(lora_model, domains)
        print(f"{tname:>14}  {scores['A_lowOmega']:>10.5f}  "
              f"{scores['B_highZeta']:>10.5f}  {scores['C_longTime']:>10.5f}  "
              f"{scores['G_general']:>10.5f}")

    # === 3. Single shared adapter trained on all three (multitask) ===
    print("\n=== Single shared LoRA trained on union of A+B+C ===")
    reset_lora(lora_model)
    X_union = torch.cat([fn(2000, seed=hash(t) % 1000)[0] for t, fn in
                         [("A_lowOmega", make_task_A),
                          ("B_highZeta", make_task_B),
                          ("C_longTime", make_task_C)]])
    y_union = torch.cat([fn(2000, seed=hash(t) % 1000)[1] for t, fn in
                         [("A_lowOmega", make_task_A),
                          ("B_highZeta", make_task_B),
                          ("C_longTime", make_task_C)]])
    Xun = (X_union - stats["mean"]) / stats["std"]
    train_one(lora_model, Xun, y_union, Xv_g_n, yv_g,
              epochs=200, lr=3e-3, batch_size=256, label="multitask")
    shared_scores = evaluate_all_domains(lora_model, domains)
    print("Shared adapter performance:")
    print(f"  A: {shared_scores['A_lowOmega']:.5f}  "
          f"B: {shared_scores['B_highZeta']:.5f}  "
          f"C: {shared_scores['C_longTime']:.5f}  "
          f"G: {shared_scores['G_general']:.5f}")


if __name__ == "__main__":
    main()
