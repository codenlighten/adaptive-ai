"""Deterministic multi-seed benchmark suite for delta-sigma networks.

Writes a single CSV (`benchmarks/results.csv`) covering:

  - Regression tasks: damped oscillator, Schrödinger E0
  - Classification: sklearn digits
  - Models: fp32 MLP, BitNet b1.58 (ternary), INT8 dynamic quant,
            DeltaSigma T=8 and T=16
  - Anytime characterization: for each DSigma model, val metric at
    k = 1, 2, 4, 8, 16 (capped at T)
  - Multi-seed: default 3 seeds, mean/std rendered downstream

Run:
    python -m scripts.benchmark_suite                  # all tasks, 3 seeds
    python -m scripts.benchmark_suite --seeds 0,1,2,3,4
    python -m scripts.benchmark_suite --tasks digits   # subset

Output: benchmarks/results.csv (long-form, one row per evaluation).
Use scripts/render_leaderboard.py to render the figure.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="torch.ao.quantization")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from delta_sigma_nn import DeltaSigmaMLP, dsigma_inference_context
from src.data import make_dataset as oscillator_dataset
from src.data import normalize as oscillator_normalize
from src.model import BitMLP, FPMLP
from src.schrodinger import make_dataset as schrodinger_dataset
from src.schrodinger import normalize as schrodinger_normalize


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_regression(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor,
                     epochs: int = 150, batch_size: int = 256,
                     lr: float = 2e-3) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return loss_fn(model(X_val), y_val).item()


def train_classification(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                         X_val: torch.Tensor, y_val: torch.Tensor,
                         epochs: int = 200, batch_size: int = 64,
                         lr: float = 2e-3) -> tuple[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        loss = F.cross_entropy(logits, y_val).item()
        acc = (logits.argmax(dim=-1) == y_val).float().mean().item()
    return loss, acc


def eval_regression(model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return nn.MSELoss()(model(X_val), y_val).item()


def eval_classification(model: nn.Module, X_val: torch.Tensor,
                        y_val: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(X_val).argmax(dim=-1) == y_val).float().mean().item()


def int8_dynamic(model: nn.Module) -> nn.Module:
    """PyTorch dynamic INT8 quantization on Linear layers."""
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8,
    )


def state_bytes(model: nn.Module) -> int:
    """Approximate state_dict size in bytes (fp32 reference)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_task(task_name: str, *, in_dim: int, out_dim: int, hidden_dim: int,
             depth: int, X_train: torch.Tensor, y_train: torch.Tensor,
             X_val: torch.Tensor, y_val: torch.Tensor,
             is_classification: bool, seeds: list[int],
             writer, epochs: int) -> None:
    """Run all models on one task across all seeds; write rows to CSV writer."""
    common = dict(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, depth=depth)
    metric = "acc" if is_classification else "mse"

    def write_row(model_name, T, k, seed, value, t_train, n_params, sb):
        writer.writerow({
            "task": task_name, "model": model_name,
            "T": "" if T is None else T,
            "k": "" if k is None else k,
            "seed": seed, "metric": metric, "value": f"{value:.6f}",
            "train_seconds": f"{t_train:.3f}",
            "params": n_params, "state_bytes": sb,
        })

    for seed in seeds:
        # ---- fp32 ----
        seed_everything(seed)
        m_fp = FPMLP(**common)
        t0 = time.perf_counter()
        if is_classification:
            _, acc = train_classification(m_fp, X_train, y_train, X_val, y_val, epochs=epochs)
            v_fp = acc
        else:
            v_fp = train_regression(m_fp, X_train, y_train, X_val, y_val, epochs=epochs)
        t_fp = time.perf_counter() - t0
        write_row("fp32", None, None, seed, v_fp, t_fp, count_params(m_fp), state_bytes(m_fp))

        # ---- INT8 dynamic (post-hoc on fp32 model) ----
        try:
            m_int8 = int8_dynamic(m_fp)
            if is_classification:
                v_int8 = eval_classification(m_int8, X_val, y_val)
            else:
                v_int8 = eval_regression(m_int8, X_val, y_val)
            sb_int8 = state_bytes(m_fp) // 4  # rough INT8 estimate
            write_row("int8", None, None, seed, v_int8, 0.0, count_params(m_fp), sb_int8)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] INT8 skipped on {task_name} seed={seed}: {exc}")

        # ---- BitNet b1.58 ternary ----
        seed_everything(seed)
        m_bit = BitMLP(**common)
        t0 = time.perf_counter()
        if is_classification:
            _, acc = train_classification(m_bit, X_train, y_train, X_val, y_val, epochs=epochs)
            v_bit = acc
        else:
            v_bit = train_regression(m_bit, X_train, y_train, X_val, y_val, epochs=epochs)
        t_bit = time.perf_counter() - t0
        write_row("bitnet", None, None, seed, v_bit, t_bit, count_params(m_bit), state_bytes(m_bit))

        # ---- DSigma at T = 8, 16 ----
        for T in [8, 16]:
            seed_everything(seed)
            m_ds = DeltaSigmaMLP(**common, T=T)
            t0 = time.perf_counter()
            if is_classification:
                _, acc = train_classification(m_ds, X_train, y_train, X_val, y_val, epochs=epochs)
                v_full = acc
            else:
                v_full = train_regression(m_ds, X_train, y_train, X_val, y_val, epochs=epochs)
            t_ds = time.perf_counter() - t0
            n_params = count_params(m_ds)
            sb = state_bytes(m_ds)
            write_row("dsigma", T, None, seed, v_full, t_ds, n_params, sb)
            # Anytime sweep — train once, evaluate at multiple k.
            m_ds.eval()
            ks = [k for k in [1, 2, 4, 8, 16] if k <= T]
            with dsigma_inference_context(m_ds) as ds_layers:
                for k in ks:
                    for blk in ds_layers:
                        blk._truncation_k = k
                    if is_classification:
                        v_k = eval_classification(m_ds, X_val, y_val)
                    else:
                        v_k = eval_regression(m_ds, X_val, y_val)
                    write_row("dsigma", T, k, seed, v_k, 0.0, n_params, sb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma-separated list of seeds")
    parser.add_argument("--tasks", default="oscillator,schrodinger,digits",
                        help="comma-separated subset of tasks to run")
    parser.add_argument("--out", default="benchmarks/results.csv")
    parser.add_argument("--epochs-regression", type=int, default=150)
    parser.add_argument("--epochs-classification", type=int, default=200)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = set(args.tasks.split(","))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["task", "model", "T", "k", "seed", "metric", "value",
              "train_seconds", "params", "state_bytes"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        if "oscillator" in tasks:
            print("\n[task] damped oscillator")
            X_tr, y_tr = oscillator_dataset(8000, seed=0)
            X_va, y_va = oscillator_dataset(2000, seed=1)
            Xtn, stats = oscillator_normalize(X_tr)
            Xvn, _ = oscillator_normalize(X_va, stats)
            run_task("oscillator", in_dim=3, out_dim=1, hidden_dim=128, depth=5,
                     X_train=Xtn, y_train=y_tr, X_val=Xvn, y_val=y_va,
                     is_classification=False, seeds=seeds, writer=writer,
                     epochs=args.epochs_regression)

        if "schrodinger" in tasks:
            print("\n[task] Schrödinger E0")
            X_tr, y_tr = schrodinger_dataset(3000, seed=0)
            X_va, y_va = schrodinger_dataset(500, seed=1)
            Xtn, stats = schrodinger_normalize(X_tr)
            Xvn, _ = schrodinger_normalize(X_va, stats)
            run_task("schrodinger", in_dim=2, out_dim=1, hidden_dim=128, depth=5,
                     X_train=Xtn, y_train=y_tr, X_val=Xvn, y_val=y_va,
                     is_classification=False, seeds=seeds, writer=writer,
                     epochs=args.epochs_regression)

        if "digits" in tasks:
            print("\n[task] sklearn digits")
            data = load_digits()
            X = data.data.astype(np.float32) / 16.0
            y = data.target.astype(np.int64)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
            run_task("digits", in_dim=64, out_dim=10, hidden_dim=128, depth=4,
                     X_train=torch.from_numpy(X_tr), y_train=torch.from_numpy(y_tr),
                     X_val=torch.from_numpy(X_te), y_val=torch.from_numpy(y_te),
                     is_classification=True, seeds=seeds, writer=writer,
                     epochs=args.epochs_classification)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
