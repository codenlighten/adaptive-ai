"""Train ternary, quaternary, and fp32 MLPs on sklearn's 8x8 digits.

External validity check outside the physics tasks. The sklearn `digits`
dataset is 1797 8x8 grayscale digit images, 10 classes. Standard tiny
classification benchmark.

Run: venv/bin/python -m scripts.train_digits
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.model import BitMLP, FPMLP
from src.quaternary import QuatMLP
from src.precision_models import BinaryMLP


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = X.shape[0]
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        loss = F.cross_entropy(logits, y_val).item()
        acc = (logits.argmax(dim=-1) == y_val).float().mean().item()
    print(f"  [{label}] done in {time.time()-t0:.1f}s  val_loss={loss:.4f}  val_acc={acc:.4f}")
    return loss, acc


def main():
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0  # in [0, 1]
    y = data.target.astype(np.int64)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} pixels (8x8), 10 classes")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr = torch.from_numpy(X_tr); X_te = torch.from_numpy(X_te)
    y_tr = torch.from_numpy(y_tr); y_te = torch.from_numpy(y_te)
    print(f"Train: {X_tr.shape[0]}  Val: {X_te.shape[0]}\n")

    common = dict(in_dim=64, hidden_dim=128, out_dim=10, depth=4)
    configs = [
        ("Binary",     BinaryMLP(**common)),
        ("Ternary",    BitMLP(**common)),
        ("Quaternary", QuatMLP(**common)),
        ("fp32",       FPMLP(**common)),
    ]
    results = []
    for label, model in configs:
        n_p = sum(p.numel() for p in model.parameters())
        loss, acc = train(model, X_tr, y_tr, X_te, y_te,
                          epochs=200, lr=2e-3, batch_size=64, label=label)
        results.append((label, n_p, loss, acc))

    print("\n=== sklearn-digits classification comparison ===")
    print(f"{'config':>12}  {'params':>9}  {'val_loss':>10}  {'val_acc':>9}")
    for label, n_p, loss, acc in results:
        print(f"{label:>12}  {n_p:>9,}  {loss:>10.4f}  {acc:>9.4f}")


if __name__ == "__main__":
    main()
