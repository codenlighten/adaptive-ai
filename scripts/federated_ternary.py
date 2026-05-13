"""Federated training with ternary gradient compression.

We simulate N clients, each holding 1/N of the damped-oscillator dataset.
At each round:
  1. Each client trains locally for K epochs starting from the global model.
  2. Each client computes its weight delta vs the round-start global model.
  3. The delta is ternarized (per-tensor) before being "uploaded".
  4. Server averages the ternarized deltas + global model.

The compression ratio is ~32× (fp32 -> ~1.6 bits/weight). We compare
final val MSE to (a) federated fp32 (no compression) and (b) centralized
fp32 (no federation, no compression).

Run: venv/bin/python -m scripts.federated_ternary
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import FPMLP
from src.ternary import ternarize


def local_train(model, X, y, epochs, lr, batch_size):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()


def evaluate(model, X, y):
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        return loss_fn(model(X), y).item()


def state_diff(s_a, s_b):
    """a - b per tensor."""
    return {k: s_a[k] - s_b[k] for k in s_a}


def state_add(s, d, scale=1.0):
    """s + scale * d in-place return."""
    return {k: s[k] + scale * d[k] for k in s}


def ternarize_state(s):
    """Apply ternarization to every weight-like tensor.

    Per-tensor scale alpha + ternary {-1, 0, +1} grid.
    Compression ratio (vs fp32): for n weights, fp32 = 32n bits; ternary =
    n * log2(3) ~ 1.585n bits, plus 32 bits for alpha. So ratio ~= 32/1.585 ~ 20x
    (we ignore the negligible alpha overhead in the print).
    """
    out = {}
    for k, t in s.items():
        if t.numel() < 4:
            out[k] = t.clone()  # bias / norm params unchanged
            continue
        w_q, alpha = ternarize(t)
        out[k] = w_q * alpha
    return out


def federated_train(global_model, clients_data, rounds, local_epochs, lr, batch_size,
                    ternary_compression=True):
    """Standard FedAvg with optional ternary compression of the per-client delta.

    Returns trained global model.
    """
    bytes_per_round_compressed = 0
    bytes_per_round_uncompressed = 0
    for r in range(rounds):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        deltas = []
        for X_c, y_c in clients_data:
            local_model = copy.deepcopy(global_model)
            local_train(local_model, X_c, y_c, local_epochs, lr, batch_size)
            delta = state_diff(local_model.state_dict(), global_state)
            if ternary_compression:
                delta_q = ternarize_state(delta)
                deltas.append(delta_q)
            else:
                deltas.append(delta)
            if r == 0:
                # Approximate byte cost for one client's update.
                for k, v in delta.items():
                    bytes_per_round_uncompressed += v.numel() * 4
                    if ternary_compression and v.numel() >= 4:
                        bytes_per_round_compressed += (v.numel() * 1.585) / 8 + 4  # +4 for alpha
                    else:
                        bytes_per_round_compressed += v.numel() * 4

        avg_delta = {}
        for k in global_state:
            avg_delta[k] = sum(d[k] for d in deltas) / len(deltas)
        new_state = state_add(global_state, avg_delta, scale=1.0)
        global_model.load_state_dict(new_state)
    return global_model, bytes_per_round_compressed, bytes_per_round_uncompressed


def main():
    torch.manual_seed(0)
    n_clients = 5
    rounds = 10
    local_epochs = 5

    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    # Partition train data across clients.
    chunk = Xtn.shape[0] // n_clients
    clients_data = [
        (Xtn[i * chunk:(i + 1) * chunk], y_train[i * chunk:(i + 1) * chunk])
        for i in range(n_clients)
    ]
    print(f"Clients: {n_clients}, samples per client: {chunk}\n")

    # === Centralized fp32 baseline ===
    print("=== Centralized fp32 (no federation) ===")
    central = FPMLP(3, 128, 1, depth=5)
    local_train(central, Xtn, y_train, epochs=rounds * local_epochs, lr=2e-3, batch_size=256)
    print(f"  centralized val MSE: {evaluate(central, Xvn, y_val):.6f}")

    # === Federated fp32 (no compression) ===
    print("\n=== Federated fp32 (no compression) ===")
    fed_fp = FPMLP(3, 128, 1, depth=5)
    t0 = time.time()
    fed_fp, b_c, b_uc = federated_train(fed_fp, clients_data, rounds, local_epochs,
                                         lr=2e-3, batch_size=256, ternary_compression=False)
    print(f"  trained in {time.time()-t0:.1f}s, val MSE: {evaluate(fed_fp, Xvn, y_val):.6f}")
    print(f"  bytes per client per round: {b_uc/n_clients/1024:.1f} KB")

    # === Federated with ternary compression ===
    print("\n=== Federated with ternary delta compression ===")
    fed_bit = FPMLP(3, 128, 1, depth=5)
    t0 = time.time()
    fed_bit, b_c, b_uc = federated_train(fed_bit, clients_data, rounds, local_epochs,
                                          lr=2e-3, batch_size=256, ternary_compression=True)
    print(f"  trained in {time.time()-t0:.1f}s, val MSE: {evaluate(fed_bit, Xvn, y_val):.6f}")
    print(f"  bytes per client per round: {b_c/n_clients/1024:.1f} KB "
          f"(compression {b_uc/b_c:.1f}x vs fp32)")


if __name__ == "__main__":
    main()
