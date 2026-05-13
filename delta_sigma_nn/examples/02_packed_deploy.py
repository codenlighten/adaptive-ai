"""Save a trained DSigma model with packed trit streams, then run pure-NumPy
inference at any precision k <= T using the same checkpoint."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from delta_sigma_nn import (
    DeltaSigmaMLP,
    dsigma_inference,
    load_dsigma_arrays,
    save_dsigma_mlp,
)


def main():
    torch.manual_seed(0)
    X = torch.randn(2000, 3)
    y = torch.sin(X[:, 0:1] * 2)

    model = DeltaSigmaMLP(in_dim=3, hidden_dim=64, out_dim=1, depth=4, T=8)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad()
        ((model(X) - y) ** 2).mean().backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        torch_out = model(X[:50]).numpy()

    # Save with packed trit streams
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ds.npz"
        breakdown = save_dsigma_mlp(model, path)
        print("Saved with breakdown:")
        for k, v in breakdown.items():
            print(f"  {k}: {v}")
        # Load and run inference at varying k
        arrays = load_dsigma_arrays(path)
        for k in [1, 2, 4, 8]:
            np_out = dsigma_inference(arrays, X[:50].numpy(), k=k)
            diff = np.abs(np_out - torch_out).max()
            print(f"  k = {k}  max diff vs torch (k=8): {diff:.6f}")


if __name__ == "__main__":
    main()
