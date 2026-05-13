import tempfile
from pathlib import Path

import numpy as np
import torch

from src.dsigma_linear import DeltaSigmaMLP
from src.dsigma_pack import dsigma_inference, load_dsigma_arrays, save_dsigma_mlp


def test_dsigma_pack_roundtrip():
    """Packed inference should match the torch model at full k."""
    torch.manual_seed(0)
    model = DeltaSigmaMLP(3, 32, 1, depth=4, T=8)
    model.eval()
    x = torch.randn(5, 3)
    with torch.no_grad():
        torch_out = model(x).numpy()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ds.npz"
        save_dsigma_mlp(model, path)
        arrays = load_dsigma_arrays(path)
        np_out = dsigma_inference(arrays, x.numpy())
    diff = np.abs(np_out - torch_out).max()
    assert diff < 1e-4, f"diff {diff}"


def test_dsigma_pack_anytime():
    """Different k values produce different outputs but full k matches torch."""
    torch.manual_seed(0)
    model = DeltaSigmaMLP(3, 32, 1, depth=4, T=8)
    model.eval()
    x = torch.randn(5, 3)
    with torch.no_grad():
        full = model(x).numpy()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ds.npz"
        save_dsigma_mlp(model, path)
        arrays = load_dsigma_arrays(path)
        outs = [dsigma_inference(arrays, x.numpy(), k=k) for k in [1, 2, 4, 8]]
    # k=8 matches torch
    assert np.abs(outs[-1] - full).max() < 1e-4
    # k=1 and k=8 should differ — anytime is doing something
    assert np.abs(outs[0] - outs[-1]).max() > 1e-4
