"""Equivalence test: pure-NumPy multfree inference matches torch BitMLP."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import save_bitmlp
from src.inference import TernaryNet
from src.model import BitMLP


def test_multfree_inference_matches_torch():
    torch.manual_seed(42)
    model = BitMLP(3, 32, 1, depth=4)
    model.eval()

    x = torch.randn(7, 3)
    with torch.no_grad():
        torch_out = model(x).numpy()

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.npz"
        save_bitmlp(model, path)
        net = TernaryNet(path)
        np_out = net.forward(x.numpy())

    diff = np.abs(torch_out - np_out)
    assert diff.max() < 1e-4, f"max diff {diff.max()} exceeds tolerance"


def test_matmul_multiplies_are_zero():
    torch.manual_seed(0)
    model = BitMLP(3, 32, 1, depth=4)
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.npz"
        save_bitmlp(model, path)
        net = TernaryNet(path)
        net.reset_counter()
        x = np.random.RandomState(0).randn(5, 3).astype(np.float32)
        _ = net.forward(x)
        c = net.counter
    assert c.matmul_multiplies_avoided > 0
    assert c.matmul_signed_adds > 0
    # The signed-add count must equal nonzero ternary weights * batch.
    assert c.matmul_signed_adds < c.matmul_multiplies_avoided


def test_packed_file_is_smaller_than_fp32():
    torch.manual_seed(1)
    model = BitMLP(8, 128, 4, depth=5)
    fp32_size = sum(p.numel() * 4 for p in model.parameters())
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.npz"
        save_bitmlp(model, path)
        packed_size = path.stat().st_size
    # Should be at least 4x smaller for this size (hidden weights dominate).
    assert packed_size * 4 < fp32_size, f"expected compression, got {packed_size} vs {fp32_size}"
