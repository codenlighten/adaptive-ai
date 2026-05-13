import numpy as np

from src.multfree import (
    count_ops,
    matmul_scalar,
    matmul_split_masks,
    matmul_ternary,
)


def test_matches_dense_matmul():
    rng = np.random.default_rng(0)
    for shape in [(4, 8), (16, 32), (64, 128)]:
        W = rng.integers(-1, 2, size=shape).astype(np.int8)
        x = rng.standard_normal(shape[1]).astype(np.float32)
        expected = W.astype(np.float32) @ x
        got = matmul_split_masks(W, x)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_scalar_matches_vectorized():
    rng = np.random.default_rng(1)
    W = rng.integers(-1, 2, size=(8, 16)).astype(np.int8)
    x = rng.standard_normal(16).astype(np.float64)
    np.testing.assert_allclose(matmul_scalar(W, x), matmul_split_masks(W, x))


def test_batch_input():
    rng = np.random.default_rng(2)
    W = rng.integers(-1, 2, size=(10, 20)).astype(np.int8)
    X = rng.standard_normal((20, 32)).astype(np.float32)
    expected = W.astype(np.float32) @ X
    np.testing.assert_allclose(matmul_split_masks(W, X), expected, rtol=1e-5, atol=1e-5)


def test_ternary_with_scale():
    rng = np.random.default_rng(3)
    W = rng.integers(-1, 2, size=(4, 6)).astype(np.int8)
    x = rng.standard_normal(6).astype(np.float32)
    alpha = 0.37
    expected = alpha * (W.astype(np.float32) @ x)
    np.testing.assert_allclose(matmul_ternary(W, x, alpha), expected, rtol=1e-5)


def test_count_ops_reports_zero_multiplies():
    W = np.array([[1, -1, 0], [0, 1, 1]], dtype=np.int8)
    stats = count_ops(W)
    assert stats["ternary_multiplies"] == 0
    assert stats["ternary_adds_subs"] == 4
    assert stats["fp32_multiplies"] == 6
    assert abs(stats["skipped_fraction"] - 2 / 6) < 1e-12
