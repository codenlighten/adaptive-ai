"""Verify the Python gate-level co-simulator matches our Setun ALU."""

from __future__ import annotations

import pytest

from src.gate_sim import (
    _trit_full_adder,
    from_trits,
    ternary_pe,
    to_trits,
    trit18_adder,
)
from src.setun import Trit18, add


def test_trit_full_adder_truth_table():
    # Every (a, b, cin) triple — sum must equal a + b + cin in balanced ternary.
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                s, co = _trit_full_adder(a, b, c)
                assert s + 3 * co == a + b + c


def test_trit18_adder_matches_setun():
    """The gate-level adder produces identical results to Trit18 software adder."""
    for a, b in [(7, 5), (-13, 4), (100, -42), (0, 0), (-1, 1), (12345, 6789)]:
        a_t = Trit18.from_int(a)
        b_t = Trit18.from_int(b)
        sw_sum = add(a_t, b_t).to_int()

        hw_sum_trits, _ = trit18_adder(a_t.digits, b_t.digits)
        hw_sum = from_trits(hw_sum_trits)
        assert sw_sum == hw_sum == (a + b), f"mismatch on {a}+{b}: hw={hw_sum} sw={sw_sum}"


def test_pe_accumulates_ternary_dot():
    """Drive the PE over many cycles and check it computes a ternary dot."""
    weights = [1, -1, 0, 1, 1, -1, 0, 0, 1, -1]
    inputs = [3, 5, 7, -2, 11, 4, 13, 6, -8, 1]
    expected = sum(w * x for w, x in zip(weights, inputs))

    acc = to_trits(0)
    for w, x in zip(weights, inputs):
        acc = ternary_pe(w, to_trits(x), acc)
    assert from_trits(acc) == expected


def test_to_from_trits_roundtrip():
    for n in [-12345, -1, 0, 1, 100, 12345, 193710244]:  # max ≈ (3^18 - 1)/2
        assert from_trits(to_trits(n)) == n


def test_to_trits_overflow():
    # 3^18 = 387,420,489 — anything above (3^18-1)/2 = 193,710,244 overflows.
    with pytest.raises(OverflowError):
        to_trits(200_000_000)
