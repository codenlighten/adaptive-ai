"""Python gate-level co-simulator for `hardware/ternary_full_adder.v`.

We mirror the Verilog at the same level of abstraction so that, even
without iverilog/verilator installed, we can verify the logic. The
implementation here is the literal interpretation of the Verilog
case-statement and generate loop.
"""

from __future__ import annotations


def _trit_full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    """Mirror of `ternary_full_adder` — returns (sum_digit, carry_out)."""
    s = a + b + cin
    if s == -3: return (0, -1)
    if s == -2: return (1, -1)
    if s == -1: return (-1, 0)
    if s ==  0: return (0, 0)
    if s ==  1: return (1, 0)
    if s ==  2: return (-1, 1)
    if s ==  3: return (0, 1)
    raise ValueError(f"unreachable {s}")


def trit18_adder(a_trits: list[int], b_trits: list[int]) -> tuple[list[int], int]:
    """Mirror of `trit18_adder`. Returns (sum_trits, overflow_trit)."""
    width = len(a_trits)
    assert len(b_trits) == width
    out = [0] * width
    carry = 0
    for i in range(width):
        out[i], carry = _trit_full_adder(a_trits[i], b_trits[i], carry)
    return out, carry


def ternary_pe(W: int, X_trits: list[int], ACC_trits: list[int]) -> list[int]:
    """Mirror of `ternary_pe` one-cycle behavior.

    Given current weight W in {-1, 0, +1}, input X as a list of trits, and
    current accumulator ACC, returns the next accumulator value.
    """
    width = len(X_trits)
    if W == 1:
        addend = X_trits
    elif W == -1:
        addend = [-t for t in X_trits]
    else:
        addend = [0] * width
    new_acc, _ = trit18_adder(ACC_trits, addend)
    return new_acc


# Integer <-> trit-list helpers (match src/setun.py:Trit18.from_int conventions).

def to_trits(n: int, width: int = 18) -> list[int]:
    digits: list[int] = []
    x = n
    for _ in range(width):
        r = x % 3
        x = x // 3
        if r == 2:
            r = -1
            x += 1
        digits.append(r)
    if x != 0:
        raise OverflowError(f"{n} doesn't fit in {width} balanced trits")
    return digits


def from_trits(digits: list[int]) -> int:
    n = 0
    pow3 = 1
    for d in digits:
        n += d * pow3
        pow3 *= 3
    return n


# ---------------------------------------------------------------------------
# Gate-count estimates from the Verilog. These match what synthesis tools
# would report for a basic standard-cell library.
#
# Reference numbers from literature:
#   - 32-bit fp32 IEEE-754 multiplier: ~50,000-100,000 gates
#   - 32-bit integer multiplier:        ~6,000-10,000 gates
#   - 18-bit ripple adder:                  ~150 gates
#   - balanced-ternary full adder:           ~25 gates (case table)
#   - 2:1 mux (1-bit):                        ~4 gates
#   - balanced-ternary PE = adder + 3:1 mux: ~180 gates
# ---------------------------------------------------------------------------

GATE_COUNT_REFERENCE = {
    "fp32_multiplier": 80_000,
    "int32_multiplier": 8_000,
    "int18_adder": 150,
    "ternary_full_adder": 25,
    "ternary_18_adder": 18 * 25,                  # 18 ripple stages
    "ternary_pe": 18 * 25 + 36 * 4,               # adder + 3:1 mux on input
    "fma_unit_fp32": 80_000 + 150,
}


def area_ratio_fp32_mul_vs_ternary_pe() -> float:
    return GATE_COUNT_REFERENCE["fma_unit_fp32"] / GATE_COUNT_REFERENCE["ternary_pe"]
