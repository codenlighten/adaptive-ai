import pytest

from src.setun import (
    SetunVM,
    Trit18,
    add,
    compare,
    matmul_as_setun_program,
    mul,
    neg,
    shift_left,
    shift_right,
    sub,
    ternary_dot,
)


def test_trit18_roundtrip():
    for n in [-1000, -1, 0, 1, 7, 13, -42, 12345, -98765]:
        assert Trit18.from_int(n).to_int() == n


def test_trit18_overflow():
    # 3^17 = 129140163; 18-trit max ≈ 1.93e8.
    with pytest.raises(OverflowError):
        Trit18.from_int(10**9)


def test_balanced_add_sub():
    for a in [-100, -1, 0, 1, 50, 12345]:
        for b in [-99, -1, 0, 1, 23, -9999]:
            assert add(Trit18.from_int(a), Trit18.from_int(b)).to_int() == a + b
            assert sub(Trit18.from_int(a), Trit18.from_int(b)).to_int() == a - b


def test_neg():
    assert neg(Trit18.from_int(42)).to_int() == -42
    assert neg(Trit18.from_int(-7)).to_int() == 7


def test_compare():
    assert compare(Trit18.from_int(5), Trit18.from_int(3)) == 1
    assert compare(Trit18.from_int(3), Trit18.from_int(5)) == -1
    assert compare(Trit18.from_int(7), Trit18.from_int(7)) == 0
    assert compare(Trit18.from_int(-5), Trit18.from_int(0)) == -1


def test_shifts():
    a = Trit18.from_int(5)
    # 5 * 3 = 15
    assert shift_left(a, 1).to_int() == 15
    # 15 // 3 = 5
    assert shift_right(Trit18.from_int(15), 1).to_int() == 5


def test_multiply_shift_and_add():
    for a in [-12, -1, 0, 1, 11, 7]:
        for b in [-9, -1, 0, 1, 13, 4]:
            assert mul(Trit18.from_int(a), Trit18.from_int(b)).to_int() == a * b


def test_ternary_dot_matches_numpy():
    weights = [1, -1, 0, 1, -1, 0]
    inputs_ints = [3, -2, 7, 5, 4, 11]
    inputs = [Trit18.from_int(x) for x in inputs_ints]
    expected = sum(w * x for w, x in zip(weights, inputs_ints))
    assert ternary_dot(weights, inputs).to_int() == expected


def test_setun_vm_executes_program():
    vm = SetunVM(mem_size=4)
    vm.mem[0] = Trit18.from_int(7)
    vm.mem[1] = Trit18.from_int(5)
    program = [
        ("LD", 0, 0),
        ("LD", 1, 1),
        ("ADD", 2, 0, 1),
        ("ST", 2, 2),
        ("HLT",),
    ]
    vm.run(program)
    assert vm.mem[2].to_int() == 12
    assert vm.stats["add"] == 1
    assert vm.stats["ld"] == 2


def test_setun_matmul_matches_dense():
    W = [
        [1, -1, 0, 1],
        [0, 0, 1, -1],
        [-1, 0, 1, 0],
    ]
    x = [3, -2, 7, 5]
    expected = [sum(W[i][j] * x[j] for j in range(4)) for i in range(3)]
    program, vm = matmul_as_setun_program(W, x)
    vm.run(program)
    actual = [vm.mem[4 + i].to_int() for i in range(3)]
    assert actual == expected, f"{actual} != {expected}"
    # The program should use ONLY adds/subs (no MULs).
    assert vm.stats["mul"] == 0
    assert vm.stats["add"] + vm.stats["sub"] > 0


def test_setun_control_flow_jump_and_loop():
    """A small loop should count from 0 to N using control flow."""
    vm = SetunVM(mem_size=4)
    program = [
        ("LDI", 0, 0),                    # R0 (counter) = 0
        ("LDI", 1, 5),                    # R1 (limit) = 5
        ("LDI", 2, 1),                    # R2 = 1
        ("LABEL", "top"),
        ("SUB",  3, 0, 1),                # R3 = R0 - R1
        ("JZ",   3, "done"),
        ("ADD",  0, 0, 2),                # R0 += 1
        ("JMP",  "top"),
        ("LABEL", "done"),
        ("ST",   0, 0),
        ("HLT",),
    ]
    vm.run(program)
    assert vm.mem[0].to_int() == 5
    assert vm.stats["jmp"] >= 5
    assert vm.stats["add"] == 5  # one ADD per iteration


def test_setun_looped_matmul_matches_unrolled():
    """The loop-based and unrolled matmul programs should agree."""
    import sys
    sys.path.insert(0, "scripts")
    from run_setun_loops import run_matmul_on_vm

    W = [
        [1, -1, 0, 1],
        [0, 0, 1, -1],
        [-1, 0, 1, 0],
    ]
    x = [3, -2, 7, 5]
    expected = [sum(W[i][j] * x[j] for j in range(4)) for i in range(3)]
    y_loop, stats = run_matmul_on_vm(W, x)
    assert y_loop == expected
    assert stats["mul"] == 0
