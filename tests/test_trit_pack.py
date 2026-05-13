import numpy as np

from delta_sigma_nn.trit_pack import (
    compression_ratio_vs_fp32,
    pack_trits,
    storage_bytes,
    unpack_trits,
)


def test_round_trip_random():
    rng = np.random.default_rng(0)
    for size in [1, 4, 5, 6, 100, 1023, 10_000]:
        arr = rng.integers(-1, 2, size=size).astype(np.int8)
        packed, n = pack_trits(arr)
        unpacked = unpack_trits(packed, n)
        assert np.array_equal(arr, unpacked), f"mismatch at size={size}"


def test_byte_capacity():
    # 5 trits should fit in exactly 1 byte.
    arr = np.array([-1, 0, 1, 1, -1], dtype=np.int8)
    packed, n = pack_trits(arr)
    assert packed.size == 1
    assert unpack_trits(packed, n).tolist() == arr.tolist()


def test_max_byte_value():
    # All +1s in a group of 5 trits => (2 + 2*3 + 2*9 + 2*27 + 2*81) = 242 < 256
    arr = np.ones(5, dtype=np.int8)
    packed, _ = pack_trits(arr)
    assert packed[0] == 242
    arr = -np.ones(5, dtype=np.int8)
    packed, _ = pack_trits(arr)
    assert packed[0] == 0


def test_storage_bytes_math():
    assert storage_bytes(0) == 0
    assert storage_bytes(1) == 1
    assert storage_bytes(5) == 1
    assert storage_bytes(6) == 2
    assert storage_bytes(10) == 2


def test_compression_ratio_vs_fp32():
    # 1000 trits = 200 bytes; fp32 would be 4000 bytes => 20x
    assert abs(compression_ratio_vs_fp32(1000) - 20.0) < 1e-9
