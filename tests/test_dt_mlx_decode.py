import sqlite3
import struct
import sys

import numpy as np
import pytest
from test_dt_tensor_store import checkpoint

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="DT decoding requires Metal")

from minimax_h3_mlx.dt_mlx_decode import DTMLXTensorStore  # noqa: E402
from minimax_h3_mlx.dt_tensor_store import DTTensorStore  # noqa: E402


@pytest.mark.parametrize("trailer", [False, True])
@pytest.mark.parametrize("columns", [3, 256])
def test_int8_gpu_decode_matches_cpu_half_rounding(tmp_path, trailer, columns):
    q = np.tile(np.arange(-128, 128, dtype=np.int8)[:columns], (7, 1))
    scales = np.array([0, -0.0, 0.1, -0.3, 0.000061, 1.998, 512], dtype=np.float16)
    payload = q.tobytes() + bytes((-q.size) % 128) + scales.tobytes()
    path = checkpoint(tmp_path, 0x8A1E9B, q.shape, payload, trailer=trailer)
    before = path.read_bytes()
    with DTTensorStore(path) as reference, DTMLXTensorStore(path) as candidate:
        with np.errstate(over="ignore"):
            expected = reference.read("w")
        actual = candidate.read("w")
        assert isinstance(actual, mx.array)
        np.testing.assert_array_equal(np.array(actual).view(np.uint16), expected.view(np.uint16))
        assert candidate.report()["gpu_decoded_tensors"] == 1
        assert candidate.report()["persistent_weight_bytes_written"] == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("count", [3, 8, 11])
def test_palette_gpu_decode_preserves_half_bits_and_partial_blocks(tmp_path, count):
    block = 4
    palettes = np.arange(3 * 256, dtype=np.uint16).reshape(3, 256)
    palettes[:, 1] = 0x8000  # negative zero must survive gathering
    payload = struct.pack("<I", block)
    for start in range(0, count, block):
        payload += palettes[start // block].tobytes()
        payload += np.arange(min(block, count - start), dtype=np.uint8).tobytes()
    path = checkpoint(tmp_path, 0x8A1E8B, (count,), payload)
    with DTTensorStore(path) as reference, DTMLXTensorStore(path) as candidate:
        expected = reference.read("w")
        actual = np.array(candidate.read("w"))
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


def test_gpu_reader_keeps_limits_size_checks_and_source_invalidation(tmp_path):
    path = checkpoint(tmp_path, 0x8A1E9B, (2, 4), bytes(132), trailer=True)
    with DTMLXTensorStore(path, max_tensor_bytes=8) as reader:
        with pytest.raises(ValueError, match="limit"):
            reader.read("w")
    with DTMLXTensorStore(path) as reader:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
        with pytest.raises(RuntimeError, match="changed"):
            reader.read("w")


def test_gpu_reader_keeps_reference_round_toward_zero(tmp_path):
    path = checkpoint(
        tmp_path,
        0x8A1E9B,
        (1, 4),
        bytes([3] * 4) + bytes(124) + np.array([0.1], dtype=np.float16).tobytes(),
    )
    with DTTensorStore(path, rounding="toward_zero") as reference:
        expected = reference.read("w")
    with DTMLXTensorStore(path, rounding="toward_zero") as candidate:
        np.testing.assert_array_equal(candidate.read("w"), expected)
        assert candidate.report()["gpu_decoded_tensors"] == 0


def test_external_palette_gpu_decode(tmp_path):
    # DT external palettes keep block size in the descriptor, not the payload.
    payload = np.arange(256, dtype=np.float16).tobytes() + bytes([255, 0, 1])
    path = checkpoint(tmp_path, 0x8A1E8B, (3,), payload)
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE tensors SET type=?, data=?",
            (
                ((0x10000000 | 0x8A1E8B) << 32) | 1,
                struct.pack("<IIQQ", 4, 0, 0, len(payload)),
            ),
        )
    path.with_name(path.name + "-tensordata").write_bytes(payload)
    with DTMLXTensorStore(path) as candidate:
        np.testing.assert_array_equal(np.array(candidate.read("w")), [255, 0, 1])


def test_gpu_int8_all_finite_half_scales_match_cpu(tmp_path):
    bits = np.concatenate(
        [np.arange(0x7C00, dtype=np.uint16), np.arange(0x8000, 0xFC00, dtype=np.uint16)]
    )
    scales = bits.view(np.float16)
    q = np.tile(np.arange(-128, 128, dtype=np.int8), (len(scales), 1))
    path = checkpoint(tmp_path, 0x8A1E9B, q.shape, q.tobytes() + scales.tobytes(), trailer=True)
    with DTTensorStore(path) as reference, DTMLXTensorStore(path) as candidate:
        with np.errstate(over="ignore"):
            expected = reference.read("w")
        actual = np.array(candidate.read("w"))
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize(
    "codec,payload",
    [
        (0x8A1E9B, b"short"),
        (0x8A1E8B, struct.pack("<I", 4) + b"short"),
        (0x8A1E8B, struct.pack("<I", 0) + bytes(516)),
    ],
)
def test_malformed_packed_payload_rejected_before_gpu(tmp_path, monkeypatch, codec, payload):
    path = checkpoint(tmp_path, codec, (1, 4), payload)

    def unexpected_gpu(*args, **kwargs):
        raise AssertionError("Malformed weights reached GPU allocation")

    monkeypatch.setattr(mx, "array", unexpected_gpu)
    with DTMLXTensorStore(path) as candidate:
        with pytest.raises(ValueError, match="size|block"):
            candidate.read("w")


def test_raw_fallback_and_closed_store(tmp_path):
    path = checkpoint(tmp_path, 0, (4,), np.arange(4, dtype=np.float16).tobytes())
    with DTMLXTensorStore(path) as candidate:
        np.testing.assert_array_equal(candidate.read("w"), np.arange(4))
        assert candidate.report()["gpu_decoded_tensors"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        candidate.read("w")


def test_gpu_rotary_and_qkv_mapping_matches_cpu():
    from minimax_h3_mlx.dt_h3_checkpoint import fuse_qkv, unpair_rotary

    value = np.arange(2 * 8 * 3, dtype=np.float16).reshape(16, 3)
    expected = unpair_rotary(value, heads=2, head_dim=8, rotary_dim=6)
    actual = unpair_rotary(mx.array(value), heads=2, head_dim=8, rotary_dim=6, array_module=mx)
    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(np.array(actual), expected)
    expected = fuse_qkv(expected, value, value + 1, heads=2, head_dim=8)
    actual = fuse_qkv(
        actual, mx.array(value), mx.array(value + 1), heads=2, head_dim=8, array_module=mx
    )
    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(np.array(actual), expected)


def test_decode_backend_rejected_before_opening_model():
    from minimax_h3_mlx.dt_h3_checkpoint import load_dt_h3_dit

    with pytest.raises(ValueError, match="decode backend"):
        load_dt_h3_dit("nonexistent.ckpt", decode_backend="unknown")
