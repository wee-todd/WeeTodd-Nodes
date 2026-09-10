import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from minimax_h3_mlx.dt_tensor_store import DTTensorStore


def checkpoint(tmp_path, codec, shape, payload, *, trailer=False):
    path = tmp_path / "model.ckpt"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE tensors(name TEXT PRIMARY KEY, type INTEGER, format INTEGER, "
            "datatype INTEGER, dim BLOB, data BLOB)"
        )
        descriptor = struct.pack("<QQ", 0, len(payload)) if trailer else payload
        db.execute(
            "INSERT INTO tensors VALUES(?,?,?,?,?,?)",
            (
                "w",
                ((codec | (0x10000000 if trailer else 0)) << 32) | 1,
                0,
                0x20000,
                struct.pack("<12i", *shape, *([0] * (12 - len(shape)))),
                descriptor,
            ),
        )
    if trailer:
        base = path.stat().st_size
        with path.open("r+b") as f:
            f.seek(60)
            f.write(struct.pack(">I", base))
            f.seek(base)
            f.write(payload)
    return path


def test_raw_trailer_rows_and_close(tmp_path):
    value = np.arange(24, dtype=np.float16).reshape(6, 4)
    p = checkpoint(tmp_path, 0, value.shape, value.tobytes(), trailer=True)
    before = p.read_bytes()
    with DTTensorStore(p, max_tensor_bytes=32) as store:
        np.testing.assert_array_equal(store.read_rows("w", 2, 2), value[2:4])
        with pytest.raises(ValueError, match="limit"):
            store.read("w")
        with pytest.raises(ValueError, match="range"):
            store.read_rows("w", 5, 2)
        assert store.report()["payload_bytes_read"] == 16
    with pytest.raises(RuntimeError, match="closed"):
        store.read_rows("w", 0, 1)
    assert p.read_bytes() == before


def test_int8_rounding_and_rows(tmp_path):
    q = np.array([[3, -3, 127], [6, -8, 1]], dtype=np.int8)
    scales = np.array([0.1, 0.3], dtype=np.float16)
    b = q.tobytes() + bytes(128 - q.size) + scales.tobytes()
    p = checkpoint(tmp_path, 0x8A1E9B, q.shape, b, trailer=True)
    with DTTensorStore(p) as s:
        expected = (q.astype(np.float32) * scales[:, None].astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(s.read("w"), expected)
        np.testing.assert_array_equal(s.read_rows("w", 1, 1), expected[1:])


def test_palette_partial_block(tmp_path):
    b = struct.pack("<I", 4) + np.arange(256, dtype=np.float16).tobytes() + bytes([1, 2, 3])
    with DTTensorStore(checkpoint(tmp_path, 0x8A1E8B, (3,), b)) as s:
        np.testing.assert_array_equal(s.read("w"), [1, 2, 3])


def test_ezm7(tmp_path):
    z = zlib.compressobj(wbits=-15)
    exp = z.compress(bytes([15, 16, 0])) + z.flush()
    b = struct.pack("<I", len(exp)) + exp + bytes([0, 128, 0])
    with DTTensorStore(checkpoint(tmp_path, 0x511, (3,), b)) as s:
        np.testing.assert_array_equal(s.read("w"), [1, -2, 0])


def test_source_replaced(tmp_path):
    p = checkpoint(tmp_path, 0, (1,), struct.pack("<e", 1))
    with DTTensorStore(p) as s:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(p.read_bytes())
        replacement.replace(p)
        with pytest.raises(RuntimeError, match="changed"):
            s.read("w")


def test_reject_wal(tmp_path):
    p = checkpoint(tmp_path, 0, (1,), struct.pack("<e", 1))
    Path(str(p) + "-wal").touch()
    with pytest.raises(ValueError, match="WAL"):
        DTTensorStore(p)


def test_bad_payload(tmp_path):
    with DTTensorStore(checkpoint(tmp_path, 0, (3,), b"00")) as s:
        with pytest.raises(ValueError, match="size"):
            s.read("w")


def test_sqlite_without_tensor_table_reports_supported_error(tmp_path):
    p = tmp_path / "unknown.ckpt"
    with sqlite3.connect(p) as db:
        db.execute("CREATE TABLE other(value TEXT)")
    with pytest.raises(ValueError, match="DT tensor store"):
        DTTensorStore(p)


def test_metadata_validation_rejects_short_external_payload(tmp_path):
    p = checkpoint(tmp_path, 0, (4,), b"00", trailer=True)
    with DTTensorStore(p) as s:
        with pytest.raises(ValueError, match="size"):
            s.validate_tensor("w")


def test_metadata_validation_rejects_non_row_embedding(tmp_path):
    p = checkpoint(tmp_path, 0, (1, 1), b"00")
    with DTTensorStore(p) as s:
        with pytest.raises(ValueError, match="row"):
            s.validate_tensor("w", row_access=True)
