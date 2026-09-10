"""Read-only, bounded access to supported Draw Things tensor stores.

No MLX import or model residency is incurred by metadata inspection. This module
never rewrites a checkpoint or creates an external tensor store.
"""

from __future__ import annotations

import math
import os
import sqlite3
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EXTERNAL = 0x10000000
_FP16 = 0x20000
_RAW, _EZM7, _Q8P, _I8X = 0, 0x511, 0x8A1E8B, 0x8A1E9B


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    codec: int
    datatype: int
    inline_bytes: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class DTTensorStore:
    """A read-only checkpoint handle; callers own and must close its lifetime."""

    def __init__(
        self, path: str | Path, *, max_tensor_bytes: int = 256 * 1024**2, rounding: str = "nearest"
    ):
        if max_tensor_bytes < 1 or rounding not in {"nearest", "toward_zero"}:
            raise ValueError("Invalid tensor allocation limit or rounding policy.")
        self.path = Path(path).expanduser().resolve(strict=True)
        self.max_tensor_bytes = max_tensor_bytes
        self.rounding = rounding
        self._db = None
        self._fd = None
        self._payload_fd = None
        self.payload_bytes_read = 0
        self.read_seconds = 0.0
        self.decode_seconds = 0.0
        self.tensors_read = 0
        try:
            if Path(str(self.path) + "-wal").exists():
                raise ValueError("DT checkpoint has a WAL file; finish its update before loading.")
            self._fd = os.open(self.path, os.O_RDONLY)
            self._source_identity = _identity(os.fstat(self._fd))
            header = os.pread(self._fd, 100, 0)
            if len(header) != 100 or header[:16] != b"SQLite format 3\0":
                raise ValueError("Not a supported DT SQLite checkpoint.")
            self._base = int.from_bytes(header[60:64], "big")
            self._db = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
            self._db.execute("PRAGMA trusted_schema=OFF")
            self._db.execute("PRAGMA query_only=ON")
            self._db.execute("PRAGMA cache_size=-2048")
            pages = self._db.execute("PRAGMA page_count").fetchone()[0]
            page_size = self._db.execute("PRAGMA page_size").fetchone()[0]
            if self._base and not pages * page_size <= self._base <= self._source_identity[2]:
                raise ValueError("Invalid DT tensor trailer boundary.")
            self.records = {}
            for name, kind, dtype, dim, size in self._db.execute(
                "SELECT name,type,datatype,dim,length(data) FROM tensors"
            ):
                if len(self.records) >= 10000:
                    raise ValueError("DT tensor store exceeds the supported inventory limit.")
                if not isinstance(dim, bytes) or not dim or len(dim) % 4 or len(dim) > 48:
                    raise ValueError("Invalid DT tensor dimensions.")
                dims = struct.unpack("<" + "i" * (len(dim) // 4), dim)
                shape = dims[: dims.index(0)] if 0 in dims else dims
                # NNC terminates dimensions at the first zero; later slots may contain scratch data.
                if not shape or any(x <= 0 for x in shape) or not isinstance(name, str):
                    raise ValueError("Invalid DT tensor shape or name.")
                if not isinstance(kind, int) or not isinstance(size, int) or size < 0:
                    raise ValueError("Invalid DT tensor storage metadata.")
                if name in self.records:
                    raise ValueError("Duplicate DT tensor name.")
                self.records[name] = TensorRecord(
                    name, shape, (kind >> 32) & 0xFFFFFFFF, dtype, size
                )
            self._payload_path = self.path if self._base else Path(str(self.path) + "-tensordata")
            if any(r.codec & _EXTERNAL for r in self.records.values()):
                self._payload_fd = os.open(self._payload_path, os.O_RDONLY)
                self._payload_identity = _identity(os.fstat(self._payload_fd))
            else:
                self._payload_identity = None
        except sqlite3.Error as exc:
            self.close()
            raise ValueError(f"Unsupported or corrupt DT tensor store: {exc}") from exc
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        for name in ("_fd", "_payload_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def _check(self):
        if self._db is None:
            raise RuntimeError("DT tensor store is closed.")
        try:
            changed = _identity(self.path.stat()) != self._source_identity
            changed |= Path(str(self.path) + "-wal").exists()
            if self._payload_identity is not None:
                changed |= _identity(self._payload_path.stat()) != self._payload_identity
        except OSError as exc:
            raise RuntimeError("DT model files changed or disappeared; reload the model.") from exc
        if changed:
            raise RuntimeError("DT model files changed; reload the model.")

    def _record(self, name):
        self._check()
        record = self.records[name]
        if record.datatype != _FP16 or record.codec & ~_EXTERNAL not in {_RAW, _EZM7, _Q8P, _I8X}:
            raise ValueError(f"Unsupported DT tensor codec or dtype for {name!r}.")
        return record

    def _inline(self, record):
        if record.inline_bytes > self.max_tensor_bytes:
            raise ValueError("DT inline tensor allocation limit exceeded.")
        return self._db.execute("SELECT data FROM tensors WHERE name=?", (record.name,)).fetchone()[
            0
        ]

    def _span(self, record, blob):
        codec = record.codec & ~_EXTERNAL
        expected = 24 if codec == _Q8P else 16
        if len(blob) != expected:
            raise ValueError("Invalid external DT tensor descriptor size.")
        block = struct.unpack_from("<I", blob)[0] if codec == _Q8P else None
        offset, length = struct.unpack_from("<QQ", blob, expected - 16)
        offset += self._base
        if offset > self._payload_identity[2] or length > self._payload_identity[2] - offset:
            raise ValueError("DT tensor span extends outside the model file.")
        return offset, length, block

    def _pread(self, offset, length):
        if length > self.max_tensor_bytes:
            raise ValueError("DT tensor read allocation limit exceeded.")
        start = time.perf_counter()
        value = os.pread(self._payload_fd, length, offset)
        self.read_seconds += time.perf_counter() - start
        self.payload_bytes_read += len(value)
        if len(value) != length:
            raise ValueError("Short DT tensor payload.")
        return value

    def _row_decode(self, quantized, scales):
        out = np.empty(quantized.shape, dtype=np.float16)
        for i in range(0, len(out), 64):
            exact = quantized[i : i + 64].astype(np.float32) * scales[i : i + 64, None].astype(
                np.float32
            )
            rounded = exact.astype(np.float16)
            if self.rounding == "toward_zero":
                rounded = np.where(
                    np.abs(rounded.astype(np.float32)) > np.abs(exact),
                    np.nextafter(rounded, np.float16(0)),
                    rounded,
                )
            out[i : i + 64] = rounded
        return out

    def validate_tensor(self, name, *, row_access=False):
        """Validate codec sizes using only metadata and small codec headers."""
        record = self._record(name)
        codec = record.codec & ~_EXTERNAL
        external = bool(record.codec & _EXTERNAL)
        if row_access and (not external or codec not in {_RAW, _I8X}):
            raise ValueError("DT embedding requires external raw or int8 row access.")
        n = record.elements
        length, block, offset = record.inline_bytes, None, None
        if external:
            offset, length, block = self._span(record, self._inline(record))
        if codec == _RAW:
            expected = n * 2
        elif codec == _I8X:
            expected = (n + 127) // 128 * 128 + n // record.shape[-1] * 2
        else:
            if length < 4:
                raise ValueError("DT codec header size mismatch.")
            if codec == _Q8P and external:
                header = None
            elif external:
                header = os.pread(self._payload_fd, 4, offset)
            else:
                header = self._db.execute(
                    "SELECT substr(data,1,4) FROM tensors WHERE name=?", (name,)
                ).fetchone()[0]
            if codec == _Q8P:
                if not external:
                    block = struct.unpack("<I", header)[0]
                if not block or block > 1048576:
                    raise ValueError("DT palette block size mismatch.")
                expected = n + (n + block - 1) // block * 512 + (0 if external else 4)
            else:
                expected = 4 + struct.unpack("<I", header)[0] + n
        if length != expected:
            raise ValueError(f"DT tensor encoded size mismatch: {name}")
        self._check()
        return record

    def read_rows(self, name: str, start: int, count: int) -> np.ndarray:
        record = self._record(name)
        codec = record.codec & ~_EXTERNAL
        if not record.codec & _EXTERNAL or codec not in {_RAW, _I8X}:
            raise ValueError("DT row reads require external FP16 or row-int8 tensors.")
        cols = record.shape[-1]
        rows = record.elements // cols
        if start < 0 or count < 1 or start + count > rows:
            raise ValueError("DT tensor row range is invalid.")
        if count * cols * 2 > self.max_tensor_bytes:
            raise ValueError("DT decoded tensor allocation limit exceeded.")
        offset, length, _ = self._span(record, self._inline(record))
        if codec == _RAW:
            if length != record.elements * 2:
                raise ValueError("DT raw tensor size mismatch.")
            data = self._pread(offset + start * cols * 2, count * cols * 2)
            out = np.frombuffer(data, dtype="<f2").reshape(count, cols).copy()
        else:
            scale_offset = (record.elements + 127) // 128 * 128
            if length != scale_offset + rows * 2:
                raise ValueError("DT int8 tensor size mismatch.")
            q = np.frombuffer(
                self._pread(offset + start * cols, count * cols), dtype=np.int8
            ).reshape(count, cols)
            scales = np.frombuffer(
                self._pread(offset + scale_offset + start * 2, count * 2), dtype="<f2"
            )
            started = time.perf_counter()
            out = self._row_decode(q, scales)
            self.decode_seconds += time.perf_counter() - started
        self._check()
        self.tensors_read += 1
        return out

    def read(self, name: str) -> np.ndarray:
        record = self._record(name)
        n = record.elements
        if n * 2 > self.max_tensor_bytes:
            raise ValueError("DT decoded tensor allocation limit exceeded; use row reads.")
        codec = record.codec & ~_EXTERNAL
        if record.codec & _EXTERNAL and codec in {_RAW, _I8X}:
            return self.read_rows(name, 0, n // record.shape[-1]).reshape(record.shape)
        blob = self._inline(record)
        block = None
        if record.codec & _EXTERNAL:
            offset, length, block = self._span(record, blob)
            blob = self._pread(offset, length)
        else:
            self.payload_bytes_read += len(blob)
            if codec == _Q8P:
                if len(blob) < 4:
                    raise ValueError("Missing DT palette block size.")
                block = struct.unpack_from("<I", blob)[0]
                blob = blob[4:]
        started = time.perf_counter()
        if codec == _RAW:
            if len(blob) != n * 2:
                raise ValueError("DT raw tensor size mismatch.")
            out = np.frombuffer(blob, dtype="<f2").copy()
        elif codec == _I8X:
            cols = record.shape[-1]
            offset = (n + 127) // 128 * 128
            if len(blob) != offset + n // cols * 2:
                raise ValueError("DT int8 tensor size mismatch.")
            q = np.frombuffer(blob, dtype=np.int8, count=n).reshape(-1, cols)
            scales = np.frombuffer(blob, dtype="<f2", offset=offset)
            out = self._row_decode(q, scales)
        elif codec == _Q8P:
            if not block or block > 1048576 or len(blob) != n + (n + block - 1) // block * 512:
                raise ValueError("DT palette tensor size mismatch.")
            out = np.empty(n, dtype=np.float16)
            for i in range((n + block - 1) // block):
                offset = i * (block + 512)
                palette = np.frombuffer(blob, dtype="<f2", count=256, offset=offset)
                count = min(block, n - i * block)
                indices = np.frombuffer(blob, dtype=np.uint8, count=count, offset=offset + 512)
                out[i * block : i * block + count] = palette[indices]
        else:
            if len(blob) < 4:
                raise ValueError("Missing DT compressed exponent size.")
            zipped = struct.unpack_from("<I", blob)[0]
            if len(blob) != 4 + zipped + n:
                raise ValueError("DT compressed tensor size mismatch.")
            decoder = zlib.decompressobj(-15)
            exp = decoder.decompress(blob[4 : 4 + zipped], n + 1)
            if len(exp) != n or not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid DT compressed exponent stream.")
            exp = np.frombuffer(exp, dtype=np.uint8).astype(np.uint16)
            if np.any(exp > 31):
                raise ValueError("Invalid DT exponent.")
            sm = np.frombuffer(blob, dtype=np.uint8, offset=4 + zipped).astype(np.uint16)
            out = (((sm >> 7) << 15) | (exp << 10) | ((sm & 127) << 3)).view(np.float16)
        self.decode_seconds += time.perf_counter() - started
        self._check()
        self.tensors_read += 1
        return out.reshape(record.shape)

    def report(self):
        return {
            "format": "draw-things",
            "payload_bytes_read": self.payload_bytes_read,
            "read_seconds": self.read_seconds,
            "decode_seconds": self.decode_seconds,
            "tensors_read": self.tensors_read,
            "persistent_weight_bytes_written": 0,
        }
