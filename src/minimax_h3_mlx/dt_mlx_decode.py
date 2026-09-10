"""Decode DT packed tensors on demand on Metal, retaining the CPU weight values.

This changes weight preparation only, not projection precision or sampler math.
The inherited reader owns read-only handles and validates all spans and limits.
MLX is imported only when a weighted tensor is requested.
"""

from __future__ import annotations

import struct
import time
from functools import lru_cache

import numpy as np

from .dt_tensor_store import _EXTERNAL, _I8X, _Q8P, DTTensorStore


@lru_cache(maxsize=1)
def _int8_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="weetodd_dt_int8_decode",
        input_names=["quantized", "scales"],
        output_names=["decoded"],
        source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= COUNT) return;
        decoded[i] = half(float(quantized[i]) * float(scales[i / COLUMNS]));
        """,
    )


@lru_cache(maxsize=1)
def _palette_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="weetodd_dt_palette_decode",
        input_names=["packed"],
        output_names=["bits"],
        source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= COUNT) return;
        uint base = (i / BLOCK) * (BLOCK + 512);
        uint index = packed[base + 512 + i % BLOCK];
        uint offset = base + index * 2;
        bits[i] = ushort(packed[offset]) | (ushort(packed[offset + 1]) << 8);
        """,
    )


class DTMLXTensorStore(DTTensorStore):
    """The same tensor values as DTTensorStore; packed expansion runs on the GPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_decoded_tensors = 0
        self.gpu_decode_seconds = 0.0

    def read(self, name):
        record = self._record(name)
        codec = record.codec & ~_EXTERNAL
        if codec not in {_I8X, _Q8P} or self.rounding != "nearest":
            return super().read(name)
        n = record.elements
        if n * 2 > self.max_tensor_bytes:
            raise ValueError("DT decoded tensor allocation limit exceeded; use row reads.")
        # Includes palette block bounds and exact encoded lengths before any GPU allocation.
        self.validate_tensor(name)
        blob = self._inline(record)
        block = None
        if record.codec & _EXTERNAL:
            offset, length, block = self._span(record, blob)
            blob = self._pread(offset, length)
        else:
            self.payload_bytes_read += len(blob)
            if codec == _Q8P:
                block = struct.unpack_from("<I", blob)[0]
                blob = blob[4:]
        import mlx.core as mx

        started = time.perf_counter()
        if codec == _I8X:
            cols = record.shape[-1]
            q = mx.array(np.frombuffer(blob, dtype=np.int8, count=n).reshape(-1, cols))
            scales = mx.array(np.frombuffer(blob, dtype="<f2", offset=(n + 127) // 128 * 128))
            # Keep the CPU reader's FP32 multiply, then round to FP16 *before*
            # the existing native BF16/FP32 policy. Direct BF16 multiplication differs.
            # A single kernel avoids full-size FP32 quantized/product scratch arrays.
            out = _int8_kernel()(
                inputs=[q, scales],
                template=[("COUNT", n), ("COLUMNS", cols)],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[record.shape],
                output_dtypes=[mx.float16],
            )[0]
        else:
            packed = mx.array(np.frombuffer(blob, dtype=np.uint8))
            bits = _palette_kernel()(
                inputs=[packed],
                template=[("COUNT", n), ("BLOCK", block)],
                grid=(n, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[record.shape],
                output_dtypes=[mx.uint16],
            )[0]
            out = bits.view(mx.float16)
        mx.eval(out)  # Bound pending inputs and include actual GPU completion in timing.
        elapsed = time.perf_counter() - started
        self.decode_seconds += elapsed
        self.gpu_decode_seconds += elapsed
        self.gpu_decoded_tensors += 1
        self.tensors_read += 1
        self._check()
        return out.reshape(record.shape)

    def report(self):
        return {
            **super().report(),
            "weight_decode_backend": "mlx",
            "gpu_decoded_tensors": self.gpu_decoded_tensors,
            "gpu_decode_seconds": self.gpu_decode_seconds,
        }
