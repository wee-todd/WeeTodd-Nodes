"""Decode DT packed tensors on demand on Metal, retaining the CPU weight values.

This changes weight preparation only, not projection precision or sampler math.
The inherited reader owns read-only handles and validates all spans and limits.
MLX is imported only when a weighted tensor is requested.
"""

from __future__ import annotations

import mmap
import struct
import time
from contextlib import contextmanager, nullcontext
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
        self.mapped_payload_reads = 0
        self.mapped_payload_bytes = 0
        self.mapped_payload_fallbacks = 0
        self._defer_decode = False
        self.deferred_decoded_tensors = 0
        self.batched_materializations = 0
        self.batched_materialization_seconds = 0.0
        self.native_layout_groups = 0

    def read_native_group(self, names, *, layout):
        """Read a bounded int8 group directly into H3's BF16 weight layout.

        Unsupported codecs/rounding return None before reading payloads so the
        caller can use ordinary reads. All spans and input allocation limits are
        checked before GPU allocation. The combined output is at most three
        individually bounded tensors; neither it nor the payloads are retained.
        """
        import mlx.core as mx

        from .dt_native_layout import decode_group

        if layout not in {"qkv", "fc1"} or len(names) != (3 if layout == "qkv" else 2):
            raise ValueError("Invalid native DT tensor group.")
        records = [self._record(name) for name in names]
        if self.rounding != "nearest" or any(r.codec & ~_EXTERNAL != _I8X for r in records):
            return None
        shape = records[0].shape
        if (
            len(shape) != 2
            or any(r.shape != shape for r in records)
            or (layout == "qkv" and shape[0] % 128)
        ):
            raise ValueError("Invalid native DT tensor group shape.")
        if any(r.elements * 2 > self.max_tensor_bytes for r in records):
            raise ValueError("DT decoded tensor allocation limit exceeded.")
        if records[0].elements * len(records) >= 2**32:
            raise ValueError("DT native tensor group index limit exceeded.")
        for name in names:
            self.validate_tensor(name)
        started = time.perf_counter()
        inputs = []
        for record in records:
            blob = self._inline(record)
            if record.codec & _EXTERNAL:
                offset, length, _ = self._span(record, blob)
                payload = self._payload_window(offset, length)
            else:
                self.payload_bytes_read += len(blob)
                payload = nullcontext(blob)
            with payload as blob:
                inputs.extend(
                    [
                        mx.array(np.frombuffer(blob, dtype=np.int8, count=record.elements)),
                        mx.array(
                            np.frombuffer(
                                blob, dtype="<f2", offset=(record.elements + 127) // 128 * 128
                            )
                        ),
                    ]
                )
        out = decode_group(inputs, shape, layout=layout)
        if self._defer_decode:
            self.deferred_decoded_tensors += len(records)
        else:
            mx.eval(out)
        self._check()
        elapsed = time.perf_counter() - started
        self.decode_seconds += elapsed
        self.gpu_decode_seconds += elapsed
        self.gpu_decoded_tensors += len(records)
        self.tensors_read += len(records)
        self.native_layout_groups += 1
        return out

    def materialize(self, prepare):
        """Prepare one bounded block and complete its graph in a single evaluation.

        Packed inputs are owned MLX copies before their file windows close. The caller
        must restrict this to one block without AdaLN weights; fixed weights and the
        initial modulation pass retain eager reads. No arrays are retained by the store.
        """
        import mlx.core as mx

        self._check()
        if self._defer_decode:
            raise RuntimeError("DT block materialization cannot be nested.")
        self._defer_decode = True
        started = time.perf_counter()
        values = None
        try:
            values = prepare()
            mx.eval(values)
            self._check()
            self.batched_materializations += 1
            return values
        finally:
            values = None
            self._defer_decode = False
            self.batched_materialization_seconds += time.perf_counter() - started

    @contextmanager
    def _payload_window(self, offset, length):
        """Expose one bounded read-only window until its owned array has materialized.

        The context closes the mapping even when an exception retains its traceback.
        Neither model arrays nor the store retain a view into the original file.
        Mapping faults occur during array copying and are included in decode time.
        """
        self._check()
        if length > self.max_tensor_bytes:
            raise ValueError("DT tensor read allocation limit exceeded.")
        size = self._payload_identity[2] if self._payload_identity else 0
        if offset < 0 or length < 0 or offset > size or length > size - offset:
            raise ValueError("DT tensor span extends outside the model file.")
        if not length:
            yield b""
            return
        start = time.perf_counter()
        base = offset // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        try:
            region = mmap.mmap(
                self._payload_fd, length + offset - base, access=mmap.ACCESS_READ, offset=base
            )
        except (OSError, ValueError):
            self.mapped_payload_fallbacks += 1
            yield super()._pread(offset, length)
            return
        view = memoryview(region)[offset - base :]
        self.read_seconds += time.perf_counter() - start
        self.payload_bytes_read += length
        self.mapped_payload_reads += 1
        self.mapped_payload_bytes += length
        try:
            yield view
        finally:
            view.release()
            region.close()

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
            payload = self._payload_window(offset, length)
        else:
            self.payload_bytes_read += len(blob)
            if codec == _Q8P:
                block = struct.unpack_from("<I", blob)[0]
                blob = blob[4:]
            payload = nullcontext(blob)
        with payload as blob:
            return self._decode_packed(record, blob, block)

    def _decode_packed(self, record, blob, block):
        import mlx.core as mx

        n = record.elements
        codec = record.codec & ~_EXTERNAL
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
        if self._defer_decode:
            self.deferred_decoded_tensors += 1
        else:
            mx.eval(out)
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
            "mapped_payload_reads": self.mapped_payload_reads,
            "mapped_payload_bytes": self.mapped_payload_bytes,
            "mapped_payload_fallbacks": self.mapped_payload_fallbacks,
            "deferred_decoded_tensors": self.deferred_decoded_tensors,
            "batched_materializations": self.batched_materializations,
            "batched_materialization_seconds": self.batched_materialization_seconds,
            "native_layout_groups": self.native_layout_groups,
            "decode_timing_scope": (
                "eager completion or deferred submission; batches timed separately"
            ),
        }
