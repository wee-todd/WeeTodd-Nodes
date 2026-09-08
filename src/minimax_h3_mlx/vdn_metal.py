"""FP32 batched Cholesky inverse for VDN's 128-wide positive-definite systems.

Each thread owns one factor row; only the current column is shared. This avoids
materializing the full factor in threadgroup memory. A second kernel solves the
triangular systems, followed by MLX's batched matrix product. No iteration-count
approximation, lower-precision factor, or regularization change is introduced.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _temporal_kernel():
    return mx.fast.metal_kernel(
        name="weetodd_vdn_temporal_five_tap",
        input_names=["spatial", "weight"], output_names=["result"],
        source=r"""
        uint i = thread_position_in_grid.x;
        if (i >= SIZE) return;
        int frame = i / FRAME_STRIDE;
        uint channel = i % CHANNELS;
        T sum = T(0);
        for (int tap = 0; tap < 5; ++tap) {
            int source_frame = frame + tap - 2;
            T product = T(0);
            if (source_frame >= 0 && source_frame < FRAMES) {
                uint source = source_frame * FRAME_STRIDE + i % FRAME_STRIDE;
                product = T(float(spatial[source]) * float(weight[channel * 5 + tap]));
            }
            // Preserve the released BF16 multiply/add rounding at every tap.
            sum = T(float(sum) + float(product));
        }
        result[i] = sum;
        """,
    )


def temporal_five_tap(spatial, weight):
    """One pass over the temporal filter, with zero-padding and original tap order."""
    if spatial.ndim != 4 or weight.shape != (spatial.shape[-1], 1, 5):
        raise ValueError("VDN temporal filter requires FHWC input and C-by-1-by-5 weights.")
    if spatial.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("VDN temporal filter requires floating-point input.")
    return _temporal_kernel()(
        inputs=[spatial, weight.astype(spatial.dtype)],
        template=[("T", spatial.dtype), ("SIZE", spatial.size),
                  ("CHANNELS", spatial.shape[-1]), ("FRAMES", spatial.shape[0]),
                  ("FRAME_STRIDE", spatial.size // spatial.shape[0])],
        grid=(spatial.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[spatial.shape], output_dtypes=[spatial.dtype],
    )[0]


class VDNFeatureKernels:
    """Verify fused temporal filtering once per geometry; latch/report any fallback."""

    def __init__(self):
        self.verified_shapes = set()
        self.fallback_reason = None
        self.metal_calls = 0
        self.reference_calls = 0

    def temporal(self, spatial, weight, reference):
        if self.fallback_reason is None:
            try:
                result = temporal_five_tap(spatial, weight)
                key = (tuple(spatial.shape), spatial.dtype)
                if key not in self.verified_shapes:
                    expected = reference(spatial, weight)
                    valid = mx.all(mx.isfinite(result)) & mx.allclose(
                        result, expected, atol=1e-6, rtol=1e-6
                    )
                    mx.eval(valid)
                    if not valid.item():
                        raise ValueError("Fused VDN temporal filter failed parity verification.")
                    self.verified_shapes.add(key)
                self.metal_calls += 1
                return result
            except (RuntimeError, ValueError) as exc:
                self.fallback_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
        self.reference_calls += 1
        return reference(spatial, weight)

    def report(self):
        return {"metal_calls": self.metal_calls, "reference_calls": self.reference_calls,
                "verified_geometries": len(self.verified_shapes),
                "fallback_reason": self.fallback_reason}


@lru_cache(maxsize=1)
def _kernels():
    cholesky = mx.fast.metal_kernel(
        name="weetodd_vdn_cholesky_fp32",
        input_names=["a"], output_names=["l"],
        source=r"""
        uint row = thread_position_in_threadgroup.x;
        uint batch = threadgroup_position_in_grid.x;
        uint base = batch * D * D;
        float values[D];
        for (uint j = 0; j < D; ++j) values[j] = a[base + row * D + j];
        threadgroup float column[D];
        threadgroup float diagonal;
        for (uint k = 0; k < D; ++k) {
            if (row == k) diagonal = metal::sqrt(values[k]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float entry = row >= k ? values[k] / diagonal : 0.0f;
            column[row] = entry;
            l[base + row * D + k] = entry;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row > k) {
                for (uint j = k + 1; j <= row; ++j)
                    values[j] = metal::fma(-entry, column[j], values[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        """,
    )
    triangular_inverse = mx.fast.metal_kernel(
        name="weetodd_vdn_triangular_inverse_fp32",
        input_names=["l"], output_names=["result"],
        source=r"""
        uint column = thread_position_in_threadgroup.x;
        uint batch = threadgroup_position_in_grid.x;
        uint base = batch * D * D;
        float values[D];
        for (uint row = 0; row < D; ++row) {
            float value = row == column ? 1.0f : 0.0f;
            for (uint j = 0; j < row; ++j)
                value = metal::fma(-l[base + row * D + j], values[j], value);
            values[row] = value / l[base + row * D + row];
            result[base + row * D + column] = values[row];
        }
        """,
    )
    return cholesky, triangular_inverse


def _metal_inverse(matrix):
    if matrix.dtype != mx.float32 or matrix.shape[-2:] != (128, 128):
        raise ValueError("VDN Metal inverse requires FP32 128-by-128 matrices.")
    cholesky, triangular_inverse = _kernels()
    options = {
        "template": [("D", 128)],
        "grid": (matrix.size // 128, 1, 1),
        "threadgroup": (128, 1, 1),
        "output_shapes": [matrix.shape],
        "output_dtypes": [mx.float32],
    }
    lower = cholesky(inputs=[matrix], **options)[0]
    inverse_lower = triangular_inverse(inputs=[lower], **options)[0]
    return inverse_lower.swapaxes(-1, -2) @ inverse_lower


class VDNMatrixSolver:
    """Run-local verified Metal dispatch, with a latched and reported CPU fallback.

    The first text/video batch of each geometry is compared against the original
    CPU inverse. Later calls stay asynchronous on Metal. Only scalar verification
    results cross to the host; no matrix arrays are retained in the solver.
    """

    def __init__(self):
        self.verified_shapes: set[tuple[int, ...]] = set()
        self.fallback_reason: str | None = None
        self.metal_calls = 0
        self.cpu_calls = 0
        self.validation_max_error = 0.0

    def inverse(self, matrix):
        eligible = matrix.dtype == mx.float32 and matrix.shape[-2:] == (128, 128)
        if eligible and self.fallback_reason is None:
            try:
                result = _metal_inverse(matrix)
                shape = tuple(matrix.shape)
                if shape not in self.verified_shapes:
                    reference = mx.linalg.inv(matrix, stream=mx.cpu)
                    error = mx.max(mx.abs(result - reference))
                    valid = mx.all(mx.isfinite(result)) & mx.all(
                        mx.abs(result - reference) <= 2e-5 + 2e-4 * mx.abs(reference)
                    )
                    mx.eval(error, valid)
                    if not valid.item():
                        raise ValueError("Metal Cholesky inverse failed CPU parity verification.")
                    self.validation_max_error = max(self.validation_max_error, float(error.item()))
                    self.verified_shapes.add(shape)
                self.metal_calls += 1
                return result
            except (RuntimeError, ValueError) as exc:
                self.fallback_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
        self.cpu_calls += 1
        return mx.linalg.inv(matrix, stream=mx.cpu)

    def report(self):
        return {
            "backend": (
                "mixed_metal_cpu" if self.metal_calls and self.cpu_calls else
                "metal_cholesky_fp32" if self.metal_calls else "cpu_inverse"
            ),
            "metal_calls": self.metal_calls,
            "cpu_calls": self.cpu_calls,
            "verified_geometries": len(self.verified_shapes),
            "validation_max_error": self.validation_max_error,
            "fallback_reason": self.fallback_reason,
        }
