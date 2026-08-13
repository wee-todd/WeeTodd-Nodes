"""Optional BF16 MPP video feed-forward projections for LTX 2.5."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from threading import RLock

import mlx.core as mx
import mlx.nn as nn

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

_SOURCE = """
auto matrix_a = tensor(
    (device bfloat*)source,
    dextents<int, 2>{INPUT_DIM, ROWS},
    array<int, 2>{1, INPUT_DIM});
auto matrix_b = tensor(
    (device bfloat*)weight,
    dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
    array<int, 2>{1, INPUT_DIM});
auto matrix_c = tensor(
    (device bfloat*)output,
    dextents<int, 2>{OUTPUT_DIM, ROWS},
    array<int, 2>{1, OUTPUT_DIM});
constexpr auto descriptor = matmul2d_descriptor(
    32, 64, static_cast<int>(dynamic_extent), false, true, false);
matmul2d<descriptor, execution_simdgroups<2>> operation;
auto tile_a = matrix_a.slice(0, threadgroup_position_in_grid.y * 32);
auto tile_b = matrix_b.slice(0, threadgroup_position_in_grid.x * 64);
auto tile_c = matrix_c.slice(
    threadgroup_position_in_grid.x * 64,
    threadgroup_position_in_grid.y * 32);
auto result = operation.template get_destination_cooperative_tensor<
    decltype(tile_a), decltype(tile_b), bfloat>();
#pragma unroll
for (ushort index = 0; index < result.get_capacity(); ++index) {
  result[index] = bfloat(0.0f);
}
operation.run(tile_a, tile_b, result);
result.store(tile_c);
"""

_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_ltx25_mpp_bf16_ff",
    input_names=["source", "weight"],
    output_names=["output"],
    source=_SOURCE,
    header=_HEADER,
)


@dataclass(frozen=True)
class FeedForwardBackendReport:
    """Resolved video feed-forward precision and projection backend."""

    requested: str
    resolved: str
    wrapped_projections: int
    approximate: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _RuntimeStatus:
    def __init__(self) -> None:
        self._lock = RLock()
        self.calls = 0
        self.cast_elements = 0

    def record(self, source: mx.array) -> None:
        with self._lock:
            self.calls += 1
            self.cast_elements += math.prod(source.shape)

    def report(self) -> dict[str, int]:
        with self._lock:
            return {"mpp_calls": self.calls, "bf16_cast_elements": self.cast_elements}

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.cast_elements = 0


_STATUS = _RuntimeStatus()


def mpp_capability() -> tuple[bool, str | None]:
    if platform.system() != "Darwin":
        return False, "BF16 MPP feed-forward projections require macOS"
    release = platform.mac_ver()[0]
    if not release or int(release.split(".", 1)[0]) < 26:
        return False, "BF16 MPP feed-forward projections require macOS 26 or newer"
    if not hasattr(mx.fast, "metal_kernel"):
        return False, "the installed MLX version has no custom Metal kernel API"
    return True, None


def mpp_bf16_linear(source: mx.array, weight: mx.array) -> mx.array:
    """Compute source @ weight.T through the measured LTX 2.5 MPP tile."""
    if source.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise TypeError("LTX 2.5 MPP feed-forward projection requires BF16 arrays")
    rows = math.prod(source.shape[:-1])
    input_dim = int(source.shape[-1])
    output_dim, weight_input_dim = map(int, weight.shape)
    if input_dim != weight_input_dim:
        raise ValueError("LTX 2.5 MPP feed-forward dimensions do not match")
    threads = 64
    return _KERNEL(
        inputs=[source, weight],
        template=[("ROWS", rows), ("OUTPUT_DIM", output_dim), ("INPUT_DIM", input_dim)],
        grid=(math.ceil(output_dim / 64) * threads, math.ceil(rows / 32), 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(*source.shape[:-1], output_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]


class BF16MPPLinear(nn.Module):
    """Cast one bias-free projection input to BF16 and execute it with MPP."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        if getattr(base, "bias", None) is not None:
            raise ValueError("LTX 2.5 BF16 MPP mode supports bias-free video FF projections only")
        self.base = base
        self.enabled = True

    def __call__(self, source: mx.array) -> mx.array:
        if not self.enabled:
            return self.base(source)
        source = source.astype(mx.bfloat16)
        _STATUS.record(source)
        return mpp_bf16_linear(source, self.base.weight)


def set_mpp_feed_forward_enabled(model, enabled: bool) -> int:
    """Enable or bypass every wrapped projection without reloading weights."""
    updated = 0
    for block in model.transformer_blocks:
        for name in ("proj_in", "proj_out"):
            layer = getattr(block.ff, name)
            if isinstance(layer, BF16MPPLinear):
                layer.enabled = bool(enabled)
                updated += 1
    return updated


def configure_feed_forward_backend(model, requested: str) -> FeedForwardBackendReport:
    """Wrap the 48 video FF pairs after strict checkpoint loading."""
    _STATUS.reset()
    if requested not in {"reference_fp32", "bf16_mpp_experimental"}:
        raise ValueError("feed-forward backend must be reference_fp32 or bf16_mpp_experimental")
    if requested == "reference_fp32":
        return FeedForwardBackendReport(requested, requested, 0, False)
    supported, reason = mpp_capability()
    if not supported:
        return FeedForwardBackendReport(requested, "reference_fp32", 0, False, reason)

    wrapped = 0
    for block in model.transformer_blocks:
        for name in ("proj_in", "proj_out"):
            layer = getattr(block.ff, name)
            if isinstance(layer, BF16MPPLinear):
                continue
            if not isinstance(layer, nn.Linear) or layer.weight.dtype != mx.bfloat16:
                continue
            setattr(block.ff, name, BF16MPPLinear(layer))
            wrapped += 1
    return FeedForwardBackendReport(
        requested,
        "bf16_mpp_experimental" if wrapped else "reference_fp32",
        wrapped,
        bool(wrapped),
        None if wrapped else "no compatible BF16 video feed-forward projections",
    )


def feed_forward_runtime_status() -> dict[str, int]:
    return _STATUS.report()


def reset_feed_forward_runtime_status() -> None:
    _STATUS.reset()


__all__ = [
    "BF16MPPLinear",
    "FeedForwardBackendReport",
    "configure_feed_forward_backend",
    "feed_forward_runtime_status",
    "mpp_bf16_linear",
    "mpp_capability",
    "reset_feed_forward_runtime_status",
    "set_mpp_feed_forward_enabled",
]
