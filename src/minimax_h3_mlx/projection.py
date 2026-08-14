"""Optional MLX-owned Metal Performance Primitives projections for MiniMax H3."""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from threading import RLock

import mlx.core as mx
import mlx.nn as nn

MPP_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
"""

MPP_SOURCE = """
// MPP does not accept a const-qualified bfloat tensor element type. The operation does not write
// either input, so remove the generated MLX signature's const qualifier before tensor wrapping.
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
    TILE_M,
    TILE_N,
    static_cast<int>(dynamic_extent),
    false,
    true,
    false);
matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;

auto tile_a = matrix_a.slice(0, threadgroup_position_in_grid.y * TILE_M);
auto tile_b = matrix_b.slice(0, threadgroup_position_in_grid.x * TILE_N);
auto tile_c = matrix_c.slice(
    threadgroup_position_in_grid.x * TILE_N,
    threadgroup_position_in_grid.y * TILE_M);
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
    name="wee_todd_mpp_bf16_nt_matmul",
    input_names=["source", "weight"],
    output_names=["output"],
    source=MPP_SOURCE,
    header=MPP_HEADER,
)


@dataclass(frozen=True)
class MPPTile:
    """Compile-time MPP output tile and cooperative simdgroup count."""

    rows: int = 32
    columns: int = 64
    simdgroups: int = 2

    def __post_init__(self) -> None:
        if min(self.rows, self.columns, self.simdgroups) < 1:
            raise ValueError("MPP tile dimensions and simdgroup count must be positive")


@dataclass(frozen=True)
class ProjectionBackendReport:
    """Result of applying one projection backend to a loaded transformer."""

    requested: str
    resolved: str
    wrapped_projections: int
    skipped_projections: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_DEFAULT_TILE = MPPTile()
_FC2_TILE = MPPTile(64, 128, 8)
_AUTO_VERIFIED_ARCHITECTURES = frozenset({"applegpu_g15d"})


def mpp_capability() -> tuple[bool, str | None]:
    """Return the static Metal 4 capability gate without compiling a kernel."""
    if platform.system() != "Darwin":
        return False, "MPP projections require macOS"
    release = platform.mac_ver()[0]
    if not release:
        return False, "macOS release could not be determined"
    if int(release.split(".", 1)[0]) < 26:
        return False, "MPP projections require macOS 26 or newer"
    if not hasattr(mx.fast, "metal_kernel"):
        return False, "the installed MLX version has no custom Metal kernel API"
    return True, None


def mpp_auto_capability(
    device_info: dict[str, object] | None = None,
) -> tuple[bool, str | None]:
    """Return whether ``auto`` may select MPP from measured complete-generation evidence."""
    info = mx.device_info() if device_info is None else device_info
    architecture = str(info.get("architecture", "")).lower()
    if architecture not in _AUTO_VERIFIED_ARCHITECTURES:
        label = architecture or "unknown"
        return False, f"MPP auto selection is not validated for Apple GPU architecture {label}"
    return True, None


def select_mpp_tile(weight: mx.array) -> MPPTile:
    """Select the measured H3 BF16 tile for a checkpoint-oriented weight."""
    if tuple(weight.shape) == (5_376, 14_336):
        return _FC2_TILE
    return _DEFAULT_TILE


def mpp_bf16_linear(
    source: mx.array,
    weight: mx.array,
    *,
    tile: MPPTile = _DEFAULT_TILE,
) -> mx.array:
    """Compute ``source @ weight.T`` with an MLX-owned MPP BF16 kernel."""
    if source.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise TypeError("MPP projection requires BF16 source and weight arrays")
    if source.ndim < 2:
        raise ValueError("MPP projection source must have at least two dimensions")
    if weight.ndim != 2:
        raise ValueError("MPP projection weight must have two dimensions")
    input_dim = source.shape[-1]
    output_dim, weight_input_dim = weight.shape
    if input_dim != weight_input_dim:
        raise ValueError(
            "MPP projection input width does not match the stored weight input width"
        )

    rows = math.prod(source.shape[:-1])
    thread_count = 32 * tile.simdgroups
    return _KERNEL(
        inputs=[source, weight],
        template=[
            ("ROWS", rows),
            ("OUTPUT_DIM", output_dim),
            ("INPUT_DIM", input_dim),
            ("TILE_M", tile.rows),
            ("TILE_N", tile.columns),
            ("SIMDGROUPS", tile.simdgroups),
        ],
        grid=(
            math.ceil(output_dim / tile.columns) * thread_count,
            math.ceil(rows / tile.rows),
            1,
        ),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[(*source.shape[:-1], output_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]


class _MPPVerificationRegistry:
    """Verify each kernel shape once and retain a process-local fallback verdict."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._verdicts: dict[tuple[int, ...], bool] = {}
        self._fallbacks: dict[tuple[int, ...], str] = {}

    @staticmethod
    def _signature(source: mx.array, weight: mx.array, tile: MPPTile) -> tuple[int, ...]:
        return (
            math.prod(source.shape[:-1]),
            int(source.shape[-1]),
            int(weight.shape[0]),
            tile.rows,
            tile.columns,
            tile.simdgroups,
        )

    def project(self, source: mx.array, layer: nn.Linear, tile: MPPTile) -> mx.array:
        signature = self._signature(source, layer.weight, tile)
        with self._lock:
            verdict = self._verdicts.get(signature)
            if verdict is False:
                return layer(source)
            if verdict is True:
                return mpp_bf16_linear(source, layer.weight, tile=tile)

            reference = layer(source)
            try:
                candidate = mpp_bf16_linear(source, layer.weight, tile=tile)
                mx.eval(reference, candidate)
                exact = bool(mx.array_equal(reference, candidate).item())
            except Exception as exc:
                self._verdicts[signature] = False
                self._fallbacks[signature] = type(exc).__name__
                mx.clear_cache()
                return reference
            self._verdicts[signature] = exact
            if not exact:
                self._fallbacks[signature] = "bitwise_mismatch"
            # The first call always returns the verified standard MLX result.
            return reference

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "verified_signatures": sum(self._verdicts.values()),
                "fallback_signatures": len(self._verdicts) - sum(self._verdicts.values()),
                "fallback_reasons": sorted(set(self._fallbacks.values())),
            }

    def reset(self) -> None:
        with self._lock:
            self._verdicts.clear()
            self._fallbacks.clear()


_VERIFICATION = _MPPVerificationRegistry()


class MPPLinear(nn.Module):
    """Inference-only wrapper that preserves the loaded MLX linear layer and its weights."""

    def __init__(self, base: nn.Linear, tile: MPPTile | None = None):
        super().__init__()
        self.base = base
        self.tile = tile or select_mpp_tile(base.weight)
        self.input_dims = int(base.weight.shape[1])
        self.output_dims = int(base.weight.shape[0])

    def __call__(self, source: mx.array) -> mx.array:
        if source.dtype != mx.bfloat16 or self.base.weight.dtype != mx.bfloat16:
            return self.base(source)
        return _VERIFICATION.project(source, self.base, self.tile)


def _eligible_linear(layer) -> bool:
    return (
        isinstance(layer, nn.Linear)
        and not isinstance(layer, MPPLinear)
        and getattr(layer, "weight", None) is not None
        and layer.weight.dtype == mx.bfloat16
        and "bias" not in layer
    )


def configure_projection_backend(dit, requested: str) -> ProjectionBackendReport:
    """Apply the requested projection backend to the 50-block H3 transformer stack.

    ``auto`` selects the verified Metal Performance Primitives path when the runtime supports it.
    Each real projection shape still runs the process-local bitwise verification gate on its first
    use. Unsupported dtypes, quantized layers, biases, kernel failures, and numerical mismatches
    retain the standard MLX implementation.
    """
    if requested not in {"auto", "mlx", "mpp_experimental"}:
        raise ValueError("projection backend must be auto, mlx, or mpp_experimental")
    if requested == "mlx":
        return ProjectionBackendReport("mlx", "mlx", 0, 0)
    supported, reason = mpp_capability()
    if not supported:
        return ProjectionBackendReport(requested, "mlx", 0, 0, reason)
    if requested == "auto":
        supported, reason = mpp_auto_capability()
        if not supported:
            return ProjectionBackendReport(requested, "mlx", 0, 0, reason)

    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        return ProjectionBackendReport(
            requested,
            "mlx",
            0,
            0,
            "the transformer exposes no resident core block stack",
        )

    wrapped = 0
    skipped = 0
    for block in blocks:
        for owner, name in (
            (block.attn, "qkv_proj"),
            (block.attn, "out_proj"),
            (block.mlp, "fc1"),
            (block.mlp, "fc2"),
        ):
            layer = getattr(owner, name)
            if _eligible_linear(layer):
                setattr(owner, name, MPPLinear(layer))
                wrapped += 1
            else:
                skipped += 1
    resolved = "mpp_experimental" if wrapped else "mlx"
    fallback_reason = None if wrapped else "no eligible BF16 projections"
    return ProjectionBackendReport(requested, resolved, wrapped, skipped, fallback_reason)


def mpp_runtime_status() -> dict[str, object]:
    """Return process-local verification and fallback counts for generation metadata."""
    return _VERIFICATION.status()


def reset_mpp_runtime_status() -> None:
    """Clear verification state for focused tests and explicit runtime teardown."""
    _VERIFICATION.reset()
