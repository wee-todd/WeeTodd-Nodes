"""Capability-gated H3 NAX attention probes built from installed MLX headers."""

from __future__ import annotations

import math
import platform
import re
from dataclasses import dataclass

import mlx.core as mx

from .steel_attention import _expand_header, _extract_attention_parts, _mlx_include_root


@dataclass(frozen=True)
class NAXAttentionTile:
    """NAX query/key tile and simdgroup layout used by MLX 0.32."""

    query_rows: int = 64
    key_rows: int = 32
    simdgroups_m: int = 4
    simdgroups_n: int = 1

    def validate(self) -> None:
        values = (
            self.query_rows,
            self.key_rows,
            self.simdgroups_m,
            self.simdgroups_n,
        )
        if min(values) < 1:
            raise ValueError("NAX attention tile values must be positive")
        if self.query_rows != 16 * self.simdgroups_m * self.simdgroups_n:
            raise ValueError(
                "NAX attention query rows must equal sixteen rows per simdgroup"
            )
        if self.key_rows % 32:
            raise ValueError("NAX attention key rows must be divisible by thirty-two")


def apple_gpu_generation(architecture: str) -> int | None:
    """Return the numeric generation from an MLX Apple GPU architecture string."""
    match = re.fullmatch(r"applegpu_g(\d+)[a-z]", architecture.lower())
    return int(match.group(1)) if match else None


def nax_attention_available(
    *,
    device_info: dict[str, object] | None = None,
    macos_version: str | None = None,
) -> bool:
    """Mirror MLX 0.32's public NAX hardware and operating-system gate."""
    info = mx.device_info() if device_info is None else device_info
    architecture = str(info.get("architecture", "")).lower()
    generation = apple_gpu_generation(architecture)
    if generation is None or not architecture:
        return False
    minimum_generation = 18 if architecture.endswith("p") else 17
    if generation < minimum_generation:
        return False

    version = platform.mac_ver()[0] if macos_version is None else macos_version
    try:
        parts = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    return parts >= (26, 2)


def _nax_sources() -> tuple[str, str]:
    include_root = _mlx_include_root()
    kernel_path = (
        include_root
        / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
    )
    if not kernel_path.is_file():
        raise FileNotFoundError(
            "The installed MLX package does not include the NAX attention headers"
        )
    raw = kernel_path.read_text()
    prefix, body = _extract_attention_parts(raw)
    dependency = include_root / "mlx/backend/metal/kernels/steel/attn/nax.h"
    expanded = _expand_header(dependency, include_root, set())
    return f"{expanded}\n{prefix}", body


_HEADER, _ATTENTION_BODY = _nax_sources()
_KERNELS: dict[NAXAttentionTile, object] = {}
_DEFAULT_TILE = NAXAttentionTile()


def _kernel_source(tile: NAXAttentionTile) -> str:
    tile.validate()
    return f"""
using namespace mlx::steel;
using T = bfloat;
using MaskType = bfloat;
using AccumType = float;
constexpr int BQ = {tile.query_rows};
constexpr int BK = {tile.key_rows};
constexpr int BD = 128;
constexpr int WM = {tile.simdgroups_m};
constexpr int WN = {tile.simdgroups_n};
constexpr float SCALE = 0.08838834764831845f;
constexpr bool align_Q = (ROWS % BQ) == 0;
constexpr bool align_K = (ROWS % BK) == 0;
constexpr bool has_mask = false;
constexpr bool do_causal = false;
constexpr bool has_sinks = false;

const device T* Q = q;
const device T* K = k;
const device T* V = v;
device T* O = output;
AttnParams params_value{{
    1, HEADS, BD,
    ROWS, ROWS,
    1, SCALE,
    (ROWS + BQ - 1) / BQ,
    (ROWS + BK - 1) / BK,
    ROWS / BQ,
    ROWS / BK,
    ROWS - (ROWS / BQ) * BQ,
    ROWS - (ROWS / BK) * BK,
    0,
    {{q_strides[0], q_strides[1], q_strides[2]}},
    {{k_strides[0], k_strides[1], k_strides[2]}},
    {{v_strides[0], v_strides[1], v_strides[2]}},
    {{ROWS * HEADS * BD, BD, HEADS * BD}}
}};
thread const AttnParams* params = &params_value;
thread const AttnMaskParams* mask_params = nullptr;
const device MaskType* mask = nullptr;
const device T* sinks = nullptr;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;
uint3 tid = threadgroup_position_in_grid;
uint3 lid = thread_position_in_threadgroup;
{_ATTENTION_BODY}
"""


def _kernel(tile: NAXAttentionTile):
    kernel = _KERNELS.get(tile)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=(
                f"wee_todd_nax_attention_bq{tile.query_rows}_bk{tile.key_rows}_"
                f"wm{tile.simdgroups_m}_wn{tile.simdgroups_n}"
            ),
            input_names=["q", "k", "v"],
            output_names=["output"],
            source=_kernel_source(tile),
            header=_HEADER,
            ensure_row_contiguous=False,
        )
        _KERNELS[tile] = kernel
    return kernel


def nax_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    tile: NAXAttentionTile = _DEFAULT_TILE,
) -> mx.array:
    """Run unmasked BF16 H3 NAX attention and return physical ``[B, L, H, D]``."""
    tile.validate()
    if not nax_attention_available():
        info = mx.device_info()
        raise RuntimeError(
            "NAX attention is unavailable on this MLX/macOS/device combination "
            f"({info.get('device_name', 'unknown')}, {info.get('architecture', 'unknown')})"
        )
    if query.dtype != mx.bfloat16 or key.dtype != mx.bfloat16 or value.dtype != mx.bfloat16:
        raise TypeError("NAX attention requires BF16 query, key, and value arrays")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("NAX attention requires matching query, key, and value shapes")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("NAX attention requires H3 shape [1, heads, rows, 128]")
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("NAX attention requires the H3 128-dimension attention scale")

    _, heads, rows, head_dim = query.shape
    threadgroup = (32, tile.simdgroups_m, tile.simdgroups_n)
    return _kernel(tile)(
        inputs=[query, key, value],
        template=[("ROWS", rows), ("HEADS", heads)],
        grid=(
            math.ceil(rows / tile.query_rows) * threadgroup[0],
            heads * threadgroup[1],
            threadgroup[2],
        ),
        threadgroup=threadgroup,
        output_shapes=[(1, rows, heads, head_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
