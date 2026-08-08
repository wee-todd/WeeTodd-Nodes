"""Exact-layout Steel attention tile probes built from the installed MLX headers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass(frozen=True)
class SteelAttentionTile:
    """Classic Steel query/key tiles and their simdgroup layout."""

    query_rows: int
    key_rows: int
    simdgroups_m: int
    simdgroups_n: int = 1

    def validate(self) -> None:
        values = (
            self.query_rows,
            self.key_rows,
            self.simdgroups_m,
            self.simdgroups_n,
        )
        if min(values) < 1:
            raise ValueError("Steel attention tile values must be positive")
        if self.query_rows != 8 * self.simdgroups_m * self.simdgroups_n:
            raise ValueError("Steel attention query rows must equal eight rows per simdgroup")
        if self.key_rows % 8:
            raise ValueError("Steel attention key rows must be divisible by eight")


_LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"\s*$')


def _mlx_include_root() -> Path:
    return Path(mx.__file__).resolve().parent / "include"


def _expand_header(path: Path, include_root: Path, seen: set[Path]) -> str:
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    output = []
    for line in path.read_text().splitlines():
        match = _LOCAL_INCLUDE.match(line)
        if match:
            child = include_root / match.group(1)
            if not child.is_file():
                raise FileNotFoundError(f"Installed MLX Metal header not found: {child}")
            output.append(_expand_header(child, include_root, seen))
        elif line.strip() != "#pragma once":
            output.append(line)
    return "\n".join(output)


def _extract_attention_parts(header: str) -> tuple[str, str]:
    marker = "[[kernel, max_total_threads_per_threadgroup"
    kernel_start = header.index(marker)
    body_start = header.index("{", kernel_start)
    depth = 0
    body_end = None
    for index in range(body_start, len(header)):
        char = header[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                body_end = index
                break
    if body_end is None:
        raise ValueError("Installed MLX Steel attention kernel has no complete body")

    prefix_start = header.index("struct MaxOp")
    template_start = header.rfind("// clang-format off", prefix_start, kernel_start)
    if template_start < prefix_start:
        raise ValueError("Installed MLX Steel attention kernel has an unknown declaration layout")
    prefix = header[prefix_start:template_start]
    return prefix, header[body_start + 1 : body_end]


def _steel_sources() -> tuple[str, str]:
    include_root = _mlx_include_root()
    kernel_path = include_root / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
    if not kernel_path.is_file():
        raise FileNotFoundError(
            "The installed MLX package does not include the classic Steel attention headers"
        )
    raw = kernel_path.read_text()
    prefix, body = _extract_attention_parts(raw)
    dependency = include_root / "mlx/backend/metal/kernels/steel/attn/attn.h"
    expanded = _expand_header(dependency, include_root, set())
    return f"{expanded}\n{prefix}", body


_HEADER, _ATTENTION_BODY = _steel_sources()
_KERNELS: dict[SteelAttentionTile, object] = {}


def _kernel_source(tile: SteelAttentionTile) -> str:
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


def _kernel(tile: SteelAttentionTile):
    kernel = _KERNELS.get(tile)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=(
                f"wee_todd_steel_attention_bq{tile.query_rows}_bk{tile.key_rows}_"
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


def steel_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    tile: SteelAttentionTile,
) -> mx.array:
    """Run unmasked BF16 H3 attention and return physical ``[B, L, H, D]`` output."""
    tile.validate()
    if query.dtype != mx.bfloat16 or key.dtype != mx.bfloat16 or value.dtype != mx.bfloat16:
        raise TypeError("Steel attention requires BF16 query, key, and value arrays")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Steel attention requires matching query, key, and value shapes")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("Steel attention requires H3 shape [1, heads, rows, 128]")
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Steel attention requires the H3 128-dimension attention scale")
    _, heads, rows, head_dim = query.shape
    threadgroup = (32, tile.simdgroups_m, tile.simdgroups_n)
    return _kernel(tile)(
        inputs=[query, key, value],
        template=[
            ("ROWS", rows),
            ("HEADS", heads),
        ],
        grid=(
            math.ceil(rows / tile.query_rows) * threadgroup[0],
            heads * threadgroup[1],
            threadgroup[2],
        ),
        threadgroup=threadgroup,
        output_shapes=[(1, rows, heads, head_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
