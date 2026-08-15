"""Research-only MPP projection that consumes head-major H3 attention output."""

from __future__ import annotations

import math

import mlx.core as mx

from minimax_h3_mlx.projection import MPP_HEADER, MPPTile

_DEFAULT_TILE = MPPTile()

_SOURCE = r"""
auto matrix_c = tensor(
    (device bfloat*)output,
    dextents<int, 2>{OUTPUT_DIM, ROWS},
    array<int, 2>{1, OUTPUT_DIM});

constexpr auto descriptor = matmul2d_descriptor(
    TILE_M,
    TILE_N,
    HEAD_DIM,
    false,
    true,
    false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;

auto tile_c = matrix_c.slice(
    threadgroup_position_in_grid.x * TILE_N,
    threadgroup_position_in_grid.y * TILE_M);

const uint output_start = threadgroup_position_in_grid.x * TILE_N;
const uint output_count = min(uint(TILE_N), uint(OUTPUT_DIM) - output_start);

auto first_a = tensor(
    (device bfloat*)head_major,
    dextents<int, 2>{HEAD_DIM, ROWS},
    array<int, 2>{int(head_major_strides[3]), int(head_major_strides[2])});
auto first_b = tensor(
    (device bfloat*)weight + output_start * weight_strides[0],
    dextents<int, 2>{HEAD_DIM, output_count},
    array<int, 2>{int(weight_strides[1]), int(weight_strides[0])});
auto first_tile_a = first_a.slice(0, threadgroup_position_in_grid.y * TILE_M);
auto first_tile_b = first_b.slice(0, 0);
auto result = operation.template get_destination_cooperative_tensor<
    decltype(first_tile_a), decltype(first_tile_b), bfloat>();
#pragma unroll
for (ushort index = 0; index < result.get_capacity(); ++index) {
    result[index] = bfloat(0.0f);
}

for (uint head = 0; head < HEADS; ++head) {
    auto matrix_a = tensor(
        (device bfloat*)head_major + head * head_major_strides[1],
        dextents<int, 2>{HEAD_DIM, ROWS},
        array<int, 2>{int(head_major_strides[3]), int(head_major_strides[2])});
    auto matrix_b = tensor(
        (device bfloat*)weight
            + output_start * weight_strides[0]
            + head * HEAD_DIM * weight_strides[1],
        dextents<int, 2>{HEAD_DIM, output_count},
        array<int, 2>{int(weight_strides[1]), int(weight_strides[0])});
    auto tile_a = matrix_a.slice(0, threadgroup_position_in_grid.y * TILE_M);
    auto tile_b = matrix_b.slice(0, 0);
    operation.run(tile_a, tile_b, result);
}
result.store(tile_c);
"""

_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_h3_head_major_mpp_output_bf16",
    input_names=["head_major", "weight"],
    output_names=["output"],
    source=_SOURCE,
    header=MPP_HEADER,
    ensure_row_contiguous=False,
)


def head_major_mpp_output(
    head_major: mx.array,
    weight: mx.array,
    *,
    tile: MPPTile = _DEFAULT_TILE,
) -> mx.array:
    """Project ``[1, heads, rows, 128]`` without a global token-major temporary."""
    if head_major.ndim != 4 or head_major.shape[0] != 1 or head_major.shape[-1] != 128:
        raise ValueError("Head-major H3 output must have shape [1, heads, rows, 128].")
    if head_major.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise TypeError("Head-major MPP output projection requires BF16 arrays.")
    if weight.ndim != 2 or weight.shape[1] != head_major.shape[1] * 128:
        raise ValueError("Output weight input width must equal heads multiplied by 128.")
    heads = int(head_major.shape[1])
    rows = int(head_major.shape[2])
    output_dim = int(weight.shape[0])
    input_dim = int(weight.shape[1])
    thread_count = 32 * tile.simdgroups
    return _KERNEL(
        inputs=[head_major, weight],
        template=[
            ("HEADS", heads),
            ("HEAD_DIM", 128),
            ("ROWS", rows),
            ("INPUT_DIM", input_dim),
            ("OUTPUT_DIM", output_dim),
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
        output_shapes=[(1, rows, output_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]


__all__ = ["head_major_mpp_output"]
