"""Research-only MPP low-bit matrix multiplication probes."""

from __future__ import annotations

import math

import mlx.core as mx

from minimax_h3_mlx.projection import MPP_HEADER, MPPTile

_SOURCE = """
auto matrix_a = tensor(
    (device bfloat*)source,
    dextents<int, 2>{INPUT_DIM, ROWS},
    array<int, 2>{1, INPUT_DIM});
WEIGHT_TENSOR
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

_INT8_WEIGHT = """
auto matrix_b = tensor(
    (device int8_t*)weight,
    dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
    array<int, 2>{1, INPUT_DIM});
"""

# Packed format tensors use a byte data handle with logical nibble extents and strides.
_INT4_WEIGHT = """
tensor<device int4b_format, dextents<int, 2>, tensor_inline> matrix_b(
    (device uchar*)weight,
    dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
    array<int, 2>{1, INPUT_DIM});
"""


def _make_kernel(name: str, weight_tensor: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=["source", "weight"],
        output_names=["output"],
        source=_SOURCE.replace("WEIGHT_TENSOR", weight_tensor),
        header=MPP_HEADER,
    )


_INT8_KERNEL = _make_kernel("wee_todd_mpp_bf16_int8_nt_matmul", _INT8_WEIGHT)
_INT4_KERNEL = _make_kernel("wee_todd_mpp_bf16_int4_nt_matmul", _INT4_WEIGHT)
_DEFAULT_TILE = MPPTile()


def mpp_low_bit_linear(
    source: mx.array,
    weight: mx.array,
    *,
    bits: int,
    output_dim: int | None = None,
    tile: MPPTile = _DEFAULT_TILE,
) -> mx.array:
    """Compute a raw BF16-by-Int8 or BF16-by-packed-Int4 projection.

    This operation deliberately has no quantization scales or biases. It is an operator probe,
    not a drop-in implementation of MLX affine quantized matrix multiplication.
    """
    if source.dtype != mx.bfloat16:
        raise TypeError("MPP low-bit projection requires a BF16 source array")
    if source.ndim < 2 or weight.ndim != 2:
        raise ValueError("MPP low-bit projection requires matrix-shaped source and weight")
    input_dim = int(source.shape[-1])
    rows = math.prod(source.shape[:-1])

    if bits == 8:
        if weight.dtype != mx.int8:
            raise TypeError("MPP Int8 projection requires an Int8 weight array")
        resolved_output_dim = int(weight.shape[0])
        if int(weight.shape[1]) != input_dim:
            raise ValueError("MPP Int8 weight input width does not match the source")
        kernel = _INT8_KERNEL
    elif bits == 4:
        if weight.dtype != mx.uint8:
            raise TypeError("MPP packed Int4 projection requires a UInt8 storage array")
        if input_dim % 2:
            raise ValueError("MPP packed Int4 projection requires an even input width")
        resolved_output_dim = int(output_dim or weight.shape[0])
        expected_shape = (resolved_output_dim, input_dim // 2)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(f"MPP packed Int4 weight storage must have shape {expected_shape}")
        kernel = _INT4_KERNEL
    else:
        raise ValueError("MPP low-bit projection supports only 8-bit and packed 4-bit weights")

    thread_count = 32 * tile.simdgroups
    return kernel(
        inputs=[source, weight],
        template=[
            ("ROWS", rows),
            ("OUTPUT_DIM", resolved_output_dim),
            ("INPUT_DIM", input_dim),
            ("TILE_M", tile.rows),
            ("TILE_N", tile.columns),
            ("SIMDGROUPS", tile.simdgroups),
        ],
        grid=(
            math.ceil(resolved_output_dim / tile.columns) * thread_count,
            math.ceil(rows / tile.rows),
            1,
        ),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[(*source.shape[:-1], resolved_output_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
