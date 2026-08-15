"""Research-only MPP H3 QKV projection with fused normalization and rotary output."""

from __future__ import annotations

import math

import mlx.core as mx

from minimax_h3_mlx.projection import MPP_HEADER

_SOURCE = r"""
constexpr uint TILE_M = 32;
constexpr uint TILE_N = 128;
constexpr uint SIMDGROUPS = 4;
constexpr uint HEAD_DIM = 128;
constexpr uint ROTARY_DIM = 96;

const uint output_tile = threadgroup_position_in_grid.x;
const uint kind = output_tile % 3;
const uint head = output_tile / 3;
const uint row_start = threadgroup_position_in_grid.y * TILE_M;

auto matrix_a = tensor(
    (device bfloat*)source,
    dextents<int, 2>{INPUT_DIM, ROWS},
    array<int, 2>{1, INPUT_DIM});
auto matrix_b = tensor(
    (device bfloat*)weight,
    dextents<int, 2>{INPUT_DIM, OUTPUT_DIM},
    array<int, 2>{1, INPUT_DIM});

constexpr auto descriptor = matmul2d_descriptor(
    TILE_M,
    TILE_N,
    static_cast<int>(dynamic_extent),
    false,
    true,
    false);
matmul2d<descriptor, execution_simdgroups<SIMDGROUPS>> operation;

auto tile_a = matrix_a.slice(0, row_start);
auto tile_b = matrix_b.slice(0, output_tile * TILE_N);
auto result = operation.template get_destination_cooperative_tensor<
    decltype(tile_a), decltype(tile_b), bfloat>();
#pragma unroll
for (ushort index = 0; index < result.get_capacity(); ++index) {
    result[index] = bfloat(0.0f);
}
operation.run(tile_a, tile_b, result);

threadgroup bfloat projected[TILE_M * TILE_N];
auto projected_tensor = tensor(
    projected,
    dextents<int, 2>{TILE_N, TILE_M},
    array<int, 2>{1, TILE_N});
result.store(projected_tensor);
threadgroup_barrier(mem_flags::mem_threadgroup);

const uint lane = thread_index_in_simdgroup;
const uint simdgroup = simdgroup_index_in_threadgroup;
if (kind == 2) {
    for (uint index = thread_index_in_threadgroup; index < TILE_M * TILE_N;
         index += SIMDGROUPS * 32) {
        const uint local_row = index / TILE_N;
        const uint dimension = index % TILE_N;
        const uint row = row_start + local_row;
        if (row < ROWS) {
            const uint destination = (head * ROWS + row) * HEAD_DIM + dimension;
            value[destination] = projected[index];
        }
    }
} else {
    for (uint local_row = simdgroup; local_row < TILE_M; local_row += SIMDGROUPS) {
        const uint row = row_start + local_row;
        if (row >= ROWS) {
            continue;
        }
        float squares = 0.0f;
        for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
            const float projected_value = float(projected[local_row * TILE_N + dimension]);
            squares += projected_value * projected_value;
        }
        const float inverse_rms = rsqrt(simd_sum(squares) / float(HEAD_DIM) + 1.0e-5f);
        for (uint dimension = lane; dimension < HEAD_DIM; dimension += 32) {
            const float norm_weight = kind == 0
                ? float(q_weight[dimension])
                : float(k_weight[dimension]);
            const bfloat normalized_value = bfloat(
                float(projected[local_row * TILE_N + dimension]) * inverse_rms);
            bfloat normalized = bfloat(float(normalized_value) * norm_weight);
            if (dimension < ROTARY_DIM) {
                const bool first_half = dimension < ROTARY_DIM / 2;
                const uint paired = first_half
                    ? dimension + ROTARY_DIM / 2
                    : dimension - ROTARY_DIM / 2;
                const float paired_weight = kind == 0
                    ? float(q_weight[paired])
                    : float(k_weight[paired]);
                const bfloat paired_normalized = bfloat(
                    float(projected[local_row * TILE_N + paired]) * inverse_rms);
                const bfloat paired_value = bfloat(float(paired_normalized) * paired_weight);
                const bfloat cosine = bfloat(cosine_table[row * ROTARY_DIM + dimension]);
                const bfloat sine = bfloat(sine_table[row * ROTARY_DIM + dimension]);
                const bfloat direct = bfloat(float(normalized) * float(cosine));
                const bfloat rotated = bfloat(float(paired_value) * float(sine));
                normalized = bfloat(
                    float(direct) + (first_half ? -1.0f : 1.0f) * float(rotated));
            }
            const uint destination = (head * ROWS + row) * HEAD_DIM + dimension;
            if (kind == 0) {
                query[destination] = normalized;
            } else {
                key[destination] = normalized;
            }
        }
    }
}
"""

_LORA_SOURCE = _SOURCE.replace(
    "operation.run(tile_a, tile_b, result);",
    r"""
operation.run(tile_a, tile_b, result);
auto matrix_lora_a = tensor(
    (device bfloat*)lora_hidden,
    dextents<int, 2>{LORA_RANK, ROWS},
    array<int, 2>{1, LORA_RANK});
auto matrix_lora_b = tensor(
    (device bfloat*)lora_weight,
    dextents<int, 2>{LORA_RANK, OUTPUT_DIM},
    array<int, 2>{1, LORA_RANK});
auto tile_lora_a = matrix_lora_a.slice(0, row_start);
auto tile_lora_b = matrix_lora_b.slice(0, output_tile * TILE_N);
auto lora_result = operation.template get_destination_cooperative_tensor<
    decltype(tile_lora_a), decltype(tile_lora_b), bfloat>();
#pragma unroll
for (ushort index = 0; index < lora_result.get_capacity(); ++index) {
    lora_result[index] = bfloat(0.0f);
}
operation.run(tile_lora_a, tile_lora_b, lora_result);
#pragma unroll
for (ushort index = 0; index < result.get_capacity(); ++index) {
    result[index] += lora_result[index];
}
""",
)

_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_h3_fused_mpp_qkv_bf16",
    input_names=["source", "weight", "q_weight", "k_weight", "cosine_table", "sine_table"],
    output_names=["query", "key", "value"],
    source=_SOURCE,
    header=MPP_HEADER,
)

_LORA_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_h3_fused_mpp_qkv_lora_bf16",
    input_names=[
        "source",
        "weight",
        "q_weight",
        "k_weight",
        "cosine_table",
        "sine_table",
        "lora_hidden",
        "lora_weight",
    ],
    output_names=["query", "key", "value"],
    source=_LORA_SOURCE,
    header=MPP_HEADER,
)


def fused_mpp_qkv(
    source: mx.array,
    weight: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    cosine: mx.array,
    sine: mx.array,
    *,
    lora_hidden: mx.array | None = None,
    lora_weight: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Project H3 BF16 rows directly into normalized, rotary head-major Q/K/V."""
    if source.ndim != 3 or source.shape[0] != 1 or source.dtype != mx.bfloat16:
        raise ValueError("Fused MPP QKV source must be BF16 [1, rows, input_dim].")
    if weight.ndim != 2 or weight.dtype != mx.bfloat16:
        raise ValueError("Fused MPP QKV weight must be a BF16 matrix.")
    rows = int(source.shape[1])
    input_dim = int(source.shape[2])
    output_dim = int(weight.shape[0])
    if weight.shape[1] != input_dim or output_dim % (3 * 128):
        raise ValueError("Fused MPP QKV weight must contain complete 3-by-128 head groups.")
    heads = output_dim // (3 * 128)
    if q_weight.shape != (128,) or k_weight.shape != (128,):
        raise ValueError("Fused MPP QKV requires 128-channel Q/K RMSNorm weights.")
    if q_weight.dtype != mx.bfloat16 or k_weight.dtype != mx.bfloat16:
        raise TypeError("Fused MPP QKV requires BF16 Q/K RMSNorm weights.")
    if cosine.shape != (rows, 96) or sine.shape != cosine.shape:
        raise ValueError("Fused MPP QKV requires matching [rows, 96] rotary tables.")
    if cosine.dtype not in {mx.float32, mx.bfloat16} or sine.dtype != cosine.dtype:
        raise TypeError("Fused MPP QKV rotary tables must use matching FP32 or BF16 values.")

    with_lora = lora_hidden is not None or lora_weight is not None
    if with_lora:
        if lora_hidden is None or lora_weight is None:
            raise ValueError("Fused MPP QKV LoRA inputs must be supplied together.")
        if lora_hidden.shape[:2] != source.shape[:2]:
            raise ValueError("Fused MPP QKV LoRA rows must match the projection source.")
        rank = int(lora_hidden.shape[-1])
        if lora_weight.shape != (output_dim, rank):
            raise ValueError("Fused MPP QKV LoRA weight shape does not match its hidden rank.")
        if lora_hidden.dtype != mx.bfloat16 or lora_weight.dtype != mx.bfloat16:
            raise TypeError("Fused MPP QKV LoRA inputs must use BF16 values.")

    shape = (1, heads, rows, 128)
    thread_count = 4 * 32
    kernel = _LORA_KERNEL if with_lora else _KERNEL
    inputs = [source, weight, q_weight, k_weight, cosine, sine]
    template = [("ROWS", rows), ("INPUT_DIM", input_dim), ("OUTPUT_DIM", output_dim)]
    if with_lora:
        inputs.extend((lora_hidden, lora_weight))
        template.append(("LORA_RANK", int(lora_hidden.shape[-1])))
    outputs = kernel(
        inputs=inputs,
        template=template,
        grid=(heads * 3 * thread_count, math.ceil(rows / 32), 1),
        threadgroup=(thread_count, 1, 1),
        output_shapes=[shape, shape, shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    return outputs[0], outputs[1], outputs[2]


__all__ = ["fused_mpp_qkv"]
