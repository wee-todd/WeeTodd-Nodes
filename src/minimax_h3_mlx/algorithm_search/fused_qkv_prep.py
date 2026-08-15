"""Research-only fused H3 Q/K normalization, rotary, and QKV layout preparation."""

from __future__ import annotations

import mlx.core as mx

_FUSED_QKV_PREP_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint group = threadgroup_position_in_grid.x;
const uint vector_count = BATCH * ROWS * HEADS;
if (group >= vector_count) {
    return;
}

uint packed = group;
const uint head = packed % HEADS;
packed /= HEADS;
const uint row = packed % ROWS;
const uint batch = packed / ROWS;
const uint interleaved_base =
    ((batch * ROWS + row) * HEADS + head) * 3 * HEAD_DIM;
const uint output_base =
    ((batch * HEADS + head) * ROWS + row) * HEAD_DIM;

float q_squares = 0.0f;
float k_squares = 0.0f;
for (uint d = lane; d < HEAD_DIM; d += 32) {
    const float q_value = float(qkv[interleaved_base + d]);
    const float k_value = float(qkv[interleaved_base + HEAD_DIM + d]);
    q_squares += q_value * q_value;
    k_squares += k_value * k_value;
}
const float q_inverse_rms = rsqrt(simd_sum(q_squares) / float(HEAD_DIM) + 1.0e-5f);
const float k_inverse_rms = rsqrt(simd_sum(k_squares) / float(HEAD_DIM) + 1.0e-5f);

for (uint d = lane; d < HEAD_DIM; d += 32) {
    float q_value = float(qkv[interleaved_base + d])
        * q_inverse_rms * float(q_weight[d]);
    float k_value = float(qkv[interleaved_base + HEAD_DIM + d])
        * k_inverse_rms * float(k_weight[d]);
    if (d < ROTARY_DIM) {
        const bool first_half = d < ROTARY_DIM / 2;
        const uint paired = first_half ? d + ROTARY_DIM / 2 : d - ROTARY_DIM / 2;
        const float q_pair = float(qkv[interleaved_base + paired])
            * q_inverse_rms * float(q_weight[paired]);
        const float k_pair = float(qkv[interleaved_base + HEAD_DIM + paired])
            * k_inverse_rms * float(k_weight[paired]);
        const float cosine = float(cosine_table[row * ROTARY_DIM + d]);
        const float sine = float(sine_table[row * ROTARY_DIM + d]);
        const float sign = first_half ? -1.0f : 1.0f;
        q_value = q_value * cosine + sign * q_pair * sine;
        k_value = k_value * cosine + sign * k_pair * sine;
    }
    query[output_base + d] = bfloat(q_value);
    key[output_base + d] = bfloat(k_value);
    value[output_base + d] = qkv[interleaved_base + 2 * HEAD_DIM + d];
}
"""


_FUSED_QKV_PREP_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_h3_fused_qkv_prep_bf16",
    input_names=["qkv", "q_weight", "k_weight", "cosine_table", "sine_table"],
    output_names=["query", "key", "value"],
    source=_FUSED_QKV_PREP_SOURCE,
)


def fused_qkv_prep(
    qkv: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    cosine: mx.array,
    sine: mx.array,
    *,
    epsilon: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array]:
    """Prepare interleaved H3 QKV as head-major BF16 Q/K/V in one Metal pass."""
    if qkv.ndim != 5 or qkv.shape[-2:] != (3, 128):
        raise ValueError("Fused H3 QKV preparation requires [batch, rows, heads, 3, 128].")
    if qkv.dtype != mx.bfloat16:
        raise TypeError("Fused H3 QKV preparation requires BF16 projection output.")
    if float(epsilon) != 1e-5:
        raise ValueError("Fused H3 QKV preparation currently requires epsilon 1e-5.")
    batch, rows, heads = map(int, qkv.shape[:3])
    if q_weight.shape != (128,) or k_weight.shape != (128,):
        raise ValueError("Fused H3 QKV preparation requires 128-channel Q/K RMSNorm weights.")
    if q_weight.dtype != mx.bfloat16 or k_weight.dtype != mx.bfloat16:
        raise TypeError("Fused H3 QKV preparation requires BF16 Q/K RMSNorm weights.")
    if cosine.shape != sine.shape or cosine.ndim != 2 or cosine.shape[0] != rows:
        raise ValueError("Fused H3 QKV preparation requires matching per-row rotary tables.")
    rotary_dim = int(cosine.shape[1])
    if rotary_dim < 2 or rotary_dim > 128 or rotary_dim % 2:
        raise ValueError("Fused H3 rotary width must be positive, even, and at most 128.")
    if cosine.dtype not in {mx.float32, mx.bfloat16} or sine.dtype != cosine.dtype:
        raise TypeError("Fused H3 rotary tables must use matching FP32 or BF16 values.")
    output_shape = (batch, heads, rows, 128)
    vector_count = batch * rows * heads
    result = _FUSED_QKV_PREP_KERNEL(
        inputs=[qkv, q_weight, k_weight, cosine, sine],
        template=[
            ("BATCH", batch),
            ("ROWS", rows),
            ("HEADS", heads),
            ("HEAD_DIM", 128),
            ("ROTARY_DIM", rotary_dim),
        ],
        grid=(vector_count * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[output_shape, output_shape, output_shape],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16],
    )
    return result[0], result[1], result[2]


__all__ = ["fused_qkv_prep"]
