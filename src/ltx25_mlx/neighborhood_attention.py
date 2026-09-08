"""Memory-bounded 3D neighborhood attention for an MLX LTX Diffusion VAE."""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import numpy as np

_METAL_NA3D_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint group = threadgroup_position_in_grid.x;
const uint query_head_count = QUERY_COUNT * HEADS;
if (group >= query_head_count) {
    return;
}

const uint local_query = group / HEADS;
uint packed = QUERY_START + local_query;
const uint query_head = group % HEADS;
const uint head = query_head;
const uint w = packed % WIDTH;
packed /= WIDTH;
const uint h = packed % HEIGHT;
packed /= HEIGHT;
const uint t = packed % FRAMES;
const uint batch = packed / FRAMES;

const int start_t = clamp(
    int(t) - int(KERNEL_T / 2),
    0,
    int(FRAMES - KERNEL_T));
const int start_h = clamp(
    int(h) - int(KERNEL_H / 2),
    0,
    int(HEIGHT - KERNEL_H));
const int start_w = clamp(
    int(w) - int(KERNEL_W / 2),
    0,
    int(WIDTH - KERNEL_W));
const uint query_base = local_query * HEADS * HEAD_DIM + query_head * HEAD_DIM;

float maximum = -INFINITY;
for (uint dt = 0; dt < KERNEL_T; ++dt) {
    for (uint dh = 0; dh < KERNEL_H; ++dh) {
        for (uint dw = 0; dw < KERNEL_W; ++dw) {
            const uint source_t = uint(start_t) + dt;
            const uint source_h = uint(start_h) + dh;
            const uint source_w = uint(start_w) + dw;
            const uint key_base =
                (((batch * FRAMES + source_t) * HEIGHT + source_h) * WIDTH + source_w)
                * HEADS * HEAD_DIM + head * HEAD_DIM;
            float partial = 0.0f;
            for (uint d = lane; d < HEAD_DIM; d += 32) {
                partial += float(q[query_base + d]) * float(k[key_base + d]);
            }
            const float score = simd_sum(partial) * 0.125f;
            maximum = max(maximum, score);
        }
    }
}

float denominator = 0.0f;
for (uint dt = 0; dt < KERNEL_T; ++dt) {
    for (uint dh = 0; dh < KERNEL_H; ++dh) {
        for (uint dw = 0; dw < KERNEL_W; ++dw) {
            const uint source_t = uint(start_t) + dt;
            const uint source_h = uint(start_h) + dh;
            const uint source_w = uint(start_w) + dw;
            const uint key_base =
                (((batch * FRAMES + source_t) * HEIGHT + source_h) * WIDTH + source_w)
                * HEADS * HEAD_DIM + head * HEAD_DIM;
            float partial = 0.0f;
            for (uint d = lane; d < HEAD_DIM; d += 32) {
                partial += float(q[query_base + d]) * float(k[key_base + d]);
            }
            const float score = simd_sum(partial) * 0.125f;
            denominator += precise::exp(score - maximum);
        }
    }
}

const uint first_dimension = lane;
const uint second_dimension = lane + 32;
float first_accumulated = 0.0f;
float second_accumulated = 0.0f;
for (uint dt = 0; dt < KERNEL_T; ++dt) {
    for (uint dh = 0; dh < KERNEL_H; ++dh) {
        for (uint dw = 0; dw < KERNEL_W; ++dw) {
            const uint source_t = uint(start_t) + dt;
            const uint source_h = uint(start_h) + dh;
            const uint source_w = uint(start_w) + dw;
            const uint key_base =
                (((batch * FRAMES + source_t) * HEIGHT + source_h) * WIDTH + source_w)
                * HEADS * HEAD_DIM + head * HEAD_DIM;
            float partial = 0.0f;
            for (uint kd = lane; kd < HEAD_DIM; kd += 32) {
                partial += float(q[query_base + kd]) * float(k[key_base + kd]);
            }
            const float score = simd_sum(partial) * 0.125f;
            const float probability = precise::exp(score - maximum) / denominator;
            first_accumulated += probability * float(v[key_base + first_dimension]);
            second_accumulated += probability * float(v[key_base + second_dimension]);
        }
    }
}
output[query_base + first_dimension] = bfloat(first_accumulated);
output[query_base + second_dimension] = bfloat(second_accumulated);
"""


_METAL_NORM_ROPE_SLICE_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint group = threadgroup_position_in_grid.x;
const uint vector_count = QUERY_COUNT * HEADS;
if (group >= vector_count) {
    return;
}

const uint local_query = group / HEADS;
uint packed = QUERY_START + local_query;
const uint head = group % HEADS;
const uint w = packed % WIDTH;
packed /= WIDTH;
const uint h = packed % HEIGHT;
packed /= HEIGHT;
const uint t = packed % FRAMES;
const uint base = local_query * HEADS * HEAD_DIM + head * HEAD_DIM;

float squares = 0.0f;
for (uint d = lane; d < HEAD_DIM; d += 32) {
    const float value = float(x[base + d]);
    squares += value * value;
}
const float inverse_rms = rsqrt(simd_sum(squares) / float(HEAD_DIM) + 1.0e-6f);

const uint pair = lane;
uint local_pair;
uint position;
uint table_offset;
float cosine;
float sine;
if (pair < 8) {
    local_pair = pair;
    position = t;
    table_offset = (position * 8 + local_pair) * 2;
    cosine = time_rope[table_offset];
    sine = time_rope[table_offset + 1];
} else if (pair < 20) {
    local_pair = pair - 8;
    position = h;
    table_offset = (position * 12 + local_pair) * 2;
    cosine = height_rope[table_offset];
    sine = height_rope[table_offset + 1];
} else {
    local_pair = pair - 20;
    position = w;
    table_offset = (position * 12 + local_pair) * 2;
    cosine = width_rope[table_offset];
    sine = width_rope[table_offset + 1];
}

const uint even_dimension = pair * 2;
const uint odd_dimension = even_dimension + 1;
const float even = float(x[base + even_dimension]) * inverse_rms
    * float(weight[even_dimension]);
const float odd = float(x[base + odd_dimension]) * inverse_rms
    * float(weight[odd_dimension]);
output[base + even_dimension] = bfloat(even * cosine - odd * sine);
output[base + odd_dimension] = bfloat(even * sine + odd * cosine);
"""


_METAL_QK_NORM_ROPE_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint group = threadgroup_position_in_grid.x;
const uint vector_count = BATCH * FRAMES * HEIGHT * WIDTH * HEADS;
if (group >= vector_count) {
    return;
}

uint packed = group;
const uint head = packed % HEADS;
packed /= HEADS;
const uint w = packed % WIDTH;
packed /= WIDTH;
const uint h = packed % HEIGHT;
packed /= HEIGHT;
const uint t = packed % FRAMES;
const uint batch = packed / FRAMES;
const uint base =
    (((batch * FRAMES + t) * HEIGHT + h) * WIDTH + w) * HEADS * HEAD_DIM
    + head * HEAD_DIM;

float q_squares = 0.0f;
float k_squares = 0.0f;
for (uint d = lane; d < HEAD_DIM; d += 32) {
    const float q_value = float(q[base + d]);
    const float k_value = float(k[base + d]);
    q_squares += q_value * q_value;
    k_squares += k_value * k_value;
}
const float q_inverse_rms = rsqrt(simd_sum(q_squares) / float(HEAD_DIM) + 1.0e-6f);
const float k_inverse_rms = rsqrt(simd_sum(k_squares) / float(HEAD_DIM) + 1.0e-6f);

const uint pair = lane;
uint local_pair;
uint axis_dimension;
uint position;
if (pair < 8) {
    local_pair = pair;
    axis_dimension = 16;
    position = t;
} else if (pair < 20) {
    local_pair = pair - 8;
    axis_dimension = 24;
    position = h;
} else {
    local_pair = pair - 20;
    axis_dimension = 24;
    position = w;
}
const uint even_dimension = pair * 2;
const uint odd_dimension = even_dimension + 1;
uint table_offset;
float cosine;
float sine;
if (pair < 8) {
    table_offset = (position * 8 + local_pair) * 2;
    cosine = time_rope[table_offset];
    sine = time_rope[table_offset + 1];
} else if (pair < 20) {
    table_offset = (position * 12 + local_pair) * 2;
    cosine = height_rope[table_offset];
    sine = height_rope[table_offset + 1];
} else {
    table_offset = (position * 12 + local_pair) * 2;
    cosine = width_rope[table_offset];
    sine = width_rope[table_offset + 1];
}

const float q_even =
    float(q[base + even_dimension]) * q_inverse_rms * float(q_weight[even_dimension]);
const float q_odd =
    float(q[base + odd_dimension]) * q_inverse_rms * float(q_weight[odd_dimension]);
const float k_even =
    float(k[base + even_dimension]) * k_inverse_rms * float(k_weight[even_dimension]);
const float k_odd =
    float(k[base + odd_dimension]) * k_inverse_rms * float(k_weight[odd_dimension]);
q_output[base + even_dimension] = bfloat(q_even * cosine - q_odd * sine);
q_output[base + odd_dimension] = bfloat(q_even * sine + q_odd * cosine);
k_output[base + even_dimension] = bfloat(k_even * cosine - k_odd * sine);
k_output[base + odd_dimension] = bfloat(k_even * sine + k_odd * cosine);
"""


_METAL_NA3D_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_ltx25_shifted_na3d_bf16",
    input_names=["q", "k", "v"],
    output_names=["output"],
    source=_METAL_NA3D_SOURCE,
)


_METAL_QK_NORM_ROPE_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_ltx25_qk_rmsnorm_rope_bf16",
    input_names=[
        "q",
        "k",
        "q_weight",
        "k_weight",
        "time_rope",
        "height_rope",
        "width_rope",
    ],
    output_names=["q_output", "k_output"],
    source=_METAL_QK_NORM_ROPE_SOURCE,
)


_METAL_NORM_ROPE_SLICE_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_ltx25_rmsnorm_rope_slice_bf16",
    input_names=["x", "weight", "time_rope", "height_rope", "width_rope"],
    output_names=["output"],
    source=_METAL_NORM_ROPE_SLICE_SOURCE,
)


@lru_cache(maxsize=16)
def _rope_tables_3d(frames: int, height: int, width: int) -> tuple[mx.array, ...]:
    tables = []
    for length, dimension in ((frames, 16), (height, 24), (width, 24)):
        inverse = mx.exp(
            -math.log(10000.0) * mx.arange(0, dimension, 2, dtype=mx.float32) / dimension
        )
        angles = mx.arange(length, dtype=mx.float32)[:, None] * inverse[None]
        table = mx.stack((mx.cos(angles), mx.sin(angles)), axis=-1)
        mx.eval(table)
        tables.append(table)
    return tuple(tables)


def metal_qk_rmsnorm_rope_3d(
    q: mx.array,
    k: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fuse trained Q/K RMSNorm and three-axis RoPE without FP32 tensor materialization."""
    if q.shape != k.shape or q.ndim != 6:
        raise ValueError("Metal Q/K preparation requires matching six-dimensional tensors.")
    if q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16:
        raise TypeError("Metal Q/K preparation requires BF16 tensors.")
    batch, frames, height, width, heads, head_dim = map(int, q.shape)
    if head_dim != 64 or q_weight.shape != (64,) or k_weight.shape != (64,):
        raise ValueError("Metal Q/K preparation requires the trained 64-dimension head layout.")
    time_rope, height_rope, width_rope = _rope_tables_3d(frames, height, width)
    groups = batch * frames * height * width * heads
    return tuple(
        _METAL_QK_NORM_ROPE_KERNEL(
            inputs=[q, k, q_weight, k_weight, time_rope, height_rope, width_rope],
            template=[
                ("BATCH", batch),
                ("FRAMES", frames),
                ("HEIGHT", height),
                ("WIDTH", width),
                ("HEADS", heads),
                ("HEAD_DIM", head_dim),
            ],
            grid=(groups * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[q.shape, k.shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
    )


def metal_neighborhood_attention_3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    kernel: tuple[int, int, int],
    scale: float,
) -> mx.array:
    """Run shifted BF16 NA3D without materializing gathered K/V neighborhoods."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 6:
        raise ValueError("Metal NA3D requires matching six-dimensional Q/K/V.")
    if q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16:
        raise TypeError("Metal NA3D requires BF16 query, key, and value tensors.")
    if abs(float(scale) - 0.125) > 1e-12:
        raise ValueError("Metal NA3D currently requires the trained head scale of 0.125.")
    batch, frames, height, width, heads, head_dim = map(int, q.shape)
    if head_dim != 64:
        raise ValueError("Metal NA3D currently requires the trained 64-dimension head layout.")
    dimensions = (frames, height, width)
    if any(dimension < size for dimension, size in zip(dimensions, kernel, strict=True)):
        raise ValueError("Metal shifted NA3D requires every dimension to cover its kernel.")
    query_count = batch * frames * height * width
    groups = query_count * heads
    return _METAL_NA3D_KERNEL(
        inputs=[q, k, v],
        template=[
            ("BATCH", batch),
            ("FRAMES", frames),
            ("HEIGHT", height),
            ("WIDTH", width),
            ("HEADS", heads),
            ("HEAD_DIM", head_dim),
            ("QUERY_START", 0),
            ("QUERY_COUNT", query_count),
            ("KERNEL_T", int(kernel[0])),
            ("KERNEL_H", int(kernel[1])),
            ("KERNEL_W", int(kernel[2])),
        ],
        grid=(groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def metal_rmsnorm_rope_3d_slice(
    x: mx.array,
    weight: mx.array,
    *,
    full_shape: tuple[int, int, int, int, int, int],
    query_start: int,
) -> mx.array:
    """Prepare one contiguous query slice using its coordinates in the complete volume."""
    batch, frames, height, width, heads, head_dim = map(int, full_shape)
    if x.ndim != 3 or x.shape[1:] != (heads, head_dim):
        raise ValueError("Metal sliced Q/K preparation requires (queries, heads, head_dim).")
    if x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16:
        raise TypeError("Metal sliced Q/K preparation requires BF16 tensors and weights.")
    if head_dim != 64 or weight.shape != (64,):
        raise ValueError("Metal sliced Q/K preparation requires the trained 64-dimension layout.")
    query_count = int(x.shape[0])
    total_queries = batch * frames * height * width
    if query_start < 0 or query_start + query_count > total_queries:
        raise ValueError("Metal sliced Q/K preparation exceeds the complete query volume.")
    time_rope, height_rope, width_rope = _rope_tables_3d(frames, height, width)
    groups = query_count * heads
    return _METAL_NORM_ROPE_SLICE_KERNEL(
        inputs=[x, weight, time_rope, height_rope, width_rope],
        template=[
            ("BATCH", batch),
            ("FRAMES", frames),
            ("HEIGHT", height),
            ("WIDTH", width),
            ("HEADS", heads),
            ("HEAD_DIM", head_dim),
            ("QUERY_START", int(query_start)),
            ("QUERY_COUNT", query_count),
        ],
        grid=(groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def metal_neighborhood_attention_3d_slice(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    query_start: int,
    kernel: tuple[int, int, int],
    scale: float,
) -> mx.array:
    """Attend one contiguous Q slice against complete prepared K/V volumes."""
    if k.shape != v.shape or k.ndim != 6:
        raise ValueError("Metal sliced NA3D requires matching complete K/V volumes.")
    batch, frames, height, width, heads, head_dim = map(int, k.shape)
    if q.ndim != 3 or q.shape[1:] != (heads, head_dim):
        raise ValueError("Metal sliced NA3D requires (queries, heads, head_dim) Q input.")
    if q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16:
        raise TypeError("Metal sliced NA3D requires BF16 query, key, and value tensors.")
    if head_dim != 64 or abs(float(scale) - 0.125) > 1e-12:
        raise ValueError("Metal sliced NA3D requires the trained head layout and scale.")
    dimensions = (frames, height, width)
    if any(dimension < size for dimension, size in zip(dimensions, kernel, strict=True)):
        raise ValueError("Metal shifted NA3D requires every dimension to cover its kernel.")
    query_count = int(q.shape[0])
    total_queries = batch * frames * height * width
    if query_start < 0 or query_start + query_count > total_queries:
        raise ValueError("Metal sliced NA3D exceeds the complete query volume.")
    groups = query_count * heads
    return _METAL_NA3D_KERNEL(
        inputs=[q, k, v],
        template=[
            ("BATCH", batch),
            ("FRAMES", frames),
            ("HEIGHT", height),
            ("WIDTH", width),
            ("HEADS", heads),
            ("HEAD_DIM", head_dim),
            ("QUERY_START", int(query_start)),
            ("QUERY_COUNT", query_count),
            ("KERNEL_T", int(kernel[0])),
            ("KERNEL_H", int(kernel[1])),
            ("KERNEL_W", int(kernel[2])),
        ],
        grid=(groups * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


@lru_cache(maxsize=32)
def _neighborhood_index(
    frames: int,
    height: int,
    width: int,
    kernel: tuple[int, int, int],
    dilation: tuple[int, int, int],
    boundary_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    radii = tuple(size // 2 for size in kernel)
    padding = tuple(radius * step for radius, step in zip(radii, dilation, strict=True))
    padded_shape = tuple(
        size + 2 * pad for size, pad in zip((frames, height, width), padding, strict=True)
    )
    coordinates = np.stack(
        np.meshgrid(np.arange(frames), np.arange(height), np.arange(width), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    if boundary_mode == "shift":
        axes = []
        for dimension, size, step in zip((frames, height, width), kernel, dilation, strict=True):
            span = (size - 1) * step + 1
            if dimension < span:
                raise ValueError(
                    "Shifted neighborhood attention requires each dimension to cover its kernel."
                )
            centers = np.arange(dimension, dtype=np.int64)
            starts = np.clip(centers - (size // 2) * step, 0, dimension - span)
            axes.append(starts[:, None] + np.arange(size, dtype=np.int64)[None] * step)
        t_idx = axes[0][coordinates[:, 0]]
        h_idx = axes[1][coordinates[:, 1]]
        w_idx = axes[2][coordinates[:, 2]]
        source = np.stack(
            np.broadcast_arrays(
                t_idx[:, :, None, None],
                h_idx[:, None, :, None],
                w_idx[:, None, None, :],
            ),
            axis=-1,
        ).reshape(len(coordinates), -1, 3)
        flat = source[..., 0] * height * width + source[..., 1] * width + source[..., 2]
        return flat.astype(np.int32), np.ones(flat.shape, dtype=bool), (0, 0, 0)
    indices = []
    valid = []
    for dt in range(-radii[0], radii[0] + 1):
        for dh in range(-radii[1], radii[1] + 1):
            for dw in range(-radii[2], radii[2] + 1):
                offset = np.array(
                    [dt * dilation[0], dh * dilation[1], dw * dilation[2]], dtype=np.int64
                )
                source = coordinates + offset
                is_valid = np.all(
                    (source >= 0) & (source < np.array([frames, height, width])), axis=1
                )
                padded = source + np.array(padding)
                flat = (
                    padded[:, 0] * padded_shape[1] * padded_shape[2]
                    + padded[:, 1] * padded_shape[2]
                    + padded[:, 2]
                )
                indices.append(flat)
                valid.append(is_valid)
    return (
        np.stack(indices, axis=1).astype(np.int32),
        np.stack(valid, axis=1),
        padding,
    )


def neighborhood_attention_3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    kernel: tuple[int, int, int] = (3, 3, 3),
    dilation: tuple[int, int, int] = (1, 1, 1),
    query_chunk_size: int = 1024,
    scale: float | None = None,
    boundary_mode: str = "mask",
    backend: str = "einsum",
) -> mx.array:
    """Attend each voxel to a bounded 3D neighborhood.

    Inputs use ``(batch, frames, height, width, heads, head_dim)``. Query chunks bound gathered
    K/V and score tensors, which is the Apple-Silicon analogue of the official chunked DiffVAE
    path. ``shift`` matches NATTEN by moving complete windows inward at borders;
    ``mask`` uses smaller masked border neighborhoods.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 6:
        raise ValueError("3D neighborhood attention requires matching six-dimensional Q/K/V.")
    if any(size < 1 or size % 2 == 0 for size in kernel):
        raise ValueError("Neighborhood kernel sizes must be positive odd integers.")
    if any(step < 1 for step in dilation):
        raise ValueError("Neighborhood dilation must be positive.")
    if query_chunk_size < 1:
        raise ValueError("Neighborhood query chunk size must be positive.")
    if boundary_mode not in {"mask", "shift"}:
        raise ValueError("Neighborhood boundary mode must be 'mask' or 'shift'.")
    if backend not in {"einsum", "sdpa", "metal"}:
        raise ValueError("Neighborhood attention backend must be 'einsum', 'sdpa', or 'metal'.")
    if backend == "sdpa" and boundary_mode != "shift":
        raise ValueError("The fused SDPA backend currently requires shifted neighborhoods.")

    batch, frames, height, width, heads, head_dim = q.shape
    attention_scale = head_dim**-0.5 if scale is None else float(scale)
    if backend == "metal":
        if boundary_mode != "shift" or dilation != (1, 1, 1):
            raise ValueError("Metal NA3D requires shifted, dilation-one neighborhoods.")
        return metal_neighborhood_attention_3d(
            q,
            k,
            v,
            kernel=kernel,
            scale=attention_scale,
        )
    host_indices, host_valid, padding = _neighborhood_index(
        frames, height, width, kernel, dilation, boundary_mode
    )
    pad_spec = (
        (0, 0),
        (padding[0], padding[0]),
        (padding[1], padding[1]),
        (padding[2], padding[2]),
        (0, 0),
        (0, 0),
    )
    padded_k = mx.pad(k, pad_spec).reshape(batch, -1, heads, head_dim)
    padded_v = mx.pad(v, pad_spec).reshape(batch, -1, heads, head_dim)
    flat_q = q.reshape(batch, -1, heads, head_dim)
    pieces = []
    for start in range(0, flat_q.shape[1], query_chunk_size):
        stop = min(start + query_chunk_size, flat_q.shape[1])
        index = mx.array(host_indices[start:stop])
        valid = mx.array(host_valid[start:stop])
        keys = mx.take(padded_k, index, axis=1)
        values = mx.take(padded_v, index, axis=1)
        queries = flat_q[:, start:stop]
        if backend == "sdpa":
            chunk_queries = queries.shape[1]
            neighborhood = keys.shape[2]
            fused_q = queries.transpose(0, 1, 2, 3).reshape(
                batch * chunk_queries, heads, 1, head_dim
            )
            fused_k = keys.transpose(0, 1, 3, 2, 4).reshape(
                batch * chunk_queries, heads, neighborhood, head_dim
            )
            fused_v = values.transpose(0, 1, 3, 2, 4).reshape(
                batch * chunk_queries, heads, neighborhood, head_dim
            )
            output = mx.fast.scaled_dot_product_attention(
                fused_q,
                fused_k,
                fused_v,
                scale=attention_scale,
            ).reshape(batch, chunk_queries, heads, head_dim)
        else:
            scores = mx.einsum("bqhd,bqkhd->bqhk", queries, keys) * attention_scale
            scores = mx.where(valid[None, :, None, :], scores, -mx.inf)
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype)
            output = mx.einsum("bqhk,bqkhd->bqhd", probabilities, values)
        mx.eval(output)
        pieces.append(output)
    return mx.concatenate(pieces, axis=1).reshape(q.shape)


__all__ = [
    "metal_neighborhood_attention_3d",
    "metal_neighborhood_attention_3d_slice",
    "metal_qk_rmsnorm_rope_3d",
    "metal_rmsnorm_rope_3d_slice",
    "neighborhood_attention_3d",
]
