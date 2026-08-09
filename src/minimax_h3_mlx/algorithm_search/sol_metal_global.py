"""Global-pool, block-routed Metal prototype for H3 sparse-attention research."""

from __future__ import annotations

import math

import mlx.core as mx

from .sol_metal import SolMetalConfig, _validate_qkv

_HEADER = """
#include <metal_math>
using namespace metal;
"""

_POOL_SOURCE = r"""
constexpr uint D = 128;
uint linear = thread_position_in_grid.x;
uint total = HEADS * KEY_BLOCKS * D;
if (linear >= total) {
  return;
}
uint dimension = linear % D;
uint key_block = (linear / D) % KEY_BLOCKS;
uint head = linear / (D * KEY_BLOCKS);
uint count = key_block + 1 == KEY_BLOCKS ? LAST_KEY_COUNT : BLOCK_SIZE;
uint block_start = PREFIX_ROWS + key_block * BLOCK_SIZE;
float key_sum = 0.0f;
float value_sum = 0.0f;
for (uint offset = 0; offset < count; ++offset) {
  uint source = (head * ROWS + block_start + offset) * D + dimension;
  key_sum += float(key[source]);
  value_sum += float(value[source]);
}
uint destination = (head * KEY_BLOCKS + key_block) * D + dimension;
pooled_key[destination] = key_sum / float(count);
summed_value[destination] = value_sum;
"""

_POOL_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_sol_attention_global_pool",
    input_names=["key", "value"],
    output_names=["pooled_key", "summed_value"],
    source=_POOL_SOURCE,
    header=_HEADER,
    ensure_row_contiguous=True,
)

_ATTENTION_SOURCE = r"""
constexpr uint D = 128;
constexpr uint LANES = 32;
constexpr uint VALUES_PER_LANE = D / LANES;
constexpr uint QUERY_BLOCK = 64;
constexpr uint SG_COUNT = SIMDGROUP_COUNT;
constexpr uint QUERIES_PER_SIMDGROUP = QUERY_BLOCK / SG_COUNT;
constexpr float SCALE = 0.08838834764831845f;

uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
uint thread_index = thread_position_in_threadgroup.x;
uint query_block = threadgroup_position_in_grid.x;
uint head = threadgroup_position_in_grid.y;
uint first_target_query = query_block * QUERY_BLOCK;

threadgroup float route_sums[SG_COUNT];
threadgroup uint route_exact_shared[1];

float q[QUERIES_PER_SIMDGROUP][VALUES_PER_LANE];
bool query_valid[QUERIES_PER_SIMDGROUP];
float thresholds[QUERIES_PER_SIMDGROUP] = {0.0f};
float maximum[QUERIES_PER_SIMDGROUP];
float denominator[QUERIES_PER_SIMDGROUP] = {0.0f};
float accumulator[QUERIES_PER_SIMDGROUP][VALUES_PER_LANE] = {{0.0f}};

#pragma unroll
for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
  uint target_query = first_target_query + simdgroup * QUERIES_PER_SIMDGROUP + local;
  query_valid[local] = target_query < TARGET_ROWS;
  maximum[local] = -3.402823466e+38f;
  if (query_valid[local]) {
    uint query_row = PREFIX_ROWS + target_query;
    const device bfloat* source = query + (head * ROWS + query_row) * D
        + lane * VALUES_PER_LANE;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      q[local][index] = float(source[index]);
    }
  } else {
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      q[local][index] = 0.0f;
    }
  }
}

// Prefix query rows are handled by dense MLX. Target queries consume every prefix key exactly.
for (uint key_row = 0; key_row < PREFIX_ROWS; ++key_row) {
  const device bfloat* key_values = key + (head * ROWS + key_row) * D
      + lane * VALUES_PER_LANE;
  const device bfloat* value_values = value + (head * ROWS + key_row) * D
      + lane * VALUES_PER_LANE;
#pragma unroll
  for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
    if (!query_valid[local]) {
      continue;
    }
    float partial = 0.0f;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      partial += q[local][index] * float(key_values[index]);
    }
    float score = simd_sum(partial) * SCALE;
    float new_maximum = max(maximum[local], score);
    float previous_scale = fast::exp(maximum[local] - new_maximum);
    float score_scale = fast::exp(score - new_maximum);
    denominator[local] = denominator[local] * previous_scale + score_scale;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      accumulator[local][index] = accumulator[local][index] * previous_scale
          + score_scale * float(value_values[index]);
    }
    maximum[local] = new_maximum;
  }
}

// Derive the diagonal threshold from the globally pooled target keys.
#pragma unroll
for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
  if (!query_valid[local]) {
    continue;
  }
  float mean_partial = 0.0f;
  float variance_partial = 0.0f;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    uint dimension = lane * VALUES_PER_LANE + index;
    float key_mean = 0.0f;
    float key_second = 0.0f;
    for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
      float pooled = pooled_key[(head * KEY_BLOCKS + key_block) * D + dimension];
      key_mean += pooled;
      key_second += pooled * pooled;
    }
    key_mean /= float(KEY_BLOCKS);
    key_second /= float(KEY_BLOCKS);
    mean_partial += q[local][index] * key_mean;
    variance_partial += q[local][index] * q[local][index]
        * max(key_second - key_mean * key_mean, 0.0f);
  }
  float mean = simd_sum(mean_partial) * SCALE;
  float variance = simd_sum(variance_partial) * SCALE * SCALE;
  thresholds[local] = mean + (float(BETA_MILLI) / 1000.0f)
      * sqrt(variance + 1.0e-12f);
}

uint exact_routes = 0;
for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
  const device float* pooled = pooled_key + (head * KEY_BLOCKS + key_block) * D
      + lane * VALUES_PER_LANE;
  float proxies[QUERIES_PER_SIMDGROUP] = {0.0f};
  float route_partial = 0.0f;
#pragma unroll
  for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
    if (!query_valid[local]) {
      continue;
    }
    float partial = 0.0f;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      partial += q[local][index] * pooled[index];
    }
    proxies[local] = simd_sum(partial) * SCALE;
    if (lane == 0) {
      route_partial += proxies[local] - thresholds[local];
    }
  }
  if (lane == 0) {
    route_sums[simdgroup] = route_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float route_total = 0.0f;
#pragma unroll
    for (uint group = 0; group < SG_COUNT; ++group) {
      route_total += route_sums[group];
    }
    route_exact_shared[0] = (FORCE_ALL_EXACT || route_total >= 0.0f) ? 1 : 0;
    if (route_exact_shared[0]) {
      exact_routes += 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint count = key_block + 1 == KEY_BLOCKS ? LAST_KEY_COUNT : BLOCK_SIZE;
  uint block_start = PREFIX_ROWS + key_block * BLOCK_SIZE;
  if (route_exact_shared[0]) {
    for (uint offset = 0; offset < count; ++offset) {
      uint key_row = block_start + offset;
      const device bfloat* key_values = key + (head * ROWS + key_row) * D
          + lane * VALUES_PER_LANE;
      const device bfloat* value_values = value + (head * ROWS + key_row) * D
          + lane * VALUES_PER_LANE;
#pragma unroll
      for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
        if (!query_valid[local]) {
          continue;
        }
        float partial = 0.0f;
#pragma unroll
        for (uint index = 0; index < VALUES_PER_LANE; ++index) {
          partial += q[local][index] * float(key_values[index]);
        }
        float score = simd_sum(partial) * SCALE;
        float new_maximum = max(maximum[local], score);
        float previous_scale = fast::exp(maximum[local] - new_maximum);
        float score_scale = fast::exp(score - new_maximum);
        denominator[local] = denominator[local] * previous_scale + score_scale;
#pragma unroll
        for (uint index = 0; index < VALUES_PER_LANE; ++index) {
          accumulator[local][index] = accumulator[local][index] * previous_scale
              + score_scale * float(value_values[index]);
        }
        maximum[local] = new_maximum;
      }
    }
  } else {
    const device float* summed = summed_value + (head * KEY_BLOCKS + key_block) * D
        + lane * VALUES_PER_LANE;
#pragma unroll
    for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
      if (!query_valid[local]) {
        continue;
      }
      float proxy = proxies[local];
      float new_maximum = max(maximum[local], proxy);
      float previous_scale = fast::exp(maximum[local] - new_maximum);
      float score_scale = fast::exp(proxy - new_maximum);
      denominator[local] = denominator[local] * previous_scale
          + float(count) * score_scale;
#pragma unroll
      for (uint index = 0; index < VALUES_PER_LANE; ++index) {
        accumulator[local][index] = accumulator[local][index] * previous_scale
            + score_scale * summed[index];
      }
      maximum[local] = new_maximum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

#pragma unroll
for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
  if (!query_valid[local]) {
    continue;
  }
  uint target_query = first_target_query + simdgroup * QUERIES_PER_SIMDGROUP + local;
  device bfloat* result = output + (head * TARGET_ROWS + target_query) * D
      + lane * VALUES_PER_LANE;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    result[index] = bfloat(
        accumulator[local][index] / max(denominator[local], 1.0e-20f));
  }
}
if (thread_index == 0) {
  route_count[head * QUERY_BLOCKS + query_block] = int(exact_routes);
}
"""

_ATTENTION_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_sol_attention_global_pool_block_route_prototype",
    input_names=["query", "key", "value", "pooled_key", "summed_value"],
    output_names=["output", "route_count"],
    source=_ATTENTION_SOURCE,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def sol_metal_global_pool_block_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    config: SolMetalConfig,
    simdgroups: int = 16,
    return_route_counts: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Pool target K/V once, then run shared 64-row routes without a route map."""
    _validate_qkv(query, key, value)
    rows = int(query.shape[-2])
    config.validate(rows)
    if simdgroups not in {8, 16}:
        raise ValueError("global-pool Sol Metal requires 8 or 16 SIMDgroups")
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Sol Metal requires the H3 128-dimension attention scale")
    prefix = config.prefix_rows
    target_rows = rows - prefix
    key_blocks = math.ceil(target_rows / config.block_size)
    query_blocks = math.ceil(target_rows / 64)
    last_key_count = target_rows - (key_blocks - 1) * config.block_size
    heads = int(query.shape[1])
    pooled_key, summed_value = _POOL_KERNEL(
        inputs=[key, value],
        template=[
            ("ROWS", rows),
            ("HEADS", heads),
            ("PREFIX_ROWS", prefix),
            ("KEY_BLOCKS", key_blocks),
            ("LAST_KEY_COUNT", last_key_count),
            ("BLOCK_SIZE", config.block_size),
        ],
        grid=(math.ceil(heads * key_blocks * 128 / 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (1, heads, key_blocks, 128),
            (1, heads, key_blocks, 128),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    target, route_counts = _ATTENTION_KERNEL(
        inputs=[query, key, value, pooled_key, summed_value],
        template=[
            ("ROWS", rows),
            ("PREFIX_ROWS", prefix),
            ("TARGET_ROWS", target_rows),
            ("KEY_BLOCKS", key_blocks),
            ("QUERY_BLOCKS", query_blocks),
            ("LAST_KEY_COUNT", last_key_count),
            ("BLOCK_SIZE", config.block_size),
            ("BETA_MILLI", int(round(config.beta * 1000))),
            ("FORCE_ALL_EXACT", config.force_all_exact),
            ("SIMDGROUP_COUNT", simdgroups),
        ],
        grid=(query_blocks * 32 * simdgroups, heads, 1),
        threadgroup=(32 * simdgroups, 1, 1),
        output_shapes=[
            (1, heads, target_rows, 128),
            (1, heads, query_blocks),
        ],
        output_dtypes=[mx.bfloat16, mx.int32],
    )
    prefix_output = mx.fast.scaled_dot_product_attention(
        query[..., :prefix, :], key, value, scale=scale
    )
    output = mx.concatenate([prefix_output, target], axis=-2)
    if return_route_counts:
        return output, route_counts
    return output
