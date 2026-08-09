"""Research-only fused Metal prototype for corrected H3 target-video attention."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx

_HEADER = """
#include <metal_math>
using namespace metal;
"""

_SOURCE = r"""
constexpr uint D = 128;
constexpr uint LANES = 32;
constexpr uint VALUES_PER_LANE = D / LANES;
constexpr uint SIMDGROUPS = 4;
constexpr float SCALE = 0.08838834764831845f;

uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
uint target_query = threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup;
uint head = threadgroup_position_in_grid.y;
if (target_query >= TARGET_ROWS) {
  return;
}

uint query_row = PREFIX_ROWS + target_query;
const device bfloat* query_values = query + (head * ROWS + query_row) * D
    + lane * VALUES_PER_LANE;
float q[VALUES_PER_LANE];
#pragma unroll
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  q[index] = float(query_values[index]);
}

float proxy_sum = 0.0f;
float proxy_square_sum = 0.0f;
for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
  const device float* pooled = pooled_key + (head * KEY_BLOCKS + key_block) * D
      + lane * VALUES_PER_LANE;
  float partial = 0.0f;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    partial += q[index] * pooled[index];
  }
  float proxy = simd_sum(partial) * SCALE;
  proxy_sum += proxy;
  proxy_square_sum += proxy * proxy;
}
float proxy_mean = proxy_sum / float(KEY_BLOCKS);
float proxy_variance = max(
    proxy_square_sum / float(KEY_BLOCKS) - proxy_mean * proxy_mean,
    0.0f);
float threshold = proxy_mean + (float(BETA_MILLI) / 1000.0f)
    * sqrt(proxy_variance + 1.0e-12f);

float maximum = -3.402823466e+38f;
float denominator = 0.0f;
float accumulator[VALUES_PER_LANE] = {0.0f};

for (uint key_row = 0; key_row < PREFIX_ROWS; ++key_row) {
  const device bfloat* key_values = key + (head * ROWS + key_row) * D
      + lane * VALUES_PER_LANE;
  const device bfloat* value_values = value + (head * ROWS + key_row) * D
      + lane * VALUES_PER_LANE;
  float partial = 0.0f;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    partial += q[index] * float(key_values[index]);
  }
  float score = simd_sum(partial) * SCALE;
  float new_maximum = max(maximum, score);
  float previous_scale = fast::exp(maximum - new_maximum);
  float score_scale = fast::exp(score - new_maximum);
  denominator = denominator * previous_scale + score_scale;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    accumulator[index] = accumulator[index] * previous_scale
        + score_scale * float(value_values[index]);
  }
  maximum = new_maximum;
}

for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
  const device float* pooled = pooled_key + (head * KEY_BLOCKS + key_block) * D
      + lane * VALUES_PER_LANE;
  float partial = 0.0f;
#pragma unroll
  for (uint index = 0; index < VALUES_PER_LANE; ++index) {
    partial += q[index] * pooled[index];
  }
  float proxy = simd_sum(partial) * SCALE;
  uint count = uint(key_count[key_block]);
  if (FORCE_ALL_EXACT || proxy >= threshold) {
    uint block_start = PREFIX_ROWS + key_block * BLOCK_SIZE;
    for (uint offset = 0; offset < count; ++offset) {
      uint key_row = block_start + offset;
      const device bfloat* key_values = key + (head * ROWS + key_row) * D
          + lane * VALUES_PER_LANE;
      const device bfloat* value_values = value + (head * ROWS + key_row) * D
          + lane * VALUES_PER_LANE;
      float exact_partial = 0.0f;
#pragma unroll
      for (uint index = 0; index < VALUES_PER_LANE; ++index) {
        exact_partial += q[index] * float(key_values[index]);
      }
      float score = simd_sum(exact_partial) * SCALE;
      float new_maximum = max(maximum, score);
      float previous_scale = fast::exp(maximum - new_maximum);
      float score_scale = fast::exp(score - new_maximum);
      denominator = denominator * previous_scale + score_scale;
#pragma unroll
      for (uint index = 0; index < VALUES_PER_LANE; ++index) {
        accumulator[index] = accumulator[index] * previous_scale
            + score_scale * float(value_values[index]);
      }
      maximum = new_maximum;
    }
  } else {
    const device float* summed = summed_value + (head * KEY_BLOCKS + key_block) * D
        + lane * VALUES_PER_LANE;
    float new_maximum = max(maximum, proxy);
    float previous_scale = fast::exp(maximum - new_maximum);
    float score_scale = fast::exp(proxy - new_maximum);
    denominator = denominator * previous_scale + float(count) * score_scale;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      accumulator[index] = accumulator[index] * previous_scale
          + score_scale * summed[index];
    }
    maximum = new_maximum;
  }
}

device bfloat* result = output + (head * TARGET_ROWS + target_query) * D
    + lane * VALUES_PER_LANE;
#pragma unroll
for (uint index = 0; index < VALUES_PER_LANE; ++index) {
  result[index] = bfloat(accumulator[index] / max(denominator, 1.0e-20f));
}
"""

_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_sol_attention_row_route_prototype",
    input_names=["query", "key", "value", "pooled_key", "summed_value", "key_count"],
    output_names=["output"],
    source=_SOURCE,
    header=_HEADER,
    ensure_row_contiguous=True,
)


_BLOCK_SOURCE = r"""
constexpr uint D = 128;
constexpr uint LANES = 32;
constexpr uint VALUES_PER_LANE = D / LANES;
constexpr uint QUERY_BLOCK = 64;
constexpr uint SIMDGROUPS = 16;
constexpr uint QUERIES_PER_SIMDGROUP = QUERY_BLOCK / SIMDGROUPS;
constexpr float SCALE = 0.08838834764831845f;

uint lane = thread_index_in_simdgroup;
uint simdgroup = simdgroup_index_in_threadgroup;
uint thread_index = thread_position_in_threadgroup.x;
uint query_block = threadgroup_position_in_grid.x;
uint head = threadgroup_position_in_grid.y;
uint first_target_query = query_block * QUERY_BLOCK;

threadgroup float pooled_key_values[D];
threadgroup float summed_value_values[D];
threadgroup float key_mean_values[D];
threadgroup float key_second_values[D];
threadgroup float route_sums[SIMDGROUPS];
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

// Every target-video query attends the complete multimodal prefix exactly.
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

// First pass: pool K once per block and derive diagonal pooled-key moments.
if (thread_index < D) {
  key_mean_values[thread_index] = 0.0f;
  key_second_values[thread_index] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
  uint count = key_block + 1 == KEY_BLOCKS ? LAST_KEY_COUNT : BLOCK_SIZE;
  uint block_start = PREFIX_ROWS + key_block * BLOCK_SIZE;
  if (thread_index < D) {
    float key_sum = 0.0f;
    for (uint offset = 0; offset < count; ++offset) {
      key_sum += float(key[(head * ROWS + block_start + offset) * D + thread_index]);
    }
    pooled_key_values[thread_index] = key_sum / float(count);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index < D) {
    float pooled = pooled_key_values[thread_index];
    key_mean_values[thread_index] += pooled;
    key_second_values[thread_index] += pooled * pooled;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (thread_index < D) {
  key_mean_values[thread_index] /= float(KEY_BLOCKS);
  key_second_values[thread_index] /= float(KEY_BLOCKS);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
for (uint local = 0; local < QUERIES_PER_SIMDGROUP; ++local) {
  if (query_valid[local]) {
    float mean_partial = 0.0f;
    float variance_partial = 0.0f;
#pragma unroll
    for (uint index = 0; index < VALUES_PER_LANE; ++index) {
      uint dimension = lane * VALUES_PER_LANE + index;
      float key_mean = key_mean_values[dimension];
      float key_variance = max(
          key_second_values[dimension] - key_mean * key_mean,
          0.0f);
      mean_partial += q[local][index] * key_mean;
      variance_partial += q[local][index] * q[local][index] * key_variance;
    }
    float mean = simd_sum(mean_partial) * SCALE;
    float variance = simd_sum(variance_partial) * SCALE * SCALE;
    thresholds[local] = mean + (float(BETA_MILLI) / 1000.0f)
        * sqrt(variance + 1.0e-12f);
  }
}

uint exact_routes = 0;
// Second pass: pool K/V once, reduce one route over 64 queries, then update online softmax.
for (uint key_block = 0; key_block < KEY_BLOCKS; ++key_block) {
  uint count = key_block + 1 == KEY_BLOCKS ? LAST_KEY_COUNT : BLOCK_SIZE;
  uint block_start = PREFIX_ROWS + key_block * BLOCK_SIZE;
  if (thread_index < D) {
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint offset = 0; offset < count; ++offset) {
      key_sum += float(key[(head * ROWS + block_start + offset) * D + thread_index]);
      value_sum += float(value[(head * ROWS + block_start + offset) * D + thread_index]);
    }
    pooled_key_values[thread_index] = key_sum / float(count);
    summed_value_values[thread_index] = value_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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
      uint dimension = lane * VALUES_PER_LANE + index;
      partial += q[local][index] * pooled_key_values[dimension];
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
    for (uint group = 0; group < SIMDGROUPS; ++group) {
      route_total += route_sums[group];
    }
    route_exact_shared[0] = (FORCE_ALL_EXACT || route_total >= 0.0f) ? 1 : 0;
    if (route_exact_shared[0]) {
      exact_routes += 1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

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
        uint dimension = lane * VALUES_PER_LANE + index;
        accumulator[local][index] = accumulator[local][index] * previous_scale
            + score_scale * summed_value_values[dimension];
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

_BLOCK_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_sol_attention_block_route_fused_pool_prototype",
    input_names=["query", "key", "value"],
    output_names=["output", "route_count"],
    source=_BLOCK_SOURCE,
    header=_HEADER,
    ensure_row_contiguous=True,
)


@dataclass(frozen=True)
class SolMetalConfig:
    """Static policy for the research Metal prototype."""

    prefix_rows: int
    beta: float = 0.75
    block_size: int = 64
    force_all_exact: bool = False

    def validate(self, rows: int) -> None:
        if not 0 < self.prefix_rows < rows:
            raise ValueError("Sol Metal prefix_rows must be inside the packed sequence")
        if self.block_size != 64:
            raise ValueError("Sol Metal prototype currently requires 64-row blocks")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Sol Metal beta must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_qkv(query: mx.array, key: mx.array, value: mx.array) -> None:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol Metal requires matching query, key, and value shapes")
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("Sol Metal requires H3 shape [1, heads, rows, 128]")
    if any(array.dtype != mx.bfloat16 for array in (query, key, value)):
        raise TypeError("Sol Metal requires BF16 query, key, and value arrays")


def _pool_target(
    key: mx.array,
    value: mx.array,
    *,
    prefix_rows: int,
    block_size: int,
) -> tuple[mx.array, mx.array, mx.array]:
    target_rows = int(key.shape[-2]) - prefix_rows
    blocks = math.ceil(target_rows / block_size)
    padded_rows = blocks * block_size
    padding = padded_rows - target_rows
    target_key = key[..., prefix_rows:, :].astype(mx.float32)
    target_value = value[..., prefix_rows:, :].astype(mx.float32)
    if padding:
        shape = (*target_key.shape[:-2], padding, target_key.shape[-1])
        target_key = mx.concatenate([target_key, mx.zeros(shape, dtype=mx.float32)], axis=-2)
        target_value = mx.concatenate([target_value, mx.zeros(shape, dtype=mx.float32)], axis=-2)
    blocked_key = target_key.reshape(*target_key.shape[:-2], blocks, block_size, 128)
    blocked_value = target_value.reshape(*target_value.shape[:-2], blocks, block_size, 128)
    counts = mx.full((blocks,), block_size, dtype=mx.int32)
    if padding:
        counts[-1] = block_size - padding
    pooled_key = mx.sum(blocked_key, axis=-2) / counts.astype(mx.float32)[None, None, :, None]
    summed_value = mx.sum(blocked_value, axis=-2)
    return pooled_key, summed_value, counts


def sol_metal_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    config: SolMetalConfig,
) -> mx.array:
    """Run the row-routed Metal prototype and return the complete packed attention output.

    This prototype uses per-query-row routing rather than Sol-Attn's query-block route reduction.
    The complete multimodal prefix and every prefix query remain dense. The deviation is explicit so
    its timing and numerical result cannot be mistaken for a production Sol-Attn implementation.
    """
    _validate_qkv(query, key, value)
    rows = int(query.shape[-2])
    config.validate(rows)
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Sol Metal requires the H3 128-dimension attention scale")
    prefix = config.prefix_rows
    target_rows = rows - prefix
    pooled_key, summed_value, counts = _pool_target(
        key, value, prefix_rows=prefix, block_size=config.block_size
    )
    heads = int(query.shape[1])
    threadgroup = (128, 1, 1)
    target = _KERNEL(
        inputs=[query, key, value, pooled_key, summed_value, counts],
        template=[
            ("ROWS", rows),
            ("PREFIX_ROWS", prefix),
            ("TARGET_ROWS", target_rows),
            ("KEY_BLOCKS", int(pooled_key.shape[-2])),
            ("BLOCK_SIZE", config.block_size),
            ("BETA_MILLI", int(round(config.beta * 1000))),
            ("FORCE_ALL_EXACT", config.force_all_exact),
        ],
        grid=(math.ceil(target_rows / 4) * 128, heads, 1),
        threadgroup=threadgroup,
        output_shapes=[(1, heads, target_rows, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]
    prefix_output = mx.fast.scaled_dot_product_attention(
        query[..., :prefix, :], key, value, scale=scale
    )
    return mx.concatenate([prefix_output, target], axis=-2)


def sol_metal_block_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
    config: SolMetalConfig,
    return_route_counts: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Run fused-pooling, 64-row query-block routing over the H3 target-video tail.

    The kernel retains the complete multimodal prefix as an exact key-value sink. Prefix query rows
    use dense MLX attention. Each target query block derives one route decision without writing a
    route tensor to device memory. The optional count output records exact target-key blocks per
    head and query block.
    """
    _validate_qkv(query, key, value)
    rows = int(query.shape[-2])
    config.validate(rows)
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Sol Metal requires the H3 128-dimension attention scale")
    prefix = config.prefix_rows
    target_rows = rows - prefix
    key_blocks = math.ceil(target_rows / config.block_size)
    query_blocks = math.ceil(target_rows / 64)
    last_key_count = target_rows - (key_blocks - 1) * config.block_size
    heads = int(query.shape[1])
    target, route_counts = _BLOCK_KERNEL(
        inputs=[query, key, value],
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
        ],
        grid=(query_blocks * 512, heads, 1),
        threadgroup=(512, 1, 1),
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
