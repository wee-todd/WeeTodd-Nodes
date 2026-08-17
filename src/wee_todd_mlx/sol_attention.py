"""Experimental fused Sol-style sparse attention for long MLX video sequences.

This module is an independent implementation of the published Sol-Attn algorithm.  It uses the
Steel primitives distributed with MLX, keeps routing inside the tiled attention kernel, and never
materializes a quadratic route map.  The backend is deliberately opt-in and has a dense fallback
for every unsupported shape.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

_BLOCK = 64
_LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"\s*$')


@dataclass(frozen=True)
class SolAttentionConfig:
    """Runtime policy for the experimental sparse-attention backend."""

    enabled: bool = False
    tau: float = 1.0
    min_tokens: int = 16384
    exact_prefix_rows: int = 0
    exact_suffix_rows: int = 0
    exact_suffix_groups: tuple[int, ...] = ()
    start_percent: float = 0.2
    end_percent: float = 1.0
    dense_blocks: int = 2

    def validate(self) -> None:
        if self.tau < 0.0:
            raise ValueError("Sol attention tau must be non-negative")
        if self.min_tokens < _BLOCK:
            raise ValueError(f"Sol attention min_tokens must be at least {_BLOCK}")
        if self.exact_prefix_rows < 0:
            raise ValueError("Sol attention exact_prefix_rows must be non-negative")
        if self.exact_suffix_rows < 0:
            raise ValueError("Sol attention exact_suffix_rows must be non-negative")
        if any(rows <= 0 for rows in self.exact_suffix_groups):
            raise ValueError("Sol attention exact suffix groups must contain positive row counts")
        if self.exact_suffix_groups and sum(self.exact_suffix_groups) != self.exact_suffix_rows:
            raise ValueError("Sol attention exact suffix groups must sum to exact_suffix_rows")
        if not 0.0 <= self.start_percent <= self.end_percent <= 1.0:
            raise ValueError("Sol attention requires 0 <= start_percent <= end_percent <= 1")
        if self.dense_blocks < 0:
            raise ValueError("Sol attention dense_blocks must be non-negative")

    def active(self, *, step_index: int, total_steps: int, block_index: int) -> bool:
        """Return whether this transformer call is inside the configured sparse region."""

        if not self.enabled or block_index < self.dense_blocks:
            return False
        denominator = max(1, total_steps - 1)
        progress = step_index / denominator
        return self.start_percent <= progress <= self.end_percent

    def with_prefix(self, rows: int) -> SolAttentionConfig:
        return SolAttentionConfig(
            enabled=self.enabled,
            tau=self.tau,
            min_tokens=self.min_tokens,
            exact_prefix_rows=rows,
            exact_suffix_rows=self.exact_suffix_rows,
            exact_suffix_groups=self.exact_suffix_groups,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            dense_blocks=self.dense_blocks,
        )

    def with_exact_rows(
        self,
        *,
        prefix: int = 0,
        suffix: int = 0,
        suffix_groups: tuple[int, ...] = (),
    ) -> SolAttentionConfig:
        """Return a policy with exact conditioning rows at either sequence boundary."""

        return SolAttentionConfig(
            enabled=self.enabled,
            tau=self.tau,
            min_tokens=self.min_tokens,
            exact_prefix_rows=prefix,
            exact_suffix_rows=suffix,
            exact_suffix_groups=tuple(suffix_groups),
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            dense_blocks=self.dense_blocks,
        )


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
    return header[prefix_start:template_start], header[body_start + 1 : body_end]


def _steel_sources() -> tuple[str, str]:
    include_root = _mlx_include_root()
    kernel_path = include_root / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
    raw = kernel_path.read_text()
    prefix, body = _extract_attention_parts(raw)
    dependency = include_root / "mlx/backend/metal/kernels/steel/attn/attn.h"
    return f"{_expand_header(dependency, include_root, set())}\n{prefix}", body


_STEEL_HEADER, _DENSE_BODY = _steel_sources()


_SUMMARY_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_sol_block_summaries_bf16_d128",
    input_names=["q", "k", "v"],
    output_names=["qc", "kc", "vc"],
    source=r"""
        uint index = thread_position_in_grid.x;
        if (index >= BATCH * HEADS * BLOCKS * HEAD_DIM) return;
        uint dim = index % HEAD_DIM;
        uint block = (index / HEAD_DIM) % BLOCKS;
        uint head = (index / (HEAD_DIM * BLOCKS)) % HEADS;
        uint batch = index / (HEAD_DIM * BLOCKS * HEADS);
        uint start = block * BLOCK_SIZE;
        uint length = min(uint(BLOCK_SIZE), uint(ROWS - start));
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float v_sum = 0.0f;
        for (uint row = 0; row < length; ++row) {
            ulong q_offset = ulong(batch) * q_strides[0] + ulong(head) * q_strides[1]
                + ulong(start + row) * q_strides[2] + dim;
            ulong k_offset = ulong(batch) * k_strides[0] + ulong(head) * k_strides[1]
                + ulong(start + row) * k_strides[2] + dim;
            ulong v_offset = ulong(batch) * v_strides[0] + ulong(head) * v_strides[1]
                + ulong(start + row) * v_strides[2] + dim;
            q_sum += float(q[q_offset]);
            k_sum += float(k[k_offset]);
            v_sum += float(v[v_offset]);
        }
        qc[index] = q_sum / float(length);
        kc[index] = bfloat(k_sum / float(length));
        vc[index] = bfloat(v_sum);
    """,
    ensure_row_contiguous=False,
)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Installed MLX Steel attention layout changed at {label}")
    return source.replace(old, new, 1)


def _sparse_body() -> str:
    """Patch the installed dense Steel loop with fused routing and summary correction.

    The MLX package remains the source of the exact dense tile implementation.  The inserted path
    changes only how a skipped key block contributes: eight identical centroid columns represent
    the block's softmax mass, and one 8-row value tile represents its summed value contribution.
    """

    body = _DENSE_BODY
    loop_marker = "  // Loop over KV seq length\n  for (int kb = 0; kb < kb_lim; kb++) {"
    loop_replacement = r"""  // One route decision is shared by every SIMD group in this query tile.
  threadgroup uint route_exact_shared;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    if (simd_group_id == 0) {
      float route_partial = 0.0f;
      const ulong summary_base =
          (tidl.z * params->H + tidl.y) * SUMMARY_BLOCKS * BD;
      const uint route_query_block = (tid.x * BQ) / SUMMARY_SIZE;
      const uint route_key_block = (kb * BK) / SUMMARY_SIZE;
      for (uint dim = simd_lane_id; dim < BD; dim += 32) {
        route_partial += QC[summary_base + route_query_block * BD + dim] *
            float(KC[summary_base + route_key_block * BD + dim]);
      }
      route_partial = simd_sum(route_partial);
      if (simd_lane_id == 0) {
        const bool query_sink = int(route_query_block) < PREFIX_BLOCKS ||
            int(route_query_block) >= SUMMARY_BLOCKS - SUFFIX_BLOCKS;
        const bool key_sink = int(route_key_block) < PREFIX_BLOCKS ||
            int(route_key_block) >= SUMMARY_BLOCKS - SUFFIX_BLOCKS;
        const bool neighbor = abs(int(route_query_block) - int(route_key_block)) <= 1;
        const ulong threshold_offset =
            (tidl.z * params->H + tidl.y) * SUMMARY_BLOCKS + route_query_block;
        route_exact_shared = uint(query_sink || key_sink || neighbor ||
            route_partial * params->scale * M_LOG2E_F > THRESHOLDS[threshold_offset]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0 && simd_lane_id == 0) {
      if (route_exact_shared != 0) {
        route_exact_count += 1;
      } else {
        route_approx_count += 1;
      }
    }
    const bool route_exact = route_exact_shared != 0;"""
    body = _replace_once(body, loop_marker, loop_replacement, "KV loop")

    load_k = """    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Apply scale in float32
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }
"""
    routed_qk = r"""    // Load exact K or synthesize the compact centroid tile for a skipped block.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (route_exact) {
      if (!align_K && kb == (params->NK_aligned)) {
        loader_k.load_safe(short2(BD, params->kL_rem));
      } else {
        loader_k.load_unsafe();
      }
    } else {
      const ulong summary_base =
          (tidl.z * params->H + tidl.y) * SUMMARY_BLOCKS * BD +
          ((kb * BK) / SUMMARY_SIZE) * BD;
      for (uint linear = lid.y * 32 + lid.x; linear < 8 * BD;
           linear += WM * WN * 32) {
        const uint token = linear / BD;
        const uint dim = linear % BD;
        Ks[dim * LDK_tgp + token] = KC[summary_base + dim];
      }
    }

    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (route_exact) {
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
        Ktile.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] *= scale;
      }
    } else {
      using ApproxKTile = MMATile<AccumType, 1, 1, MMAFrag_acc_t>;
      using ApproxSTile = MMATile<AccumType, TQ, 1, MMAFrag_acc_t>;
      ApproxKTile approx_k;
      ApproxSTile approx_s;
      approx_s.clear();
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
        approx_k.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(approx_s, Qtile, approx_k, approx_s);
      }
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
        Stile.elems()[ii] = neg_inf;
      }
      const uint summary_start = ((kb * BK) / SUMMARY_SIZE) * SUMMARY_SIZE;
      const uint block_length = min(uint(SUMMARY_SIZE), uint(params->kL - summary_start));
      const float mass_adjust = fast::log2(float(block_length) / 8.0f);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < TQ; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short element = 0; element < MMAFrag_acc_t::kElemsPerFrag; ++element) {
          Stile.frag_at(i, 0)[element] =
              approx_s.frag_at(i, 0)[element] * scale + mass_adjust;
        }
      }
    }
"""
    body = _replace_once(body, load_k, routed_qk, "QK path")

    load_v = """    // Load V blocks
    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }
"""
    routed_v = r"""    // Load exact V or eight repeated rows representing the block value sum.
    if (route_exact) {
      if (!align_K && kb == (params->NK_aligned)) {
        loader_v.load_safe(short2(BD, params->kL_rem));
      } else {
        loader_v.load_unsafe();
      }
    } else {
      const ulong summary_base =
          (tidl.z * params->H + tidl.y) * SUMMARY_BLOCKS * BD +
          ((kb * BK) / SUMMARY_SIZE) * BD;
      const uint summary_start = ((kb * BK) / SUMMARY_SIZE) * SUMMARY_SIZE;
      const float block_length =
          float(min(uint(SUMMARY_SIZE), uint(params->kL - summary_start)));
      for (uint linear = lid.y * 32 + lid.x; linear < 8 * BD;
           linear += WM * WN * 32) {
        const uint token = linear / BD;
        const uint dim = linear % BD;
        Vs[token * LDV_tgp + dim] = T(float(VC[summary_base + dim]) / block_length);
      }
    }
"""
    body = _replace_once(body, load_v, routed_v, "V path")

    pv_loop = """        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
"""
    routed_pv = r"""        const short ik_limit = route_exact ? TK : 1;
        for (short ik = 0; ik < ik_limit; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
"""
    return _replace_once(body, pv_loop, routed_pv, "PV path")


_SPARSE_BODY = _sparse_body()
_MASK_NONE = 0
_MASK_FULL = 1
_MASK_COMPACT_SUFFIX = 2
_KERNELS: dict[tuple[int, int, int, int, int, tuple[int, ...]], object] = {}


def _kernel(
    prefix_blocks: int,
    suffix_blocks: int,
    summary_blocks: int,
    *,
    mask_mode: int,
    exact_suffix_rows: int,
    suffix_groups: tuple[int, ...],
):
    key = (
        prefix_blocks,
        suffix_blocks,
        summary_blocks,
        mask_mode,
        exact_suffix_rows,
        suffix_groups,
    )
    kernel = _KERNELS.get(key)
    if kernel is not None:
        return kernel
    masked = mask_mode != _MASK_NONE
    if mask_mode == _MASK_FULL:
        mask_params_declaration = (
            "AttnMaskParams mask_params_value{{"
            "attention_mask_strides[0], 0, attention_mask_strides[1]}};"
        )
        mask_pointer = "attention_mask"
    elif mask_mode == _MASK_COMPACT_SUFFIX:
        mask_params_declaration = (
            "AttnMaskParams mask_params_value{{"
            "attention_mask_strides[0], 0, 0}};"
        )
        boundaries = []
        running = 0
        for index, rows in enumerate(suffix_groups, start=1):
            running += rows
            boundaries.append((index, running))
        query = "threadgroup_position_in_grid.x * BQ"
        template_index = "0"
        for index, boundary in reversed(boundaries):
            template_index = (
                f"({query} < TARGET_ROWS + {boundary} ? {index} : {template_index})"
            )
        template_index = f"({query} < TARGET_ROWS ? 0 : {template_index})"
        mask_pointer = (
            f"attention_mask + ({template_index}) * attention_mask_strides[1]"
        )
    else:
        mask_params_declaration = ""
        mask_pointer = "nullptr"
    source = f"""
using namespace mlx::steel;
using T = bfloat;
using MaskType = bfloat;
using AccumType = float;
constexpr int BQ = 32;
constexpr int BK = 64;
constexpr int BD = 128;
constexpr int WM = 4;
constexpr int WN = 1;
constexpr int SUMMARY_SIZE = 64;
constexpr int PREFIX_BLOCKS = {prefix_blocks};
constexpr int SUFFIX_BLOCKS = {suffix_blocks};
constexpr int SUMMARY_BLOCKS = {summary_blocks};
constexpr int TARGET_ROWS = ROWS - {exact_suffix_rows};
constexpr bool align_Q = (ROWS % BQ) == 0;
constexpr bool align_K = (ROWS % BK) == 0;
constexpr bool has_mask = {"true" if masked else "false"};
constexpr bool do_causal = false;
constexpr bool has_sinks = false;

const device T* Q = q;
const device T* K = k;
const device T* V = v;
auto QC = qc;
auto KC = kc;
auto VC = vc;
auto THRESHOLDS = thresholds;
device T* O = output;
device uint* ROUTE_COUNTS = route_counts;
uint route_exact_count = 0;
uint route_approx_count = 0;
AttnParams params_value{{
    BATCH, HEADS, BD,
    ROWS, ROWS,
    1, 0.08838834764831845f,
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
{mask_params_declaration}
thread const AttnMaskParams* mask_params = {"&mask_params_value" if masked else "nullptr"};
const device MaskType* mask = {mask_pointer};
const device T* sinks = nullptr;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;
uint3 tid = threadgroup_position_in_grid;
uint3 lid = thread_position_in_threadgroup;
{_SPARSE_BODY}
if (simd_group_id == 0 && simd_lane_id == 0) {{
    const uint query_tiles = (ROWS + BQ - 1) / BQ;
    const ulong route_offset =
        ((ulong(tid.z) * HEADS + tid.y) * query_tiles + tid.x) * 2;
    ROUTE_COUNTS[route_offset] = route_exact_count;
    ROUTE_COUNTS[route_offset + 1] = route_approx_count;
}}
"""
    kernel = mx.fast.metal_kernel(
        name=(
            f"wee_todd_sol_attention_p{prefix_blocks}_s{suffix_blocks}_n{summary_blocks}"
            f"_m{mask_mode}_g{'_'.join(str(rows) for rows in suffix_groups) or '0'}"
        ),
        input_names=[
            "q",
            "k",
            "v",
            "qc",
            "kc",
            "vc",
            "thresholds",
            *(["attention_mask"] if masked else []),
        ],
        output_names=["output", "route_counts"],
        source=source,
        header=_STEEL_HEADER,
        ensure_row_contiguous=False,
    )
    _KERNELS[key] = kernel
    return kernel


def _summaries(q: mx.array, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    batch, heads, rows, head_dim = q.shape
    blocks = math.ceil(rows / _BLOCK)
    return tuple(
        _SUMMARY_KERNEL(
            inputs=[q, k, v],
            template=[
                ("BATCH", batch),
                ("HEADS", heads),
                ("ROWS", rows),
                ("HEAD_DIM", head_dim),
                ("BLOCKS", blocks),
                ("BLOCK_SIZE", _BLOCK),
            ],
            grid=(batch * heads * blocks * head_dim, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[
                (batch, heads, blocks, head_dim),
                (batch, heads, blocks, head_dim),
                (batch, heads, blocks, head_dim),
            ],
            output_dtypes=[mx.float32, mx.bfloat16, mx.bfloat16],
        )
    )


def _thresholds(qc: mx.array, kc: mx.array, *, scale: float, tau: float) -> mx.array:
    keys = kc.astype(mx.float32)
    key_mean = mx.mean(keys, axis=2)
    key_variance = mx.maximum(mx.mean(keys * keys, axis=2) - key_mean * key_mean, 0.0)
    log2_scale = scale * math.log2(math.e)
    mean = mx.sum(qc * key_mean[:, :, None, :], axis=-1) * log2_scale
    variance = mx.sum(qc * qc * key_variance[:, :, None, :], axis=-1) * (log2_scale**2)
    return mean + tau * mx.sqrt(variance + 1.0e-6)


def sol_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    config: SolAttentionConfig,
    mask: mx.array | None = None,
    return_route_counts: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Return sparse H3 attention in logical ``[B, H, L, D]`` layout."""

    config.validate()
    if not config.enabled:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Sol attention requires matching Q, K, and V shapes")
    if q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("Sol attention currently requires [batch, heads, rows, 128]")
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Sol attention currently requires the H3 128-dimension scale")
    if q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16:
        raise TypeError("Sol attention currently requires BF16 Q, K, and V")
    batch, heads, rows, head_dim = q.shape
    mask_mode = _MASK_NONE
    suffix_groups = tuple(config.exact_suffix_groups)
    if mask is not None:
        if mask.ndim == 3 and tuple(mask.shape) == (batch, rows, rows):
            mask_mode = _MASK_FULL
        elif mask.ndim == 3 and mask.shape[0] == batch and mask.shape[2] == rows:
            mask_mode = _MASK_COMPACT_SUFFIX
        else:
            raise ValueError(
                "Masked Sol attention requires a full [batch, rows, rows] mask or a compact "
                "suffix mask shaped [batch, 2, rows]."
            )
        if config.exact_suffix_rows <= 0 and config.exact_prefix_rows <= 0:
            raise ValueError(
                "Masked Sol attention requires exact conditioning rows at a sequence boundary."
            )
        if mask_mode == _MASK_COMPACT_SUFFIX and not suffix_groups:
            suffix_groups = (config.exact_suffix_rows,)
        if mask_mode == _MASK_COMPACT_SUFFIX and (
            config.exact_prefix_rows > 0
            or not suffix_groups
            or any(group % _BLOCK for group in suffix_groups)
            or mask.shape[1] != len(suffix_groups) + 1
        ):
            raise ValueError(
                "Compact Sol masks require one template per 64-row-aligned suffix group, "
                "one target template, and no exact prefix."
            )
        mask = mask.astype(mx.bfloat16)
    if rows < config.min_tokens:
        dense_mask = (
            materialize_compact_attention_mask(mask, suffix_groups)
            if mask_mode == _MASK_COMPACT_SUFFIX
            else mask
        )
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=dense_mask)
    blocks = math.ceil(rows / _BLOCK)
    prefix_blocks = min(blocks, math.ceil(config.exact_prefix_rows / _BLOCK))
    suffix_blocks = min(
        blocks - prefix_blocks,
        math.ceil(config.exact_suffix_rows / _BLOCK),
    )
    qc, kc, vc = _summaries(q, k, v)
    thresholds = _thresholds(qc, kc, scale=scale, tau=config.tau)
    threadgroup = (32, 4, 1)
    query_tiles = math.ceil(rows / 32)
    physical, route_counts = _kernel(
        prefix_blocks,
        suffix_blocks,
        blocks,
        mask_mode=mask_mode,
        exact_suffix_rows=config.exact_suffix_rows,
        suffix_groups=suffix_groups,
    )(
        inputs=[q, k, v, qc, kc, vc, thresholds, *([mask] if mask is not None else [])],
        template=[
            ("BATCH", batch),
            ("HEADS", heads),
            ("ROWS", rows),
        ],
        grid=(math.ceil(rows / 32) * 32, heads * 4, batch),
        threadgroup=threadgroup,
        output_shapes=[
            (batch, rows, heads, head_dim),
            (batch, heads, query_tiles, 2),
        ],
        output_dtypes=[mx.bfloat16, mx.uint32],
    )
    output = physical.transpose(0, 2, 1, 3)
    if return_route_counts:
        return output, route_counts
    return output


def supports_sol_attention(q: mx.array, mask: mx.array | None, config: SolAttentionConfig) -> bool:
    """Return whether a call can safely use the current experimental kernel."""

    return bool(
        config.enabled
        and (
            mask is None
            or (
                mask.ndim == 3
                and (
                    (
                        tuple(mask.shape) == (q.shape[0], q.shape[-2], q.shape[-2])
                        and (config.exact_prefix_rows > 0 or config.exact_suffix_rows > 0)
                    )
                    or (
                        mask.shape[0] == q.shape[0]
                        and mask.shape[2] == q.shape[-2]
                        and mask.shape[1]
                        == len(
                            config.exact_suffix_groups or (config.exact_suffix_rows,)
                        )
                        + 1
                        and config.exact_prefix_rows == 0
                        and config.exact_suffix_rows > 0
                        and all(
                            rows % _BLOCK == 0
                            for rows in (
                                config.exact_suffix_groups or (config.exact_suffix_rows,)
                            )
                        )
                    )
                )
            )
        )
        and q.ndim == 4
        and q.dtype == mx.bfloat16
        and q.shape[-1] == 128
        and q.shape[-2] >= config.min_tokens
    )


def materialize_compact_attention_mask(
    mask: mx.array,
    suffix_groups: tuple[int, ...] = (),
) -> mx.array:
    """Expand target/reference templates for a dense fallback."""

    if mask.ndim != 3 or mask.shape[1] < 2:
        return mask
    rows = int(mask.shape[2])
    if suffix_groups:
        if len(suffix_groups) + 1 != int(mask.shape[1]):
            raise ValueError("Compact attention template count does not match suffix groups.")
        target_rows = rows - sum(suffix_groups)
        if target_rows <= 0:
            raise ValueError("Compact attention suffix groups leave no target rows.")
        query_templates = mx.concatenate(
            [
                mx.zeros((target_rows,), dtype=mx.int32),
                *(
                    mx.full((count,), index, dtype=mx.int32)
                    for index, count in enumerate(suffix_groups, start=1)
                ),
            ]
        )
        return mx.take_along_axis(
            mask,
            query_templates[None, :, None],
            axis=1,
        )
    if int(mask.shape[1]) != 2:
        raise ValueError("Grouped compact attention masks require explicit suffix row counts.")
    differs = mask[:, 0, :] != mask[:, 0, :1]
    transition = mx.argmax(differs.astype(mx.int32), axis=1)
    mx.eval(transition)
    starts = [int(value) for value in transition.tolist()]
    if any(value <= 0 or value >= rows for value in starts):
        raise ValueError("Compact attention mask does not contain a valid suffix boundary.")
    query_rows = mx.arange(rows)[None, :, None]
    boundary = mx.array(starts, dtype=mx.int32)[:, None, None]
    target_template = mask[:, 0, :][:, None, :]
    reference_template = mask[:, 1, :][:, None, :]
    return mx.where(query_rows < boundary, target_template, reference_template)


def route_telemetry_report(
    records: list[tuple[int, int, mx.array]],
) -> dict[str, object]:
    """Aggregate fused-kernel route counters with one synchronization."""

    if not records:
        return {}
    totals = mx.stack([mx.sum(counts, axis=(0, 1, 2)) for _step, _block, counts in records])
    mx.eval(totals)
    values = totals.tolist()
    by_step: dict[int, list[int]] = {}
    by_block: dict[int, list[int]] = {}
    for (step, block, _counts), (exact, approximate) in zip(records, values, strict=True):
        for destination, key in ((by_step, step), (by_block, block)):
            current = destination.setdefault(int(key), [0, 0])
            current[0] += int(exact)
            current[1] += int(approximate)

    def summarize(values_by_key: dict[int, list[int]]) -> dict[str, dict[str, object]]:
        result = {}
        for key, (exact, approximate) in sorted(values_by_key.items()):
            total = exact + approximate
            result[str(key)] = {
                "exact_routes": exact,
                "approximate_routes": approximate,
                "approximate_fraction": approximate / total if total else 0.0,
            }
        return result

    exact_routes = sum(int(item[0]) for item in values)
    approximate_routes = sum(int(item[1]) for item in values)
    total_routes = exact_routes + approximate_routes
    dense_key_row_units = total_routes * 64
    processed_key_row_units = exact_routes * 64 + approximate_routes * 8
    avoided_key_row_units = dense_key_row_units - processed_key_row_units
    return {
        "route_count_scope": "query-tile/head/key-block decisions",
        "exact_routes": exact_routes,
        "approximate_routes": approximate_routes,
        "total_routes": total_routes,
        "approximate_fraction": approximate_routes / total_routes if total_routes else 0.0,
        "dense_key_row_units": dense_key_row_units,
        "processed_key_row_units": processed_key_row_units,
        "avoided_key_row_units": avoided_key_row_units,
        "avoided_key_row_fraction": (
            avoided_key_row_units / dense_key_row_units if dense_key_row_units else 0.0
        ),
        "route_by_step": summarize(by_step),
        "route_by_block": summarize(by_block),
    }


__all__ = [
    "SolAttentionConfig",
    "materialize_compact_attention_mask",
    "route_telemetry_report",
    "sol_attention",
    "supports_sol_attention",
]
