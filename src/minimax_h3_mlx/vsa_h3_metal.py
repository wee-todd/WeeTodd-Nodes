"""Indexed Metal consumer for trained FastH3 VSA routes.

The router remains in :mod:`minimax_h3_mlx.vsa_h3`. This module independently specializes the
MLX-distributed Steel attention body so each target-video query tile visits only its explicit
dense-prefix and selected video key tiles. It consumes compact preordered storage directly and
does not materialize padded or gathered Q/K/V tensors.
"""

from __future__ import annotations

import math

import mlx.core as mx

import wee_todd_mlx.sol_attention as _steel


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Installed MLX Steel attention layout changed at {label}.")
    return source.replace(old, new, 1)


def _indexed_body() -> str:
    body = _steel._DENSE_BODY
    body = _replace_once(
        body,
        """  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence
""",
        """  const uint route_query_tile = tid.x / 2;
  const uint query_half = tid.x % 2;
  const uint query_tile = PREFIX_TILES + route_query_tile;
  const int query_block_size = int(TILE_SIZES[query_tile]) - int(query_half * BQ);
  if (query_block_size <= 0) return;
  const uint query_offset = uint(TILE_OFFSETS[query_tile] - TILE_OFFSETS[PREFIX_TILES])
      + query_half * BQ;
  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      ulong(query_offset) * params->Q_strides[2]; // Compact sequence
""",
        "compact Q offset",
    )
    body = _replace_once(
        body,
        """  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence
""",
        """  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      ulong(query_offset) * params->O_strides[2]; // Compact sequence
""",
        "compact output offset",
    )
    body = _replace_once(
        body,
        """  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
""",
        """  if (query_block_size < BQ) {
    loader_q.load_safe(short2(BD, query_block_size));
  } else {
    loader_q.load_unsafe();
  }
""",
        "compact Q load",
    )
    body = _replace_once(
        body,
        "  // Loop over KV seq length\n  for (int kb = 0; kb < kb_lim; kb++) {",
        r"""  // Each pair of compact 32-row query blocks shares one trained 64-row VSA route.
  const ulong route_base =
      ((ulong(tidl.z) * HEADS + tidl.y) * VIDEO_TILES + route_query_tile) * ROUTE_TILES;
  for (uint route_slot = 0; route_slot < ROUTE_TILES; ++route_slot) {
    const int kb = int(ROUTES[route_base + route_slot]);
    const uint selected_size = uint(TILE_SIZES[kb]);
    KBlockLoader selected_loader_k(
        K + ulong(TILE_OFFSETS[kb]) * params->K_strides[2],
        params->K_strides[2], Ks, simd_group_id, simd_lane_id);
    VBlockLoader selected_loader_v(
        V + ulong(TILE_OFFSETS[kb]) * params->V_strides[2],
        params->V_strides[2], Vs, simd_group_id, simd_lane_id);""",
        "indexed KV loop",
    )
    body = _replace_once(
        body,
        """    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }
""",
        """    if (selected_size < BK) {
      selected_loader_k.load_safe(short2(BD, selected_size));
    } else {
      selected_loader_k.load_unsafe();
    }
""",
        "indexed K load",
    )
    body = _replace_once(
        body,
        """    // Mask out length sequence
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }
""",
        """    // Mask internal padding in every ragged prefix or video tile.
    if (selected_size < BK) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= selected_size) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }
""",
        "indexed padding mask",
    )
    body = _replace_once(
        body,
        """    if (!align_K && kb == (params->NK_aligned)) {
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_v.load_unsafe();
    }
""",
        """    if (selected_size < BK) {
      selected_loader_v.load_safe(short2(BD, selected_size));
    } else {
      selected_loader_v.load_unsafe();
    }
""",
        "indexed V load",
    )
    body = _replace_once(
        body,
        """    // Prepare for next iteration
    loader_k.next();
    loader_v.next();
""",
        """    // Indexed loaders are reconstructed at the selected tile address.
""",
        "indexed loader advance",
    )
    body = _replace_once(
        body,
        """  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
""",
        """  if (query_block_size < BQ) {
    auto dst_tile_dims = short2(BD - sn, query_block_size - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
""",
        "compact output store",
    )
    return body


_INDEXED_BODY = _indexed_body()
_COMPACT_SUMMARY_KERNEL = mx.fast.metal_kernel(
    name="wee_todd_fasth3_vsa_compact_summaries_bf16_d128",
    input_names=["q", "k", "v", "tile_offsets", "tile_sizes"],
    output_names=["qc", "kc", "vc"],
    source=r"""
        uint index = thread_position_in_grid.x;
        if (index >= BATCH * HEADS * TILES * HEAD_DIM) return;
        uint dim = index % HEAD_DIM;
        uint tile = (index / HEAD_DIM) % TILES;
        uint head = (index / (HEAD_DIM * TILES)) % HEADS;
        uint batch = index / (HEAD_DIM * TILES * HEADS);
        uint start = uint(tile_offsets[tile]);
        uint length = uint(tile_sizes[tile]);
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
        kc[index] = k_sum / float(length);
        vc[index] = v_sum / float(length);
    """,
    ensure_row_contiguous=False,
)


def vsa_h3_compact_summaries(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    tile_offsets: mx.array,
    tile_sizes: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Pool compact preordered Q/K/V rows without materializing padded tile tensors."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("FastH3 compact summaries require matching [B,H,R,D] tensors.")
    if q.dtype != mx.bfloat16 or k.dtype != mx.bfloat16 or v.dtype != mx.bfloat16:
        raise TypeError("FastH3 compact summaries require BF16 Q/K/V tensors.")
    batch, heads, _rows, head_dim = map(int, q.shape)
    if head_dim != 128:
        raise ValueError("FastH3 compact summaries require head dimension 128.")
    if tile_offsets.ndim != 1 or tile_sizes.shape != tile_offsets.shape:
        raise ValueError("FastH3 compact summary tile metadata is inconsistent.")
    tiles = int(tile_sizes.shape[0])
    values = _COMPACT_SUMMARY_KERNEL(
        inputs=[
            q,
            k,
            v,
            tile_offsets.astype(mx.int32),
            tile_sizes.astype(mx.int32),
        ],
        template=[
            ("BATCH", batch),
            ("HEADS", heads),
            ("TILES", tiles),
            ("HEAD_DIM", head_dim),
        ],
        grid=(batch * heads * tiles * head_dim, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (batch, heads, tiles, head_dim),
            (batch, heads, tiles, head_dim),
            (batch, heads, tiles, head_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return tuple(values)


_KERNELS: dict[tuple[int, int, int], object] = {}


def _kernel(prefix_tiles: int, video_tiles: int, route_tiles: int):
    key = (prefix_tiles, video_tiles, route_tiles)
    kernel = _KERNELS.get(key)
    if kernel is not None:
        return kernel
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
constexpr int VIDEO_TILES = {video_tiles};
constexpr int ROUTE_TILES = {route_tiles};
constexpr int PREFIX_TILES = {prefix_tiles};
constexpr bool align_Q = true;
constexpr bool align_K = true;
constexpr bool has_mask = false;
constexpr bool do_causal = false;
constexpr bool has_sinks = false;

const device T* Q = q;
const device T* K = k;
const device T* V = v;
auto ROUTES = routes;
auto TILE_OFFSETS = tile_offsets;
auto TILE_SIZES = tile_sizes;
device T* O = output;
AttnParams params_value{{
    BATCH, HEADS, BD,
    QUERY_ROWS, KEY_ROWS,
    1, 0.08838834764831845f,
    QUERY_ROWS / BQ, KEY_ROWS / BK,
    QUERY_ROWS / BQ, KEY_ROWS / BK,
    0, 0, 0,
    {{q_strides[0], q_strides[1], q_strides[2]}},
    {{k_strides[0], k_strides[1], k_strides[2]}},
    {{v_strides[0], v_strides[1], v_strides[2]}},
    {{QUERY_ROWS * HEADS * BD, BD, HEADS * BD}}
}};
thread const AttnParams* params = &params_value;
thread const AttnMaskParams* mask_params = nullptr;
const device MaskType* mask = nullptr;
const device T* sinks = nullptr;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;
uint3 tid = threadgroup_position_in_grid;
uint3 lid = thread_position_in_threadgroup;
{_INDEXED_BODY}
"""
    kernel = mx.fast.metal_kernel(
        name=f"wee_todd_fasth3_vsa_compact_p{prefix_tiles}_v{video_tiles}_r{route_tiles}",
        input_names=["q", "k", "v", "routes", "tile_offsets", "tile_sizes"],
        output_names=["output"],
        source=source,
        header=_steel._STEEL_HEADER,
        ensure_row_contiguous=False,
    )
    _KERNELS[key] = kernel
    return kernel


def vsa_h3_indexed_attention(
    video_q: mx.array,
    compact_k: mx.array,
    compact_v: mx.array,
    routes: mx.array,
    tile_offsets: mx.array,
    tile_sizes: mx.array,
    *,
    prefix_tiles: int,
    prefix_rows: int,
    scale: float,
) -> mx.array:
    """Return compact target-video attention without padded or gathered Q/K/V buffers."""

    if video_q.ndim != 4 or video_q.dtype != mx.bfloat16:
        raise TypeError("FastH3 indexed Metal attention requires BF16 [B,H,Q,128] queries.")
    if video_q.shape[-1] != 128:
        raise ValueError("FastH3 indexed Metal attention requires head dimension 128.")
    if compact_k.shape != compact_v.shape or compact_k.ndim != 4:
        raise ValueError("FastH3 indexed Metal attention requires matching compact K/V tensors.")
    if compact_k.dtype != mx.bfloat16 or compact_v.dtype != mx.bfloat16:
        raise TypeError("FastH3 indexed Metal attention requires BF16 K/V tensors.")
    batch, heads, query_rows, head_dim = map(int, video_q.shape)
    key_rows = int(compact_k.shape[2])
    if tuple(compact_k.shape[:2]) != (batch, heads) or int(compact_k.shape[3]) != head_dim:
        raise ValueError("FastH3 indexed Metal K/V batch or head dimensions do not match Q.")
    key_tiles = int(tile_sizes.shape[0])
    video_tiles = key_tiles - int(prefix_tiles)
    if prefix_tiles < 1 or video_tiles < 1:
        raise ValueError("FastH3 indexed Metal tile partition is invalid.")
    if key_rows - int(prefix_rows) != query_rows:
        raise ValueError("FastH3 indexed Metal compact video rows do not match K/V rows.")
    if routes.ndim != 4 or tuple(routes.shape[:3]) != (batch, heads, video_tiles):
        raise ValueError("FastH3 indexed Metal routes do not match Q tiles and heads.")
    if routes.dtype != mx.int32:
        routes = routes.astype(mx.int32)
    route_tiles = int(routes.shape[-1])
    if route_tiles < 1 or route_tiles > key_tiles:
        raise ValueError("FastH3 indexed Metal route width is invalid.")
    if tile_sizes.shape != (key_tiles,) or tile_offsets.shape != (key_tiles,):
        raise ValueError("FastH3 indexed Metal tile metadata does not match compact K/V.")
    if not math.isclose(scale, 128**-0.5, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("FastH3 indexed Metal attention requires the H3 D=128 scale.")
    physical = _kernel(prefix_tiles, video_tiles, route_tiles)(
        inputs=[
            video_q,
            compact_k,
            compact_v,
            routes,
            tile_offsets.astype(mx.int32),
            tile_sizes.astype(mx.int32),
        ],
        template=[
            ("BATCH", batch),
            ("HEADS", heads),
            ("QUERY_ROWS", query_rows),
            ("KEY_ROWS", key_rows),
        ],
        grid=(video_tiles * 64, heads * 4, batch),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, query_rows, heads, head_dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return physical.transpose(0, 2, 1, 3)


__all__ = ["vsa_h3_compact_summaries", "vsa_h3_indexed_attention"]
