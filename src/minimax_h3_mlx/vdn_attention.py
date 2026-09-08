"""Direct indexed attention for VDN's static windows, without K/V gathers.

Uses this project's MLX Steel specialization, not FastH3's trained routing or gates.
Every query visits the same ordered global/local/anchor segments as grouped SDPA.
"""

import math
from functools import lru_cache

import mlx.core as mx

import wee_todd_mlx.sol_attention as _steel

from .vsa_h3_metal import _INDEXED_BODY, _replace_once


@lru_cache(maxsize=16)
def window_plan(layout, key_tile=64):
    from .vdn import _window_bounds

    groups = []
    if layout.video_start:
        groups.append((0, layout.video_start, [(0, layout.sequence)]))
    frame = 0
    bounds = _window_bounds(layout.frames)
    size = layout.tokens_per_frame
    while frame < layout.frames:
        lo, hi = bounds[frame]
        stop = frame + 1
        if frame not in (0, layout.frames - 1):
            while stop < layout.frames - 1 and bounds[stop] == (lo, hi):
                stop += 1
            lo, hi = max(lo, 0), min(hi, layout.frames - 1)
            segments = [
                (0, layout.video_start),
                (layout.video_end, layout.sequence),
                (layout.video_start + lo * size, layout.video_start + (hi + 1) * size),
            ]
            for anchor in (0, layout.frames - 1):
                if not lo <= anchor <= hi:
                    segments.append(
                        (
                            layout.video_start + anchor * size,
                            layout.video_start + (anchor + 1) * size,
                        )
                    )
        else:
            segments = [(0, layout.sequence)]
        groups.append(
            (layout.video_start + frame * size, layout.video_start + stop * size, segments)
        )
        frame = stop
    if layout.video_end < layout.sequence:
        groups.append((layout.video_end, layout.sequence, [(0, layout.sequence)]))
    qs, qn, qg, ks, kn, rs, rn = [], [], [], [], [], [], []
    for group, (start, stop, segments) in enumerate(groups):
        rs.append(len(ks))
        for a, b in segments:
            for offset in range(a, b, key_tile):
                ks.append(offset)
                kn.append(min(key_tile, b - offset))
        rn.append(len(ks) - rs[-1])
        for offset in range(start, stop, 32):
            qs.append(offset)
            qn.append(min(32, stop - offset))
            qg.append(group)
    return tuple(mx.array(v, dtype=mx.int32) for v in (qs, qn, qg, ks, kn, rs, rn))


@lru_cache(maxsize=3)
def _kernel(key_tile=64):
    body = _replace_once(
        _INDEXED_BODY,
        """  const uint route_query_tile = tid.x / 2;
  const uint query_half = tid.x % 2;
  const uint query_tile = PREFIX_TILES + route_query_tile;
  const int query_block_size = int(TILE_SIZES[query_tile]) - int(query_half * BQ);
  if (query_block_size <= 0) return;
  const uint query_offset = uint(TILE_OFFSETS[query_tile] - TILE_OFFSETS[PREFIX_TILES])
      + query_half * BQ;""",
        """  const uint query_offset = uint(query_offsets[tid.x]);
  const int query_block_size = int(query_sizes[tid.x]);
  const uint query_group = uint(query_groups[tid.x]);""",
        "VDN query plan",
    )
    body = _replace_once(
        body,
        """  const ulong route_base =
      ((ulong(tidl.z) * HEADS + tidl.y) * VIDEO_TILES + route_query_tile) * ROUTE_TILES;
  for (uint route_slot = 0; route_slot < ROUTE_TILES; ++route_slot) {
    const int kb = int(ROUTES[route_base + route_slot]);""",
        """  const uint route_base = uint(route_starts[query_group]);
  for (uint route_slot = 0; route_slot < uint(route_counts[query_group]); ++route_slot) {
    const int kb = int(route_base + route_slot);""",
        "VDN static routes",
    )
    body = body.replace(
        "Each pair of compact 32-row query blocks shares one trained 64-row VSA route.",
        "Each 32-row query block uses its static VDN global/local/anchor key segments.",
    )
    source = (
        """
using namespace mlx::steel;
using T = bfloat;
using MaskType = bfloat;
using AccumType = float;
constexpr int BQ = 32, BK = 64, BD = 128, WM = 4, WN = 1;
constexpr bool align_Q = true, align_K = true, has_mask = false;
constexpr bool do_causal = false, has_sinks = false;
const device T* Q = q;
const device T* K = k;
const device T* V = v;
auto TILE_OFFSETS = key_offsets;
auto TILE_SIZES = key_sizes;
device T* O = output;
AttnParams params_value{
    BATCH, HEADS, BD, ROWS, ROWS, 1, 0.08838834764831845f,
    ROWS / BQ, ROWS / BK, ROWS / BQ, ROWS / BK, 0, 0, 0,
    {q_strides[0], q_strides[1], q_strides[2]},
    {k_strides[0], k_strides[1], k_strides[2]},
    {v_strides[0], v_strides[1], v_strides[2]},
    {ROWS * HEADS * BD, BD, HEADS * BD}
};
thread const AttnParams* params = &params_value;
thread const AttnMaskParams* mask_params = nullptr;
const device MaskType* mask = nullptr;
const device T* sinks = nullptr;
uint simd_lane_id = thread_index_in_simdgroup;
uint simd_group_id = simdgroup_index_in_threadgroup;
uint3 tid = threadgroup_position_in_grid;
uint3 lid = thread_position_in_threadgroup;
""".replace("BK = 64", f"BK = {key_tile}")
        + body
    )
    return mx.fast.metal_kernel(
        name=f"weetodd_vdn_static_indexed_attention_k{key_tile}",
        source=source,
        header=_steel._STEEL_HEADER,
        input_names=[
            "q",
            "k",
            "v",
            "query_offsets",
            "query_sizes",
            "query_groups",
            "key_offsets",
            "key_sizes",
            "route_starts",
            "route_counts",
        ],
        output_names=["output"],
        ensure_row_contiguous=False,
    )


def indexed_attention(q, k, v, layout, scale, *, key_tile=16):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("VDN indexed attention requires matching BHSD Q/K/V.")
    if any(a.dtype != mx.bfloat16 for a in (q, k, v)) or q.shape[-1] != 128:
        raise ValueError("VDN indexed attention requires BF16 D=128.")
    if q.shape[2] != layout.sequence or not math.isclose(scale, 128**-0.5, abs_tol=1e-12):
        raise ValueError("VDN indexed attention layout or scale mismatch.")
    if key_tile not in {16, 32, 64}:
        raise ValueError("VDN indexed key tile must be 16, 32, or 64.")
    plan = window_plan(layout, key_tile)
    batch, heads, rows, dim = q.shape
    output = _kernel(key_tile)(
        inputs=[q, k, v, *plan],
        template=[("BATCH", batch), ("HEADS", heads), ("ROWS", rows)],
        grid=(plan[0].size * 32, heads * 4, batch),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, rows, heads, dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return output.transpose(0, 2, 1, 3)
