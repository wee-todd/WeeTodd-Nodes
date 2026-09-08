"""FastH3 VSA-H3 routing and grouped MLX SDPA execution.

This is an independent MLX implementation of FastVideo's public VSA-H3 contract.  It preserves
segment boundaries in the multimodal prefix, tiles generated video in 4x4x4 token cubes, keeps
prefix queries dense, keeps prefix keys visible to every video query, selects the top 10 percent
of video key tiles per query tile and head, and adds the checkpoint's learned pooled-value
compression branch.

The first backend deliberately composes MLX's fused dense SDPA over small gathered route groups.
It is portable and testable on Apple Silicon; a later Metal block-sparse kernel can replace that
consumer without changing routing or checkpoint semantics.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace

import mlx.core as mx
import numpy as np

_TILE = 64
_TILE_SHAPE = (4, 4, 4)


@dataclass(frozen=True)
class FastH3VSAConfig:
    """Runtime policy for the trained FastH3 VSA-H3 checkpoint."""

    enabled: bool = True
    sparsity: float = 0.9
    min_tokens: int = 4096
    query_tile_batch: int = 8
    prefix_segments: tuple[int, ...] = ()
    video_grid: tuple[int, int, int] = ()
    block_stack_preorder: bool = True
    consumer_backend: str = "grouped_sdpa"
    qkv_prep_backend: str = "mlx"

    def validate(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("FastH3 VSA sparsity must be in [0, 1).")
        if self.min_tokens < _TILE:
            raise ValueError(f"FastH3 VSA min_tokens must be at least {_TILE}.")
        if self.query_tile_batch < 1:
            raise ValueError("FastH3 VSA query_tile_batch must be positive.")
        if self.consumer_backend not in {"grouped_sdpa", "metal_indexed"}:
            raise ValueError(
                "FastH3 VSA consumer_backend must be 'grouped_sdpa' or 'metal_indexed'."
            )
        if self.qkv_prep_backend not in {"mlx", "metal_fused"}:
            raise ValueError(
                "FastH3 VSA qkv_prep_backend must be 'mlx' or 'metal_fused'."
            )
        if any(value < 0 for value in self.prefix_segments):
            raise ValueError("FastH3 VSA prefix segment sizes must be non-negative.")
        if self.video_grid and (len(self.video_grid) != 3 or any(v < 1 for v in self.video_grid)):
            raise ValueError("FastH3 VSA video_grid must contain three positive dimensions.")

    def with_layout(
        self,
        *,
        prefix_segments: tuple[int, ...],
        video_grid: tuple[int, int, int],
    ) -> FastH3VSAConfig:
        configured = replace(
            self,
            prefix_segments=tuple(int(v) for v in prefix_segments),
            video_grid=tuple(int(v) for v in video_grid),
        )
        configured.validate()
        return configured


@dataclass(frozen=True)
class FastH3VSAGeometry:
    scatter_index: mx.array
    preordered_scatter_index: mx.array
    stack_permutation: mx.array
    stack_inverse_permutation: mx.array
    variable_sizes: mx.array
    compact_tile_offsets: mx.array
    compact_row_tiles: mx.array
    prefix_rows: int
    prefix_tiles: int
    video_tiles: int
    padded_rows: int


@functools.lru_cache(maxsize=16)
def build_vsa_h3_geometry(
    prefix_segments: tuple[int, ...],
    video_grid: tuple[int, int, int],
) -> FastH3VSAGeometry:
    """Build the exact segment-pure prefix and 4x4x4 video tile permutation."""

    prefix_segments = tuple(int(value) for value in prefix_segments if value > 0)
    if len(video_grid) != 3 or any(int(value) < 1 for value in video_grid):
        raise ValueError("FastH3 VSA requires a positive three-dimensional video token grid.")
    video_grid = tuple(int(value) for value in video_grid)
    prefix_rows = sum(prefix_segments)
    total_rows = prefix_rows + math.prod(video_grid)
    scatter = np.empty(total_rows, dtype=np.int32)
    sizes: list[int] = []
    tile = 0
    source = 0
    for segment in prefix_segments:
        for start in range(0, segment, _TILE):
            count = min(_TILE, segment - start)
            scatter[source + start : source + start + count] = tile * _TILE + np.arange(
                count, dtype=np.int32
            )
            sizes.append(count)
            tile += 1
        source += segment
    prefix_tiles = tile

    t_size, h_size, w_size = video_grid
    t_tiles = math.ceil(t_size / _TILE_SHAPE[0])
    h_tiles = math.ceil(h_size / _TILE_SHAPE[1])
    w_tiles = math.ceil(w_size / _TILE_SHAPE[2])
    for tile_t in range(t_tiles):
        for tile_h in range(h_tiles):
            for tile_w in range(w_tiles):
                rows: list[int] = []
                for local_t in range(_TILE_SHAPE[0]):
                    current_t = tile_t * _TILE_SHAPE[0] + local_t
                    if current_t >= t_size:
                        continue
                    for local_h in range(_TILE_SHAPE[1]):
                        current_h = tile_h * _TILE_SHAPE[1] + local_h
                        if current_h >= h_size:
                            continue
                        for local_w in range(_TILE_SHAPE[2]):
                            current_w = tile_w * _TILE_SHAPE[2] + local_w
                            if current_w < w_size:
                                rows.append(
                                    prefix_rows
                                    + (current_t * h_size + current_h) * w_size
                                    + current_w
                                )
                count = len(rows)
                scatter[np.asarray(rows, dtype=np.int32)] = tile * _TILE + np.arange(
                    count, dtype=np.int32
                )
                sizes.append(count)
                tile += 1

    if sorted(scatter.tolist()) != sorted(set(scatter.tolist())):
        raise ValueError("FastH3 VSA geometry does not map packed rows injectively.")
    variable_sizes = np.asarray(sizes, dtype=np.int32)
    if int(variable_sizes.sum()) != total_rows or variable_sizes.min() < 1:
        raise ValueError("FastH3 VSA tile sizes do not cover the packed sequence.")
    video_order = np.argsort(scatter[prefix_rows:], kind="stable").astype(np.int32)
    stack_permutation = np.concatenate(
        (
            np.arange(prefix_rows, dtype=np.int32),
            prefix_rows + video_order,
        )
    )
    stack_inverse = np.argsort(stack_permutation, kind="stable").astype(np.int32)
    preordered_scatter = np.concatenate(
        (
            scatter[:prefix_rows],
            np.sort(scatter[prefix_rows:]),
        )
    ).astype(np.int32)
    compact_tile_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(variable_sizes[:-1], dtype=np.int32))
    )
    compact_row_tiles = np.repeat(
        np.arange(len(variable_sizes), dtype=np.int32), variable_sizes
    )
    return FastH3VSAGeometry(
        scatter_index=mx.array(scatter),
        preordered_scatter_index=mx.array(preordered_scatter),
        stack_permutation=mx.array(stack_permutation),
        stack_inverse_permutation=mx.array(stack_inverse),
        variable_sizes=mx.array(variable_sizes),
        compact_tile_offsets=mx.array(compact_tile_offsets),
        compact_row_tiles=mx.array(compact_row_tiles),
        prefix_rows=prefix_rows,
        prefix_tiles=prefix_tiles,
        video_tiles=tile - prefix_tiles,
        padded_rows=tile * _TILE,
    )


def _tile_rows(
    value: mx.array,
    geometry: FastH3VSAGeometry,
    *,
    preordered: bool,
) -> mx.array:
    batch, heads, _rows, width = value.shape
    if preordered:
        prefix_padded_rows = geometry.prefix_tiles * _TILE
        prefix = mx.zeros((batch, heads, prefix_padded_rows, width), dtype=value.dtype)
        if geometry.prefix_rows:
            prefix[:, :, geometry.preordered_scatter_index[: geometry.prefix_rows], :] = value[
                :, :, : geometry.prefix_rows, :
            ]
        video = value[:, :, geometry.prefix_rows :, :]
        video_padded_rows = geometry.video_tiles * _TILE
        if int(video.shape[2]) == video_padded_rows:
            return mx.concatenate((prefix, video), axis=2)
        padded_video = mx.zeros(
            (batch, heads, video_padded_rows, width), dtype=value.dtype
        )
        destinations = (
            geometry.preordered_scatter_index[geometry.prefix_rows :] - prefix_padded_rows
        )
        padded_video[:, :, destinations, :] = video
        return mx.concatenate((prefix, padded_video), axis=2)
    tiled = mx.zeros((batch, heads, geometry.padded_rows, width), dtype=value.dtype)
    tiled[:, :, geometry.scatter_index, :] = value
    return tiled


def vsa_h3_stack_order(
    value: mx.array,
    geometry: FastH3VSAGeometry,
    *,
    inverse: bool = False,
    axis: int = -2,
) -> mx.array:
    """Apply the exact target-video cube order used across the VSA block stack."""

    index = (
        geometry.stack_inverse_permutation
        if inverse
        else geometry.stack_permutation
    )
    if int(value.shape[axis]) != int(index.shape[0]):
        raise ValueError("FastH3 VSA stack permutation does not match the sequence axis.")
    return mx.take(value, index, axis=axis)


def _pool_tiles(value: mx.array, geometry: FastH3VSAGeometry) -> mx.array:
    batch, heads, _rows, width = value.shape
    blocks = value.reshape(batch, heads, -1, _TILE, width)
    summed = mx.sum(blocks.astype(mx.float32), axis=3)
    return summed / geometry.variable_sizes.astype(mx.float32)[None, None, :, None]


def _route_indices(
    scores: mx.array,
    *,
    prefix_tiles: int,
    video_tiles: int,
    sparsity: float,
) -> tuple[mx.array, int]:
    """Return prefix-plus-top-k routes for every video query tile and head."""

    keep = min(video_tiles, max(1, math.ceil((1.0 - sparsity) * video_tiles)))
    video_scores = scores[:, :, prefix_tiles:, prefix_tiles:]
    if keep == video_tiles:
        selected = mx.broadcast_to(
            mx.arange(video_tiles, dtype=mx.int32)[None, None, None, :],
            (*video_scores.shape[:-1], video_tiles),
        )
    else:
        selected = mx.argpartition(-video_scores, kth=keep - 1, axis=-1)[..., :keep]
    selected = selected.astype(mx.int32) + prefix_tiles
    if prefix_tiles:
        prefix = mx.broadcast_to(
            mx.arange(prefix_tiles, dtype=mx.int32)[None, None, None, :],
            (*selected.shape[:-1], prefix_tiles),
        )
        selected = mx.concatenate((prefix, selected), axis=-1)
    return selected, keep


def _gather_routed_blocks(
    blocks: mx.array,
    routes: mx.array,
) -> mx.array:
    batch, heads, total_tiles, _tile, width = blocks.shape
    offsets = (
        mx.arange(batch * heads, dtype=mx.int32).reshape(batch, heads, 1, 1) * total_tiles
    )
    flat_indices = routes + offsets
    return mx.take(
        blocks.reshape(batch * heads * total_tiles, _TILE, width),
        flat_indices,
        axis=0,
    )


def vsa_h3_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    gate: mx.array,
    *,
    scale: float,
    config: FastH3VSAConfig,
) -> tuple[mx.array, mx.array]:
    """Execute VSA-H3 and return ``(output, [selected, skipped] route counts)``."""

    config.validate()
    if not config.enabled:
        raise ValueError("FastH3 VSA attention received a disabled configuration.")
    if q.shape != k.shape or q.shape != v.shape or q.shape != gate.shape:
        raise ValueError("FastH3 VSA requires matching Q, K, V, and gate shapes.")
    if q.ndim != 4 or q.shape[-1] != 128:
        raise ValueError("FastH3 VSA currently requires [batch, heads, rows, 128].")
    if not config.prefix_segments or not config.video_grid:
        raise ValueError("FastH3 VSA runtime layout has not been resolved.")
    geometry = build_vsa_h3_geometry(config.prefix_segments, config.video_grid)
    if int(q.shape[-2]) != geometry.prefix_rows + math.prod(config.video_grid):
        raise ValueError("FastH3 VSA layout does not match the packed attention sequence.")

    preordered = bool(config.block_stack_preorder)
    compact_indexed = config.consumer_backend == "metal_indexed"
    if compact_indexed and not preordered:
        raise ValueError("FastH3 indexed Metal attention requires block-stack preordering.")
    if compact_indexed:
        from .vsa_h3_metal import vsa_h3_compact_summaries

        pooled_q, pooled_k, pooled_v = vsa_h3_compact_summaries(
            q,
            k,
            v,
            geometry.compact_tile_offsets,
            geometry.variable_sizes,
        )
        tiled_q = tiled_k = tiled_v = None
    else:
        tiled_q = _tile_rows(q, geometry, preordered=preordered)
        tiled_k = _tile_rows(k, geometry, preordered=preordered)
        tiled_v = _tile_rows(v, geometry, preordered=preordered)
        pooled_q = _pool_tiles(tiled_q, geometry)
        pooled_k = _pool_tiles(tiled_k, geometry)
        pooled_v = _pool_tiles(tiled_v, geometry)
    scores = (pooled_q @ pooled_k.swapaxes(-1, -2)) * scale

    # The learned correction branch exists even for a dense fallback.
    compressed_tiles = mx.softmax(scores, axis=-1) @ pooled_v

    use_sparse = int(q.shape[-2]) >= config.min_tokens and config.sparsity > 0.0
    if not use_sparse:
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        total_tiles = geometry.prefix_tiles + geometry.video_tiles
        selected = int(q.shape[0]) * int(q.shape[1]) * total_tiles * total_tiles
        counts = mx.array([selected, 0], dtype=mx.uint32).reshape(1, 1, 1, 2)
    else:
        routes, kept_video_tiles = _route_indices(
            scores,
            prefix_tiles=geometry.prefix_tiles,
            video_tiles=geometry.video_tiles,
            sparsity=config.sparsity,
        )
        mx.eval(routes, scores, pooled_v)
        batch, heads, _rows, width = q.shape
        total_tiles = geometry.prefix_tiles + geometry.video_tiles
        prefix_output = mx.fast.scaled_dot_product_attention(
            q[:, :, : geometry.prefix_rows], k, v, scale=scale
        )
        if compact_indexed:
            if q.dtype != mx.bfloat16:
                raise TypeError(
                    "FastH3 VSA metal_indexed was requested but Q/K/V are not BF16."
                )
            from .vsa_h3_metal import vsa_h3_indexed_attention

            video_output = vsa_h3_indexed_attention(
                q[:, :, geometry.prefix_rows :],
                k,
                v,
                routes,
                geometry.compact_tile_offsets,
                geometry.variable_sizes,
                prefix_tiles=geometry.prefix_tiles,
                prefix_rows=geometry.prefix_rows,
                scale=scale,
            )
        else:
            if tiled_q is None or tiled_k is None or tiled_v is None:
                raise RuntimeError("FastH3 grouped VSA tile materialization is missing.")
            k_blocks = tiled_k.reshape(batch, heads, total_tiles, _TILE, width)
            v_blocks = tiled_v.reshape(batch, heads, total_tiles, _TILE, width)
            video_q = tiled_q.reshape(batch, heads, total_tiles, _TILE, width)[
                :, :, geometry.prefix_tiles :
            ]
            pieces = []
            for start in range(0, geometry.video_tiles, config.query_tile_batch):
                stop = min(start + config.query_tile_batch, geometry.video_tiles)
                chunk_routes = routes[:, :, start:stop]
                gathered_k = _gather_routed_blocks(k_blocks, chunk_routes)
                gathered_v = _gather_routed_blocks(v_blocks, chunk_routes)
                size_offsets = (
                    mx.arange(batch * heads, dtype=mx.int32).reshape(batch, heads, 1, 1)
                    * total_tiles
                )
                gathered_sizes = mx.take(
                    mx.broadcast_to(
                        geometry.variable_sizes[None, None, :],
                        (batch, heads, total_tiles),
                    ).reshape(-1),
                    chunk_routes + size_offsets,
                )
                query_tiles = stop - start
                route_tiles = int(chunk_routes.shape[-1])
                call_batch = batch * heads * query_tiles
                chunk_q = video_q[:, :, start:stop].reshape(call_batch, 1, _TILE, width)
                chunk_k = gathered_k.reshape(call_batch, 1, route_tiles * _TILE, width)
                chunk_v = gathered_v.reshape(call_batch, 1, route_tiles * _TILE, width)
                valid = (
                    mx.arange(_TILE, dtype=mx.int32)[None, None, None, None, :]
                    < gathered_sizes[..., None]
                ).reshape(call_batch, 1, 1, route_tiles * _TILE)
                additive_mask = mx.where(
                    valid,
                    mx.array(0.0, dtype=q.dtype),
                    mx.array(float("-inf"), dtype=q.dtype),
                )
                piece = mx.fast.scaled_dot_product_attention(
                    chunk_q,
                    chunk_k,
                    chunk_v,
                    scale=scale,
                    mask=additive_mask,
                ).reshape(batch, heads, query_tiles, _TILE, width)
                mx.eval(piece)
                pieces.append(piece)
            video_tiled = mx.concatenate(pieces, axis=2).reshape(
                batch, heads, geometry.video_tiles * _TILE, width
            )
            scatter = (
                geometry.preordered_scatter_index if preordered else geometry.scatter_index
            )
            video_scatter = scatter[geometry.prefix_rows :] - geometry.prefix_tiles * _TILE
            video_output = mx.take(video_tiled, video_scatter, axis=2)
        output = mx.concatenate((prefix_output, video_output), axis=2)

        selected_per_video_query = geometry.prefix_tiles + kept_video_tiles
        skipped_per_video_query = total_tiles - selected_per_video_query
        selected_routes = (
            geometry.prefix_tiles * total_tiles + geometry.video_tiles * selected_per_video_query
        ) * batch * heads
        skipped_routes = geometry.video_tiles * skipped_per_video_query * batch * heads
        counts = mx.array(
            [selected_routes, skipped_routes], dtype=mx.uint32
        ).reshape(1, 1, 1, 2)

    if compact_indexed:
        compressed = mx.take(
            compressed_tiles, geometry.compact_row_tiles, axis=2
        ).astype(output.dtype)
    else:
        compressed_tiled = mx.repeat(compressed_tiles, _TILE, axis=2)
        scatter = geometry.preordered_scatter_index if preordered else geometry.scatter_index
        compressed = mx.take(compressed_tiled, scatter, axis=2).astype(output.dtype)
    return output + compressed * gate.astype(output.dtype), counts


def vsa_h3_route_report(records: list[tuple[int, int, mx.array]]) -> dict[str, object]:
    """Aggregate strict VSA selected/skipped tile routes."""

    if not records:
        return {}
    totals = mx.stack([mx.sum(counts, axis=(0, 1, 2)) for _s, _b, counts in records])
    mx.eval(totals)
    values = totals.tolist()
    selected = sum(int(item[0]) for item in values)
    skipped = sum(int(item[1]) for item in values)
    total = selected + skipped
    return {
        "route_count_scope": "query-tile/head/key-tile decisions",
        "selected_routes": selected,
        "skipped_routes": skipped,
        "total_routes": total,
        "skipped_fraction": skipped / total if total else 0.0,
        "dense_key_row_units": total * _TILE,
        "processed_key_row_units": selected * _TILE,
        "avoided_key_row_units": skipped * _TILE,
        "avoided_key_row_fraction": skipped / total if total else 0.0,
    }


__all__ = [
    "FastH3VSAConfig",
    "FastH3VSAGeometry",
    "build_vsa_h3_geometry",
    "vsa_h3_stack_order",
    "vsa_h3_attention",
    "vsa_h3_route_report",
]
