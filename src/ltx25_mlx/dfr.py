"""LTX 2.5 Diffusion Fidelity Rendering helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx

DFR_SEGMENT_CANDIDATES = (24, 32)
DFR_TILE_LEAD_SEGMENTS = 1


class DFRTemporalTile(NamedTuple):
    pixel_start: int
    pixel_end: int
    latent_start: int
    latent_end_exclusive: int
    anchor_frames: tuple[int, ...]
    slot_frames: tuple[int, ...]
    drop_latent_prefix: int


class DFRTemporalImageAnchor(NamedTuple):
    pixel_frame: int
    latent_tokens: mx.array
    strength: float
    replace: bool


def extract_dfr_temporal_image_anchors(
    conditionings: Sequence[object], *, latent_h: int, latent_w: int
) -> tuple[DFRTemporalImageAnchor, ...]:
    """Retain already encoded explicit image anchors for temporal DFR rounds."""
    rows_per_frame = int(latent_h) * int(latent_w)
    anchors: list[DFRTemporalImageAnchor] = []
    for conditioning in conditionings:
        frame_indices = getattr(conditioning, "frame_indices", None)
        clean_latent = getattr(conditioning, "clean_latent", None)
        if frame_indices is not None and clean_latent is not None:
            for index, pixel_frame in enumerate(frame_indices):
                start = index * rows_per_frame
                end = start + rows_per_frame
                if end > clean_latent.shape[1]:
                    raise ValueError("An LTX 2.5 image anchor has an invalid latent row count.")
                anchors.append(
                    DFRTemporalImageAnchor(
                        int(pixel_frame),
                        mx.contiguous(clean_latent[:, start:end]),
                        float(conditioning.strength),
                        True,
                    )
                )
            continue
        pixel_frame = getattr(conditioning, "frame_idx", None)
        keyframe_latent = getattr(conditioning, "keyframe_latent", None)
        if pixel_frame is not None and keyframe_latent is not None:
            if keyframe_latent.shape[1] != rows_per_frame:
                raise ValueError("An LTX 2.5 keyframe anchor has an invalid latent row count.")
            anchors.append(
                DFRTemporalImageAnchor(
                    int(pixel_frame),
                    mx.contiguous(keyframe_latent),
                    float(conditioning.strength),
                    False,
                )
            )
    if len({anchor.pixel_frame for anchor in anchors}) != len(anchors):
        raise ValueError("Temporal DFR image anchors must target distinct pixel frames.")
    return tuple(sorted(anchors, key=lambda anchor: anchor.pixel_frame))


def scale_dfr_temporal_image_anchors(
    anchors: Sequence[DFRTemporalImageAnchor], factor: int = 2
) -> tuple[DFRTemporalImageAnchor, ...]:
    if factor < 1:
        raise ValueError("Temporal DFR image-anchor scale must be positive.")
    return tuple(
        DFRTemporalImageAnchor(
            anchor.pixel_frame * factor,
            anchor.latent_tokens,
            anchor.strength,
            anchor.replace,
        )
        for anchor in anchors
    )


def select_dfr_generated_slot_tokens(latent: mx.array, slot_rows: int) -> mx.array:
    """Select the trailing rows marked as generated keyframe slots."""
    rows = int(slot_rows)
    if rows < 0 or rows > latent.shape[1]:
        raise ValueError("Temporal DFR generated-slot row count is invalid.")
    return latent[:, latent.shape[1] - rows :] if rows else latent[:, :0]


def choose_dfr_segment_length(content_frames: int) -> int:
    """Choose the 24/32-frame segment that needs the least tail padding."""
    if content_frames < 1:
        raise ValueError("DFR needs at least one content frame interval.")

    def padding(segment: int) -> int:
        return (segment - content_frames % segment) % segment

    return min(DFR_SEGMENT_CANDIDATES, key=lambda value: (padding(value), -value))


def resolve_dfr_canvas(
    num_frames: int, temporal_scale: int = 8
) -> tuple[int, int, tuple[int, ...]]:
    """Return padded frame count, segment length, and generated keyframe positions."""
    if num_frames < 2 or (num_frames - 1) % temporal_scale:
        raise ValueError(
            "DFR frame count must be at least two and align to "
            f"x{temporal_scale} temporal compression."
        )
    content = num_frames - 1
    segment = choose_dfr_segment_length(content)
    padded = content + (segment - content % segment) % segment
    positions = tuple(range(segment, padded + 1, segment))
    return padded + 1, segment, positions


def _latent_index(pixel_frame: int, temporal_scale: int = 8) -> int:
    if pixel_frame < 0 or (pixel_frame and pixel_frame % temporal_scale):
        raise ValueError(f"DFR frame {pixel_frame} is not on the temporal latent grid.")
    return pixel_frame // temporal_scale


def plan_dfr_temporal_tiles(
    seam_frames: Sequence[int],
    num_frames: int,
    tile_count: int,
    *,
    temporal_scale: int = 8,
) -> tuple[DFRTemporalTile, ...]:
    """Partition one temporal round into seam-aware denoise tiles."""
    seams = tuple(int(value) for value in seam_frames)
    if not seams or seams[-1] != num_frames - 1:
        raise ValueError("DFR temporal seams must end on the final output frame.")
    boundaries = (0, *seams)
    spans = tuple(
        right - left for left, right in zip(boundaries, boundaries[1:], strict=False)
    )
    if any(span < temporal_scale * 2 or span % temporal_scale for span in spans):
        raise ValueError("DFR temporal seam spans must contain at least two latent intervals.")
    count = min(max(1, int(tile_count)), len(spans))
    base, remainder = divmod(len(spans), count)
    owned_counts = tuple(base + (1 if index < remainder else 0) for index in range(count))
    tiles = []
    own_start = 0
    for index, owned in enumerate(owned_counts):
        own_end = own_start + owned
        window_start = max(0, own_start - (DFR_TILE_LEAD_SEGMENTS if index else 0))
        pixel_start = boundaries[window_start]
        pixel_end = boundaries[own_end]
        latent_start = _latent_index(pixel_start, temporal_scale)
        drop = _latent_index(boundaries[own_start], temporal_scale) - latent_start
        if own_start:
            drop += 1
        anchors = tuple(
            boundaries[item]
            for item in range(window_start, own_end + 1)
            if boundaries[item]
        )
        slots = tuple(
            (boundaries[item] + boundaries[item + 1]) // 2
            for item in range(window_start, own_end)
        )
        tiles.append(
            DFRTemporalTile(
                pixel_start,
                pixel_end,
                latent_start,
                _latent_index(pixel_end, temporal_scale) + 1,
                anchors,
                slots,
                drop,
            )
        )
        own_start = own_end
    return tuple(tiles)


def stitch_dfr_temporal_tiles(
    latents: Sequence[mx.array], tiles: Sequence[DFRTemporalTile]
) -> mx.array:
    if len(latents) != len(tiles) or not latents:
        raise ValueError("DFR temporal tile outputs must match the tile plan.")
    pieces = []
    for latent, tile in zip(latents, tiles, strict=True):
        expected = tile.latent_end_exclusive - tile.latent_start
        if latent.ndim != 5 or latent.shape[2] != expected:
            raise ValueError("A DFR temporal tile returned an invalid latent shape.")
        if not 0 <= tile.drop_latent_prefix < expected:
            raise ValueError("A DFR temporal tile has an invalid discarded prefix.")
        pieces.append(latent[:, :, tile.drop_latent_prefix :])
    return mx.concatenate(pieces, axis=2)


__all__ = [
    "DFR_SEGMENT_CANDIDATES",
    "DFRTemporalImageAnchor",
    "DFRTemporalTile",
    "choose_dfr_segment_length",
    "extract_dfr_temporal_image_anchors",
    "plan_dfr_temporal_tiles",
    "resolve_dfr_canvas",
    "scale_dfr_temporal_image_anchors",
    "select_dfr_generated_slot_tokens",
    "stitch_dfr_temporal_tiles",
]
