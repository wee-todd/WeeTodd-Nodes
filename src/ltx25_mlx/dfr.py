"""LTX 2.5 Diffusion Fidelity Rendering helpers."""

from __future__ import annotations

DFR_SEGMENT_CANDIDATES = (24, 32)


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


__all__ = ["DFR_SEGMENT_CANDIDATES", "choose_dfr_segment_length", "resolve_dfr_canvas"]
