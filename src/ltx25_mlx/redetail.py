"""Pure planning utilities for bounded LTX 2.5 generative video re-detailing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LTX25RedetailChunk:
    index: int
    start_frame: int
    end_frame: int
    input_frames: int
    vae_frames: int
    start_seconds: float
    end_seconds: float
    boundary_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ltx25_vae_frame_count(frames: int) -> int:
    """Return the smallest supported temporal extent that contains ``frames``."""
    if frames < 1:
        raise ValueError("LTX 2.5 chunk frame count must be positive.")
    return 1 + 8 * math.ceil((frames - 1) / 8)


def detect_scene_cut_scores(video: Any) -> list[float]:
    """Measure adjacent-frame change on a bounded, low-resolution RGB sample."""
    import numpy as np

    frames = np.asarray(video, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Scene-cut input must have shape (frames, height, width, 3).")
    if frames.shape[0] < 2:
        return []
    stride_y = max(1, math.ceil(frames.shape[1] / 64))
    stride_x = max(1, math.ceil(frames.shape[2] / 64))
    sample = frames[:, ::stride_y, ::stride_x, :]
    luminance = (
        sample[..., 0] * 0.2126 + sample[..., 1] * 0.7152 + sample[..., 2] * 0.0722
    )
    return [
        float(value)
        for value in np.mean(np.abs(luminance[1:] - luminance[:-1]), axis=(1, 2))
    ]


def scene_cut_candidates(scores: Sequence[float]) -> tuple[int, ...]:
    """Return frame boundaries with change well above local motion noise."""
    if not scores:
        return ()
    import numpy as np

    values = np.asarray(scores, dtype=np.float32)
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    threshold = max(0.08, median + 6.0 * max(deviation, 1e-4))
    return tuple(int(index + 1) for index, value in enumerate(values) if value >= threshold)


def plan_redetail_chunks(
    total_frames: int,
    *,
    fps: float,
    output_width: int,
    output_height: int,
    frame_megapixel_budget: float,
    cut_frames: Sequence[int] = (),
    min_chunk_frames: int = 49,
) -> tuple[LTX25RedetailChunk, ...]:
    """Partition a clip without gaps while respecting a per-chunk workload ceiling."""
    if total_frames < 1:
        raise ValueError("LTX 2.5 re-detail input must contain at least one frame.")
    if fps <= 0:
        raise ValueError("LTX 2.5 re-detail fps must be positive.")
    if output_width < 1 or output_height < 1:
        raise ValueError("LTX 2.5 re-detail output dimensions must be positive.")
    if frame_megapixel_budget <= 0:
        raise ValueError("Chunk frame-megapixel budget must be positive.")
    per_frame = output_width * output_height / 1_000_000.0
    max_frames = int(frame_megapixel_budget // per_frame)
    if max_frames < 9:
        raise ValueError(
            "Chunk frame-megapixel budget permits fewer than nine output frames. "
            "Reduce the output dimensions or raise the budget."
        )
    if total_frames <= max_frames:
        boundaries = [0, total_frames]
        reasons = ["complete clip"]
    else:
        chunk_count = math.ceil(total_frames / max_frames)
        effective_min = max(9, min_chunk_frames)
        if max_frames < effective_min:
            raise ValueError(
                "Chunk budget permits less than the 49-frame quality floor. "
                "Raise the budget or reduce the output dimensions."
            )
        if chunk_count * effective_min > total_frames:
            raise ValueError(
                "Chunk budget cannot partition this clip into supported temporal windows. "
                "Raise the frame-megapixel budget or use a single pass."
            )
        valid_cuts = sorted({int(value) for value in cut_frames if 0 < value < total_frames})
        boundaries = [0]
        reasons = []
        for index in range(1, chunk_count):
            ideal = round(index * total_frames / chunk_count)
            remaining = chunk_count - index
            candidates = [
                cut
                for cut in valid_cuts
                if cut - boundaries[-1] >= effective_min
                and cut - boundaries[-1] <= max_frames
                and total_frames - cut >= effective_min * remaining
                and total_frames - cut <= max_frames * remaining
                and abs(cut - ideal) <= max_frames * 0.35
            ]
            if candidates:
                boundary = min(candidates, key=lambda value: (abs(value - ideal), value))
                reasons.append("scene cut")
            else:
                low = boundaries[-1] + effective_min
                high = min(boundaries[-1] + max_frames, total_frames - effective_min * remaining)
                boundary = min(max(ideal, low), high)
                reasons.append("workload boundary")
            boundaries.append(boundary)
        boundaries.append(total_frames)
        reasons.append("complete clip")

    chunks = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        count = end - start
        if count < 1 or count > max_frames:
            raise RuntimeError("LTX 2.5 chunk planner produced an invalid partition.")
        chunks.append(
            LTX25RedetailChunk(
                index=index,
                start_frame=start,
                end_frame=end,
                input_frames=count,
                vae_frames=ltx25_vae_frame_count(count),
                start_seconds=start / fps,
                end_seconds=end / fps,
                boundary_reason=reasons[index],
            )
        )
    if chunks[0].start_frame != 0 or chunks[-1].end_frame != total_frames:
        raise RuntimeError("LTX 2.5 chunk plan does not cover the complete clip.")
    if any(
        left.end_frame != right.start_frame
        for left, right in zip(chunks, chunks[1:], strict=False)
    ):
        raise RuntimeError("LTX 2.5 chunk plan contains a frame gap or overlap.")
    return tuple(chunks)


def audio_sample_bounds(
    start_frame: int,
    end_frame: int,
    *,
    fps: float,
    sample_rate: int,
    total_samples: int,
) -> tuple[int, int]:
    """Map a half-open video-frame interval to a contiguous audio-sample interval."""
    if fps <= 0 or sample_rate < 1 or total_samples < 1:
        raise ValueError("Audio-bound inputs must be positive.")
    start = min(total_samples, max(0, round(start_frame * sample_rate / fps)))
    end = min(total_samples, max(start, round(end_frame * sample_rate / fps)))
    return start, end


__all__ = [
    "LTX25RedetailChunk",
    "audio_sample_bounds",
    "detect_scene_cut_scores",
    "ltx25_vae_frame_count",
    "plan_redetail_chunks",
    "scene_cut_candidates",
]
