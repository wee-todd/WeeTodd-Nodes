"""Packed reference-conditioning geometry for MiniMax H3 Ref2VA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import mlx.core as mx
import numpy as np

from .config import TAG_AUDIO, TAG_VIDEO
from .packing import (
    _ROPE_FRAME_RESCALE,
    _ROPE_FRAMES_PER_LATENT,
    AUDIO_CHANNELS,
    PackedSequence,
    _spatial_position_grid,
    _temporal_position_grid,
    video_latent_num_frames,
)

MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12


@dataclass(frozen=True)
class ReferenceDensityDecision:
    """Resolved persistent-video density and the activity evidence behind it."""

    density: float
    sampled_pairs: int = 0
    activity_mean: float | None = None
    activity_p95: float | None = None
    reason: str = "explicit"


@dataclass
class PreparedReference:
    """Latent geometry of one prepared Ref2VA reference block."""

    kind: Literal["image", "video", "audio"]
    num_latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0
    image: Any | None = None
    frames: np.ndarray | None = None
    waveform: np.ndarray | None = None
    block_timestamps: list[float] = field(default_factory=list)
    qwen_frames: np.ndarray | None = None
    source_num_latent_frames: int = 0
    target_frame: int | None = None
    temporal_density_requested: str = "full"
    temporal_density_resolved: float = 1.0
    temporal_activity_mean: float | None = None
    temporal_activity_p95: float | None = None
    temporal_density_reason: str | None = None

    @property
    def has_audio(self) -> bool:
        return self.waveform is not None or self.num_audio_latents > 0

    def validate(self, patch_size: tuple[int, int, int]) -> None:
        _, patch_height, patch_width = patch_size
        if self.kind not in {"image", "video", "audio"}:
            raise ValueError("Ref2VA reference kind must be 'image', 'video', or 'audio'.")
        if self.num_audio_latents < 0:
            raise ValueError("Ref2VA audio latent count must not be negative.")
        if self.target_frame is not None and self.target_frame < 0:
            raise ValueError("A prepared timeline guide must use a resolved non-negative frame.")
        if self.kind == "audio":
            if self.num_audio_latents < 1:
                raise ValueError("A Ref2VA audio reference must contain audio latents.")
            if self.num_latent_frames or self.latent_height or self.latent_width:
                raise ValueError(
                    "A standalone Ref2VA audio reference cannot contain video geometry."
                )
            return
        if self.num_latent_frames < 1:
            raise ValueError("A Ref2VA image or video reference must contain latent frames.")
        if self.latent_height < patch_height or self.latent_width < patch_width:
            raise ValueError("Ref2VA visual latent dimensions are smaller than the H3 patch.")
        if self.latent_height % patch_height or self.latent_width % patch_width:
            raise ValueError("Ref2VA visual latent dimensions must be divisible by the H3 patch.")
        if self.kind == "image" and self.num_latent_frames != 1:
            raise ValueError("A Ref2VA image reference must contain exactly one latent frame.")
        if self.kind == "image" and self.has_audio:
            raise ValueError("A Ref2VA image reference cannot contain soundtrack latents.")
        if self.kind == "video" and self.source_num_latent_frames:
            if self.source_num_latent_frames < self.num_latent_frames:
                raise ValueError(
                    "Ref2VA source latent-frame count cannot be smaller than encoded frames."
                )

    def video_rows(self, patch_size: tuple[int, int, int]) -> int:
        if self.kind == "audio":
            return 0
        _, patch_height, patch_width = patch_size
        return (
            self.num_latent_frames
            * (self.latent_height // patch_height)
            * (self.latent_width // patch_width)
        )

    @property
    def audio_rows(self) -> int:
        return self.num_audio_latents * AUDIO_CHANNELS


def validate_reference_set(
    references: tuple[PreparedReference, ...] | list[PreparedReference],
    patch_size: tuple[int, int, int],
) -> None:
    """Validate released Ref2VA reference-count and modality constraints."""
    if not references:
        raise ValueError("Ref2VA requires at least one reference.")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"Ref2VA supports at most {MAX_REFERENCES} references.")
    for reference in references:
        if not isinstance(reference, PreparedReference):
            raise TypeError("Ref2VA references must be PreparedReference values.")
        reference.validate(patch_size)
    images = sum(reference.kind == "image" for reference in references)
    videos = sum(reference.kind == "video" for reference in references)
    audios = sum(reference.has_audio for reference in references)
    if images > MAX_REFERENCE_IMAGES:
        raise ValueError(f"Ref2VA supports at most {MAX_REFERENCE_IMAGES} image references.")
    if videos > MAX_REFERENCE_VIDEOS:
        raise ValueError(f"Ref2VA supports at most {MAX_REFERENCE_VIDEOS} video references.")
    if audios > MAX_REFERENCE_AUDIOS:
        raise ValueError(f"Ref2VA supports at most {MAX_REFERENCE_AUDIOS} audio references.")
    if (
        not images
        and not videos
        and not all(reference.target_frame is not None for reference in references)
    ):
        raise ValueError(
            "Untimed Ref2VA audio references require at least one image or video reference."
        )


def resample_reference_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Resample decoded frames to H3's 24 fps grid by dropping or repeating frames."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError("A Ref2VA video must have shape (frames, height, width, 3).")
    if fps <= 0:
        raise ValueError("A Ref2VA video frame rate must be positive.")
    if fps == 24.0:
        return frames
    scale = 24.0 / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def trim_reference_num_frames(num_frames: int) -> int:
    """Snap a reference down to the ``17 * n + 5`` frame count encoded without padding."""
    if num_frames < 5:
        raise ValueError("A Ref2VA video must contain at least five frames after resampling.")
    return (num_frames - 5) // 17 * 17 + 5


def resolve_reference_video_density(
    frames: np.ndarray,
    policy: Literal["full", "half", "quarter", "automatic"],
) -> ReferenceDensityDecision:
    """Resolve a conservative density from adjacent-frame activity on the H3 24 fps grid.

    The scan uses a bounded spatial thumbnail of every aligned frame. It never changes the
    full-resolution Qwen presentation frames; it controls only the persistent video-VAE rows
    carried through each transformer evaluation.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Prepared Ref2VA video frames have invalid geometry.")
    explicit = {"full": 1.0, "half": 0.5, "quarter": 0.25}
    if policy in explicit:
        return ReferenceDensityDecision(explicit[policy], reason=f"explicit_{policy}")
    if policy != "automatic":
        raise ValueError(f"Unknown Ref2VA temporal-density policy: {policy!r}.")

    count = trim_reference_num_frames(int(frames.shape[0]))
    if count <= 22:
        return ReferenceDensityDecision(1.0, reason="short_reference_kept_full")

    height, width = frames.shape[1:3]
    row_stride = max(1, math.ceil(height / 64))
    column_stride = max(1, math.ceil(width / 64))
    thumbnail = frames[:count, ::row_stride, ::column_stride, :3].astype(np.float32)
    deltas = np.mean(np.abs(np.diff(thumbnail, axis=0)), axis=(1, 2, 3)) / 255.0
    mean = float(np.mean(deltas))
    p95 = float(np.quantile(deltas, 0.95))
    evidence = {
        "sampled_pairs": int(deltas.shape[0]),
        "activity_mean": mean,
        "activity_p95": p95,
    }
    if mean >= 0.02 or p95 >= 0.06:
        return ReferenceDensityDecision(
            1.0, reason="high_motion_or_cut_kept_full", **evidence
        )
    if mean >= 0.004 or p95 >= 0.015:
        return ReferenceDensityDecision(
            0.5, reason="moderate_redundancy_selected_half", **evidence
        )
    return ReferenceDensityDecision(
        0.25, reason="near_static_reference_selected_quarter", **evidence
    )


def reduce_reference_video_frames(frames: np.ndarray, density: float) -> tuple[np.ndarray, int]:
    """Uniformly reduce aligned Ref2VA frames while retaining the source latent duration."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Prepared Ref2VA video frames have invalid geometry.")
    if not 0.0 < density <= 1.0:
        raise ValueError("Ref2VA temporal density must be greater than zero and at most one.")
    count = trim_reference_num_frames(int(frames.shape[0]))
    source_latents = video_latent_num_frames(count)
    source_chunks = (count - 5) // 17
    target_chunks = round((source_latents * density - 2) / 5)
    target_chunks = min(source_chunks, max(0, target_chunks))
    target_count = 5 + 17 * target_chunks
    aligned = frames[:count]
    if target_count >= count:
        return aligned, source_latents
    indices = np.rint(np.linspace(0, count - 1, target_count)).astype(np.int64)
    return np.ascontiguousarray(aligned[indices]), source_latents


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """Select the 2 fps Qwen vision frames and timestamps from a prepared 24 fps video."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError("Prepared Ref2VA video frames have invalid geometry.")
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        index = round(cursor)
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += 12.0
    timestamps = [index / 2.0 for index in range(len(indices))]
    if len(timestamps) % 2:
        timestamps.append(timestamps[-1])
    blocks = [
        (timestamps[index] + timestamps[index + 1]) / 2.0 for index in range(0, len(timestamps), 2)
    ]
    return [frames[index] for index in indices], blocks


def encode_reference_video_rows(
    video_vae,
    references: list[PreparedReference],
    patch_size: tuple[int, int, int],
) -> mx.array | None:
    """Encode image and video references into normalized, packed H3 video rows."""
    from .packing import PIXEL_MEAN, PIXEL_STD, patchify_video_latents

    cfg = video_vae.config
    latent_mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
    latent_std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
    pixel_mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    rows = []
    for reference in references:
        if reference.kind == "audio":
            continue
        if reference.kind == "image":
            pixels = np.asarray(reference.image, dtype=np.float32)
            pixels = pixels.transpose(2, 0, 1)[None, :, None]
            normalized = (pixels / 255.0 - pixel_mean) / pixel_std
            moments = video_vae._encode_clip(mx.array(normalized).transpose(0, 2, 3, 4, 1))
            moments = moments.transpose(0, 4, 1, 2, 3)
        else:
            frames = np.asarray(reference.frames)
            count = trim_reference_num_frames(int(frames.shape[0]))
            pixels = frames[:count].astype(np.float32).transpose(3, 0, 1, 2)[None]
            normalized = (pixels / 255.0 - pixel_mean) / pixel_std
            moments = video_vae.encode(mx.array(normalized))
        channels = cfg.latent_channels
        # Ref2VA uses the deterministic posterior mean. FL2VA keyframes use a seeded posterior
        # sample, but carrying that behavior across tasks destabilizes genuine Ref2VA weights.
        latent = moments[:, :channels].astype(mx.float32)
        reference.num_latent_frames = int(latent.shape[2])
        reference.latent_height = int(latent.shape[3])
        reference.latent_width = int(latent.shape[4])
        rows.append(patchify_video_latents((latent - latent_mean) / latent_std, patch_size))
    return mx.concatenate(rows) if rows else None


def encode_reference_audio_rows(
    audio_vae,
    references: list[PreparedReference],
) -> mx.array | None:
    """Encode reference waveforms with the audio posterior mean and channel-major packing."""
    cfg = audio_vae.config
    latent_mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(1, 1, -1)
    latent_std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(1, 1, -1)
    rows = []
    for reference in references:
        if not reference.has_audio:
            continue
        waveform = mx.array(np.asarray(reference.waveform, dtype=np.float32))[:, None, :]
        mean, _ = audio_vae.encode(waveform)
        latents = mean.astype(mx.float32).transpose(0, 2, 1)
        reference.num_audio_latents = int(latents.shape[1])
        rows.append(((latents - latent_mean) / latent_std).reshape(-1, cfg.latent_channels))
    return mx.concatenate(rows) if rows else None


def _frame_grid(
    latent_height: int,
    latent_width: int,
    patch_height: int,
    patch_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_height, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_width, sqrt_area)
    height, width = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack([height.reshape(-1), width.reshape(-1)], axis=-1), width_grid


def _fill_audio_positions(
    position_ids: np.ndarray,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: np.ndarray,
) -> None:
    if num_audio_latents == 0:
        return
    time = rotary_time + np.arange(num_audio_latents, dtype=np.float64)
    position_ids[rows, 0] = np.tile(time, AUDIO_CHANNELS)
    position_ids[rows, 2] = np.concatenate(
        [
            np.full(num_audio_latents, width_grid[0], dtype=np.float64),
            np.full(num_audio_latents, width_grid[-1], dtype=np.float64),
        ]
    )


def _reference_video_span(num_latent_frames: int) -> float:
    """Return the sequentially summed rotary span used between reference blocks."""
    return sum(
        _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
        for index in range(num_latent_frames)
    )


def build_ref2va_packed_sequence(
    text_token_tags: np.ndarray | list[int],
    references: tuple[PreparedReference, ...] | list[PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    continuation_video_frames: int = 0,
    continuation_audio_latents: int = 0,
) -> PackedSequence:
    """Build ordered reference rows, optional overlap context, and target geometry.

    Continuation rows share the target rotary origin. They are indexed before the reference rows
    within their modality so callers can provide ``[continuation, references, target]`` tensors
    and keep the overlap pinned to its own timestep.
    """
    validate_reference_set(references, patch_size)
    _, patch_height, patch_width = patch_size
    if num_latent_frames < 1 or num_audio_latents < 1:
        raise ValueError("Ref2VA target video and audio latent counts must be positive.")
    if latent_height % patch_height or latent_width % patch_width:
        raise ValueError("Ref2VA target latent dimensions must be divisible by the H3 patch.")

    text_tags = np.asarray(text_token_tags, dtype=np.int64)
    if text_tags.ndim != 1 or text_tags.size == 0:
        raise ValueError("Ref2VA text token tags must be a non-empty one-dimensional array.")

    target_grid, target_width_grid = _frame_grid(
        latent_height, latent_width, patch_height, patch_width
    )
    reference_video_rows = sum(reference.video_rows(patch_size) for reference in references)
    reference_audio_rows = sum(reference.audio_rows for reference in references)
    rows_per_frame = target_grid.shape[0]
    continuation_video_rows = continuation_video_frames * rows_per_frame
    continuation_audio_rows = continuation_audio_latents * AUDIO_CHANNELS
    target_audio_rows = num_audio_latents * AUDIO_CHANNELS
    target_video_rows = num_latent_frames * target_grid.shape[0]
    sequence_length = (
        text_tags.size
        + reference_video_rows
        + reference_audio_rows
        + continuation_audio_rows
        + continuation_video_rows
        + target_audio_rows
        + target_video_rows
    )

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[: text_tags.size, 0] = np.arange(text_tags.size, dtype=np.float64)
    video_indices: list[np.ndarray] = []
    audio_indices: list[np.ndarray] = []
    cursor = int(text_tags.size)
    rotary_time = float(text_tags.size)

    # Ordinary Ref2VA blocks occupy their own rotary prefix. Timeline guides are positioned
    # against the target clock and do not lengthen that prefix.
    for reference in references:
        if reference.target_frame is not None:
            continue
        if reference.kind == "image":
            rotary_time += 1.0
        elif reference.kind == "audio":
            rotary_time += float(reference.num_audio_latents)
        else:
            rotary_time += max(
                float(reference.num_audio_latents),
                _reference_video_span(
                    max(reference.num_latent_frames, reference.source_num_latent_frames)
                ),
            )
    target_rotary_time = rotary_time
    rotary_time = float(text_tags.size)

    for reference in references:
        reference_time = (
            target_rotary_time + _ROPE_FRAME_RESCALE * reference.target_frame
            if reference.target_frame is not None
            else rotary_time
        )
        if reference.kind == "image":
            stop = cursor + reference.video_rows(patch_size)
            rows = slice(cursor, stop)
            frame_grid, _ = _frame_grid(
                reference.latent_height,
                reference.latent_width,
                patch_height,
                patch_width,
            )
            position_ids[rows, 0] = reference_time
            position_ids[rows, 1:] = frame_grid
            video_indices.append(np.arange(cursor, stop, dtype=np.int64))
            cursor = stop
            if reference.target_frame is None:
                rotary_time += 1.0
            continue

        if reference.kind == "audio":
            stop = cursor + reference.audio_rows
            rows = slice(cursor, stop)
            _fill_audio_positions(
                position_ids,
                rows,
                reference.num_audio_latents,
                reference_time,
                target_width_grid,
            )
            audio_indices.append(np.arange(cursor, stop, dtype=np.int64))
            cursor = stop
            if reference.target_frame is None:
                rotary_time += float(reference.num_audio_latents)
            continue

        audio_stop = cursor + reference.audio_rows
        video_stop = audio_stop + reference.video_rows(patch_size)
        audio_rows = slice(cursor, audio_stop)
        video_rows = slice(audio_stop, video_stop)
        frame_grid, width_grid = _frame_grid(
            reference.latent_height,
            reference.latent_width,
            patch_height,
            patch_width,
        )
        _fill_audio_positions(
            position_ids,
            audio_rows,
            reference.num_audio_latents,
            reference_time,
            width_grid,
        )
        source_latent_frames = max(reference.num_latent_frames, reference.source_num_latent_frames)
        frame_time = _temporal_position_grid(reference.num_latent_frames, reference_time)
        if source_latent_frames > reference.num_latent_frames:
            source_time = _temporal_position_grid(source_latent_frames, reference_time)
            frame_time = np.linspace(source_time[0], source_time[-1], reference.num_latent_frames)
        position_ids[video_rows, 0] = np.repeat(frame_time, frame_grid.shape[0])
        position_ids[video_rows, 1:] = np.tile(frame_grid, (reference.num_latent_frames, 1))
        if reference.audio_rows:
            audio_indices.append(np.arange(cursor, audio_stop, dtype=np.int64))
        video_indices.append(np.arange(audio_stop, video_stop, dtype=np.int64))
        cursor = video_stop
        if reference.target_frame is None:
            rotary_time += max(
                float(reference.num_audio_latents),
                _reference_video_span(source_latent_frames),
            )

    rotary_time = target_rotary_time

    continuation_audio_start = cursor
    continuation_video_start = continuation_audio_start + continuation_audio_rows
    target_audio_start = continuation_video_start + continuation_video_rows
    target_video_start = target_audio_start + target_audio_rows
    if continuation_audio_rows:
        _fill_audio_positions(
            position_ids,
            slice(continuation_audio_start, continuation_video_start),
            continuation_audio_latents,
            rotary_time,
            target_width_grid,
        )
    if continuation_video_rows:
        continuation_time = _temporal_position_grid(continuation_video_frames, rotary_time)
        position_ids[continuation_video_start:target_audio_start, 0] = np.repeat(
            continuation_time, rows_per_frame
        )
        position_ids[continuation_video_start:target_audio_start, 1:] = np.tile(
            target_grid, (continuation_video_frames, 1)
        )
    _fill_audio_positions(
        position_ids,
        slice(target_audio_start, target_video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    target_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[target_video_start:, 0] = np.repeat(target_time, target_grid.shape[0])
    position_ids[target_video_start:, 1:] = np.tile(target_grid, (num_latent_frames, 1))

    target_video_index = np.arange(target_video_start, sequence_length, dtype=np.int64)
    target_audio_index = np.arange(target_audio_start, target_video_start, dtype=np.int64)
    reference_video_index = np.concatenate(video_indices)
    reference_audio_index = (
        np.concatenate(audio_indices) if audio_indices else np.empty(0, np.int64)
    )
    continuation_video_index = np.arange(
        continuation_video_start, target_audio_start, dtype=np.int64
    )
    continuation_audio_index = np.arange(
        continuation_audio_start, continuation_video_start, dtype=np.int64
    )
    video_index = np.concatenate(
        [continuation_video_index, reference_video_index, target_video_index]
    )
    audio_index = np.concatenate(
        [continuation_audio_index, reference_audio_index, target_audio_index]
    )
    text_index = np.arange(text_tags.size, dtype=np.int64)
    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_index] = text_tags
    token_tags[video_index] = TAG_VIDEO
    token_tags[audio_index] = TAG_AUDIO

    return PackedSequence(
        sequence_length=sequence_length,
        position_ids=mx.array(position_ids.astype(np.float32)),
        token_tags=mx.array(token_tags.astype(np.int32)),
        video_indices=mx.array(video_index.astype(np.int32)),
        audio_indices=mx.array(audio_index.astype(np.int32)),
        text_indices=mx.array(text_index.astype(np.int32)),
        num_condition_video_rows=reference_video_rows + continuation_video_rows,
        num_condition_audio_rows=reference_audio_rows + continuation_audio_rows,
        num_continuation_video_rows=continuation_video_rows,
        num_continuation_audio_rows=continuation_audio_rows,
    )
