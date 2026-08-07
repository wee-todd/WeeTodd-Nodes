"""Packed-row accounting for modality-specific H3 experiments."""

from __future__ import annotations

from dataclasses import dataclass

from minimax_h3_mlx.packing import (
    AUDIO_CHANNELS,
    FPS,
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)


@dataclass(frozen=True)
class ModalityRows:
    text: slice
    audio: slice
    video: slice
    counts: dict[str, int]


def t2va_modality_rows(
    *,
    total_rows: int,
    text_rows: int,
    duration_seconds: float,
    height: int,
    width: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> ModalityRows:
    """Reconstruct text/audio/video row slices for an unconditioned T2VA packed sequence."""
    if min(total_rows, text_rows, height, width) < 1 or duration_seconds <= 0:
        raise ValueError("row counts, dimensions, and duration must be positive")
    frames = align_num_frames(int(round(duration_seconds * FPS)))
    audio_rows = audio_latent_num_frames(frames) * AUDIO_CHANNELS
    latent_frames = video_latent_num_frames(frames)
    _, patch_height, patch_width = patch_size
    latent_height, latent_width = height // 16, width // 16
    if latent_height % patch_height or latent_width % patch_width:
        raise ValueError("latent dimensions must be divisible by the patch size")
    video_rows = (
        latent_frames
        * (latent_height // patch_height)
        * (latent_width // patch_width)
    )
    expected = text_rows + audio_rows + video_rows
    if expected != total_rows:
        raise ValueError(f"packed row count mismatch: expected {expected}, captured {total_rows}")
    audio_start = text_rows
    video_start = audio_start + audio_rows
    return ModalityRows(
        text=slice(0, text_rows),
        audio=slice(audio_start, video_start),
        video=slice(video_start, total_rows),
        counts={"text": text_rows, "audio": audio_rows, "video": video_rows},
    )
