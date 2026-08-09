"""Validated atomic publication of synchronized H3 video and audio."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MediaWriter = Callable[..., Path]


@dataclass(frozen=True)
class H3PublicationResult:
    """Published media paths and measured synchronization metadata."""

    video_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


def validate_synchronized_media(
    video: Any,
    audio: Any,
    sample_rate: int,
    fps: float,
    max_av_drift_seconds: float,
) -> dict[str, float | int]:
    """Validate final media arrays before filesystem mutation."""
    import numpy as np

    if video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < 1:
        raise ValueError(
            "Video must have shape (frames, height, width, 3) with at least one frame; "
            f"got {video.shape}."
        )
    if video.dtype != np.uint8:
        raise ValueError(f"Video must use uint8 RGB values; got {video.dtype}.")
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < 1:
        raise ValueError(
            f"Audio must have shape (2, samples) with stereo channels; got {audio.shape}."
        )
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError(f"Audio must use floating-point samples; got {audio.dtype}.")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples. Decode the audio again.")
    if sample_rate != 32000:
        raise ValueError(f"MiniMax H3 publication requires 32000 Hz audio; got {sample_rate}.")
    if fps != 24.0:
        raise ValueError(f"MiniMax H3 publication requires 24 fps video; got {fps}.")
    if max_av_drift_seconds < 0:
        raise ValueError("Maximum audio-video drift must be zero or positive.")

    video_seconds = video.shape[0] / fps
    audio_seconds = audio.shape[1] / sample_rate
    drift_seconds = abs(video_seconds - audio_seconds)
    if drift_seconds > max_av_drift_seconds + 1e-9:
        raise ValueError(
            "Audio and video durations differ by "
            f"{drift_seconds:.6f} seconds, above the allowed "
            f"{max_av_drift_seconds:.6f} seconds."
        )
    return {
        "frames": int(video.shape[0]),
        "width": int(video.shape[2]),
        "height": int(video.shape[1]),
        "fps": fps,
        "audio_channels": int(audio.shape[0]),
        "audio_samples": int(audio.shape[1]),
        "sample_rate": sample_rate,
        "video_duration_seconds": video_seconds,
        "audio_duration_seconds": audio_seconds,
        "av_drift_seconds": drift_seconds,
    }


def _metadata_object(value: str) -> dict[str, Any]:
    try:
        metadata = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("Generation metadata must be a valid JSON object.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Generation metadata must be a JSON object.")
    return metadata


def _available_target(base: Path) -> Path:
    if not base.exists() and not base.with_suffix(".json").exists():
        return base
    for index in range(1, 100000):
        candidate = base.with_name(f"{base.stem}_{index:05d}{base.suffix}")
        if not candidate.exists() and not candidate.with_suffix(".json").exists():
            return candidate
    raise RuntimeError(f"No available output filename remains for: {base.name}")


def publish_synchronized_media(
    target: str | Path,
    video: Any,
    audio: Any,
    *,
    sample_rate: int = 32000,
    fps: float = 24.0,
    crf: int = 18,
    max_av_drift_seconds: float = 0.025,
    generation_metadata: str = "{}",
    writer: MediaWriter | None = None,
    check_interrupted: Callable[[], None] | None = None,
    ffmpeg_path: str | Path | None = None,
) -> H3PublicationResult:
    """Validate, atomically encode, and describe one synchronized H3 result."""
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51.")
    measured = validate_synchronized_media(video, audio, sample_rate, fps, max_av_drift_seconds)
    supplied = _metadata_object(generation_metadata)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _available_target(target)
    metadata_path = target.with_suffix(".json")
    temporary_video = target.with_name(f".{target.stem}.partial{target.suffix}")
    temporary_audio = temporary_video.with_suffix(".wav")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.partial")
    if writer is None:
        from minimax_h3_mlx.media import save_mp4

        def resolved_writer(path, video, fps, audio, sample_rate, crf):
            return save_mp4(
                path,
                video,
                fps,
                audio,
                sample_rate,
                crf,
                ffmpeg_path=ffmpeg_path,
            )

        writer = resolved_writer

    published_video = False
    try:
        if check_interrupted is not None:
            check_interrupted()
        started = time.perf_counter()
        writer(
            temporary_video,
            video,
            fps,
            audio,
            sample_rate,
            crf=crf,
        )
        encode_seconds = time.perf_counter() - started
        if check_interrupted is not None:
            check_interrupted()
        metadata = {
            **supplied,
            "format": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
            "crf": crf,
            "encode_seconds": encode_seconds,
            "output_file": target.name,
            **measured,
        }
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_video, target)
        published_video = True
        os.replace(temporary_metadata, metadata_path)
        return H3PublicationResult(target, metadata_path, metadata)
    except BaseException:
        temporary_video.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        if published_video:
            target.unlink(missing_ok=True)
        raise
    finally:
        temporary_audio.unlink(missing_ok=True)
