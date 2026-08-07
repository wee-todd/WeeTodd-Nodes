"""Direct staged publication from synchronized MLX H3 latents."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimax_h3_mlx.media import save_wav

from .decoding import (
    AUDIO_VAE_RUNTIME,
    VIDEO_VAE_RUNTIME,
    H3AudioVAESpec,
    H3VideoVAESpec,
)
from .publishing import _available_target, _metadata_object


@dataclass(frozen=True)
class H3DirectPublicationResult:
    """Published file paths and direct-decode metadata."""

    video_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


class RawVideoEncoder:
    """Write uint8 RGB chunks to one silent H.264 stream."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for direct H3 publication.")
        self.path = path
        self.width = width
        self.height = height
        self.frames = 0
        self._closed = False
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk)
        if chunk.dtype != np.uint8 or chunk.ndim != 4 or chunk.shape[-1] != 3:
            raise ValueError("Direct video chunks must use uint8 RGB frame layout.")
        if chunk.shape[1:3] != (self.height, self.width):
            raise ValueError(
                f"Direct video chunk geometry is {chunk.shape[2]}x{chunk.shape[1]}; "
                f"expected {self.width}x{self.height}."
            )
        if self._process.stdin is None:
            raise RuntimeError("Direct video encoder has no input stream.")
        self._process.stdin.write(np.ascontiguousarray(chunk).tobytes())
        self.frames += int(chunk.shape[0])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        error = self._process.stderr.read() if self._process.stderr is not None else b""
        code = self._process.wait()
        if code:
            raise RuntimeError(f"ffmpeg video encode failed: {error.decode()[:500]}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._closed = True


def _mux_audio(silent_video: Path, audio_path: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for direct H3 publication.")
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent_video),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg audio mux failed: {completed.stderr.decode()[:500]}")


def _timing(
    frames: int,
    width: int,
    height: int,
    fps: float,
    audio: np.ndarray,
    sample_rate: int,
    max_av_drift_seconds: float,
) -> dict[str, float | int]:
    if frames < 1 or width < 1 or height < 1:
        raise ValueError("Direct publication requires non-empty video geometry.")
    if fps != 24.0 or sample_rate != 32000:
        raise ValueError("Direct H3 publication requires 24 fps video and 32000 Hz audio.")
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < 1:
        raise ValueError("Direct H3 publication requires stereo audio with shape (2, samples).")
    if not np.isfinite(audio).all():
        raise ValueError("Direct H3 publication audio contains non-finite samples.")
    video_seconds = frames / fps
    audio_seconds = audio.shape[1] / sample_rate
    drift = abs(video_seconds - audio_seconds)
    if drift > max_av_drift_seconds + 1e-9:
        raise ValueError(
            f"Audio and video durations differ by {drift:.6f} seconds, above the allowed "
            f"{max_av_drift_seconds:.6f} seconds."
        )
    return {
        "frames": frames,
        "width": width,
        "height": height,
        "fps": fps,
        "audio_channels": int(audio.shape[0]),
        "audio_samples": int(audio.shape[1]),
        "sample_rate": sample_rate,
        "video_duration_seconds": video_seconds,
        "audio_duration_seconds": audio_seconds,
        "av_drift_seconds": drift,
    }


def publish_latents_direct(
    target: str | Path,
    components,
    latents,
    *,
    crf: int = 18,
    max_av_drift_seconds: float = 0.025,
    generation_metadata: str = "{}",
    check_interrupted: Callable[[], None] | None = None,
    prepare_video_stage: Callable[[], None] | None = None,
    prepare_audio_stage: Callable[[], None] | None = None,
    metadata_updates: Callable[[], dict[str, Any]] | None = None,
    video_cache=VIDEO_VAE_RUNTIME,
    audio_cache=AUDIO_VAE_RUNTIME,
    encoder_factory=RawVideoEncoder,
    muxer=_mux_audio,
) -> H3DirectPublicationResult:
    """Decode, stream, mux, and atomically publish one synchronized latent result."""
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51.")
    if max_av_drift_seconds < 0:
        raise ValueError("Maximum audio-video drift must be zero or positive.")
    if float(latents.fps) != 24.0 or int(latents.sample_rate) != 32000:
        raise ValueError("Direct H3 publication requires 24 fps video and 32000 Hz audio.")
    supplied = _metadata_object(generation_metadata)
    target = _available_target(Path(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_suffix(".json")
    silent_video = target.with_name(f".{target.stem}.video.partial.mp4")
    audio_path = target.with_name(f".{target.stem}.audio.partial.wav")
    partial_video = target.with_name(f".{target.stem}.partial.mp4")
    partial_metadata = target.with_name(f".{target.stem}.metadata.partial.json")
    encoder = None
    published = False
    started = time.perf_counter()

    try:
        if check_interrupted is not None:
            check_interrupted()
        encoder = encoder_factory(
            silent_video,
            latents.width,
            latents.height,
            float(latents.fps),
            crf,
        )
        video = video_cache.decode_stream(
            H3VideoVAESpec.from_components(components),
            latents,
            encoder.write,
            unload_after=True,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_video_stage,
        )
        encoder.close()
        if encoder.frames != video.num_frames:
            raise ValueError(
                f"Direct encoder received {encoder.frames} frames; expected {video.num_frames}."
            )

        audio = audio_cache.decode(
            H3AudioVAESpec.from_components(components),
            latents,
            unload_after=True,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_audio_stage,
        )
        measured = _timing(
            video.num_frames,
            video.width,
            video.height,
            float(video.fps),
            audio.waveform,
            audio.sample_rate,
            max_av_drift_seconds,
        )
        save_wav(audio_path, audio.waveform, audio.sample_rate)
        if check_interrupted is not None:
            check_interrupted()
        muxer(silent_video, audio_path, partial_video)
        if check_interrupted is not None:
            check_interrupted()

        metadata = {
            **supplied,
            "format": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
            "publication_mode": "direct_mlx_stream",
            "crf": crf,
            "video_decode_seconds": video.decode_seconds,
            "audio_decode_seconds": audio.decode_seconds,
            "peak_rgb8_chunk_bytes": video.peak_rgb8_chunk_bytes,
            "tile_decode_batch": video.decode_batch,
            "publish_seconds": time.perf_counter() - started,
            "output_file": target.name,
            **measured,
        }
        if metadata_updates is not None:
            updates = metadata_updates()
            if not isinstance(updates, dict):
                raise TypeError("Direct publication metadata updates must be a dictionary.")
            metadata.update(updates)
        partial_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial_video, target)
        published = True
        os.replace(partial_metadata, metadata_path)
        return H3DirectPublicationResult(target, metadata_path, metadata)
    except BaseException:
        if encoder is not None:
            encoder.abort()
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        video_cache.unload()
        audio_cache.unload()
        silent_video.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        partial_video.unlink(missing_ok=True)
        partial_metadata.unlink(missing_ok=True)
