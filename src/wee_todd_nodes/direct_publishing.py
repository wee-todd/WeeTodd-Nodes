"""Direct staged publication from synchronized MLX H3 latents."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimax_h3_mlx.media import FFmpegExecutable, resolve_ffmpeg, save_wav

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

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        ffmpeg: FFmpegExecutable | None = None,
    ) -> None:
        ffmpeg = ffmpeg or resolve_ffmpeg()
        self.path = path
        self.width = width
        self.height = height
        self.frames = 0
        self._closed = False
        command = [
            str(ffmpeg.path),
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


def _cosine_ramp(length: int) -> np.ndarray:
    """Return a raised-cosine transition that selects both aligned endpoints."""
    if length < 2:
        raise ValueError("A join transition must contain at least two samples.")
    position = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(np.pi * position)


def _motion_matched_overlap(
    previous: np.ndarray,
    following: np.ndarray,
    *,
    blend_frames: int = 4,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Join two aligned video overlaps at their least disruptive temporal seam."""
    previous = np.asarray(previous)
    following = np.asarray(following)
    if previous.shape != following.shape or previous.ndim != 4:
        raise ValueError("Aligned video overlaps must have the same four-dimensional shape.")
    count = int(previous.shape[0])
    if count < 2:
        raise ValueError("A motion-matched video overlap requires at least two frames.")
    if previous.dtype != np.uint8 or following.dtype != np.uint8:
        raise ValueError("Motion-matched video overlaps must use uint8 RGB frames.")

    # A sparse view keeps seam selection cheap while preserving the large composition changes
    # that make a chained cut visible. Candidate k transitions from previous[k - 1] to following[k].
    prior_view = previous[:, ::4, ::4].astype(np.int16, copy=False)
    next_view = following[:, ::4, ::4].astype(np.int16, copy=False)
    scores = np.asarray(
        [np.mean(np.abs(prior_view[index - 1] - next_view[index])) for index in range(1, count)],
        dtype=np.float64,
    )
    seam = int(np.argmin(scores)) + 1
    blend_frames = min(max(int(blend_frames), 1), count)
    start = max(0, min(seam - blend_frames // 2, count - blend_frames))
    stop = start + blend_frames
    ramp = _cosine_ramp(blend_frames)

    # The preceding overlap is no longer needed after this call, so blend it in place. Mixing
    # one frame at a time avoids a full float32 overlap allocation at native H3 resolutions.
    joined = previous
    for offset, alpha in enumerate(ramp):
        frame = start + offset
        mixed = (
            joined[frame].astype(np.float32) * (1.0 - alpha)
            + following[frame].astype(np.float32) * alpha
        )
        joined[frame] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    joined[stop:] = following[stop:]
    return joined, {
        "seam_frame_in_overlap": seam,
        "blend_start_frame": start,
        "blend_frames": blend_frames,
        "seam_score": float(scores[seam - 1]),
        "fixed_end_score": float(scores[-1]),
    }


class _ChainedVideoAssembler:
    """Stream windows while retaining only the RGB overlap needed to repair each join."""

    def __init__(self, encoder, target_frames: int, overlap_frames: int) -> None:
        self.encoder = encoder
        self.remaining = int(target_frames)
        self.overlap_frames = int(overlap_frames)
        self.previous_overlap: np.ndarray | None = None
        self.current_tail: np.ndarray | None = None
        self.head_parts: list[np.ndarray] = []
        self.head_frames = 0
        self.window_index = -1
        self.window_published = 0
        self.join_reports: list[dict[str, float | int]] = []

    def begin_window(self, index: int) -> None:
        self.window_index = index
        self.current_tail = None
        self.head_parts = []
        self.head_frames = 0
        self.window_published = 0

    def _emit(self, frames: np.ndarray) -> None:
        if self.remaining <= 0 or frames.shape[0] == 0:
            return
        frames = frames[: self.remaining]
        self.encoder.write(frames)
        written = int(frames.shape[0])
        self.window_published += written
        self.remaining -= written

    def _retain_tail(self, frames: np.ndarray) -> None:
        if frames.shape[0] == 0:
            return
        combined = (
            frames
            if self.current_tail is None
            else np.concatenate((self.current_tail, frames), axis=0)
        )
        if combined.shape[0] > self.overlap_frames:
            split = int(combined.shape[0]) - self.overlap_frames
            self._emit(combined[:split])
            combined = combined[split:]
        self.current_tail = np.ascontiguousarray(combined)

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk)
        if self.window_index == 0:
            self._retain_tail(chunk)
            return
        if self.previous_overlap is None:
            raise RuntimeError("A chained video window is missing its preceding overlap.")
        if self.head_frames < self.overlap_frames:
            take = min(self.overlap_frames - self.head_frames, int(chunk.shape[0]))
            if take:
                self.head_parts.append(np.ascontiguousarray(chunk[:take]))
                self.head_frames += take
                chunk = chunk[take:]
            if self.head_frames == self.overlap_frames:
                following = np.concatenate(self.head_parts, axis=0)
                self.head_parts.clear()
                joined, report = _motion_matched_overlap(
                    self.previous_overlap,
                    following,
                )
                report = {"join": self.window_index, **report}
                self.join_reports.append(report)
                self._emit(joined)
        self._retain_tail(chunk)

    def end_window(self) -> int:
        if self.window_index > 0 and self.head_frames != self.overlap_frames:
            raise ValueError("A chained video window ended before its overlap was decoded.")
        if self.current_tail is None or self.current_tail.shape[0] != self.overlap_frames:
            raise ValueError("A chained video window is too short for its configured overlap.")
        self.previous_overlap = self.current_tail
        return self.window_published

    def finish(self) -> None:
        if self.previous_overlap is None:
            raise ValueError("The chained video assembler did not receive any windows.")
        self._emit(self.previous_overlap)


def _splice_audio_windows(
    waveforms: list[np.ndarray],
    overlap_samples: int,
    video_joins: list[dict[str, float | int]],
    context_frames: int,
    *,
    crossfade_samples: int = 1600,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Splice aligned audio overlaps at the video seam with a short cosine crossfade."""
    if not waveforms:
        raise ValueError("An audio chain requires at least one waveform.")
    merged = np.asarray(waveforms[0])
    reports: list[dict[str, float | int]] = []
    for index, following in enumerate(waveforms[1:]):
        following = np.asarray(following)
        overlap = min(int(overlap_samples), merged.shape[1], following.shape[1])
        if overlap < 2:
            raise ValueError("An audio chain overlap must contain at least two samples.")
        video_seam = int(video_joins[index]["seam_frame_in_overlap"])
        seam = round(video_seam / context_frames * overlap)
        fade = min(max(int(crossfade_samples), 2), overlap)
        start = max(0, min(seam - fade // 2, overlap - fade))
        stop = start + fade
        prior_overlap = merged[:, -overlap:]
        next_overlap = following[:, :overlap]
        joined_overlap = prior_overlap.copy()
        ramp = _cosine_ramp(fade).reshape((1, -1))
        joined_overlap[:, start:stop] = (
            prior_overlap[:, start:stop] * (1.0 - ramp) + next_overlap[:, start:stop] * ramp
        )
        joined_overlap[:, stop:] = next_overlap[:, stop:]
        hard_delta = float(np.max(np.abs(next_overlap[:, seam] - prior_overlap[:, seam - 1])))
        mixed_delta = float(
            np.max(
                np.abs(
                    joined_overlap[:, start]
                    - (prior_overlap[:, start - 1] if start else prior_overlap[:, 0])
                )
            )
        )
        merged = np.concatenate(
            (merged[:, :-overlap], joined_overlap, following[:, overlap:]),
            axis=1,
        )
        reports.append(
            {
                "join": index + 1,
                "seam_sample_in_overlap": seam,
                "crossfade_start_sample": start,
                "crossfade_samples": fade,
                "hard_cut_peak_delta": hard_delta,
                "crossfade_entry_peak_delta": mixed_delta,
            }
        )
    return merged, reports


def _mux_audio(
    silent_video: Path,
    audio_path: Path,
    output: Path,
    ffmpeg: FFmpegExecutable | None = None,
) -> None:
    ffmpeg = ffmpeg or resolve_ffmpeg()
    command = [
        str(ffmpeg.path),
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


def _mlx_process_peak_bytes() -> int | None:
    """Report MLX Metal allocations; process RSS does not include them reliably on macOS."""
    try:
        import mlx.core as mx

        return int(mx.get_peak_memory())
    except (AttributeError, ImportError):
        return None


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
    ffmpeg_path: str | Path | None = None,
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
    ffmpeg = None

    try:
        if encoder_factory is RawVideoEncoder or muxer is _mux_audio:
            ffmpeg = resolve_ffmpeg(ffmpeg_path)
        if check_interrupted is not None:
            check_interrupted()
        if encoder_factory is RawVideoEncoder:
            encoder = encoder_factory(
                silent_video,
                latents.width,
                latents.height,
                float(latents.fps),
                crf,
                ffmpeg,
            )
        else:
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
        if muxer is _mux_audio:
            muxer(silent_video, audio_path, partial_video, ffmpeg)
        else:
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
            "video_vae_quantization": video.quantization,
            "tile_decode_batch": video.decode_batch,
            "mlx_process_peak_bytes": _mlx_process_peak_bytes(),
            "publish_seconds": time.perf_counter() - started,
            "output_file": target.name,
            "ffmpeg": ffmpeg.to_dict() if ffmpeg is not None else {"source": "injected"},
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


def publish_latent_chain_direct(
    target: str | Path,
    components,
    chain,
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
    ffmpeg_path: str | Path | None = None,
) -> H3DirectPublicationResult:
    """Decode an overlapping latent chain, trim joins, and atomically publish one A/V file."""
    chain.validate_complete()
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51.")
    if max_av_drift_seconds < 0:
        raise ValueError("Maximum audio-video drift must be zero or positive.")
    timeline = chain.timeline
    windows = chain.windows
    first = windows[0]
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
    ffmpeg = None
    started = time.perf_counter()
    target_frames = timeline.published_frames
    video_reports = []
    audio_reports = []

    try:
        if encoder_factory is RawVideoEncoder or muxer is _mux_audio:
            ffmpeg = resolve_ffmpeg(ffmpeg_path)
        if check_interrupted is not None:
            check_interrupted()
        encoder_args = (
            silent_video,
            first.width,
            first.height,
            float(first.fps),
            crf,
        )
        encoder = (
            encoder_factory(*encoder_args, ffmpeg)
            if encoder_factory is RawVideoEncoder
            else encoder_factory(*encoder_args)
        )

        assembler = _ChainedVideoAssembler(
            encoder,
            target_frames,
            timeline.context_frames,
        )
        for index, latents in enumerate(windows):
            assembler.begin_window(index)

            report = video_cache.decode_stream(
                H3VideoVAESpec.from_components(components),
                latents,
                assembler.write,
                unload_after=index == len(windows) - 1,
                check_interrupted=check_interrupted,
                prepare_stage=prepare_video_stage if index == 0 else None,
            )
            published = assembler.end_window()
            video_reports.append(
                {
                    "window": index + 1,
                    "decoded_frames": report.num_frames,
                    "overlap_frames_reconciled": timeline.context_frames if index else 0,
                    "published_frames_before_final_flush": published,
                    "decode_seconds": report.decode_seconds,
                    "peak_rgb8_chunk_bytes": report.peak_rgb8_chunk_bytes,
                }
            )
        assembler.finish()
        if assembler.remaining:
            raise ValueError(f"H3 chain ended {assembler.remaining} frames before its target.")
        encoder.close()
        if encoder.frames != target_frames:
            raise ValueError(
                f"Direct chain encoder received {encoder.frames} frames; expected {target_frames}."
            )

        audio_parts = []
        overlap_samples = round(timeline.context_frames / 24 * 32000)
        for index, latents in enumerate(windows):
            report = audio_cache.decode(
                H3AudioVAESpec.from_components(components),
                latents,
                unload_after=index == len(windows) - 1,
                check_interrupted=check_interrupted,
                prepare_stage=prepare_audio_stage if index == 0 else None,
            )
            audio_parts.append(report.waveform)
            audio_reports.append(
                {
                    "window": index + 1,
                    "decoded_samples": report.num_samples,
                    "overlap_samples_reconciled": overlap_samples if index else 0,
                    "decode_seconds": report.decode_seconds,
                }
            )
        waveform, audio_join_reports = _splice_audio_windows(
            audio_parts,
            overlap_samples,
            assembler.join_reports,
            timeline.context_frames,
        )
        target_samples = round(target_frames / 24 * 32000)
        if waveform.shape[1] < target_samples:
            waveform = np.pad(waveform, ((0, 0), (0, target_samples - waveform.shape[1])))
            audio_adjustment = "zero_padded"
        elif waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
            audio_adjustment = "truncated"
        else:
            audio_adjustment = "none"
        measured = _timing(
            target_frames,
            first.width,
            first.height,
            24.0,
            waveform,
            32000,
            max_av_drift_seconds,
        )
        save_wav(audio_path, waveform, 32000)
        if check_interrupted is not None:
            check_interrupted()
        if muxer is _mux_audio:
            muxer(silent_video, audio_path, partial_video, ffmpeg)
        else:
            muxer(silent_video, audio_path, partial_video)

        metadata = {
            **supplied,
            "format": "mp4",
            "video_codec": "h264",
            "audio_codec": "aac",
            "publication_mode": "direct_mlx_chained_stream",
            "crf": crf,
            "timeline": timeline.metadata(),
            "video_windows": video_reports,
            "video_joins": assembler.join_reports,
            "audio_windows": audio_reports,
            "audio_joins": audio_join_reports,
            "join_policy": "motion_matched_video_and_50ms_cosine_audio",
            "mlx_process_peak_bytes": _mlx_process_peak_bytes(),
            "audio_adjustment": audio_adjustment,
            "publish_seconds": time.perf_counter() - started,
            "output_file": target.name,
            "ffmpeg": ffmpeg.to_dict() if ffmpeg is not None else {"source": "injected"},
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
