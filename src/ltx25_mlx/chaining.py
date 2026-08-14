"""Latent-native timeline planning and assembly for LTX 2.5 continuation."""

from __future__ import annotations

import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

LTX25_CHAIN_CONTINUATION_STRENGTH = 0.5
LTX25_CHAIN_VIDEO_BLEND_FRAMES = 4
LTX25_CHAIN_AUDIO_CROSSFADE_SECONDS = 0.05


@dataclass(frozen=True)
class LTX25ChainPlan:
    """An exact, temporally aligned LTX 2.5 chained timeline."""

    total_frames: int
    window_count: int
    window_frames: int
    overlap_frames: int
    video_overlap_latent_frames: int
    window_audio_tokens: int
    join_audio_tokens: tuple[int, ...]
    frame_rate: float

    @property
    def expected_audio_tokens(self) -> int:
        return round(self.total_frames / self.frame_rate * 25.0)

    @property
    def window_start_frames(self) -> tuple[int, ...]:
        stride = self.window_frames - self.overlap_frames
        return tuple(index * stride for index in range(self.window_count))

    @property
    def assembled_video_seam_frames(self) -> tuple[int, ...]:
        """Frame indices where the assembled output first switches windows."""
        stride = self.window_frames - self.overlap_frames
        return tuple(self.window_frames + index * stride for index in range(self.window_count - 1))

    @property
    def assembled_audio_seam_tokens(self) -> tuple[int, ...]:
        """Audio-token indices where the assembled output first switches windows."""
        seams = []
        total = self.window_audio_tokens
        for trim in self.join_audio_tokens:
            seams.append(total)
            total += self.window_audio_tokens - trim
        return tuple(seams)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_frames": self.total_frames,
            "window_count": self.window_count,
            "window_frames": self.window_frames,
            "overlap_frames": self.overlap_frames,
            "video_overlap_latent_frames": self.video_overlap_latent_frames,
            "window_audio_tokens": self.window_audio_tokens,
            "join_audio_tokens": list(self.join_audio_tokens),
            "window_start_frames": list(self.window_start_frames),
            "assembled_video_seam_frames": list(self.assembled_video_seam_frames),
            "assembled_audio_seam_tokens": list(self.assembled_audio_seam_tokens),
            "frame_rate": self.frame_rate,
            "delivered_duration_seconds": (self.total_frames - 1) / self.frame_rate,
        }


@dataclass(frozen=True)
class LTX25LatentContinuation:
    """Small synchronized latent tails carried from one LTX window to the next."""

    stage1_video_tokens: mx.array
    stage2_video_tokens: mx.array
    audio_tokens: mx.array
    video_latent_frames: int
    audio_token_count: int


class LatentGuideConditioning:
    """Append prior-window tokens as timeline-aligned continuation guides.

    The noisy target remains independent. Guide tokens reuse the positions of
    the target's leading overlap and are cropped from the denoised output. This
    matches LTX's native extension topology and avoids forcing clean history
    directly into the new window's causal prefix.
    """

    def __init__(self, clean_latent: mx.array, *, strength: float = 1.0) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("LTX 2.5 continuation strength must be between 0 and 1.")
        if clean_latent.ndim != 3:
            raise ValueError("LTX 2.5 continuation tokens must have shape (B, N, C).")
        self.clean_latent = clean_latent
        self.strength = float(strength)

    def apply(self, state: Any, _spatial_dims: tuple[int, int, int]):
        from ltx_core_mlx.conditioning.types.latent_cond import LatentState

        count = int(self.clean_latent.shape[1])
        if self.clean_latent.shape[0] != state.latent.shape[0]:
            raise ValueError("LTX 2.5 continuation batch size does not match the target state.")
        if self.clean_latent.shape[2] != state.latent.shape[2]:
            raise ValueError("LTX 2.5 continuation channel count does not match the target state.")
        target_count = int(state.latent.shape[1])
        if count > target_count:
            raise ValueError("LTX 2.5 continuation is longer than the target latent timeline.")
        clean = self.clean_latent.astype(state.latent.dtype)
        mask = mx.full(
            (state.denoise_mask.shape[0], count, 1),
            1.0 - self.strength,
            dtype=state.denoise_mask.dtype,
        )
        positions = state.positions
        if positions is not None:
            positions = mx.concatenate([positions, positions[:, :count, :]], axis=1)
        return LatentState(
            latent=mx.concatenate([state.latent, clean], axis=1),
            clean_latent=mx.concatenate([state.clean_latent, clean], axis=1),
            denoise_mask=mx.concatenate([state.denoise_mask, mask], axis=1),
            positions=positions,
            attention_mask=state.attention_mask,
        )


def plan_ltx25_chain(
    *,
    total_frames: int,
    window_count: int,
    overlap_frames: int,
    frame_rate: float,
) -> LTX25ChainPlan:
    """Resolve an exact equal-window chain on the LTX temporal grid."""
    if window_count < 2 or window_count > 4:
        raise ValueError("LTX 2.5 chained timelines support two to four windows.")
    if frame_rate <= 0:
        raise ValueError("LTX 2.5 chained timeline frame rate must be positive.")
    if total_frames < 1 or (total_frames - 1) % 8:
        raise ValueError("LTX 2.5 total frames must equal 8n+1.")
    if overlap_frames < 1 or (overlap_frames - 1) % 8:
        raise ValueError("LTX 2.5 overlap frames must equal 8n+1.")
    numerator = total_frames + (window_count - 1) * overlap_frames
    if numerator % window_count:
        raise ValueError(
            "The selected total, window count, and overlap do not produce equal integer windows."
        )
    window_frames = numerator // window_count
    if window_frames <= overlap_frames:
        raise ValueError("LTX 2.5 chained windows must be longer than their overlap.")
    if (window_frames - 1) % 8:
        raise ValueError("Resolved LTX 2.5 window frames must equal 8n+1.")

    window_audio_tokens = round(window_frames / frame_rate * 25.0)
    stride = window_frames - overlap_frames
    previous_total_audio = window_audio_tokens
    join_audio_tokens: list[int] = []
    for index in range(1, window_count):
        cumulative_frames = window_frames + index * stride
        cumulative_audio = round(cumulative_frames / frame_rate * 25.0)
        new_audio = cumulative_audio - previous_total_audio
        trim = window_audio_tokens - new_audio
        if trim <= 0 or trim >= window_audio_tokens:
            raise ValueError("Resolved LTX 2.5 audio overlap is invalid.")
        join_audio_tokens.append(trim)
        previous_total_audio = cumulative_audio

    return LTX25ChainPlan(
        total_frames=total_frames,
        window_count=window_count,
        window_frames=window_frames,
        overlap_frames=overlap_frames,
        video_overlap_latent_frames=(overlap_frames - 1) // 8 + 1,
        window_audio_tokens=window_audio_tokens,
        join_audio_tokens=tuple(join_audio_tokens),
        frame_rate=float(frame_rate),
    )


def assemble_ltx25_latents(
    video_windows: list[mx.array],
    audio_windows: list[mx.array],
    plan: LTX25ChainPlan,
) -> tuple[mx.array, mx.array]:
    """Assemble synchronized windows using LTX's causal latent transition.

    Later video windows reinterpret their first latent as a causal one-frame
    token. Drop it, then blend the remaining overlap in latent space. Audio is
    kept in the joint transformer timeline and concatenated before one decode.
    """
    if len(video_windows) != plan.window_count or len(audio_windows) != plan.window_count:
        raise ValueError("LTX 2.5 chain window count does not match its plan.")
    video = video_windows[0]
    audio_parts = [audio_windows[0]]
    blend_count = plan.video_overlap_latent_frames - 1
    if blend_count < 1:
        raise ValueError("LTX 2.5 continuation requires at least two overlap latents.")
    for index in range(1, plan.window_count):
        following = video_windows[index]
        audio = audio_windows[index]
        if following.shape[2] <= plan.video_overlap_latent_frames:
            raise ValueError("LTX 2.5 video window is not longer than its latent overlap.")
        audio_trim = plan.join_audio_tokens[index - 1]
        if audio.shape[2] <= audio_trim:
            raise ValueError("LTX 2.5 audio window is not longer than its overlap.")
        following = following[:, :, 1:, :, :]
        alpha = mx.linspace(0.0, 1.0, blend_count).astype(video.dtype)
        alpha = alpha.reshape(1, 1, blend_count, 1, 1)
        blended = (
            video[:, :, -blend_count:, :, :] * (1.0 - alpha)
            + following[:, :, :blend_count, :, :] * alpha
        )
        video = mx.concatenate(
            [
                video[:, :, :-blend_count, :, :],
                blended,
                following[:, :, blend_count:, :, :],
            ],
            axis=2,
        )
        audio_parts.append(audio[:, :, audio_trim:, :])

    audio = mx.concatenate(audio_parts, axis=2)
    expected_video_latent_frames = (plan.total_frames - 1) // 8 + 1
    if video.shape[2] != expected_video_latent_frames:
        raise RuntimeError(
            "LTX 2.5 assembled video latent length is incorrect: "
            f"{video.shape[2]} != {expected_video_latent_frames}."
        )
    if audio.shape[2] != plan.expected_audio_tokens:
        raise RuntimeError(
            "LTX 2.5 assembled audio latent length is incorrect: "
            f"{audio.shape[2]} != {plan.expected_audio_tokens}."
        )
    return video, audio


def _cosine_ramp(length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("An LTX 2.5 join transition requires at least two samples.")
    position = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(np.pi * position)


def motion_matched_overlap(
    previous: np.ndarray,
    following: np.ndarray,
    *,
    blend_frames: int = LTX25_CHAIN_VIDEO_BLEND_FRAMES,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Select and blend the least disruptive seam inside two decoded overlaps."""
    previous = np.asarray(previous)
    following = np.asarray(following)
    if previous.shape != following.shape or previous.ndim != 4:
        raise ValueError("Aligned LTX 2.5 video overlaps must have matching frame layouts.")
    count = int(previous.shape[0])
    if count < 2:
        raise ValueError("An LTX 2.5 video overlap requires at least two frames.")
    if previous.dtype != np.uint8 or following.dtype != np.uint8:
        raise ValueError("LTX 2.5 decoded overlap frames must use uint8 RGB.")

    prior_view = previous[:, ::4, ::4].astype(np.int16, copy=False)
    next_view = following[:, ::4, ::4].astype(np.int16, copy=False)
    scores = np.asarray(
        [np.mean(np.abs(prior_view[index - 1] - next_view[index])) for index in range(1, count)],
        dtype=np.float64,
    )
    seam = int(np.argmin(scores)) + 1
    blend_frames = min(max(int(blend_frames), 2), count)
    start = max(0, min(seam - blend_frames // 2, count - blend_frames))
    stop = start + blend_frames
    ramp = _cosine_ramp(blend_frames)

    joined = previous.copy()
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


class DecodedChainAssembler:
    """Stream decoded windows while retaining only the overlap needed for repair."""

    def __init__(self, writer, target_frames: int, overlap_frames: int) -> None:
        self.writer = writer
        self.remaining = int(target_frames)
        self.overlap_frames = int(overlap_frames)
        self.previous_overlap: np.ndarray | None = None
        self.current_tail: np.ndarray | None = None
        self.head_parts: list[np.ndarray] = []
        self.head_frames = 0
        self.window_index = -1
        self.emitted_frames = 0
        self.join_reports: list[dict[str, float | int]] = []

    def begin_window(self, index: int) -> None:
        self.window_index = int(index)
        self.current_tail = None
        self.head_parts = []
        self.head_frames = 0

    def _emit(self, frames: np.ndarray) -> None:
        if self.remaining <= 0 or not frames.size:
            return
        frames = np.ascontiguousarray(frames[: self.remaining])
        self.writer.write(frames)
        written = int(frames.shape[0])
        self.emitted_frames += written
        self.remaining -= written

    def _retain_tail(self, frames: np.ndarray) -> None:
        if not frames.size:
            return
        combined = (
            np.ascontiguousarray(frames)
            if self.current_tail is None
            else np.concatenate((self.current_tail, frames), axis=0)
        )
        if combined.shape[0] > self.overlap_frames:
            split = int(combined.shape[0]) - self.overlap_frames
            self._emit(combined[:split])
            combined = combined[split:]
        self.current_tail = np.ascontiguousarray(combined)

    def write(self, frames: np.ndarray) -> None:
        frames = np.asarray(frames)
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("LTX 2.5 decoded chunks must use uint8 RGB frame layout.")
        if self.window_index == 0:
            self._retain_tail(frames)
            return
        if self.previous_overlap is None:
            raise RuntimeError("An LTX 2.5 chain is missing its preceding decoded overlap.")
        if self.head_frames < self.overlap_frames:
            take = min(self.overlap_frames - self.head_frames, int(frames.shape[0]))
            if take:
                self.head_parts.append(np.ascontiguousarray(frames[:take]))
                self.head_frames += take
                frames = frames[take:]
            if self.head_frames == self.overlap_frames:
                following = np.concatenate(self.head_parts, axis=0)
                self.head_parts.clear()
                joined, report = motion_matched_overlap(self.previous_overlap, following)
                overlap_start = self.emitted_frames
                self.join_reports.append(
                    {
                        "join": self.window_index,
                        "assembled_overlap_start_frame": overlap_start,
                        "assembled_seam_frame": overlap_start
                        + int(report["seam_frame_in_overlap"]),
                        **report,
                    }
                )
                self._emit(joined)
        self._retain_tail(frames)

    def end_window(self) -> None:
        if self.window_index > 0 and self.head_frames != self.overlap_frames:
            raise ValueError("An LTX 2.5 window ended before its overlap was decoded.")
        if self.current_tail is None or self.current_tail.shape[0] != self.overlap_frames:
            raise ValueError("An LTX 2.5 window is too short for its configured overlap.")
        self.previous_overlap = self.current_tail

    def finish(self) -> None:
        if self.previous_overlap is None:
            raise ValueError("The LTX 2.5 chain did not decode any windows.")
        self._emit(self.previous_overlap)
        if self.remaining:
            raise ValueError(f"The LTX 2.5 chain is {self.remaining} frames short.")


def splice_audio_windows(
    waveforms: list[np.ndarray],
    *,
    overlap_samples: int,
    video_joins: list[dict[str, float | int]],
    context_frames: int,
    crossfade_samples: int,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Join decoded audio at each selected visual seam with a short cosine transition."""
    if not waveforms:
        raise ValueError("An LTX 2.5 audio chain requires at least one waveform.")
    merged = np.asarray(waveforms[0], dtype=np.float32)
    reports: list[dict[str, float | int]] = []
    for index, following in enumerate(waveforms[1:]):
        following = np.asarray(following, dtype=np.float32)
        overlap = min(int(overlap_samples), merged.shape[1], following.shape[1])
        if overlap < 2:
            raise ValueError("An LTX 2.5 audio overlap requires at least two samples.")
        visual_seam = int(video_joins[index]["seam_frame_in_overlap"])
        seam = round(visual_seam / context_frames * overlap)
        fade = min(max(int(crossfade_samples), 2), overlap)
        start = max(0, min(seam - fade // 2, overlap - fade))
        stop = start + fade
        prior = merged[:, -overlap:]
        next_overlap = following[:, :overlap]
        assembled_overlap_start = int(merged.shape[1]) - overlap
        joined = prior.copy()
        ramp = _cosine_ramp(fade).reshape((1, -1))
        joined[:, start:stop] = (
            prior[:, start:stop] * (1.0 - ramp) + next_overlap[:, start:stop] * ramp
        )
        joined[:, stop:] = next_overlap[:, stop:]
        hard_delta = float(np.max(np.abs(next_overlap[:, seam] - prior[:, seam - 1])))
        merged = np.concatenate((merged[:, :-overlap], joined, following[:, overlap:]), axis=1)
        reports.append(
            {
                "join": index + 1,
                "seam_sample_in_overlap": seam,
                "assembled_overlap_start_sample": assembled_overlap_start,
                "assembled_seam_sample": assembled_overlap_start + seam,
                "crossfade_start_sample": start,
                "assembled_crossfade_start_sample": assembled_overlap_start + start,
                "crossfade_samples": fade,
                "hard_cut_peak_delta": hard_delta,
            }
        )
    return merged, reports


class _RawVideoEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: float, ffmpeg: str) -> None:
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
            "18",
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

    def write(self, frames: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("The LTX 2.5 video encoder has no input stream.")
        self._process.stdin.write(np.ascontiguousarray(frames).tobytes())
        self.frames += int(frames.shape[0])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        error = self._process.stderr.read() if self._process.stderr is not None else b""
        code = self._process.wait()
        if code:
            raise RuntimeError(f"LTX 2.5 video encode failed: {error.decode()[:500]}")

    def abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait()
        self._closed = True


def _save_waveform(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(waveform.T, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(int(waveform.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def mlx_audio_to_numpy(waveform: mx.array) -> np.ndarray:
    """Transfer decoded audio without exposing MLX bfloat16 to NumPy's buffer API."""
    host_ready = waveform.astype(mx.float32)
    mx.eval(host_ready)
    return np.asarray(host_ready)


def fit_audio_window(waveform: np.ndarray, target_samples: int) -> tuple[np.ndarray, str]:
    """Match one decoded window to the video clock without moving its leading audio."""
    samples = int(waveform.shape[1])
    if samples < target_samples:
        return (
            np.pad(waveform, ((0, 0), (0, target_samples - samples))),
            f"zero_padded_{target_samples - samples}",
        )
    if samples > target_samples:
        return waveform[:, :target_samples], f"truncated_{samples - target_samples}"
    return waveform, "none"


def publish_decoded_ltx25_chain(
    *,
    output_path: str,
    video_decoder_block,
    audio_decoder_block,
    video_windows: list[mx.array],
    audio_windows: list[mx.array],
    plan: LTX25ChainPlan,
    width: int,
    height: int,
    check_interrupted=None,
) -> dict[str, object]:
    """Decode, reconcile, and atomically publish an overlapping LTX 2.5 chain."""
    from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling
    from ltx_core_mlx.utils.ffmpeg import find_ffmpeg
    from ltx_core_mlx.utils.memory import aggressive_cleanup

    if len(video_windows) != plan.window_count or len(audio_windows) != plan.window_count:
        raise ValueError("LTX 2.5 publication window count does not match its plan.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    silent = target.with_name(f".{target.stem}.chain-video.partial.mp4")
    audio_path = target.with_name(f".{target.stem}.chain-audio.partial.wav")
    partial = target.with_name(f".{target.stem}.chain.partial.mp4")
    ffmpeg = find_ffmpeg()
    encoder = _RawVideoEncoder(silent, width, height, plan.frame_rate, ffmpeg)
    assembler = DecodedChainAssembler(encoder, plan.total_frames, plan.overlap_frames)
    audio_parts: list[np.ndarray] = []
    audio_window_adjustments: list[str] = []
    sample_rate = 48000
    try:
        decoder = video_decoder_block.load()
        for index, latent in enumerate(video_windows):
            if check_interrupted is not None:
                check_interrupted()
            assembler.begin_window(index)
            tiling = _compute_decode_tiling(latent.shape, frame_rate=plan.frame_rate)
            for chunk in decoder.tiled_decode(latent, tiling):
                for frame_index in range(int(chunk.shape[2])):
                    frame = mx.clip(chunk[0, :, frame_index], -1.0, 1.0)
                    frame = ((frame + 1.0) * 127.5).astype(mx.uint8).transpose(1, 2, 0)
                    mx.eval(frame)
                    assembler.write(np.asarray(frame)[None])
                del chunk
                aggressive_cleanup()
            assembler.end_window()
        assembler.finish()
        encoder.close()
        if encoder.frames != plan.total_frames:
            raise RuntimeError(
                f"LTX 2.5 chain encoded {encoder.frames} frames; expected {plan.total_frames}."
            )

        for latent in audio_windows:
            if check_interrupted is not None:
                check_interrupted()
            waveform = audio_decoder_block(latent)
            array = mlx_audio_to_numpy(waveform)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            elif array.ndim == 2 and array.shape[0] == 1:
                pass
            if array.ndim != 2:
                raise ValueError("LTX 2.5 decoded audio must have shape (channels, samples).")
            target_window_samples = round(plan.window_frames / plan.frame_rate * sample_rate)
            array, adjustment = fit_audio_window(array, target_window_samples)
            audio_parts.append(np.ascontiguousarray(array))
            audio_window_adjustments.append(adjustment)
            del waveform
            aggressive_cleanup()

        overlap_samples = round(plan.overlap_frames / plan.frame_rate * sample_rate)
        waveform, audio_reports = splice_audio_windows(
            audio_parts,
            overlap_samples=overlap_samples,
            video_joins=assembler.join_reports,
            context_frames=plan.overlap_frames,
            crossfade_samples=round(LTX25_CHAIN_AUDIO_CROSSFADE_SECONDS * sample_rate),
        )
        target_samples = round(plan.total_frames / plan.frame_rate * sample_rate)
        if waveform.shape[1] < target_samples:
            waveform = np.pad(waveform, ((0, 0), (0, target_samples - waveform.shape[1])))
            audio_adjustment = "zero_padded"
        elif waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
            audio_adjustment = "truncated"
        else:
            audio_adjustment = "none"
        _save_waveform(audio_path, waveform, sample_rate)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(silent),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(partial),
        ]
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"LTX 2.5 audio mux failed: {completed.stderr.decode()[:500]}")
        os.replace(partial, target)
        return {
            "video_joins": assembler.join_reports,
            "audio_joins": audio_reports,
            "join_policy": "motion_matched_video_and_50ms_cosine_audio",
            "audio_window_adjustments": audio_window_adjustments,
            "audio_adjustment": audio_adjustment,
            "output_frames": plan.total_frames,
            "output_audio_samples": target_samples,
        }
    except BaseException:
        encoder.abort()
        raise
    finally:
        for path in (silent, audio_path, partial):
            path.unlink(missing_ok=True)
        aggressive_cleanup()


__all__ = [
    "LTX25ChainPlan",
    "LTX25LatentContinuation",
    "LTX25_CHAIN_CONTINUATION_STRENGTH",
    "LatentGuideConditioning",
    "DecodedChainAssembler",
    "assemble_ltx25_latents",
    "fit_audio_window",
    "mlx_audio_to_numpy",
    "motion_matched_overlap",
    "plan_ltx25_chain",
    "publish_decoded_ltx25_chain",
    "splice_audio_windows",
]
