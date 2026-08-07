#!/usr/bin/env python3
"""Decode and publish a synchronized comparison from two saved H3 latent trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from minimax_h3_mlx.load import load_compact_audio_vae, load_compact_video_vae
from minimax_h3_mlx.media import save_mp4, save_wav
from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD


def _metrics(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    left = np.asarray(baseline, dtype=np.float32)
    right = np.asarray(candidate, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"comparison shapes must match: {left.shape} != {right.shape}")
    delta = right - left
    left_norm = float(np.linalg.norm(left.reshape(-1)))
    right_norm = float(np.linalg.norm(right.reshape(-1)))
    rmse = math.sqrt(float(np.mean(delta * delta)))
    peak = 255.0 if np.issubdtype(np.asarray(baseline).dtype, np.integer) else 2.0
    return {
        "relative_l2_error": float(np.linalg.norm(delta.reshape(-1)) / max(left_norm, 1e-12)),
        "cosine_similarity": float(
            np.clip(
                np.dot(left.reshape(-1), right.reshape(-1)) / max(left_norm * right_norm, 1e-12),
                -1.0,
                1.0,
            )
        ),
        "mean_absolute_error": float(np.mean(np.abs(delta))),
        "max_absolute_error": float(np.max(np.abs(delta))),
        "rmse": rmse,
        "psnr_db": float(20.0 * math.log10(peak / max(rmse, 1e-12))),
    }


def _decode_video(model, normalized: mx.array, num_frames: int) -> np.ndarray:
    config = model.config
    mean = mx.array(np.asarray(config.latents_mean, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
    std = mx.array(np.asarray(config.latents_std, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
    decoded = np.asarray(model.decode((normalized * std + mean).astype(mx.float32)))
    pixel_mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    frames = np.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
    frames = frames[0, :, :num_frames].transpose(1, 2, 3, 0)
    return np.ascontiguousarray(frames * 255.0 + 0.5, dtype=np.uint8)


def _decode_audio(model, normalized: mx.array) -> np.ndarray:
    config = model.config
    mean = mx.array(np.asarray(config.latents_mean, dtype=np.float32)).reshape(1, -1, 1)
    std = mx.array(np.asarray(config.latents_std, dtype=np.float32)).reshape(1, -1, 1)
    decoded = np.asarray(
        model.decode((normalized * std + mean).astype(mx.float32)), dtype=np.float32
    )
    if decoded.ndim != 3 or decoded.shape[:2] != (2, 1):
        raise ValueError(f"audio VAE output must have shape (2, 1, samples), got {decoded.shape}")
    return np.ascontiguousarray(decoded[:, 0], dtype=np.float32)


def _labeled(frames: np.ndarray, label: str) -> np.ndarray:
    font = ImageFont.load_default(size=20)
    rendered = []
    for frame in frames:
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        bounds = draw.textbbox((0, 0), label, font=font)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        draw.rectangle((8, 8, 24 + width, 20 + height), fill=(0, 0, 0))
        draw.text((16, 10), label, font=font, fill=(255, 255, 255))
        rendered.append(np.asarray(image))
    return np.ascontiguousarray(rendered, dtype=np.uint8)


def _contact_sheet(
    baseline: np.ndarray,
    candidate: np.ndarray,
    output: Path,
    baseline_label: str,
    candidate_label: str,
) -> Path:
    indices = np.linspace(0, len(baseline) - 1, 5, dtype=int)
    top = _labeled(baseline[indices], baseline_label)
    bottom = _labeled(candidate[indices], candidate_label)
    rows = [np.concatenate(list(top), axis=1), np.concatenate(list(bottom), axis=1)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(output)
    return output


def _load_latents(path: Path) -> dict[str, mx.array]:
    values = mx.load(str(path))
    expected = {"video_latents", "audio_latents"}
    if set(values) != expected:
        raise ValueError(f"latent file must contain {sorted(expected)}, got {sorted(values)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video-decode-batch", type=int, default=8)
    parser.add_argument("--baseline-label", default="BF16 baseline")
    parser.add_argument("--candidate-label", default="Q8 blocks 38-49")
    args = parser.parse_args()
    for path in (args.baseline, args.candidate, args.video_vae, args.audio_vae):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.frames, args.fps, args.video_decode_batch) < 1:
        raise ValueError("frames, fps, and video decode batch must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"comparison output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_latents = _load_latents(args.baseline)
    candidate_latents = _load_latents(args.candidate)
    for key in baseline_latents:
        if baseline_latents[key].shape != candidate_latents[key].shape:
            raise ValueError(f"{key} shape mismatch")

    timings: dict[str, float] = {}
    peaks: dict[str, int] = {}
    video_model = load_compact_video_vae(args.video_vae)
    video_model.decode_batch = args.video_decode_batch
    decoded_video = []
    for label, values in (
        ("baseline", baseline_latents),
        ("candidate", candidate_latents),
    ):
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        decoded_video.append(_decode_video(video_model, values["video_latents"], args.frames))
        timings[f"{label}_video_decode_seconds"] = time.perf_counter() - started
        peaks[f"{label}_video_decode_peak_bytes"] = int(mx.get_peak_memory())
    video_model = None
    gc.collect()
    mx.clear_cache()

    audio_model = load_compact_audio_vae(args.audio_vae)
    decoded_audio = []
    for label, values in (
        ("baseline", baseline_latents),
        ("candidate", candidate_latents),
    ):
        mx.clear_cache()
        mx.reset_peak_memory()
        started = time.perf_counter()
        decoded_audio.append(_decode_audio(audio_model, values["audio_latents"]))
        timings[f"{label}_audio_decode_seconds"] = time.perf_counter() - started
        peaks[f"{label}_audio_decode_peak_bytes"] = int(mx.get_peak_memory())
    audio_model = None
    baseline_latents = candidate_latents = None
    gc.collect()
    mx.clear_cache()

    baseline_video, candidate_video = decoded_video
    baseline_audio, candidate_audio = decoded_audio
    if (
        baseline_video.shape != candidate_video.shape
        or baseline_audio.shape != candidate_audio.shape
    ):
        raise ValueError("decoded comparison shapes do not match")

    baseline_mp4 = save_mp4(
        args.output / "bf16_baseline.mp4", baseline_video, args.fps, baseline_audio, 32000
    )
    candidate_mp4 = save_mp4(
        args.output / "q8_blocks_38_49.mp4", candidate_video, args.fps, candidate_audio, 32000
    )
    comparison = np.concatenate(
        [
            _labeled(baseline_video, args.baseline_label),
            _labeled(candidate_video, args.candidate_label),
        ],
        axis=2,
    )
    comparison_mp4 = save_mp4(args.output / "side_by_side_silent.mp4", comparison, args.fps)
    contact_sheet = _contact_sheet(
        baseline_video,
        candidate_video,
        args.output / "contact_sheet.png",
        args.baseline_label,
        args.candidate_label,
    )
    silence = np.zeros((2, 16000), dtype=np.float32)
    audio_ab = save_wav(
        args.output / "audio_ab_bf16_then_q8.wav",
        np.concatenate([baseline_audio, silence, candidate_audio], axis=1),
        32000,
    )
    payload = {
        "geometry": {
            "frames": int(baseline_video.shape[0]),
            "height": int(baseline_video.shape[1]),
            "width": int(baseline_video.shape[2]),
            "fps": args.fps,
            "audio_samples": int(baseline_audio.shape[1]),
            "sample_rate": 32000,
        },
        "decoded_metrics": {
            "video_uint8": _metrics(baseline_video, candidate_video),
            "audio_float32": _metrics(baseline_audio, candidate_audio),
        },
        "timings": timings,
        "peaks": peaks,
        "audio_ab_segments": {
            "bf16_seconds": [0.0, baseline_audio.shape[1] / 32000],
            "silence_seconds": [
                baseline_audio.shape[1] / 32000,
                baseline_audio.shape[1] / 32000 + 0.5,
            ],
            "q8_seconds": [
                baseline_audio.shape[1] / 32000 + 0.5,
                baseline_audio.shape[1] / 32000 + 0.5 + candidate_audio.shape[1] / 32000,
            ],
        },
        "artifacts": {
            "bf16": baseline_mp4.name,
            "q8": candidate_mp4.name,
            "side_by_side": comparison_mp4.name,
            "contact_sheet": contact_sheet.name,
            "audio_ab": audio_ab.name,
        },
    }
    (args.output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
