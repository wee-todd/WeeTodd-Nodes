#!/usr/bin/env python3
"""Verify, compare, and publish the FastH3 audiovisual acceptance matrix."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

VARIANTS = (
    "grouped_vsa",
    "indexed_metal",
    "indexed_metal_fused_qkv",
    "indexed_metal_layer40",
    "dense_control",
    "dense_token_pair",
)
LABELS = {
    "grouped_vsa": "Grouped VSA control",
    "indexed_metal": "Indexed Metal",
    "indexed_metal_fused_qkv": "Indexed + fused QKV",
    "indexed_metal_layer40": "Indexed + 40 layers",
    "dense_control": "Dense control",
    "dense_token_pair": "Dense token pairing",
}
COMPARISONS = (
    ("grouped_vsa", "indexed_metal"),
    ("grouped_vsa", "indexed_metal_fused_qkv"),
    ("indexed_metal", "indexed_metal_fused_qkv"),
    ("grouped_vsa", "indexed_metal_layer40"),
    ("dense_control", "dense_token_pair"),
)


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=True, capture_output=True)


def _decode_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No video frames decoded from {path}")
    return np.stack(frames)


def _decode_audio(path: Path, ffmpeg: Path) -> np.ndarray:
    result = _run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            "32000",
            "pipe:1",
        ]
    )
    values = np.frombuffer(result.stdout, dtype=np.float32)
    if values.size == 0 or values.size % 2:
        raise ValueError(f"Invalid decoded stereo audio from {path}")
    return values.reshape(-1, 2).T


def _array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Comparison shapes differ: {reference.shape} != {candidate.shape}")
    left = reference.astype(np.float64, copy=False).reshape(-1)
    right = candidate.astype(np.float64, copy=False).reshape(-1)
    difference = right - left
    reference_norm = np.linalg.norm(left)
    candidate_norm = np.linalg.norm(right)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "values": int(left.size),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(np.linalg.norm(difference) / max(reference_norm, 1e-12)),
        "cosine_similarity": float(
            np.dot(left, right) / max(reference_norm * candidate_norm, 1e-12)
        ),
    }


def _video_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    metrics = _array_metrics(reference, candidate)
    metrics["frames"] = int(reference.shape[0])
    metrics["psnr_db"] = (
        float("inf") if metrics["rmse"] == 0 else 20 * math.log10(255.0 / metrics["rmse"])
    )
    return metrics


def _video_health(video: np.ndarray) -> dict[str, float | int]:
    temporal = np.mean(
        np.abs(video[1:].astype(np.float32) - video[:-1].astype(np.float32)),
        axis=(1, 2, 3),
    )
    return {
        "frames": int(video.shape[0]),
        "mean_rgb": float(np.mean(video)),
        "mean_temporal_absolute_difference": float(np.mean(temporal)),
        "minimum_temporal_absolute_difference": float(np.min(temporal)),
        "maximum_temporal_absolute_difference": float(np.max(temporal)),
        "near_frozen_frame_pair_fraction": float(np.mean(temporal < 0.5)),
        "black_pixel_fraction": float(np.mean(video <= 1)),
        "white_pixel_fraction": float(np.mean(video >= 254)),
    }


def _audio_health(audio: np.ndarray, sample_rate: int = 32000) -> dict[str, float | int]:
    mono = np.mean(audio.astype(np.float64), axis=0)
    window = 1024
    hop = 512
    frames = np.stack(
        [mono[start : start + window] for start in range(0, len(mono) - window + 1, hop)]
    )
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    window_values = np.hanning(window)
    spectrum = np.abs(np.fft.rfft(frames * window_values[None, :], axis=1))
    frequencies = np.fft.rfftfreq(window, 1.0 / sample_rate)
    centroid = np.sum(spectrum * frequencies[None, :], axis=1) / np.maximum(
        np.sum(spectrum, axis=1),
        1e-12,
    )
    energy_change = np.diff(frame_rms, prepend=frame_rms[0])
    transient_threshold = float(np.mean(energy_change) + 2.0 * np.std(energy_change))
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {
        "channels": int(audio.shape[0]),
        "samples": int(audio.shape[1]),
        "duration_seconds": float(audio.shape[1] / sample_rate),
        "rms": rms,
        "rms_dbfs": float(20.0 * math.log10(max(rms, 1e-12))),
        "peak_absolute": float(np.max(np.abs(audio))),
        "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
        "silent_window_fraction": float(np.mean(frame_rms < 1e-4)),
        "mean_spectral_centroid_hz": float(np.mean(centroid)),
        "transient_window_count": int(np.sum(energy_change > transient_threshold)),
    }


def _latent_arrays(path: Path) -> dict[str, np.ndarray]:
    values = mx.load(str(path))
    return {
        name: np.asarray(value.astype(mx.float32))
        for name, value in values.items()
    }


def _make_contact_sheet(videos: dict[str, np.ndarray], target: Path) -> None:
    frame_indices = (0, 30, 60, 90, 123)
    cell_width, cell_height = 320, 192
    label_height = 34
    sheet = np.full(
        (
            len(VARIANTS) * (cell_height + label_height),
            len(frame_indices) * cell_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    for row, name in enumerate(VARIANTS):
        top = row * (cell_height + label_height)
        cv2.putText(
            sheet,
            LABELS[name],
            (10, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for column, frame_index in enumerate(frame_indices):
            rgb = videos[name][frame_index]
            resized = cv2.resize(rgb, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            left = column * cell_width
            sheet[
                top + label_height : top + label_height + cell_height,
                left : left + cell_width,
            ] = resized
            cv2.putText(
                sheet,
                f"{frame_index / 24:.2f}s",
                (left + 8, top + label_height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(target), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])


def _make_audio_spectrograms(audio: dict[str, np.ndarray], target: Path) -> None:
    width, plot_height, label_height = 1200, 150, 34
    sheet = np.full(
        (len(VARIANTS) * (plot_height + label_height), width, 3),
        245,
        dtype=np.uint8,
    )
    window_size, hop = 2048, 512
    window = np.hanning(window_size)
    for row, name in enumerate(VARIANTS):
        mono = np.mean(audio[name].astype(np.float64), axis=0)
        frames = np.stack(
            [
                mono[start : start + window_size] * window
                for start in range(0, len(mono) - window_size + 1, hop)
            ]
        )
        spectrum = np.abs(np.fft.rfft(frames, axis=1)).T
        decibels = 20.0 * np.log10(np.maximum(spectrum, 1e-8))
        peak = float(np.max(decibels))
        normalized = np.clip((decibels - (peak - 80.0)) / 80.0, 0.0, 1.0)
        image = np.flipud((normalized * 255.0).astype(np.uint8))
        image = cv2.resize(image, (width, plot_height), interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        top = row * (plot_height + label_height)
        cv2.putText(
            sheet,
            LABELS[name],
            (10, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        sheet[top + label_height : top + label_height + plot_height] = colored
    cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])


def _comparison_video(
    ffmpeg: Path,
    inputs: list[tuple[Path, str]],
    target: Path,
    columns: int,
) -> None:
    command = [str(ffmpeg), "-y", "-v", "error"]
    filters = []
    for index, (path, label) in enumerate(inputs):
        command.extend(["-i", str(path)])
        escaped = label.replace("'", "\\'")
        filters.append(
            f"[{index}:v]drawtext=text='{escaped}':x=12:y=12:fontsize=24:"
            "fontcolor=white:borderw=2:bordercolor=black[v" + str(index) + "]"
        )
    layout = []
    for index in range(len(inputs)):
        row, column = divmod(index, columns)
        layout.append(f"{column * 640}_{row * 384}")
    stacks = "".join(f"[v{index}]" for index in range(len(inputs)))
    filters.append(
        f"{stacks}xstack=inputs={len(inputs)}:layout={'|'.join(layout)}:fill=black[outv]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ]
    )
    _run(command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
    )
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--publish-prefix", default="FastH3-1-5")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    args = parser.parse_args()
    root = args.matrix.expanduser().resolve()
    publish_root = args.publish_root.expanduser().resolve()
    ffmpeg = args.ffmpeg.expanduser().resolve()

    matrix = json.loads((root / "matrix.json").read_text())
    videos = {
        name: _decode_video(root / name / f"{name}.mp4") for name in VARIANTS
    }
    audio = {
        name: _decode_audio(root / name / f"{name}.mp4", ffmpeg) for name in VARIANTS
    }
    latents = {
        name: _latent_arrays(root / name / f"{name}-latents.safetensors")
        for name in VARIANTS
    }

    comparisons = {}
    for reference, candidate in COMPARISONS:
        key = f"{reference}__vs__{candidate}"
        comparisons[key] = {
            "reference": reference,
            "candidate": candidate,
            "video_latents": _array_metrics(
                latents[reference]["video_latents"],
                latents[candidate]["video_latents"],
            ),
            "audio_latents": _array_metrics(
                latents[reference]["audio_latents"],
                latents[candidate]["audio_latents"],
            ),
            "decoded_video": _video_metrics(videos[reference], videos[candidate]),
            "decoded_audio": _array_metrics(audio[reference], audio[candidate]),
        }

    health = {
        name: {
            "video": _video_health(videos[name]),
            "audio": _audio_health(audio[name]),
        }
        for name in VARIANTS
    }

    timings = {}
    for name in VARIANTS:
        sampling = matrix[name]["sampling"]
        timings[name] = {
            "seconds_per_evaluation": sampling["seconds_per_evaluation"],
            "transformer_total_seconds": sampling["total_seconds"],
            "sample_wall_seconds": sampling["sample_wall_seconds"],
            "sample_peak_memory_bytes": sampling["sample_peak_memory_bytes"],
            "pages_loaded": sampling["paging"]["pages_loaded"],
            "pages_avoided": sampling["paging"]["pages_avoided"],
            "publish_seconds": matrix[name]["publication"]["publish_seconds"],
        }
    for reference, candidate in (
        ("grouped_vsa", "indexed_metal"),
        ("indexed_metal", "indexed_metal_fused_qkv"),
        ("indexed_metal", "indexed_metal_layer40"),
        ("dense_control", "dense_token_pair"),
    ):
        baseline = timings[reference]["seconds_per_evaluation"]
        candidate_time = timings[candidate]["seconds_per_evaluation"]
        timings[candidate]["speedup_vs_reference_percent"] = (
            (baseline - candidate_time) / baseline * 100.0
        )
        timings[candidate]["speedup_reference"] = reference

    prefix = args.publish_prefix
    contact_sheet = publish_root / f"{prefix}-Contact-Sheet.jpg"
    _make_contact_sheet(videos, contact_sheet)
    spectrograms = publish_root / f"{prefix}-Audio-Spectrograms.jpg"
    _make_audio_spectrograms(audio, spectrograms)
    vsa_compare = publish_root / f"{prefix}-VSA-Compare.mp4"
    _comparison_video(
        ffmpeg,
        [
            (root / name / f"{name}.mp4", LABELS[name])
            for name in VARIANTS[:4]
        ],
        vsa_compare,
        columns=2,
    )
    dense_compare = publish_root / f"{prefix}-Dense-Pair-Compare.mp4"
    _comparison_video(
        ffmpeg,
        [
            (root / name / f"{name}.mp4", LABELS[name])
            for name in VARIANTS[4:]
        ],
        dense_compare,
        columns=2,
    )
    metrics_path = publish_root / f"{prefix}-Metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "timings": timings,
                "comparisons": comparisons,
                "health": health,
                "scenario": json.loads((root / "scenario.json").read_text())
                if (root / "scenario.json").is_file()
                else None,
                "contact_sheet": str(contact_sheet),
                "audio_spectrograms": str(spectrograms),
                "vsa_comparison_video": str(vsa_compare),
                "dense_pair_comparison_video": str(dense_compare),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for name in VARIANTS:
        shutil.copy2(
            root / name / f"{name}.mp4",
            publish_root / f"{prefix}-{name}.mp4",
        )
        _run(
            [
                str(ffmpeg),
                "-y",
                "-v",
                "error",
                "-i",
                str(root / name / f"{name}.mp4"),
                "-map",
                "0:a:0",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(publish_root / f"{prefix}-{name}-audio.mp3"),
            ]
        )
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
