#!/usr/bin/env python3
"""Verify and publish the FastH3 resolution scaling matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

RESOLUTIONS = (
    (512, 256),
    (768, 448),
    (1024, 576),
    (1280, 704),
    (1536, 832),
    (1920, 1088),
)


def _key(width: int, height: int) -> str:
    return f"{width}x{height}"


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, check=True, capture_output=True)


def _probe(path: Path, ffprobe: Path) -> dict[str, object]:
    result = _run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError(f"Expected synchronized video and audio streams in {path}")
    rate_numerator, rate_denominator = video["avg_frame_rate"].split("/")
    fps = float(rate_numerator) / float(rate_denominator)
    frames = int(video.get("nb_frames") or round(float(video["duration"]) * fps))
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frames": frames,
        "fps": fps,
        "video_duration_seconds": float(video["duration"]),
        "video_codec": video["codec_name"],
        "audio_codec": audio["codec_name"],
        "audio_channels": int(audio["channels"]),
        "audio_sample_rate": int(audio["sample_rate"]),
        "audio_duration_seconds": float(audio["duration"]),
        "av_drift_seconds": abs(float(video["duration"]) - float(audio["duration"])),
        "file_bytes": path.stat().st_size,
    }


def _decode_audio_health(path: Path, ffmpeg: Path) -> dict[str, float | int]:
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
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0 or audio.size % 2 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Invalid decoded stereo audio in {path}")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return {
        "samples_per_channel": int(audio.size // 2),
        "rms": rms,
        "rms_dbfs": float(20 * math.log10(max(rms, 1e-12))),
        "peak_absolute": float(np.max(np.abs(audio))),
        "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }


def _read_contact_frames(path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode frame {index} from {path}")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _make_contact_sheet(matrix: dict[str, object], target: Path) -> None:
    frame_indices = (0, 53, 106)
    cell_width, cell_height = 384, 216
    label_height = 42
    sheet = np.full(
        (len(RESOLUTIONS) * (cell_height + label_height), len(frame_indices) * cell_width, 3),
        248,
        dtype=np.uint8,
    )
    for row, (width, height) in enumerate(RESOLUTIONS):
        name = _key(width, height)
        frames = _read_contact_frames(Path(matrix[name]["video"]), frame_indices)
        top = row * (cell_height + label_height)
        cv2.putText(
            sheet,
            f"{name}  ({width * height / 1_000_000:.2f} MP)",
            (12, top + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        for column, (frame_index, frame) in enumerate(zip(frame_indices, frames, strict=True)):
            left = column * cell_width
            sheet[
                top + label_height : top + label_height + cell_height,
                left : left + cell_width,
            ] = _fit(frame, cell_width, cell_height)
            cv2.putText(
                sheet,
                f"{frame_index / 24:.2f}s",
                (left + 10, top + label_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet,
                f"{frame_index / 24:.2f}s",
                (left + 10, top + label_height + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"Could not write contact sheet: {target}")


def _format_seconds(seconds: float) -> str:
    rounded = int(round(seconds))
    minutes, remainder = divmod(rounded, 60)
    return f"{minutes}m {remainder:02d}s" if minutes else f"{remainder}s"


def _make_chart(matrix: dict[str, object], target: Path) -> None:
    chart_width, chart_height = 1500, 940
    left, right, top, bottom = 250, 150, 160, 90
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom
    canvas = np.full((chart_height, chart_width, 3), 255, dtype=np.uint8)
    times = [
        float(matrix[_key(*resolution)]["complete_wall_seconds"])
        for resolution in RESOLUTIONS
    ]
    maximum = max(times) * 1.12
    tick = max(1, math.ceil(maximum / 5 / 60) * 60)
    axis_max = math.ceil(maximum / tick) * tick

    cv2.putText(
        canvas,
        "FastH3 ~35B Q8/BF16-gate MLX - identical seed and prompt",
        (left, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (95, 95, 95),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "What each resolution costs on a Mac Studio M3 Ultra",
        (left, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.96,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "4.0-second request aligns to 107 frames (4.458s at 24 fps); lower is better",
        (left, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (95, 95, 95),
        1,
        cv2.LINE_AA,
    )

    for seconds in range(0, axis_max + 1, tick):
        x = left + round(seconds / axis_max * plot_width)
        cv2.line(canvas, (x, top), (x, top + plot_height), (230, 230, 230), 1)
        cv2.putText(
            canvas,
            str(seconds),
            (x - 10, top + plot_height + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (85, 85, 85),
            1,
            cv2.LINE_AA,
        )

    row_height = plot_height / len(RESOLUTIONS)
    bar_height = round(row_height * 0.56)
    for row, ((width, height), seconds) in enumerate(zip(RESOLUTIONS, times, strict=True)):
        name = _key(width, height)
        center = top + round((row + 0.5) * row_height)
        bar_top = center - bar_height // 2
        bar_right = left + round(seconds / axis_max * plot_width)
        cv2.rectangle(
            canvas,
            (left, bar_top),
            (bar_right, bar_top + bar_height),
            (210, 126, 47),
            -1,
        )
        label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0]
        cv2.putText(
            canvas,
            name,
            (left - label_size[0] - 18, center + 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        label_x = min(bar_right + 15, chart_width - 130)
        cv2.putText(
            canvas,
            _format_seconds(seconds),
            (label_x, center - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{width * height / 1_000_000:.2f} MP | "
            f"{matrix[name]['complete_peak_memory_bytes'] / 1_000_000_000:.2f} GB MLX peak",
            (label_x, center + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (105, 105, 105),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "seconds for sampling + direct video/audio decode + mux (shared text encoding excluded)",
        (left + 165, chart_height - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (85, 85, 85),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(target), canvas):
        raise OSError(f"Could not write scaling chart: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument("--ffmpeg", type=Path, default=Path("/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--publish-prefix", default="FastH3-Scaling-M3-Ultra-20260829")
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    matrix = json.loads((source / "matrix.json").read_text(encoding="utf-8"))
    scenario = json.loads((source / "scenario.json").read_text(encoding="utf-8"))
    missing = [_key(*resolution) for resolution in RESOLUTIONS if _key(*resolution) not in matrix]
    if missing:
        raise ValueError(f"Scaling matrix is incomplete; missing rows: {missing}")
    ffmpeg = args.ffmpeg.expanduser().resolve()
    ffprobe = ffmpeg.with_name("ffprobe")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        raise FileNotFoundError("ffmpeg and ffprobe are required")

    verification = {}
    rows = []
    for width, height in RESOLUTIONS:
        name = _key(width, height)
        video = Path(matrix[name]["video"])
        if not video.is_file() or not Path(matrix[name]["latents"]).is_file():
            raise FileNotFoundError(f"Scaling artifacts are missing for {name}")
        probe = _probe(video, ffprobe)
        if (probe["width"], probe["height"]) != (width, height):
            raise ValueError(f"Published geometry mismatch for {name}: {probe}")
        if probe["frames"] != 107 or probe["fps"] != 24:
            raise ValueError(f"Published frame contract mismatch for {name}: {probe}")
        if probe["av_drift_seconds"] > 0.025:
            raise ValueError(f"A/V drift exceeds 25 ms for {name}: {probe}")
        audio = _decode_audio_health(video, ffmpeg)
        verification[name] = {"probe": probe, "audio_health": audio}
        rows.append(
            {
                "resolution": name,
                "width": width,
                "height": height,
                "megapixels": matrix[name]["megapixels"],
                "aligned_frames": matrix[name]["aligned_frames"],
                "delivered_duration_seconds": matrix[name]["delivered_duration_seconds"],
                "sampling_wall_seconds": matrix[name]["sampling"]["sampling_wall_seconds"],
                "transformer_seconds": matrix[name]["sampling"]["transformer_seconds"],
                "publication_seconds": matrix[name]["publication_seconds"],
                "complete_wall_seconds": matrix[name]["complete_wall_seconds"],
                "sampling_peak_memory_gb": (
                    matrix[name]["sampling"]["sampling_peak_memory_bytes"] / 1_000_000_000
                ),
                "complete_peak_memory_gb": (
                    matrix[name]["complete_peak_memory_bytes"] / 1_000_000_000
                ),
                "vsa_storage_layout": matrix[name]["profile"]["storage_layout"],
                "vsa_backend": matrix[name]["profile"]["attention_backend"],
                "video": str(video),
            }
        )

    chart = source / "fasth3-scaling-chart.png"
    contact = source / "fasth3-scaling-contact-sheet.jpg"
    csv_path = source / "fasth3-scaling-matrix.csv"
    report_path = source / "fasth3-scaling-verification.json"
    _make_chart(matrix, chart)
    _make_contact_sheet(matrix, contact)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path.write_text(
        json.dumps(
            {
                "status": "verified",
                "model": "FastH3 ~35.05B DiT, Q8 core with BF16 VSA gates",
                "hardware": scenario.get("hardware"),
                "frame_contract": "4.0-second request -> 107 aligned frames -> 4.458s at 24fps",
                "rows": rows,
                "verification": verification,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    publish_root = args.publish_root.expanduser().resolve()
    publish_root.mkdir(parents=True, exist_ok=True)
    publications = {
        "chart": (chart, publish_root / f"{args.publish_prefix}-Chart.png"),
        "contact_sheet": (contact, publish_root / f"{args.publish_prefix}-Contact-Sheet.jpg"),
        "csv": (csv_path, publish_root / f"{args.publish_prefix}-Matrix.csv"),
        "report": (report_path, publish_root / f"{args.publish_prefix}-Report.json"),
    }
    for name, (origin, target) in publications.items():
        shutil.copy2(origin, target)
        if not target.is_file() or target.stat().st_size != origin.stat().st_size:
            raise OSError(f"Published {name} failed verification: {target}")
    for width, height in RESOLUTIONS:
        name = _key(width, height)
        origin = Path(matrix[name]["video"])
        target = publish_root / f"{args.publish_prefix}-{name}.mp4"
        shutil.copy2(origin, target)
        if not target.is_file() or target.stat().st_size != origin.stat().st_size:
            raise OSError(f"Published video failed verification: {target}")

    print(
        json.dumps(
            {
                "status": "verified_and_published",
                "chart": str(publications["chart"][1]),
                "contact_sheet": str(publications["contact_sheet"][1]),
                "csv": str(publications["csv"][1]),
                "report": str(publications["report"][1]),
                "videos": [
                    str(publish_root / f"{args.publish_prefix}-{_key(*resolution)}.mp4")
                    for resolution in RESOLUTIONS
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
