"""Sparse motion-track guide rendering for LTX IC-LoRA conditioning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class MotionTrackConfig:
    width: int = 768
    height: int = 512
    num_frames: int = 121
    coordinate_space: str = "normalized"
    track_format: str = "spline control points"
    trail_frames: int = 50
    reference_short_side: int = 1080

    def validate(self) -> None:
        if not 64 <= self.width <= 1920 or self.width % 8:
            raise ValueError("Motion-track width must be 64..1920 and divisible by 8.")
        if not 64 <= self.height <= 1920 or self.height % 8:
            raise ValueError("Motion-track height must be 64..1920 and divisible by 8.")
        if not 2 <= self.num_frames <= 1000:
            raise ValueError("Motion-track frame count must be between 2 and 1000.")
        if self.coordinate_space not in {"normalized", "pixels"}:
            raise ValueError("Motion-track coordinate space must be normalized or pixels.")
        if self.track_format not in {"spline control points", "per-frame coordinates"}:
            raise ValueError(
                "Motion-track format must be spline control points or per-frame coordinates."
            )
        if not 1 <= self.trail_frames <= 200:
            raise ValueError("Motion-track trail length must be between 1 and 200 frames.")
        if not 256 <= self.reference_short_side <= 2160:
            raise ValueError("Motion-track reference short side must be between 256 and 2160.")


def _point(value) -> tuple[float, float]:
    if isinstance(value, dict) and "x" in value and "y" in value:
        return float(value["x"]), float(value["y"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError("Each motion-track point must contain numeric x and y coordinates.")


def _catmull_rom(points: list[tuple[float, float]], count: int) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(np.asarray(points, dtype=np.float32), count, axis=0)
    if len(points) == 2:
        return np.linspace(points[0], points[1], count, dtype=np.float32)
    padded = [points[0], *points, points[-1]]
    segments = len(padded) - 3
    result = np.empty((count, 2), dtype=np.float32)
    for index in range(count):
        position = index * segments / (count - 1)
        segment = min(int(position), segments - 1)
        t = position - segment
        t2 = t * t
        t3 = t2 * t
        p0, p1, p2, p3 = (np.asarray(padded[segment + offset]) for offset in range(4))
        result[index] = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
    return result


def _parse_tracks(raw_tracks, config: MotionTrackConfig) -> list[np.ndarray]:
    try:
        parsed = json.loads(raw_tracks) if isinstance(raw_tracks, str) else raw_tracks
    except json.JSONDecodeError as exc:
        raise ValueError(f"Motion-track JSON is invalid: {exc.msg}.") from exc
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Motion-track JSON must contain at least one track.")
    if len(parsed) > 16:
        raise ValueError("Motion-track guides support at most 16 simultaneous tracks.")
    tracks = []
    for track_index, raw_track in enumerate(parsed):
        if not isinstance(raw_track, list) or not raw_track:
            raise ValueError(f"Motion track {track_index + 1} has no points.")
        if len(raw_track) > 1000:
            raise ValueError(f"Motion track {track_index + 1} has too many points.")
        points = [_point(value) for value in raw_track]
        values = np.asarray(points, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Motion track {track_index + 1} contains non-finite coordinates.")
        if config.coordinate_space == "normalized":
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError("Normalized motion-track coordinates must remain between 0 and 1.")
            values *= np.asarray([config.width - 1, config.height - 1], dtype=np.float32)
        elif (
            np.any(values[:, 0] < 0.0)
            or np.any(values[:, 0] >= config.width)
            or np.any(values[:, 1] < 0.0)
            or np.any(values[:, 1] >= config.height)
        ):
            raise ValueError("Pixel motion-track coordinates must remain inside the output canvas.")
        if config.track_format == "spline control points":
            sampled = _catmull_rom([tuple(value) for value in values], config.num_frames)
        elif len(values) == 1:
            sampled = np.repeat(values, config.num_frames, axis=0)
        elif len(values) != config.num_frames:
            raise ValueError(
                f"Per-frame motion track {track_index + 1} has {len(values)} points; "
                f"expected {config.num_frames}."
            )
        else:
            sampled = values
        sampled[:, 0] = np.clip(np.rint(sampled[:, 0]), 0, config.width - 1)
        sampled[:, 1] = np.clip(np.rint(sampled[:, 1]), 0, config.height - 1)
        tracks.append(sampled.astype(np.int32))
    return tracks


def _training_color(ratio: float) -> np.ndarray:
    """Return the BGR-ordered guide color used by the motion-track training representation."""
    if ratio <= 1.0 / 3.0:
        local = ratio * 3.0
        rgb = (0.0, local, 1.0 - local)
    elif ratio <= 2.0 / 3.0:
        local = (ratio - 1.0 / 3.0) * 3.0
        rgb = (local, 1.0, 0.0)
    else:
        local = (ratio - 2.0 / 3.0) * 3.0
        rgb = (1.0, 1.0 - local, 0.0)
    return np.asarray(rgb[::-1], dtype=np.float32)


def _stamp_cache(config: MotionTrackConfig):
    scale = min(config.width, config.height) / config.reference_short_side
    result = []
    for age in range(config.trail_frames + 1):
        ratio = 1.0 - age / config.trail_frames
        radius = max(0.5, (2.0 + 6.0 * ratio) * scale)
        half = max(1, int(math.ceil(radius + 1.0)))
        axis = mx.arange(-half, half + 1, dtype=mx.float32)
        yy, xx = mx.meshgrid(axis, axis, indexing="ij")
        alpha = mx.clip(radius + 0.5 - mx.sqrt(xx * xx + yy * yy), 0.0, 1.0)[..., None]
        color = mx.array(_training_color(ratio))
        mx.eval(alpha, color)
        result.append((half, alpha, color))
    return result


def render_motion_tracks(
    raw_tracks,
    config: MotionTrackConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or MotionTrackConfig()
    config.validate()
    tracks = _parse_tracks(raw_tracks, config)
    stamps = _stamp_cache(config)
    output = []
    for frame_index in range(config.num_frames):
        if interruption_callback is not None:
            interruption_callback()
        frame = mx.zeros((config.height, config.width, 3), dtype=mx.float32)
        oldest = max(0, frame_index - config.trail_frames)
        for sample_index in range(oldest, frame_index + 1):
            age = frame_index - sample_index
            half, alpha, color = stamps[age]
            for track in tracks:
                x, y = (int(value) for value in track[sample_index])
                x0 = max(0, x - half)
                x1 = min(config.width, x + half + 1)
                y0 = max(0, y - half)
                y1 = min(config.height, y + half + 1)
                ax0 = x0 - (x - half)
                ax1 = ax0 + (x1 - x0)
                ay0 = y0 - (y - half)
                ay1 = ay0 + (y1 - y0)
                local_alpha = alpha[ay0:ay1, ax0:ax1]
                frame = frame.at[y0:y1, x0:x1].multiply(1.0 - local_alpha)
                frame = frame.at[y0:y1, x0:x1].add(local_alpha * color)
        mx.eval(frame)
        output.append(np.asarray(frame, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(frame_index + 1, config.num_frames)
    return np.stack(output), {
        "backend": "mlx_sparse_raster",
        "algorithm": "ltx_motion_track_guide",
        "frames": config.num_frames,
        "width": config.width,
        "height": config.height,
        "tracks": len(tracks),
        "coordinate_space": config.coordinate_space,
        "track_format": config.track_format,
        "trail_frames": config.trail_frames,
        "channel_order": "training_bgr",
        "reference_short_side": config.reference_short_side,
    }
