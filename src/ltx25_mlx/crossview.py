"""Depth-based camera warp preparation for LTX CrossView IC-LoRA control."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

_MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)
_PATH_FIELDS = (
    "azimuth",
    "elevation",
    "distance",
    "vertical_shift",
    "pivot_x",
    "pivot_y",
    "pivot_z",
)


@dataclass(frozen=True)
class CrossViewPose:
    frame: int
    azimuth: float
    elevation: float
    distance: float
    vertical_shift: float
    pivot_x: float
    pivot_y: float
    pivot_z: float


def _host_images(value: Any, *, name: str) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] < 1 or array.shape[-1] < 1:
        raise ValueError(f"CrossView {name} must be a ComfyUI IMAGE frame batch.")
    if not np.isfinite(array).all():
        raise ValueError(f"CrossView {name} contains non-finite pixels.")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        array = np.clip(array, 0.0, 1.0)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _unwrap(values: list[float]) -> list[float]:
    result = [float(values[0])]
    for value in values[1:]:
        result.append(result[-1] + _wrap_degrees(float(value) - result[-1]))
    return result


def _ease(value: float, mode: str) -> float:
    if mode == "ease_in":
        return value * value
    if mode == "ease_out":
        return 1.0 - (1.0 - value) ** 2
    if mode == "ease_in_out":
        return 0.5 - 0.5 * math.cos(math.pi * value)
    return value


def _catmull(values: list[float], index: int, value: float) -> float:
    p1, p2 = values[index], values[index + 1]
    p0 = values[index - 1] if index else 2.0 * p1 - p2
    p3 = values[index + 2] if index + 2 < len(values) else 2.0 * p2 - p1
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * value
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * value**2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * value**3
    )


def parse_crossview_keyframes(
    raw: str,
    *,
    frame_count: int,
    default_pose: CrossViewPose,
) -> tuple[CrossViewPose, ...]:
    """Parse a 1-based camera path and preserve the trained short-arc convention."""

    if not str(raw).strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("CrossView camera keyframes must be valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("CrossView camera keyframes must be a JSON list.")
    result: list[CrossViewPose] = []
    aliases = {
        "frame": ("frame", "f"),
        "azimuth": ("azimuth", "az"),
        "elevation": ("elevation", "el"),
        "distance": ("distance", "dist"),
        "vertical_shift": ("vertical_shift", "vs"),
        "pivot_x": ("pivot_x", "px"),
        "pivot_y": ("pivot_y", "py"),
        "pivot_z": ("pivot_z", "pz"),
    }
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"CrossView keyframe {index} must be a JSON object.")

        def read(
            field: str,
            fallback: float | int,
            data: dict[str, object] = item,
            item_index: int = index,
        ) -> float:
            for key in aliases[field]:
                if key in data:
                    try:
                        return float(data[key])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"CrossView keyframe {item_index} has a non-numeric {field}."
                        ) from exc
            return float(fallback)

        frame = int(round(read("frame", 0)))
        if not 1 <= frame <= frame_count:
            raise ValueError(
                f"CrossView keyframe {index} targets frame {frame}; use 1..{frame_count}."
            )
        result.append(
            CrossViewPose(
                frame=frame,
                azimuth=read("azimuth", default_pose.azimuth),
                elevation=read("elevation", default_pose.elevation),
                distance=read("distance", default_pose.distance),
                vertical_shift=read("vertical_shift", default_pose.vertical_shift),
                pivot_x=read("pivot_x", default_pose.pivot_x),
                pivot_y=read("pivot_y", default_pose.pivot_y),
                pivot_z=read("pivot_z", default_pose.pivot_z),
            )
        )
    result.sort(key=lambda item: item.frame)
    if len({item.frame for item in result}) != len(result):
        raise ValueError("CrossView camera keyframes must target different frames.")
    return tuple(result)


def sample_crossview_path(
    keyframes: tuple[CrossViewPose, ...],
    *,
    frame: int,
    interpolation: str,
) -> CrossViewPose:
    if not keyframes:
        raise ValueError("CrossView path sampling requires at least one keyframe.")
    if len(keyframes) == 1 or frame <= keyframes[0].frame:
        return CrossViewPose(frame, *tuple(getattr(keyframes[0], field) for field in _PATH_FIELDS))
    if frame >= keyframes[-1].frame:
        return CrossViewPose(frame, *tuple(getattr(keyframes[-1], field) for field in _PATH_FIELDS))
    segment = next(
        index
        for index in range(len(keyframes) - 1)
        if keyframes[index].frame <= frame <= keyframes[index + 1].frame
    )
    left, right = keyframes[segment], keyframes[segment + 1]
    fraction = (frame - left.frame) / (right.frame - left.frame)
    smooth = interpolation == "smooth"
    if not smooth:
        fraction = _ease(fraction, interpolation)
    channels: dict[str, list[float]] = {
        field: [float(getattr(item, field)) for item in keyframes] for field in _PATH_FIELDS
    }
    channels["azimuth"] = _unwrap(channels["azimuth"])
    sampled = {}
    for field, values in channels.items():
        sampled[field] = (
            _catmull(values, segment, fraction)
            if smooth and len(values) > 2
            else values[segment] + (values[segment + 1] - values[segment]) * fraction
        )
    sampled["azimuth"] = _wrap_degrees(sampled["azimuth"])
    sampled["elevation"] = float(np.clip(sampled["elevation"], -90.0, 90.0))
    sampled["distance"] = float(np.clip(sampled["distance"], 0.1, 3.0))
    sampled["vertical_shift"] = float(np.clip(sampled["vertical_shift"], -1.0, 1.0))
    sampled["pivot_z"] = max(sampled["pivot_z"], 0.01)
    return CrossViewPose(frame=frame, **sampled)


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    if np.linalg.norm(right) < 1e-7:
        right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    camera = np.eye(4, dtype=np.float64)
    camera[:3, :3] = np.stack([right, down, forward], axis=1)
    camera[:3, 3] = eye
    return camera


def _orbit_camera(pose: CrossViewPose, *, keep_source_aim: bool) -> np.ndarray:
    pivot = np.array([pose.pivot_x, pose.pivot_y, pose.pivot_z], dtype=np.float64)
    azimuth = math.radians(-pose.azimuth)
    elevation = math.radians(-pose.elevation)
    ry = np.array(
        [
            [math.cos(azimuth), 0.0, math.sin(azimuth)],
            [0.0, 1.0, 0.0],
            [-math.sin(azimuth), 0.0, math.cos(azimuth)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elevation), -math.sin(elevation)],
            [0.0, math.sin(elevation), math.cos(elevation)],
        ],
        dtype=np.float64,
    )
    eye = pivot + pose.distance * ((ry @ rx) @ -pivot)
    aim = (
        np.array([0.0, 0.0, max(float(np.linalg.norm(pivot)), 1e-3)]) if keep_source_aim else pivot
    )
    return _look_at(eye, aim)


def normalize_crossview_depth(depth: np.ndarray, *, invert: bool, depth_ratio: float) -> np.ndarray:
    """Convert a relative near-white depth sequence into stable metric-like Z values."""

    values = np.asarray(depth, dtype=np.float64)
    if invert:
        values = -values
    low, high = np.percentile(values, (1.0, 99.0))
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    ratio = max(float(depth_ratio), 1.01)
    return 1.0 / (1.0 / ratio + (1.0 - 1.0 / ratio) * normalized)


def _crossview_depth_range(depth: np.ndarray, *, invert: bool) -> tuple[float, float]:
    low, high = (float(value) for value in np.percentile(depth, (1.0, 99.0)))
    return (-high, -low) if invert else (low, high)


def _normalize_crossview_depth_frame(
    depth: np.ndarray,
    *,
    invert: bool,
    low: float,
    high: float,
    depth_ratio: float,
) -> np.ndarray:
    values = depth.astype(np.float64, copy=False)
    if invert:
        values = -values
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    ratio = max(float(depth_ratio), 1.01)
    return 1.0 / (1.0 / ratio + (1.0 - 1.0 / ratio) * normalized)


@lru_cache(maxsize=2)
def _projection_grid(
    height: int,
    width: int,
    horizontal_fov: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Reuse the immutable projection grid without retaining source media."""

    focal = width / (2.0 * math.tan(math.radians(horizontal_fov) / 2.0))
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    normalized_x = (columns - width / 2.0) / focal
    normalized_y = (rows - height / 2.0) / focal
    normalized_x.setflags(write=False)
    normalized_y.setflags(write=False)
    return focal, normalized_x, normalized_y


def clear_crossview_geometry_cache() -> int:
    """Release cached projection grids and return the number of retained entries."""

    count = _projection_grid.cache_info().currsize
    _projection_grid.cache_clear()
    return int(count)


def _warp_frame(
    rgb: np.ndarray,
    depth: np.ndarray,
    *,
    camera: np.ndarray,
    focal: float,
    vertical_shift: float,
    splat_radius: int,
    normalized_x: np.ndarray,
    normalized_y: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    world = np.stack([normalized_x * depth, normalized_y * depth, depth], axis=-1).reshape(-1, 3)
    target = (world - camera[:3, 3]) @ camera[:3, :3]
    z_target = target[:, 2]
    flat_depth = depth.reshape(-1)
    finite_depth = np.isfinite(flat_depth) & (flat_depth > 0.0)
    far_limit = np.percentile(flat_depth[finite_depth], 99.5) if finite_depth.any() else 0.0
    source_valid = finite_depth & (flat_depth < far_limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        projected_x = np.rint(target[:, 0] / z_target * focal + width / 2.0)
        projected_y = np.rint(
            target[:, 1] / z_target * focal + height / 2.0 + vertical_shift * height
        )
    valid = source_valid & np.isfinite(projected_x) & np.isfinite(projected_y) & (z_target > 0.0)
    indices = np.flatnonzero(valid)
    # Far rows write first and near rows write last. This painter order is the
    # conditioning format used to train the adapter.
    indices = indices[np.argsort(-z_target[indices], kind="stable")]
    x0 = projected_x[indices].astype(np.int64)
    y0 = projected_y[indices].astype(np.int64)
    colors = rgb.reshape(-1, 3)[indices]
    output = np.broadcast_to(_MAGENTA, (height, width, 3)).copy()
    flat = output.reshape(-1, 3)
    for offset_y in range(-splat_radius, splat_radius + 1):
        for offset_x in range(-splat_radius, splat_radius + 1):
            x = x0 + offset_x
            y = y0 + offset_y
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            flat[y[inside] * width + x[inside]] = colors[inside]
    return output


def build_crossview_warp(
    source_images: Any,
    depth_images: Any,
    *,
    azimuth: float = -30.0,
    elevation: float = 20.0,
    distance: float = 1.0,
    horizontal_fov: float = 50.0,
    vertical_shift: float = 0.0,
    depth_ratio: float = 6.0,
    invert_depth: bool = False,
    pivot_x: float = 0.0,
    pivot_y: float = 0.0,
    pivot_z: float = 1.05,
    keep_source_aim: bool = True,
    keyframes: str = "",
    interpolation: str = "linear",
    splat_radius: int = 2,
    progress_callback: Callable[[int, int], None] | None = None,
    interruption_callback: Callable[[], None] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build full-resolution CrossView conditioning without loading model weights."""

    started = time.perf_counter()
    source = _host_images(source_images, name="source video")[..., :3]
    depth_batch = _host_images(depth_images, name="depth video")
    if source.shape[:3] != depth_batch.shape[:3]:
        raise ValueError(
            "CrossView source and depth batches must have identical frame counts, "
            "heights, and widths."
        )
    if not 20.0 <= horizontal_fov <= 120.0:
        raise ValueError("CrossView horizontal field of view must be between 20 and 120 degrees.")
    if interpolation not in {"linear", "ease_in", "ease_out", "ease_in_out", "smooth"}:
        raise ValueError(f"Unsupported CrossView interpolation: {interpolation!r}.")
    if not 0 <= splat_radius <= 3:
        raise ValueError("CrossView splat radius must be between zero and three pixels.")
    depth = depth_batch[..., :3].mean(axis=-1)
    depth_low, depth_high = _crossview_depth_range(depth, invert=invert_depth)
    frame_count, height, width = source.shape[:3]
    cache_before = _projection_grid.cache_info()
    focal, normalized_x, normalized_y = _projection_grid(
        int(height),
        int(width),
        round(float(horizontal_fov), 6),
    )
    geometry_cache_hit = _projection_grid.cache_info().hits > cache_before.hits
    default_pose = CrossViewPose(
        frame=1,
        azimuth=float(azimuth),
        elevation=float(elevation),
        distance=float(distance),
        vertical_shift=float(vertical_shift),
        pivot_x=float(pivot_x),
        pivot_y=float(pivot_y),
        pivot_z=float(pivot_z),
    )
    parsed = parse_crossview_keyframes(
        keyframes,
        frame_count=frame_count,
        default_pose=default_pose,
    )
    static_camera = (
        _orbit_camera(default_pose, keep_source_aim=keep_source_aim) if not parsed else None
    )
    result = np.empty_like(source, dtype=np.float32)
    for frame_index in range(frame_count):
        if interruption_callback is not None:
            interruption_callback()
        pose = (
            sample_crossview_path(parsed, frame=frame_index + 1, interpolation=interpolation)
            if parsed
            else CrossViewPose(
                frame_index + 1, *tuple(getattr(default_pose, field) for field in _PATH_FIELDS)
            )
        )
        result[frame_index] = _warp_frame(
            source[frame_index],
            _normalize_crossview_depth_frame(
                depth[frame_index],
                invert=invert_depth,
                low=depth_low,
                high=depth_high,
                depth_ratio=depth_ratio,
            ),
            camera=(
                static_camera
                if static_camera is not None
                else _orbit_camera(pose, keep_source_aim=keep_source_aim)
            ),
            focal=focal,
            vertical_shift=pose.vertical_shift,
            splat_radius=splat_radius,
            normalized_x=normalized_x,
            normalized_y=normalized_y,
        )
        if progress_callback is not None:
            progress_callback(frame_index + 1, frame_count)
    hole_fraction = float(np.mean(np.all(result == _MAGENTA, axis=-1)))
    return result, {
        "backend": "vectorized_numpy",
        "conditioning_contract": "crossview_warp_then_source",
        "frames": int(frame_count),
        "width": int(width),
        "height": int(height),
        "horizontal_fov": float(horizontal_fov),
        "keep_source_aim": bool(keep_source_aim),
        "keyframe_count": len(parsed),
        "interpolation": interpolation,
        "splat_radius": int(splat_radius),
        "projection_grid_cache_hit": geometry_cache_hit,
        "projection_grid_cache_entries": _projection_grid.cache_info().currsize,
        "magenta_hole_fraction": hole_fraction,
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = [
    "CrossViewPose",
    "build_crossview_warp",
    "clear_crossview_geometry_cache",
    "normalize_crossview_depth",
    "parse_crossview_keyframes",
    "sample_crossview_path",
]
