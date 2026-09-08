"""Deterministic sparse optical-flow tracks for LTX Motion Track guides."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class OpticalFlowTrackConfig:
    max_tracks: int = 8
    quality_level: float = 0.02
    minimum_distance: int = 32
    window_size: int = 21
    pyramid_levels: int = 3
    forward_backward_limit: float = 1.5

    def validate(self) -> None:
        if not 1 <= self.max_tracks <= 16:
            raise ValueError("Optical-flow max_tracks must be between 1 and 16.")
        if not 0.001 <= self.quality_level <= 0.5:
            raise ValueError("Optical-flow quality_level must be in [0.001, 0.5].")
        if not 2 <= self.minimum_distance <= 512:
            raise ValueError("Optical-flow minimum_distance must be between 2 and 512.")
        if not 5 <= self.window_size <= 63 or self.window_size % 2 == 0:
            raise ValueError("Optical-flow window_size must be an odd integer in [5, 63].")
        if not 0 <= self.pyramid_levels <= 6:
            raise ValueError("Optical-flow pyramid_levels must be between 0 and 6.")
        if not math.isfinite(self.forward_backward_limit) or not (
            0.1 <= self.forward_backward_limit <= 20.0
        ):
            raise ValueError("Optical-flow forward_backward_limit must be in [0.1, 20].")


def _host_frames(frames) -> np.ndarray:
    try:
        import torch

        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
    except ImportError:
        pass
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] not in {1, 3, 4} or values.shape[0] < 2:
        raise ValueError("Optical-flow input must be at least two HWC image frames.")
    if values.shape[1] < 8 or values.shape[2] < 8:
        raise ValueError("Optical-flow input frames must be at least 8x8 pixels.")
    if not np.isfinite(values).all():
        raise ValueError("Optical-flow input contains non-finite pixels.")
    if np.issubdtype(values.dtype, np.floating):
        if values.min() < 0 or values.max() > 1:
            raise ValueError("Floating optical-flow input pixels must be in [0, 1].")
        values = np.rint(values * 255.0).astype(np.uint8)
    else:
        values = np.clip(values, 0, 255).astype(np.uint8)
    if values.shape[-1] == 1:
        return values[..., 0]
    conversion = cv2.COLOR_RGBA2GRAY if values.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
    return np.stack([cv2.cvtColor(frame, conversion) for frame in values])


def extract_optical_flow_tracks(
    frames,
    config: OpticalFlowTrackConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    """Track strong first-frame features with forward/backward LK validation."""
    config = config or OpticalFlowTrackConfig()
    config.validate()
    gray = _host_frames(frames)
    frame_count, height, width = gray.shape
    points = cv2.goodFeaturesToTrack(
        gray[0],
        maxCorners=config.max_tracks,
        qualityLevel=config.quality_level,
        minDistance=config.minimum_distance,
        blockSize=7,
        useHarrisDetector=False,
    )
    fallback = points is None or len(points) == 0
    if fallback:
        points = np.array([[[0.5 * (width - 1), 0.5 * (height - 1)]]], dtype=np.float32)
    points = points.reshape(-1, 2).astype(np.float32)
    # OpenCV's response order can vary across implementations for equal scores.
    order = np.lexsort((points[:, 0], points[:, 1]))
    points = points[order]
    trajectories = np.empty((len(points), frame_count, 2), dtype=np.float32)
    trajectories[:, 0] = points
    held = np.zeros(len(points), dtype=np.int32)
    valid_observations = len(points)
    lk = {
        "winSize": (config.window_size, config.window_size),
        "maxLevel": config.pyramid_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    previous_points = points.reshape(-1, 1, 2)
    for frame_index in range(1, frame_count):
        if interruption_callback is not None:
            interruption_callback()
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame_index - 1], gray[frame_index], previous_points, None, **lk
        )
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame_index], gray[frame_index - 1], forward, None, **lk
        )
        error = np.linalg.norm(previous_points - backward, axis=2).reshape(-1)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(forward.reshape(-1, 2)).all(axis=1)
            & (error <= config.forward_backward_limit)
        )
        next_points = forward.reshape(-1, 2)
        in_bounds = (
            (next_points[:, 0] >= 0)
            & (next_points[:, 0] <= width - 1)
            & (next_points[:, 1] >= 0)
            & (next_points[:, 1] <= height - 1)
        )
        valid &= in_bounds
        next_points[~valid] = previous_points.reshape(-1, 2)[~valid]
        held += (~valid).astype(np.int32)
        valid_observations += int(valid.sum())
        trajectories[:, frame_index] = next_points
        previous_points = next_points.reshape(-1, 1, 2)
        if progress_callback is not None:
            progress_callback(frame_index, frame_count - 1)

    normalized = trajectories / np.array([max(width - 1, 1), max(height - 1, 1)])
    normalized = np.clip(normalized, 0.0, 1.0)
    tracks = [
        [{"x": round(float(x), 7), "y": round(float(y), 7)} for x, y in track]
        for track in normalized
    ]
    report = {
        "backend": "opencv_lucas_kanade",
        "algorithm": "sparse_pyramidal_lk_forward_backward",
        "frames": frame_count,
        "width": width,
        "height": height,
        "tracks": len(tracks),
        "fallback_center_track": fallback,
        "valid_observation_ratio": valid_observations / (len(points) * frame_count),
        "held_observations": int(held.sum()),
        "held_per_track": held.tolist(),
        "coordinate_space": "normalized",
        "track_format": "per-frame coordinates",
        "settings": {
            "max_tracks": config.max_tracks,
            "quality_level": config.quality_level,
            "minimum_distance": config.minimum_distance,
            "window_size": config.window_size,
            "pyramid_levels": config.pyramid_levels,
            "forward_backward_limit": config.forward_backward_limit,
        },
    }
    return json.dumps(tracks, separators=(",", ":")), report


__all__ = ["OpticalFlowTrackConfig", "extract_optical_flow_tracks"]
