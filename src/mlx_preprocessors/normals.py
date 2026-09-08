"""Weightless MLX surface-normal estimation from relative depth maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class NormalMapConfig:
    # Relative-depth maps are normalized to [0, 1], so their per-pixel slopes are
    # small.  A value near 40 gives a useful +Z-blue normal map at common video
    # resolutions while leaving room for gentler or more sculpted guides.
    strength: float = 40.0
    method: str = "sobel"
    depth_polarity: str = "near white"
    discontinuity_threshold: float = 0.0
    flip_y: bool = False
    frame_chunk_size: int = 16

    def validate(self):
        if not 0.1 <= self.strength <= 400.0:
            raise ValueError("Normal-map strength must be between 0.1 and 400.")
        if self.method not in {"central", "sobel"}:
            raise ValueError("Normal-map gradient method must be central or sobel.")
        if self.depth_polarity not in {"near white", "near black"}:
            raise ValueError("Normal-map depth polarity must be near white or near black.")
        if not 0.0 <= self.discontinuity_threshold <= 1.0:
            raise ValueError("Normal-map discontinuity threshold must be between 0 and 1.")
        if self.frame_chunk_size not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("Normal-map frame chunk size must be 1, 2, 4, 8, 16, 32, or 64.")


def _prepare(images: Any):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    value = np.asarray(images, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] < 1:
        raise ValueError("Normal maps require a ComfyUI IMAGE depth batch.")
    if value.shape[-1] >= 3:
        value = value[..., :3].mean(axis=-1)
    else:
        value = value[..., 0]
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0))


def _replicate_pad(value):
    value = mx.concatenate((value[:, :1], value, value[:, -1:]), axis=1)
    return mx.concatenate((value[:, :, :1], value, value[:, :, -1:]), axis=2)


def _gradients(value, method: str):
    padded = _replicate_pad(value)
    if method == "central":
        dx = 0.5 * (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2])
        dy = 0.5 * (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1])
        return dx, dy
    top = padded[:, :-2]
    middle = padded[:, 1:-1]
    bottom = padded[:, 2:]
    dx = (
        top[:, :, 2:]
        + 2.0 * middle[:, :, 2:]
        + bottom[:, :, 2:]
        - top[:, :, :-2]
        - 2.0 * middle[:, :, :-2]
        - bottom[:, :, :-2]
    ) / 8.0
    left = padded[:, :, :-2]
    center = padded[:, :, 1:-1]
    right = padded[:, :, 2:]
    dy = (
        left[:, 2:]
        + 2.0 * center[:, 2:]
        + right[:, 2:]
        - left[:, :-2]
        - 2.0 * center[:, :-2]
        - right[:, :-2]
    ) / 8.0
    return dx, dy


def depth_to_normals(
    depth_images,
    config: NormalMapConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or NormalMapConfig()
    config.validate()
    source = _prepare(depth_images)
    output = []
    for start in range(0, len(source), config.frame_chunk_size):
        if interruption_callback is not None:
            interruption_callback()
        depth = mx.array(source[start : start + config.frame_chunk_size])
        if config.depth_polarity == "near black":
            depth = 1.0 - depth
        dx, dy = _gradients(depth, config.method)
        normal_x = -dx * config.strength
        normal_y = dy * config.strength if config.flip_y else -dy * config.strength
        normal_z = mx.ones_like(normal_x)
        magnitude = mx.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
        normals = mx.stack((normal_x, normal_y, normal_z), axis=-1) / magnitude[..., None]
        if config.discontinuity_threshold:
            discontinuity = mx.sqrt(dx * dx + dy * dy) > config.discontinuity_threshold
            forward = mx.array([0.0, 0.0, 1.0])
            normals = mx.where(discontinuity[..., None], forward, normals)
        normals = mx.clip(normals * 0.5 + 0.5, 0.0, 1.0)
        mx.eval(normals)
        output.append(np.asarray(normals, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(min(start + config.frame_chunk_size, len(source)), len(source))
    return np.concatenate(output), {
        "backend": "mlx",
        "algorithm": "depth_derived_normals",
        "frames": len(source),
        "method": config.method,
        "strength": config.strength,
        "depth_polarity": config.depth_polarity,
        "discontinuity_threshold": config.discontinuity_threshold,
        "flip_y": config.flip_y,
        "frame_chunk_size": config.frame_chunk_size,
    }
