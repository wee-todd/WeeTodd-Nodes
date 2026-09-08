"""MLX-native Canny edge detection for ComfyUI IMAGE batches."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CannyConfig:
    low_threshold: float = 0.4
    high_threshold: float = 0.8
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    hysteresis: bool = True
    frame_chunk_size: int = 16

    def validate(self) -> None:
        if not 0.0 < self.low_threshold <= self.high_threshold < 1.0:
            raise ValueError(
                "MLX Canny thresholds must satisfy 0 < low_threshold <= "
                "high_threshold < 1."
            )
        if self.gaussian_kernel_size not in {3, 5, 7}:
            raise ValueError("MLX Canny Gaussian kernel size must be 3, 5, or 7.")
        if not 0.1 <= self.gaussian_sigma <= 4.0:
            raise ValueError("MLX Canny Gaussian sigma must be between 0.1 and 4.0.")
        if self.frame_chunk_size not in {4, 8, 16, 32, 64}:
            raise ValueError("MLX Canny frame chunk size must be 4, 8, 16, 32, or 64.")


def _host_image_batch(images: Any) -> np.ndarray:
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    value = np.asarray(images, dtype=np.float32)
    if value.ndim != 4 or value.shape[0] < 1 or value.shape[-1] < 3:
        raise ValueError(
            "MLX Canny requires a ComfyUI IMAGE batch with shape "
            "(frames, height, width, channels)."
        )
    value = value[..., :3]
    if not np.isfinite(value).all():
        raise ValueError("MLX Canny input contains non-finite pixels.")
    return np.ascontiguousarray(np.clip(value, 0.0, 1.0))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _conv2d_same(value, kernel, *, padding_mode: str):
    import mlx.core as mx

    pad_y = int(kernel.shape[1]) // 2
    pad_x = int(kernel.shape[2]) // 2
    if padding_mode == "reflect":
        top = mx.flip(value[:, 1 : pad_y + 1], axis=1) if pad_y else value[:, :0]
        bottom = mx.flip(value[:, -pad_y - 1 : -1], axis=1) if pad_y else value[:, :0]
        padded = mx.concatenate((top, value, bottom), axis=1)
        left = mx.flip(padded[:, :, 1 : pad_x + 1], axis=2) if pad_x else padded[:, :, :0]
        right = (
            mx.flip(padded[:, :, -pad_x - 1 : -1], axis=2)
            if pad_x
            else padded[:, :, :0]
        )
        padded = mx.concatenate((left, padded, right), axis=2)
    elif padding_mode == "edge":
        top = mx.repeat(value[:, :1], pad_y, axis=1)
        bottom = mx.repeat(value[:, -1:], pad_y, axis=1)
        padded = mx.concatenate((top, value, bottom), axis=1)
        left = mx.repeat(padded[:, :, :1], pad_x, axis=2)
        right = mx.repeat(padded[:, :, -1:], pad_x, axis=2)
        padded = mx.concatenate((left, padded, right), axis=2)
    else:
        raise ValueError(f"Unsupported MLX Canny padding mode: {padding_mode!r}.")
    return mx.conv2d(padded, kernel)


def _shift(value, offset_y: int, offset_x: int):
    import mlx.core as mx

    height, width = int(value.shape[1]), int(value.shape[2])
    padded = mx.pad(value, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    y = 1 + offset_y
    x = 1 + offset_x
    return padded[:, y : y + height, x : x + width, :]


def _hysteresis(edges, weak, *, maximum_iterations: int):
    import mlx.core as mx

    strong = edges
    iterations = 0
    for _iteration in range(maximum_iterations):
        iterations += 1
        neighbors = mx.maximum(
            mx.maximum(_shift(strong, -1, -1), _shift(strong, -1, 0)),
            mx.maximum(_shift(strong, -1, 1), _shift(strong, 0, -1)),
        )
        neighbors = mx.maximum(
            neighbors,
            mx.maximum(_shift(strong, 0, 1), _shift(strong, 1, -1)),
        )
        neighbors = mx.maximum(
            neighbors,
            mx.maximum(_shift(strong, 1, 0), _shift(strong, 1, 1)),
        )
        updated = mx.logical_or(strong, mx.logical_and(weak, neighbors))
        changed = bool(np.asarray(mx.any(updated != strong)).item())
        strong = updated
        if not changed:
            break
    return strong, iterations


def _canny_mlx_batch(source: np.ndarray, config: CannyConfig):
    import mlx.core as mx

    value = mx.array(source, dtype=mx.float32)
    gray = (
        value[..., 0:1] * 0.299
        + value[..., 1:2] * 0.587
        + value[..., 2:3] * 0.114
    )

    gaussian = _gaussian_kernel(config.gaussian_kernel_size, config.gaussian_sigma)
    kernel_x = mx.array(gaussian.reshape(1, 1, -1, 1), dtype=mx.float32)
    kernel_y = mx.array(gaussian.reshape(1, -1, 1, 1), dtype=mx.float32)
    blurred = _conv2d_same(gray, kernel_x, padding_mode="reflect")
    blurred = _conv2d_same(blurred, kernel_y, padding_mode="reflect")

    sobel_x = mx.array(
        [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]], [[-1.0], [0.0], [1.0]]],
        dtype=mx.float32,
    ).reshape(1, 3, 3, 1)
    sobel_y = mx.transpose(sobel_x, (0, 2, 1, 3))
    gradient_x = _conv2d_same(blurred, sobel_x, padding_mode="edge")
    gradient_y = _conv2d_same(blurred, sobel_y, padding_mode="edge")
    magnitude = mx.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-6)
    direction = mx.round(mx.arctan2(gradient_y, gradient_x) * (4.0 / np.pi)) % 4

    horizontal = mx.logical_and(
        magnitude >= _shift(magnitude, 0, -1), magnitude >= _shift(magnitude, 0, 1)
    )
    diagonal_up = mx.logical_and(
        magnitude >= _shift(magnitude, -1, 1), magnitude >= _shift(magnitude, 1, -1)
    )
    vertical = mx.logical_and(
        magnitude >= _shift(magnitude, -1, 0), magnitude >= _shift(magnitude, 1, 0)
    )
    diagonal_down = mx.logical_and(
        magnitude >= _shift(magnitude, -1, -1), magnitude >= _shift(magnitude, 1, 1)
    )
    is_maximum = mx.where(
        direction == 0,
        horizontal,
        mx.where(direction == 1, diagonal_up, mx.where(direction == 2, vertical, diagonal_down)),
    )
    suppressed = mx.where(is_maximum, magnitude, 0.0)
    strong = suppressed > config.high_threshold
    weak = suppressed > config.low_threshold
    iterations = 0
    if config.hysteresis:
        maximum_iterations = max(1, ceil(max(source.shape[1:3]) / 2))
        edges, iterations = _hysteresis(strong, weak, maximum_iterations=maximum_iterations)
    else:
        edges = strong.astype(mx.float32) + mx.logical_and(weak, ~strong).astype(mx.float32) * 0.5
    output = mx.repeat(edges.astype(mx.float32), 3, axis=-1)
    mx.eval(output)
    return np.asarray(output, dtype=np.float32), iterations


def canny_edges(images: Any, config: CannyConfig | None = None):
    """Return RGB edge frames and a compact execution report.

    MLX owns Gaussian blur, gradients, non-maximum suppression, thresholding,
    and hysteresis. NumPy is used only at the ComfyUI input and output boundary.
    """
    import mlx.core as mx

    config = config or CannyConfig()
    config.validate()
    source = _host_image_batch(images)
    host_output = np.empty(source.shape, dtype=np.float32)
    iteration_counts = []
    for start in range(0, int(source.shape[0]), config.frame_chunk_size):
        stop = min(int(source.shape[0]), start + config.frame_chunk_size)
        host_output[start:stop], iterations = _canny_mlx_batch(source[start:stop], config)
        iteration_counts.append(iterations)
        mx.clear_cache()
    return host_output, {
        "backend": "mlx",
        "algorithm": "canny",
        "frames": int(source.shape[0]),
        "height": int(source.shape[1]),
        "width": int(source.shape[2]),
        "low_threshold": config.low_threshold,
        "high_threshold": config.high_threshold,
        "gaussian_kernel_size": config.gaussian_kernel_size,
        "gaussian_sigma": config.gaussian_sigma,
        "hysteresis": config.hysteresis,
        "frame_chunk_size": config.frame_chunk_size,
        "hysteresis_iterations": max(iteration_counts, default=0),
        "hysteresis_iterations_per_chunk": iteration_counts,
    }
