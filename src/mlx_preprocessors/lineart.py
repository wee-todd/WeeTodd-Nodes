"""MLX-native realistic line-art extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import load_file

from .video_depth import _resize


def _reflect_pad(value, amount: int):
    if amount == 0:
        return value
    height = np.pad(np.arange(value.shape[1]), amount, mode="reflect")
    width = np.pad(np.arange(value.shape[2]), amount, mode="reflect")
    return mx.take(mx.take(value, mx.array(height), axis=1), mx.array(width), axis=2)


def _instance_norm(value, epsilon=1e-5):
    mean = mx.mean(value, axis=(1, 2), keepdims=True)
    variance = mx.mean((value - mean) ** 2, axis=(1, 2), keepdims=True)
    return (value - mean) * mx.rsqrt(variance + epsilon)


class _ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)

    def __call__(self, value):
        residual = self.conv1(_reflect_pad(value, 1))
        residual = mx.maximum(_instance_norm(residual), 0.0)
        residual = _instance_norm(self.conv2(_reflect_pad(residual, 1)))
        return value + residual


class RealisticLineArt(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(3, 64, 7)
        self.down_convs = [
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        ]
        self.residuals = [_ResidualBlock() for _ in range(3)]
        self.up_convs = [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        ]
        self.output_conv = nn.Conv2d(64, 1, 7)

    def __call__(self, value):
        value = mx.maximum(_instance_norm(self.input_conv(_reflect_pad(value, 3))), 0.0)
        for convolution in self.down_convs:
            value = mx.maximum(_instance_norm(convolution(value)), 0.0)
        for residual in self.residuals:
            value = residual(value)
        for convolution in self.up_convs:
            value = mx.maximum(_instance_norm(convolution(value)), 0.0)
        return mx.sigmoid(self.output_conv(_reflect_pad(value, 3)))


def load_realistic_lineart(path: str | Path):
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(
            "The MLX line-art loader requires a converted .safetensors checkpoint. "
            "Run scripts/convert_lineart_mlx.py once in an environment with PyTorch."
        )
    model = RealisticLineArt()
    expected = dict(tree_flatten(model.parameters()))
    source = load_file(path)
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Line-art checkpoint mismatch: {len(missing)} missing and {len(extra)} extra tensors."
        )
    model.load_weights([(name, mx.array(source[name])) for name in expected], strict=True)
    return model


@dataclass(frozen=True)
class LineArtConfig:
    detect_resolution: int = 512
    frame_chunk_size: int = 2
    output_mode: str = "white lines"

    def validate(self):
        if self.detect_resolution not in {256, 384, 512, 640, 768, 1024}:
            raise ValueError("Line-art resolution must be 256, 384, 512, 640, 768, or 1024.")
        if self.frame_chunk_size not in {1, 2, 4, 8, 16}:
            raise ValueError("Line-art frame chunk size must be 1, 2, 4, 8, or 16.")
        if self.output_mode not in {"black lines", "white lines"}:
            raise ValueError("Line-art output mode must be black lines or white lines.")


def _prepare(images: Any):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    value = np.asarray(images, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError("Line art requires a ComfyUI IMAGE frame batch.")
    return np.ascontiguousarray(np.clip(value[..., :3], 0.0, 1.0))


def _network_size(height: int, width: int, resolution: int):
    scale = resolution / min(height, width)
    target_h = max(8, round(height * scale / 8) * 8)
    target_w = max(8, round(width * scale / 8) * 8)
    return target_h, target_w


def infer_realistic_lineart(
    images,
    model: RealisticLineArt,
    config: LineArtConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or LineArtConfig()
    config.validate()
    source = _prepare(images)
    target_h, target_w = _network_size(source.shape[1], source.shape[2], config.detect_resolution)
    output = []
    for start in range(0, len(source), config.frame_chunk_size):
        if interruption_callback is not None:
            interruption_callback()
        value = mx.array(source[start : start + config.frame_chunk_size])
        if value.shape[1:3] != (target_h, target_w):
            value = _resize(value, target_h, target_w, mode="cubic")
        line_probability = model(value)
        line = 1.0 - line_probability if config.output_mode == "white lines" else line_probability
        line = _resize(line, source.shape[1], source.shape[2], mode="cubic")
        line = mx.repeat(mx.clip(line, 0.0, 1.0), 3, axis=-1)
        mx.eval(line)
        output.append(np.asarray(line, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(min(start + config.frame_chunk_size, len(source)), len(source))
    return np.concatenate(output), {
        "backend": "mlx",
        "algorithm": "realistic_lineart",
        "frames": len(source),
        "detect_resolution": config.detect_resolution,
        "network_height": target_h,
        "network_width": target_w,
        "frame_chunk_size": config.frame_chunk_size,
        "output_mode": config.output_mode,
    }
