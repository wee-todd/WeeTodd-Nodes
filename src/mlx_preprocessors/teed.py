"""Independent MLX implementation of the Tiny and Efficient Edge Detector."""

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


def _smish(value):
    return value * mx.tanh(mx.log1p(mx.sigmoid(value)))


class _DoubleConv(nn.Module):
    def __init__(self, inputs, middle, outputs, *, stride=1, final_activation=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, middle, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(middle, outputs, 3, padding=1)
        self.final_activation = final_activation

    def __call__(self, value):
        value = self.conv2(_smish(self.conv1(value)))
        return _smish(value) if self.final_activation else value


class _SingleConv(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 1, stride=stride)

    def __call__(self, value):
        return self.conv(value)


class _DenseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 48, 3, padding=2)
        self.conv2 = nn.Conv2d(48, 48, 3)

    def __call__(self, value, residual):
        value = self.conv2(_smish(self.conv1(_smish(value))))
        return 0.5 * (value + residual)


class _DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.denselayer1 = _DenseLayer()

    def __call__(self, value, residual):
        return self.denselayer1(value, residual)


class _UpBlock(nn.Module):
    def __init__(self, inputs, scale):
        super().__init__()
        if scale == 1:
            self.features = [
                nn.Conv2d(inputs, 1, 1),
                nn.Identity(),
                nn.ConvTranspose2d(1, 1, 2, stride=2),
            ]
        else:
            self.features = [
                nn.Conv2d(inputs, 16, 1),
                nn.Identity(),
                nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
                nn.Conv2d(16, 1, 1),
                nn.Identity(),
                nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1),
            ]

    def __call__(self, value):
        for index, layer in enumerate(self.features):
            value = _smish(value) if index in {1, 4} else layer(value)
        return value


class _DoubleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.DWconv1 = nn.Conv2d(3, 24, 3, padding=1, groups=3)
        self.DWconv2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)

    def __call__(self, value):
        attention = self.DWconv1(_smish(value))
        attention_2 = self.DWconv2(_smish(attention))
        return _smish(mx.sum(attention + attention_2, axis=-1, keepdims=True))


class TEED(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = _DoubleConv(3, 16, 16, stride=2)
        self.block_2 = _DoubleConv(16, 32, 32, final_activation=False)
        self.dblock_3 = _DenseBlock()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.side_1 = _SingleConv(16, 32, 2)
        self.pre_dense_3 = _SingleConv(32, 48, 1)
        self.up_block_1 = _UpBlock(16, 1)
        self.up_block_2 = _UpBlock(32, 1)
        self.up_block_3 = _UpBlock(48, 2)
        self.block_cat = _DoubleFusion()

    def __call__(self, value):
        block_1 = self.block_1(value)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_3 = self.dblock_3(block_2_down + self.side_1(block_1), self.pre_dense_3(block_2_down))
        outputs = (
            self.up_block_1(block_1),
            self.up_block_2(block_2),
            self.up_block_3(block_3),
        )
        return (*outputs, self.block_cat(mx.concatenate(outputs, axis=-1)))


def load_teed(path: str | Path):
    model = TEED()
    expected = dict(tree_flatten(model.parameters()))
    source = load_file(path)
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise ValueError(f"TEED checkpoint mismatch: {len(missing)} missing, {len(extra)} extra.")
    model.load_weights([(name, mx.array(source[name])) for name in expected], strict=True)
    return model


@dataclass(frozen=True)
class TEEDConfig:
    safe_steps: int = 2
    frame_chunk_size: int = 8

    def validate(self):
        if self.safe_steps not in range(0, 11):
            raise ValueError("TEED safe steps must be between 0 and 10.")
        if self.frame_chunk_size not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("TEED frame chunk size must be 1, 2, 4, 8, 16, or 32.")


def _prepare(images: Any):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    value = np.asarray(images)
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError("TEED requires a ComfyUI IMAGE frame batch.")
    value = value[..., :3]
    if value.dtype != np.uint8:
        value = np.clip(value.astype(np.float32), 0, 1) * 255
    return np.ascontiguousarray(value.astype(np.float32))


def infer_teed(
    images,
    model,
    config: TEEDConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or TEEDConfig()
    config.validate()
    source = _prepare(images)
    height, width = source.shape[1:3]
    target_h = ((height + 7) // 8) * 8
    target_w = ((width + 7) // 8) * 8
    output = []
    for start in range(0, len(source), config.frame_chunk_size):
        if interruption_callback is not None:
            interruption_callback()
        value = mx.array(source[start : start + config.frame_chunk_size])
        if value.shape[1:3] != (target_h, target_w):
            value = _resize(value, target_h, target_w, mode="cubic")
        logits = model(value)
        resized = [_resize(item, height, width) for item in logits]
        edge = mx.sigmoid(mx.mean(mx.concatenate(resized, axis=-1), axis=-1, keepdims=True))
        if config.safe_steps:
            edge = mx.floor(edge * (config.safe_steps + 1)) / config.safe_steps
        edge = mx.clip(edge, 0, 1)
        edge = mx.repeat(edge, 3, axis=-1)
        mx.eval(edge)
        output.append(np.asarray(edge, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(min(start + config.frame_chunk_size, len(source)), len(source))
    return np.concatenate(output), {
        "backend": "mlx",
        "algorithm": "teed_soft_edge",
        "frames": len(source),
        "safe_steps": config.safe_steps,
        "frame_chunk_size": config.frame_chunk_size,
    }
