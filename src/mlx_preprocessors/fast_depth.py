"""MLX-native Depth Anything V2 Small for fast per-frame relative depth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import load_file

from .video_depth import (
    _convert_weight,
    _cubic_coefficient,
    _DinoV2Small,
    _FeatureFusionBlock,
    _network_size,
    _resize,
)


def _hf_bicubic_position(value, height: int, width: int):
    """Match torch interpolate(size=..., bicubic, align_corners=False)."""
    source_h, source_w = int(value.shape[1]), int(value.shape[2])

    def samples(source_size, target_size):
        coordinates = (np.arange(target_size, dtype=np.float32) + 0.5) * (
            source_size / target_size
        ) - 0.5
        base = np.floor(coordinates).astype(np.int32)
        offsets = np.arange(-1, 3, dtype=np.int32)
        indices = np.clip(base[:, None] + offsets[None], 0, source_size - 1)
        weights = _cubic_coefficient(coordinates[:, None] - (base[:, None] + offsets[None]))
        return mx.array(indices), mx.array(weights.astype(np.float32))

    y_indices, y_weights = samples(source_h, height)
    x_indices, x_weights = samples(source_w, width)
    output = mx.take(value, y_indices.reshape(-1), axis=1).reshape(
        value.shape[0], height, 4, source_w, value.shape[-1]
    )
    output = mx.sum(output * y_weights[None, :, :, None, None], axis=2)
    output = mx.take(output, x_indices.reshape(-1), axis=2).reshape(
        value.shape[0], height, width, 4, value.shape[-1]
    )
    return mx.sum(output * x_weights[None, None, :, :, None], axis=3)


class _HFDinoV2Small(_DinoV2Small):
    def _position_embedding(self, patch_h: int, patch_w: int, dtype):
        class_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:].reshape(1, 37, 37, 384).astype(mx.float32)
        patch_position = _hf_bicubic_position(patch_position, patch_h, patch_w)
        return mx.concatenate((class_position, patch_position.reshape(1, -1, 384)), axis=1).astype(
            dtype
        )


class DepthAnythingV2Small(nn.Module):
    """The 24.8M-parameter Apache-2.0 Depth Anything V2 Small model."""

    def __init__(self):
        super().__init__()
        self.backbone = _HFDinoV2Small()
        self.projects = [
            nn.Conv2d(384, 48, 1),
            nn.Conv2d(384, 96, 1),
            nn.Conv2d(384, 192, 1),
            nn.Conv2d(384, 384, 1),
        ]
        self.resize_layers = [
            nn.ConvTranspose2d(48, 48, 4, stride=4),
            nn.ConvTranspose2d(96, 96, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(384, 384, 3, stride=2, padding=1),
        ]
        self.neck_convs = [
            nn.Conv2d(48, 64, 3, padding=1, bias=False),
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.Conv2d(192, 64, 3, padding=1, bias=False),
            nn.Conv2d(384, 64, 3, padding=1, bias=False),
        ]
        self.fusion = [_FeatureFusionBlock() for _ in range(4)]
        self.head_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.head_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.head_conv3 = nn.Conv2d(32, 1, 1)

    def __call__(self, value):
        if value.ndim != 4 or value.shape[-1] != 3:
            raise ValueError("Depth Anything V2 requires an NHWC RGB frame batch.")
        height, width = int(value.shape[1]), int(value.shape[2])
        if height % 14 or width % 14:
            raise ValueError("Depth Anything V2 input dimensions must be divisible by 14.")
        features = self.backbone.intermediate_layers(value)
        patch_h, patch_w = height // 14, width // 14
        reassembled = []
        for index, (tokens, _class_token) in enumerate(features):
            feature = tokens.reshape(-1, patch_h, patch_w, 384)
            feature = self.resize_layers[index](self.projects[index](feature))
            reassembled.append(self.neck_convs[index](feature))

        fused = self.fusion[0](reassembled[3], size=reassembled[2].shape[1:3])
        fused = self.fusion[1](fused, reassembled[2], size=reassembled[1].shape[1:3])
        fused = self.fusion[2](fused, reassembled[1], size=reassembled[0].shape[1:3])
        fused = self.fusion[3](fused, reassembled[0])
        depth = self.head_conv1(fused)
        depth = _resize(depth, height, width, mode="linear", align_corners=True)
        depth = self.head_conv2(depth)
        depth = self.head_conv3(mx.maximum(depth, 0.0))
        return mx.maximum(depth[..., 0], 0.0)


def _source_mapping():
    mapping = {
        "backbone.cls_token": "backbone.embeddings.cls_token",
        "backbone.mask_token": "backbone.embeddings.mask_token",
        "backbone.pos_embed": "backbone.embeddings.position_embeddings",
        "backbone.patch_embed.proj.weight": (
            "backbone.embeddings.patch_embeddings.projection.weight"
        ),
        "backbone.patch_embed.proj.bias": "backbone.embeddings.patch_embeddings.projection.bias",
        "backbone.norm.weight": "backbone.layernorm.weight",
        "backbone.norm.bias": "backbone.layernorm.bias",
    }
    for index in range(12):
        target = f"backbone.blocks.{index}"
        source = f"backbone.encoder.layer.{index}"
        mapping.update(
            {
                f"{target}.norm1.weight": f"{source}.norm1.weight",
                f"{target}.norm1.bias": f"{source}.norm1.bias",
                f"{target}.attn.proj.weight": f"{source}.attention.output.dense.weight",
                f"{target}.attn.proj.bias": f"{source}.attention.output.dense.bias",
                f"{target}.ls1.gamma": f"{source}.layer_scale1.lambda1",
                f"{target}.norm2.weight": f"{source}.norm2.weight",
                f"{target}.norm2.bias": f"{source}.norm2.bias",
                f"{target}.mlp.fc1.weight": f"{source}.mlp.fc1.weight",
                f"{target}.mlp.fc1.bias": f"{source}.mlp.fc1.bias",
                f"{target}.mlp.fc2.weight": f"{source}.mlp.fc2.weight",
                f"{target}.mlp.fc2.bias": f"{source}.mlp.fc2.bias",
                f"{target}.ls2.gamma": f"{source}.layer_scale2.lambda1",
            }
        )
    for index in range(4):
        mapping[f"projects.{index}.weight"] = (
            f"neck.reassemble_stage.layers.{index}.projection.weight"
        )
        mapping[f"projects.{index}.bias"] = f"neck.reassemble_stage.layers.{index}.projection.bias"
        if index != 2:
            mapping[f"resize_layers.{index}.weight"] = (
                f"neck.reassemble_stage.layers.{index}.resize.weight"
            )
            mapping[f"resize_layers.{index}.bias"] = (
                f"neck.reassemble_stage.layers.{index}.resize.bias"
            )
        mapping[f"neck_convs.{index}.weight"] = f"neck.convs.{index}.weight"
        target = f"fusion.{index}"
        source = f"neck.fusion_stage.layers.{index}"
        mapping[f"{target}.out_conv.weight"] = f"{source}.projection.weight"
        mapping[f"{target}.out_conv.bias"] = f"{source}.projection.bias"
        for residual_index in (1, 2):
            for convolution_index in (1, 2):
                mapping[f"{target}.resConfUnit{residual_index}.conv{convolution_index}.weight"] = (
                    f"{source}.residual_layer{residual_index}.convolution{convolution_index}.weight"
                )
                mapping[f"{target}.resConfUnit{residual_index}.conv{convolution_index}.bias"] = (
                    f"{source}.residual_layer{residual_index}.convolution{convolution_index}.bias"
                )
    for index in (1, 2, 3):
        mapping[f"head_conv{index}.weight"] = f"head.conv{index}.weight"
        mapping[f"head_conv{index}.bias"] = f"head.conv{index}.bias"
    return mapping


def load_depth_anything_v2_small(path: str | Path):
    """Load the standard Hugging Face safetensors checkpoint directly into MLX."""
    path = Path(path)
    source = load_file(path)
    model = DepthAnythingV2Small()
    expected = dict(tree_flatten(model.parameters()))
    mapping = _source_mapping()
    weights = []
    for name, parameter in expected.items():
        if name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias"):
            prefix = name.removesuffix("qkv.weight").removesuffix("qkv.bias")
            block = prefix.split(".")[2]
            suffix = "weight" if name.endswith("weight") else "bias"
            source_prefix = f"backbone.encoder.layer.{block}.attention.attention"
            value = np.concatenate(
                [source[f"{source_prefix}.{part}.{suffix}"] for part in ("query", "key", "value")],
                axis=0,
            )
        else:
            source_name = mapping.get(name)
            if source_name is None:
                raise ValueError(f"Depth Anything V2 has no source mapping for {name!r}.")
            value = source[source_name]
        if name in {"resize_layers.0.weight", "resize_layers.1.weight"}:
            converted = np.transpose(np.asarray(value, dtype=np.float32), (1, 2, 3, 0))
            if converted.shape != tuple(parameter.shape):
                raise ValueError(
                    f"Depth Anything V2 transposed convolution {name!r} has shape "
                    f"{converted.shape}; expected {tuple(parameter.shape)}."
                )
        else:
            converted = _convert_weight(name, value, parameter.shape)
        weights.append((name, mx.array(converted)))
    model.load_weights(weights, strict=True)
    return model


@dataclass(frozen=True)
class FastDepthConfig:
    input_size: int = 392
    frame_chunk_size: int = 4
    output_invert: bool = False
    normalize: str = "per clip"

    def validate(self):
        if self.input_size not in {280, 336, 392, 448, 518}:
            raise ValueError("Fast depth input size must be 280, 336, 392, 448, or 518.")
        if self.frame_chunk_size not in {1, 2, 4, 8, 16}:
            raise ValueError("Fast depth frame chunk size must be 1, 2, 4, 8, or 16.")
        if self.normalize not in {"per clip", "per frame"}:
            raise ValueError("Fast depth normalization must be per clip or per frame.")


def _prepare(images: Any, input_size: int):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    source = np.asarray(images, dtype=np.float32)
    if source.ndim != 4 or source.shape[-1] < 3:
        raise ValueError("Fast depth requires a ComfyUI IMAGE frame batch.")
    source = np.ascontiguousarray(np.clip(source[..., :3], 0.0, 1.0))
    target_h, target_w = _network_size(source.shape[1], source.shape[2], input_size)
    return source, target_h, target_w


def infer_fast_depth(
    images,
    model: DepthAnythingV2Small,
    config: FastDepthConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or FastDepthConfig()
    config.validate()
    source, target_h, target_w = _prepare(images, config.input_size)
    parameter_dtype = tree_flatten(model.parameters())[0][1].dtype
    mean = mx.array([0.485, 0.456, 0.406])
    deviation = mx.array([0.229, 0.224, 0.225])
    chunks = []
    for start in range(0, len(source), config.frame_chunk_size):
        if interruption_callback is not None:
            interruption_callback()
        prepared = mx.array(source[start : start + config.frame_chunk_size])
        prepared = _resize(prepared, target_h, target_w, mode="cubic")
        prepared = ((prepared - mean) / deviation).astype(parameter_dtype)
        depth = model(prepared)
        depth = _resize(depth[..., None], source.shape[1], source.shape[2], align_corners=True)[
            ..., 0
        ]
        mx.eval(depth)
        chunks.append(np.asarray(depth, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(min(start + config.frame_chunk_size, len(source)), len(source))
    result = np.concatenate(chunks)
    if config.normalize == "per frame":
        minimum = result.min(axis=(1, 2), keepdims=True)
        maximum = result.max(axis=(1, 2), keepdims=True)
    else:
        minimum = result.min()
        maximum = result.max()
    result = (result - minimum) / np.maximum(maximum - minimum, 1e-6)
    if config.output_invert:
        result = 1.0 - result
    return np.repeat(result[..., None], 3, axis=-1).astype(np.float32), {
        "backend": "mlx",
        "algorithm": "depth_anything_v2_small",
        "frames": len(source),
        "input_size": config.input_size,
        "network_height": target_h,
        "network_width": target_w,
        "frame_chunk_size": config.frame_chunk_size,
        "normalization": config.normalize,
        "temporal_consistency": "independent frames",
    }
