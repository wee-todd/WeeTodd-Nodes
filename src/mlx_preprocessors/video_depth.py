"""Independent MLX implementation of Video Depth Anything Small."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

INFER_LENGTH = 32
OVERLAP = 10
KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
INTERPOLATION_LENGTH = 8


def _resize(value, height: int, width: int, *, mode="linear", align_corners=False):
    source_h, source_w = int(value.shape[1]), int(value.shape[2])
    if (source_h, source_w) == (height, width):
        return value
    # MLX derives the output extent with floor(source * scale). Roundoff in an exact
    # target/source ratio can otherwise remove the final row or column.
    scale_h = np.nextafter(height / source_h, np.inf)
    scale_w = np.nextafter(width / source_w, np.inf)
    output = nn.Upsample(
        (scale_h, scale_w), mode=mode, align_corners=align_corners
    )(value)
    if output.shape[1:3] != (height, width):
        raise RuntimeError(
            f"MLX resize produced {output.shape[1]}x{output.shape[2]}; "
            f"expected {height}x{width}."
        )
    return output


def _gelu(value):
    return value * 0.5 * (1.0 + mx.erf(value / sqrt(2.0)))


def _cubic_coefficient(distance):
    absolute = np.abs(distance)
    first = ((-0.75 + 2.0) * absolute - (-0.75 + 3.0)) * absolute * absolute + 1.0
    second = (
        ((-0.75 * absolute - 5.0 * -0.75) * absolute + 8.0 * -0.75) * absolute
        - 4.0 * -0.75
    )
    return np.where(absolute <= 1.0, first, np.where(absolute < 2.0, second, 0.0))


def _pytorch_bicubic_position(value, height: int, width: int):
    """Match the DINOv2 scale-factor bicubic path used by PyTorch."""
    source_h, source_w = int(value.shape[1]), int(value.shape[2])
    if (source_h, source_w) == (height, width):
        return value

    def samples(source_size, target_size):
        scale_factor = (target_size + 0.1) / source_size
        coordinates = (np.arange(target_size, dtype=np.float32) + 0.5) / scale_factor - 0.5
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


class _DinoAttention(nn.Module):
    def __init__(self, width=384, heads=6):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3, bias=True)
        self.proj = nn.Linear(width, width, bias=True)

    def __call__(self, value):
        batch, tokens, width = value.shape
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.heads, width // self.heads)
        query, key, val = (mx.transpose(qkv[:, :, index], (0, 2, 1, 3)) for index in range(3))
        attended = mx.fast.scaled_dot_product_attention(
            query, key, val, scale=(width // self.heads) ** -0.5
        )
        return self.proj(mx.transpose(attended, (0, 2, 1, 3)).reshape(batch, tokens, width))


class _DinoMLP(nn.Module):
    def __init__(self, width=384):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, value):
        return self.fc2(_gelu(self.fc1(value)))


class _LayerScale(nn.Module):
    def __init__(self, width=384):
        super().__init__()
        self.gamma = mx.ones((width,))

    def __call__(self, value):
        return value * self.gamma


class _DinoBlock(nn.Module):
    def __init__(self, width=384, heads=6):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = _DinoAttention(width, heads)
        self.ls1 = _LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = _DinoMLP(width)
        self.ls2 = _LayerScale(width)

    def __call__(self, value):
        value = value + self.ls1(self.attn(self.norm1(value)))
        return value + self.ls2(self.mlp(self.norm2(value)))


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 384, 14, stride=14)

    def __call__(self, value):
        value = self.proj(value)
        return value.reshape(value.shape[0], -1, value.shape[-1])


class _DinoV2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.zeros((1, 1, 384))
        self.pos_embed = mx.zeros((1, 1370, 384))
        self.mask_token = mx.zeros((1, 384))
        self.patch_embed = _PatchEmbed()
        self.blocks = [_DinoBlock() for _ in range(12)]
        self.norm = nn.LayerNorm(384, eps=1e-6)

    def _position_embedding(self, patch_h: int, patch_w: int, dtype):
        class_position = self.pos_embed[:, :1]
        patch_position = self.pos_embed[:, 1:].reshape(1, 37, 37, 384).astype(mx.float32)
        patch_position = _pytorch_bicubic_position(patch_position, patch_h, patch_w)
        return mx.concatenate((class_position, patch_position.reshape(1, -1, 384)), axis=1).astype(
            dtype
        )

    def intermediate_layers(self, value, requested=(2, 5, 8, 11)):
        patch_h, patch_w = int(value.shape[1]) // 14, int(value.shape[2]) // 14
        value = self.patch_embed(value)
        class_tokens = mx.broadcast_to(self.cls_token, (value.shape[0], 1, value.shape[-1]))
        value = mx.concatenate((class_tokens, value), axis=1)
        value = value + self._position_embedding(patch_h, patch_w, value.dtype)
        outputs = []
        for index, block in enumerate(self.blocks):
            value = block(value)
            if index in requested:
                normalized = self.norm(value)
                outputs.append((normalized[:, 1:], normalized[:, 0]))
        return outputs


class _ResidualConvUnit(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)

    def __call__(self, value):
        output = self.conv1(mx.maximum(value, 0.0))
        output = self.conv2(mx.maximum(output, 0.0))
        return output + value


class _FeatureFusionBlock(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1)
        self.resConfUnit1 = _ResidualConvUnit(features)
        self.resConfUnit2 = _ResidualConvUnit(features)

    def __call__(self, value, residual=None, *, size=None):
        if residual is not None:
            value = value + self.resConfUnit1(residual)
        value = self.resConfUnit2(value)
        if size is None:
            size = (int(value.shape[1]) * 2, int(value.shape[2]) * 2)
        return self.out_conv(_resize(value, *size, mode="linear", align_corners=True))


class _Scratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1_rn = nn.Conv2d(48, 64, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(96, 64, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(192, 64, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(384, 64, 3, padding=1, bias=False)
        self.refinenet1 = _FeatureFusionBlock()
        self.refinenet2 = _FeatureFusionBlock()
        self.refinenet3 = _FeatureFusionBlock()
        self.refinenet4 = _FeatureFusionBlock()
        self.output_conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.output_conv2 = [nn.Conv2d(32, 32, 3, padding=1), nn.Identity(), nn.Conv2d(32, 1, 1)]


class _TemporalAttention(nn.Module):
    def __init__(self, width: int, heads=8):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.to_out = [nn.Linear(width, width), nn.Identity()]
        self.pos_encoder = _PositionalEncoding(width)

    def __call__(self, value, *, video_length: int):
        batch_frames, spatial, width = value.shape
        batch = batch_frames // video_length
        value = value.reshape(batch, video_length, spatial, width)
        value = mx.transpose(value, (0, 2, 1, 3)).reshape(batch * spatial, video_length, width)
        value = value + self.pos_encoder.pe[:, :video_length].astype(value.dtype)
        query = self.to_q(value)
        key = self.to_k(value)
        val = self.to_v(value)
        head_width = width // self.heads

        def heads(tensor):
            return mx.transpose(
                tensor.reshape(tensor.shape[0], video_length, self.heads, head_width),
                (0, 2, 1, 3),
            )

        output = mx.fast.scaled_dot_product_attention(
            heads(query), heads(key), heads(val), scale=head_width**-0.5
        )
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(
            batch * spatial, video_length, width
        )
        output = self.to_out[0](output)
        output = output.reshape(batch, spatial, video_length, width)
        return mx.transpose(output, (0, 2, 1, 3)).reshape(batch_frames, spatial, width)


class _PositionalEncoding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.pe = mx.zeros((1, 32, width))


class _GEGLU(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width * 8)

    def __call__(self, value):
        hidden, gate = mx.split(self.proj(value), 2, axis=-1)
        return hidden * _gelu(gate)


class _TemporalFeedForward(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = [_GEGLU(width), nn.Identity(), nn.Linear(width * 4, width)]

    def __call__(self, value):
        return self.net[2](self.net[0](value))


class _TemporalBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention_blocks = [_TemporalAttention(width), _TemporalAttention(width)]
        self.norms = [nn.LayerNorm(width), nn.LayerNorm(width)]
        self.ff = _TemporalFeedForward(width)
        self.ff_norm = nn.LayerNorm(width)

    def __call__(self, value, *, video_length: int):
        for attention, norm in zip(self.attention_blocks, self.norms, strict=True):
            value = value + attention(norm(value), video_length=video_length)
        return value + self.ff(self.ff_norm(value))


class _TemporalTransformer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = [_TemporalBlock(channels)]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, value):
        batch, frames, height, width, channels = value.shape
        flattened = value.reshape(batch * frames, height, width, channels)
        residual = flattened
        hidden = self.norm(flattened).reshape(batch * frames, height * width, channels)
        hidden = self.proj_in(hidden)
        hidden = self.transformer_blocks[0](hidden, video_length=frames)
        hidden = self.proj_out(hidden).reshape(batch * frames, height, width, channels)
        return (hidden + residual).reshape(batch, frames, height, width, channels)


class _TemporalModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.temporal_transformer = _TemporalTransformer(channels)

    def __call__(self, value):
        return self.temporal_transformer(value)


class _DPTHeadTemporal(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.scratch = _Scratch()
        self.motion_modules = [
            _TemporalModule(192),
            _TemporalModule(384),
            _TemporalModule(64),
            _TemporalModule(64),
        ]

    def __call__(
        self,
        features,
        patch_h: int,
        patch_w: int,
        frames: int,
        decoder_chunk_size: int,
    ):
        projected = []
        for index, (patch_tokens, _class_token) in enumerate(features):
            value = patch_tokens.reshape(-1, patch_h, patch_w, 384)
            projected.append(self.resize_layers[index](self.projects[index](value)))
        layer_1, layer_2, layer_3, layer_4 = projected
        batch = int(layer_1.shape[0]) // frames

        def temporal(value, module):
            shape = value.shape
            return module(value.reshape(batch, frames, *shape[1:])).reshape(shape)

        layer_3 = temporal(layer_3, self.motion_modules[0])
        layer_4 = temporal(layer_4, self.motion_modules[1])
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[1:3])
        path_4 = temporal(path_4, self.motion_modules[2])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[1:3])
        path_3 = temporal(path_3, self.motion_modules[3])
        outputs = []
        for start in range(0, int(path_3.shape[0]), decoder_chunk_size):
            stop = min(int(path_3.shape[0]), start + decoder_chunk_size)
            path_2 = self.scratch.refinenet2(
                path_3[start:stop],
                layer_2_rn[start:stop],
                size=layer_1_rn.shape[1:3],
            )
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn[start:stop])
            output = self.scratch.output_conv1(path_1)
            output = _resize(
                output, patch_h * 14, patch_w * 14, mode="linear", align_corners=True
            )
            output = self.scratch.output_conv2[0](output.astype(mx.float32))
            output = self.scratch.output_conv2[2](mx.maximum(output, 0.0))
            output = mx.maximum(output, 0.0)
            mx.eval(output)
            outputs.append(output)
        return mx.concatenate(outputs, axis=0)


class VideoDepthAnythingSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _DinoV2Small()
        self.head = _DPTHeadTemporal()

    def __call__(self, value, *, encoder_chunk_size=4, decoder_chunk_size=4):
        batch, frames, height, width, channels = value.shape
        if channels != 3 or height % 14 or width % 14:
            raise ValueError("Video Depth Anything input must be RGB and divisible by 14.")
        flattened = value.reshape(batch * frames, height, width, channels)
        feature_chunks = []
        for start in range(0, int(flattened.shape[0]), encoder_chunk_size):
            chunk = self.pretrained.intermediate_layers(
                flattened[start : start + encoder_chunk_size]
            )
            mx.eval(*[value for pair in chunk for value in pair])
            feature_chunks.append(chunk)
        features = []
        for layer_index in range(4):
            features.append(
                (
                    mx.concatenate(
                        [chunk[layer_index][0] for chunk in feature_chunks], axis=0
                    ),
                    mx.concatenate(
                        [chunk[layer_index][1] for chunk in feature_chunks], axis=0
                    ),
                )
            )
        depth = self.head(
            features,
            height // 14,
            width // 14,
            frames,
            decoder_chunk_size,
        )
        return depth.reshape(batch, frames, height, width)


def _convert_weight(name: str, value, expected_shape):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == tuple(expected_shape):
        return np.ascontiguousarray(array)
    if array.ndim == 4:
        if name in {"head.resize_layers.0.weight", "head.resize_layers.1.weight"}:
            candidate = np.transpose(array, (1, 2, 3, 0))
        else:
            candidate = np.transpose(array, (0, 2, 3, 1))
        if candidate.shape == tuple(expected_shape):
            return np.ascontiguousarray(candidate)
    raise ValueError(
        f"Video Depth Anything tensor {name!r} has shape {array.shape}; "
        f"expected {tuple(expected_shape)}."
    )


def load_video_depth_anything_small(path: str | Path):
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(
            "The MLX Video Depth Anything loader requires a converted .safetensors checkpoint. "
            "Run scripts/convert_video_depth_anything_mlx.py once in the ComfyUI environment."
        )
    from safetensors.numpy import load_file

    model = VideoDepthAnythingSmall()
    expected = dict(tree_flatten(model.parameters()))
    source = load_file(path)
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Video Depth Anything checkpoint mismatch: {len(missing)} missing and "
            f"{len(extra)} extra tensors."
        )
    weights = [
        (name, mx.array(_convert_weight(name, source[name], value.shape)))
        for name, value in expected.items()
    ]
    model.load_weights(weights, strict=True)
    return model


@dataclass(frozen=True)
class VideoDepthConfig:
    input_size: int = 518
    output_invert: bool = False
    encoder_chunk_size: int = 4
    decoder_chunk_size: int = 4

    def validate(self):
        if self.input_size not in {392, 448, 518, 560, 644}:
            raise ValueError("Video depth input size must be 392, 448, 518, 560, or 644.")
        if self.encoder_chunk_size not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("Video depth encoder chunk size must be 1, 2, 4, 8, 16, or 32.")
        if self.decoder_chunk_size not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("Video depth decoder chunk size must be 1, 2, 4, 8, 16, or 32.")


def _network_size(height: int, width: int, input_size: int):
    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:
        input_size = round((input_size * 1.777 / ratio) / 14) * 14
    scale = max(input_size / width, input_size / height)
    target_h = max(input_size, round((height * scale) / 14) * 14)
    target_w = max(input_size, round((width * scale) / 14) * 14)
    return int(target_h), int(target_w)


def _prepare_frames(images: Any, input_size: int):
    detach = getattr(images, "detach", None)
    if detach is not None:
        images = detach()
    cpu = getattr(images, "cpu", None)
    if cpu is not None:
        images = cpu()
    source = np.asarray(images, dtype=np.float32)
    if source.ndim != 4 or source.shape[-1] < 3:
        raise ValueError("Video depth requires a ComfyUI IMAGE frame batch.")
    source = np.ascontiguousarray(np.clip(source[..., :3], 0.0, 1.0))
    target_h, target_w = _network_size(source.shape[1], source.shape[2], input_size)
    value = _resize(mx.array(source), target_h, target_w, mode="cubic")
    mean = mx.array([0.485, 0.456, 0.406])
    standard_deviation = mx.array([0.229, 0.224, 0.225])
    return source, (value - mean) / standard_deviation


def _scale_and_shift(prediction, target):
    prediction = np.concatenate(prediction).astype(np.float64)
    target = np.concatenate(target).astype(np.float64)
    a00 = np.sum(prediction * prediction)
    a01 = np.sum(prediction)
    a11 = prediction.size
    b0 = np.sum(prediction * target)
    b1 = np.sum(target)
    determinant = a00 * a11 - a01 * a01
    if abs(determinant) < 1e-12:
        return 1.0, 0.0
    return (a11 * b0 - a01 * b1) / determinant, (-a01 * b0 + a00 * b1) / determinant


def infer_video_depth(
    images: Any,
    model: VideoDepthAnythingSmall,
    config: VideoDepthConfig | None = None,
    *,
    progress_callback=None,
    interruption_callback=None,
):
    config = config or VideoDepthConfig()
    config.validate()
    source, prepared = _prepare_frames(images, config.input_size)
    parameter_dtype = tree_flatten(model.parameters())[0][1].dtype
    prepared = prepared.astype(parameter_dtype)
    original_frames = int(source.shape[0])
    step = INFER_LENGTH - OVERLAP
    append = (step - (original_frames % step)) % step + (INFER_LENGTH - step)
    prepared = mx.concatenate(
        (prepared, mx.repeat(prepared[-1:], append, axis=0)), axis=0
    )
    depth_windows = []
    previous_input = None
    for frame_index in range(0, original_frames, step):
        if interruption_callback is not None:
            interruption_callback()
        window = prepared[frame_index : frame_index + INFER_LENGTH]
        if previous_input is not None:
            window = mx.concatenate(
                (mx.take(previous_input, mx.array(KEYFRAMES), axis=0), window[OVERLAP:]), axis=0
            )
        depth = model(
            window[None],
            encoder_chunk_size=config.encoder_chunk_size,
            decoder_chunk_size=config.decoder_chunk_size,
        )[0]
        depth = _resize(
            depth[..., None], source.shape[1], source.shape[2], align_corners=True
        )[..., 0]
        mx.eval(depth)
        depth_windows.append(np.asarray(depth, dtype=np.float32))
        previous_input = window
        if progress_callback is not None:
            progress_callback(min(frame_index + step, original_frames), original_frames)

    aligned = []
    reference = []
    alignment_length = OVERLAP - INTERPOLATION_LENGTH
    alignment_keyframes = KEYFRAMES[:alignment_length]
    for window_index, depth in enumerate(depth_windows):
        if window_index == 0:
            aligned.extend(depth)
            reference = [depth[index] for index in alignment_keyframes]
            continue
        current = [depth[index] for index in range(alignment_length)]
        scale, shift = _scale_and_shift(current, reference)
        post = np.maximum(depth[alignment_length:OVERLAP] * scale + shift, 0.0)
        pre = np.asarray(aligned[-INTERPOLATION_LENGTH:])
        weights = np.linspace(0.0, 1.0, INTERPOLATION_LENGTH + 2, dtype=np.float32)[1:-1]
        blended = pre * (1.0 - weights[:, None, None]) + post * weights[:, None, None]
        aligned[-INTERPOLATION_LENGTH:] = list(blended)
        aligned.extend(np.maximum(depth[OVERLAP:] * scale + shift, 0.0))
        reference = [
            reference[0],
            *[
                np.maximum(depth[index] * scale + shift, 0.0)
                for index in alignment_keyframes[1:]
            ],
        ]

    result = np.asarray(aligned[:original_frames], dtype=np.float32)
    minimum = float(result.min())
    maximum = float(result.max())
    result = (result - minimum) / max(maximum - minimum, 1e-6)
    if config.output_invert:
        result = 1.0 - result
    return np.repeat(result[..., None], 3, axis=-1), {
        "backend": "mlx",
        "algorithm": "video_depth_anything_small",
        "frames": original_frames,
        "input_size": config.input_size,
        "network_height": int(prepared.shape[1]),
        "network_width": int(prepared.shape[2]),
        "windows": len(depth_windows),
        "encoder_chunk_size": config.encoder_chunk_size,
        "decoder_chunk_size": config.decoder_chunk_size,
        "output_invert": config.output_invert,
    }
