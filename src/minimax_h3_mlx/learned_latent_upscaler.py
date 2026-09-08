"""MLX-native learned spatial upscaling for MiniMax H3 video latents.

The supported checkpoint is a scale-conditioned 3D convolutional network over H3's 24-channel
video latent.  Audio is deliberately outside this module: the ComfyUI adapter keeps the original
audio latent unchanged through the visual refinement stage.
"""

from __future__ import annotations

import gc
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .hires_fix import resize_video_latents_bilinear

H3_LATENT_CHANNELS = 24
H3_LATENT_UPSCALER_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
H3_LATENT_UPSCALER_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


class _DepthwiseTemporalConv3d(nn.Module):
    """Depthwise temporal convolution for NDHWC tensors.

    MLX 0.32 exposes the ``groups`` argument on ``conv3d`` but currently implements only one
    group.  A five-tap temporal depthwise convolution is cheaper and clearer as five shifted,
    channel-wise multiply-adds.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.weight = mx.zeros((self.kernel_size, channels))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        radius = self.kernel_size // 2
        padded = mx.pad(x, ((0, 0), (radius, radius), (0, 0), (0, 0), (0, 0)))
        result = self.bias
        frames = int(x.shape[1])
        for offset in range(self.kernel_size):
            result = result + padded[:, offset : offset + frames] * self.weight[offset]
        return result


class _ResidualBlock3d(nn.Module):
    def __init__(self, channels: int, embedding_channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            32, channels, eps=1e-5, affine=True, pytorch_compatible=True
        )
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.emb = nn.Linear(embedding_channels, channels * 2)
        self.norm2 = nn.GroupNorm(
            32, channels, eps=1e-5, affine=True, pytorch_compatible=True
        )
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array, embedding: mx.array) -> mx.array:
        hidden = self.conv1(nn.silu(self.norm1(x)))
        modulation = self.emb(nn.silu(embedding)).astype(hidden.dtype)
        scale, shift = mx.split(modulation, 2, axis=-1)
        hidden = self.norm2(hidden) * (1.0 + scale[:, None, None, None, :])
        hidden = hidden + shift[:, None, None, None, :]
        return x + self.conv2(nn.silu(hidden))


class _TemporalBlock3d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.norm = nn.GroupNorm(
            32, channels, eps=1e-5, affine=True, pytorch_compatible=True
        )
        self.dwconv = _DepthwiseTemporalConv3d(channels, kernel_size)
        self.pwconv = nn.Conv3d(channels, channels, 1)

    def __call__(self, x: mx.array, _embedding: mx.array) -> mx.array:
        return x + self.pwconv(self.dwconv(nn.silu(self.norm(x))))


class H3LearnedLatentUpscaler(nn.Module):
    """Scale-conditioned learned H3 latent upscaler operating on NDHWC internally."""

    def __init__(
        self,
        *,
        channels: int,
        input_channels: int,
        input_block_kinds: tuple[str, ...],
        output_block_kinds: tuple[str, ...],
        temporal_kernel: int,
        embedding_channels: int = 64,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(input_channels, channels, 3, padding=1)
        self.embed = [
            nn.Linear(1, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        ]
        self.in_blocks = [
            (
                _ResidualBlock3d(channels, embedding_channels)
                if kind == "residual"
                else _TemporalBlock3d(channels, temporal_kernel)
            )
            for kind in input_block_kinds
        ]
        self.out_blocks = [
            (
                _ResidualBlock3d(channels, embedding_channels)
                if kind == "residual"
                else _TemporalBlock3d(channels, temporal_kernel)
            )
            for kind in output_block_kinds
        ]
        self.norm_out = nn.GroupNorm(
            32, channels, eps=1e-5, affine=True, pytorch_compatible=True
        )
        self.conv_out = nn.Conv3d(channels, input_channels, 3, padding=1)

    @staticmethod
    def _resize_spatial(x: mx.array, target_height: int, target_width: int) -> mx.array:
        # Reuse the tested half-pixel bilinear implementation. Time is never interpolated.
        channels_first = x.transpose(0, 4, 1, 2, 3)
        resized = resize_video_latents_bilinear(
            channels_first, target_height, target_width
        )
        return resized.transpose(0, 2, 3, 4, 1)

    def __call__(
        self,
        x: mx.array,
        *,
        scale: float,
        target_height: int,
        target_width: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> mx.array:
        embedding = mx.array([[float(scale) - 1.0]], dtype=x.dtype)
        embedding = self.embed[2](self.embed[1](self.embed[0](embedding)))
        total = len(self.in_blocks) + len(self.out_blocks) + 3
        completed = 0

        def materialize(value: mx.array) -> mx.array:
            nonlocal completed
            mx.eval(value)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
            return value

        x = materialize(self.conv_in(x))
        for block in self.in_blocks:
            x = materialize(block(x, embedding))
        x = materialize(self._resize_spatial(x, target_height, target_width))
        for block in self.out_blocks:
            x = materialize(block(x, embedding))
        return materialize(self.conv_out(nn.silu(self.norm_out(x))))


def _block_kinds(weights: Mapping[str, mx.array], prefix: str) -> tuple[str, ...]:
    indices = sorted(
        {
            int(match.group(1))
            for key in weights
            if (match := re.match(rf"{re.escape(prefix)}\.(\d+)\.", key))
        }
    )
    if indices != list(range(len(indices))):
        raise ValueError(f"H3 learned latent upscaler has sparse {prefix} indices: {indices}")
    return tuple(
        "temporal"
        if f"{prefix}.{index}.dwconv.weight" in weights
        else "residual"
        for index in indices
    )


def _map_checkpoint_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    mapped: dict[str, mx.array] = {}
    replacements = (
        (".in_layers.0.", ".norm1."),
        (".in_layers.2.", ".conv1."),
        (".emb_layers.1.", ".emb."),
        (".out_norm.", ".norm2."),
        (".out_layers.2.", ".conv2."),
    )
    for original_key, value in weights.items():
        key = original_key.removeprefix("upscaler.")
        for old, new in replacements:
            key = key.replace(old, new)
        if key.endswith(".dwconv.weight"):
            if value.ndim != 5 or tuple(value.shape[1:])[:1] != (1,):
                raise ValueError(f"Unexpected H3 temporal depthwise weight: {value.shape}")
            value = value[:, 0, :, 0, 0].transpose(1, 0)
        elif value.ndim == 5:
            # PyTorch OI(DHW) -> MLX O(DHW)I.
            value = value.transpose(0, 2, 3, 4, 1)
        mapped[key] = value
    return mapped


def load_h3_learned_latent_upscaler(
    checkpoint: str | Path,
) -> tuple[H3LearnedLatentUpscaler, dict[str, int | str]]:
    """Load a BF16/FP16 H3 latent-upscaler SafeTensors file directly into MLX."""
    source = Path(checkpoint).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"H3 learned latent upscaler not found: {source}")
    raw = dict(mx.load(str(source)))
    try:
        conv_in = raw.get("conv_in.weight")
        if conv_in is None:
            conv_in = raw.get("upscaler.conv_in.weight")
        if conv_in is None or conv_in.ndim != 5:
            raise ValueError("Checkpoint does not contain an H3 3D latent-upscaler conv_in.")
        channels = int(conv_in.shape[0])
        input_channels = int(conv_in.shape[1])
        if input_channels != H3_LATENT_CHANNELS:
            raise ValueError(
                f"H3 learned latent upscaler requires 24 channels; checkpoint has {input_channels}."
            )
        unprefixed = {
            key.removeprefix("upscaler."): value for key, value in raw.items()
        }
        temporal = next(
            (value for key, value in unprefixed.items() if key.endswith("dwconv.weight")),
            None,
        )
        temporal_kernel = int(temporal.shape[2]) if temporal is not None else 5
        model = H3LearnedLatentUpscaler(
            channels=channels,
            input_channels=input_channels,
            input_block_kinds=_block_kinds(unprefixed, "in_blocks"),
            output_block_kinds=_block_kinds(unprefixed, "out_blocks"),
            temporal_kernel=temporal_kernel,
        )
        mapped = _map_checkpoint_weights(raw)
        model.load_weights(list(mapped.items()), strict=True)
        mx.eval(model.parameters())
        resident_bytes = sum(value.nbytes for _, value in tree_flatten(model.parameters()))
        report: dict[str, int | str] = {
            "checkpoint": source.name,
            "tensor_count": len(mapped),
            "channels": channels,
            "input_channels": input_channels,
            "resident_bytes": resident_bytes,
        }
        return model, report
    finally:
        del raw
        if "mapped" in locals():
            del mapped
        gc.collect()


def upscale_h3_video_latents_learned(
    latents: mx.array,
    target_height: int,
    target_width: int,
    checkpoint: str | Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[mx.array, dict[str, int | float | str | bool]]:
    """Run the learned model and return only the enlarged H3 video latent."""
    source = Path(checkpoint).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"H3 learned latent upscaler not found: {source}")
    if latents.ndim != 5 or int(latents.shape[1]) != H3_LATENT_CHANNELS:
        raise ValueError("Learned H3 upscaling requires (B,24,T,H,W) video latents.")
    source_height = int(latents.shape[3])
    source_width = int(latents.shape[4])
    if target_height <= source_height or target_width <= source_width:
        raise ValueError("Learned H3 upscaling target dimensions must exceed the source.")
    scale = ((target_height * target_width) / (source_height * source_width)) ** 0.5
    if not 1.0 < scale <= 4.0:
        raise ValueError("Learned H3 latent upscaler supports spatial scales above 1x through 4x.")

    model = None
    total_started = time.perf_counter()
    active_before = mx.get_active_memory()
    try:
        load_started = time.perf_counter()
        model, load_report = load_h3_learned_latent_upscaler(source)
        load_seconds = time.perf_counter() - load_started
        dtype = latents.dtype
        working = latents.transpose(0, 2, 3, 4, 1)
        mean = mx.array(H3_LATENT_UPSCALER_MEAN, dtype=working.dtype)
        std = mx.array(H3_LATENT_UPSCALER_STD, dtype=working.dtype)
        working = (working - mean) / std
        inference_started = time.perf_counter()
        output = model(
            working,
            scale=scale,
            target_height=target_height,
            target_width=target_width,
            progress_callback=progress_callback,
        )
        output = (output * std + mean).transpose(0, 4, 1, 2, 3).astype(dtype)
        mx.eval(output)
        inference_seconds = time.perf_counter() - inference_started
        return output, {
            **load_report,
            "scale": scale,
            "source_height": source_height,
            "source_width": source_width,
            "target_height": target_height,
            "target_width": target_width,
            "unloaded_after_upscale": True,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - total_started,
            "active_memory_before_bytes": active_before,
            "active_memory_after_output_bytes": mx.get_active_memory(),
            "mlx_process_peak_bytes": mx.get_peak_memory(),
        }
    finally:
        if model is not None:
            del model
            gc.collect()
            mx.clear_cache()
