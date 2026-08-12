"""Small MLX-native decoder for live MiniMax H3 latent previews.

The decoder consumes the 24-channel normalized video latents produced by the H3 sampler.  It is
deliberately isolated from the full video VAE: previewing must not make the transformer coexist
with the multi-gigabyte production decoder in unified memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .hires_fix import resize_video_latents_bilinear


@dataclass(frozen=True)
class PreviewStatistics:
    finite: bool
    luminance_std: float
    luminance_range_02_98: float
    spatial_gradient: float

    @property
    def collapsed(self) -> bool:
        """Conservative signal for featureless output, evaluated over several checkpoints."""
        # Failed H3 decodes can retain a broad brightness gradient even when they contain no
        # recognizable spatial structure. Requiring both low contrast and low local structure
        # catches that failure while preserving flat backgrounds with a sharp subject or edge.
        return self.luminance_std < 0.090 and self.spatial_gradient < 0.006


def preview_statistics(frames: np.ndarray) -> PreviewStatistics:
    """Measure cheap structural signals on ``(frames, height, width, 3)`` RGB in ``[0, 1]``."""
    values = np.asarray(frames, dtype=np.float32)
    finite = bool(np.isfinite(values).all())
    if not finite or values.size == 0:
        return PreviewStatistics(finite, 0.0, 0.0, 0.0)
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    dy = np.abs(np.diff(luminance, axis=1)).mean() if luminance.shape[1] > 1 else 0.0
    dx = np.abs(np.diff(luminance, axis=2)).mean() if luminance.shape[2] > 1 else 0.0
    return PreviewStatistics(
        finite=True,
        luminance_std=float(luminance.std()),
        luminance_range_02_98=float(np.quantile(luminance, 0.98) - np.quantile(luminance, 0.02)),
        spatial_gradient=float((dx + dy) * 0.5),
    )


class H3TinyPreviewDecoder:
    """Inference-only MLX implementation of the compact H3 TAE decoder architecture."""

    latent_channels = 24
    spatial_scale = 16
    temporal_scale = 4

    def __init__(self, weights: dict[str, mx.array]):
        self.weights = {
            name: tensor for name, tensor in weights.items() if name.startswith("decoder.")
        }
        required = {
            "decoder.1.weight",
            "decoder.22.weight",
            "decoder.7.conv.weight",
            "decoder.13.conv.weight",
            "decoder.19.conv.weight",
        }
        missing = sorted(required.difference(self.weights))
        if missing:
            raise ValueError(f"H3 preview TAE is missing decoder tensors: {missing}")
        if tuple(self.weights["decoder.1.weight"].shape[:2]) != (256, self.latent_channels):
            raise ValueError("H3 preview TAE must accept exactly 24 latent channels.")

    @classmethod
    def from_safetensors(cls, path: str | Path) -> H3TinyPreviewDecoder:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"H3 preview TAE not found: {resolved}")
        return cls(dict(mx.load(str(resolved))))

    def release(self) -> None:
        self.weights.clear()

    def _conv(self, x: mx.array, key: str, *, stride: int = 1) -> mx.array:
        weight = self.weights[f"{key}.weight"].transpose(0, 2, 3, 1)
        bias = self.weights.get(f"{key}.bias")
        output = mx.conv2d(x, weight, stride=stride, padding=1)
        return output + bias if bias is not None else output

    def _conv1x1(self, x: mx.array, key: str) -> mx.array:
        weight = self.weights[f"{key}.weight"].transpose(0, 2, 3, 1)
        bias = self.weights.get(f"{key}.bias")
        output = mx.conv2d(x, weight)
        return output + bias if bias is not None else output

    def _memblock(self, x: mx.array, index: int) -> mx.array:
        past = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        y = mx.concatenate([x, past], axis=-1)
        batch, frames, height, width, channels = y.shape
        y = y.reshape(batch * frames, height, width, channels)
        y = mx.maximum(self._conv(y, f"decoder.{index}.conv.0"), 0)
        y = mx.maximum(self._conv(y, f"decoder.{index}.conv.2"), 0)
        y = self._conv(y, f"decoder.{index}.conv.4")
        y = y.reshape(batch, frames, height, width, -1)
        return mx.maximum(y + x, 0)

    @staticmethod
    def _spatial_grow(x: mx.array) -> mx.array:
        return mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)

    def _temporal_grow(self, x: mx.array, index: int, stride: int) -> mx.array:
        batch, frames, height, width, channels = x.shape
        y = self._conv1x1(
            x.reshape(batch * frames, height, width, channels),
            f"decoder.{index}.conv",
        )
        y = y.reshape(batch, frames, height, width, stride, channels)
        y = y.transpose(0, 1, 4, 2, 3, 5)
        return y.reshape(batch, frames * stride, height, width, channels)

    @staticmethod
    def _pixel_shuffle_2(x: mx.array) -> mx.array:
        batch, frames, height, width, channels = x.shape
        if channels != 12:
            raise ValueError(f"H3 preview TAE expected 12 output channels, got {channels}.")
        x = x.reshape(batch, frames, height, width, 3, 2, 2)
        x = x.transpose(0, 1, 2, 5, 3, 6, 4)
        return x.reshape(batch, frames, height * 2, width * 2, 3)

    def decode(self, latents: mx.array, *, max_edge: int = 384) -> np.ndarray:
        """Decode a compact true-color preview and return host RGB frames in ``[0, 1]``."""
        if latents.ndim != 5 or int(latents.shape[1]) != self.latent_channels:
            raise ValueError(
                "H3 preview latents must have shape (batch, 24, frames, height, width)."
            )
        if max_edge < 64:
            raise ValueError("H3 preview max_edge must be at least 64 pixels.")
        target_latent_edge = max(4, max_edge // self.spatial_scale)
        source_height, source_width = int(latents.shape[3]), int(latents.shape[4])
        scale = min(1.0, target_latent_edge / max(source_height, source_width))
        target_height = max(2, int(round(source_height * scale)))
        target_width = max(2, int(round(source_width * scale)))
        working = resize_video_latents_bilinear(
            latents.astype(mx.float16), target_height, target_width
        )
        x = working.transpose(0, 2, 3, 4, 1)
        x = mx.tanh(x / 3.0) * 3.0
        batch, frames, height, width, channels = x.shape
        x = self._conv(x.reshape(batch * frames, height, width, channels), "decoder.1").reshape(
            batch, frames, height, width, -1
        )
        x = mx.maximum(x, 0)

        for index in (3, 4, 5):
            x = self._memblock(x, index)
        x = self._spatial_grow(x)
        x = self._temporal_grow(x, 7, 1)
        batch, frames, height, width, channels = x.shape
        x = self._conv(x.reshape(batch * frames, height, width, channels), "decoder.8").reshape(
            batch, frames, height, width, -1
        )

        for index in (9, 10, 11):
            x = self._memblock(x, index)
        x = self._spatial_grow(x)
        x = self._temporal_grow(x, 13, 2)
        batch, frames, height, width, channels = x.shape
        x = self._conv(x.reshape(batch * frames, height, width, channels), "decoder.14").reshape(
            batch, frames, height, width, -1
        )

        for index in (15, 16, 17):
            x = self._memblock(x, index)
        x = self._spatial_grow(x)
        x = self._temporal_grow(x, 19, 2)
        batch, frames, height, width, channels = x.shape
        x = self._conv(x.reshape(batch * frames, height, width, channels), "decoder.20")
        x = mx.maximum(x, 0)
        x = self._conv(x, "decoder.22")
        x = x.reshape(batch, frames, height, width, -1)
        x = self._pixel_shuffle_2(x)

        # H3 decodes five latent tokens per temporal chunk. Each chunk has a three-frame causal
        # prefix, and the final three latent tokens are padding. Previewing ten latent tokens
        # therefore yields two useful temporal blocks without running the full production VAE.
        chunk = 5 * self.temporal_scale
        padding = (-int(x.shape[1])) % chunk
        if padding:
            x = mx.concatenate([x, mx.zeros_like(x[:, :padding])], axis=1)
        x = x.reshape(batch, -1, chunk, *x.shape[2:])[:, :, 3:]
        x = x.reshape(batch, -1, *x.shape[3:])
        tail = 3 * self.temporal_scale
        if int(x.shape[1]) > tail:
            x = x[:, :-tail]
        x = mx.clip(x, 0.0, 1.0)
        mx.eval(x)
        return np.asarray(x[0], dtype=np.float32)
