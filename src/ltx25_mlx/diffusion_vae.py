"""Native MLX implementation of the LTX 2.5 one-step Diffusion VAE decoder.

The decoder is intentionally separate from the convolutional VAE.  Its parameter
tree mirrors the official safetensors metadata so loading can remain strict.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .neighborhood_attention import (
    metal_neighborhood_attention_3d_slice,
    metal_qk_rmsnorm_rope_3d,
    metal_rmsnorm_rope_3d_slice,
    neighborhood_attention_3d,
)


@dataclass(frozen=True)
class DiffusionVAEConfig:
    in_channels: int = 128
    out_channels: int = 3
    patch_size: int = 4
    head_dim: int = 64
    stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256)
    stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8)
    stage_kernels: tuple[tuple[int, int, int], ...] = (
        (3, 7, 7),
        (3, 7, 7),
        (3, 5, 5),
        (3, 5, 5),
        (11, 11, 11),
    )
    upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
        ((1, 2, 2), 2),
        ((2, 1, 1), 2),
        ((2, 2, 2), 1),
        ((2, 2, 2), 2),
    )
    timestep_scale_multiplier: float = 1000.0
    default_num_inference_steps: int = 1
    model_output_type: str = "x0"
    stage5_channels: int | None = None
    stage5_kernel: tuple[int, int, int] = (11, 11, 11)

    @classmethod
    def from_metadata(cls, metadata: dict) -> DiffusionVAEConfig:
        vae = metadata.get("vae", {})
        decoder = vae.get("decoder", {}) if isinstance(vae, dict) else {}
        if (
            not isinstance(decoder, dict)
            or "diffusion" not in str(decoder.get("_class_name", "")).lower()
        ):
            raise ValueError("The selected video VAE is not an LTX Diffusion VAE checkpoint.")
        return cls(
            in_channels=int(decoder.get("in_channels", 128)),
            out_channels=int(decoder.get("out_channels", 3)),
            patch_size=int(decoder.get("patch_size", 4)),
            head_dim=int(decoder.get("head_dim", 64)),
            stage_channels=tuple(int(v) for v in decoder.get("stage_channels", cls.stage_channels)),
            stage_depths=tuple(int(v) for v in decoder.get("stage_depths", cls.stage_depths)),
            stage_kernels=tuple(
                tuple(int(v) for v in row)
                for row in decoder.get("stage_kernels", cls.stage_kernels)
            ),
            upsamples=tuple(
                (tuple(int(v) for v in row[0]), int(row[1]))
                for row in decoder.get("upsamples", cls.upsamples)
            ),
            timestep_scale_multiplier=float(decoder.get("timestep_scale_multiplier", 1.0)),
            default_num_inference_steps=int(decoder.get("default_num_inference_steps", 1)),
            model_output_type=str(vae.get("model_output_type", "v")),
            stage5_channels=(
                int(decoder["stage5_channels"])
                if decoder.get("stage5_channels") is not None
                else None
            ),
            stage5_kernel=tuple(int(v) for v in decoder.get("stage5_kernel", (11, 11, 11))),
        )


class _Statistics(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mean = mx.zeros((channels,))
        self.std = mx.ones((channels,))

    def unnormalize(self, latent: mx.array) -> mx.array:
        return latent * self.std.reshape(1, 1, 1, 1, -1) + self.mean.reshape(1, 1, 1, 1, -1)


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    scale = mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x * scale.astype(x.dtype)) * weight


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _rope_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _rotate_axis(x: mx.array, positions: mx.array, *, axis: int) -> mx.array:
    dim = x.shape[-1]
    inv = mx.exp(-math.log(10000.0) * mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    angles = positions[:, None] * inv[None]
    shape = [1] * x.ndim
    shape[axis] = positions.shape[0]
    shape[-1] = dim // 2
    cosine = mx.cos(angles).reshape(shape)
    sine = mx.sin(angles).reshape(shape)
    pairs = x.reshape(*x.shape[:-1], dim // 2, 2).astype(mx.float32)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack((even * cosine - odd * sine, even * sine + odd * cosine), axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def _apply_rope(x: mx.array) -> mx.array:
    d_t, d_h, _ = _rope_split(x.shape[-1])
    t = _rotate_axis(x[..., :d_t], mx.arange(x.shape[1], dtype=mx.float32), axis=1)
    h = _rotate_axis(x[..., d_t : d_t + d_h], mx.arange(x.shape[2], dtype=mx.float32), axis=2)
    w = _rotate_axis(x[..., d_t + d_h :], mx.arange(x.shape[3], dtype=mx.float32), axis=3)
    return mx.concatenate((t, h, w), axis=-1)


class _Attention(nn.Module):
    def __init__(
        self, dim: int, kernel: tuple[int, int, int], head_dim: int, attention_backend: str
    ) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel = kernel
        self.attention_backend = attention_backend

    def _project_qkv_slice(self, x: mx.array, index: int) -> mx.array:
        start = index * self.dim
        stop = start + self.dim
        weight = self.qkv.weight[start:stop]
        if "bias" in self.qkv:
            return mx.addmm(self.qkv.bias[start:stop], x, weight.T)
        return mx.matmul(x, weight.T)

    def __call__(self, x: mx.array, *, query_chunk_size: int) -> mx.array:
        batch, frames, height, width, _ = x.shape
        if self.attention_backend in {"metal", "metal_tiled"}:
            head_shape = (batch, frames, height, width, self.num_heads, self.head_dim)
            query_count = batch * frames * height * width
            if self.attention_backend == "metal_tiled" and query_chunk_size < query_count:
                flat_x = x.reshape(query_count, self.dim)
                full_shape = (
                    batch,
                    frames,
                    height,
                    width,
                    self.num_heads,
                    self.head_dim,
                )
                k = self._project_qkv_slice(flat_x, 1).reshape(
                    query_count, self.num_heads, self.head_dim
                )
                k = metal_rmsnorm_rope_3d_slice(
                    k,
                    self.k_norm.weight,
                    full_shape=full_shape,
                    query_start=0,
                ).reshape(head_shape)
                mx.eval(k)
                v = self._project_qkv_slice(flat_x, 2).reshape(head_shape)
                mx.eval(v)
                pieces = []
                for start in range(0, query_count, query_chunk_size):
                    stop = min(start + query_chunk_size, query_count)
                    q = self._project_qkv_slice(flat_x[start:stop], 0).reshape(
                        stop - start, self.num_heads, self.head_dim
                    )
                    q = metal_rmsnorm_rope_3d_slice(
                        q,
                        self.q_norm.weight,
                        full_shape=full_shape,
                        query_start=start,
                    )
                    out = metal_neighborhood_attention_3d_slice(
                        q,
                        k,
                        v,
                        query_start=start,
                        kernel=self.kernel,
                        scale=self.head_dim**-0.5,
                    )
                    piece = self.proj(out.reshape(stop - start, self.dim))
                    mx.eval(piece)
                    pieces.append(piece)
                return mx.concatenate(pieces, axis=0).reshape(
                    batch, frames, height, width, self.dim
                )
            q = self._project_qkv_slice(x, 0).reshape(head_shape)
            k = self._project_qkv_slice(x, 1).reshape(head_shape)
            q, k = metal_qk_rmsnorm_rope_3d(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
            )
            mx.eval(q, k)
            v = self._project_qkv_slice(x, 2).reshape(head_shape)
        else:
            qkv = self.qkv(x).reshape(
                batch, frames, height, width, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
            q = _apply_rope(self.q_norm(q))
            k = _apply_rope(self.k_norm(k))
        out = neighborhood_attention_3d(
            q,
            k,
            v,
            kernel=self.kernel,
            query_chunk_size=query_chunk_size,
            boundary_mode="shift",
            backend="metal" if self.attention_backend == "metal_tiled" else self.attention_backend,
        )
        return self.proj(out.reshape(batch, frames, height, width, self.dim))


class _SwiGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = math.ceil((dim * 4.0) / 16) * 16
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array, *, token_chunk_size: int) -> mx.array:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        pieces = []
        for start in range(0, flat.shape[0], token_chunk_size):
            tile = flat[start : start + token_chunk_size]
            out = self.w_down(_silu(self.w_gate(tile)) * self.w_up(tile))
            mx.eval(out)
            pieces.append(out)
        return mx.concatenate(pieces, axis=0).reshape(shape)


class _NABlock(nn.Module):
    def __init__(
        self, dim: int, kernel: tuple[int, int, int], head_dim: int, attention_backend: str
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, kernel, head_dim, attention_backend)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = _SwiGLU(dim)

    def __call__(self, x: mx.array, *, query_chunk_size: int, token_chunk_size: int) -> mx.array:
        x = x + self.attn(self.norm1(x), query_chunk_size=query_chunk_size)
        x = x + self.mlp(self.norm2(x), token_chunk_size=token_chunk_size)
        return x


class _Upsample(nn.Module):
    def __init__(self, channels: int, stride: tuple[int, int, int], reduction: int) -> None:
        super().__init__()
        self.stride = stride
        self.out_channels = channels // reduction
        self.proj = nn.Linear(channels, math.prod(stride) * self.out_channels)

    def __call__(self, x: mx.array, *, drop_leading_frame: bool = True) -> mx.array:
        p_t, p_h, p_w = self.stride
        x = self.proj(x)
        batch, frames, height, width, _ = x.shape
        x = x.reshape(batch, frames, height, width, self.out_channels, p_t, p_h, p_w)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(batch, frames * p_t, height * p_h, width * p_w, self.out_channels)
        if p_t == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x


class _DiffusionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        context_channels: int,
        kernel: tuple[int, int, int],
        head_dim: int,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim)
        self.scale_shift_table = mx.zeros((7, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, kernel, head_dim, attention_backend)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = _SwiGLU(dim)

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        modulation: mx.array,
        *,
        query_chunk_size: int,
        token_chunk_size: int,
    ) -> mx.array:
        values = (
            modulation[:, :, None, None, None, :]
            + self.scale_shift_table[None, :, None, None, None, :]
        )
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [values[:, i] for i in range(7)]
        x = x + self.context_proj(context)
        y = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn(y, query_chunk_size=query_chunk_size)
        y = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(y, token_chunk_size=token_chunk_size)

    def forward_deferred(
        self,
        x: mx.array,
        low_resolution_context: mx.array,
        context_upsample: _Upsample,
        modulation: mx.array,
        *,
        width_chunks: int,
        query_chunk_size: int,
        token_chunk_size: int,
    ) -> mx.array:
        """Project stage-four context in bounded width stripes at each point of use."""
        low_width = low_resolution_context.shape[3]
        chunk_width = max(1, math.ceil(low_width / width_chunks))
        pieces = []
        for start in range(0, low_width, chunk_width):
            stop = min(start + chunk_width, low_width)
            context_piece = context_upsample(low_resolution_context[:, :, :, start:stop])
            target_start = start * context_upsample.stride[2]
            target_stop = stop * context_upsample.stride[2]
            x_piece = x[:, : context_piece.shape[1], :, target_start:target_stop]
            context_piece = context_piece[:, : x_piece.shape[1]]
            piece = x_piece + self.context_proj(context_piece)
            mx.eval(piece)
            pieces.append(piece)
        x = mx.concatenate(pieces, axis=3)
        values = (
            modulation[:, :, None, None, None, :]
            + self.scale_shift_table[None, :, None, None, None, :]
        )
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [values[:, i] for i in range(7)]
        y = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + self.attn(y, query_chunk_size=query_chunk_size)
        y = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(y, token_chunk_size=token_chunk_size)


class _TimestepEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = [nn.Linear(256, 384), nn.Identity(), nn.Linear(384, 384)]

    def __call__(self, timestep: mx.array, dtype) -> mx.array:
        half = 128
        exponent = -math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half
        angles = timestep.astype(mx.float32)[:, None] * mx.exp(exponent)[None]
        projected = mx.concatenate((mx.cos(angles), mx.sin(angles)), axis=-1).astype(dtype)
        return self.mlp[2](_silu(self.mlp[0](projected)))


class _SharedAdaLN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(384, 7 * dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(_silu(x)).reshape(x.shape[0], 7, -1)


class MLXDiffusionVideoDecoder(nn.Module):
    """One-step MLX Diffusion VAE with bounded NA and SwiGLU intermediates."""

    def __init__(
        self,
        config: DiffusionVAEConfig,
        *,
        query_chunk_size: int = 512,
        token_chunk_size: int = 4096,
        stage4_tile_width: int = 0,
        attention_backend: str = "einsum",
        deferred_stage4: bool = False,
        context_width_chunks: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if config.default_num_inference_steps < 1 or config.model_output_type not in {"x0", "v"}:
            raise ValueError(
                "The native MLX decoder requires a positive step count and x0 or v output."
            )
        if attention_backend not in {"einsum", "sdpa", "metal", "metal_tiled"}:
            raise ValueError(
                "Diffusion VAE attention backend must be 'einsum', 'sdpa', or 'metal'."
            )
        if context_width_chunks < 1:
            raise ValueError("Diffusion VAE context width chunks must be positive.")
        if deferred_stage4 and stage4_tile_width:
            raise ValueError("Deferred stage four and stage-four tiling cannot be combined.")
        self.config = config
        self.query_chunk_size = query_chunk_size
        self.token_chunk_size = token_chunk_size
        self.stage4_tile_width = stage4_tile_width
        self.attention_backend = attention_backend
        self.deferred_stage4 = deferred_stage4
        self.context_width_chunks = context_width_chunks
        self.seed = seed
        self.per_channel_statistics = _Statistics(config.in_channels)
        self.conv_in = nn.Linear(config.in_channels, config.stage_channels[0])
        self.det_stages = [
            [
                _NABlock(
                    config.stage_channels[i],
                    config.stage_kernels[i],
                    config.head_dim,
                    attention_backend,
                )
                for _ in range(config.stage_depths[i])
            ]
            for i in range(4)
        ]
        self.upsamples = [
            _Upsample(config.stage_channels[i], stride, reduction)
            for i, (stride, reduction) in enumerate(config.upsamples)
        ]
        context_dim = config.stage_channels[-1]
        final_dim = config.stage5_channels or context_dim
        self.t_embedder = _TimestepEmbedder()
        self.conv_in_x_t = nn.Linear(config.out_channels * config.patch_size**2, final_dim)
        self.shared_adaln = _SharedAdaLN(final_dim)
        self.diff_blocks = [
            _DiffusionBlock(
                final_dim,
                context_dim,
                config.stage5_kernel,
                config.head_dim,
                attention_backend,
            )
            for _ in range(config.stage_depths[-1])
        ]
        self.norm_out = nn.RMSNorm(final_dim, eps=1e-6)
        self.conv_out = nn.Linear(final_dim, config.out_channels * config.patch_size**2)
        # Preserved for strict compatibility with the distributed 2.5 checkpoint.
        self.type_emb = mx.zeros((config.in_channels,))

    def _run_stage_blocks(self, x: mx.array, index: int) -> mx.array:
        for block in self.det_stages[index]:
            x = block(
                x,
                query_chunk_size=self.query_chunk_size,
                token_chunk_size=self.token_chunk_size,
            )
        mx.eval(x)
        return x

    def _run_stage(self, x: mx.array, index: int) -> mx.array:
        x = self._run_stage_blocks(x, index)
        x = self.upsamples[index](x)
        mx.eval(x)
        return x

    def _stage5_halo_in_stage4_cells(self) -> int:
        """Return the exact horizontal context needed around a stage-4 output core."""
        stage4_radius = self.config.stage_depths[3] * (self.config.stage_kernels[3][2] // 2)
        stage5_radius = self.config.stage_depths[4] * (self.config.stage5_kernel[2] // 2)
        stage4_stride = self.config.upsamples[3][0][2]
        return stage4_radius + math.ceil(stage5_radius / stage4_stride)

    def _initial_pixels(
        self,
        *,
        batch: int,
        frames: int,
        height: int,
        width: int,
        dtype,
    ) -> mx.array:
        return mx.random.normal(
            (batch, self.config.out_channels, frames, height, width),
            key=mx.random.key(self.seed),
            dtype=dtype,
        )

    def _patch_pixels(self, pixels: mx.array) -> mx.array:
        batch, _, frames, pixel_height, pixel_width = pixels.shape
        patch = self.config.patch_size
        if pixel_height % patch or pixel_width % patch:
            raise ValueError("Diffusion VAE pixel noise must be divisible by its patch size.")
        height = pixel_height // patch
        width = pixel_width // patch
        patched = pixels.reshape(
            batch,
            self.config.out_channels,
            frames,
            height,
            patch,
            width,
            patch,
        )
        # LTX stores each patch as (channel, width-subpixel, height-subpixel).
        return patched.transpose(0, 2, 3, 5, 1, 6, 4).reshape(batch, frames, height, width, -1)

    def _predict_diffusion_stage(
        self,
        context: mx.array,
        patched_pixels: mx.array,
        timestep_value: float,
        *,
        deferred_context: bool = False,
    ) -> mx.array:
        batch = context.shape[0]
        x = self.conv_in_x_t(patched_pixels)
        timestep = mx.full(
            (batch,),
            timestep_value * self.config.timestep_scale_multiplier,
            dtype=mx.float32,
        )
        modulation = self.shared_adaln(self.t_embedder(timestep, x.dtype))
        for block in self.diff_blocks:
            if deferred_context:
                x = block.forward_deferred(
                    x,
                    context,
                    self.upsamples[3],
                    modulation,
                    width_chunks=self.context_width_chunks,
                    query_chunk_size=self.query_chunk_size,
                    token_chunk_size=self.token_chunk_size,
                )
            else:
                x = block(
                    x,
                    context,
                    modulation,
                    query_chunk_size=self.query_chunk_size,
                    token_chunk_size=self.token_chunk_size,
                )
            mx.eval(x)
        return self.conv_out(self.norm_out(x))

    def _run_diffusion_stage(
        self, context: mx.array, patched_pixels: mx.array, *, deferred_context: bool = False
    ) -> mx.array:
        x_t = patched_pixels
        steps = self.config.default_num_inference_steps
        timesteps = [1.0 - index / steps for index in range(steps)]
        for index, timestep in enumerate(timesteps):
            prediction = self._predict_diffusion_stage(
                context, x_t, timestep, deferred_context=deferred_context
            )
            next_timestep = timesteps[index + 1] if index + 1 < steps else 0.0
            if self.config.model_output_type == "x0":
                velocity = (x_t - prediction) / timestep
            else:
                velocity = prediction
            x_t = x_t - (timestep - next_timestep) * velocity
            mx.eval(x_t)
        return x_t

    def _unpatch(self, x: mx.array) -> mx.array:
        batch, frames, height, width, _ = x.shape
        patch = self.config.patch_size
        x = x.reshape(batch, frames, height, width, self.config.out_channels, patch, patch)
        return x.transpose(0, 4, 1, 2, 6, 3, 5).reshape(
            batch,
            self.config.out_channels,
            frames,
            height * patch,
            width * patch,
        )

    def _decode_stage4_width_tiles(
        self,
        stage4_input: mx.array,
        patched_pixels: mx.array,
        *,
        target_frames: int,
    ) -> mx.array:
        """Decode stages 4–5 in haloed width stripes, then join non-overlap cores."""
        core_width = self.stage4_tile_width
        full_width = stage4_input.shape[3]
        if core_width < 1 or core_width >= full_width:
            if self.deferred_stage4:
                context = self._run_stage_blocks(stage4_input, 3)
                return self._run_diffusion_stage(context, patched_pixels, deferred_context=True)
            context = self._run_stage(stage4_input, 3)
            context = context[:, : max(target_frames, self.config.stage5_kernel[0])]
            return self._run_diffusion_stage(context, patched_pixels)

        halo = self._stage5_halo_in_stage4_cells()
        upsample_width = self.config.upsamples[3][0][2]
        pieces = []
        for core_start in range(0, full_width, core_width):
            core_stop = min(core_start + core_width, full_width)
            tile_start = max(0, core_start - halo)
            tile_stop = min(full_width, core_stop + halo)
            stage4_tile = stage4_input[:, :, :, tile_start:tile_stop]
            context = self._run_stage(stage4_tile, 3)
            context = context[:, : max(target_frames, self.config.stage5_kernel[0])]
            pixel_start = tile_start * upsample_width
            pixel_stop = tile_stop * upsample_width
            noise_tile = patched_pixels[:, :, :, pixel_start:pixel_stop]
            decoded = self._run_diffusion_stage(context, noise_tile)
            keep_start = (core_start - tile_start) * upsample_width
            keep_stop = keep_start + (core_stop - core_start) * upsample_width
            piece = decoded[:, :, :, keep_start:keep_stop]
            mx.eval(piece)
            pieces.append(piece)
        return mx.concatenate(pieces, axis=3)

    def decode(self, latent: mx.array) -> mx.array:
        if latent.ndim != 5 or latent.shape[1] != self.config.in_channels:
            raise ValueError("Diffusion VAE latent must use (B, 128, F, H, W) layout.")
        minimum = self.config.stage_kernels[0]
        if any(latent.shape[axis + 2] < minimum[axis] for axis in range(3)):
            raise ValueError(
                "Initial MLX Diffusion VAE decode requires latent F/H/W to cover "
                f"the first-stage kernel {minimum}."
            )
        output_dtype = latent.dtype
        x = latent.transpose(0, 2, 3, 4, 1)
        # Match the NATTEN final-frame workaround while stages 1-4 build context.
        x = mx.concatenate((x, mx.repeat(x[:, -1:], 2, axis=1)), axis=1)
        x = self.conv_in(self.per_channel_statistics.unnormalize(x))
        for stage in range(3):
            x = self._run_stage(x, stage)
        target_frames = latent.shape[2] * 8 - 7
        batch = x.shape[0]
        stage4_stride = self.config.upsamples[3][0]
        frames = max(target_frames, self.config.stage5_kernel[0])
        height = x.shape[2] * stage4_stride[1]
        width = x.shape[3] * stage4_stride[2]
        pixels = self._initial_pixels(
            batch=batch,
            frames=frames,
            height=height * self.config.patch_size,
            width=width * self.config.patch_size,
            dtype=output_dtype,
        )
        patched = self._patch_pixels(pixels)
        x = self._decode_stage4_width_tiles(x, patched, target_frames=target_frames)
        x = self._unpatch(x)
        return x[:, :, :target_frames].astype(output_dtype)

    def tiled_decode(self, latent: mx.array, _tiling_config=None):
        """Compatibility iterator; bounded internal kernels precede overlap tiling."""
        pixels = self.decode(latent)
        mx.eval(pixels)
        yield pixels

    def decode_and_stream(
        self,
        latent: mx.array,
        output_path: str,
        *,
        frame_rate: float,
        audio_path: str | None = None,
    ) -> None:
        # Reuse the installed Apache-compatible media publication contract while
        # dispatching decoding back through this class's iterator.
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder

        VideoDecoder.decode_and_stream(
            self,
            latent,
            output_path,
            frame_rate=frame_rate,
            audio_path=audio_path,
        )


def load_diffusion_video_decoder(
    path: str | Path,
    metadata: dict,
    *,
    query_chunk_size: int | None = None,
    token_chunk_size: int = 4096,
    stage4_tile_width: int | None = None,
    attention_backend: str | None = None,
    deferred_stage4: bool | None = None,
    context_width_chunks: int | None = None,
    seed: int = 0,
) -> MLXDiffusionVideoDecoder:
    """Construct and strictly load the decoder subtree from an official checkpoint."""
    config = DiffusionVAEConfig.from_metadata(metadata)
    resolved_query_chunk = int(
        os.environ.get("LTX25_DIFFVAE_QUERY_CHUNK_SIZE", query_chunk_size or 512)
    )
    if resolved_query_chunk < 1:
        raise ValueError("LTX25_DIFFVAE_QUERY_CHUNK_SIZE must be positive.")
    resolved_tile_width = int(
        os.environ.get("LTX25_DIFFVAE_STAGE4_TILE_WIDTH", stage4_tile_width or 0)
    )
    if resolved_tile_width < 0:
        raise ValueError("LTX25_DIFFVAE_STAGE4_TILE_WIDTH must be zero or positive.")
    resolved_attention_backend = os.environ.get(
        "LTX25_DIFFVAE_ATTENTION_BACKEND", attention_backend or "einsum"
    ).lower()
    resolved_deferred_stage4 = os.environ.get(
        "LTX25_DIFFVAE_DEFERRED_STAGE4", "1" if deferred_stage4 else "0"
    ).lower() in {"1", "true", "yes", "on"}
    resolved_context_width_chunks = int(
        os.environ.get("LTX25_DIFFVAE_CONTEXT_WIDTH_CHUNKS", context_width_chunks or 4)
    )
    decoder = MLXDiffusionVideoDecoder(
        config,
        query_chunk_size=resolved_query_chunk,
        token_chunk_size=token_chunk_size,
        stage4_tile_width=resolved_tile_width,
        attention_backend=resolved_attention_backend,
        deferred_stage4=resolved_deferred_stage4,
        context_width_chunks=resolved_context_width_chunks,
        seed=seed,
    )
    weights = mx.load(str(Path(path).expanduser()))
    mapped = {}
    for key, value in weights.items():
        if key.startswith("decoder."):
            mapped[key.removeprefix("decoder.")] = value
        elif key == "per_channel_statistics.mean-of-means":
            mapped["per_channel_statistics.mean"] = value
        elif key == "per_channel_statistics.std-of-means":
            mapped["per_channel_statistics.std"] = value
    mapped = _fold_legacy_diffusion_gates(mapped)
    decoder.load_weights(list(mapped.items()), strict=True)
    mx.eval(decoder.parameters())
    return decoder


def _fold_legacy_diffusion_gates(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Fold legacy static DiffVAE gates into their projections once at load time."""
    folded = dict(weights)
    targets = {
        "gate_msa": "attn.proj",
        "gate_mlp": "mlp.w_down",
        "gate_ctx": "context_proj",
    }
    for gate_key in [key for key in folded if key.rsplit(".", 1)[-1] in targets]:
        prefix, gate_name = gate_key.rsplit(".", 1)
        gate = folded.pop(gate_key)
        target = targets[gate_name]
        for suffix in ("weight", "bias"):
            parameter_key = f"{prefix}.{target}.{suffix}"
            if parameter_key not in folded:
                continue
            parameter = folded[parameter_key]
            multiplier = gate[:, None] if parameter.ndim == 2 else gate
            folded[parameter_key] = (
                parameter.astype(mx.float32) * multiplier.astype(mx.float32)
            ).astype(parameter.dtype)
    return folded


__all__ = ["DiffusionVAEConfig", "MLXDiffusionVideoDecoder", "load_diffusion_video_decoder"]
