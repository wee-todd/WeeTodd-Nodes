"""Direct loaders for official split LTX 2.5 components."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def _metadata_config(path: str | Path) -> dict:
    with safe_open(Path(path).expanduser(), framework="numpy") as handle:
        raw = (handle.metadata() or {}).get("config", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError(f"Component config in {path} must be a JSON object.")
    return parsed


def _cleanup() -> None:
    gc.collect()
    mx.clear_cache()


def remap_convolution_layout(model, weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Transpose PyTorch convolution kernels only when MLX shapes prove the map."""
    from mlx.utils import tree_flatten

    expected = dict(tree_flatten(model.parameters()))
    remapped: dict[str, mx.array] = {}
    for key, value in weights.items():
        target = expected.get(key)
        if target is None or tuple(value.shape) == tuple(target.shape):
            remapped[key] = value
            continue
        permutations = {
            3: ((0, 2, 1), (1, 2, 0)),
            4: ((0, 2, 3, 1), (1, 2, 3, 0)),
            5: ((0, 2, 3, 4, 1), (1, 2, 3, 4, 0)),
        }.get(value.ndim, ())
        converted = None
        for axes in permutations:
            candidate = value.transpose(*axes)
            if tuple(candidate.shape) == tuple(target.shape):
                converted = candidate
                break
        remapped[key] = converted if converted is not None else value
    return remapped


class LTX25ImageConditioner:
    """Own the convolutional video-VAE encoder used for I2V conditioning."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._encoder = None

    def load(self):
        if self._encoder is not None:
            return self._encoder
        from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        encoder = VideoEncoder()
        weights = load_split_safetensors(self.path, prefix="encoder.")
        shared = load_split_safetensors(self.path)
        shared_stats = {
            "per_channel_statistics.mean-of-means": "per_channel_statistics.mean_of_means",
            "per_channel_statistics.std-of-means": "per_channel_statistics.std_of_means",
        }
        for source, target in shared_stats.items():
            if source in shared:
                weights[target] = shared[source]
        weights = {
            key.replace("._mean_of_means", ".mean_of_means").replace(
                "._std_of_means", ".std_of_means"
            ): value
            for key, value in weights.items()
        }
        weights = remap_convolution_layout(encoder, weights)
        encoder.load_weights(list(weights.items()), strict=True)
        mx.eval(encoder.parameters())
        self._encoder = encoder
        _cleanup()
        return encoder

    def free(self) -> None:
        self._encoder = None
        _cleanup()


class LTX25LatentNormalizer:
    """Load only the two VAE statistics needed around latent upscaling."""

    def __init__(self, path: str | Path) -> None:
        source = Path(path).expanduser()
        weights = mx.load(str(source))
        self.mean = weights["per_channel_statistics.mean-of-means"]
        self.std = weights["per_channel_statistics.std-of-means"]
        mx.eval(self.mean, self.std)
        del weights

    def normalize_latent(self, latent: mx.array) -> mx.array:
        mean = self.mean.reshape(1, 1, 1, 1, -1)
        std = self.std.reshape(1, 1, 1, 1, -1)
        return (latent - mean) / std

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        mean = self.mean.reshape(1, 1, 1, 1, -1)
        std = self.std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean


class LTX25VideoDecoder:
    """Own either official LTX 2.5 video-VAE decoder and streamed publication."""

    def __init__(
        self,
        path: str | Path,
        *,
        verbose: bool = True,
        diffvae_optimization: str = "combined",
        diffvae_query_chunk_size: int = 512,
        diffvae_context_width_chunks: int = 4,
        diffvae_stage4_tile_width: int = 0,
    ) -> None:
        self.path = Path(path).expanduser()
        self.verbose = verbose
        self._decoder = None
        self.last_decode_report: dict[str, object] = {}
        self.diffvae_optimization = diffvae_optimization
        self.diffvae_query_chunk_size = diffvae_query_chunk_size
        self.diffvae_context_width_chunks = diffvae_context_width_chunks
        self.diffvae_stage4_tile_width = diffvae_stage4_tile_width

    def load(self):
        if self._decoder is not None:
            return self._decoder
        component_config = _metadata_config(self.path)
        vae_config = component_config.get("vae", {})
        decoder_config = vae_config.get("decoder", {}) if isinstance(vae_config, dict) else {}
        decoder_name = str(decoder_config.get("_class_name", ""))
        if "diffusion" in decoder_name.lower():
            from .diffusion_vae import load_diffusion_video_decoder

            self._decoder = load_diffusion_video_decoder(
                self.path,
                component_config,
                query_chunk_size=self.diffvae_query_chunk_size,
                attention_backend=(
                    "metal_tiled"
                    if self.diffvae_optimization == "metal_na3d_query_tiled_experimental"
                    else (
                        "metal"
                        if self.diffvae_optimization == "metal_na3d_experimental"
                        else "einsum"
                    )
                ),
                deferred_stage4=self.diffvae_optimization == "deferred_stage4",
                context_width_chunks=self.diffvae_context_width_chunks,
                stage4_tile_width=(
                    self.diffvae_stage4_tile_width
                    if self.diffvae_optimization == "stage4_width_tiles"
                    else 0
                ),
            )
            _cleanup()
            return self._decoder
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        decoder = VideoDecoder(
            causal=bool(vae_config.get("causal_decoder", False)),
            spatial_padding_mode=str(vae_config.get("spatial_padding_mode", "zeros")),
        )
        weights = load_split_safetensors(self.path, prefix="decoder.")
        shared = load_split_safetensors(self.path)
        shared_stats = {
            "per_channel_statistics.mean-of-means": "per_channel_statistics.mean",
            "per_channel_statistics.std-of-means": "per_channel_statistics.std",
        }
        for source, target in shared_stats.items():
            if source in shared:
                weights[target] = shared[source]
        weights = remap_convolution_layout(decoder, weights)
        decoder.load_weights(list(weights.items()), strict=True)
        mx.eval(decoder.parameters())
        self._decoder = decoder
        _cleanup()
        return decoder

    def free(self) -> None:
        self._decoder = None
        _cleanup()

    def decode_and_stream(
        self,
        video_latent,
        output_path: str,
        *,
        frame_rate: float = 24.0,
        audio_path: str | None = None,
    ) -> str:
        decoder = self.load()
        # ltx-core-mlx computes a conservative temporal tile from the actual
        # latent shape and streams completed RGB frames directly to ffmpeg. Keep
        # that bounded path visible in generation metadata instead of presenting
        # decode as an opaque final allocation.
        try:
            import os

            from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling

            is_diffusion = decoder.__class__.__name__ == "MLXDiffusionVideoDecoder"
            tiling = (
                None
                if is_diffusion
                else _compute_decode_tiling(video_latent.shape, frame_rate=frame_rate)
            )
            temporal = getattr(tiling, "temporal_config", None)
            self.last_decode_report = {
                "publication": "direct_ffmpeg_stream",
                "decoder": "diffusion" if is_diffusion else "convolutional",
                "diffvae_optimization": self.diffvae_optimization if is_diffusion else None,
                "diffvae_attention_backend": (
                    str(getattr(decoder, "attention_backend", "einsum"))
                    if is_diffusion
                    else None
                ),
                "diffvae_inference_steps": (
                    int(decoder.config.default_num_inference_steps) if is_diffusion else None
                ),
                "diffvae_model_output_type": (
                    str(decoder.config.model_output_type) if is_diffusion else None
                ),
                "query_chunk_size": (
                    int(getattr(decoder, "query_chunk_size", 0)) if is_diffusion else None
                ),
                "stage4_tile_width": (
                    int(getattr(decoder, "stage4_tile_width", 0)) if is_diffusion else None
                ),
                "decode_budget_gib": float(os.environ.get("LTX2_VAE_DECODE_BUDGET_GB", "8.0")),
                "temporal_tiling": temporal is not None,
                "tile_frames": (
                    int(temporal.tile_size_in_frames) if temporal is not None else None
                ),
                "overlap_frames": (
                    int(temporal.tile_overlap_in_frames) if temporal is not None else None
                ),
            }
        except (ImportError, AttributeError, TypeError, ValueError):
            self.last_decode_report = {"publication": "direct_ffmpeg_stream"}
        decoder.decode_and_stream(
            video_latent,
            output_path,
            frame_rate=frame_rate,
            audio_path=audio_path,
        )
        return output_path


class LTX25AudioDecoder:
    """Own the official combined audio-VAE, vocoder, and BWE checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._audio_decoder = None
        self._vocoder = None

    def load(self):
        if self._audio_decoder is None:
            from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
            from ltx_core_mlx.utils.weights import (
                load_split_safetensors,
                remap_audio_vae_keys,
            )

            decoder = AudioVAEDecoder()
            weights = load_split_safetensors(self.path, prefix="audio_vae.decoder.")
            statistics = load_split_safetensors(self.path, prefix="audio_vae.")
            weights.update(
                (
                    key.replace("mean-of-means", "mean_of_means").replace(
                        "std-of-means", "std_of_means"
                    ),
                    value,
                )
                for key, value in statistics.items()
                if key.startswith("per_channel_statistics.")
            )
            weights = remap_audio_vae_keys(weights)
            weights = remap_convolution_layout(decoder, weights)
            decoder.load_weights(list(weights.items()), strict=True)
            mx.eval(decoder.parameters())
            self._audio_decoder = decoder
            _cleanup()
        if self._vocoder is None:
            from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE
            from ltx_core_mlx.utils.weights import load_split_safetensors

            vocoder = VocoderWithBWE()
            weights = load_split_safetensors(self.path, prefix="vocoder.")
            weights = {key.removeprefix("vocoder."): value for key, value in weights.items()}
            weights = remap_convolution_layout(vocoder, weights)
            vocoder.load_weights(list(weights.items()), strict=True)
            vocoder.upcast_weights_to_fp32()
            mx.eval(vocoder.parameters())
            self._vocoder = vocoder
            _cleanup()
        return self._audio_decoder, self._vocoder

    def free(self) -> None:
        self._audio_decoder = None
        self._vocoder = None
        _cleanup()

    def __call__(self, audio_latent):
        decoder, vocoder = self.load()
        return vocoder(decoder.decode(audio_latent))


class LTX25AudioConditioner:
    """Own the audio-VAE encoder used for frozen refinement context."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._encoder = None
        self._processor = None

    def load(self):
        if self._encoder is not None and self._processor is not None:
            return self._encoder, self._processor
        from ltx_core_mlx.model.audio_vae import AudioProcessor, AudioVAEEncoder
        from ltx_core_mlx.utils.weights import (
            load_split_safetensors,
            remap_audio_vae_keys,
        )

        encoder = AudioVAEEncoder()
        weights = load_split_safetensors(self.path, prefix="audio_vae.encoder.")
        statistics = load_split_safetensors(self.path, prefix="audio_vae.")
        weights.update(
            (
                key.replace("mean-of-means", "mean_of_means").replace(
                    "std-of-means", "std_of_means"
                ),
                value,
            )
            for key, value in statistics.items()
            if key.startswith("per_channel_statistics.")
        )
        weights = remap_audio_vae_keys(weights)
        weights = remap_convolution_layout(encoder, weights)
        encoder.load_weights(list(weights.items()), strict=True)
        mx.eval(encoder.parameters())
        self._encoder = encoder
        self._processor = AudioProcessor(sample_rate=16000)
        _cleanup()
        return self._encoder, self._processor

    def free(self) -> None:
        self._encoder = None
        self._processor = None
        _cleanup()


def load_ltx25_latent_upsampler(path: str | Path):
    """Load an official spatial or temporal latent upsampler from checkpoint metadata."""
    from ltx_core_mlx.model.upsampler.model import LatentUpsampler
    from ltx_core_mlx.utils.weights import load_split_safetensors

    source = Path(path).expanduser()
    config = _metadata_config(source)
    upsampler = LatentUpsampler.from_config(config)
    weights = load_split_safetensors(source)
    weights = remap_convolution_layout(upsampler, weights)
    upsampler.load_weights(list(weights.items()), strict=True)
    mx.eval(upsampler.parameters())
    _cleanup()
    return upsampler


def inspect_ltx25_latent_upsampler(path: str | Path) -> dict[str, object]:
    """Validate an upsampler header without loading its tensors."""
    source = Path(path).expanduser()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX 2.5 latent upsampler not found: {source}")
    config = _metadata_config(source)
    if config.get("_class_name") != "LatentUpsampler":
        raise ValueError("The selected checkpoint is not an LTX latent upsampler.")
    if int(config.get("in_channels", 0)) != 128 or int(config.get("dims", 0)) != 3:
        raise ValueError("The selected latent upsampler has an incompatible LTX 2.5 layout.")
    return {
        "path": str(source),
        "spatial_upsample": bool(config.get("spatial_upsample", False)),
        "temporal_upsample": bool(config.get("temporal_upsample", False)),
        "rational_resampler": bool(config.get("rational_resampler", False)),
    }


def load_ltx25_spatial_upsampler(path: str | Path):
    """Load the official spatial latent upsampler."""
    model = load_ltx25_latent_upsampler(path)
    if not model.spatial_upsample or model.temporal_upsample:
        raise ValueError("The selected LTX 2.5 checkpoint is not a spatial-only upsampler.")
    return model


__all__ = [
    "LTX25AudioConditioner",
    "LTX25AudioDecoder",
    "LTX25ImageConditioner",
    "LTX25LatentNormalizer",
    "LTX25VideoDecoder",
    "inspect_ltx25_latent_upsampler",
    "load_ltx25_latent_upsampler",
    "load_ltx25_spatial_upsampler",
    "remap_convolution_layout",
]
