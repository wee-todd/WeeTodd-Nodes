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
    """Own the convolutional video-VAE decoder and streamed publication."""

    def __init__(self, path: str | Path, *, verbose: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.verbose = verbose
        self._decoder = None

    def load(self):
        if self._decoder is not None:
            return self._decoder
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        vae_config = _metadata_config(self.path).get("vae", {})
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
        self.load().decode_and_stream(
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


def load_ltx25_spatial_upsampler(path: str | Path):
    """Load the official 2x latent upscaler from embedded configuration."""
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


__all__ = [
    "LTX25AudioDecoder",
    "LTX25ImageConditioner",
    "LTX25LatentNormalizer",
    "LTX25VideoDecoder",
    "load_ltx25_spatial_upsampler",
    "remap_convolution_layout",
]
