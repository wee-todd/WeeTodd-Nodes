"""Process-local staged video and audio VAE decoding for MiniMax H3 latents."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .preflight import H3ComponentSetSpec
from .sampling import H3Latents


@dataclass(frozen=True)
class H3VideoVAESpec:
    """Immutable location of one H3 video decoder."""

    video_vae: str

    @classmethod
    def from_components(cls, components: H3ComponentSetSpec) -> H3VideoVAESpec:
        return cls(video_vae=str(components.resolved_paths()["video_vae"]))

    def validate(self) -> None:
        path = Path(self.video_vae).expanduser()
        if path.is_file():
            if path.suffix != ".safetensors":
                raise ValueError(f"Compact video VAE must be a safetensors file: {path}")
            return
        if not path.is_dir():
            raise FileNotFoundError(f"MiniMax H3 video VAE not found: {path}")
        required = (path / "config.json", path / "source" / "config.json")
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"MiniMax H3 video VAE config not found: {missing[0]}")
        weights = path / "source" / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"MiniMax H3 video VAE weights not found: {weights}")


@dataclass(frozen=True)
class H3VideoFrames:
    """Decoded Comfy-ready RGB frames plus synchronized timing provenance."""

    frames: Any
    num_frames: int
    width: int
    height: int
    fps: int
    decode_seconds: float


VideoVAEFactory = Callable[[H3VideoVAESpec], Any]


def _default_video_vae_factory(spec: H3VideoVAESpec):
    from minimax_h3_mlx.load import load_compact_video_vae, load_video_vae

    path = Path(spec.video_vae).expanduser()
    if path.is_file():
        return load_compact_video_vae(path)
    return load_video_vae(path)


class H3VideoVAECache:
    """Cache and explicitly release one final-quality H3 video decoder."""

    def __init__(self, factory: VideoVAEFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_video_vae_factory
        self._spec: H3VideoVAESpec | None = None
        self._vae: Any = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._vae is not None

    def decode(
        self,
        spec: H3VideoVAESpec,
        latents: H3Latents,
        *,
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
    ) -> H3VideoFrames:
        import time

        spec.validate()
        if Path(latents.transformer_spec.video_vae).expanduser() != Path(
            spec.video_vae
        ).expanduser():
            raise ValueError("Latents were produced for a different MiniMax H3 video VAE.")
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if latents.generation_config.memory_mode == "low_memory_bf16":
                    # Keep the existing tile geometry, but hold one decoder activation set at a
                    # time. Weights and decoder arithmetic remain in their checkpoint dtypes.
                    self._vae.decode_batch = 1
                if check_interrupted is not None:
                    check_interrupted()
                started = time.perf_counter()
                frames = self._decode_normalized(latents.video, latents.num_frames)
                elapsed = time.perf_counter() - started
                if check_interrupted is not None:
                    check_interrupted()
                result = H3VideoFrames(
                    frames=frames,
                    num_frames=frames.shape[0],
                    width=frames.shape[2],
                    height=frames.shape[1],
                    fps=latents.fps,
                    decode_seconds=elapsed,
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after or latents.generation_config.memory_mode == "low_memory_bf16":
                self._release_locked()
            return result

    def _decode_normalized(self, normalized: Any, num_frames: int) -> Any:
        import mlx.core as mx
        import numpy as np

        from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD

        cfg = self._vae.config
        mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(
            1, -1, 1, 1, 1
        )
        std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(
            1, -1, 1, 1, 1
        )
        decoded = np.asarray(
            self._vae.decode((normalized * std + mean).astype(mx.float32))
        )
        pixel_mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        frames = np.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
        frames = frames[0, :, :num_frames].transpose(1, 2, 3, 0)
        return np.ascontiguousarray(frames, dtype=np.float32)

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._vae = None
        self._spec = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


VIDEO_VAE_RUNTIME = H3VideoVAECache()


@dataclass(frozen=True)
class H3AudioVAESpec:
    """Immutable location of one H3 audio decoder."""

    audio_vae: str

    @classmethod
    def from_components(cls, components: H3ComponentSetSpec) -> H3AudioVAESpec:
        return cls(audio_vae=str(components.resolved_paths()["audio_vae"]))

    def validate(self) -> None:
        path = Path(self.audio_vae).expanduser()
        if path.is_file():
            if path.suffix != ".safetensors":
                raise ValueError(f"Compact audio VAE must be a safetensors file: {path}")
            return
        if not path.is_dir():
            raise FileNotFoundError(f"MiniMax H3 audio VAE not found: {path}")
        required = (path / "config.json", path / "metadata.json", path / "model.safetensors")
        missing = [str(item) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"MiniMax H3 audio VAE file not found: {missing[0]}")


@dataclass(frozen=True)
class H3AudioWaveform:
    """Decoded stereo waveform plus synchronized timing provenance."""

    waveform: Any
    sample_rate: int
    channels: int
    num_samples: int
    duration_seconds: float
    video_frames: int
    fps: int
    decode_seconds: float


AudioVAEFactory = Callable[[H3AudioVAESpec], Any]


def _default_audio_vae_factory(spec: H3AudioVAESpec):
    from minimax_h3_mlx.load import load_audio_vae, load_compact_audio_vae

    path = Path(spec.audio_vae).expanduser()
    if path.is_file():
        return load_compact_audio_vae(path)
    return load_audio_vae(path)


class H3AudioVAECache:
    """Cache and explicitly release one final-quality H3 audio decoder."""

    def __init__(self, factory: AudioVAEFactory | None = None) -> None:
        self._lock = RLock()
        self._factory = factory or _default_audio_vae_factory
        self._spec: H3AudioVAESpec | None = None
        self._vae: Any = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._vae is not None

    def decode(
        self,
        spec: H3AudioVAESpec,
        latents: H3Latents,
        *,
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
    ) -> H3AudioWaveform:
        import time

        spec.validate()
        if Path(latents.transformer_spec.audio_vae).expanduser() != Path(
            spec.audio_vae
        ).expanduser():
            raise ValueError("Latents were produced for a different MiniMax H3 audio VAE.")
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if self._vae.config.sampling_rate != latents.sample_rate:
                    raise ValueError(
                        "Audio VAE sample rate does not match the synchronized latent contract: "
                        f"{self._vae.config.sampling_rate} != {latents.sample_rate}."
                    )
                if check_interrupted is not None:
                    check_interrupted()
                started = time.perf_counter()
                waveform = self._decode_normalized(latents.audio)
                elapsed = time.perf_counter() - started
                if check_interrupted is not None:
                    check_interrupted()
                result = H3AudioWaveform(
                    waveform=waveform,
                    sample_rate=latents.sample_rate,
                    channels=waveform.shape[0],
                    num_samples=waveform.shape[1],
                    duration_seconds=waveform.shape[1] / latents.sample_rate,
                    video_frames=latents.num_frames,
                    fps=latents.fps,
                    decode_seconds=elapsed,
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after or latents.generation_config.memory_mode == "low_memory_bf16":
                self._release_locked()
            return result

    def _decode_normalized(self, normalized: Any) -> Any:
        import mlx.core as mx
        import numpy as np

        cfg = self._vae.config
        mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(1, -1, 1)
        std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(1, -1, 1)
        decoded = np.asarray(
            self._vae.decode((normalized * std + mean).astype(mx.float32)),
            dtype=np.float32,
        )
        if decoded.ndim != 3 or decoded.shape[0] != 2 or decoded.shape[1] != 1:
            raise ValueError(
                "Audio VAE output must have shape (2, 1, samples); "
                f"got {decoded.shape}."
            )
        return np.ascontiguousarray(decoded[:, 0, :], dtype=np.float32)

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._vae = None
        self._spec = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


AUDIO_VAE_RUNTIME = H3AudioVAECache()
