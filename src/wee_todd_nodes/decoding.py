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

    def quantization(self) -> str:
        """Return header-derived precision metadata without loading VAE tensors."""
        path = Path(self.video_vae).expanduser()
        if not path.is_file():
            return "unquantized-or-directory-managed"
        import json

        from minimax_h3_mlx.load import safetensor_metadata
        from minimax_h3_mlx.video_vae_checkpoint import (
            VIDEO_VAE_METADATA_KEY,
            validate_video_vae_quantization,
        )

        metadata = safetensor_metadata(path)
        value = metadata.get(VIDEO_VAE_METADATA_KEY)
        if value is None:
            return "unquantized-or-self-describing"
        recipe = validate_video_vae_quantization(json.loads(value))
        if recipe is None:
            return "unquantized"
        return f"mlx-affine-{recipe['bits']}bit-group-{recipe['group_size']}"


@dataclass(frozen=True)
class H3VideoFrames:
    """Decoded Comfy-ready RGB frames plus synchronized timing provenance."""

    frames: Any
    num_frames: int
    width: int
    height: int
    fps: int
    decode_seconds: float
    decode_batch: int
    quantization: str = "unquantized-or-self-describing"


@dataclass(frozen=True)
class H3VideoStream:
    """Timing and geometry for a streamed video VAE decode."""

    num_frames: int
    width: int
    height: int
    fps: int
    decode_seconds: float
    decode_batch: int
    peak_rgb8_chunk_bytes: int
    quantization: str = "unquantized-or-self-describing"


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
        prepare_stage: Callable[[], None] | None = None,
    ) -> H3VideoFrames:
        import time

        spec.validate()
        if (
            Path(latents.transformer_spec.video_vae).expanduser()
            != Path(spec.video_vae).expanduser()
        ):
            raise ValueError("Latents were produced for a different MiniMax H3 video VAE.")
        if prepare_stage is not None:
            prepare_stage()
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
                    decode_batch=int(self._vae.decode_batch),
                    quantization=spec.quantization(),
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after or latents.generation_config.memory_mode == "low_memory_bf16":
                self._release_locked()
            return result

    def encode_keyframes(
        self,
        spec: H3VideoVAESpec,
        images: list[Any],
        *,
        height: int,
        width: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
        prepare_stage: Callable[[], None] | None = None,
    ) -> Any:
        """Encode normalized FL2VA keyframe rows with staged VAE residency."""
        if not images:
            raise ValueError("FL2VA keyframe encoding requires at least one image.")
        if len(images) > 8:
            raise ValueError("Timed FL2VA supports at most eight keyframes per window.")
        spec.validate()
        if prepare_stage is not None:
            prepare_stage()
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if check_interrupted is not None:
                    check_interrupted()
                from minimax_h3_mlx.pipeline import encode_keyframe_rows

                rows = encode_keyframe_rows(
                    self._vae,
                    images,
                    height,
                    width,
                    patch_size,
                )
                try:
                    import mlx.core as mx

                    mx.eval(rows)
                except ImportError:
                    pass
                if check_interrupted is not None:
                    check_interrupted()
            except BaseException:
                self._release_locked()
                raise
            if unload_after:
                self._release_locked()
            return rows

    def encode_references(
        self,
        spec: H3VideoVAESpec,
        references: list[Any],
        *,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
        prepare_stage: Callable[[], None] | None = None,
    ) -> Any:
        """Encode Ref2VA image and video rows with staged VAE residency."""
        spec.validate()
        if prepare_stage is not None:
            prepare_stage()
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if check_interrupted is not None:
                    check_interrupted()
                from minimax_h3_mlx.ref2va import encode_reference_video_rows

                rows = encode_reference_video_rows(self._vae, references, patch_size)
                if rows is not None:
                    import mlx.core as mx

                    mx.eval(rows)
                if check_interrupted is not None:
                    check_interrupted()
            except BaseException:
                self._release_locked()
                raise
            if unload_after:
                self._release_locked()
            return rows

    def _decode_normalized(self, normalized: Any, num_frames: int) -> Any:
        import mlx.core as mx
        import numpy as np

        from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD

        cfg = self._vae.config
        mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        decoded = np.asarray(self._vae.decode((normalized * std + mean).astype(mx.float32)))
        pixel_mean = np.asarray(PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.asarray(PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
        frames = np.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
        frames = frames[0, :, :num_frames].transpose(1, 2, 3, 0)
        return np.ascontiguousarray(frames, dtype=np.float32)

    def decode_stream(
        self,
        spec: H3VideoVAESpec,
        latents: H3Latents,
        write_chunk: Callable[[Any], None],
        *,
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
        prepare_stage: Callable[[], None] | None = None,
    ) -> H3VideoStream:
        """Decode temporal chunks to uint8 RGB and release each chunk after writing."""
        import time

        spec.validate()
        if (
            Path(latents.transformer_spec.video_vae).expanduser()
            != Path(spec.video_vae).expanduser()
        ):
            raise ValueError("Latents were produced for a different MiniMax H3 video VAE.")
        if prepare_stage is not None:
            prepare_stage()
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if latents.generation_config.memory_mode == "low_memory_bf16":
                    self._vae.decode_batch = 1
                started = time.perf_counter()
                frame_count = 0
                width = height = 0
                peak_rgb8_chunk_bytes = 0
                for chunk in self._decode_normalized_chunks(latents.video, latents.num_frames):
                    if check_interrupted is not None:
                        check_interrupted()
                    if chunk.ndim != 4 or chunk.shape[-1] != 3:
                        raise ValueError(
                            "Streamed video chunk must have shape (frames, H, W, 3); "
                            f"got {chunk.shape}."
                        )
                    if frame_count + chunk.shape[0] > latents.num_frames:
                        chunk = chunk[: latents.num_frames - frame_count]
                    if chunk.shape[0] == 0:
                        continue
                    height, width = int(chunk.shape[1]), int(chunk.shape[2])
                    peak_rgb8_chunk_bytes = max(peak_rgb8_chunk_bytes, int(chunk.nbytes))
                    write_chunk(chunk)
                    frame_count += int(chunk.shape[0])
                    del chunk
                if frame_count != latents.num_frames:
                    raise ValueError(
                        f"Streamed video decode produced {frame_count} frames; "
                        f"expected {latents.num_frames}."
                    )
                result = H3VideoStream(
                    num_frames=frame_count,
                    width=width,
                    height=height,
                    fps=latents.fps,
                    decode_seconds=time.perf_counter() - started,
                    decode_batch=int(self._vae.decode_batch),
                    peak_rgb8_chunk_bytes=peak_rgb8_chunk_bytes,
                    quantization=spec.quantization(),
                )
            except BaseException:
                self._release_locked()
                raise
            if unload_after or latents.generation_config.memory_mode == "low_memory_bf16":
                self._release_locked()
            return result

    def _decode_normalized_chunks(self, normalized: Any, num_frames: int):
        import mlx.core as mx
        import numpy as np

        from minimax_h3_mlx.packing import PIXEL_MEAN, PIXEL_STD

        cfg = self._vae.config
        mean = mx.array(np.asarray(cfg.latents_mean, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.asarray(cfg.latents_std, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        denormalized = (normalized * std + mean).astype(mx.float32)
        pixel_mean = mx.array(np.asarray(PIXEL_MEAN, dtype=np.float32)).reshape(1, 3, 1, 1, 1)
        pixel_std = mx.array(np.asarray(PIXEL_STD, dtype=np.float32)).reshape(1, 3, 1, 1, 1)
        emitted = 0
        for decoded in self._vae.decode_chunks(denormalized):
            frames = mx.clip(decoded * pixel_std + pixel_mean, 0.0, 1.0)
            frames = mx.round(frames * 255.0).astype(mx.uint8)
            frames = frames[0].transpose(1, 2, 3, 0)
            remaining = num_frames - emitted
            frames = frames[:remaining]
            emitted += frames.shape[0]
            yield np.ascontiguousarray(np.asarray(frames), dtype=np.uint8)
            if emitted >= num_frames:
                break

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
        prepare_stage: Callable[[], None] | None = None,
    ) -> H3AudioWaveform:
        import time

        spec.validate()
        if (
            Path(latents.transformer_spec.audio_vae).expanduser()
            != Path(spec.audio_vae).expanduser()
        ):
            raise ValueError("Latents were produced for a different MiniMax H3 audio VAE.")
        if prepare_stage is not None:
            prepare_stage()
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

    def encode_references(
        self,
        spec: H3AudioVAESpec,
        references: list[Any],
        *,
        unload_after: bool = True,
        check_interrupted: Callable[[], None] | None = None,
        prepare_stage: Callable[[], None] | None = None,
    ) -> Any:
        """Encode clean Ref2VA soundtrack rows with staged audio-VAE residency."""
        spec.validate()
        if prepare_stage is not None:
            prepare_stage()
        with self._lock:
            if self._vae is None or self._spec != spec:
                self._release_locked()
                self._vae = self._factory(spec)
                self._spec = spec
            try:
                if check_interrupted is not None:
                    check_interrupted()
                from minimax_h3_mlx.ref2va import encode_reference_audio_rows

                rows = encode_reference_audio_rows(self._vae, references)
                if rows is not None:
                    import mlx.core as mx

                    mx.eval(rows)
                if check_interrupted is not None:
                    check_interrupted()
            except BaseException:
                self._release_locked()
                raise
            if unload_after:
                self._release_locked()
            return rows

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
                f"Audio VAE output must have shape (2, 1, samples); got {decoded.shape}."
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
