"""Versioned LTX 2.5 split-component contracts and MLX lifecycle management."""

from __future__ import annotations

import gc
import inspect
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from safetensors import safe_open

from .gemma_pack import inspect_gemma4_pack

LTX25_CONFIG_MODES = ("distilled",)
LTX25_PROMPT_CONTEXTS = ("official_1024", "auto", "128", "256", "512", "1024")
LTX25_FEED_FORWARD_BACKENDS = ("reference_fp32", "bf16_mpp_experimental")
LTX25_GENERATION_PRESETS = (
    "Custom",
    "Official parity — 768×512, 5 s, reference FP32, 8+3 ancestral",
)
LTX25_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX25_STAGE2_SIGMAS = (0.85, 0.725, 0.421875, 0.0)


def _metadata(path: Path) -> dict[str, object]:
    """Read and JSON-decode a safetensors metadata header without loading weights."""
    with safe_open(path, framework="numpy") as handle:
        raw = handle.metadata() or {}
    parsed: dict[str, object] = {}
    for key, value in raw.items():
        try:
            parsed[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed[key] = value
    return parsed


def _version_tuple(value: object) -> tuple[int, ...]:
    parts = []
    for item in str(value or "").replace("-", ".").split("."):
        if item.isdigit():
            parts.append(int(item))
        elif parts:
            break
    return tuple(parts)


def _scale_factors(video_vae_metadata: dict[str, object]) -> tuple[int, int, int]:
    config = video_vae_metadata.get("config")
    if not isinstance(config, dict):
        return (8, 32, 32)
    vae = config.get("vae")
    if not isinstance(vae, dict):
        return (8, 32, 32)
    blocks = vae.get("encoder_blocks") or vae.get("decoder_blocks")
    if not isinstance(blocks, list):
        return (8, 32, 32)
    spatial_steps = 0
    temporal_steps = 0
    for entry in blocks:
        if not isinstance(entry, (list, tuple)) or not entry:
            continue
        name = str(entry[0])
        if name.startswith(("compress_space", "compress_all")):
            spatial_steps += 1
        if name.startswith(("compress_time", "compress_all")):
            temporal_steps += 1
    patch_size = int(vae.get("patch_size", 4))
    spatial = patch_size * (2**spatial_steps)
    return (2**temporal_steps, spatial, spatial)


def _decoder_kind(video_vae_metadata: dict[str, object]) -> str:
    config = video_vae_metadata.get("config")
    vae = config.get("vae", {}) if isinstance(config, dict) else {}
    class_name = str(vae.get("_class_name", "")) if isinstance(vae, dict) else ""
    return "diffusion" if "diffusion" in class_name.lower() else "convolutional"


def _transformer_architecture(metadata: dict[str, object]) -> dict[str, object]:
    config = metadata.get("config")
    transformer = config.get("transformer", {}) if isinstance(config, dict) else {}
    if not isinstance(transformer, dict):
        transformer = {}
    return {
        "num_layers": int(transformer.get("num_layers", 48)),
        "use_prompt_adaln_single": bool(transformer.get("use_prompt_adaln_single", True)),
        "ff_bias": bool(transformer.get("ff_bias", True)),
        "audio_ff_bias": bool(transformer.get("audio_ff_bias", True)),
        "caption_proj_before_connector": bool(
            transformer.get("caption_proj_before_connector", False)
        ),
        "cross_attention_adaln": bool(transformer.get("cross_attention_adaln", False)),
        "use_keyframes_abs_pos_embedding": bool(
            transformer.get("use_keyframes_abs_pos_embedding", False)
        ),
        "frequencies_precision": str(transformer.get("frequencies_precision", "float32")),
    }


@dataclass(frozen=True)
class LTX25ComponentSpec:
    """Explicit LTX 2.5 split components; no implicit downloads are permitted."""

    transformer_path: str
    text_encoder_path: str
    video_vae_path: str
    audio_vae_path: str
    spatial_upscaler_path: str
    duration_head_path: str = ""

    def paths(self) -> dict[str, Path]:
        values = asdict(self)
        return {name: Path(value).expanduser() for name, value in values.items() if value}

    def validate(self, mode: str = "distilled") -> dict[str, object]:
        if mode not in LTX25_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.5 pipeline mode: {mode!r}.")
        required = {
            "transformer_path",
            "text_encoder_path",
            "video_vae_path",
            "audio_vae_path",
            "spatial_upscaler_path",
        }
        paths = self.paths()
        missing = [
            name for name in sorted(required) if name not in paths or not paths[name].is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "LTX 2.5 split component files are missing: " + ", ".join(missing)
            )
        for name, path in paths.items():
            if path.suffix != ".safetensors":
                raise ValueError(f"LTX 2.5 {name} must be a .safetensors file: {path}")

        transformer_meta = _metadata(paths["transformer_path"])
        text_meta = _metadata(paths["text_encoder_path"])
        gemma_pack = inspect_gemma4_pack(paths["text_encoder_path"])
        video_vae_meta = _metadata(paths["video_vae_path"])
        model_version = transformer_meta.get("model_version")
        if _version_tuple(model_version) < (2, 5):
            raise ValueError(
                "The selected transformer is not identified as LTX 2.5 or newer; "
                f"model_version={model_version!r}. Preserve upstream checkpoint metadata."
            )

        transformer_gemma = transformer_meta.get("gemma_source_checkpoint")
        text_gemma = text_meta.get("gemma_source_checkpoint")
        expected_gemma_version = (
            transformer_gemma.get("gemma_version") if isinstance(transformer_gemma, dict) else None
        )
        packed_gemma_version = gemma_pack.get("gemma_version")
        if expected_gemma_version and packed_gemma_version != expected_gemma_version:
            raise ValueError(
                "The LTX 2.5 transformer and Gemma 4 pack declare different Gemma versions: "
                f"{expected_gemma_version!r} != {packed_gemma_version!r}."
            )
        if transformer_gemma and text_gemma and transformer_gemma != text_gemma:
            raise ValueError(
                "The LTX 2.5 transformer and Gemma 4 component declare different sources."
            )
        architecture = _transformer_architecture(transformer_meta)
        incompatible = []
        if architecture["use_prompt_adaln_single"] is not True:
            incompatible.append("use_prompt_adaln_single=true")
        if architecture["ff_bias"] is not False:
            incompatible.append("ff_bias=false")
        if architecture["audio_ff_bias"] is not True:
            incompatible.append("audio_ff_bias=true")
        if architecture["cross_attention_adaln"] is not True:
            incompatible.append("cross_attention_adaln=true")
        if architecture["caption_proj_before_connector"] is not True:
            incompatible.append("caption_proj_before_connector=true")
        if architecture["use_keyframes_abs_pos_embedding"] is not True:
            incompatible.append("use_keyframes_abs_pos_embedding=true")
        if incompatible:
            raise ValueError(
                "The selected LTX 2.5 distilled transformer metadata does not declare the "
                "required architecture: " + ", ".join(incompatible)
            )

        scale_factors = _scale_factors(video_vae_meta)
        inventory = []
        total = 0
        for name, path in paths.items():
            size = path.stat().st_size
            total += size
            inventory.append({"component": name, "file": path.name, "bytes": size})
        return {
            "model_version": str(model_version),
            "gemma_source_checkpoint": transformer_gemma or text_gemma,
            "gemma_pack": gemma_pack,
            "video_scale_factors": list(scale_factors),
            "video_decoder": _decoder_kind(video_vae_meta),
            "transformer_architecture": architecture,
            "components": inventory,
            "checkpoint_bytes": total,
        }


@dataclass(frozen=True)
class LTX25GenerationConfig:
    """Validated distilled LTX 2.5 generation settings."""

    pipeline_mode: str = "distilled"
    width: int = 768
    height: int = 512
    duration_seconds: float = 5.0
    frame_rate: float = 24.0
    seed: int = 0
    stage1_steps: int = 8
    stage2_steps: int = 3
    stage1_sampler: str = "euler_ancestral"
    stage2_sampler: str = "euler_ancestral"
    stage1_eta: float = 1.0
    stage1_s_noise: float = 1.0
    ancestral_seed_offset: int = 10000
    low_memory: bool = True
    low_ram_streaming: bool = False
    prompt_context: str = "official_1024"
    feed_forward_backend: str = "reference_fp32"

    @property
    def num_frames(self) -> int:
        intervals = max(1, round(self.duration_seconds * self.frame_rate / 8.0))
        return intervals * 8 + 1

    @property
    def delivered_duration_seconds(self) -> float:
        return (self.num_frames - 1) / self.frame_rate

    def validate(self, *, scale_factors: tuple[int, int, int] = (8, 32, 32)) -> None:
        if self.pipeline_mode not in LTX25_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.5 pipeline mode: {self.pipeline_mode!r}.")
        if self.prompt_context not in LTX25_PROMPT_CONTEXTS:
            raise ValueError(f"Unsupported LTX 2.5 prompt context mode: {self.prompt_context!r}.")
        if self.feed_forward_backend not in LTX25_FEED_FORWARD_BACKENDS:
            raise ValueError(
                f"Unsupported LTX 2.5 feed-forward backend: {self.feed_forward_backend!r}."
            )
        if self.low_ram_streaming and self.feed_forward_backend != "reference_fp32":
            raise ValueError(
                "LTX 2.5 BF16 MPP feed-forward mode is not compatible with low-RAM block "
                "streaming. Select reference_fp32 or disable low_ram_streaming."
            )
        temporal, spatial_h, spatial_w = scale_factors
        modulus_h = spatial_h * 2
        modulus_w = spatial_w * 2
        if self.width < modulus_w or self.height < modulus_h:
            raise ValueError("LTX 2.5 distilled dimensions are below the two-stage latent grid.")
        if self.width % modulus_w or self.height % modulus_h:
            raise ValueError(
                f"LTX 2.5 distilled dimensions must be divisible by {modulus_w}x{modulus_h}."
            )
        if self.width > 1920 or self.height > 1920:
            raise ValueError("LTX 2.5 dimensions must not exceed 1920 pixels.")
        if not 0.25 <= self.duration_seconds <= 30.0:
            raise ValueError("LTX 2.5 duration must be between 0.25 and 30 seconds.")
        if not 1.0 <= self.frame_rate <= 60.0:
            raise ValueError("LTX 2.5 frame rate must be between 1 and 60 fps.")
        if self.stage1_steps != 8 or self.stage2_steps != 3:
            raise ValueError(
                "The LTX 2.5 distilled path requires eight stage-one and three stage-two "
                "transformer evaluations."
            )
        if (
            self.stage1_sampler != "euler_ancestral"
            or self.stage2_sampler != "euler_ancestral"
        ):
            raise ValueError(
                "LTX 2.5 distilled requires Euler ancestral sampling in both stages."
            )
        if self.stage1_eta != 1.0 or self.stage1_s_noise != 1.0:
            raise ValueError("LTX 2.5 distilled stage one requires eta=1.0 and s_noise=1.0.")
        if self.ancestral_seed_offset != 10000:
            raise ValueError("LTX 2.5 distilled requires the stage-one noise seed offset 10000.")
        if (self.num_frames - 1) % temporal:
            raise ValueError(
                f"LTX 2.5 frame count must align to temporal compression factor {temporal}."
            )


def apply_ltx25_generation_preset(name: str, values: dict[str, object]) -> dict[str, object]:
    """Apply a named LTX 2.5 recipe without changing the selected seed."""
    if name not in LTX25_GENERATION_PRESETS:
        raise ValueError(f"Unsupported LTX 2.5 generation preset: {name!r}.")
    resolved = dict(values)
    if name == LTX25_GENERATION_PRESETS[1]:
        resolved.update(
            width=768,
            height=512,
            duration_seconds=5.0,
            frame_rate=24.0,
            low_memory=True,
            low_ram_streaming=False,
            prompt_context="official_1024",
            feed_forward_backend="reference_fp32",
        )
    return resolved


def _pipeline_class():
    from .pipeline import LTX25DistilledPipeline

    return LTX25DistilledPipeline


def backend_capability() -> dict[str, object]:
    """Report whether the installed optional backend exposes the versioned entry point."""
    try:
        pipeline = _pipeline_class()
    except ImportError as exc:
        try:
            from importlib.metadata import version

            installed = version("ltx-pipelines-mlx")
        except Exception:
            installed = None
        return {
            "ready": False,
            "installed_ltx_pipelines_mlx": installed,
            "project_ancestral_sampler": True,
            "project_gemma4_conditioner": True,
            "remaining_engine_gates": ["transformer", "video_vae", "audio_vae", "vocoder"],
            "reason": str(exc),
        }
    return {"ready": True, "pipeline_class": pipeline.__name__}


@contextmanager
def _comfy_sampler_progress(check_interrupted, step_callback, expected_steps: int):
    if check_interrupted is None and step_callback is None:
        yield
        return
    from ltx_pipelines_mlx.utils import samplers

    original = samplers.tqdm
    completed = 0

    def iter_with_callbacks(iterable, *_args, **_kwargs):
        nonlocal completed
        for item in iterable:
            if check_interrupted is not None:
                check_interrupted()
            yield item
            completed += 1
            if step_callback is not None:
                step_callback(completed, max(expected_steps, completed))

    samplers.tqdm = iter_with_callbacks
    try:
        yield
    finally:
        samplers.tqdm = original


class LTX25RuntimeCache:
    """One process-local LTX 2.5 pipeline with explicit staged release."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[object, ...] | None = None
        self._pipeline: Any = None
        self._previous_cache_limit: int | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._pipeline is not None

    def get(self, spec: LTX25ComponentSpec, config: LTX25GenerationConfig):
        report = spec.validate(config.pipeline_mode)
        scales = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(scale_factors=scales)
        key = (
            spec,
            config.low_memory,
            config.low_ram_streaming,
            config.feed_forward_backend,
        )
        with self._lock:
            if self._pipeline is None or self._key != key:
                self._release_locked()
                pipeline_class = _pipeline_class()
                if config.low_ram_streaming:
                    import mlx.core as mx

                    self._previous_cache_limit = int(mx.set_cache_limit(0))
                kwargs = {
                    **{name: str(path) for name, path in spec.paths().items()},
                    "low_memory": config.low_memory,
                    "low_ram_streaming": config.low_ram_streaming,
                    "feed_forward_backend": config.feed_forward_backend,
                }
                signature = inspect.signature(pipeline_class)
                accepted = {
                    key: value for key, value in kwargs.items() if key in signature.parameters
                }
                try:
                    self._pipeline = pipeline_class(**accepted)
                except BaseException:
                    if self._previous_cache_limit is not None:
                        mx.set_cache_limit(self._previous_cache_limit)
                        self._previous_cache_limit = None
                    raise
                self._key = key
            return self._pipeline

    def generate_to_file(
        self,
        spec: LTX25ComponentSpec,
        config: LTX25GenerationConfig,
        prompt: str,
        output_path: str | Path,
        *,
        image_path: str | None = None,
        unload_after: bool = True,
        check_interrupted=None,
        step_callback=None,
    ) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("LTX 2.5 prompt must not be empty.")
        report = spec.validate(config.pipeline_mode)
        scales = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(scale_factors=scales)
        if check_interrupted is not None:
            check_interrupted()
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            mx = None
        started = time.perf_counter()
        pipeline = self.get(spec, config)
        kwargs: dict[str, object] = {
            "prompt": prompt,
            "output_path": str(output_path),
            "height": config.height,
            "width": config.width,
            "num_frames": config.num_frames,
            "frame_rate": config.frame_rate,
            "seed": config.seed,
            "image": image_path,
            "stage1_steps": config.stage1_steps,
            "stage2_steps": config.stage2_steps,
            "stage1_sigmas": LTX25_DISTILLED_SIGMAS,
            "stage2_sigmas": LTX25_STAGE2_SIGMAS,
            "stage1_sampler": config.stage1_sampler,
            "stage2_sampler": config.stage2_sampler,
            "stage1_eta": config.stage1_eta,
            "stage1_s_noise": config.stage1_s_noise,
            "ancestral_noise_seed": config.seed + config.ancestral_seed_offset,
            "check_interrupted": check_interrupted,
            "step_callback": step_callback,
            "prompt_context": config.prompt_context,
        }
        signature = inspect.signature(pipeline.generate_and_save)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        accepted = {
            key: value
            for key, value in kwargs.items()
            if accepts_kwargs or key in signature.parameters
        }
        succeeded = False
        try:
            with (
                self._lock,
                _comfy_sampler_progress(
                    check_interrupted, step_callback, config.stage1_steps + config.stage2_steps
                ),
            ):
                result_path = pipeline.generate_and_save(**accepted)
            if check_interrupted is not None:
                check_interrupted()
            succeeded = True
            try:
                from .feed_forward import feed_forward_runtime_status

                feed_forward_runtime = feed_forward_runtime_status()
            except (ImportError, AttributeError):
                feed_forward_runtime = None
            return {
                "prompt": prompt,
                "video_path": str(result_path),
                "generation": asdict(config),
                "num_frames": config.num_frames,
                "delivered_duration_seconds": config.delivered_duration_seconds,
                "pipeline_mode": config.pipeline_mode,
                "model_version": report["model_version"],
                "video_decoder": report["video_decoder"],
                "video_scale_factors": report["video_scale_factors"],
                "mlx_peak_bytes": int(mx.get_peak_memory()) if mx is not None else None,
                "total_seconds": time.perf_counter() - started,
                "stage_timings": getattr(pipeline, "last_timings", {}),
                "resolved_prompt_context": getattr(pipeline, "last_prompt_context", None),
                "feed_forward_backend": getattr(pipeline, "feed_forward_report", None),
                "feed_forward_runtime": feed_forward_runtime,
                "runtime_cached": not unload_after,
            }
        finally:
            if unload_after or not succeeded:
                self.unload()

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        self._pipeline = None
        self._key = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
            if self._previous_cache_limit is not None:
                mx.set_cache_limit(self._previous_cache_limit)
                self._previous_cache_limit = None
        except (ImportError, AttributeError):
            pass


RUNTIME = LTX25RuntimeCache()
