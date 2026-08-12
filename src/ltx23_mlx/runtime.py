"""Process-local LTX 2.3 pipeline selection and lifecycle management."""

from __future__ import annotations

import gc
import inspect
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

LTX23_CONFIG_MODES = (
    "two_stage",
    "two_stage_hq",
    "distilled",
    "one_stage",
)


def _has_safetensors(root: Path, stem: str) -> bool:
    return (root / f"{stem}.safetensors").is_file() or any(
        root.glob(f"{stem}-*-of-*.safetensors")
    )


def _required_files(mode: str) -> tuple[str, ...]:
    common = (
        "connector.safetensors",
        "vae_encoder.safetensors",
        "vae_decoder.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    )
    if mode in {"two_stage", "two_stage_hq"}:
        return (
            *common,
            "transformer-dev",
            "ltx-2.3-22b-distilled-lora-384",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x2_v1_1",
        )
    if mode == "distilled":
        return (
            *common,
            "transformer-distilled",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x2_v1_1",
        )
    return (*common, "transformer-dev")


@dataclass(frozen=True)
class LTX23ModelSpec:
    """Local model bundle and Gemma text-encoder selection."""

    model_dir: str
    gemma_model: str = "mlx-community/gemma-3-12b-it-4bit"

    def root(self) -> Path:
        return Path(self.model_dir).expanduser()

    def gemma_root(self) -> Path:
        """Resolve Gemma locally without permitting an implicit model download."""
        path = Path(self.gemma_model).expanduser()
        if path.is_dir():
            return path
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(self.gemma_model, local_files_only=True))
        except Exception as exc:
            raise FileNotFoundError(
                "LTX 2.3 Gemma model is not a local directory or a complete cached "
                f"snapshot: {self.gemma_model!r}. Install it before running ComfyUI."
            ) from exc

    def validate(self, mode: str | None = None) -> None:
        root = self.root()
        if not root.is_dir():
            raise FileNotFoundError(f"LTX 2.3 model directory not found: {root}")
        if mode is None:
            return
        self.gemma_root()
        if mode not in LTX23_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.3 pipeline mode: {mode!r}.")
        missing = []
        for name in _required_files(mode):
            if name.endswith((".safetensors", ".json")):
                present = (root / name).is_file()
            else:
                present = _has_safetensors(root, name)
            if not present:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"LTX 2.3 {mode} model bundle is incomplete under {root}: "
                + ", ".join(missing)
            )

    def inventory(self, mode: str) -> dict[str, object]:
        root = self.root()
        entries = []
        total = 0
        for name in _required_files(mode):
            if name.endswith((".safetensors", ".json")):
                paths = [root / name] if (root / name).is_file() else []
            else:
                single = root / f"{name}.safetensors"
                paths = (
                    [single]
                    if single.is_file()
                    else sorted(root.glob(f"{name}-*-of-*.safetensors"))
                )
            size = sum(path.stat().st_size for path in paths)
            total += size
            entries.append(
                {"component": name, "files": [path.name for path in paths], "bytes": size}
            )
        return {"components": entries, "checkpoint_bytes": total}


@dataclass(frozen=True)
class LTX23GenerationConfig:
    """Validated user-facing generation settings for LTX 2.3."""

    pipeline_mode: str = "two_stage"
    width: int = 704
    height: int = 448
    duration_seconds: float = 5.0
    frame_rate: float = 24.0
    seed: int = 0
    stage1_steps: int = 30
    stage2_steps: int = 3
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    low_memory: bool = True
    low_ram_streaming: bool = False

    @property
    def num_frames(self) -> int:
        intervals = max(1, round(self.duration_seconds * self.frame_rate / 8.0))
        return intervals * 8 + 1

    @property
    def delivered_duration_seconds(self) -> float:
        return (self.num_frames - 1) / self.frame_rate

    def validate(self) -> None:
        if self.pipeline_mode not in LTX23_CONFIG_MODES:
            raise ValueError(f"Unsupported LTX 2.3 pipeline mode: {self.pipeline_mode!r}.")
        modulus = 32 if self.pipeline_mode == "one_stage" else 64
        if self.width < modulus or self.height < modulus:
            raise ValueError(f"LTX 2.3 dimensions must be at least {modulus} pixels.")
        if self.width % modulus or self.height % modulus:
            raise ValueError(
                f"LTX 2.3 {self.pipeline_mode} dimensions must be divisible by {modulus}."
            )
        if self.width > 1920 or self.height > 1920:
            raise ValueError("LTX 2.3 dimensions must not exceed 1920 pixels.")
        if not 0.25 <= self.duration_seconds <= 30.0:
            raise ValueError("LTX 2.3 duration must be between 0.25 and 30 seconds.")
        if not 1.0 <= self.frame_rate <= 60.0:
            raise ValueError("LTX 2.3 frame rate must be between 1 and 60 fps.")
        if self.stage1_steps < 1 or self.stage2_steps < 1:
            raise ValueError("LTX 2.3 stage step counts must be positive.")
        if self.cfg_scale < 0 or self.stg_scale < 0:
            raise ValueError("LTX 2.3 guidance scales must be zero or positive.")
        if (self.num_frames - 1) % 8:
            raise AssertionError("LTX 2.3 frame normalization failed to produce 8n+1 frames.")


def _pipeline_class(mode: str):
    try:
        import ltx_pipelines_mlx
    except ImportError as exc:
        raise ImportError(
            "LTX 2.3 support is optional. Install this project with its 'ltx' extra "
            "using the same Python interpreter that runs ComfyUI."
        ) from exc
    names = {
        "two_stage": "TI2VidTwoStagesPipeline",
        "two_stage_hq": "TI2VidTwoStagesHQPipeline",
        "distilled": "DistilledPipeline",
        "one_stage": "TI2VidOneStagePipeline",
    }
    try:
        return getattr(ltx_pipelines_mlx, names[mode])
    except AttributeError as exc:
        raise ImportError(
            "The installed ltx-2-mlx revision is too old for WeeTodd LTX 2.3 support."
        ) from exc


@contextmanager
def _comfy_sampler_progress(check_interrupted, step_callback, expected_steps: int):
    """Bridge upstream sampler iteration to Comfy cancellation and progress."""
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


class LTX23RuntimeCache:
    """One optional LTX pipeline instance with explicit unload behavior."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[object, ...] | None = None
        self._pipeline: Any = None
        self._previous_cache_limit: int | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._pipeline is not None

    def get(self, spec: LTX23ModelSpec, config: LTX23GenerationConfig):
        config.validate()
        spec.validate(config.pipeline_mode)
        key = (
            spec,
            config.pipeline_mode,
            config.low_memory,
            config.low_ram_streaming,
        )
        with self._lock:
            if self._pipeline is None or self._key != key:
                self._release_locked()
                pipeline_class = _pipeline_class(config.pipeline_mode)
                if config.low_ram_streaming:
                    import mlx.core as mx

                    self._previous_cache_limit = int(mx.set_cache_limit(0))
                try:
                    self._pipeline = pipeline_class(
                        model_dir=str(spec.root()),
                        gemma_model_id=str(spec.gemma_root()),
                        low_memory=config.low_memory,
                        low_ram_streaming=config.low_ram_streaming,
                    )
                except BaseException:
                    if self._previous_cache_limit is not None:
                        mx.set_cache_limit(self._previous_cache_limit)
                        self._previous_cache_limit = None
                    raise
                self._key = key
            return self._pipeline

    def generate_to_file(
        self,
        spec: LTX23ModelSpec,
        config: LTX23GenerationConfig,
        prompt: str,
        output_path: str | Path,
        *,
        image_path: str | None = None,
        unload_after: bool = True,
        check_interrupted=None,
        step_callback=None,
    ) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("LTX 2.3 prompt must not be empty.")
        config.validate()
        spec.validate(config.pipeline_mode)
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
        }
        if config.pipeline_mode == "one_stage":
            kwargs.update(
                num_steps=config.stage1_steps,
                cfg_scale=config.cfg_scale,
                stg_scale=config.stg_scale,
            )
        else:
            kwargs.update(stage1_steps=config.stage1_steps, stage2_steps=config.stage2_steps)
            if config.pipeline_mode in {"two_stage", "two_stage_hq"}:
                kwargs.update(cfg_scale=config.cfg_scale, stg_scale=config.stg_scale)
        signature = inspect.signature(pipeline.generate_and_save)
        accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        succeeded = False
        expected_steps = config.stage1_steps
        if config.pipeline_mode != "one_stage":
            expected_steps += config.stage2_steps
        try:
            with self._lock, _comfy_sampler_progress(
                check_interrupted,
                step_callback,
                expected_steps,
            ):
                result_path = pipeline.generate_and_save(**accepted)
            if check_interrupted is not None:
                check_interrupted()
            peak = int(mx.get_peak_memory()) if mx is not None else None
            succeeded = True
            return {
                "prompt": prompt,
                "video_path": str(result_path),
                "generation": asdict(config),
                "num_frames": config.num_frames,
                "delivered_duration_seconds": config.delivered_duration_seconds,
                "pipeline_mode": config.pipeline_mode,
                "model_dir": spec.root().name,
                "gemma_model": spec.gemma_root().name,
                "mlx_peak_bytes": peak,
                "total_seconds": time.perf_counter() - started,
                "runtime_cached": not unload_after,
                "weighted_components_may_be_resident": (
                    not unload_after and not config.low_memory
                ),
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


RUNTIME = LTX23RuntimeCache()
