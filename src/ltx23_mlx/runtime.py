"""Process-local LTX 2.3 pipeline selection and lifecycle management."""

from __future__ import annotations

import gc
import inspect
import math
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .ic_lora import (
    INGREDIENTS_DEV_TRANSFORMER,
    INGREDIENTS_DISTILLED_LORA,
    INGREDIENTS_DISTILLED_LORA_STRENGTH,
    LTX23ICLoRASpec,
    ingredients_distillation_spec,
    install_ic_fusion,
    validate_control_inputs,
    validate_ic_stack,
)
from .lora import LTX23LoRASpec, install_loader, validate_stack

LTX23_CONFIG_MODES = (
    "two_stage",
    "two_stage_hq",
    "distilled",
    "one_stage",
    "distilled_single_stage",
)

LTX23_IC_TOPOLOGIES = (
    "auto",
    "two_stage_clean",
    "control_refine",
    "upsample_only",
    "single_stage",
)


def _has_safetensors(root: Path, stem: str) -> bool:
    return (root / f"{stem}.safetensors").is_file() or any(root.glob(f"{stem}-*-of-*.safetensors"))


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
    if mode == "distilled_single_stage":
        return (*common, "transformer-distilled-1.1")
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
    loras: tuple[LTX23LoRASpec, ...] = ()
    ic_loras: tuple[LTX23ICLoRASpec, ...] = ()

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
                f"LTX 2.3 {mode} model bundle is incomplete under {root}: " + ", ".join(missing)
            )
        validate_stack(self.loras, root, mode)

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
    ic_lora_topology: str = "auto"
    shift: float = 5.0

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
        modulus = 32 if self.pipeline_mode in {"one_stage", "distilled_single_stage"} else 64
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
        if self.pipeline_mode == "distilled_single_stage":
            if self.cfg_scale != 1 or self.stg_scale != 0 or self.stage2_steps != 0:
                raise ValueError("Single-pass distilled requires CFG 1, STG 0, and no refinement")
            if type(self.stage1_steps) is not int or not 1 <= self.stage1_steps <= 100:
                raise ValueError("Single-pass distilled steps must be an integer in [1, 100]")
        if not math.isfinite(self.shift) or not 1 <= self.shift <= 20:
            raise ValueError("LTX 2.3 Shift must be finite and between 1 and 20")
        if self.pipeline_mode != "distilled_single_stage" and self.shift != 5:
            raise ValueError("Shift is only supported for single-pass distilled")
        if self.stage1_steps < 1 or (
            self.stage2_steps < 1 and self.pipeline_mode != "distilled_single_stage"
        ):
            raise ValueError("LTX 2.3 stage step counts must be positive.")
        if self.cfg_scale < 0 or self.stg_scale < 0:
            raise ValueError("LTX 2.3 guidance scales must be zero or positive.")
        if self.ic_lora_topology not in LTX23_IC_TOPOLOGIES:
            raise ValueError(f"Unsupported LTX 2.3 IC-LoRA topology: {self.ic_lora_topology!r}.")
        if (self.num_frames - 1) % 8:
            raise AssertionError("LTX 2.3 frame normalization failed to produce 8n+1 frames.")


def resolve_ic_topology(
    spec: LTX23ModelSpec, config: LTX23GenerationConfig, conditioning_task: str | None
) -> str:
    """Resolve the qualified topology without changing non-control generation."""
    if config.ic_lora_topology != "auto":
        return config.ic_lora_topology
    if spec.ic_loras and spec.ic_loras[0].family == "ingredients_reference_sheet":
        return "two_stage_dev"
    if (
        conditioning_task == "control"
        and spec.ic_loras
        and spec.ic_loras[0].family == "motion_track"
    ):
        return "single_stage"
    return "two_stage_clean"


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
        "keyframe": "KeyframeInterpolationPipeline",
        "a2v": "A2VidPipelineTwoStage",
        "control": "ICLoraPipeline",
        "extension": "RetakePipeline",
    }
    if mode == "distilled_single_stage":
        from .single_stage import LTX23SingleStageDistilledPipeline

        return LTX23SingleStageDistilledPipeline
    if mode == "extension_distilled":
        from .distilled_extension import LTX23DistilledExtendPipeline

        return LTX23DistilledExtendPipeline
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


@contextmanager
def _audio_temporary_cleanup(pipeline, enabled):
    """Reclaim A2V's freshly allocated waveform file when upstream muxing fails."""
    original = getattr(pipeline, "_save_waveform", None)
    if not enabled or original is None:
        yield
        return
    owned = []
    previous = pipeline.__dict__.get("_save_waveform")

    def tracked(waveform, path, *args, **kwargs):
        filename = Path(path)
        # Upstream creates an empty NamedTemporaryFile immediately before this call.
        # Never take ownership of an existing populated waveform or arbitrary output.
        if filename.parent.resolve() == Path(tempfile.gettempdir()).resolve():
            stat = filename.stat()
            if stat.st_size == 0 and filename.suffix == ".wav":
                owned.append((filename, stat.st_dev, stat.st_ino))
        return original(waveform, path, *args, **kwargs)

    pipeline._save_waveform = tracked
    try:
        yield
    finally:
        if previous is None:
            del pipeline._save_waveform
        else:
            pipeline._save_waveform = previous
        for filename, device, inode in owned:
            try:
                stat = filename.stat()
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    filename.unlink()
            except FileNotFoundError:
                pass


class LTX23RuntimeCache:
    """One optional LTX pipeline instance with explicit unload behavior."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._key: tuple[object, ...] | None = None
        self._pipeline: Any = None
        self._previous_cache_limit: int | None = None
        self._lora_reports: list[dict] = []
        self._ic_reports: list[dict] = []
        self._auxiliary_lora_reports: list[dict] = []

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._pipeline is not None

    def get(self, spec: LTX23ModelSpec, config: LTX23GenerationConfig, *, conditioning_task=None):
        config.validate()
        validate_ic_stack(spec, config)
        if config.pipeline_mode == "distilled_single_stage" and conditioning_task:
            raise ValueError("Single-pass distilled currently supports text-to-video only")
        ic_topology = resolve_ic_topology(spec, config, conditioning_task)
        if conditioning_task == "extension" and (
            config.pipeline_mode not in {"one_stage", "distilled"} or spec.ic_loras
        ):
            raise ValueError(
                "LTX 2.3 extension requires Dev one_stage or distilled without IC-LoRAs"
            )
        spec.validate(config.pipeline_mode)
        key = (
            spec,
            config.pipeline_mode,
            config.low_memory,
            config.low_ram_streaming,
            conditioning_task,
            ic_topology,
        )
        with self._lock:
            if self._pipeline is None or self._key != key:
                self._release_locked()
                pipeline_mode = (
                    "extension_distilled"
                    if conditioning_task == "extension" and config.pipeline_mode == "distilled"
                    else conditioning_task or config.pipeline_mode
                )
                pipeline_class = _pipeline_class(pipeline_mode)
                if config.low_ram_streaming:
                    import mlx.core as mx

                    self._previous_cache_limit = int(mx.set_cache_limit(0))
                try:
                    task_options = (
                        {
                            "dev_transformer": "transformer-dev.safetensors",
                            "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
                        }
                        if conditioning_task in {"keyframe", "a2v"}
                        else {}
                    )
                    if conditioning_task == "control":
                        task_options = {
                            "lora_paths": [
                                (str(Path(s.path).expanduser().resolve()), s.strength)
                                for s in spec.ic_loras
                            ]
                        }
                        if spec.ic_loras[0].family == "ingredients_reference_sheet":
                            task_options.update(
                                dev_transformer=INGREDIENTS_DEV_TRANSFORMER,
                                distilled_lora=INGREDIENTS_DISTILLED_LORA,
                                distilled_lora_strength=INGREDIENTS_DISTILLED_LORA_STRENGTH,
                            )
                    if conditioning_task == "extension" and config.pipeline_mode == "one_stage":
                        task_options = {"dev_transformer": "transformer-dev.safetensors"}
                    constructor_options = {
                        "model_dir": str(spec.root()),
                        "gemma_model_id": str(spec.gemma_root()),
                        "low_memory": config.low_memory,
                        **task_options,
                    }
                    if "low_ram_streaming" in inspect.signature(pipeline_class.__init__).parameters:
                        constructor_options["low_ram_streaming"] = config.low_ram_streaming
                    self._pipeline = pipeline_class(**constructor_options)
                    if spec.loras:
                        self._lora_reports = install_loader(self._pipeline, spec.loras)
                    if spec.ic_loras:
                        self._pipeline._weetodd_ic_stages = {
                            "two_stage_dev": "all_two_stage",
                            "two_stage_clean": "stage1_only",
                            "control_refine": "stage1_and_control_refine",
                            "upsample_only": "stage1_only",
                            "single_stage": "all_single_stage",
                        }[ic_topology]
                        auxiliary_specs = (
                            (ingredients_distillation_spec(spec),)
                            if ic_topology == "two_stage_dev"
                            else ()
                        )
                        self._ic_reports = install_ic_fusion(
                            self._pipeline, spec.ic_loras, auxiliary_specs
                        )
                        self._auxiliary_lora_reports = getattr(
                            self._pipeline, "_weetodd_auxiliary_lora_reports", []
                        )
                except BaseException:
                    self._pipeline = None
                    self._lora_reports = []
                    self._ic_reports = []
                    self._auxiliary_lora_reports = []
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
        image_inputs: list[dict[str, object]] | None = None,
        audio_path: str | None = None,
        control_inputs: list[dict[str, object]] | None = None,
        extension_input: dict[str, object] | None = None,
        unload_after: bool = True,
        check_interrupted=None,
        step_callback=None,
    ) -> dict[str, object]:
        if not prompt.strip():
            raise ValueError("LTX 2.3 prompt must not be empty.")
        config.validate()
        validate_ic_stack(spec, config)
        conditioning_task = (
            "extension"
            if extension_input is not None
            else "control"
            if control_inputs
            else "a2v"
            if audio_path is not None
            else "keyframe"
            if image_inputs
            else None
        )
        if config.pipeline_mode == "distilled_single_stage" and (
            conditioning_task or image_path is not None or spec.ic_loras
        ):
            raise ValueError("Single-pass distilled currently supports text-to-video only")
        ic_topology = resolve_ic_topology(spec, config, conditioning_task)
        if spec.ic_loras or control_inputs is not None:
            if image_path is not None or image_inputs or audio_path is not None:
                raise ValueError("Combined LTX 2.3 IC-LoRA/keyframe/audio inputs are not qualified")
            ingredients = spec.ic_loras[0].family == "ingredients_reference_sheet"
            if ingredients and ic_topology != "two_stage_dev":
                raise ValueError(
                    "LTX 2.3 Ingredients requires ic_lora_topology=auto with the resident "
                    "Dev two_stage pipeline"
                )
            validate_control_inputs(spec, config, control_inputs, inspect_files=not ingredients)
            signature = inspect.signature(_pipeline_class("control").generate_and_save)
            if "video_conditioning" not in signature.parameters:
                raise ValueError("Installed IC pipeline does not consume video_conditioning")
        if conditioning_task in {"keyframe", "a2v"}:
            if config.pipeline_mode != "two_stage":
                raise ValueError("LTX 2.3 keyframe and A2V tasks require Dev two_stage")
            if image_path is not None or (audio_path is not None and image_inputs):
                raise ValueError(
                    "Combined LTX 2.3 audio/keyframe/initial-image inputs are not qualified"
                )
            seen = set()
            for item in image_inputs or ():
                if set(item) != {"path", "frame_index", "strength"}:
                    raise ValueError("LTX 2.3 keyframes require path, frame_index, and strength")
                index = item["frame_index"]
                if type(index) is not int or not 0 <= index < config.num_frames or index in seen:
                    raise ValueError(
                        "LTX 2.3 keyframe indices must be unique and inside the output"
                    )
                seen.add(index)
                if not 0 <= float(item["strength"]) <= 1 or not Path(item["path"]).is_file():
                    raise ValueError(
                        "LTX 2.3 keyframe requires an existing image and strength in [0, 1]"
                    )
            if audio_path is not None and not Path(audio_path).is_file():
                raise FileNotFoundError(f"LTX 2.3 audio not found: {audio_path}")
            # Inspect the actual consuming signature, not **kwargs that may discard inputs.
            signature = inspect.signature(_pipeline_class(conditioning_task).generate_and_save)
            required = (
                {"audio_path"}
                if audio_path is not None
                else {
                    "keyframe_images",
                    "keyframe_indices",
                    "keyframe_strengths",
                    "video_guider_params",
                }
            )
            if required - signature.parameters.keys():
                raise ValueError(
                    "Installed LTX 2.3 pipeline does not consume the requested conditioning"
                )
        extension_info = None
        if conditioning_task == "extension":
            if image_path is not None or image_inputs or audio_path is not None or control_inputs:
                raise ValueError("LTX 2.3 extension cannot be combined with other conditioning")
            if config.pipeline_mode not in {"one_stage", "distilled"}:
                raise ValueError(
                    "LTX 2.3 extension requires the Dev one_stage or distilled pipeline"
                )
            if config.pipeline_mode == "distilled" and config.stage1_steps != 8:
                raise ValueError("LTX 2.3 distilled extension requires exactly eight steps")
            if spec.ic_loras:
                raise ValueError("LTX 2.3 extension with IC-LoRAs is not qualified")
            if set(extension_input) != {"path", "direction", "additional_frames"}:
                raise ValueError(
                    "LTX 2.3 extension requires path, direction, and additional_frames"
                )
            if extension_input["direction"] not in {"before", "after"}:
                raise ValueError("LTX 2.3 extension direction must be before or after")
            additional_frames = extension_input["additional_frames"]
            if (
                type(additional_frames) is not int
                or not 8 <= additional_frames <= 720
                or additional_frames % 8
            ):
                raise ValueError(
                    "LTX 2.3 extension additional_frames must be a multiple of 8 in [8, 720]"
                )
            source = Path(str(extension_input["path"])).expanduser()
            if not source.is_file():
                raise FileNotFoundError(f"LTX 2.3 extension source not found: {source}")
            from ltx_core_mlx.utils.ffmpeg import probe_video_info

            extension_info = probe_video_info(str(source))
            if (
                extension_info.width != config.width
                or extension_info.height != config.height
                or abs(extension_info.fps - config.frame_rate) > 0.001
                or extension_info.num_frames != config.num_frames
            ):
                raise ValueError(
                    "LTX 2.3 extension config must exactly match source width, height, fps, "
                    "and 8n+1 frame count"
                )
            if (extension_info.num_frames - 1) % 8:
                raise ValueError("LTX 2.3 extension source must contain exactly 8n+1 frames")
            output_frames = extension_info.num_frames + additional_frames
            if (output_frames - 1) / extension_info.fps > 30:
                raise ValueError("LTX 2.3 extended output must not exceed 30 seconds")
        if conditioning_task != "control" and ic_topology != "two_stage_clean":
            raise ValueError("Alternate IC-LoRA topology requires an LTX 2.3 control task")
        spec.validate(config.pipeline_mode)
        if check_interrupted is not None:
            check_interrupted()
        try:
            import mlx.core as mx

            mx.reset_peak_memory()
        except (ImportError, AttributeError):
            mx = None
        started = time.perf_counter()
        # Refinement mutates weights in place. Never reuse an adapter-baked instance
        # as a fresh stage-one model on a later request.
        if spec.loras or conditioning_task:
            self.unload()
            unload_after = True
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
        if config.pipeline_mode == "distilled_single_stage":
            kwargs.update(num_steps=config.stage1_steps, shift=config.shift)
        elif config.pipeline_mode == "one_stage":
            kwargs.update(
                num_steps=config.stage1_steps,
                cfg_scale=config.cfg_scale,
                stg_scale=config.stg_scale,
            )
        else:
            kwargs.update(stage1_steps=config.stage1_steps, stage2_steps=config.stage2_steps)
            if config.pipeline_mode in {"two_stage", "two_stage_hq"}:
                kwargs.update(cfg_scale=config.cfg_scale, stg_scale=config.stg_scale)
        if conditioning_task == "keyframe":
            from ltx_core_mlx.components.guiders import MultiModalGuiderParams

            ordered = sorted(image_inputs, key=lambda item: item["frame_index"])
            kwargs.update(
                keyframe_images=[str(i["path"]) for i in ordered],
                keyframe_indices=[i["frame_index"] for i in ordered],
                keyframe_strengths=[i["strength"] for i in ordered],
                video_guider_params=MultiModalGuiderParams(
                    cfg_scale=config.cfg_scale,
                    stg_scale=config.stg_scale,
                    stg_blocks=[28],
                ),
            )
        elif conditioning_task == "a2v":
            kwargs["audio_path"] = str(audio_path)
        elif conditioning_task == "control":
            kwargs["video_conditioning"] = [(str(i["path"]), i["strength"]) for i in control_inputs]
            if ic_topology == "control_refine":
                kwargs.update(upsample_only=True, refine_steps=config.stage2_steps)
            elif ic_topology == "upsample_only":
                kwargs["upsample_only"] = True
            elif ic_topology == "single_stage":
                kwargs["single_stage"] = True
        pipeline = self.get(spec, config, conditioning_task=conditioning_task)
        if spec.loras or spec.ic_loras:
            pipeline._weetodd_check_interrupted = check_interrupted
        accepted = {}
        if conditioning_task != "extension":
            signature = inspect.signature(pipeline.generate_and_save)
            accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
        required_inputs = {
            key
            for key in (
                "image",
                "audio_path",
                "keyframe_images",
                "keyframe_indices",
                "keyframe_strengths",
                "video_conditioning",
            )
            if kwargs.get(key) is not None
        }
        if conditioning_task != "extension" and required_inputs - accepted.keys():
            self.unload()
            raise ValueError("Installed LTX 2.3 pipeline cannot consume requested conditioning")
        succeeded = False
        expected_steps = config.stage1_steps
        if (
            conditioning_task != "extension"
            and config.pipeline_mode not in {"one_stage", "distilled_single_stage"}
            and ic_topology
            in {
                "two_stage_dev",
                "two_stage_clean",
                "control_refine",
            }
        ):
            expected_steps += config.stage2_steps
        try:
            with (
                self._lock,
                _comfy_sampler_progress(
                    check_interrupted,
                    step_callback,
                    expected_steps,
                ),
                _audio_temporary_cleanup(pipeline, conditioning_task == "a2v"),
            ):
                if conditioning_task == "extension":
                    video_latent, audio_latent = pipeline.extend_from_video(
                        prompt=prompt,
                        video_path=str(extension_input["path"]),
                        extend_frames=extension_input["additional_frames"] // 8,
                        direction=extension_input["direction"],
                        seed=config.seed,
                        num_steps=config.stage1_steps,
                        cfg_scale=config.cfg_scale,
                        stg_scale=config.stg_scale,
                    )
                    pipeline._load_decoders()
                    result_path = pipeline._decode_and_save_video(
                        video_latent,
                        audio_latent,
                        str(output_path),
                        frame_rate=extension_info.fps,
                    )
                else:
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
                "sampling": (
                    {
                        "schedule": "linear_trailing",
                        "shift": config.shift,
                        "evaluations": config.stage1_steps,
                        "audio_cfg": 1,
                        "video_cfg": 1,
                        "transformer": "transformer-distilled-1.1",
                    }
                    if config.pipeline_mode == "distilled_single_stage"
                    else None
                ),
                "ic_lora_topology_effective": (
                    "none"
                    if extension_input is not None
                    or config.pipeline_mode == "distilled_single_stage"
                    else ic_topology
                ),
                "conditioning_task": conditioning_task,
                "audio_policy": (
                    "source_latent_reconstructed_and_generated_extension"
                    if extension_input is not None
                    else "source_resampled_48khz"
                    if audio_path is not None
                    else "generated"
                ),
                "extension": (
                    {
                        "direction": extension_input["direction"],
                        "source_frames": extension_info.num_frames,
                        "additional_frames": extension_input["additional_frames"],
                        "output_frames": extension_info.num_frames
                        + extension_input["additional_frames"],
                    }
                    if extension_input is not None
                    else None
                ),
                "extension_sampler": (
                    "distilled_positive_only"
                    if extension_input is not None and config.pipeline_mode == "distilled"
                    else "dev_cfg_stg"
                    if extension_input is not None
                    else None
                ),
                "model_dir": spec.root().name,
                "gemma_model": spec.gemma_root().name,
                "loras": list(self._lora_reports),
                "ic_loras": list(self._ic_reports),
                "auxiliary_loras": list(self._auxiliary_lora_reports),
                "mlx_peak_bytes": peak,
                "total_seconds": time.perf_counter() - started,
                "runtime_cached": not unload_after,
                "weighted_components_may_be_resident": (not unload_after and not config.low_memory),
            }
        finally:
            if unload_after or not succeeded:
                self.unload()

    def unload(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        for source in getattr(self._pipeline, "_weetodd_lora_sources", ()):
            close = getattr(source, "close", None)
            if callable(close):
                close()
        self._pipeline = None
        self._key = None
        self._lora_reports = []
        self._ic_reports = []
        self._auxiliary_lora_reports = []
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
