"""Classic ComfyUI node contracts backed by the MLX MiniMax H3 pipeline."""

import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
from .decoding import (
    AUDIO_VAE_RUNTIME,
    VIDEO_VAE_RUNTIME,
    H3AudioVAESpec,
    H3VideoVAESpec,
)
from .direct_publishing import publish_latents_direct
from .preflight import H3ComponentSetSpec, H3PreflightRequest, preflight_components
from .publishing import publish_synchronized_media
from .residency import prepare_low_memory_stage
from .runtime import RUNTIME, H3GenerationConfig, H3ModelSpec
from .sampling import TRANSFORMER_RUNTIME, H3TransformerSpec


def _lora_choices():
    try:
        import folder_paths

        choices = folder_paths.get_filename_list("loras")
        return choices or [""]
    except ImportError:
        return [""]


def _resolve_lora_path(name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute() or path.exists():
        return path
    try:
        import folder_paths

        resolved = folder_paths.get_full_path("loras", name)
        if resolved:
            return Path(resolved)
        return Path(folder_paths.models_dir) / "loras" / name
    except ImportError:
        return path


def _output_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path.cwd() / "output"


def _safe_output_target(output_directory: Path, filename_prefix: str, seed: int) -> Path:
    """Resolve a user prefix below ComfyUI's output directory."""
    prefix = Path(filename_prefix.replace("\\", "/"))
    if prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError(
            "filename_prefix must be a relative path inside ComfyUI's output directory"
        )
    if not prefix.name or prefix.name in {".", ".."}:
        raise ValueError("filename_prefix must include a filename")
    root = output_directory.resolve()
    target = (root / prefix.parent / f"{prefix.name}_{seed}.mp4").resolve()
    if target != root and root not in target.parents:
        raise ValueError("filename_prefix resolves outside ComfyUI's output directory")
    return target


def _resolve_component_root(checkpoint: str) -> str:
    """Resolve a relative H3 root below ComfyUI's model directory when available."""
    path = Path(checkpoint).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    try:
        import folder_paths

        return str(Path(folder_paths.models_dir) / path)
    except ImportError:
        return str(path)


_H3_RESOLUTION_SHORT_EDGES = {
    "384P (fast smoke)": 384,
    "512P (balanced)": 512,
    "768P (native quality)": 768,
    "2K (experimental, very high memory)": 1152,
}
_H3_ASPECT_RATIOS = {
    "21:9": (21, 9),
    "16:9": (16, 9),
    "5:3": (5, 3),
    "3:2": (3, 2),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "2:3": (2, 3),
    "3:5": (3, 5),
    "9:16": (9, 16),
    "9:21": (9, 21),
}


def _resolve_h3_resolution(
    mode: str,
    resolution_tier: str,
    aspect_ratio: str,
    custom_width: int,
    custom_height: int,
) -> tuple[int, int]:
    """Resolve an intuitive tier and ratio selection to the H3 32-pixel grid."""
    if mode == "custom":
        return custom_width, custom_height
    if mode != "preset":
        raise ValueError("Resolution mode must be 'preset' or 'custom'.")
    try:
        short_edge = _H3_RESOLUTION_SHORT_EDGES[resolution_tier]
    except KeyError as exc:
        raise ValueError(f"Unknown H3 resolution tier: {resolution_tier!r}.") from exc
    try:
        ratio_width, ratio_height = _H3_ASPECT_RATIOS[aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unknown H3 aspect ratio: {aspect_ratio!r}.") from exc
    if aspect_ratio == "16:9":
        if short_edge == 1152:
            return 2048, 1152
        return short_edge * 7 // 4, short_edge
    if aspect_ratio == "9:16":
        if short_edge == 1152:
            return 1152, 2048
        return short_edge, short_edge * 7 // 4
    if ratio_width >= ratio_height:
        height = short_edge
        width = round(short_edge * ratio_width / ratio_height / 32) * 32
    else:
        width = short_edge
        height = round(short_edge * ratio_height / ratio_width / 32) * 32
    return width, height


class WeeToddH3ComponentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {"default": "MiniMax-H3/FL2VA"}),
                "task": (["t2va", "fl2va", "ref2va"], {"default": "t2va"}),
            },
            "optional": {
                "transformer": ("STRING", {"default": ""}),
                "text_encoder": ("STRING", {"default": ""}),
                "processor": ("STRING", {"default": ""}),
                "tokenizer": ("STRING", {"default": ""}),
                "video_vae": ("STRING", {"default": ""}),
                "audio_vae": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_COMPONENTS",)
    RETURN_NAMES = ("components",)
    FUNCTION = "specify"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = "Describe every MiniMax H3 component. This node does not load tensor weights."

    def specify(
        self,
        checkpoint,
        task,
        transformer="",
        text_encoder="",
        processor="",
        tokenizer="",
        video_vae="",
        audio_vae="",
    ):
        return (
            H3ComponentSetSpec(
                checkpoint=_resolve_component_root(checkpoint),
                task=task,
                transformer=transformer or None,
                text_encoder=text_encoder or None,
                processor=processor or None,
                tokenizer=tokenizer or None,
                video_vae=video_vae or None,
                audio_vae=audio_vae or None,
            ),
        )


class WeeToddH3QuantizedTransformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "profile": (
                    ["q8_conservative", "q8_extended"],
                    {"default": "q8_conservative"},
                ),
                "transformer_root": (
                    "STRING",
                    {"default": "MiniMax-H3/transformers"},
                ),
            },
            "optional": {
                "transformer_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_COMPONENTS", "STRING")
    RETURN_NAMES = ("components", "profile_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Select and validate a named mixed-precision H3 transformer without loading weights. "
        "Both q8 profiles are approximate and keep BlockCache disabled by default."
    )

    def select(self, components, profile, transformer_root, transformer_override=""):
        from minimax_h3_mlx.mixed_checkpoint import validate_named_q8_checkpoint

        if transformer_override.strip():
            transformer = Path(_resolve_component_root(transformer_override))
        else:
            root = Path(_resolve_component_root(transformer_root))
            transformer = root / profile
        info = validate_named_q8_checkpoint(transformer, profile)
        selected = replace(components, transformer=str(transformer))
        return (selected, json.dumps(info, indent=2, sort_keys=True))


class WeeToddH3Preflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "prompt_tokens": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "available_memory_gb": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.25},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_COMPONENTS", "STRING")
    RETURN_NAMES = ("components", "preflight_report")
    FUNCTION = "inspect"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Validate MiniMax H3 components and estimate staged memory from file headers. "
        "Set available memory to zero when unknown."
    )

    def inspect(
        self,
        components,
        config,
        prompt_tokens,
        available_memory_gb,
    ):
        config.validate()
        report = preflight_components(
            components,
            H3PreflightRequest(
                duration_seconds=config.duration_seconds,
                steps=config.steps,
                width=config.width,
                height=config.height,
                prompt_tokens=prompt_tokens,
                available_memory_gb=available_memory_gb,
            ),
        )
        return components, report.to_json()


class WeeToddH3TextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "unload_after_encode": ("BOOLEAN", {"default": True}),
            },
            "optional": {"config": ("WEETODD_H3_CONFIG",)},
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Encode a text-only H3 prompt with Qwen3-VL. The vision tower stays unloaded. "
        "The encoder can unload after it produces conditioning."
    )

    def encode(self, components, prompt, unload_after_encode, config=None):
        memory_mode = getattr(config, "memory_mode", "normal")
        staged_releases = ()

        def prepare_stage():
            nonlocal staged_releases
            staged_releases = prepare_low_memory_stage("text_encoder", memory_mode)

        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        if check_interrupted is not None:
            check_interrupted()
        conditioning = TEXT_ENCODER_RUNTIME.encode(
            H3TextEncoderSpec.from_components(components, load_vision=False),
            prompt,
            unload_after=unload_after_encode or memory_mode == "low_memory_bf16",
            prepare_stage=prepare_stage,
        )
        if check_interrupted is not None:
            check_interrupted()
        info = {
            "token_count": conditioning.token_count,
            "vision_loaded": conditioning.load_vision,
            "encoder_resident": TEXT_ENCODER_RUNTIME.loaded,
            "memory_mode": memory_mode,
            "staged_releases": list(staged_releases),
        }
        return conditioning, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3UnloadTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = "Release the process-local Qwen3-VL conditioner and clear the MLX cache."

    def release(self, unload):
        if unload:
            TEXT_ENCODER_RUNTIME.unload()
            return ("MiniMax H3 Qwen3-VL conditioner unloaded",)
        return ("MiniMax H3 Qwen3-VL conditioner kept warm",)


class WeeToddH3Sample:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "conditioning": ("WEETODD_H3_CONDITIONING",),
                "config": ("WEETODD_H3_CONFIG",),
                "unload_after_sample": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "easycache": ("WEETODD_H3_EASYCACHE",),
                "blockcache": ("WEETODD_H3_BLOCKCACHE",),
                "trajectory_forecast": ("WEETODD_H3_TRAJECTORY_FORECAST",),
                "loras": ("WEETODD_H3_LORAS",),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_LATENTS", "STRING")
    RETURN_NAMES = ("latents", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Sample synchronized MiniMax H3 video and audio latents with MLX. "
        "This node does not load or run either VAE."
    )

    def sample(
        self,
        components,
        conditioning,
        config,
        unload_after_sample,
        easycache=None,
        blockcache=None,
        trajectory_forecast=None,
        loras=None,
    ):
        if sum(value is not None for value in (easycache, blockcache, trajectory_forecast)) > 1:
            raise ValueError("Connect only one of EasyCache, BlockCache, or Trajectory Forecast.")
        staged_releases = ()

        def prepare_stage():
            nonlocal staged_releases
            staged_releases = prepare_low_memory_stage("transformer", config.memory_mode)

        progress = None
        check_interrupted = None
        try:
            import comfy.model_management
            import comfy.utils

            progress = comfy.utils.ProgressBar(config.steps - 1)
            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass

        def on_step(completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None:
                progress.update_absolute(completed, total)

        latents = TRANSFORMER_RUNTIME.sample(
            H3TransformerSpec.from_components(components),
            conditioning,
            config,
            unload_after=unload_after_sample,
            step_callback=on_step,
            easycache=easycache,
            blockcache=blockcache,
            trajectory_forecast=trajectory_forecast,
            loras=loras,
            prepare_stage=prepare_stage,
        )
        info = {
            "prompt": conditioning.prompt,
            "frames": latents.num_frames,
            "width": latents.width,
            "height": latents.height,
            "fps": latents.fps,
            "sample_rate": latents.sample_rate,
            "transformer_evaluations": latents.transformer_evaluations,
            "easycache_skipped_steps": latents.easycache_skipped_steps,
            "easycache_resolved_threshold": latents.easycache_resolved_threshold,
            "easycache": asdict(easycache) if easycache is not None else None,
            "blockcache_hits": getattr(latents, "blockcache_hits", 0),
            "blockcache_resolved_threshold": getattr(
                latents, "blockcache_resolved_threshold", None
            ),
            "blockcache_cache_bytes": getattr(latents, "blockcache_cache_bytes", 0),
            "blockcache_segment_hits": list(
                getattr(latents, "blockcache_segment_hits", ())
            ),
            "blockcache_segment_thresholds": list(
                getattr(latents, "blockcache_segment_thresholds", ())
            ),
            "blockcache_executed_blocks": getattr(
                latents, "blockcache_executed_blocks", 0
            ),
            "blockcache_skipped_blocks": getattr(
                latents, "blockcache_skipped_blocks", 0
            ),
            "blockcache": asdict(blockcache) if blockcache is not None else None,
            "trajectory_forecasts": getattr(latents, "trajectory_forecasts", 0),
            "trajectory_bootstrap_forecasts": getattr(
                latents, "trajectory_bootstrap_forecasts", 0
            ),
            "trajectory_fallbacks": getattr(latents, "trajectory_fallbacks", 0),
            "trajectory_history_bytes": getattr(latents, "trajectory_history_bytes", 0),
            "trajectory_forecast": (
                asdict(trajectory_forecast) if trajectory_forecast is not None else None
            ),
            "loras": loras.metadata() if loras is not None else [],
            "lora_report": list(getattr(latents, "lora_report", ())),
            "seconds_per_evaluation": latents.seconds_per_evaluation,
            "total_seconds": latents.total_seconds,
            "transformer_resident": TRANSFORMER_RUNTIME.loaded,
            "memory_mode": config.memory_mode,
            "attention_query_chunk_size": config.attention_query_chunk_size,
            "compute_dtype": "bfloat16",
            "projection_backend": getattr(latents, "projection_backend_report", None),
            "projection_backend_runtime": getattr(
                latents, "projection_backend_runtime", None
            ),
            "preview_policy": "none",
            "staged_releases": list(staged_releases),
        }
        return latents, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3LoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (_lora_choices(),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "profile": (["auto", "standard", "turbo"], {"default": "auto"}),
                "adaln_input_grid": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Required when a LoRA targets original AdaLN weights but the selected "
                            "H3 transformer uses a pruned AdaLN curve."
                        ),
                    },
                ),
            },
            "optional": {"previous_loras": ("WEETODD_H3_LORAS",)},
        }

    RETURN_TYPES = ("WEETODD_H3_LORAS", "STRING")
    RETURN_NAMES = ("loras", "lora_info")
    FUNCTION = "load"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Build a lazy, ordered MiniMax H3 LoRA stack. Validate safetensors headers now and load "
        "adapter tensors only when the H3 transformer executes."
    )

    def load(self, lora_name, strength, profile, adaln_input_grid="", previous_loras=None):
        from .lora import H3LoRASpec, H3LoRAStack

        path = _resolve_lora_path(lora_name)
        grid = _resolve_lora_path(adaln_input_grid) if adaln_input_grid.strip() else None
        if grid is None:
            candidate = path.parent / "h3_silu_temb_grid.safetensors"
            grid = candidate if candidate.is_file() else None
        spec = H3LoRASpec(
            path=str(path),
            strength=strength,
            profile=profile,
            adaln_input_grid=str(grid) if grid is not None else None,
        )
        stack = (previous_loras or H3LoRAStack()).append(spec)
        info = {
            "file": path.name,
            "strength": strength,
            "profile": spec.resolved_profile,
            "tensor_bytes": spec.tensor_bytes,
            "adaln_input_grid": grid.name if grid is not None else None,
            "stack_size": len(stack.adapters),
            "loads_at_sampling": True,
        }
        return stack, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3EasyCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "manual",
                        "automatic_conservative",
                        "automatic_balanced",
                        "automatic_speed",
                    ],
                    {"default": "manual"},
                ),
                "reuse_threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 3.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "auto_multiplier": (
                    "FLOAT",
                    {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "max_skip_fraction": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_EASYCACHE",)
    RETURN_NAMES = ("easycache",)
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Configure joint MLX EasyCache residual reuse for H3 video and audio sampling. "
        "Choose quality-first, balanced, or speed-first bounded automatic reuse."
    )

    def configure(
        self,
        mode,
        reuse_threshold,
        start_percent,
        end_percent,
        auto_multiplier,
        max_skip_fraction,
    ):
        from minimax_h3_mlx.easycache import H3EasyCacheConfig

        config = H3EasyCacheConfig(
            mode=mode,
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            auto_multiplier=auto_multiplier,
            max_skip_fraction=max_skip_fraction,
        )
        config.validate()
        return (config,)


class WeeToddH3TrajectoryForecast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "manual",
                        "automatic_conservative",
                        "automatic_balanced",
                        "automatic_speed",
                    ],
                    {"default": "automatic_balanced"},
                ),
                "forecast_strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "warmup_steps": ("INT", {"default": 2, "min": 2, "max": 20}),
                "tail_actual_steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                "max_history": ("INT", {"default": 2, "min": 2, "max": 2}),
                "max_forecast_fraction": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 0.5, "step": 0.05},
                ),
                "max_delta_ratio": (
                    "FLOAT",
                    {"default": 1.75, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
            },
            "optional": {
                "bootstrap_first_forecast": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental speed-mode zero-order hold for the second sampling step. "
                            "It changes output and does not increase the total forecast budget."
                        ),
                    },
                )
            },
        }

    RETURN_TYPES = ("WEETODD_H3_TRAJECTORY_FORECAST",)
    RETURN_NAMES = ("trajectory_forecast",)
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Experimentally forecast compact post-transformer H3 video and audio features. "
        "Current timestep output heads still run on every step. Turbo LoRA is supported."
    )

    def configure(
        self,
        mode,
        forecast_strength,
        warmup_steps,
        tail_actual_steps,
        max_history,
        max_forecast_fraction,
        max_delta_ratio,
        bootstrap_first_forecast=False,
    ):
        from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig

        config = H3TrajectoryForecastConfig(
            mode=mode,
            forecast_strength=forecast_strength,
            warmup_steps=warmup_steps,
            tail_actual_steps=tail_actual_steps,
            max_history=max_history,
            max_forecast_fraction=max_forecast_fraction,
            max_delta_ratio=max_delta_ratio,
            bootstrap_first_forecast=bootstrap_first_forecast,
        )
        config.validate()
        return (config,)


class WeeToddH3BlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "manual",
                        "automatic_conservative",
                        "automatic_balanced",
                        "automatic_speed",
                    ],
                    {"default": "automatic_balanced"},
                ),
                "reuse_threshold": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "auto_multiplier": (
                    "FLOAT",
                    {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.05},
                ),
                "max_hit_fraction": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 0.6, "step": 0.05},
                ),
                "allow_turbo_experimental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental: permit BlockCache with a Turbo LoRA. This combines "
                            "two approximations and may change motion, detail, or audio."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_BLOCKCACHE",)
    RETURN_NAMES = ("blockcache",)
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Always run H3 block zero and the current output heads, then safely reuse the cached "
        "joint audio/video residual of later transformer blocks when both modality indicators "
        "agree."
    )

    def configure(
        self,
        mode,
        reuse_threshold,
        start_percent,
        end_percent,
        auto_multiplier,
        max_hit_fraction,
        allow_turbo_experimental=False,
    ):
        from minimax_h3_mlx.blockcache import H3BlockCacheConfig

        config = H3BlockCacheConfig(
            mode=mode,
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            auto_multiplier=auto_multiplier,
            max_hit_fraction=max_hit_fraction,
            allow_turbo_experimental=allow_turbo_experimental,
        )
        config.validate()
        return (config,)


class WeeToddH3HierarchicalBlockCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    [
                        "automatic_conservative",
                        "automatic_balanced",
                        "automatic_speed",
                    ],
                    {"default": "automatic_balanced"},
                ),
                "allow_turbo_experimental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Permit hierarchical BlockCache with Turbo. Each segment remains an "
                            "independent approximation and may change motion, detail, or audio."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_BLOCKCACHE",)
    RETURN_NAMES = ("blockcache",)
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Split the 50 H3 blocks into three contiguous segments. Always evaluate each segment's "
        "anchor block, accept video and audio together, and reuse eligible segment tails "
        "independently."
    )

    def configure(self, mode, allow_turbo_experimental=False):
        from minimax_h3_mlx.blockcache import H3HierarchicalBlockCacheConfig

        config = H3HierarchicalBlockCacheConfig(
            mode=mode,
            allow_turbo_experimental=allow_turbo_experimental,
        )
        config.validate()
        return (config,)


class WeeToddH3UnloadTransformer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = "Release the process-local H3 transformer and clear the MLX cache."

    def release(self, unload):
        if unload:
            TRANSFORMER_RUNTIME.unload()
            return ("MiniMax H3 transformer unloaded",)
        return ("MiniMax H3 transformer kept warm",)


class WeeToddH3VideoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "latents": ("WEETODD_H3_LATENTS",),
                "unload_after_decode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "decode_info")
    FUNCTION = "decode"
    CATEGORY = "WeeTodd/H3/decoding"
    DESCRIPTION = (
        "Decode the video stream from synchronized H3 latents with the final video VAE. "
        "The audio latent stream remains available on the original latent output."
    )

    def decode(self, components, latents, unload_after_decode):
        staged_releases = ()

        def prepare_stage():
            nonlocal staged_releases
            staged_releases = prepare_low_memory_stage(
                "video_vae", latents.generation_config.memory_mode
            )

        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        result = VIDEO_VAE_RUNTIME.decode(
            H3VideoVAESpec.from_components(components),
            latents,
            unload_after=unload_after_decode,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_stage,
        )
        import torch

        frames = torch.from_numpy(result.frames)
        info = {
            "frames": result.num_frames,
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
            "decode_seconds": result.decode_seconds,
            "video_vae_resident": VIDEO_VAE_RUNTIME.loaded,
            "video_vae_quantization": result.quantization,
            "memory_mode": latents.generation_config.memory_mode,
            "tile_decode_batch": result.decode_batch,
            "staged_releases": list(staged_releases),
        }
        return frames, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3UnloadVideoVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3/decoding"
    DESCRIPTION = "Release the process-local H3 video VAE and clear the MLX cache."

    def release(self, unload):
        if unload:
            VIDEO_VAE_RUNTIME.unload()
            return ("MiniMax H3 video VAE unloaded",)
        return ("MiniMax H3 video VAE kept warm",)


class WeeToddH3AudioVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "latents": ("WEETODD_H3_LATENTS",),
                "unload_after_decode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "decode_info")
    FUNCTION = "decode"
    CATEGORY = "WeeTodd/H3/decoding"
    DESCRIPTION = (
        "Decode the audio stream from synchronized H3 latents as 32 kHz stereo audio. "
        "The video latent stream remains available on the original latent output."
    )

    def decode(self, components, latents, unload_after_decode):
        staged_releases = ()

        def prepare_stage():
            nonlocal staged_releases
            staged_releases = prepare_low_memory_stage(
                "audio_vae", latents.generation_config.memory_mode
            )

        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        result = AUDIO_VAE_RUNTIME.decode(
            H3AudioVAESpec.from_components(components),
            latents,
            unload_after=unload_after_decode,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_stage,
        )
        import torch

        audio = {
            "waveform": torch.from_numpy(result.waveform).unsqueeze(0),
            "sample_rate": result.sample_rate,
        }
        info = {
            "channels": result.channels,
            "num_samples": result.num_samples,
            "sample_rate": result.sample_rate,
            "duration_seconds": result.duration_seconds,
            "video_frames": result.video_frames,
            "fps": result.fps,
            "decode_seconds": result.decode_seconds,
            "audio_vae_resident": AUDIO_VAE_RUNTIME.loaded,
            "memory_mode": latents.generation_config.memory_mode,
            "staged_releases": list(staged_releases),
        }
        return audio, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3UnloadAudioVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3/decoding"
    DESCRIPTION = "Release the process-local H3 audio VAE and clear the MLX cache."

    def release(self, unload):
        if unload:
            AUDIO_VAE_RUNTIME.unload()
            return ("MiniMax H3 audio VAE unloaded",)
        return ("MiniMax H3 audio VAE kept warm",)


class WeeToddH3PublishVideoAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "filename_prefix": ("STRING", {"default": "WeeTodd/H3"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "max_av_drift_seconds": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.001},
                ),
            },
            "optional": {
                "generation_metadata": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
                "sampling_info": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "publish"
    CATEGORY = "WeeTodd/H3/output"
    DESCRIPTION = (
        "Validate and publish synchronized H3 images and 32 kHz stereo audio as MP4. "
        "The node writes an atomic JSON metadata sidecar."
    )

    def publish(
        self,
        images,
        audio,
        components,
        config,
        filename_prefix,
        crf,
        max_av_drift_seconds,
        generation_metadata="{}",
        sampling_info="",
    ):
        config.validate()
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("Audio must contain waveform and sample_rate fields.")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                "ComfyUI IMAGE input must have shape (frames, height, width, 3); "
                f"got {tuple(images.shape)}."
            )
        if images.shape[1] != config.height or images.shape[2] != config.width:
            raise ValueError(
                "Decoded image dimensions do not match the generation configuration: "
                f"{images.shape[2]}x{images.shape[1]} != {config.width}x{config.height}."
            )
        from minimax_h3_mlx.packing import align_num_frames

        expected_frames = align_num_frames(int(round(config.duration_seconds * 24)))
        if images.shape[0] != expected_frames:
            raise ValueError(
                "Decoded frame count does not match the generation configuration: "
                f"{images.shape[0]} != {expected_frames}."
            )
        waveform = audio["waveform"]
        if waveform.ndim != 3 or waveform.shape[0] != 1:
            raise ValueError(
                "ComfyUI AUDIO waveform must have shape (1, channels, samples); "
                f"got {tuple(waveform.shape)}."
            )

        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        if check_interrupted is not None:
            check_interrupted()

        import torch

        if not torch.isfinite(images).all():
            raise ValueError("Image input contains non-finite values. Decode the video again.")
        video = (
            images.detach()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .contiguous()
            .numpy()
        )
        host_audio = waveform.detach().cpu().float().contiguous().numpy()[0]
        try:
            supplied_metadata = json.loads(generation_metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Generation metadata must be a valid JSON object.") from exc
        if not isinstance(supplied_metadata, dict):
            raise ValueError("Generation metadata must be a JSON object.")
        if sampling_info:
            try:
                supplied_metadata["sampling"] = json.loads(sampling_info)
            except json.JSONDecodeError as exc:
                raise ValueError("Sampling information must be valid JSON.") from exc
        component_paths = components.resolved_paths()
        try:
            package_version = version("comfyui-weetodd-nodes")
        except PackageNotFoundError:
            package_version = "uninstalled"
        try:
            mlx_version = version("mlx")
        except PackageNotFoundError:
            mlx_version = "uninstalled"
        metadata = {
            **supplied_metadata,
            "generation": asdict(config),
            "precision_policy": (
                "component-specific checkpoint precision; verify quantization in preflight"
            ),
            "components": {
                "checkpoint": Path(components.checkpoint).name,
                "task": components.task,
                **{name: path.name for name, path in component_paths.items()},
            },
            "software": {
                "python": platform.python_version(),
                "mlx": mlx_version,
                "weetodd_nodes": package_version,
            },
        }
        target = _safe_output_target(_output_directory(), filename_prefix, config.seed)
        result = publish_synchronized_media(
            target,
            video,
            host_audio,
            sample_rate=int(audio["sample_rate"]),
            fps=24.0,
            crf=crf,
            max_av_drift_seconds=max_av_drift_seconds,
            generation_metadata=json.dumps(metadata),
            check_interrupted=check_interrupted,
        )
        info = json.dumps(result.metadata, indent=2, sort_keys=True)
        output_root = _output_directory().resolve()
        relative = result.video_path.resolve().relative_to(output_root)
        preview = {
            "filename": relative.name,
            "subfolder": str(relative.parent) if str(relative.parent) != "." else "",
            "type": "output",
            "format": "video/mp4",
        }
        return {
            "ui": {"gifs": [preview]},
            "result": (str(result.video_path), info),
        }


class WeeToddH3DirectPublishLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "latents": ("WEETODD_H3_LATENTS",),
                "filename_prefix": ("STRING", {"default": "WeeTodd/H3_direct"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "max_av_drift_seconds": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.001},
                ),
            },
            "optional": {
                "generation_metadata": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
                "sampling_info": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "publish"
    CATEGORY = "WeeTodd/H3/output"
    DESCRIPTION = (
        "Decode synchronized H3 latents directly to MP4 through staged MLX VAEs. "
        "The node avoids a persistent ComfyUI IMAGE tensor and unloads each VAE after use."
    )

    def publish(
        self,
        components,
        latents,
        filename_prefix,
        crf,
        max_av_drift_seconds,
        generation_metadata="{}",
        sampling_info="",
    ):
        config = latents.generation_config
        config.validate()
        try:
            supplied_metadata = json.loads(generation_metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Generation metadata must be a valid JSON object.") from exc
        if not isinstance(supplied_metadata, dict):
            raise ValueError("Generation metadata must be a JSON object.")
        if sampling_info:
            try:
                supplied_metadata["sampling"] = json.loads(sampling_info)
            except json.JSONDecodeError as exc:
                raise ValueError("Sampling information must be valid JSON.") from exc

        component_paths = components.resolved_paths()
        try:
            package_version = version("comfyui-weetodd-nodes")
        except PackageNotFoundError:
            package_version = "uninstalled"
        try:
            mlx_version = version("mlx")
        except PackageNotFoundError:
            mlx_version = "uninstalled"
        metadata = {
            **supplied_metadata,
            "generation": asdict(config),
            "precision_policy": (
                "component-specific checkpoint precision; verify quantization in preflight"
            ),
            "components": {
                "checkpoint": Path(components.checkpoint).name,
                "task": components.task,
                **{name: path.name for name, path in component_paths.items()},
            },
            "software": {
                "python": platform.python_version(),
                "mlx": mlx_version,
                "weetodd_nodes": package_version,
            },
        }
        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass

        video_releases = ()
        audio_releases = ()

        def prepare_video_stage():
            nonlocal video_releases
            video_releases = prepare_low_memory_stage("video_vae", config.memory_mode)

        def prepare_audio_stage():
            nonlocal audio_releases
            audio_releases = prepare_low_memory_stage("audio_vae", config.memory_mode)

        target = _safe_output_target(_output_directory(), filename_prefix, config.seed)
        result = publish_latents_direct(
            target,
            components,
            latents,
            crf=crf,
            max_av_drift_seconds=max_av_drift_seconds,
            generation_metadata=json.dumps(metadata),
            check_interrupted=check_interrupted,
            prepare_video_stage=prepare_video_stage,
            prepare_audio_stage=prepare_audio_stage,
            metadata_updates=lambda: {
                "staged_releases": {
                    "video": list(video_releases),
                    "audio": list(audio_releases),
                }
            },
        )
        info = json.dumps(result.metadata, indent=2, sort_keys=True)
        output_root = _output_directory().resolve()
        relative = result.video_path.resolve().relative_to(output_root)
        preview = {
            "filename": relative.name,
            "subfolder": str(relative.parent) if str(relative.parent) != "." else "",
            "type": "output",
            "format": "video/mp4",
        }
        return {
            "ui": {"gifs": [preview]},
            "result": (str(result.video_path), info),
        }


class WeeToddH3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {"default": "models/MiniMax-H3/FL2VA"}),
                "transformer": ("STRING", {"default": ""}),
                "load_vision": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "WeeTodd/H3"
    DESCRIPTION = "Describe an MLX MiniMax H3 checkpoint. Weights load lazily at generation time."

    def load(self, checkpoint, transformer, load_vision):
        return (H3ModelSpec(checkpoint, transformer or None, load_vision),)


class WeeToddH3GenerationConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.1},
                ),
                "steps": ("INT", {"default": 16, "min": 2, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "resolution_mode": (["preset", "custom"], {"default": "preset"}),
                "resolution_tier": (
                    list(_H3_RESOLUTION_SHORT_EDGES),
                    {"default": "768P (native quality)"},
                ),
                "aspect_ratio": (list(_H3_ASPECT_RATIOS), {"default": "16:9"}),
                "custom_width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 4096, "step": 32, "advanced": True},
                ),
                "custom_height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 4096, "step": 32, "advanced": True},
                ),
                "drop_adaln": ("BOOLEAN", {"default": True}),
                "memory_mode": (
                    ["normal", "low_memory_bf16"],
                    {"default": "normal", "advanced": True},
                ),
                "attention_chunk_size": (
                    ["automatic", "512", "1024", "2048"],
                    {
                        "default": "automatic",
                        "advanced": True,
                        "tooltip": (
                            "Used only by low_memory_bf16. Automatic selects the measured "
                            "512-row policy. Larger values are diagnostic overrides."
                        ),
                    },
                ),
                "projection_backend": (
                    ["mlx", "mpp_experimental"],
                    {
                        "default": "mlx",
                        "advanced": True,
                        "tooltip": (
                            "Experimental Metal Performance Primitives acceleration for eligible "
                            "BF16 transformer projections. Unsupported projections use MLX."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONFIG", "STRING")
    RETURN_NAMES = ("config", "resolved_resolution")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3"

    DESCRIPTION = (
        "Choose an H3 quality tier and aspect ratio, or use exact custom dimensions. "
        "The node resolves the canvas to the required 32-pixel grid."
    )

    def configure(
        self,
        duration_seconds,
        steps,
        seed,
        resolution_mode,
        resolution_tier,
        aspect_ratio,
        custom_width,
        custom_height,
        drop_adaln,
        memory_mode="normal",
        attention_chunk_size="automatic",
        projection_backend="mlx",
    ):
        width, height = _resolve_h3_resolution(
            resolution_mode,
            resolution_tier,
            aspect_ratio,
            custom_width,
            custom_height,
        )
        config = H3GenerationConfig(
            duration_seconds=duration_seconds,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
            drop_adaln=drop_adaln,
            resolution_mode=resolution_mode,
            resolution_tier=resolution_tier if resolution_mode == "preset" else "custom",
            aspect_ratio=aspect_ratio if resolution_mode == "preset" else "custom",
            memory_mode=memory_mode,
            attention_chunk_size=attention_chunk_size,
            projection_backend=projection_backend,
        )
        config.validate()
        return config, f"{width} x {height} pixels"


class WeeToddH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_H3_MODEL",),
                "config": ("WEETODD_H3_CONFIG",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "WeeTodd/H3"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/H3"
    DESCRIPTION = "Generate synchronized video and audio with MiniMax H3 through MLX."

    def generate(self, model, config, prompt, filename_prefix):
        config.validate()
        prepare_low_memory_stage("pipeline", config.memory_mode)
        progress = None
        check_interrupted = None
        try:
            import comfy.model_management
            import comfy.utils

            progress = comfy.utils.ProgressBar(config.steps - 1)
            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass

        def on_step(completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None:
                progress.update_absolute(completed, total)

        result = RUNTIME.get(model, config.projection_backend)(
            prompt,
            duration_seconds=config.duration_seconds,
            num_inference_steps=config.steps,
            seed=config.seed,
            height=config.height,
            width=config.width,
            drop_adaln=config.drop_adaln,
            step_callback=on_step,
        )
        from minimax_h3_mlx.media import save_mp4

        target = _safe_output_target(_output_directory(), filename_prefix, config.seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_mp4(target, result.video, result.fps, result.audio, result.sample_rate)
        info = {
            "prompt": prompt,
            **asdict(config),
            "video_path": str(target),
            "seconds_per_step": result.seconds_per_step,
            "total_seconds": result.total_seconds,
            "projection_backend": RUNTIME.projection_backend_report,
        }
        target.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        return (str(target), json.dumps(info, indent=2))


class WeeToddH3Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3"

    def release(self, unload):
        if unload:
            RUNTIME.unload()
            return ("MiniMax H3 MLX runtime unloaded",)
        return ("MiniMax H3 MLX runtime kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddH3ComponentLoader": WeeToddH3ComponentLoader,
    "WeeToddH3QuantizedTransformerLoader": WeeToddH3QuantizedTransformerLoader,
    "WeeToddH3Preflight": WeeToddH3Preflight,
    "WeeToddH3TextEncode": WeeToddH3TextEncode,
    "WeeToddH3UnloadTextEncoder": WeeToddH3UnloadTextEncoder,
    "WeeToddH3Sample": WeeToddH3Sample,
    "WeeToddH3LoRALoader": WeeToddH3LoRALoader,
    "WeeToddH3EasyCache": WeeToddH3EasyCache,
    "WeeToddH3TrajectoryForecast": WeeToddH3TrajectoryForecast,
    "WeeToddH3BlockCache": WeeToddH3BlockCache,
    "WeeToddH3HierarchicalBlockCache": WeeToddH3HierarchicalBlockCache,
    "WeeToddH3UnloadTransformer": WeeToddH3UnloadTransformer,
    "WeeToddH3VideoVAEDecode": WeeToddH3VideoVAEDecode,
    "WeeToddH3UnloadVideoVAE": WeeToddH3UnloadVideoVAE,
    "WeeToddH3AudioVAEDecode": WeeToddH3AudioVAEDecode,
    "WeeToddH3UnloadAudioVAE": WeeToddH3UnloadAudioVAE,
    "WeeToddH3PublishVideoAudio": WeeToddH3PublishVideoAudio,
    "WeeToddH3DirectPublishLatents": WeeToddH3DirectPublishLatents,
    "WeeToddH3ModelLoader": WeeToddH3ModelLoader,
    "WeeToddH3GenerationConfig": WeeToddH3GenerationConfig,
    "WeeToddH3Generate": WeeToddH3Generate,
    "WeeToddH3Unload": WeeToddH3Unload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddH3ComponentLoader": "WeeTodd H3 Component Loader",
    "WeeToddH3QuantizedTransformerLoader": "WeeTodd H3 Quantized Transformer Loader",
    "WeeToddH3Preflight": "WeeTodd H3 Component Preflight",
    "WeeToddH3TextEncode": "WeeTodd H3 Text Encode (Qwen3-VL)",
    "WeeToddH3UnloadTextEncoder": "WeeTodd H3 Unload Qwen3-VL",
    "WeeToddH3Sample": "WeeTodd H3 Sample Video + Audio Latents",
    "WeeToddH3LoRALoader": "WeeTodd H3 LoRA Loader (MLX)",
    "WeeToddH3EasyCache": "WeeTodd H3 EasyCache (MLX)",
    "WeeToddH3TrajectoryForecast": "WeeTodd H3 Trajectory Forecast (MLX)",
    "WeeToddH3BlockCache": "WeeTodd H3 BlockCache (MLX)",
    "WeeToddH3HierarchicalBlockCache": "WeeTodd H3 Hierarchical BlockCache (MLX)",
    "WeeToddH3UnloadTransformer": "WeeTodd H3 Unload Transformer",
    "WeeToddH3VideoVAEDecode": "WeeTodd H3 Decode Video VAE",
    "WeeToddH3UnloadVideoVAE": "WeeTodd H3 Unload Video VAE",
    "WeeToddH3AudioVAEDecode": "WeeTodd H3 Decode Audio VAE",
    "WeeToddH3UnloadAudioVAE": "WeeTodd H3 Unload Audio VAE",
    "WeeToddH3PublishVideoAudio": "WeeTodd H3 Publish Video + Audio",
    "WeeToddH3DirectPublishLatents": "WeeTodd H3 Direct Publish Latents (MLX)",
    "WeeToddH3ModelLoader": "WeeTodd H3 Model Loader (MLX)",
    "WeeToddH3GenerationConfig": "WeeTodd H3 Generation Config",
    "WeeToddH3Generate": "WeeTodd H3 Generate Video + Audio",
    "WeeToddH3Unload": "WeeTodd H3 Unload MLX Runtime",
}
