"""Classic ComfyUI node contracts backed by the MLX MiniMax H3 pipeline."""

import json
import platform
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
from .conditioning_inputs import H3KeyframeConditioning, H3ReferenceInput, H3ReferenceStack
from .continuation import (
    SUPPORTED_CONTEXT_FRAMES,
    continuation_context_from_latents,
    trim_continuation_overlap,
)
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


def _publication_environment(ffmpeg_path: str = "") -> dict[str, object]:
    """Describe the output and encoder state seen by this ComfyUI process."""
    from minimax_h3_mlx.media import ffmpeg_status

    return {
        "output_directory": str(_output_directory().resolve()),
        "ffmpeg": ffmpeg_status(ffmpeg_path or None),
    }


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


def _parse_media_timing_info(raw: str, *, image_frames: int, sample_rate: int):
    if not raw:
        return None
    try:
        timing = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Media timing information must be valid JSON.") from exc
    if not isinstance(timing, dict):
        raise ValueError("Media timing information must be a JSON object.")
    if timing.get("fps") != 24:
        raise ValueError("Media timing information must declare 24 fps video.")
    if timing.get("sample_rate") != sample_rate:
        raise ValueError("Media timing sample rate does not match the AUDIO sample rate.")
    if timing.get("output_frames") != image_frames:
        raise ValueError(
            "Media timing frame count does not match the decoded IMAGE frame count."
        )
    return timing


_COMPONENT_MODEL_CATEGORIES = {
    "checkpoint": ("checkpoints", "diffusers", "diffusion_models"),
    "transformer": ("diffusion_models", "checkpoints", "diffusers"),
    "text_encoder": ("text_encoders", "checkpoints"),
    "processor": ("text_encoders", "checkpoints"),
    "tokenizer": ("text_encoders", "checkpoints"),
    "video_vae": ("vae", "checkpoints"),
    "audio_vae": ("vae", "checkpoints"),
}


def _resolve_component_root(checkpoint: str, component: str = "checkpoint") -> str:
    """Resolve an H3 path through every model root registered with ComfyUI."""
    path = Path(checkpoint).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    if ".." in path.parts:
        raise ValueError("Relative H3 component paths cannot contain '..'.")
    try:
        import folder_paths

        roots = []
        seen = set()

        def add_root(value):
            root = Path(value).expanduser()
            key = str(root)
            if key not in seen:
                seen.add(key)
                roots.append(root)

        get_folder_paths = getattr(folder_paths, "get_folder_paths", None)
        if get_folder_paths is not None:
            for category in _COMPONENT_MODEL_CATEGORIES.get(component, ()):
                try:
                    for root in get_folder_paths(category):
                        add_root(root)
                except KeyError:
                    continue

        models_dir = Path(folder_paths.models_dir)
        add_root(models_dir)

        registered = getattr(folder_paths, "folder_names_and_paths", {})
        for category, entry in registered.items():
            if category in {"custom_nodes", "datasets"} or not entry:
                continue
            for root in entry[0]:
                add_root(root)

        for root in roots:
            candidate = root / path
            if candidate.exists():
                return str(candidate)
        return str(models_dir / path)
    except ImportError:
        return str(path)


_H3_RESOLUTION_MODES = ("ratio + size", "exact dimensions")
_H3_RESOLUTION_PRESETS = {
    "Use size slider — 32 px steps": 768,
    "384 px short edge — fast smoke": 384,
    "480 px short edge — fast preview": 480,
    "512 px short edge — balanced preview": 512,
    "576 px short edge — detailed preview": 576,
    "640 px short edge — quality preview": 640,
    "672 px short edge — quality preview+": 672,
    "704 px short edge — high quality": 704,
    "736 px short edge — near-native": 736,
    "768 px short edge — native": 768,
    "896 px short edge — high detail / high memory": 896,
    "1024 px short edge — very high memory": 1024,
    "1088 px short edge — maximum slider size": 1088,
}
_H3_LEGACY_RESOLUTION_PRESETS = {
    "384P (fast mode)": 384,
    "384P (fast smoke)": 384,
    "512P (balanced)": 512,
    "640P (quality preview)": 640,
    "768P (native quality)": 768,
    "2K (experimental, very high memory)": 1088,
}
_H3_RESOLUTION_SHORT_EDGES = {
    **_H3_RESOLUTION_PRESETS,
    **_H3_LEGACY_RESOLUTION_PRESETS,
}
_H3_ASPECT_RATIOS = {
    "21:9 — ultrawide landscape": (21, 9),
    "16:9 — widescreen landscape": (16, 9),
    "5:3 — wide landscape": (5, 3),
    "3:2 — classic landscape": (3, 2),
    "4:3 — standard landscape": (4, 3),
    "5:4 — near-square landscape": (5, 4),
    "1:1 — square": (1, 1),
    "4:5 — near-square portrait": (4, 5),
    "3:4 — standard portrait": (3, 4),
    "2:3 — classic portrait": (2, 3),
    "3:5 — tall portrait": (3, 5),
    "9:16 — vertical portrait": (9, 16),
    "9:21 — ultratall portrait": (9, 21),
}
_H3_LEGACY_ASPECT_RATIOS = {
    label.split(" — ", 1)[0]: value for label, value in _H3_ASPECT_RATIOS.items()
}

_H3_VALIDATED_SAMPLING_PRESETS = {
    "Chained context — Dense Turbo LightX2V rank 21 — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": (
            "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
        ),
        "measurement": {
            "task": "t2va_continuation",
            "windows": 4,
            "context_frames": 22,
            "canvas": [960, 544],
            "final_duration_seconds": 15.0,
            "transformer_evaluations_per_window": 4,
            "complete_workflow_seconds": 1570,
        },
    },
    "Chained context — Trajectory target-only replay — 20 points / up to 11 evaluations": {
        "steps": 20,
        "policy": "trajectory_speed_offline_replay",
        "measurement": {
            "task": "t2va_continuation",
            "windows": 4,
            "context_frames": 22,
            "canvas": [960, 544],
            "final_duration_seconds": 15.0,
            "transformer_evaluations_per_window": 11,
            "forecasts_per_window": 8,
            "fallbacks": 0,
            "complete_workflow_seconds": 3765,
            "conditioned_row_policy": "target_only",
        },
    },
    "Dense baseline — 20 points / 19 evaluations": {
        "steps": 20,
        "policy": "dense",
    },
    "Trajectory speed + offline replay — 20 points / up to 11 evaluations": {
        "steps": 20,
        "policy": "trajectory_speed_offline_replay",
    },
    "Ref2VA four-reference BF16 — Forward Attention replay — 20 points / up to 11 evaluations": {
        "steps": 20,
        "policy": "trajectory_speed_offline_replay",
        "measurement": {
            "task": "ref2va",
            "reference_images": 4,
            "canvas": [896, 512],
            "duration_seconds": 5.0,
            "memory_mode": "normal",
            "checkpoint_policy": "experimental_fl2va_weights_for_ref2va",
            "transformer_evaluations": 11,
            "mlx_peak_bytes": 47323507330,
        },
    },
    "Turbo — Larry EMA-850 — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
    },
    "Turbo — Larry v4 step-600 — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
    },
    "Turbo — LightX2V full rank — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
    },
    "Turbo — LightX2V dynamic rank 21 — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": (
            "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
        ),
    },
}


def _h3_aspect_ratio_key(aspect_ratio: str) -> str:
    return aspect_ratio.split(" — ", 1)[0]


def _bounded_h3_canvas(width: int, height: int) -> tuple[int, int]:
    if width > 1920 or height > 1920:
        raise ValueError("Resolved H3 width and height must not exceed 1920 pixels.")
    return width, height


def _resolve_h3_resolution(
    mode: str,
    resolution_tier: str,
    aspect_ratio: str,
    custom_width: int,
    custom_height: int,
    short_edge: int | None = None,
) -> tuple[int, int]:
    """Resolve ratio-and-size or exact dimensions to the H3 32-pixel grid."""
    if mode in {"custom", "exact dimensions"}:
        return _bounded_h3_canvas(custom_width, custom_height)
    if mode not in {"preset", "ratio + size"}:
        raise ValueError("Resolution mode must be 'ratio + size' or 'exact dimensions'.")
    if short_edge is None:
        try:
            short_edge = _H3_RESOLUTION_SHORT_EDGES[resolution_tier]
        except KeyError as exc:
            raise ValueError(f"Unknown H3 resolution preset: {resolution_tier!r}.") from exc
    if not 32 <= short_edge <= 1088 or short_edge % 32:
        raise ValueError("H3 short edge must be 32 through 1088 in 32-pixel steps.")
    try:
        ratio_width, ratio_height = _H3_ASPECT_RATIOS[aspect_ratio]
    except KeyError as exc:
        try:
            ratio_width, ratio_height = _H3_LEGACY_ASPECT_RATIOS[aspect_ratio]
        except KeyError:
            raise ValueError(f"Unknown H3 aspect ratio: {aspect_ratio!r}.") from exc
    ratio_key = _h3_aspect_ratio_key(aspect_ratio)
    # Preserve the established 1344x768 and 1120x640 H3 widescreen canvases. Values
    # between named presets are snapped again because a 32-pixel short-edge increment can
    # otherwise produce a long edge that falls between grid points.
    if ratio_key in {"16:9", "9:16"}:
        long_edge = round(short_edge * 7 / 4 / 32) * 32
        canvas = (long_edge, short_edge) if ratio_key == "16:9" else (short_edge, long_edge)
        return _bounded_h3_canvas(*canvas)
    if ratio_width >= ratio_height:
        height = short_edge
        width = round(short_edge * ratio_width / ratio_height / 32) * 32
    else:
        width = short_edge
        height = round(short_edge * ratio_height / ratio_width / 32) * 32
    return _bounded_h3_canvas(width, height)


class WeeToddH3ComponentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": ("STRING", {"default": "MiniMax-H3/FL2VA"}),
                "task": (["t2va", "fl2va", "ref2va"], {"default": "t2va"}),
            },
            "optional": {
                "transformer": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Select a shared or optimized transformer. Leave blank only when "
                            "the checkpoint contains a native transformer directory."
                        ),
                    },
                ),
                "text_encoder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Select a Qwen3-VL text-encoder root. Leave blank only when the "
                            "checkpoint contains a native text_encoder directory."
                        ),
                    },
                ),
                "processor": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Select the processor asset directory. T2VA can use tokenizer-only "
                            "assets. Image and reference tasks require vision processor files."
                        ),
                    },
                ),
                "tokenizer": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Select the directory that directly contains tokenizer.json.",
                    },
                ),
                "video_vae": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Select a native video-VAE directory or a self-describing MLX "
                            "safetensors file."
                        ),
                    },
                ),
                "audio_vae": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Select the licensed audio-VAE directory or a self-describing MLX "
                            "safetensors file."
                        ),
                    },
                ),
                "allow_fl2va_weights_for_ref2va": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": (
                            "Experimental: run Ref2VA packing with an FL2VA checkpoint. "
                            "The official partitions share an architecture but not weights."
                        ),
                    },
                ),
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
        allow_fl2va_weights_for_ref2va=False,
    ):
        return (
            H3ComponentSetSpec(
                checkpoint=_resolve_component_root(checkpoint, "checkpoint"),
                task=task,
                transformer=(
                    _resolve_component_root(transformer, "transformer") if transformer else None
                ),
                text_encoder=(
                    _resolve_component_root(text_encoder, "text_encoder") if text_encoder else None
                ),
                processor=(_resolve_component_root(processor, "processor") if processor else None),
                tokenizer=(_resolve_component_root(tokenizer, "tokenizer") if tokenizer else None),
                video_vae=(_resolve_component_root(video_vae, "video_vae") if video_vae else None),
                audio_vae=(_resolve_component_root(audio_vae, "audio_vae") if audio_vae else None),
                allow_fl2va_weights_for_ref2va=bool(allow_fl2va_weights_for_ref2va),
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
            },
            "optional": {
                "ffmpeg_path": (
                    "STRING",
                    {
                        "default": "",
                        "advanced": True,
                        "tooltip": (
                            "Optional executable override. Leave empty to use WEETODD_FFMPEG, "
                            "the ComfyUI process PATH, or a compatible packaged encoder."
                        ),
                    },
                )
            },
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
        ffmpeg_path="",
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
        payload = report.to_dict()
        payload["publication"] = _publication_environment(ffmpeg_path)
        return components, json.dumps(payload, indent=2, sort_keys=True)


class WeeToddH3FirstFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first_frame": ("IMAGE",)}}

    RETURN_TYPES = ("WEETODD_H3_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = "Use one image as the first-frame endpoint for an FL2VA generation."

    def configure(self, first_frame):
        conditioning = H3KeyframeConditioning(first_frame=first_frame)
        return conditioning, json.dumps(conditioning.metadata(), indent=2, sort_keys=True)


class WeeToddH3LastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"last_frame": ("IMAGE",)}}

    RETURN_TYPES = ("WEETODD_H3_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = "Use one image as the last-frame endpoint for an FL2VA generation."

    def configure(self, last_frame):
        conditioning = H3KeyframeConditioning(last_frame=last_frame)
        return conditioning, json.dumps(conditioning.metadata(), indent=2, sort_keys=True)


class WeeToddH3FirstLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)}}

    RETURN_TYPES = ("WEETODD_H3_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = "Use two images as the first-frame and last-frame endpoints for FL2VA."

    def configure(self, first_frame, last_frame):
        conditioning = H3KeyframeConditioning(
            first_frame=first_frame,
            last_frame=last_frame,
        )
        return conditioning, json.dumps(conditioning.metadata(), indent=2, sort_keys=True)


def _append_reference(previous_references, reference):
    stack = (previous_references or H3ReferenceStack()).append(reference)
    return stack, json.dumps(stack.metadata(), indent=2, sort_keys=True)


def _checkpoint_task_policy(components) -> str:
    if getattr(components, "allow_fl2va_weights_for_ref2va", False):
        return "experimental_fl2va_weights_for_ref2va"
    return "strict_manifest"


class WeeToddH3ReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_budget_percent": (
                    "INT",
                    {
                        "default": 100,
                        "min": 50,
                        "max": 400,
                        "step": 10,
                        "display": "slider",
                    },
                ),
            },
            "optional": {"previous_references": ("WEETODD_H3_REFERENCES",)},
        }

    RETURN_TYPES = ("WEETODD_H3_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "reference_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Append an image identity, subject, style, or scene reference. Reference order controls "
        "the prompt labels and packed rotary positions. A 100% pixel budget matches the output "
        "canvas area; lower values reduce persistent reference tokens and higher values retain "
        "more source detail."
    )

    def append(self, image, pixel_budget_percent, previous_references=None):
        return _append_reference(
            previous_references,
            H3ReferenceInput(
                "image",
                image,
                image_pixel_budget_percent=pixel_budget_percent,
            ),
        )


class WeeToddH3ReferenceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.01}),
            },
            "optional": {
                "soundtrack": ("AUDIO",),
                "previous_references": ("WEETODD_H3_REFERENCES",),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "reference_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Append a video motion and camera reference, with an optional synchronized soundtrack. "
        "Supply the source frame rate explicitly."
    )

    def append(self, video_frames, fps, soundtrack=None, previous_references=None):
        return _append_reference(
            previous_references,
            H3ReferenceInput("video", video_frames, fps=fps, soundtrack=soundtrack),
        )


class WeeToddH3ReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {"previous_references": ("WEETODD_H3_REFERENCES",)},
        }

    RETURN_TYPES = ("WEETODD_H3_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "reference_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Append a standalone voice, sound, or music reference. Ref2VA also requires at least "
        "one image or video reference."
    )

    def append(self, audio, previous_references=None):
        return _append_reference(previous_references, H3ReferenceInput("audio", audio))


class WeeToddH3KeyframeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "keyframes": ("WEETODD_H3_KEYFRAMES",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Encode FL2VA prompt vision rows and first/last-frame VAE rows in separate staged phases. "
        "Each weighted component unloads before the next phase."
    )

    def encode(self, components, config, keyframes, prompt):
        if components.task != "fl2va":
            raise ValueError("H3 keyframe encoding requires an FL2VA component set.")
        config.validate()
        keyframes.validate()
        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        if check_interrupted is not None:
            check_interrupted()

        from minimax_h3_mlx.packing import prepare_keyframe_image

        images = [
            prepare_keyframe_image(
                image,
                config.height,
                config.width,
                stretch=anchor == "first",
            )
            for anchor, image in zip(keyframes.anchors, keyframes.images(), strict=True)
        ]
        text_releases = ()
        video_vae_releases = ()

        def prepare_text_stage():
            nonlocal text_releases
            text_releases = prepare_low_memory_stage("text_encoder", config.memory_mode)

        conditioning = TEXT_ENCODER_RUNTIME.encode(
            H3TextEncoderSpec.from_components(components, load_vision=True),
            prompt,
            images=images,
            task="fl2va",
            unload_after=True,
            prepare_stage=prepare_text_stage,
        )
        if check_interrupted is not None:
            check_interrupted()

        def prepare_video_vae_stage():
            nonlocal video_vae_releases
            video_vae_releases = prepare_low_memory_stage("video_vae", config.memory_mode)

        rows = VIDEO_VAE_RUNTIME.encode_keyframes(
            H3VideoVAESpec.from_components(components),
            images,
            height=config.height,
            width=config.width,
            unload_after=True,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_video_vae_stage,
        )
        conditioning = replace(
            conditioning,
            condition_video_rows=rows,
            keyframe_anchors=keyframes.anchors,
        )
        info = {
            "task": "fl2va",
            "prompt": prompt,
            "token_count": conditioning.token_count,
            "anchors": list(keyframes.anchors),
            "condition_video_rows": int(rows.shape[0]),
            "vision_loaded": True,
            "encoder_resident": TEXT_ENCODER_RUNTIME.loaded,
            "video_vae_resident": VIDEO_VAE_RUNTIME.loaded,
            "staged_releases": {
                "text_encoder": list(text_releases),
                "video_vae": list(video_vae_releases),
            },
        }
        return conditioning, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3ReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "references": ("WEETODD_H3_REFERENCES",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Prepare ordered Ref2VA media, then stage Qwen3-VL, the video VAE, and the audio VAE. "
        "Each weighted component unloads before the next stage."
    )

    def encode(self, components, config, references, prompt):
        if components.task != "ref2va":
            raise ValueError("H3 reference encoding requires a Ref2VA component set.")
        config.validate()
        references.validate_request()
        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        if check_interrupted is not None:
            check_interrupted()

        from minimax_h3_mlx.packing import align_num_frames

        num_frames = align_num_frames(round(config.duration_seconds * 24))
        prepared = references.prepare(
            target_width=config.width,
            target_height=config.height,
            target_num_frames=num_frames,
        )
        staged = {}

        def prepare_text_stage():
            staged["text_encoder"] = list(
                prepare_low_memory_stage("text_encoder", config.memory_mode)
            )

        conditioning = TEXT_ENCODER_RUNTIME.encode(
            H3TextEncoderSpec.from_components(components, load_vision=True),
            prompt,
            references=prepared,
            task="ref2va",
            unload_after=True,
            prepare_stage=prepare_text_stage,
        )
        if check_interrupted is not None:
            check_interrupted()

        def prepare_video_stage():
            staged["video_vae"] = list(prepare_low_memory_stage("video_vae", config.memory_mode))

        video_rows = VIDEO_VAE_RUNTIME.encode_references(
            H3VideoVAESpec.from_components(components),
            prepared,
            unload_after=True,
            check_interrupted=check_interrupted,
            prepare_stage=prepare_video_stage,
        )

        audio_rows = None
        if any(reference.has_audio for reference in prepared):

            def prepare_audio_stage():
                staged["audio_vae"] = list(
                    prepare_low_memory_stage("audio_vae", config.memory_mode)
                )

            audio_rows = AUDIO_VAE_RUNTIME.encode_references(
                H3AudioVAESpec.from_components(components),
                prepared,
                unload_after=True,
                check_interrupted=check_interrupted,
                prepare_stage=prepare_audio_stage,
            )

        conditioning = replace(
            conditioning,
            condition_video_rows=video_rows,
            condition_audio_rows=audio_rows,
            references=tuple(prepared),
        )
        info = {
            "task": "ref2va",
            "prompt": prompt,
            "token_count": conditioning.token_count,
            "reference_count": len(prepared),
            "condition_video_rows": int(video_rows.shape[0]),
            "condition_audio_rows": 0 if audio_rows is None else int(audio_rows.shape[0]),
            "references": [
                {
                    **metadata,
                    "latent_frames": reference.num_latent_frames,
                    "latent_height": reference.latent_height,
                    "latent_width": reference.latent_width,
                    "audio_latents": reference.num_audio_latents,
                }
                for metadata, reference in zip(
                    references.metadata()["references"], prepared, strict=True
                )
            ],
            "staged_releases": staged,
            "encoder_resident": TEXT_ENCODER_RUNTIME.loaded,
            "video_vae_resident": VIDEO_VAE_RUNTIME.loaded,
            "audio_vae_resident": AUDIO_VAE_RUNTIME.loaded,
        }
        return conditioning, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3ReferenceStrength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("WEETODD_H3_CONDITIONING",),
                "visual_strength": (
                    "FLOAT",
                    {
                        "default": 0.999,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": (
                            "Lower values add more noise to image and video conditioning. "
                            "For FL2VA, values below 0.7 can weaken the last-frame anchor."
                        ),
                    },
                ),
                "audio_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": (
                            "1.0 keeps reference audio clean. Lower values add seeded noise and "
                            "can change generated audio as well as motion."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "strength_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Adjust how strongly FL2VA or Ref2VA trusts visual and audio condition rows. "
        "Defaults preserve the released H3 behavior."
    )

    def configure(self, conditioning, visual_strength, audio_strength):
        if conditioning.task not in {"fl2va", "ref2va"}:
            raise ValueError("H3 reference strength requires FL2VA or Ref2VA conditioning.")
        for name, value in (
            ("visual_strength", visual_strength),
            ("audio_strength", audio_strength),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"H3 {name} must be between 0 and 1.")
        warning = None
        if conditioning.task == "fl2va" and visual_strength < 0.7:
            warning = "Visual strength below 0.7 can weaken or remove the last-frame anchor."
        configured = replace(
            conditioning,
            visual_condition_strength=float(visual_strength),
            audio_condition_strength=float(audio_strength),
        )
        info = {
            "task": conditioning.task,
            "visual_strength": configured.visual_condition_strength,
            "audio_strength": configured.audio_condition_strength,
            "visual_noise_fraction": 1.0 - configured.visual_condition_strength,
            "audio_noise_fraction": 1.0 - configured.audio_condition_strength,
            "warning": warning,
        }
        return configured, json.dumps(info, indent=2, sort_keys=True)


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
            "paged_weights": getattr(conditioning, "paging_report", None),
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


class WeeToddH3ContinuationContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("WEETODD_H3_LATENTS",),
                "context_frames": ([str(value) for value in SUPPORTED_CONTEXT_FRAMES],),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONTINUATION", "STRING")
    RETURN_NAMES = ("continuation", "continuation_info")
    FUNCTION = "extract"
    CATEGORY = "WeeTodd/H3/continuation"
    DESCRIPTION = (
        "Copy a synchronized tail from H3 video and audio latents for motion continuation. "
        "The recommended 22-frame overlap is about 0.92 seconds at 24 fps."
    )

    def extract(self, latents, context_frames):
        context = continuation_context_from_latents(latents, int(context_frames))
        info = {
            "context_frames": context.context_frames,
            "context_seconds": context.context_frames / context.fps,
            "video_latent_frames": context.video_latent_frames,
            "audio_latent_frames": context.audio_latent_frames,
            "width": context.width,
            "height": context.height,
            "fps": context.fps,
            "sample_rate": context.sample_rate,
            "checkpoint": Path(context.transformer_checkpoint).name,
            "transformer": Path(context.transformer_path).name,
        }
        return context, json.dumps(info, indent=2, sort_keys=True)


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
                "continuation": ("WEETODD_H3_CONTINUATION",),
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
        continuation=None,
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

            progress_steps = config.steps - 1
            if trajectory_forecast is not None and getattr(
                trajectory_forecast, "offline_smoothing_replay", False
            ):
                progress_steps *= 2
            progress = comfy.utils.ProgressBar(progress_steps)
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
            continuation=continuation,
            loras=loras,
            prepare_stage=prepare_stage,
        )
        info = {
            "prompt": conditioning.prompt,
            "task": conditioning.task,
            "checkpoint_task_policy": _checkpoint_task_policy(components),
            "keyframe_anchors": list(conditioning.keyframe_anchors),
            "references": [
                {
                    "kind": reference.kind,
                    "video_rows": reference.video_rows((1, 2, 2)),
                    "audio_rows": reference.audio_rows,
                }
                for reference in conditioning.references
            ],
            "reference_strength": {
                "visual": conditioning.visual_condition_strength,
                "audio": conditioning.audio_condition_strength,
            },
            "continuation": (
                {
                    "context_frames": continuation.context_frames,
                    "context_seconds": continuation.context_frames / continuation.fps,
                    "video_latent_frames": continuation.video_latent_frames,
                    "audio_latent_frames": continuation.audio_latent_frames,
                }
                if continuation is not None
                else None
            ),
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
            "blockcache_segment_hits": list(getattr(latents, "blockcache_segment_hits", ())),
            "blockcache_segment_thresholds": list(
                getattr(latents, "blockcache_segment_thresholds", ())
            ),
            "blockcache_executed_blocks": getattr(latents, "blockcache_executed_blocks", 0),
            "blockcache_skipped_blocks": getattr(latents, "blockcache_skipped_blocks", 0),
            "blockcache": asdict(blockcache) if blockcache is not None else None,
            "trajectory_forecasts": getattr(latents, "trajectory_forecasts", 0),
            "trajectory_bootstrap_forecasts": getattr(latents, "trajectory_bootstrap_forecasts", 0),
            "trajectory_fallbacks": getattr(latents, "trajectory_fallbacks", 0),
            "trajectory_history_bytes": getattr(latents, "trajectory_history_bytes", 0),
            "trajectory_offline_replay": getattr(latents, "trajectory_offline_replay", False),
            "trajectory_replay_steps": getattr(latents, "trajectory_replay_steps", 0),
            "trajectory_replay_anchor_steps": getattr(latents, "trajectory_replay_anchor_steps", 0),
            "trajectory_replay_smoothed_steps": getattr(
                latents, "trajectory_replay_smoothed_steps", 0
            ),
            "trajectory_capture_seconds": getattr(latents, "trajectory_capture_seconds", 0.0),
            "trajectory_replay_seconds": getattr(latents, "trajectory_replay_seconds", 0.0),
            "trajectory_replay_fallback_reason": getattr(
                latents, "trajectory_replay_fallback_reason", None
            ),
            "trajectory_conditioned_row_policy": getattr(
                latents, "trajectory_conditioned_row_policy", None
            ),
            "trajectory_excluded_condition_rows": {
                "video": getattr(latents, "trajectory_excluded_video_rows", 0),
                "audio": getattr(latents, "trajectory_excluded_audio_rows", 0),
            },
            "trajectory_forecast": (
                asdict(trajectory_forecast) if trajectory_forecast is not None else None
            ),
            "loras": loras.metadata() if loras is not None else [],
            "lora_report": list(getattr(latents, "lora_report", ())),
            "seconds_per_evaluation": latents.seconds_per_evaluation,
            "total_seconds": latents.total_seconds,
            "transformer_resident": TRANSFORMER_RUNTIME.loaded,
            "memory_mode": config.memory_mode,
            "sampling_method": config.sampling_method,
            "attention_query_chunk_size": config.attention_query_chunk_size,
            "compute_dtype": "bfloat16",
            "projection_backend": getattr(latents, "projection_backend_report", None),
            "projection_backend_runtime": getattr(latents, "projection_backend_runtime", None),
            "paged_weights": {
                "transformer": getattr(latents, "paging_report", None),
                "text_encoder": getattr(latents, "text_encoder_paging_report", None),
            },
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
                "qkv_layout": (
                    ["auto", "native_interleaved", "contiguous_qkv"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Turbo adapters normally use contiguous Q/K/V rows and are converted "
                            "to the H3 MLX per-head layout. Override only for a verified adapter."
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

    def load(
        self,
        lora_name,
        strength,
        profile,
        adaln_input_grid="",
        qkv_layout="auto",
        previous_loras=None,
    ):
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
            qkv_layout=qkv_layout,
        )
        stack = (previous_loras or H3LoRAStack()).append(spec)
        info = {
            "file": path.name,
            "strength": strength,
            "profile": spec.resolved_profile,
            "qkv_layout": spec.resolved_qkv_layout,
            "tensor_bytes": spec.tensor_bytes,
            "adaln_input_grid": grid.name if grid is not None else None,
            "stack_size": len(stack.adapters),
            "loads_at_sampling": True,
        }
        return stack, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3ValidatedSamplingPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_H3_CONFIG",),
                "preset": (
                    list(_H3_VALIDATED_SAMPLING_PRESETS),
                    {
                        "default": "Dense baseline — 20 points / 19 evaluations",
                        "tooltip": (
                            "Apply one measured sampling schedule. The node preserves the canvas, "
                            "duration, seed, memory mode, and component paths."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "WEETODD_H3_CONFIG",
        "WEETODD_H3_LORAS",
        "WEETODD_H3_TRAJECTORY_FORECAST",
        "STRING",
    )
    RETURN_NAMES = ("config", "loras", "trajectory_forecast", "preset_info")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Apply a measured dense, trajectory-replay, or Turbo sampling policy. "
        "Connect all three typed outputs to the H3 sampler."
    )

    def apply(self, config, preset):
        try:
            selected = _H3_VALIDATED_SAMPLING_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(f"Unknown H3 validated sampling preset: {preset!r}.") from exc

        steps = int(selected["steps"])
        configured = replace(config, steps=steps, sampling_method="euler")
        configured.validate()
        policy = str(selected["policy"])
        loras = None
        trajectory_forecast = None

        if policy == "turbo":
            lora_name = str(selected["lora"])
            try:
                loras, _ = WeeToddH3LoRALoader().load(
                    lora_name,
                    1.0,
                    "turbo",
                    qkv_layout="auto",
                )
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"H3 preset requires LoRA {lora_name!r}. "
                    "Place the file in a ComfyUI LoRA model folder and refresh model files."
                ) from exc
        elif policy == "trajectory_speed_offline_replay":
            from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig

            trajectory_forecast = H3TrajectoryForecastConfig(
                mode="automatic_speed",
                forecast_strength=1.0,
                warmup_steps=2,
                tail_actual_steps=1,
                max_history=2,
                max_forecast_fraction=0.5,
                max_delta_ratio=2.5,
                bootstrap_first_forecast=False,
                offline_smoothing_replay=True,
                offline_video_blend=0.5,
                offline_audio_blend=0.0,
                conditioned_row_policy="target_only",
            )
            trajectory_forecast.validate()

        lora_file = str(selected["lora"]) if "lora" in selected else None
        info = {
            "preset": preset,
            "policy": policy,
            "requested_schedule_points": steps,
            "transformer_evaluations_without_forecast": steps - 1,
            "sampling_method": configured.sampling_method,
            "lora_file": lora_file,
            "lora_strength": 1.0 if lora_file is not None else None,
            "trajectory_offline_replay": bool(
                trajectory_forecast is not None and trajectory_forecast.offline_smoothing_replay
            ),
            "canvas": [configured.width, configured.height],
            "duration_seconds": configured.duration_seconds,
            "seed": configured.seed,
            "measurement": selected.get("measurement"),
        }
        return configured, loras, trajectory_forecast, json.dumps(info, indent=2, sort_keys=True)


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
                ),
                "offline_smoothing_replay": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Run a transformer-free second pass from archived actual anchors. "
                            "This can protect audio from later joint-transformer forecast error "
                            "but uses more memory."
                        ),
                    },
                ),
                "offline_video_blend": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend global affine smoothing into local video interpolation during "
                            "offline replay."
                        ),
                    },
                ),
                "offline_audio_blend": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend global affine smoothing into local audio interpolation during "
                            "offline replay. Keep zero for the audio-isolation default."
                        ),
                    },
                ),
                "conditioned_row_policy": (
                    ["target_only", "all_rows_legacy"],
                    {
                        "default": "target_only",
                        "tooltip": (
                            "Forecast only generated rows when continuation or reference rows "
                            "are present. This is the recommended chained-context policy."
                        ),
                    },
                ),
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
        offline_smoothing_replay=False,
        offline_video_blend=0.5,
        offline_audio_blend=0.0,
        conditioned_row_policy="target_only",
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
            offline_smoothing_replay=offline_smoothing_replay,
            offline_video_blend=offline_video_blend,
            offline_audio_blend=offline_audio_blend,
            conditioned_row_policy=conditioned_row_policy,
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


class WeeToddH3TrimContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "continuation": ("WEETODD_H3_CONTINUATION",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "trim_info")
    FUNCTION = "trim"
    CATEGORY = "WeeTodd/H3/continuation"
    DESCRIPTION = (
        "Remove the repeated motion-continuation overlap from decoded video and audio, then "
        "normalize audio to the exact remaining video duration."
    )

    def trim(self, images, audio, continuation):
        trimmed_images, trimmed_audio, info = trim_continuation_overlap(
            images, audio, continuation.context_frames, continuation.fps
        )
        return trimmed_images, trimmed_audio, json.dumps(info, indent=2, sort_keys=True)


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
                "media_timing_info": ("STRING", {"default": ""}),
                "ffmpeg_path": (
                    "STRING",
                    {
                        "default": "",
                        "advanced": True,
                        "tooltip": "Optional ffmpeg executable override for this publication.",
                    },
                ),
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
        media_timing_info="",
        ffmpeg_path="",
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
        timing_metadata = _parse_media_timing_info(
            media_timing_info,
            image_frames=int(images.shape[0]),
            sample_rate=int(audio["sample_rate"]),
        )
        if timing_metadata is None and images.shape[0] != expected_frames:
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
        if timing_metadata is not None:
            supplied_metadata["media_timing"] = timing_metadata
        component_paths = components.resolved_paths()
        try:
            package_version = version("comfyui-weetodd-nodes")
        except PackageNotFoundError:
            package_version = "uninstalled"
        try:
            mlx_version = version("mlx")
        except PackageNotFoundError:
            mlx_version = "uninstalled"
        publication = _publication_environment(ffmpeg_path)
        metadata = {
            **supplied_metadata,
            "generation": asdict(config),
            "precision_policy": (
                "component-specific checkpoint precision; verify quantization in preflight"
            ),
            "components": {
                "checkpoint": Path(components.checkpoint).name,
                "task": components.task,
                "checkpoint_task_policy": _checkpoint_task_policy(components),
                **{name: path.name for name, path in component_paths.items()},
            },
            "software": {
                "python": platform.python_version(),
                "mlx": mlx_version,
                "weetodd_nodes": package_version,
            },
            "publication": publication,
        }
        output_root = Path(publication["output_directory"])
        target = _safe_output_target(output_root, filename_prefix, config.seed)
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
            ffmpeg_path=ffmpeg_path or None,
        )
        info = json.dumps(result.metadata, indent=2, sort_keys=True)
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
                "ffmpeg_path": (
                    "STRING",
                    {
                        "default": "",
                        "advanced": True,
                        "tooltip": "Optional ffmpeg executable override for this publication.",
                    },
                ),
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
        ffmpeg_path="",
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
        publication = _publication_environment(ffmpeg_path)
        metadata = {
            **supplied_metadata,
            "generation": asdict(config),
            "precision_policy": (
                "component-specific checkpoint precision; verify quantization in preflight"
            ),
            "components": {
                "checkpoint": Path(components.checkpoint).name,
                "task": components.task,
                "checkpoint_task_policy": _checkpoint_task_policy(components),
                **{name: path.name for name, path in component_paths.items()},
            },
            "software": {
                "python": platform.python_version(),
                "mlx": mlx_version,
                "weetodd_nodes": package_version,
            },
            "publication": publication,
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

        output_root = Path(publication["output_directory"])
        target = _safe_output_target(output_root, filename_prefix, config.seed)
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
            ffmpeg_path=ffmpeg_path or None,
        )
        info = json.dumps(result.metadata, indent=2, sort_keys=True)
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
                "resolution_mode": (
                    [*_H3_RESOLUTION_MODES, "preset", "custom"],
                    {"default": "ratio + size"},
                ),
                "resolution_tier": (
                    list(_H3_RESOLUTION_SHORT_EDGES),
                    {"default": "768 px short edge — native"},
                ),
                "aspect_ratio": (
                    [
                        *_H3_ASPECT_RATIOS,
                        "custom — exact dimensions",
                        *_H3_LEGACY_ASPECT_RATIOS,
                    ],
                    {"default": "16:9 — widescreen landscape"},
                ),
                "custom_width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 1920, "step": 32, "advanced": True},
                ),
                "custom_height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 1920, "step": 32, "advanced": True},
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
            },
            "optional": {
                "short_edge": (
                    "INT",
                    {
                        "default": 768,
                        "min": 32,
                        "max": 1088,
                        "step": 32,
                        "display": "slider",
                        "tooltip": (
                            "Move in 32-pixel steps. The selected aspect ratio resolves the "
                            "compatible width and height live in the ComfyUI node."
                        ),
                    },
                ),
                "sampling_method": (
                    ["euler", "res_multistep"],
                    {
                        "default": "euler",
                        "tooltip": (
                            "Use res_multistep for the native H3 base-model quality regime; "
                            "Euler remains available for established Turbo and benchmark recipes."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_CONFIG", "STRING")
    RETURN_NAMES = ("config", "resolved_resolution")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3"

    DESCRIPTION = (
        "Choose a clearly labeled aspect ratio and move the short-edge size slider, or use exact "
        "dimensions. The live canvas remains on H3's required 32-pixel grid."
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
        short_edge=None,
        sampling_method="euler",
    ):
        width, height = _resolve_h3_resolution(
            resolution_mode,
            resolution_tier,
            aspect_ratio,
            custom_width,
            custom_height,
            short_edge,
        )
        ratio_mode = resolution_mode in {"preset", "ratio + size"}
        normalized_ratio = _h3_aspect_ratio_key(aspect_ratio) if ratio_mode else "custom"
        normalized_mode = "ratio + size" if ratio_mode else "exact dimensions"
        config = H3GenerationConfig(
            duration_seconds=duration_seconds,
            steps=steps,
            seed=seed,
            width=width,
            height=height,
            drop_adaln=drop_adaln,
            resolution_mode=normalized_mode,
            resolution_tier=resolution_tier if ratio_mode else "custom",
            aspect_ratio=normalized_ratio,
            memory_mode=memory_mode,
            attention_chunk_size=attention_chunk_size,
            projection_backend=projection_backend,
            sampling_method=sampling_method,
        )
        config.validate()
        if ratio_mode:
            short_edge_label = f"{min(width, height)} px short edge"
            return (
                config,
                f"{width} × {height} pixels — {normalized_ratio} — {short_edge_label}",
            )
        return config, f"{width} × {height} pixels — custom"


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
    "WeeToddH3FirstFrame": WeeToddH3FirstFrame,
    "WeeToddH3LastFrame": WeeToddH3LastFrame,
    "WeeToddH3FirstLastFrame": WeeToddH3FirstLastFrame,
    "WeeToddH3ReferenceImage": WeeToddH3ReferenceImage,
    "WeeToddH3ReferenceVideo": WeeToddH3ReferenceVideo,
    "WeeToddH3ReferenceAudio": WeeToddH3ReferenceAudio,
    "WeeToddH3KeyframeEncode": WeeToddH3KeyframeEncode,
    "WeeToddH3ReferenceEncode": WeeToddH3ReferenceEncode,
    "WeeToddH3ReferenceStrength": WeeToddH3ReferenceStrength,
    "WeeToddH3TextEncode": WeeToddH3TextEncode,
    "WeeToddH3UnloadTextEncoder": WeeToddH3UnloadTextEncoder,
    "WeeToddH3ContinuationContext": WeeToddH3ContinuationContext,
    "WeeToddH3Sample": WeeToddH3Sample,
    "WeeToddH3LoRALoader": WeeToddH3LoRALoader,
    "WeeToddH3ValidatedSamplingPreset": WeeToddH3ValidatedSamplingPreset,
    "WeeToddH3EasyCache": WeeToddH3EasyCache,
    "WeeToddH3TrajectoryForecast": WeeToddH3TrajectoryForecast,
    "WeeToddH3BlockCache": WeeToddH3BlockCache,
    "WeeToddH3HierarchicalBlockCache": WeeToddH3HierarchicalBlockCache,
    "WeeToddH3UnloadTransformer": WeeToddH3UnloadTransformer,
    "WeeToddH3VideoVAEDecode": WeeToddH3VideoVAEDecode,
    "WeeToddH3UnloadVideoVAE": WeeToddH3UnloadVideoVAE,
    "WeeToddH3AudioVAEDecode": WeeToddH3AudioVAEDecode,
    "WeeToddH3UnloadAudioVAE": WeeToddH3UnloadAudioVAE,
    "WeeToddH3TrimContinuation": WeeToddH3TrimContinuation,
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
    "WeeToddH3FirstFrame": "WeeTodd H3 First Frame",
    "WeeToddH3LastFrame": "WeeTodd H3 Last Frame",
    "WeeToddH3FirstLastFrame": "WeeTodd H3 First + Last Frame",
    "WeeToddH3ReferenceImage": "WeeTodd H3 Reference Image",
    "WeeToddH3ReferenceVideo": "WeeTodd H3 Reference Video",
    "WeeToddH3ReferenceAudio": "WeeTodd H3 Reference Audio",
    "WeeToddH3KeyframeEncode": "WeeTodd H3 Encode First / Last Frames",
    "WeeToddH3ReferenceEncode": "WeeTodd H3 Encode References",
    "WeeToddH3ReferenceStrength": "WeeTodd H3 Reference Strength",
    "WeeToddH3TextEncode": "WeeTodd H3 Text Encode (Qwen3-VL)",
    "WeeToddH3UnloadTextEncoder": "WeeTodd H3 Unload Qwen3-VL",
    "WeeToddH3ContinuationContext": "WeeTodd H3 Motion Continuation Context",
    "WeeToddH3Sample": "WeeTodd H3 Sample Video + Audio Latents",
    "WeeToddH3LoRALoader": "WeeTodd H3 LoRA Loader (MLX)",
    "WeeToddH3ValidatedSamplingPreset": "WeeTodd H3 Validated Sampling Preset",
    "WeeToddH3EasyCache": "WeeTodd H3 EasyCache (MLX)",
    "WeeToddH3TrajectoryForecast": "WeeTodd H3 Trajectory Forecast (MLX)",
    "WeeToddH3BlockCache": "WeeTodd H3 BlockCache (MLX)",
    "WeeToddH3HierarchicalBlockCache": "WeeTodd H3 Hierarchical BlockCache (MLX)",
    "WeeToddH3UnloadTransformer": "WeeTodd H3 Unload Transformer",
    "WeeToddH3VideoVAEDecode": "WeeTodd H3 Decode Video VAE",
    "WeeToddH3UnloadVideoVAE": "WeeTodd H3 Unload Video VAE",
    "WeeToddH3AudioVAEDecode": "WeeTodd H3 Decode Audio VAE",
    "WeeToddH3UnloadAudioVAE": "WeeTodd H3 Unload Audio VAE",
    "WeeToddH3TrimContinuation": "WeeTodd H3 Trim Continuation Overlap",
    "WeeToddH3PublishVideoAudio": "WeeTodd H3 Publish Video + Audio",
    "WeeToddH3DirectPublishLatents": "WeeTodd H3 Direct Publish Latents (MLX)",
    "WeeToddH3ModelLoader": "WeeTodd H3 Model Loader (MLX)",
    "WeeToddH3GenerationConfig": "WeeTodd H3 Generation Config",
    "WeeToddH3Generate": "WeeTodd H3 Generate Video + Audio",
    "WeeToddH3Unload": "WeeTodd H3 Unload MLX Runtime",
}

# LTX 2.3 remains an optional, independently loaded engine. Importing these
# adapter contracts does not import MLX or ltx-2-mlx.
from .ltx_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as LTX23_NODE_CLASS_MAPPINGS,
)
from .ltx_nodes import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as LTX23_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(LTX23_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LTX23_NODE_DISPLAY_NAME_MAPPINGS)
