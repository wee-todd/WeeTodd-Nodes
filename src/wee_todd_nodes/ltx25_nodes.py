"""ComfyUI adapters for the standalone LTX 2.5 MLX split-component pipeline."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict
from pathlib import Path

from ltx25_mlx.runtime import (
    LTX25_GENERATION_PRESETS,
    RUNTIME,
    LTX25ComponentSpec,
    LTX25GenerationConfig,
    apply_ltx25_generation_preset,
    backend_capability,
)
from ltx25_mlx.upscale_contracts import (
    LTX25_INPUT_SIZE_POLICIES,
    LTX25_PIXEL_SPATIAL_MODE,
    LTX25_SOURCE_FRAME_ANCHORS,
    LTX25_UPSCALE_MODES,
)

from .ltx_nodes import (
    _check_interrupted,
    _comfy_progress,
    _preview,
    _release_h3_stages,
    _safe_target,
    _software_versions,
)


def _register_ltx25_model_folders() -> None:
    try:
        import folder_paths

        add_path = getattr(folder_paths, "add_model_folder_path", None)
        if add_path is None:
            return
        model_root = Path(folder_paths.models_dir) / "LTX-2.5"
        for category, subdir in (
            ("ltx25", ""),
            ("diffusion_models", "diffusion_models"),
            ("text_encoders", "text_encoders"),
            ("vae", "vae"),
            ("latent_upscale_models", "latent_upscale_models"),
        ):
            add_path(category, str(model_root / subdir), is_default=True)
    except ImportError:
        pass


def _resolve_component(value: str, categories: tuple[str, ...]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    if ".." in path.parts:
        raise ValueError("Relative LTX 2.5 component paths cannot contain '..'.")
    try:
        import folder_paths

        _register_ltx25_model_folders()
        roots: list[Path] = []
        seen: set[str] = set()
        for category in categories:
            try:
                candidates = folder_paths.get_folder_paths(category)
            except KeyError:
                continue
            for root in candidates:
                resolved = Path(root).expanduser()
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    roots.append(resolved)
        models_dir = Path(folder_paths.models_dir)
        roots.extend([models_dir / "LTX-2.5", models_dir])
        candidates = [root / path for root in roots]
        if path.parts and path.parts[0].lower() in {"ltx-2.5", "ltx2.5", "ltx25"}:
            tail = Path(*path.parts[1:])
            candidates.extend(root / tail for root in roots)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return models_dir / path
    except ImportError:
        return path


class WeeToddLTX25ComponentLoader:
    @classmethod
    def INPUT_TYPES(cls):
        _register_ltx25_model_folders()
        return {
            "required": {
                "transformer": (
                    "STRING",
                    {"default": "ltx-2.5-22b-distilled-transformer-bf16.safetensors"},
                ),
                "text_encoder": (
                    "STRING",
                    {"default": "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"},
                ),
                "video_vae": (
                    "STRING",
                    {
                        "default": "ltx-2.5-video-vae-conv-bf16.safetensors",
                        "tooltip": (
                            "The convolutional VAE is the first MLX target. The diffusion VAE "
                            "needs an efficient MLX neighborhood-attention implementation."
                        ),
                    },
                ),
                "audio_vae": (
                    "STRING",
                    {"default": "ltx-2.5-audio-vae-bf16.safetensors"},
                ),
                "spatial_upscaler": (
                    "STRING",
                    {"default": ("ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")},
                ),
                "duration_head": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional. Explicit duration is used when this is empty.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "specify"
    CATEGORY = "WeeTodd/LTX 2.5/loaders"
    DESCRIPTION = "Select LTX 2.5 split components without loading weights or downloading files."

    def specify(
        self,
        transformer,
        text_encoder,
        video_vae,
        audio_vae,
        spatial_upscaler,
        duration_head,
    ):
        optional_duration = (
            str(_resolve_component(duration_head, ("ltx25", "model_patches")))
            if duration_head.strip()
            else ""
        )
        return (
            LTX25ComponentSpec(
                transformer_path=str(
                    _resolve_component(transformer, ("diffusion_models", "ltx25", "checkpoints"))
                ),
                text_encoder_path=str(_resolve_component(text_encoder, ("text_encoders", "ltx25"))),
                video_vae_path=str(_resolve_component(video_vae, ("vae", "ltx25"))),
                audio_vae_path=str(_resolve_component(audio_vae, ("vae", "ltx25"))),
                spatial_upscaler_path=str(
                    _resolve_component(spatial_upscaler, ("latent_upscale_models", "ltx25"))
                ),
                duration_head_path=optional_duration,
            ),
        )


class WeeToddLTX25GenerationConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    list(LTX25_GENERATION_PRESETS),
                    {
                        "default": LTX25_GENERATION_PRESETS[0],
                        "tooltip": (
                            "Select a validated two-stage recipe or Custom to use the controls. "
                            "The selected seed is always preserved. The 1920×1088 option is "
                            "substantially slower and requires more unified memory."
                        ),
                    },
                ),
                "width": ("INT", {"default": 768, "min": 64, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1920, "step": 32}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.25, "max": 30.0, "step": 0.25},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "low_memory": ("BOOLEAN", {"default": True}),
                "low_ram_streaming": ("BOOLEAN", {"default": False}),
                "prompt_context": (
                    ["official_1024", "auto", "128", "256", "512", "1024"],
                    {
                        "default": "official_1024",
                        "tooltip": (
                            "Cap real prompt tokens before Gemma. The trained connector always "
                            "appends registers to at least 1024 tokens."
                        ),
                    },
                ),
                "feed_forward_backend": (
                    ["reference_fp32", "bf16_mpp_experimental"],
                    {
                        "default": "reference_fp32",
                        "tooltip": (
                            "bf16_mpp_experimental casts video feed-forward inputs to BF16 and "
                            "uses Metal Performance Primitives. It is faster but approximate."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "resolved_settings")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = "Configure the official distilled 8+3-evaluation LTX 2.5 two-stage schedule."

    def configure(
        self,
        preset,
        width,
        height,
        duration_seconds,
        frame_rate,
        seed,
        low_memory,
        low_ram_streaming,
        prompt_context,
        feed_forward_backend,
    ):
        values = apply_ltx25_generation_preset(
            preset,
            {
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds,
                "frame_rate": frame_rate,
                "low_memory": low_memory,
                "low_ram_streaming": low_ram_streaming,
                "prompt_context": prompt_context,
                "feed_forward_backend": feed_forward_backend,
            },
        )
        config = LTX25GenerationConfig(
            width=int(values["width"]),
            height=int(values["height"]),
            duration_seconds=float(values["duration_seconds"]),
            frame_rate=float(values["frame_rate"]),
            seed=secrets.randbelow(0x80000000) if seed < 0 else seed,
            low_memory=bool(values["low_memory"]),
            low_ram_streaming=bool(values["low_ram_streaming"]),
            prompt_context=str(values["prompt_context"]),
            feed_forward_backend=str(values["feed_forward_backend"]),
        )
        config.validate()
        info = {
            "preset": preset,
            **asdict(config),
            "num_frames": config.num_frames,
            "delivered_duration_seconds": config.delivered_duration_seconds,
            "real_evaluations": config.stage1_steps + config.stage2_steps,
        }
        return config, json.dumps(info, indent=2, sort_keys=True)


class WeeToddLTX25Preflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "config": ("WEETODD_LTX25_CONFIG",),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_MODEL", "STRING")
    RETURN_NAMES = ("model", "preflight_report")
    FUNCTION = "check"
    CATEGORY = "WeeTodd/LTX 2.5/loaders"

    def check(self, model, config):
        report = model.validate(config.pipeline_mode)
        scale_factors = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(scale_factors=scale_factors)
        result = {
            "component_contract_ok": True,
            "mlx_backend": backend_capability(),
            "pipeline_mode": config.pipeline_mode,
            "canvas": [config.width, config.height],
            "num_frames": config.num_frames,
            **report,
        }
        return model, json.dumps(result, indent=2, sort_keys=True)


class WeeToddLTX25Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "config": ("WEETODD_LTX25_CONFIG",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "WeeTodd/LTX25"}),
                "unload_after_generate": ("BOOLEAN", {"default": True}),
            },
            "optional": {"first_frame": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = "Generate synchronized LTX 2.5 video and audio through the MLX adapter."

    def generate(
        self,
        model,
        config,
        prompt,
        filename_prefix,
        unload_after_generate,
        first_frame=None,
    ):
        import numpy as np
        from PIL import Image

        report = model.validate(config.pipeline_mode)
        config.validate(scale_factors=tuple(int(value) for value in report["video_scale_factors"]))
        released = _release_h3_stages()
        final = _safe_target(filename_prefix, config.seed)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f".{final.stem}.partial{final.suffix}")
        metadata_path = final.with_suffix(".json")
        partial_metadata = final.with_name(f".{final.stem}.metadata.partial.json")
        image_path = None
        try:
            if first_frame is not None:
                frame = first_frame[0]
                detach = getattr(frame, "detach", None)
                if detach is not None:
                    frame = detach()
                cpu = getattr(frame, "cpu", None)
                if cpu is not None:
                    frame = cpu()
                frame = np.asarray(frame, dtype=np.float32)
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("LTX 2.5 first frame must be an RGB ComfyUI IMAGE.")
                image_path = final.with_name(f".{final.stem}.input.partial.png")
                Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)).save(image_path)
            info = RUNTIME.generate_to_file(
                model,
                config,
                prompt,
                partial,
                image_path=str(image_path) if image_path is not None else None,
                unload_after=unload_after_generate,
                check_interrupted=_check_interrupted(),
                step_callback=_comfy_progress(config.stage1_steps + config.stage2_steps),
            )
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("LTX 2.5 pipeline did not produce a video file.")
            info.update(
                video_path=str(final),
                h3_components_released=released,
                software=_software_versions(),
            )
            partial_metadata.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
            os.replace(partial, final)
            os.replace(partial_metadata, metadata_path)
            return {
                "ui": {"gifs": [_preview(final)]},
                "result": (str(final), json.dumps(info, indent=2, sort_keys=True)),
            }
        except BaseException:
            partial.unlink(missing_ok=True)
            partial_metadata.unlink(missing_ok=True)
            raise
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)


class WeeToddLTX25GenerateChained:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "config": ("WEETODD_LTX25_CONFIG",),
                "window_count": ("INT", {"default": 3, "min": 2, "max": 4, "step": 1}),
                "overlap_frames": (
                    "INT",
                    {
                        "default": 25,
                        "min": 9,
                        "max": 57,
                        "step": 8,
                        "tooltip": (
                            "Must equal 8n+1. Twenty-five frames is the balanced 24 fps default."
                        ),
                    },
                ),
                "prompt_1": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "prompt_2": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Leave empty to reuse the preceding window prompt.",
                    },
                ),
                "prompt_3": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": "Leave empty to reuse the preceding window prompt.",
                    },
                ),
                "prompt_4": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "",
                        "tooltip": (
                            "Used only when window_count is four; empty reuses window three."
                        ),
                    },
                ),
                "filename_prefix": ("STRING", {"default": "WeeTodd/LTX25_chained"}),
                "unload_after_generate": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = (
        "Generate two to four overlapping LTX 2.5 windows with timeline-aligned latent guides, "
        "causal-aware latent transitions, and one synchronized audio/video decode."
    )

    def generate(
        self,
        model,
        config,
        window_count,
        overlap_frames,
        prompt_1,
        prompt_2,
        prompt_3,
        prompt_4,
        filename_prefix,
        unload_after_generate,
    ):
        from ltx25_mlx.chaining import plan_ltx25_chain

        report = model.validate(config.pipeline_mode)
        config.validate(scale_factors=tuple(int(value) for value in report["video_scale_factors"]))
        plan = plan_ltx25_chain(
            total_frames=config.num_frames,
            window_count=int(window_count),
            overlap_frames=int(overlap_frames),
            frame_rate=config.frame_rate,
        )
        candidates = [prompt_1, prompt_2, prompt_3, prompt_4]
        prompts = []
        for index in range(int(window_count)):
            value = str(candidates[index]).strip()
            if not value:
                if not prompts:
                    raise ValueError("LTX 2.5 chained window one requires a prompt.")
                value = prompts[-1]
            prompts.append(value)

        released = _release_h3_stages()
        final = _safe_target(filename_prefix, config.seed)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f".{final.stem}.partial{final.suffix}")
        metadata_path = final.with_suffix(".json")
        partial_metadata = final.with_name(f".{final.stem}.metadata.partial.json")
        try:
            info = RUNTIME.generate_chain_to_file(
                model,
                config,
                prompts,
                partial,
                window_count=int(window_count),
                overlap_frames=int(overlap_frames),
                unload_after=unload_after_generate,
                check_interrupted=_check_interrupted(),
                step_callback=_comfy_progress(int(window_count) * 11),
            )
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("LTX 2.5 chained pipeline did not produce a video file.")
            info.update(
                video_path=str(final),
                h3_components_released=released,
                software=_software_versions(),
                chain_plan=plan.as_dict(),
            )
            partial_metadata.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
            os.replace(partial, final)
            os.replace(partial_metadata, metadata_path)
            return {
                "ui": {"gifs": [_preview(final)]},
                "result": (str(final), json.dumps(info, indent=2, sort_keys=True)),
            }
        except BaseException:
            partial.unlink(missing_ok=True)
            partial_metadata.unlink(missing_ok=True)
            raise


class WeeToddLTX25VideoUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "images": ("IMAGE",),
                "mode": (
                    list(LTX25_UPSCALE_MODES),
                    {
                        "default": LTX25_UPSCALE_MODES[0],
                        "tooltip": (
                            "All modes encode movie frames into LTX 2.5 latent space and use the "
                            "official learned 2× upscaler. Pixel-spatial mode also conditions "
                            "every output frame on the complete source clip through the native "
                            "LTX 2.5 IC-LoRA. Original movie audio is always preserved."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": (
                            "Describe the existing movie accurately for visual refinement."
                        ),
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "input_size_policy": (
                    list(LTX25_INPUT_SIZE_POLICIES),
                    {
                        "default": LTX25_INPUT_SIZE_POLICIES[0],
                        "tooltip": (
                            "LTX requires a 32-pixel grid. The recommended policy removes at "
                            "most 31 pixels from each axis with a centered crop and records the "
                            "exact crop in generation metadata."
                        ),
                    },
                ),
                "refinement_strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 0.85,
                        "step": 0.05,
                        "tooltip": (
                            "0.35 is the conservative cross-model default. 0.85 is the official "
                            "native LTX stage-two strength but may redraw source content."
                        ),
                    },
                ),
                "source_frame_anchors": (
                    list(LTX25_SOURCE_FRAME_ANCHORS),
                    {
                        "default": "first frame",
                        "tooltip": (
                            "Use the movie's own endpoint frames as native LTX anchors. "
                            "First frame is the balanced default; first + last provides stronger "
                            "protection but adds keyframe attention cost. External reference "
                            "sockets override the matching source endpoint."
                        ),
                    },
                ),
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Strength for optional native LTX first/last image conditioning. "
                            "Scene-like stills work better than multi-view character sheets."
                        ),
                    },
                ),
                "filename_prefix": ("STRING", {"default": "WeeTodd/LTX25_video_2x"}),
                "max_av_drift_seconds": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "low_ram_streaming": ("BOOLEAN", {"default": False}),
                "prompt_context": (
                    ["official_1024", "auto", "128", "256", "512", "1024"],
                    {"default": "official_1024"},
                ),
                "generation_metadata": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
                "pixel_spatial_lora": (
                    "STRING",
                    {
                        "default": (
                            "LTX-2.5/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
                        ),
                        "tooltip": (
                            "Required only by the recommended pixel-spatial mode. The file is "
                            "resolved through every configured ComfyUI LoRA folder."
                        ),
                    },
                ),
                "pixel_spatial_lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.05,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "The official LTX 2.5 checkpoint recommends 1.0.",
                    },
                ),
            },
            "optional": {
                "first_reference": ("IMAGE",),
                "last_reference": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "upscale"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = (
        "Upscale decoded ComfyUI IMAGE+AUDIO from any movie through LTX 2.5 latent space, "
        "optionally adding video-only refinement while preserving the source audio."
    )

    def upscale(
        self,
        model,
        images,
        mode,
        prompt,
        seed,
        fps,
        input_size_policy,
        refinement_strength,
        source_frame_anchors,
        reference_strength,
        filename_prefix,
        max_av_drift_seconds,
        low_ram_streaming,
        prompt_context,
        generation_metadata,
        pixel_spatial_lora,
        pixel_spatial_lora_strength,
        first_reference=None,
        last_reference=None,
        audio=None,
    ):
        import numpy as np
        from PIL import Image

        metadata = json.loads(generation_metadata or "{}")
        if not isinstance(metadata, dict):
            raise ValueError("generation_metadata must be a JSON object.")
        model.validate()
        released = _release_h3_stages()
        final = _safe_target(filename_prefix, seed)
        reference_paths = []

        def save_reference(value, label):
            if value is None:
                return None
            frame = value[0]
            detach = getattr(frame, "detach", None)
            if detach is not None:
                frame = detach()
            cpu = getattr(frame, "cpu", None)
            if cpu is not None:
                frame = cpu()
            frame = np.asarray(frame, dtype=np.float32)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError("LTX 2.5 references must be RGB ComfyUI IMAGE inputs.")
            path = final.with_name(f".{final.stem}.{label}.partial.png")
            Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
            reference_paths.append(path)
            return str(path)

        try:
            first_path = save_reference(first_reference, "first-reference")
            last_path = save_reference(last_reference, "last-reference")
            pixel_lora_path = None
            if mode == LTX25_PIXEL_SPATIAL_MODE:
                pixel_lora_path = str(_resolve_component(pixel_spatial_lora, ("loras", "ltx25")))
            from ltx25_mlx.upscale import upscale_video_to_file

            result = upscale_video_to_file(
                model,
                images,
                audio,
                final,
                mode=mode,
                prompt=prompt,
                seed=seed,
                fps=fps,
                input_size_policy=input_size_policy,
                refinement_strength=refinement_strength,
                source_frame_anchors=source_frame_anchors,
                first_reference_path=first_path,
                last_reference_path=last_path,
                reference_strength=reference_strength,
                max_av_drift_seconds=max_av_drift_seconds,
                low_ram_streaming=low_ram_streaming,
                prompt_context=prompt_context,
                pixel_spatial_lora_path=pixel_lora_path,
                pixel_spatial_lora_strength=pixel_spatial_lora_strength,
                generation_metadata={
                    **metadata,
                    "h3_components_released": released,
                    "software": _software_versions(),
                },
                check_interrupted=_check_interrupted(),
                step_callback=(_comfy_progress(3) if mode != LTX25_UPSCALE_MODES[0] else None),
            )
        finally:
            for path in reference_paths:
                path.unlink(missing_ok=True)
        return {
            "ui": {"gifs": [_preview(result.video_path)]},
            "result": (
                str(result.video_path),
                json.dumps(result.metadata, indent=2, sort_keys=True),
            ),
        }


class WeeToddLTX25Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/LTX 2.5"

    def release(self, unload):
        if unload:
            RUNTIME.unload()
            return ("LTX 2.5 MLX runtime unloaded",)
        return ("LTX 2.5 MLX runtime kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddLTX25ComponentLoader": WeeToddLTX25ComponentLoader,
    "WeeToddLTX25GenerationConfig": WeeToddLTX25GenerationConfig,
    "WeeToddLTX25Preflight": WeeToddLTX25Preflight,
    "WeeToddLTX25Generate": WeeToddLTX25Generate,
    "WeeToddLTX25GenerateChained": WeeToddLTX25GenerateChained,
    "WeeToddLTX25VideoUpscale": WeeToddLTX25VideoUpscale,
    "WeeToddLTX25Unload": WeeToddLTX25Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddLTX25ComponentLoader": "WeeTodd LTX 2.5 Component Loader (MLX)",
    "WeeToddLTX25GenerationConfig": "WeeTodd LTX 2.5 Generation Config",
    "WeeToddLTX25Preflight": "WeeTodd LTX 2.5 Preflight",
    "WeeToddLTX25Generate": "WeeTodd LTX 2.5 Generate Video + Audio",
    "WeeToddLTX25GenerateChained": "WeeTodd LTX 2.5 Generate Chained Timeline",
    "WeeToddLTX25VideoUpscale": "WeeTodd LTX 2.5 Video Upscale / Refine",
    "WeeToddLTX25Unload": "WeeTodd LTX 2.5 Unload MLX Runtime",
}
