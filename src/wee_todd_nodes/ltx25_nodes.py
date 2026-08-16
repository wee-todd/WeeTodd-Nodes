"""ComfyUI adapters for the standalone LTX 2.5 MLX split-component pipeline."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ltx25_mlx.runtime import (
    LTX25_DEFAULT_NEGATIVE_PROMPT,
    LTX25_DIFFVAE_OPTIMIZATIONS,
    LTX25_GENERATION_PRESETS,
    RUNTIME,
    LTX25ComponentSpec,
    LTX25GenerationConfig,
    apply_ltx25_generation_preset,
    backend_capability,
    validate_ltx25_dfr_prebaked_pair,
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


@dataclass(frozen=True)
class LTX25Keyframe:
    image: object
    frame_index: int
    strength: float = 1.0


@dataclass(frozen=True)
class LTX25KeyframeStack:
    keyframes: tuple[LTX25Keyframe, ...] = ()

    def append(self, value: LTX25Keyframe) -> LTX25KeyframeStack:
        if not 0.0 <= value.strength <= 1.0:
            raise ValueError("LTX 2.5 keyframe strength must be between zero and one.")
        shape = getattr(value.image, "shape", None)
        if shape is None or len(shape) != 4 or int(shape[0]) < 1 or int(shape[-1]) < 3:
            raise ValueError("LTX 2.5 keyframe must be a ComfyUI IMAGE batch.")
        result = LTX25KeyframeStack((*self.keyframes, value))
        if len(result.keyframes) > 8:
            raise ValueError("LTX 2.5 supports at most eight explicit keyframes per window.")
        if len({item.frame_index for item in result.keyframes}) != len(result.keyframes):
            raise ValueError("Two LTX 2.5 keyframes cannot target the same frame.")
        return result

    def validate(self, num_frames: int) -> None:
        if not self.keyframes:
            raise ValueError("LTX 2.5 keyframe conditioning requires at least one image.")
        for item in self.keyframes:
            if not 0 <= item.frame_index < num_frames:
                raise ValueError(
                    f"LTX 2.5 keyframe {item.frame_index} is outside 0..{num_frames - 1}."
                )

    def metadata(self) -> dict[str, object]:
        return {
            "keyframes": [
                {"frame_index": item.frame_index, "strength": item.strength}
                for item in sorted(self.keyframes, key=lambda value: value.frame_index)
            ]
        }


LTX25_MEDIA_ROLES = (
    "image_keyframe",
    "video_reference",
    "audio_reference",
    "inpaint_mask",
)

LTX25_IC_CONTROL_TYPES = (
    "canny_edges",
    "depth_map",
    "pose_skeleton",
    "motion_track",
    "ingredients_reference_sheet",
    "custom_preprocessed",
)


@dataclass(frozen=True)
class LTX25MediaConditioningItem:
    role: str
    start_frame: int
    end_frame: int
    strength: float
    attention_strength: float = 1.0
    images: object | None = None
    audio: object | None = None
    mask: object | None = None
    control_type: str = "custom_preprocessed"


@dataclass(frozen=True)
class LTX25MediaConditioningStack:
    items: tuple[LTX25MediaConditioningItem, ...] = ()

    def append(self, item: LTX25MediaConditioningItem) -> LTX25MediaConditioningStack:
        if item.role not in LTX25_MEDIA_ROLES:
            raise ValueError(f"Unsupported LTX 2.5 media role: {item.role!r}.")
        if not 0.0 <= item.strength <= 1.0:
            raise ValueError("LTX 2.5 media strength must be between zero and one.")
        if not 0.0 <= item.attention_strength <= 1.0:
            raise ValueError("LTX 2.5 media attention strength must be between zero and one.")
        if item.start_frame < 0 or item.end_frame < item.start_frame:
            raise ValueError("LTX 2.5 media frame range is invalid.")
        if item.role == "image_keyframe":
            shape = getattr(item.images, "shape", None)
            if shape is None or len(shape) != 4 or int(shape[0]) != 1:
                raise ValueError("An image keyframe requires exactly one ComfyUI IMAGE.")
            if item.end_frame != item.start_frame:
                raise ValueError("An image keyframe targets one exact frame.")
        elif item.role == "video_reference":
            shape = getattr(item.images, "shape", None)
            if shape is None or len(shape) != 4 or int(shape[0]) < 1:
                raise ValueError("A video reference requires a ComfyUI IMAGE frame batch.")
            if item.end_frame == item.start_frame and int(shape[0]) > 1:
                item = replace(item, end_frame=item.start_frame + int(shape[0]) - 1)
            if item.control_type not in LTX25_IC_CONTROL_TYPES:
                raise ValueError(
                    f"Unsupported LTX 2.5 IC-LoRA control type: {item.control_type!r}."
                )
        elif item.role == "audio_reference":
            if item.audio is None:
                raise ValueError("An audio reference requires a ComfyUI AUDIO input.")
            if item.strength != 1.0:
                raise ValueError(
                    "LTX 2.5 audio-driven conditioning freezes and preserves its source; "
                    "audio strength must be 1.0."
                )
        elif item.mask is None:
            raise ValueError("An inpaint mask requires a ComfyUI MASK input.")
        if len(self.items) >= 20:
            raise ValueError("LTX 2.5 media conditioning supports at most twenty items.")
        return LTX25MediaConditioningStack((*self.items, item))

    def validate_for_generation(self, num_frames: int) -> None:
        for item in self.items:
            if item.end_frame >= num_frames:
                raise ValueError(
                    f"LTX 2.5 media range {item.start_frame}..{item.end_frame} is outside "
                    f"0..{num_frames - 1}."
                )
            if item.role not in {"image_keyframe", "video_reference", "audio_reference"}:
                raise ValueError(
                    f"LTX 2.5 {item.role} is represented by the shared media contract but "
                    "requires its IC-LoRA or audio pipeline, which is not enabled by the "
                    "current Generate node yet."
                )

    def metadata(self) -> dict[str, object]:
        return {
            "items": [
                {
                    "role": item.role,
                    "start_frame": item.start_frame,
                    "end_frame": item.end_frame,
                    "strength": item.strength,
                    "attention_strength": item.attention_strength,
                    **(
                        {"control_type": item.control_type}
                        if item.role == "video_reference"
                        else {}
                    ),
                }
                for item in self.items
            ]
        }


class WeeToddLTX25MediaConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (list(LTX25_MEDIA_ROLES), {"default": "image_keyframe"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "attention_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "IC-LoRA attention weight. Leave at 1.0 for full reference "
                            "attention; lower it to relax the control signal."
                        ),
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mask": ("MASK",),
                "previous_conditioning": ("WEETODD_LTX25_MEDIA_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_MEDIA_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Build a shared LTX 2.5 image, video, audio, or mask conditioning stack. "
        "Image keyframes, IC-LoRA video references, and one frozen audio-driven source execute. "
        "Standalone inpaint masks remain gated."
    )

    def append(
        self,
        role,
        start_frame,
        end_frame,
        strength,
        attention_strength=1.0,
        images=None,
        audio=None,
        mask=None,
        previous_conditioning=None,
    ):
        stack = previous_conditioning or LTX25MediaConditioningStack()
        updated = stack.append(
            LTX25MediaConditioningItem(
                role=str(role),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                strength=float(strength),
                attention_strength=float(attention_strength),
                images=images,
                audio=audio,
                mask=mask,
            )
        )
        return updated, json.dumps(updated.metadata(), indent=2, sort_keys=True)


class WeeToddLTX25ICLoRAControlGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_video": ("IMAGE",),
                "control_type": (
                    list(LTX25_IC_CONTROL_TYPES),
                    {
                        "default": "canny_edges",
                        "tooltip": (
                            "Connect preprocessed Canny, depth, or DWPose frames. Motion Track "
                            "accepts the source video frames directly."
                        ),
                    },
                ),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "end_frame": ("INT", {"default": 120, "min": 0, "max": 100000}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "attention_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "attention_mask": ("MASK",),
                "previous_conditioning": ("WEETODD_LTX25_MEDIA_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_MEDIA_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "guide_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Add one preprocessed Canny, depth, pose, Motion Track, or custom IC-LoRA guide. "
        "Use the LTX 2.5 distilled model and the matching task adapter."
    )

    def append(
        self,
        control_video,
        control_type,
        start_frame,
        end_frame,
        strength,
        attention_strength,
        attention_mask=None,
        previous_conditioning=None,
    ):
        stack = previous_conditioning or LTX25MediaConditioningStack()
        updated = stack.append(
            LTX25MediaConditioningItem(
                role="video_reference",
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                strength=float(strength),
                attention_strength=float(attention_strength),
                images=control_video,
                mask=attention_mask,
                control_type=str(control_type),
            )
        )
        control_groups = sorted(
            {
                item.control_type
                for item in updated.items
                if item.role == "video_reference"
            }
        )
        return updated, json.dumps(
            {
                **updated.metadata(),
                "control_groups": control_groups,
                "single_group_recommended": len(control_groups) <= 1,
                "preprocessing": (
                    "raw_source_video"
                    if control_type == "motion_track"
                    else "preprocessed_image_batch"
                ),
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddLTX25ICLoRAPipelineMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "mode": (
                    [
                        "CFG++ quality — 8 steps / 15 real forwards",
                        "Fast single stage — 8 ancestral steps / 8 real forwards",
                        "Two stage — 8 low-resolution + 3 clean refinement",
                    ],
                    {
                        "default": "CFG++ quality — 8 steps / 15 real forwards",
                        "tooltip": (
                            "CFG++ quality runs both branches where the correction is useful; "
                            "the terminal step needs only its conditional prediction. Fast "
                            "single stage uses the distinct ancestral sampler."
                        ),
                    },
                ),
            },
            "optional": {
                "cfg_pp_execution": (
                    ["automatic", "batched", "serial"],
                    {
                        "default": "automatic",
                        "tooltip": (
                            "Automatic selects serial, which is fastest on the measured Apple "
                            "Silicon system. Batched is experimental: it reduces peak memory "
                            "but was slower at the trained 768x448 bucket."
                        ),
                    },
                ),
                "cfg_pp_schedule": (
                    ["full", "balanced", "speed"],
                    {
                        "default": "full",
                        "tooltip": (
                            "Full uses 15 useful forwards; balanced uses 12; speed uses 10. "
                            "Hybrid schedules trade some CFG++ influence for real compute savings."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "mode_info")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Select full or hybrid CFG++, the eight-forward single-stage shortcut, "
        "or the existing two-stage stage-one-control pipeline."
    )

    def apply(self, config, mode, cfg_pp_execution="automatic", cfg_pp_schedule="full"):
        mode = str(mode)
        cfg_pp_mode = mode.startswith("CFG++ quality") or mode.startswith(
            "Official Comfy parity"
        )
        fast_single_stage = mode.startswith("Fast single stage")
        single_stage = cfg_pp_mode or fast_single_stage
        cfg_pp_batched = cfg_pp_mode and str(cfg_pp_execution) == "batched"
        updated = replace(
            config,
            ic_lora_single_stage=single_stage,
            stage2_steps=0 if single_stage else 3,
            stage1_sampler=(
                "euler_ancestral_cfg_pp" if cfg_pp_mode else "euler_ancestral"
            ),
            cfg_pp_batched=cfg_pp_batched,
            cfg_pp_schedule=str(cfg_pp_schedule) if cfg_pp_mode else "full",
            negative_prompt="" if cfg_pp_mode else config.negative_prompt,
        )
        updated.validate()
        return updated, json.dumps(
            {
                "mode": (
                    "cfg_pp_single_stage"
                    if cfg_pp_mode
                    else "single_stage_fast"
                    if fast_single_stage
                    else "two_stage"
                ),
                "stage1_steps": updated.stage1_steps,
                "stage1_real_forwards": updated.stage1_forward_passes,
                "stage2_steps": updated.stage2_steps,
                "total_real_forwards": updated.real_forward_passes,
                "sampler": updated.stage1_sampler,
                "cfg_pp_execution": "batched" if updated.cfg_pp_batched else "serial",
                "cfg_pp_schedule": updated.cfg_pp_schedule,
                "negative_prompt": updated.negative_prompt,
                "recommended_ingredients_lora_strength": 1.2,
                "ic_lora_stage_scope": (
                    "full_resolution_complete_generation"
                    if single_stage
                    else "low_resolution_stage_1_only"
                ),
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddLTX25ReferenceSheetGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_sheet": ("IMAGE",),
                "reference_description": (
                    "STRING",
                    {
                        "default": "Describe the characters, props, and location panels.",
                        "multiline": True,
                    },
                ),
                "generated_video": (
                    "STRING",
                    {
                        "default": "Describe the shot, action, camera, and synchronized audio.",
                        "multiline": True,
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "attention_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {
                "attention_mask": ("MASK",),
                "previous_conditioning": ("WEETODD_LTX25_MEDIA_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_MEDIA_CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "prompt", "reference_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Condition LTX 2.5 from one Ingredients reference sheet. The image is repeated "
        "internally across the full clip and encoded as IC-LoRA reference context."
    )

    def append(
        self,
        reference_sheet,
        reference_description,
        generated_video,
        strength,
        attention_strength,
        attention_mask=None,
        previous_conditioning=None,
    ):
        shape = getattr(reference_sheet, "shape", None)
        if shape is None or len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) < 3:
            raise ValueError("An LTX 2.5 Ingredients reference sheet requires one RGB image.")
        reference_description = str(reference_description).strip()
        generated_video = str(generated_video).strip()
        if not reference_description or not generated_video:
            raise ValueError(
                "Ingredients prompting requires both a reference-sheet description and a "
                "generated-video description."
            )
        stack = (previous_conditioning or LTX25MediaConditioningStack()).append(
            LTX25MediaConditioningItem(
                role="video_reference",
                start_frame=0,
                end_frame=0,
                strength=float(strength),
                attention_strength=float(attention_strength),
                images=reference_sheet,
                mask=attention_mask,
                control_type="ingredients_reference_sheet",
            )
        )
        prompt = (
            f"Reference sheet: {reference_description}\n\n"
            f"Generated video: {generated_video}"
        )
        return stack, prompt, json.dumps(
            {
                **stack.metadata(),
                "conditioning_mode": "static_reference_sheet_repeated_to_target",
                "recommended_canvas": "768x448",
                "recommended_frames": 121,
                "recommended_frame_rate": 24,
                "recommended_pipeline": "single_stage_full_resolution",
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddLTX25Keyframe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100000, "step": 1},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            },
            "optional": {"previous_keyframes": ("WEETODD_LTX25_KEYFRAMES",)},
        }

    RETURN_TYPES = ("WEETODD_LTX25_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Append a first, middle, or last image at an exact zero-based pixel-frame index. "
        "The image is encoded as reference conditioning; generated keyframe slots are separate."
    )

    def append(self, image, frame_index, strength, previous_keyframes=None):
        stack = (previous_keyframes or LTX25KeyframeStack()).append(
            LTX25Keyframe(image, int(frame_index), float(strength))
        )
        return stack, json.dumps(stack.metadata(), indent=2, sort_keys=True)


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
                            "Choose the lower-cost convolutional VAE or the official one-step "
                            "Diffusion VAE. The MLX Diffusion VAE uses bounded neighborhood "
                            "attention. Use the separate optimization node to select its layout."
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
            },
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
        optional_spatial_upscaler = (
            str(
                _resolve_component(
                    spatial_upscaler,
                    ("latent_upscale_models", "ltx25"),
                )
            )
            if spatial_upscaler.strip()
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
                spatial_upscaler_path=optional_spatial_upscaler,
                duration_head_path=optional_duration,
            ),
        )


class WeeToddLTX25LoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "lora": ("STRING", {"default": ""}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_MODEL", "STRING")
    RETURN_NAMES = ("model", "lora_info")
    FUNCTION = "attach"
    CATEGORY = "WeeTodd/LTX 2.5/loaders"
    DESCRIPTION = (
        "Attach a generic LTX 2.5 transformer LoRA, including block and non-block targets. "
        "Multiple loader nodes may be chained. Use the dedicated loader for IC-LoRA task "
        "adapters."
    )

    def attach(self, model, lora, strength):
        resolved = _resolve_component(lora, ("loras", "ltx25"))
        from ltx25_mlx.transformer import inspect_ltx25_lora

        report = inspect_ltx25_lora(resolved)
        if report["adapter_role"] == "ic_lora":
            raise ValueError(
                "This checkpoint declares IC-LoRA reference metadata. Use the dedicated "
                "LTX 2.5 IC-LoRA Loader so it is scoped to the reference-conditioned stage."
            )
        attached = replace(
            model,
            loras=(*model.loras, (str(resolved), float(strength))),
        )
        report["path"] = str(report["path"])
        return attached, json.dumps(
            {**report, "strength": float(strength), "stack_size": len(attached.loras)},
            indent=2,
            sort_keys=True,
        )


class WeeToddLTX25ICLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "ic_lora": ("STRING", {"default": ""}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 2.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_MODEL", "STRING")
    RETURN_NAMES = ("model", "ic_lora_info")
    FUNCTION = "attach"
    CATEGORY = "WeeTodd/LTX 2.5/loaders"
    DESCRIPTION = (
        "Attach one LTX 2.5-compatible IC-LoRA for video/reference conditioning. "
        "Official LTX 2.3 22B adapters pass an additional shape check. The selected IC-LoRA "
        "Pipeline Mode determines whether the adapter runs for stage one or the full generation."
    )

    def attach(self, model, ic_lora, strength):
        if model.ic_loras:
            raise ValueError(
                "Only one IC-LoRA adapter may be active. Remove the existing IC-LoRA "
                "loader before attaching another adapter."
            )
        resolved = _resolve_component(ic_lora, ("loras", "ltx25"))
        from ltx25_mlx.transformer import inspect_ltx25_lora

        report = inspect_ltx25_lora(resolved)
        if report["adapter_role"] != "ic_lora":
            raise ValueError(
                "The selected checkpoint is a transformer LoRA, not an IC-LoRA with "
                "reference scale metadata."
            )
        if report["ic_lora_task"] == "pixel_spatial_upscaler":
            raise ValueError(
                "The selected IC-LoRA is the Pixel-Spatial Upscaler, not a general "
                "video/reference adapter. Use LTX 2.5 Video Upscale / Refine for it."
            )
        attached = replace(
            model,
            ic_loras=(*model.ic_loras, (str(resolved), float(strength))),
        )
        report["path"] = str(report["path"])
        return attached, json.dumps(
            {
                **report,
                "strength": float(strength),
                "stack_size": len(attached.ic_loras),
                "stage_scope": "selected_by_ic_lora_pipeline_mode",
            },
            indent=2,
            sort_keys=True,
        )
class WeeToddLTX25GuidedModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX25_MODEL",),
                "development_transformer": (
                    "STRING",
                    {"default": "ltx-2.5-22b-dev-transformer-bf16.safetensors"},
                ),
                "distilled_lora": (
                    "STRING",
                    {"default": "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_MODEL", "STRING")
    RETURN_NAMES = ("model", "guided_model_info")
    FUNCTION = "attach"
    CATEGORY = "WeeTodd/LTX 2.5/loaders"
    DESCRIPTION = (
        "Select the LTX 2.5 development transformer for guided stage one and the official "
        "rank-450 distilled LoRA for stage two. No weights load in this node."
    )

    def attach(self, model, development_transformer, distilled_lora):
        transformer = _resolve_component(
            development_transformer, ("diffusion_models", "ltx25", "checkpoints")
        )
        adapter = _resolve_component(distilled_lora, ("loras", "ltx25"))
        from ltx25_mlx.transformer import inspect_ltx25_lora

        report = inspect_ltx25_lora(adapter)
        if (
            report["adapter_role"] != "transformer_lora"
            or report["lora_rank"] != 450
            or report["lora_alpha"] != 450
        ):
            raise ValueError(
                "Guided LTX 2.5 requires the official rank-450/alpha-450 distilled LoRA."
            )
        updated = replace(
            model,
            transformer_path=str(transformer),
            distilled_lora_path=str(adapter),
        )
        return updated, json.dumps(
            {
                "stage1_transformer": str(transformer),
                "stage2_distilled_lora": str(adapter),
                "lora_rank": report["lora_rank"],
                "lora_alpha": report["lora_alpha"],
            },
            indent=2,
            sort_keys=True,
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
                    [
                        "reference_fp32",
                        "mlx_fused_experimental",
                        "bf16_mpp_experimental",
                    ],
                    {
                        "default": "reference_fp32",
                        "tooltip": (
                            "mlx_fused_experimental compiles exact RMS-AdaLN, audiovisual FF, "
                            "gate, and residual graphs. bf16_mpp_experimental casts video FF "
                            "inputs to BF16 and is faster but approximate."
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
            "sampler_steps": config.stage1_steps + config.stage2_steps,
            "real_forward_passes": config.real_forward_passes,
        }
        return config, json.dumps(info, indent=2, sort_keys=True)


LTX25_QUALITY_MODES = (
    "Fast distilled — 8 ancestral + 3 deterministic",
    "Production guided — 30 Euler + 3 deterministic",
    "HQ guided — 15 res_2s + 3 deterministic",
)


class WeeToddLTX25QualityMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "mode": (list(LTX25_QUALITY_MODES), {"default": LTX25_QUALITY_MODES[0]}),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": LTX25_DEFAULT_NEGATIVE_PROMPT,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "quality_mode_info")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = (
        "Choose fast distilled inference, production guided Euler, or the official HQ "
        "second-order res_2s recipe without changing the base Generation Config schema."
    )

    def apply(self, config, mode, negative_prompt):
        if mode == LTX25_QUALITY_MODES[0]:
            values = {
                "pipeline_mode": "distilled",
                "stage1_steps": 8,
                "stage1_sampler": "euler_ancestral",
                "video_cfg_scale": 1.0,
                "audio_cfg_scale": 1.0,
                "stg_scale": 0.0,
                "video_rescale_scale": 0.0,
                "audio_rescale_scale": 0.0,
                "modality_scale": 1.0,
                "stg_blocks": (),
            }
        elif mode == LTX25_QUALITY_MODES[1]:
            values = {
                "pipeline_mode": "guided",
                "stage1_steps": 30,
                "stage1_sampler": "euler_guided",
                "video_cfg_scale": 3.0,
                "audio_cfg_scale": 7.0,
                "stg_scale": 1.0,
                "video_rescale_scale": 0.7,
                "audio_rescale_scale": 0.7,
                "modality_scale": 3.0,
                "stg_blocks": (28,),
            }
        elif mode == LTX25_QUALITY_MODES[2]:
            values = {
                "pipeline_mode": "guided_hq",
                "stage1_steps": 15,
                "stage1_sampler": "res_2s_guided",
                "video_cfg_scale": 3.0,
                "audio_cfg_scale": 7.0,
                "stg_scale": 0.0,
                "video_rescale_scale": 0.45,
                "audio_rescale_scale": 1.0,
                "modality_scale": 3.0,
                "stg_blocks": (),
            }
        else:
            raise ValueError(f"Unsupported LTX 2.5 quality mode: {mode!r}.")
        updated = replace(config, negative_prompt=str(negative_prompt), **values)
        updated.validate()
        info = {
            "mode": mode,
            "pipeline_mode": updated.pipeline_mode,
            "stage1_sampler": updated.stage1_sampler,
            "stage1_iterations": updated.stage1_steps,
            "stage2_iterations": updated.stage2_steps,
            "guidance": {
                "video_cfg": updated.video_cfg_scale,
                "audio_cfg": updated.audio_cfg_scale,
                "stg": updated.stg_scale,
                "modality": updated.modality_scale,
            },
            "requires_guided_model_loader": updated.pipeline_mode != "distilled",
        }
        return updated, json.dumps(info, indent=2, sort_keys=True)


class WeeToddLTX25AutoDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "minimum_seconds": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 30.0, "step": 0.25},
                ),
                "maximum_seconds": (
                    "FLOAT",
                    {"default": 20.0, "min": 0.25, "max": 30.0, "step": 0.25},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "duration_settings")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5"
    DESCRIPTION = (
        "Predict one-shot duration from the prompt with the official LTX 2.5 duration head. "
        "The manual duration remains unchanged when this modifier is not connected."
    )

    def apply(self, config, minimum_seconds, maximum_seconds):
        updated = replace(
            config,
            duration_mode="automatic",
            auto_duration_min_seconds=float(minimum_seconds),
            auto_duration_max_seconds=float(maximum_seconds),
        )
        updated.validate()
        info = {
            "mode": "automatic",
            "minimum_seconds": updated.auto_duration_min_seconds,
            "maximum_seconds": updated.auto_duration_max_seconds,
            "requires_component": "ltx-2.5-duration-head-bf16.safetensors",
            "scope": "one-shot generation",
        }
        return updated, json.dumps(info, indent=2, sort_keys=True)


class WeeToddLTX25GeneratedKeyframes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "generated_keyframes": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Append learned interior keyframe slots during stage one. Higher "
                            "values can help difficult motion but add attention cost."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = "Apply LTX 2.5 generated interior keyframe slots as a composable config modifier."

    def apply(self, config, generated_keyframes):
        updated = replace(config, generated_keyframes=int(generated_keyframes))
        updated.validate()
        return (updated,)


class WeeToddLTX25DiffVAEOptimization:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "optimization": (
                    list(LTX25_DIFFVAE_OPTIMIZATIONS),
                    {
                        "default": "combined",
                        "tooltip": (
                            "Combined preserves the reference MLX path. Metal NA3D is an "
                            "approximate experimental kernel measured 4.49x faster at 512px with "
                            "a 3.00 GB peak reduction against the reference decoder. Query-tiled "
                            "Metal saves more memory but is slower; use a 65536-row query tile. "
                            "Deferred stage four is exact. Width tiles trade substantial decode "
                            "time for a modest peak reduction."
                        ),
                    },
                ),
                "query_chunk_size": (
                    "INT",
                    {"default": 512, "min": 32, "max": 262144, "step": 32},
                ),
                "context_width_chunks": (
                    "INT",
                    {"default": 4, "min": 1, "max": 32, "step": 1},
                ),
                "stage4_tile_width": (
                    "INT",
                    {"default": 32, "min": 1, "max": 512, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "diffvae_info")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5/optimization"
    DESCRIPTION = (
        "Select an MLX Diffusion VAE execution layout. It does not affect the convolutional VAE."
    )

    def apply(
        self,
        config,
        optimization,
        query_chunk_size,
        context_width_chunks,
        stage4_tile_width,
    ):
        updated = replace(
            config,
            diffvae_optimization=str(optimization),
            diffvae_query_chunk_size=int(query_chunk_size),
            diffvae_context_width_chunks=int(context_width_chunks),
            diffvae_stage4_tile_width=int(stage4_tile_width),
        )
        updated.validate()
        details = {
            "optimization": updated.diffvae_optimization,
            "query_chunk_size": updated.diffvae_query_chunk_size,
            "context_width_chunks": updated.diffvae_context_width_chunks,
            "stage4_tile_width": updated.diffvae_stage4_tile_width,
            "applies_only_to": "Diffusion VAE checkpoints",
        }
        return updated, json.dumps(details, indent=2, sort_keys=True)


class WeeToddLTX25DFRDetailing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "detailing_lora": (
                    "STRING",
                    {
                        "default": (
                            "LTX-2.5/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
                        ),
                        "tooltip": (
                            "Official 2x Pixel-Spatial IC-LoRA. DFR applies it only during "
                            "the full-resolution stage-two detailing pass."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.05},
                ),
            },
            "optional": {
                "prebaked_stage2_transformer": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Optional paged Q8 transformer with the base and Pixel-Spatial "
                            "LoRAs already baked together. Leave empty for live fusion."
                        ),
                    },
                )
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "dfr_info")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Enable MLX Diffusion Fidelity Rendering: segment-grid generated keyframes, "
        "stage-one latent reference conditioning, stage-two-only Pixel-Spatial IC-LoRA, "
        "optional exact prebaked Q8 adapter pages, and untouched stage-one audio publication."
    )

    def apply(self, config, detailing_lora, strength, prebaked_stage2_transformer=""):
        resolved = _resolve_component(detailing_lora, ("loras", "ltx25"))
        prebaked = (
            _resolve_component(prebaked_stage2_transformer, ("diffusion_models", "ltx25"))
            if prebaked_stage2_transformer.strip()
            else None
        )
        updated = replace(
            config,
            generated_keyframes=0,
            dfr_enabled=True,
            dfr_detailing_lora_path=str(resolved),
            dfr_detailing_lora_strength=float(strength),
            dfr_prebaked_transformer_path=str(prebaked) if prebaked is not None else "",
        )
        updated.validate()
        from ltx25_mlx.dfr import resolve_dfr_canvas

        padded, segment, positions = resolve_dfr_canvas(updated.num_frames)
        return updated, json.dumps(
            {
                "enabled": True,
                "detailing_lora": str(resolved),
                "strength": float(strength),
                "prebaked_stage2_transformer": (
                    str(prebaked) if prebaked is not None else None
                ),
                "requested_frames": updated.num_frames,
                "internal_canvas_frames": padded,
                "segment_frames": segment,
                "generated_keyframe_positions": positions,
                "audio_source": "stage_1",
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddLTX25DFRTemporalRefinement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_LTX25_CONFIG",),
                "temporal_upsampler": (
                    "STRING",
                    {
                        "default": (
                            "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
                        )
                    },
                ),
                "rounds": ("INT", {"default": 1, "min": 1, "max": 2, "step": 1}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX25_CONFIG", "STRING")
    RETURN_NAMES = ("config", "temporal_settings")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/LTX 2.5/conditioning"
    DESCRIPTION = (
        "Experimentally add one or two learned x2 temporal DFR rounds. Each round preserves "
        "stage-one audio, doubles playback frame rate, reapplies one-shot image anchors, and "
        "adds four transformer evaluations per temporal tile. Current MLX visual parity is not "
        "yet production-validated."
    )

    def apply(self, config, temporal_upsampler, rounds):
        resolved = _resolve_component(
            temporal_upsampler,
            ("latent_upscale_models", "ltx25"),
        )
        from ltx25_mlx.components import inspect_ltx25_latent_upsampler

        upsampler_report = inspect_ltx25_latent_upsampler(resolved)
        if upsampler_report["spatial_upsample"] or not upsampler_report[
            "temporal_upsample"
        ]:
            raise ValueError("LTX 2.5 temporal refinement requires a temporal-only upsampler.")
        updated = replace(
            config,
            dfr_temporal_upsampler_path=str(resolved),
            dfr_temporal_rounds=int(rounds),
        )
        updated.validate()
        info = {
            "rounds": updated.dfr_temporal_rounds,
            "output_frame_multiplier": 2**updated.dfr_temporal_rounds,
            "output_fps_multiplier": 2**updated.dfr_temporal_rounds,
            "audio_policy": "preserve stage-one audio",
            "transformer_evaluations_per_tile_per_round": 4,
            "upsampler": upsampler_report,
        }
        return updated, json.dumps(info, indent=2, sort_keys=True)


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
        report = model.validate(
            config.pipeline_mode,
            require_spatial_upscaler=not config.ic_lora_single_stage,
        )
        scale_factors = tuple(int(value) for value in report["video_scale_factors"])
        config.validate(scale_factors=scale_factors)
        validate_ltx25_dfr_prebaked_pair(config, report)
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
            "optional": {
                "first_frame": ("IMAGE",),
                "keyframes": ("WEETODD_LTX25_KEYFRAMES",),
                "media_conditioning": ("WEETODD_LTX25_MEDIA_CONDITIONING",),
            },
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
        keyframes=None,
        media_conditioning=None,
    ):
        import numpy as np
        from PIL import Image

        if media_conditioning is not None:
            media_conditioning.validate_for_generation(config.num_frames)
            media_keyframes = LTX25KeyframeStack()
            for item in media_conditioning.items:
                if item.role == "image_keyframe":
                    media_keyframes = media_keyframes.append(
                        LTX25Keyframe(item.images, item.start_frame, item.strength)
                    )
            if keyframes is not None:
                for item in keyframes.keyframes:
                    media_keyframes = media_keyframes.append(item)
            keyframes = media_keyframes if media_keyframes.keyframes else None
        report = model.validate(
            config.pipeline_mode,
            require_spatial_upscaler=not config.ic_lora_single_stage,
        )
        config.validate(scale_factors=tuple(int(value) for value in report["video_scale_factors"]))
        validate_ltx25_dfr_prebaked_pair(config, report)
        released = _release_h3_stages()
        final = _safe_target(filename_prefix, config.seed)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(f".{final.stem}.partial{final.suffix}")
        metadata_path = final.with_suffix(".json")
        partial_metadata = final.with_name(f".{final.stem}.metadata.partial.json")
        image_path = None
        keyframe_paths = []
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
            image_inputs = []
            video_references = []
            audio_reference = None
            if media_conditioning is not None:
                video_references = [
                    {
                        "images": item.images,
                        "start_frame": item.start_frame,
                        "end_frame": (
                            config.num_frames - 1
                            if item.control_type == "ingredients_reference_sheet"
                            else item.end_frame
                        ),
                        "strength": item.strength,
                        "attention_strength": item.attention_strength,
                        "mask": item.mask,
                        "control_type": item.control_type,
                    }
                    for item in media_conditioning.items
                    if item.role == "video_reference"
                ]
                if video_references and not model.ic_loras:
                    raise ValueError(
                        "Connect the LTX 2.5 IC-LoRA Loader before generating with a "
                        "video_reference media item."
                    )
                if video_references:
                    ic_components = [
                        item
                        for item in report["components"]
                        if str(item.get("component", "")).startswith("ic_lora_")
                    ]
                    adapter_family = (
                        ic_components[0].get("adapter_family") if ic_components else None
                    )
                    control_types = {
                        str(item["control_type"]) for item in video_references
                    }
                    required_family = {
                        "ingredients_reference_sheet": "ingredients_reference_sheet",
                        "motion_track": "motion_track",
                        "canny_edges": "union_control",
                        "depth_map": "union_control",
                        "pose_skeleton": "union_control",
                    }
                    mismatches = {
                        control: family
                        for control, family in required_family.items()
                        if control in control_types and adapter_family != family
                    }
                    if mismatches:
                        expected = ", ".join(sorted(set(mismatches.values())))
                        raise ValueError(
                            "The selected IC-LoRA does not match the reference guide. "
                            f"Expected adapter family: {expected}; found: {adapter_family}."
                        )
                    if any(
                        item["control_type"] == "ingredients_reference_sheet"
                        for item in video_references
                    ) and not config.ic_lora_single_stage:
                        raise ValueError(
                            "Ingredients reference sheets require the IC-LoRA Pipeline Mode "
                            "node set to Official single stage."
                        )
                audio_items = [
                    item
                    for item in media_conditioning.items
                    if item.role == "audio_reference"
                ]
                if len(audio_items) > 1:
                    raise ValueError(
                        "LTX 2.5 currently accepts one audio-driven conditioning source."
                    )
                if audio_items:
                    audio_item = audio_items[0]
                    if audio_item.start_frame != 0 or audio_item.end_frame != 0:
                        raise ValueError(
                            "Audio-driven conditioning currently spans the complete output; "
                            "leave start_frame and end_frame at zero."
                        )
                    audio_reference = audio_item.audio
                if video_references and audio_reference is not None:
                    raise ValueError(
                        "Combined video-reference plus audio-reference is a LipDub topology. "
                        "No compatible LTX 2.5 LipDub IC-LoRA is currently validated; use "
                        "either video-reference or audio-driven conditioning in this node."
                    )
            if keyframes is not None:
                keyframes.validate(config.num_frames)
                if first_frame is not None and any(
                    item.frame_index == 0 for item in keyframes.keyframes
                ):
                    raise ValueError(
                        "Use either first_frame or a frame-zero LTX 2.5 keyframe, not both."
                    )
                for index, item in enumerate(
                    sorted(keyframes.keyframes, key=lambda value: value.frame_index)
                ):
                    frame = item.image[0]
                    detach = getattr(frame, "detach", None)
                    if detach is not None:
                        frame = detach()
                    cpu = getattr(frame, "cpu", None)
                    if cpu is not None:
                        frame = cpu()
                    frame = np.asarray(frame, dtype=np.float32)
                    keyframe_path = final.with_name(f".{final.stem}.keyframe-{index}.partial.png")
                    Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)).save(
                        keyframe_path
                    )
                    keyframe_paths.append(keyframe_path)
                    image_inputs.append(
                        {
                            "path": str(keyframe_path),
                            "frame_index": item.frame_index,
                            "strength": item.strength,
                        }
                    )
            info = RUNTIME.generate_to_file(
                model,
                config,
                prompt,
                partial,
                image_path=str(image_path) if image_path is not None else None,
                image_inputs=image_inputs,
                video_references=video_references,
                audio_reference=audio_reference,
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
            for keyframe_path in keyframe_paths:
                keyframe_path.unlink(missing_ok=True)


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
    "WeeToddLTX25LoRALoader": WeeToddLTX25LoRALoader,
    "WeeToddLTX25ICLoRALoader": WeeToddLTX25ICLoRALoader,
    "WeeToddLTX25GuidedModelLoader": WeeToddLTX25GuidedModelLoader,
    "WeeToddLTX25GenerationConfig": WeeToddLTX25GenerationConfig,
    "WeeToddLTX25QualityMode": WeeToddLTX25QualityMode,
    "WeeToddLTX25AutoDuration": WeeToddLTX25AutoDuration,
    "WeeToddLTX25GeneratedKeyframes": WeeToddLTX25GeneratedKeyframes,
    "WeeToddLTX25DiffVAEOptimization": WeeToddLTX25DiffVAEOptimization,
    "WeeToddLTX25DFRDetailing": WeeToddLTX25DFRDetailing,
    "WeeToddLTX25DFRTemporalRefinement": WeeToddLTX25DFRTemporalRefinement,
    "WeeToddLTX25Preflight": WeeToddLTX25Preflight,
    "WeeToddLTX25Keyframe": WeeToddLTX25Keyframe,
    "WeeToddLTX25MediaConditioning": WeeToddLTX25MediaConditioning,
    "WeeToddLTX25ICLoRAControlGuide": WeeToddLTX25ICLoRAControlGuide,
    "WeeToddLTX25ICLoRAPipelineMode": WeeToddLTX25ICLoRAPipelineMode,
    "WeeToddLTX25ReferenceSheetGuide": WeeToddLTX25ReferenceSheetGuide,
    "WeeToddLTX25Generate": WeeToddLTX25Generate,
    "WeeToddLTX25GenerateChained": WeeToddLTX25GenerateChained,
    "WeeToddLTX25VideoUpscale": WeeToddLTX25VideoUpscale,
    "WeeToddLTX25Unload": WeeToddLTX25Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddLTX25ComponentLoader": "WeeTodd LTX 2.5 Component Loader (MLX)",
    "WeeToddLTX25LoRALoader": "WeeTodd LTX 2.5 LoRA Loader (MLX)",
    "WeeToddLTX25ICLoRALoader": "WeeTodd LTX 2.5 IC-LoRA Loader (MLX)",
    "WeeToddLTX25GuidedModelLoader": "WeeTodd LTX 2.5 Guided Model Loader (MLX)",
    "WeeToddLTX25GenerationConfig": "WeeTodd LTX 2.5 Generation Config",
    "WeeToddLTX25QualityMode": "WeeTodd LTX 2.5 Quality Mode",
    "WeeToddLTX25AutoDuration": "WeeTodd LTX 2.5 Automatic Duration",
    "WeeToddLTX25GeneratedKeyframes": "WeeTodd LTX 2.5 Generated Keyframes",
    "WeeToddLTX25DiffVAEOptimization": "WeeTodd LTX 2.5 Diffusion VAE Optimization",
    "WeeToddLTX25DFRDetailing": "WeeTodd LTX 2.5 DFR Detail Refinement",
    "WeeToddLTX25DFRTemporalRefinement": "WeeTodd LTX 2.5 DFR Temporal Refinement",
    "WeeToddLTX25Preflight": "WeeTodd LTX 2.5 Preflight",
    "WeeToddLTX25Keyframe": "WeeTodd LTX 2.5 Timed Keyframe",
    "WeeToddLTX25MediaConditioning": "WeeTodd LTX 2.5 Media Conditioning",
    "WeeToddLTX25ICLoRAControlGuide": "WeeTodd LTX 2.5 IC-LoRA Control Guide",
    "WeeToddLTX25ICLoRAPipelineMode": "WeeTodd LTX 2.5 IC-LoRA Pipeline Mode",
    "WeeToddLTX25ReferenceSheetGuide": "WeeTodd LTX 2.5 Ingredients Reference Sheet",
    "WeeToddLTX25Generate": "WeeTodd LTX 2.5 Generate Video + Audio",
    "WeeToddLTX25GenerateChained": "WeeTodd LTX 2.5 Generate Chained Timeline",
    "WeeToddLTX25VideoUpscale": "WeeTodd LTX 2.5 Video Upscale / Refine",
    "WeeToddLTX25Unload": "WeeTodd LTX 2.5 Unload MLX Runtime",
}
