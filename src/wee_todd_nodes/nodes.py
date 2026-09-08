"""Classic ComfyUI node contracts backed by the MLX MiniMax H3 pipeline."""

import json
import math
import os
import platform
import struct
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
from .conditioning_inputs import (
    H3KeyframeConditioning,
    H3ReferenceInput,
    H3ReferenceStack,
    H3TimedKeyframe,
    H3TimedKeyframeStack,
)
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
from .direct_publishing import publish_latent_chain_direct, publish_latents_direct
from .preflight import (
    H3ComponentSetSpec,
    H3PreflightRequest,
    estimate_h3_token_budget,
    preflight_components,
)
from .preview import PREVIEW_BACKENDS, PREVIEW_GUARD_MODES, H3PreviewConfig
from .publishing import publish_synchronized_media
from .residency import prepare_low_memory_stage
from .runtime import RUNTIME, H3GenerationConfig, H3ModelSpec
from .sampling import (
    TRANSFORMER_RUNTIME,
    H3LearnedLatentUpscalerSpec,
    H3TransformerSpec,
)
from .timeline import H3ChainedTimeline, H3LatentChain
from .vdn import VDN_REPOSITORY_ID, VDN_STAGES, resolve_vdn_spec

_PORTABLE_H3_LORA_NAMES = (
    "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    "fasth3_dense_datafree_4step_rank64.safetensors",
)
_PORTABLE_H3_LATENT_UPSCALER_NAMES = (
    "minimax_h3_latent_upscaler_3d_bf16.safetensors",
)


def _lora_choices():
    try:
        import folder_paths

        discovered = folder_paths.get_filename_list("loras")
        return list(dict.fromkeys((*_PORTABLE_H3_LORA_NAMES, *discovered)))
    except ImportError:
        return list(_PORTABLE_H3_LORA_NAMES)


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


def _h3_latent_upscaler_choices():
    try:
        import folder_paths

        category = "latent_upscale_models"
        if category not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                category, str(Path(folder_paths.models_dir) / category)
            )
        discovered = []
        for name in folder_paths.get_filename_list(category):
            resolved = folder_paths.get_full_path(category, name)
            if resolved and _is_h3_latent_upscaler_checkpoint(Path(resolved)):
                discovered.append(name)
        return list(
            dict.fromkeys((*_PORTABLE_H3_LATENT_UPSCALER_NAMES, *discovered))
        )
    except ImportError:
        return list(_PORTABLE_H3_LATENT_UPSCALER_NAMES)


def _is_h3_latent_upscaler_checkpoint(path: Path) -> bool:
    """Identify the H3 24-channel 3D architecture from its SafeTensors header only."""
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size <= 4 * 1024 * 1024:
                return False
            header = json.loads(stream.read(header_size))
        conv = header.get("conv_in.weight") or header.get("upscaler.conv_in.weight")
        shape = conv.get("shape") if isinstance(conv, dict) else None
        return bool(
            isinstance(shape, list)
            and len(shape) == 5
            and shape[1] == 24
            and any(key.endswith("dwconv.weight") for key in header)
        )
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False


def _resolve_h3_latent_upscaler_path(name: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    try:
        import folder_paths

        category = "latent_upscale_models"
        if category not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                category, str(Path(folder_paths.models_dir) / category)
            )
        resolved = folder_paths.get_full_path(category, name)
        if resolved:
            return Path(resolved)
        return Path(folder_paths.models_dir) / category / name
    except ImportError:
        return candidate


def _output_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path.cwd() / "output"


def _load_h3_frame_image(image_name: str):
    """Load one annotated ComfyUI input path only when the frame node executes."""
    try:
        import folder_paths
        import nodes as comfy_nodes
    except ImportError as error:
        raise RuntimeError(
            "H3 Frames image loading requires a running ComfyUI installation."
        ) from error
    if not folder_paths.exists_annotated_filepath(image_name):
        raise ValueError(f"H3 Frames image does not exist in a ComfyUI media folder: {image_name}")
    image, _mask = comfy_nodes.LoadImage().load_image(image_name)
    return image


def _frames_from_manifest(frame_manifest: str, num_frames: int, image_loader=None):
    """Convert the one-based Frames editor manifest to the existing timed-keyframe contract."""
    if image_loader is None:
        image_loader = _load_h3_frame_image
    try:
        entries = json.loads(frame_manifest)
    except json.JSONDecodeError as error:
        raise ValueError(
            "H3 Frames data is not valid JSON; reopen the node and select frames."
        ) from error
    if not isinstance(entries, list) or not entries:
        raise ValueError("H3 Frames requires at least one selected image.")
    if len(entries) > 8:
        raise ValueError("H3 Frames supports at most eight images per generation.")

    resolved = []
    roles = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"H3 Frames entry {index} must be an object.")
        role = entry.get("role", "middle")
        if role not in {"first", "middle", "last"}:
            raise ValueError(f"H3 Frames entry {index} has an unknown role: {role!r}.")
        if role in {"first", "last"}:
            if role in roles:
                raise ValueError(f"H3 Frames can contain only one {role} image.")
            roles.add(role)
        image_name = entry.get("image")
        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"H3 Frames entry {index} does not have a selected image.")
        if role == "first":
            frame_number = 1
        elif role == "last":
            frame_number = num_frames
        else:
            frame_number = entry.get("frame")
            if isinstance(frame_number, bool) or not isinstance(frame_number, int):
                raise ValueError(f"H3 Frames middle entry {index} needs an integer frame number.")
            if not 2 <= frame_number < num_frames:
                raise ValueError(
                    f"H3 Frames middle entry {index} must be between frame 2 and {num_frames - 1}."
                )
        resolved.append((frame_number, image_name.strip()))

    frame_numbers = [item[0] for item in resolved]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("H3 Frames contains two images assigned to the same frame.")
    stack = H3TimedKeyframeStack()
    for frame_number, image_name in sorted(resolved):
        stack = stack.append(
            H3TimedKeyframe(
                image_loader(image_name),
                timestamp_seconds=(frame_number - 1) / 24.0,
            )
        )
    return stack


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
        raise ValueError("Media timing frame count does not match the decoded IMAGE frame count.")
    return timing


def _h3_preview_contact_sheet(frames, completed: int, total: int):
    """Build one compact sampler preview without retaining decoded video frames."""
    import numpy as np
    from PIL import Image, ImageDraw

    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.shape[0] < 1:
        raise ValueError("H3 TAE preview must contain RGB video frames.")
    uint8 = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    images = [Image.fromarray(frame, mode="RGB") for frame in uint8]
    columns = min(3, len(images))
    rows = math.ceil(len(images) / columns)
    label_height = 24
    width, height = images[0].size
    sheet = Image.new("RGB", (columns * width, rows * height + label_height), "black")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height + label_height))
    ImageDraw.Draw(sheet).text(
        (8, 6),
        f"H3 predicted clean latent — evaluation {completed}/{total}",
        fill="white",
    )
    return sheet


def _save_h3_preview_contact_sheet(image, completed: int, total: int) -> Path | None:
    """Publish a stable sampler-preview artifact beside normal ComfyUI outputs."""
    try:
        import folder_paths
    except ImportError:
        return None
    directory = Path(folder_paths.get_output_directory()) / "WeeTodd" / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"h3_live_preview_eval_{completed:02d}_of_{total:02d}.jpg"
    temporary = path.with_suffix(".tmp.jpg")
    image.save(temporary, format="JPEG", quality=92)
    temporary.replace(path)
    return path


_COMPONENT_MODEL_CATEGORIES = {
    "checkpoint": ("checkpoints", "diffusers", "diffusion_models"),
    "transformer": ("diffusion_models", "checkpoints", "diffusers"),
    "text_encoder": ("text_encoders", "checkpoints"),
    "processor": ("text_encoders", "checkpoints"),
    "tokenizer": ("text_encoders", "checkpoints"),
    "video_vae": ("vae", "checkpoints"),
    "audio_vae": ("vae", "checkpoints"),
    "preview_tae": ("vae_approx", "vae"),
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


def _resolve_h3_coreml_model(value: str) -> str:
    """Honor an existing requested model, then try its compiled/package sibling."""
    resolved = _resolve_component_root(value, "preview_tae")
    if Path(resolved).exists():
        return resolved
    suffix = Path(value).suffix
    alternate = {".mlpackage": ".mlmodelc", ".mlmodelc": ".mlpackage"}.get(suffix)
    if alternate:
        sibling = _resolve_component_root(str(Path(value).with_suffix(alternate)), "preview_tae")
        if Path(sibling).exists():
            return sibling
    return resolved


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
    "FastH3 Preview v1 — Native VSA student — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "distilled_checkpoint",
        "required_transformer": "weetodd-fasth3-vsa-datafree-q8-paged",
        "required_attention_profile": "fasth3_vsa_90_metal",
        "source": {
            "model_id": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
            "revision": "b65818d41939b5085451074fe8ca8b799f8d4921",
            "attention": "VSA-H3 64-token tiles at 90% sparsity",
            "training": "data-free DMD2 with trained compression gates",
            "tasks": ["t2va"],
        },
    },
    "FastH3 Preview v1 — Native dense student — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "distilled_checkpoint",
        "required_transformer": "weetodd-fasth3-dense-q8-paged",
        "source": {
            "model_id": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",
            "revision": "f624f08c6c279ab43534c003e556fc5b295b6558",
            "attention": "dense",
            "training": "data-free DMD2",
            "tasks": ["t2va"],
        },
    },
    "FastH3 Preview v1 — Dense Data-Free — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "fasth3_dense_datafree_4step_rank64.safetensors",
        "source": {
            "model_id": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
            "revision": "bcf40ca6f457ed66f8badf13514943e390205fca",
            "variant": "dense-datafree",
            "sha256": "4ce198c83132251b7fd0de2503823aa49c53983f068318f66cb19eaefb7fcc12",
            "attention": "dense",
            "training": "data-free DMD2",
        },
    },
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
    "Turbo — drbaph v4 step-600 — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
    },
    "Turbo — drbaph v4 step-600 — 384p low-memory — 5 points / 4 evaluations": {
        "steps": 5,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "measurement": {
            "task": "t2va",
            "seed": 0,
            "canvas": [640, 384],
            "duration_seconds": 5.0,
            "memory_mode": "low_memory_bf16",
            "transformer_evaluations": 4,
            "cache": "disabled",
            "complete_workflow_seconds": 150.965,
            "complete_process_peak_bytes": 14951286752,
            "seconds_per_evaluation": 28.621493291517254,
            "output_frames": 124,
            "audio_sample_rate": 32000,
        },
    },
    "Staged Turbo — drbaph v4 step-600 — 2 base + 4 Turbo evaluations": {
        "steps": 7,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "start_after_evaluations": 2,
        "measurement": {
            "task": "fl2va",
            "canvas": [1344, 768],
            "duration_seconds": 5.0,
            "transformer_evaluations": 6,
            "base_evaluations": 2,
            "lora_evaluations": 4,
            "sampling_seconds": 1275.5262,
            "complete_workflow_seconds": 1415.689,
            "av_drift_seconds": 0.0083333,
        },
    },
    "Staged Turbo — drbaph v4 step-600 — 384p low-memory — 2 base + 4 Turbo evaluations": {
        "steps": 7,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "start_after_evaluations": 2,
        "measurement": {
            "task": "t2va",
            "seed": 0,
            "canvas": [640, 384],
            "duration_seconds": 5.0,
            "memory_mode": "low_memory_bf16",
            "transformer_evaluations": 6,
            "base_evaluations": 2,
            "lora_evaluations": 4,
            "complete_workflow_seconds": 203.493,
            "complete_process_peak_bytes": 14908409896,
            "output_frames": 124,
            "audio_sample_rate": 32000,
        },
    },
    "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline": {
        "steps": 7,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "start_after_evaluations": 2,
        "measurement": {
            "task": "t2va",
            "seed": 20260811,
            "canvas": [1344, 768],
            "duration_seconds": 15.083333333333334,
            "transformer_evaluations": 6,
            "base_evaluations": 2,
            "lora_evaluations": 4,
            "sampling_seconds": 7840.3384475,
            "complete_workflow_seconds": 8207.699172,
            "mlx_peak_bytes": 30783349650,
            "av_drift_seconds": 0.0083333,
        },
    },
    "Chained staged Turbo — drbaph v4 step-600 — 4 windows / 22-frame context": {
        "steps": 7,
        "policy": "turbo",
        "lora": "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors",
        "start_after_evaluations": 2,
        "measurement": {
            "task": "t2va_continuation",
            "canvas": [1344, 768],
            "windows": 4,
            "window_duration_seconds": 4.0,
            "final_duration_seconds": 15.0,
            "context_frames": 22,
            "transformer_evaluations_per_window": 6,
            "base_evaluations": 2,
            "lora_evaluations": 4,
            "base_evaluations_per_window": 2,
            "lora_evaluations_per_window": 4,
            "sampling_seconds": 4636.656131541,
            "publication_seconds": 416.626592958,
            "complete_workflow_seconds": 5089.0,
            "mlx_peak_bytes": 14453992534,
            "av_drift_seconds": 0.0,
            "join_policy": "motion-matched overlap + 4-frame cosine blend + 50-ms audio crossfade",
            "perceptual_status": "promising; review motion immediately after each join",
        },
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


class WeeToddH3PreviewOverride:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "tae_model": (
                    "STRING",
                    {
                        "default": "taeh3.safetensors",
                        "tooltip": (
                            "A 24-channel MiniMax H3 tiny preview decoder. Relative names are "
                            "searched across ComfyUI's shared vae_approx and VAE model roots."
                        ),
                    },
                ),
                "preview_backend": (
                    list(PREVIEW_BACKENDS),
                    {
                        "default": "auto",
                        "tooltip": (
                            "Auto prefers a compiled Core ML model on the Apple Neural Engine "
                            "and falls back to MLX. Neural engine requires the Core ML model."
                        ),
                    },
                ),
                "coreml_model": (
                    "STRING",
                    {
                        "default": "taeh3_coreml_256.mlpackage",
                        "tooltip": (
                            "Optional compiled H3 preview model searched across ComfyUI's "
                            "vae_approx and VAE roots."
                        ),
                    },
                ),
                "preview_every": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 20,
                        "tooltip": "Decode a live contact-sheet preview every N evaluations.",
                    },
                ),
                "preview_frames": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 12,
                        "tooltip": "Frames shown across the sampler preview contact sheet.",
                    },
                ),
                "max_preview_edge": (
                    "INT",
                    {
                        "default": 256,
                        "min": 64,
                        "max": 1024,
                        "step": 32,
                        "tooltip": (
                            "Maximum decoded preview edge. This does not change generation size."
                        ),
                    },
                ),
                "safety_guard": (
                    list(PREVIEW_GUARD_MODES),
                    {
                        "default": "conservative collapse guard",
                        "tooltip": (
                            "The conservative guard requires repeated featureless previews after "
                            "the schedule midpoint. Non-finite video latents always stop sampling."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_COMPONENTS",)
    RETURN_NAMES = ("components",)
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Attach a true-color TAE preview and optional collapse guard to H3 sampling. Core ML "
        "can keep preview decoding on the Apple Neural Engine; MLX remains the fallback. Place "
        "this node between the component loader and sampler."
    )

    def apply(
        self,
        components,
        tae_model,
        preview_backend,
        coreml_model,
        preview_every,
        preview_frames,
        max_preview_edge,
        safety_guard,
    ):
        config = H3PreviewConfig(
            tae_path=_resolve_component_root(tae_model, "preview_tae"),
            backend=preview_backend,
            coreml_model_path=(
                _resolve_h3_coreml_model(coreml_model) if coreml_model.strip() else None
            ),
            every_n_evaluations=int(preview_every),
            preview_frames=int(preview_frames),
            max_edge=int(max_preview_edge),
            guard_mode=safety_guard,
        )
        config.validate()
        return (replace(components, preview_override=config),)


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
        "Vision-capable paged Qwen is supported; reference workspace is not included. "
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


class WeeToddH3TokenBudget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_H3_CONFIG",),
                "prompt_tokens": ("INT", {"default": 512, "min": 1, "max": 32768}),
                "condition_video_rows": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "advanced": True},
                ),
                "condition_audio_rows": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "advanced": True},
                ),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("packed_rows", "token_report")
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Estimate H3 text, condition, audio, and video rows plus dense-attention scale without "
        "loading a checkpoint. Add encoded reference rows for Ref2VA planning."
    )

    def estimate(self, config, prompt_tokens, condition_video_rows, condition_audio_rows):
        report = estimate_h3_token_budget(
            H3PreflightRequest(
                duration_seconds=config.duration_seconds,
                steps=config.steps,
                width=config.width,
                height=config.height,
                prompt_tokens=prompt_tokens,
            ),
            condition_video_rows=condition_video_rows,
            condition_audio_rows=condition_audio_rows,
        )
        return int(report["packed_rows"]), json.dumps(report, indent=2, sort_keys=True)


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


class WeeToddH3ChainedTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "window_duration_seconds": (
                    "FLOAT",
                    {"default": 4.0, "min": 2.5, "max": 15.0, "step": 0.1},
                ),
                "window_count": ("INT", {"default": 4, "min": 2, "max": 16}),
                "context_frames": (
                    [str(value) for value in SUPPORTED_CONTEXT_FRAMES],
                    {"default": "22"},
                ),
                "target_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 0.0,
                        "max": 120.0,
                        "step": 1.0 / 24.0,
                        "tooltip": "Use 0 to publish the complete assembled timeline.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_TIMELINE", "STRING")
    RETURN_NAMES = ("timeline", "timeline_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/continuation"
    DESCRIPTION = (
        "Map global timestamps onto equal-length H3 windows and define exact overlap trimming."
    )

    def configure(
        self,
        window_duration_seconds,
        window_count,
        context_frames,
        target_duration_seconds,
    ):
        from minimax_h3_mlx.packing import align_num_frames

        window_frames = align_num_frames(round(window_duration_seconds * 24))
        target_frames = (
            None if target_duration_seconds == 0 else round(target_duration_seconds * 24)
        )
        timeline = H3ChainedTimeline(
            window_frames=window_frames,
            window_count=window_count,
            context_frames=int(context_frames),
            target_frames=target_frames,
        )
        timeline.validate()
        return timeline, json.dumps(timeline.metadata(), indent=2, sort_keys=True)


class WeeToddH3TimedKeyframe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "timestamp_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 120.0, "step": 1.0 / 24.0},
                ),
                "timestamp_scope": (["local window", "global timeline"],),
                "window_index": ("INT", {"default": 1, "min": 1, "max": 16}),
            },
            "optional": {
                "timeline": ("WEETODD_H3_TIMELINE",),
                "previous_keyframes": ("WEETODD_H3_TIMED_KEYFRAMES",),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_TIMED_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Append an FL2VA image at an exact 24 fps local timestamp, or map a global chained "
        "timeline timestamp into one window."
    )

    def append(
        self,
        image,
        timestamp_seconds,
        timestamp_scope,
        window_index,
        timeline=None,
        previous_keyframes=None,
    ):
        global_seconds = None
        local_seconds = timestamp_seconds
        if timestamp_scope == "global timeline":
            if timeline is None:
                raise ValueError("Global timed keyframes require an H3 Chained Timeline.")
            global_seconds = timestamp_seconds
            local_seconds = timeline.local_timestamp(window_index, timestamp_seconds)
        stack = (previous_keyframes or H3TimedKeyframeStack()).append(
            H3TimedKeyframe(image, local_seconds, global_seconds)
        )
        return stack, json.dumps(stack.metadata(), indent=2, sort_keys=True)


class WeeToddH3Frames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_H3_CONFIG",),
                "frame_manifest": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": True,
                        "tooltip": "Managed by the visual frame strip. One-based frame numbers.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_TIMED_KEYFRAMES", "STRING")
    RETURN_NAMES = ("keyframes", "keyframe_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Select first, last, and up to six numbered middle images in one visual frame strip. "
        "Frame numbers are one-based; the last frame follows the connected generation duration."
    )

    def configure(self, config, frame_manifest):
        config.validate()
        from minimax_h3_mlx.packing import align_num_frames

        num_frames = align_num_frames(round(config.duration_seconds * 24))
        stack = _frames_from_manifest(frame_manifest, num_frames)
        info = stack.metadata(num_frames)
        info.update(
            {
                "editor_frame_numbering": "one_based",
                "first_frame": 1,
                "last_frame": num_frames,
            }
        )
        return stack, json.dumps(info, indent=2, sort_keys=True)


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
                "video_size": (
                    [
                        "match output (recommended)",
                        "native H3 reference canvas (high detail / slow)",
                    ],
                    {
                        "default": "match output (recommended)",
                        "tooltip": (
                            "Match output preserves aspect and downsizes only to the generated "
                            "video's pixel area. Native H3 canvas may improve fine reference "
                            "detail, but persistent reference tokens make every forward pass "
                            "slower and increase memory use."
                        ),
                    },
                ),
            },
            "optional": {
                "temporal_density": (
                    [
                        "all frames (recommended)",
                        "automatic (conservative, experimental)",
                        "uniform 50% (experimental)",
                        "uniform 25% (experimental)",
                    ],
                    {
                        "default": "all frames (recommended)",
                        "tooltip": (
                            "Automatic mode scans adjacent-frame activity and conservatively "
                            "selects full, half, or quarter persistent video-VAE density. Qwen "
                            "still inspects the full clip at 2 fps, and rotary timestamps retain "
                            "the original duration. Reduced density can change motion or identity "
                            "fidelity."
                        ),
                    },
                ),
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
        "Supply the source frame rate explicitly. The recommended default matches the output "
        "pixel area; native reference resolution is available but can be dramatically slower."
    )

    def append(
        self,
        video_frames,
        fps,
        video_size="match output (recommended)",
        temporal_density="all frames (recommended)",
        soundtrack=None,
        previous_references=None,
    ):
        return _append_reference(
            previous_references,
            H3ReferenceInput(
                "video",
                video_frames,
                fps=fps,
                soundtrack=soundtrack,
                video_size_mode=video_size,
                temporal_density=temporal_density,
            ),
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


class WeeToddH3TimelineVisualGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_or_clip": ("IMAGE",),
                "target_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": -100000,
                        "max": 100000,
                        "tooltip": "Zero-based target frame; negative values count from the end.",
                    },
                ),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0}),
            },
            "optional": {
                "soundtrack": ("AUDIO",),
                "previous_references": ("WEETODD_H3_REFERENCES",),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "guide_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Place one image or an aligned H3 clip on the Ref2VA target timeline. A clip must contain "
        "5, 22, 39, ... frames after 24 fps alignment; optional audio starts at the same frame."
    )

    def append(self, image_or_clip, target_frame, fps, soundtrack=None, previous_references=None):
        frame_count = int(image_or_clip.shape[0])
        kind = "image" if frame_count == 1 else "video"
        if kind == "video" and frame_count < 5:
            raise ValueError("An H3 timeline clip must contain at least five source frames.")
        return _append_reference(
            previous_references,
            H3ReferenceInput(
                kind,
                image_or_clip,
                fps=(fps if kind == "video" else None),
                soundtrack=soundtrack,
                target_frame=target_frame,
            ),
        )


class WeeToddH3TimelineAudioGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_frame": (
                    "INT",
                    {
                        "default": 0,
                        "min": -100000,
                        "max": 100000,
                        "tooltip": "Zero-based target frame; negative values count from the end.",
                    },
                ),
            },
            "optional": {"previous_references": ("WEETODD_H3_REFERENCES",)},
        }

    RETURN_TYPES = ("WEETODD_H3_REFERENCES", "STRING")
    RETURN_NAMES = ("references", "guide_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = "Place an audio guide at an exact Ref2VA target frame."

    def append(self, audio, target_frame, previous_references=None):
        return _append_reference(
            previous_references,
            H3ReferenceInput("audio", audio, target_frame=target_frame),
        )


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


class WeeToddH3TimedKeyframeEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "keyframes": ("WEETODD_H3_TIMED_KEYFRAMES",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Encode up to eight sparse FL2VA images at exact 24 fps timestamps, unloading "
        "Qwen3-VL before the video VAE stage."
    )

    def encode(self, components, config, keyframes, prompt):
        if components.task != "fl2va":
            raise ValueError("H3 timed keyframe encoding requires an FL2VA component set.")
        config.validate()
        from minimax_h3_mlx.packing import align_num_frames, prepare_keyframe_image

        num_frames = align_num_frames(round(config.duration_seconds * 24))
        anchors, source_images = keyframes.resolve(num_frames)
        images = [
            prepare_keyframe_image(
                image,
                config.height,
                config.width,
                stretch=anchor == 0,
            )
            for anchor, image in zip(anchors, source_images, strict=True)
        ]
        check_interrupted = None
        try:
            import comfy.model_management

            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass
        if check_interrupted is not None:
            check_interrupted()

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
            keyframe_anchors=anchors,
        )
        info = {
            "task": "fl2va",
            "prompt": prompt,
            "token_count": conditioning.token_count,
            "anchors": list(anchors),
            "anchor_seconds": [frame / 24 for frame in anchors],
            "rope_times": [frame * (5.0 / 3.0) for frame in anchors],
            "condition_video_rows": int(rows.shape[0]),
            "vision_loaded": True,
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
        "Resident and vision-capable paged Qwen are supported. "
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
                    **(
                        {
                            "temporal_density_policy": getattr(
                                reference, "temporal_density_requested", "full"
                            ),
                            "temporal_density_resolved": getattr(
                                reference, "temporal_density_resolved", 1.0
                            ),
                            "temporal_activity_mean": getattr(
                                reference, "temporal_activity_mean", None
                            ),
                            "temporal_activity_p95": getattr(
                                reference, "temporal_activity_p95", None
                            ),
                            "temporal_density_reason": getattr(
                                reference, "temporal_density_reason", None
                            ),
                        }
                        if reference.kind == "video"
                        else {}
                    ),
                }
                for metadata, reference in zip(
                    references.metadata()["references"], prepared, strict=True
                )
            ],
            "staged_releases": staged,
            "encoder_resident": TEXT_ENCODER_RUNTIME.loaded,
            "video_vae_resident": VIDEO_VAE_RUNTIME.loaded,
            "audio_vae_resident": AUDIO_VAE_RUNTIME.loaded,
            "reference_feature_reuse": (
                "encoded_once_and_reused_for_every_transformer_evaluation"
            ),
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
            "optional": {
                "config": ("WEETODD_H3_CONFIG",),
                "persistent_cache": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse text-only features; bounded 1 GiB local cache.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/conditioning"
    DESCRIPTION = (
        "Encode a text-only H3 prompt with Qwen3-VL. The vision tower stays unloaded. "
        "A bounded persistent feature cache can skip repeat encodes without keeping weights loaded."
    )

    def encode(self, components, prompt, unload_after_encode, config=None, persistent_cache=True):
        from .conditioning_cache import default_cache_directory

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
            task=components.task,
            unload_after=unload_after_encode or memory_mode == "low_memory_bf16",
            prepare_stage=prepare_stage,
            **({"cache_directory": default_cache_directory()} if persistent_cache else {}),
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
            "conditioning_cache": getattr(conditioning, "cache_report", None),
            "phase_memory": getattr(conditioning, "phase_memory", None),
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
                "context_frames": (
                    [str(value) for value in SUPPORTED_CONTEXT_FRAMES],
                    {
                        "default": "22",
                        "tooltip": (
                            "22 frames is the quality-first default. Shorter context is faster "
                            "but can weaken motion and identity continuity at the join."
                        ),
                    },
                ),
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


class WeeToddH3ChainAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("WEETODD_H3_TIMELINE",),
                "latents": ("WEETODD_H3_LATENTS",),
            },
            "optional": {"previous_chain": ("WEETODD_H3_LATENT_CHAIN",)},
        }

    RETURN_TYPES = ("WEETODD_H3_LATENT_CHAIN", "STRING")
    RETURN_NAMES = ("chain", "chain_info")
    FUNCTION = "append"
    CATEGORY = "WeeTodd/H3/continuation"
    DESCRIPTION = "Append one synchronized latent window to a validated H3 chained timeline."

    def append(self, timeline, latents, previous_chain=None):
        chain = previous_chain or H3LatentChain(timeline)
        if chain.timeline != timeline:
            raise ValueError("The previous latent chain uses a different H3 timeline.")
        chain = chain.append(latents)
        info = {
            **timeline.metadata(),
            "windows_present": len(chain.windows),
            "windows_complete": len(chain.windows) == timeline.window_count,
            "transformer_evaluations": [window.transformer_evaluations for window in chain.windows],
        }
        return chain, json.dumps(info, indent=2, sort_keys=True)


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
                "sol_attention": ("WEETODD_H3_SOL_ATTENTION",),
                "production_profile_info": ("STRING", {"forceInput": True}),
                "fastvideo": ("WEETODD_H3_FASTVIDEO",),
                "vdn": ("WEETODD_H3_VDN",),
                "continuation": ("WEETODD_H3_CONTINUATION",),
                "loras": ("WEETODD_H3_LORAS",),
                "fun_control": ("WEETODD_H3_FUN_CONTROL",),
                "block_residency": (
                    ["checkpoint_default", "resident"],
                    {
                        "default": "checkpoint_default",
                        "advanced": True,
                        "tooltip": "Resident retains all transformer blocks during sampling. "
                        "Requires ample memory; unload_after_sample controls release.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_LATENTS", "STRING")
    RETURN_NAMES = ("latents", "sampling_info")
    FUNCTION = "sample"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Sample synchronized MiniMax H3 video and audio latents with MLX. "
        "This node does not load or run either VAE. Optional resident block loading avoids "
        "repeated paging; staged unloading remains the default."
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
        sol_attention=None,
        production_profile_info=None,
        fastvideo=None,
        vdn=None,
        continuation=None,
        loras=None,
        fun_control=None,
        block_residency="checkpoint_default",
    ):
        if sum(value is not None for value in (easycache, blockcache, trajectory_forecast)) > 1:
            raise ValueError("Connect only one of EasyCache, BlockCache, or Trajectory Forecast.")
        production_profile = WeeToddH3FastH3ProductionProfile.validate_sampling_inputs(
            production_profile_info,
            components,
            config,
            sol_attention,
            fastvideo,
            modifiers=(
                easycache,
                blockcache,
                trajectory_forecast,
                vdn,
                continuation,
                fun_control,
            ),
            loras=loras,
        )
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

        preview_config = getattr(components, "preview_override", None)

        def on_preview(update, completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None and update.frames is not None:
                image = _h3_preview_contact_sheet(update.frames, completed, total)
                _save_h3_preview_contact_sheet(image, completed, total)
                progress.update_absolute(
                    completed,
                    total,
                    ("JPEG", image, preview_config.max_edge),
                )

        latents = TRANSFORMER_RUNTIME.sample(
            H3TransformerSpec.from_components(components),
            conditioning,
            config,
            unload_after=unload_after_sample,
            block_residency=block_residency,
            step_callback=on_step,
            easycache=easycache,
            blockcache=blockcache,
            trajectory_forecast=trajectory_forecast,
            sol_attention=sol_attention,
            fastvideo=fastvideo,
            vdn=vdn,
            continuation=continuation,
            loras=loras,
            preview_config=preview_config,
            preview_callback=on_preview if preview_config is not None else None,
            prepare_stage=prepare_stage,
            fun_control_spec=(fun_control.spec if fun_control is not None else None),
            fun_control_latent=(fun_control.latent if fun_control is not None else None),
        )
        try:
            WeeToddH3FastH3ProductionProfile.validate_execution(production_profile, latents)
        except BaseException:
            TRANSFORMER_RUNTIME.unload()
            raise
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
            "fun_control": getattr(latents, "fun_control_report", None),
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
            "easycache_reuse_strategy": getattr(latents, "easycache_reuse_strategy", None),
            "easycache_cache_bytes": getattr(latents, "easycache_cache_bytes", 0),
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
            "sol_attention": getattr(latents, "sol_attention_report", None)
            or (asdict(sol_attention) if sol_attention is not None else None),
            "fastvideo": getattr(latents, "fast_h3_approximation_report", None)
            or (asdict(fastvideo) if fastvideo is not None else None),
            "vdn": getattr(latents, "vdn_report", None),
            "production_profile": production_profile,
            "loras": loras.metadata() if loras is not None else [],
            "lora_report": list(getattr(latents, "lora_report", ())),
            "seconds_per_evaluation": latents.seconds_per_evaluation,
            "total_seconds": latents.total_seconds,
            "transformer_resident": TRANSFORMER_RUNTIME.loaded,
            "block_residency": getattr(latents, "block_residency_report", None),
            "memory_mode": config.memory_mode,
            "sampling_method": config.sampling_method,
            "attention_query_chunk_size": config.attention_query_chunk_size,
            "compute_dtype": "bfloat16",
            "projection_backend": getattr(latents, "projection_backend_report", None),
            "phase_memory": getattr(latents, "phase_memory", None),
            "conditioning_cache": getattr(latents, "conditioning_cache_report", None),
            "projection_backend_runtime": getattr(latents, "projection_backend_runtime", None),
            "paged_weights": {
                "transformer": getattr(latents, "paging_report", None),
                "text_encoder": getattr(latents, "text_encoder_paging_report", None),
            },
            "prepared_state": getattr(latents, "prepared_state_report", None),
            "preview_policy": (
                {
                    "model": Path(preview_config.tae_path).name,
                    "every_n_evaluations": preview_config.every_n_evaluations,
                    "preview_frames": preview_config.preview_frames,
                    "max_edge": preview_config.max_edge,
                    "guard_mode": preview_config.guard_mode,
                    "checkpoints": list(latents.preview_report),
                }
                if preview_config is not None
                else None
            ),
            "staged_releases": list(staged_releases),
        }
        return latents, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3LearnedLatentUpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    _h3_latent_upscaler_choices(),
                    {
                        "default": _PORTABLE_H3_LATENT_UPSCALER_NAMES[0],
                        "tooltip": (
                            "Place the BF16 SafeTensors checkpoint in ComfyUI/models/"
                            "latent_upscale_models. Weights load only during Hi-Res Fix and "
                            "unload before the H3 transformer refinement stage."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_LATENT_UPSCALER",)
    RETURN_NAMES = ("latent_upscaler",)
    FUNCTION = "select"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Select an MLX-native learned 3D latent upscaler for H3 Hi-Res Fix. "
        "The checkpoint is validated now and loaded only when the graph executes."
    )

    def select(self, model_name):
        spec = H3LearnedLatentUpscalerSpec(
            checkpoint=str(_resolve_h3_latent_upscaler_path(model_name))
        )
        spec.validate()
        return (spec,)


class WeeToddH3LatentHiresFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "conditioning": ("WEETODD_H3_CONDITIONING",),
                "source_latents": ("WEETODD_H3_LATENTS",),
                "scale": (
                    [
                        "1.5x — balanced",
                        "2.0x — experimental",
                        "maximum canvas — 1920×1088",
                    ],
                    {"default": "1.5x — balanced"},
                ),
                "refinement_schedule_points": (
                    "INT",
                    {
                        "default": 5,
                        "min": 2,
                        "max": 20,
                        "tooltip": "Requested H3 schedule points for the second visual pass.",
                    },
                ),
                "refinement_strength": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Fraction of the refinement schedule used to noise and reconstruct "
                            "the enlarged video latent. Higher values can change composition."
                        ),
                    },
                ),
                "unload_after_refine": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "learned_latent_upscaler": ("WEETODD_H3_LATENT_UPSCALER",),
                "loras": ("WEETODD_H3_LORAS",),
                "trajectory_forecast": ("WEETODD_H3_TRAJECTORY_FORECAST",),
                "latent_resize_method": (
                    ["bilinear", "nearest exact", "bicubic", "lanczos-3"],
                    {
                        "default": "bilinear",
                        "tooltip": (
                            "Select the MLX-native spatial interpolation used to enlarge the "
                            "video latent before refinement. Audio is not resized."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_LATENTS", "WEETODD_H3_CONFIG", "STRING")
    RETURN_NAMES = ("refined_latents", "refined_config", "refinement_info")
    FUNCTION = "refine"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Enlarge an H3 video latent and run a second H3 visual refinement pass. "
        "The original synchronized audio latent is returned unchanged."
    )

    def refine(
        self,
        components,
        conditioning,
        source_latents,
        scale,
        refinement_schedule_points,
        refinement_strength,
        unload_after_refine,
        loras=None,
        trajectory_forecast=None,
        latent_resize_method="bilinear",
        learned_latent_upscaler=None,
    ):
        from minimax_h3_mlx.hires_fix import (
            resolve_hires_canvas,
            resolve_hires_maximum_canvas,
        )

        if scale.startswith("maximum canvas"):
            scale_value = None
            scale_label = "maximum canvas"
            target_width, target_height = resolve_hires_maximum_canvas(
                source_latents.width,
                source_latents.height,
            )
        else:
            scale_value = 1.5 if scale.startswith("1.5") else 2.0
            scale_label = f"{scale_value:g}x"
            target_width, target_height = resolve_hires_canvas(
                source_latents.width,
                source_latents.height,
                scale_value,
            )
        config = replace(
            source_latents.generation_config,
            width=target_width,
            height=target_height,
            steps=int(refinement_schedule_points),
            resolution_mode="custom",
            resolution_tier=f"H3 Hi Res Fix {scale_label}",
            aspect_ratio="custom",
        )
        config.validate()

        progress = None
        check_interrupted = None
        active_steps = max(
            1,
            int(math.ceil((config.steps - 1) * float(refinement_strength))),
        )
        try:
            import comfy.model_management
            import comfy.utils

            upscaler_steps = 39 if learned_latent_upscaler is not None else 0
            progress = comfy.utils.ProgressBar(active_steps + upscaler_steps)
            check_interrupted = comfy.model_management.throw_exception_if_processing_interrupted
        except ImportError:
            pass

        def on_step(completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None:
                offset = 39 if learned_latent_upscaler is not None else 0
                progress.update_absolute(offset + completed, offset + total)

        def on_upscaler_step(completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None:
                progress.update_absolute(completed, total + active_steps)

        preview_config = getattr(components, "preview_override", None)

        def on_preview(update, completed, total):
            if check_interrupted is not None:
                check_interrupted()
            if progress is not None and update.frames is not None:
                image = _h3_preview_contact_sheet(update.frames, completed, total)
                _save_h3_preview_contact_sheet(image, completed, total)
                progress.update_absolute(
                    (39 if learned_latent_upscaler is not None else 0) + completed,
                    (39 if learned_latent_upscaler is not None else 0) + total,
                    ("JPEG", image, preview_config.max_edge),
                )

        refined = TRANSFORMER_RUNTIME.sample(
            H3TransformerSpec.from_components(components),
            conditioning,
            config,
            unload_after=unload_after_refine,
            step_callback=on_step,
            refinement_source=source_latents,
            refinement_strength=float(refinement_strength),
            refinement_resize_method=str(latent_resize_method),
            refinement_learned_upscaler=learned_latent_upscaler,
            refinement_upscaler_callback=(
                on_upscaler_step if learned_latent_upscaler is not None else None
            ),
            loras=loras,
            trajectory_forecast=trajectory_forecast,
            preview_config=preview_config,
            preview_callback=on_preview if preview_config is not None else None,
        )
        info = {
            "mode": "h3_native_latent_hires_fix",
            "source_canvas": [source_latents.width, source_latents.height],
            "target_canvas": [target_width, target_height],
            "requested_scale": scale_value,
            "scale_mode": scale_label,
            "resolved_scale": [
                target_width / source_latents.width,
                target_height / source_latents.height,
            ],
            "refinement_schedule_points": config.steps,
            "refinement_strength": refinement_strength,
            "latent_resize_method": latent_resize_method,
            "latent_upscale_backend": (
                "learned 3D MLX" if learned_latent_upscaler is not None else "interpolation"
            ),
            "learned_upscaler": getattr(refined, "refinement_upscaler_report", None),
            "condition_rows_resized": getattr(
                refined, "refinement_condition_rows_resized", False
            ),
            "transformer_evaluations": refined.transformer_evaluations,
            "trajectory_forecasts": refined.trajectory_forecasts,
            "trajectory_fallbacks": refined.trajectory_fallbacks,
            "trajectory_offline_replay": refined.trajectory_offline_replay,
            "trajectory_replay_steps": refined.trajectory_replay_steps,
            "trajectory_replay_anchor_steps": refined.trajectory_replay_anchor_steps,
            "trajectory_replay_smoothed_steps": refined.trajectory_replay_smoothed_steps,
            "trajectory_replay_seconds": refined.trajectory_replay_seconds,
            "trajectory_replay_fallback_reason": refined.trajectory_replay_fallback_reason,
            "trajectory_forecast": (
                asdict(trajectory_forecast) if trajectory_forecast is not None else None
            ),
            "audio_preserved": refined.refinement_audio_preserved,
            "preview_policy": (
                {
                    "model": Path(preview_config.tae_path).name,
                    "guard_mode": preview_config.guard_mode,
                    "checkpoints": list(refined.preview_report),
                }
                if preview_config is not None
                else None
            ),
            "audio_identity": "original H3 audio latent; second-pass audio discarded",
            "transformer_resident": TRANSFORMER_RUNTIME.loaded,
            "loras": loras.metadata() if loras is not None else [],
        }
        return refined, config, json.dumps(info, indent=2, sort_keys=True)


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
                "profile": (
                    ["auto", "standard", "turbo"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Auto uses checkpoint metadata and otherwise defaults to standard. "
                            "Select turbo explicitly when a downloaded adapter does not declare "
                            "its distillation schedule. Filenames never select adapter math."
                        ),
                    },
                ),
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
            "optional": {
                "start_after_evaluations": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": (
                            "Keep this LoRA disabled for the first N transformer evaluations, "
                            "then activate it for the rest of the same noise schedule. Use 2 for "
                            "the staged two-base-plus-four-Turbo recipe."
                        ),
                    },
                ),
                "previous_loras": ("WEETODD_H3_LORAS",),
            },
        }

    RETURN_TYPES = ("WEETODD_H3_LORAS", "STRING")
    RETURN_NAMES = ("loras", "lora_info")
    FUNCTION = "load"
    CATEGORY = "WeeTodd/H3/loaders"
    DESCRIPTION = (
        "Build a lazy, ordered MiniMax H3 LoRA stack. Validate safetensors headers now and load "
        "adapter tensors only when the H3 transformer executes. Reject malformed A/B pairs "
        "and unsupported tensor fields before loading weights."
    )

    def load(
        self,
        lora_name,
        strength,
        profile,
        adaln_input_grid="",
        qkv_layout="auto",
        start_after_evaluations=0,
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
            start_after_evaluations=start_after_evaluations,
        )
        stack = (previous_loras or H3LoRAStack()).append(spec)
        info = {
            "file": path.name,
            "strength": strength,
            "profile": spec.resolved_profile,
            "profile_classification_basis": spec.profile_classification_basis,
            "qkv_layout": spec.resolved_qkv_layout,
            "structural_descriptor": spec.structural_descriptor,
            "tensor_bytes": spec.tensor_bytes,
            "adaln_input_grid": grid.name if grid is not None else None,
            "stack_size": len(stack.adapters),
            "loads_at_sampling": True,
            "start_after_evaluations": start_after_evaluations,
        }
        return stack, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3VDNCheckpoint:
    """Select OpenVDN's hybrid branch and its required adapter stack."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "repository": (
                    "STRING",
                    {
                        "default": VDN_REPOSITORY_ID,
                        "tooltip": (
                            "Local directory produced by hf download "
                            "OpenVDN/vdn-minimax-h3 --local-dir <directory>."
                        ),
                    },
                ),
                "stage": (list(VDN_STAGES), {"default": next(iter(VDN_STAGES))}),
            },
            "optional": {
                "adaln_input_grid": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Original H3 SiLU timestep grid for an AdaLN-pruned base. "
                            "Uses the normal ComfyUI LoRA/model paths; blank checks the "
                            "Turbo adapter and transformer directories for "
                            "h3_silu_temb_grid.safetensors."
                        ),
                    },
                ),
                "inference_backend": (
                    ["verified", "reference", "indexed_experimental"],
                    {
                        "default": "verified",
                        "tooltip": (
                            "Verified inference fusion; reference disables new optimizations. "
                            "Indexed attention is numerically approximate and experimental."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "WEETODD_H3_COMPONENTS",
        "WEETODD_H3_CONFIG",
        "WEETODD_H3_VDN",
        "WEETODD_H3_LORAS",
        "STRING",
    )
    RETURN_NAMES = ("components", "config", "vdn", "loras", "vdn_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Select the VDN-H3 hybrid-attention branch and required LoRAs from a downloaded "
        "OpenVDN/vdn-minimax-h3 repository. Supports T2VA with resident or paged H3 base "
        "transformers. Experimental MLX runtime with a verified FP32 Metal matrix solve "
        "and CPU fallback; "
        "full-checkpoint render parity has not been established."
    )

    def select(
        self,
        components,
        config,
        repository,
        stage,
        adaln_input_grid="",
        inference_backend="verified",
    ):
        if components.task != "t2va":
            raise ValueError("VDN-H3 currently supports WeeTodd T2VA components only.")
        transformer = Path(components.resolved_paths()["transformer"])
        vdn = resolve_vdn_spec(_resolve_component_root(repository), stage)
        vdn = replace(vdn, inference_backend=inference_backend)
        vdn.validate()
        from .lora import H3LoRASpec, H3LoRAStack

        adapters = H3LoRAStack().append(
            H3LoRASpec(
                path=vdn.default_adapter,
                strength=1.0,
                profile="standard",
                qkv_layout="contiguous_qkv",
            )
        )
        if vdn.turbo_adapter is not None:
            grid = _resolve_lora_path(adaln_input_grid) if adaln_input_grid.strip() else None
            if grid is None:
                for directory in (Path(vdn.turbo_adapter).parent, transformer, transformer.parent):
                    candidate = directory / "h3_silu_temb_grid.safetensors"
                    if candidate.is_file():
                        grid = candidate
                        break
            adapters = adapters.append(
                H3LoRASpec(
                    path=vdn.turbo_adapter,
                    strength=1.0,
                    profile="turbo",
                    qkv_layout="contiguous_qkv",
                    adaln_input_grid=str(grid) if grid is not None else None,
                )
            )
        configured = replace(
            config,
            steps=vdn.schedule_points,
            sampling_method="euler",
            attention_chunk_size="automatic",
        )
        configured.validate()
        info = {
            "repository_id": VDN_REPOSITORY_ID,
            "stage": vdn.stage,
            "checkpoint": Path(vdn.checkpoint).name,
            "schedule_points": vdn.schedule_points,
            "transformer_evaluations": vdn.schedule_points - 1,
            "hybrid_attention": "window_softmax_plus_bidirectional_vdn_solve",
            "softmax_window": {"chunk": 5, "radius": 1, "anchor_frames": "both"},
            "linear_branch": Path(vdn.linear_branch).name,
            "adapters": [Path(item.path).name for item in adapters.adapters],
            "task": "t2va",
            "runtime": "mlx_metal_solve_with_cpu_fallback",
            "inference_backend": inference_backend,
            "full_checkpoint_parity_validated": False,
            "adaln_input_grid": (
                Path(adapters.adapters[-1].adaln_input_grid).name
                if adapters.adapters[-1].adaln_input_grid is not None else None
            ),
        }
        return components, configured, vdn, adapters, json.dumps(info, indent=2, sort_keys=True)


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
            start_after_evaluations = int(selected.get("start_after_evaluations", 0))
            try:
                loras, _ = WeeToddH3LoRALoader().load(
                    lora_name,
                    1.0,
                    "turbo",
                    qkv_layout="auto",
                    start_after_evaluations=start_after_evaluations,
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
            "lora_start_after_evaluations": (
                int(selected.get("start_after_evaluations", 0)) if lora_file is not None else None
            ),
            "trajectory_offline_replay": bool(
                trajectory_forecast is not None and trajectory_forecast.offline_smoothing_replay
            ),
            "canvas": [configured.width, configured.height],
            "duration_seconds": configured.duration_seconds,
            "seed": configured.seed,
            "source": selected.get("source"),
            "required_transformer": selected.get("required_transformer"),
            "required_attention_profile": selected.get("required_attention_profile"),
            "measurement": selected.get("measurement"),
        }
        return configured, loras, trajectory_forecast, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3FastH3ProductionProfile:
    """Apply one fail-closed native FastH3 VSA production policy."""

    _PROFILES = {
        "Balanced — compact indexed Metal (recommended)": {
            "attention": "fasth3_vsa_90_metal",
            "status": "validated_numerically_approximate",
            "storage_layout": "compact_preordered",
        },
        "Speed candidate — compact Metal + 40 layers": {
            "attention": "fasth3_vsa_90_metal",
            "status": "generatively_approximate_candidate",
            "storage_layout": "compact_preordered",
            "active_layers": 40,
        },
        "Conservative — grouped MLX fallback": {
            "attention": "fasth3_vsa_90",
            "status": "experimental_control",
            "storage_layout": "padded_tiles",
        },
        "Experimental — compact Metal + fused QKV": {
            "attention": "fasth3_vsa_90_metal_fused_qkv",
            "status": "numerically_approximate_research",
            "storage_layout": "compact_preordered",
        },
    }
    _TRANSFORMER = "weetodd-fasth3-vsa-datafree-q8-paged"
    _SOURCE = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
    _SOURCE_REVISION = "b65818d41939b5085451074fe8ca8b799f8d4921"
    _KEEP_RESOLUTION = "Keep Generation Config"
    _RESOLUTION_MATRIX = {
        "512×256 — 1m 00s / 5.79 GB MLX": {
            "width": 512,
            "height": 256,
            "complete_wall_seconds": 60.29744724999182,
            "complete_peak_memory_bytes": 5791476444,
        },
        "768×448 — 2m 22s / 7.18 GB MLX": {
            "width": 768,
            "height": 448,
            "complete_wall_seconds": 141.7950401660055,
            "complete_peak_memory_bytes": 7176383420,
        },
        "1024×576 — 4m 14s / 9.29 GB MLX": {
            "width": 1024,
            "height": 576,
            "complete_wall_seconds": 254.22458666702732,
            "complete_peak_memory_bytes": 9286711236,
        },
        "1280×704 — 6m 43s / 11.96 GB MLX": {
            "width": 1280,
            "height": 704,
            "complete_wall_seconds": 402.85038945800625,
            "complete_peak_memory_bytes": 11959781484,
        },
        "1536×832 — 9m 28s / 15.23 GB MLX": {
            "width": 1536,
            "height": 832,
            "complete_wall_seconds": 568.4969888750347,
            "complete_peak_memory_bytes": 15226952612,
        },
        "1920×1088 — 17m 00s / 22.16 GB MLX": {
            "width": 1920,
            "height": 1088,
            "complete_wall_seconds": 1019.617088708037,
            "complete_peak_memory_bytes": 22160015292,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "profile": (
                    list(cls._PROFILES),
                    {"default": "Balanced — compact indexed Metal (recommended)"},
                ),
                "resolution_preset": (
                    [cls._KEEP_RESOLUTION, *cls._RESOLUTION_MATRIX],
                    {
                        "default": "768×448 — 2m 22s / 7.18 GB MLX",
                        "tooltip": (
                            "Select one measured M3 Ultra canvas or preserve the dimensions from "
                            "H3 Generation Config. Times include sampling, decode, and mux for a "
                            "107-frame run; shared text encoding is excluded."
                        ),
                    },
                ),
            },
            "optional": {
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 1024,
                        "max": 131072,
                        "step": 1024,
                        "advanced": True,
                        "tooltip": (
                            "Keep 4096 for the shipped scaling range so the trained VSA route "
                            "remains active at 512x256. Raise only for deliberate dense fallback."
                        ),
                    },
                ),
                "advisory_memory_budget_gb": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1024.0,
                        "step": 1.0,
                        "round": 0.1,
                        "advanced": True,
                        "tooltip": (
                            "Optional planning budget in GiB. Zero uses detected physical memory. "
                            "A manual value changes advisory warnings only; it does not simulate "
                            "or validate a lower-memory render."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "WEETODD_H3_COMPONENTS",
        "WEETODD_H3_CONFIG",
        "WEETODD_H3_SOL_ATTENTION",
        "STRING",
        "WEETODD_H3_FASTVIDEO",
    )
    RETURN_NAMES = ("components", "config", "sol_attention", "profile_info", "fastvideo")
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Recommended native FastH3 VSA entry point. It applies the four-evaluation schedule, "
        "selects one measured resolution and compact Metal backend or an explicit fallback, "
        "reports hardware-aware advisories, and rejects incompatible checkpoints before sampling. "
        "The opt-in 40-layer Speed candidate skips joint video/audio layers and still requires "
        "sound-effect listening acceptance. Connect its fastvideo output to H3 Sample."
    )

    @classmethod
    def validate_sampling_inputs(
        cls, raw, components, config, attention, fastvideo, *, modifiers=(), loras=None
    ):
        """Check the profile's execution contract before transformer loading."""
        if not raw:
            return None
        try:
            info = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Production profile info must be valid JSON.") from exc
        if not isinstance(info, dict):
            raise ValueError("Production profile info must decode to a JSON object.")
        # Older metadata-only callers remain readable. Current profile outputs are versioned
        # and must be connected together; their declared policy is an execution contract.
        if "contract_version" not in info:
            return info
        if info["contract_version"] != 1 or info.get("profile") not in cls._PROFILES:
            raise ValueError("Unsupported FastH3 production profile contract.")
        selected = cls._PROFILES[info["profile"]]
        if (
            components.task != "t2va"
            or components.resolved_paths()["transformer"].name != cls._TRANSFORMER
        ):
            raise ValueError("FastH3 profile requires its native VSA student and T2VA task.")
        if config.steps != 5 or config.sampling_method != "euler":
            raise ValueError(
                "FastH3 profile requires five Euler schedule points / four evaluations."
            )
        if [config.width, config.height] != info.get(
            "canvas"
        ) or config.duration_seconds != info.get("duration_seconds_requested"):
            raise ValueError(
                "FastH3 profile config changed after selection; reconnect its config output."
            )
        if any(value is not None for value in modifiers) or (loras is not None and loras.adapters):
            raise ValueError(
                "FastH3 production profiles cannot be combined with cache, forecast, VDN, "
                "continuation, or LoRA modifiers."
            )
        expected_attention = {
            "enabled": True,
            "consumer_backend": "grouped_sdpa"
            if selected["attention"] == "fasth3_vsa_90"
            else "metal_indexed",
            "block_stack_preorder": selected["attention"] != "fasth3_vsa_90",
            "qkv_prep_backend": "metal_fused"
            if selected["attention"].endswith("fused_qkv")
            else "mlx",
            "sparsity": 0.9,
            "min_tokens": info.get("min_tokens"),
        }
        if attention is None or any(
            getattr(attention, key, None) != value for key, value in expected_attention.items()
        ):
            raise ValueError(
                "FastH3 profile attention changed or is disconnected; reconnect its "
                "sol_attention output."
            )
        expected_layers = selected.get("active_layers", 50)
        if (
            info.get("layers_per_evaluation") != expected_layers
            or info.get("expected_attention_calls") != expected_layers * 4
        ):
            raise ValueError("FastH3 profile execution counts changed after selection.")
        if expected_layers == 40:
            if fastvideo is None or (
                fastvideo.active_layers != 40
                or fastvideo.pair_target_video
                or fastvideo.protect_first_layers != 2
                or fastvideo.protect_last_layers != 2
            ):
                raise ValueError(
                    "FastH3 40-layer profile requires its fastvideo output connected unchanged "
                    "to H3 Sample."
                )
            fastvideo.validate(50)
            root = components.resolved_paths()["transformer"]
            try:
                manifest = json.loads((root / "paged_manifest.json").read_text())
                quantization = json.loads((root / "quant_config.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "FastH3 40-layer profile requires readable native checkpoint metadata."
                ) from exc
            if (
                not isinstance(manifest, dict)
                or not isinstance(quantization, dict)
                or manifest.get("format") != "weetodd-h3-paged-v1"
                or manifest.get("source") != cls._SOURCE
                or manifest.get("source_revision") != cls._SOURCE_REVISION
                or manifest.get("num_blocks") != 50
                or manifest.get("attention") != "vsa_h3_64_90"
                or manifest.get("sampling") != {"schedule_points": 5, "transformer_evaluations": 4}
                or quantization.get("bits") != 8
                or quantization.get("group_size") != 64
                or quantization.get("quantize_core") is not True
                or quantization.get("quantize_adaln") is not True
                or quantization.get("adaln_bits") != 8
                or quantization.get("overrides") != {}
            ):
                raise ValueError(
                    "FastH3 40-layer profile requires the validated native Q8 VSA checkpoint "
                    "revision; renaming another checkpoint is not sufficient."
                )
        elif fastvideo is not None and fastvideo.enabled:
            raise ValueError(
                "The selected FastH3 profile requires all 50 layers; select the Speed "
                "candidate for thinning."
            )
        return info

    @staticmethod
    def validate_execution(info, latents):
        """Do not publish a Speed result unless the engine proves the requested path ran."""
        if not info or info.get("contract_version") != 1 or info.get("layers_per_evaluation") != 40:
            return
        report = getattr(latents, "fast_h3_approximation_report", None) or {}
        kept = report.get("active_layer_indices", [])
        skipped = report.get("skipped_layer_indices", [])
        attention = getattr(latents, "sol_attention_report", None) or {}
        if (
            latents.transformer_evaluations != 4
            or report.get("executed_layers") != 40
            or report.get("skipped_layers") != 10
            or len(kept) != 40
            or len(skipped) != 10
            or sorted([*kept, *skipped]) != list(range(50))
            or not {0, 1, 48, 49}.issubset(kept)
            or attention.get("executed_calls") != 160
            or attention.get("fallback_calls") != 0
            or attention.get("storage_layout") != "compact_preordered"
        ):
            raise RuntimeError(
                "FastH3 40-layer execution proof failed; no production-profile artifact was "
                "published. Inspect layer and attention telemetry."
            )

    @staticmethod
    def _hardware_report():
        memory_bytes = None
        try:
            memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, TypeError, ValueError):
            pass
        chip = None
        if platform.system() == "Darwin":
            try:
                import subprocess

                completed = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                chip = completed.stdout.strip() or None
            except (OSError, subprocess.SubprocessError):
                pass
        return {
            "architecture": platform.machine(),
            "chip": chip,
            "unified_memory_bytes": memory_bytes,
            "unified_memory_gib": (memory_bytes / 1024**3 if memory_bytes is not None else None),
        }

    @classmethod
    def _measurement_for_canvas(cls, width, height):
        return next(
            (
                {"label": label, **measurement}
                for label, measurement in cls._RESOLUTION_MATRIX.items()
                if (measurement["width"], measurement["height"]) == (width, height)
            ),
            None,
        )

    def apply(
        self,
        components,
        config,
        profile,
        resolution_preset="768×448 — 2m 22s / 7.18 GB MLX",
        min_tokens=4096,
        advisory_memory_budget_gb=0.0,
    ):
        try:
            selected = self._PROFILES[profile]
        except KeyError as exc:
            raise ValueError(f"Unknown FastH3 production profile: {profile!r}.") from exc

        transformer = components.resolved_paths()["transformer"]
        if transformer.name != self._TRANSFORMER:
            raise ValueError(
                "FastH3 production profiles require the native VSA student transformer "
                f"{self._TRANSFORMER!r}; selected {transformer.name!r}."
            )
        if components.task != "t2va":
            raise ValueError(
                "FastH3 Preview v1 native VSA supports T2VA only; "
                f"selected task {components.task!r}."
            )

        if resolution_preset == self._KEEP_RESOLUTION:
            configured = replace(config, steps=5, sampling_method="euler")
        else:
            try:
                resolution = self._RESOLUTION_MATRIX[resolution_preset]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown FastH3 resolution preset: {resolution_preset!r}."
                ) from exc
            configured = replace(
                config,
                steps=5,
                sampling_method="euler",
                width=resolution["width"],
                height=resolution["height"],
                resolution_mode="exact dimensions",
                resolution_tier="custom",
                aspect_ratio="custom",
            )
        configured.validate()
        attention, raw_attention = WeeToddH3SolAttention().configure(
            selected["attention"],
            0.75,
            0.2,
            1.0,
            2,
            int(min_tokens),
        )
        attention_info = json.loads(raw_attention)
        fastvideo = None
        active_layers = selected.get("active_layers", 50)
        if active_layers == 40:
            from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig

            fastvideo = FastH3ApproximationConfig(active_layers=40)
            fastvideo.validate(50)
        measurement = self._measurement_for_canvas(configured.width, configured.height)
        hardware = self._hardware_report()
        memory_budget_gib = float(advisory_memory_budget_gb)
        if not math.isfinite(memory_budget_gib) or memory_budget_gib < 0.0:
            raise ValueError("FastH3 advisory memory budget must be a finite non-negative value.")
        budget_is_override = memory_budget_gib > 0.0
        advisory_memory_bytes = (
            round(memory_budget_gib * 1024**3)
            if budget_is_override
            else hardware["unified_memory_bytes"]
        )
        advisory_memory_source = (
            "manual_override"
            if budget_is_override
            else (
                "detected_physical_memory" if advisory_memory_bytes is not None else "unavailable"
            )
        )
        warnings = []
        if budget_is_override:
            warnings.append(
                f"Manual advisory memory budget of {memory_budget_gib:.1f} GiB is active; it "
                "changes warning evaluation only and does not simulate or validate a "
                "lower-memory render."
            )
        measurement_applicable = measurement is not None
        if configured.inference_optimization != "off":
            measurement_applicable = False
            warnings.append(
                "An opt-in H3 arithmetic experiment is active; the reference timing matrix "
                "does not measure this policy. Generic QMM/dense rounding may differ."
            )
        if fastvideo is not None:
            measurement_applicable = False
            warnings.append(
                "The 40-layer Speed candidate changes the joint video/audio trajectory. "
                "Three 640x384 scenarios passed visual continuity, audio health, and dialogue "
                "transcription; sound-effect fidelity and synchronization still need listening "
                "acceptance. The resolution matrix is the 50-layer Balanced reference, not "
                "a 40-layer timing or memory measurement."
            )
        if measurement is None:
            warnings.append("The selected canvas is outside the measured FastH3 resolution matrix.")
        if configured.duration_seconds != 4.0:
            measurement_applicable = False
            warnings.append(
                "Measured time and memory apply to a 4.0-second request aligned to 107 frames; "
                f"the selected request is {configured.duration_seconds:.1f} seconds."
            )
        if selected["attention"] != "fasth3_vsa_90_metal":
            measurement_applicable = False
            warnings.append(
                "Measured matrix values use the Balanced compact indexed-Metal backend; the "
                "selected attention policy has different performance."
            )
        if hardware["chip"] is None:
            measurement_applicable = False
            warnings.append(
                "Apple chip identity is unavailable; M3 Ultra timing is reference-only."
            )
        elif hardware["chip"] != "Apple M3 Ultra":
            measurement_applicable = False
            warnings.append(
                f"Detected {hardware['chip']}; measured times are specific to Apple M3 Ultra."
            )

        advisory_minimum_bytes = None
        headroom_status = "unmeasured"
        if measurement is not None:
            peak_bytes = int(measurement["complete_peak_memory_bytes"])
            advisory_minimum_bytes = max(round(peak_bytes * 1.35), peak_bytes + 4_000_000_000)
            if advisory_memory_bytes is None:
                headroom_status = "unknown"
                warnings.append(
                    "Unified-memory capacity is unavailable; confirm headroom in H3 Preflight."
                )
            elif advisory_memory_bytes < advisory_minimum_bytes:
                headroom_status = "limited"
                budget_description = (
                    "the manual advisory budget is"
                    if budget_is_override
                    else "the detected system has"
                )
                warnings.append(
                    "Limited measured headroom: this row used "
                    f"{peak_bytes / 1_000_000_000:.2f} GB of MLX allocations on the reference "
                    f"machine, while {budget_description} "
                    f"{advisory_memory_bytes / 1024**3:.1f} GiB. "
                    "This is advisory; H3 Preflight and live free memory remain authoritative."
                )
            else:
                headroom_status = "comfortable"
        info = {
            "contract_version": 1,
            "profile": profile,
            "status": selected["status"],
            "transformer": transformer.name,
            "task": components.task,
            "requested_schedule_points": configured.steps,
            "transformer_evaluations": configured.steps - 1,
            "layers_per_evaluation": active_layers,
            "expected_attention_calls": active_layers * (configured.steps - 1),
            "fastvideo": asdict(fastvideo) if fastvideo is not None else None,
            "sampling_method": configured.sampling_method,
            "inference_optimization": configured.inference_optimization,
            "attention_profile": selected["attention"],
            "attention_backend": attention_info["backend"],
            "qkv_prep_backend": attention_info["qkv_prep_backend"],
            "storage_layout": selected["storage_layout"],
            "min_tokens": attention.min_tokens,
            "canvas": [configured.width, configured.height],
            "resolution_selector": resolution_preset,
            "duration_seconds_requested": configured.duration_seconds,
            "measurement": (
                {
                    **measurement,
                    "reference_hardware": "Mac Studio M3 Ultra, 256 GB",
                    "requested_duration_seconds": 4.0,
                    "aligned_frames": 107,
                    "fps": 24,
                    "timing_scope": "sampling + direct video/audio decode + mux",
                    "shared_text_encoding_excluded": True,
                    "applicable_to_current_selection": measurement_applicable,
                }
                if measurement is not None
                else None
            ),
            "hardware": {
                **hardware,
                "advisory_minimum_bytes": advisory_minimum_bytes,
                "advisory_minimum_gib": (
                    advisory_minimum_bytes / 1024**3 if advisory_minimum_bytes is not None else None
                ),
                "policy": "advisory_only",
                "advisory_memory_budget_bytes": advisory_memory_bytes,
                "advisory_memory_budget_gib": (
                    advisory_memory_bytes / 1024**3 if advisory_memory_bytes is not None else None
                ),
                "advisory_memory_source": advisory_memory_source,
                "advisory_budget_is_override": budget_is_override,
                "headroom_status": headroom_status,
                "validation_scope": (
                    "warning_policy_only_not_lower_memory_hardware"
                    if budget_is_override
                    else "detected_hardware"
                ),
            },
            "warnings": warnings,
            "compatibility": "native FastH3 VSA student; T2VA only",
            "fallback": (
                "select Conservative — grouped MLX fallback if compact Metal is unavailable"
                if selected["attention"] == "fasth3_vsa_90_metal"
                else None
            ),
        }
        return (
            components,
            configured,
            attention,
            json.dumps(info, indent=2, sort_keys=True),
            fastvideo,
        )


class WeeToddH3FastVideoApproximation:
    """Configure disclosed FastH3 layer and target-video token approximations."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": (
                    ["layer_thinning_40", "token_pairing", "combined_40_pairing", "manual"],
                    {"default": "layer_thinning_40"},
                ),
                "active_layers": (
                    "INT",
                    {
                        "default": 40,
                        "min": 4,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Used only by manual; 50 disables layer thinning.",
                    },
                ),
                "pair_target_video": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Pair adjacent horizontal target-video tokens between the selected "
                            "layers. Prefix text, condition, and audio rows remain full size."
                        ),
                    },
                ),
                "pair_start_layer": (
                    "INT",
                    {"default": 4, "min": 0, "max": 49, "step": 1},
                ),
                "pair_end_layer": (
                    "INT",
                    {"default": 30, "min": 1, "max": 50, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_FASTVIDEO", "STRING")
    RETURN_NAMES = ("fastvideo", "policy_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Opt-in generative FastH3 approximations. Layer thinning is ranked once from the full "
        "AdaLN schedule; token pairing keeps a full-resolution residual bypass."
    )

    def configure(
        self,
        profile,
        active_layers,
        pair_target_video,
        pair_start_layer,
        pair_end_layer,
    ):
        from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig

        if profile == "layer_thinning_40":
            active_layers, pair_target_video = 40, False
        elif profile == "token_pairing":
            active_layers, pair_target_video = 50, True
        elif profile == "combined_40_pairing":
            active_layers, pair_target_video = 40, True
        config = FastH3ApproximationConfig(
            active_layers=None if int(active_layers) == 50 else int(active_layers),
            pair_target_video=bool(pair_target_video),
            pair_start_layer=int(pair_start_layer),
            pair_end_layer=int(pair_end_layer),
        )
        config.validate(50)
        info = {
            **asdict(config),
            "profile": profile,
            "status": "generatively_approximate",
            "layer_policy": "schedule-wide AdaLN attention+MLP gate magnitude",
            "token_policy": (
                "horizontal target-video pairs with full-resolution residual bypass"
                if config.pair_target_video
                else "disabled"
            ),
            "prefix_policy": "text, condition video, and audio rows remain full resolution",
            "vsa_pairing_compatibility": "layer thinning only; token pairing requires dense FastH3",
        }
        return config, json.dumps(info, indent=2, sort_keys=True)


class WeeToddH3SolAttention:
    """Configure independent MLX Sol-style or trained FastH3 VSA attention."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": (
                    [
                        "quality",
                        "balanced",
                        "speed",
                        "fasth3_vsa_90",
                        "fasth3_vsa_90_metal",
                        "fasth3_vsa_90_metal_fused_qkv",
                        "manual",
                    ],
                    {"default": "balanced"},
                ),
                "tau": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Higher values route fewer exact attention blocks. Used only by "
                            "the manual profile."
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "dense_blocks": (
                    "INT",
                    {"default": 2, "min": 0, "max": 50, "step": 1},
                ),
                "min_tokens": (
                    "INT",
                    {"default": 16384, "min": 1024, "max": 131072, "step": 1024},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_SOL_ATTENTION", "STRING")
    RETURN_NAMES = ("sol_attention", "policy_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Experimental H3 sparse attention. Sol profiles use the fused MLX Metal backend; the "
        "FastH3 profiles preserve trained 64-token routing and compression gates with either "
        "grouped SDPA or an indexed Metal consumer. Both preserve the complete multimodal prefix."
    )

    def configure(
        self,
        profile,
        tau,
        start_percent,
        end_percent,
        dense_blocks,
        min_tokens,
    ):
        if profile.startswith("fasth3_vsa_90"):
            from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig

            config = FastH3VSAConfig(
                enabled=True,
                sparsity=0.9,
                min_tokens=int(min_tokens),
                query_tile_batch=8,
                block_stack_preorder=profile != "fasth3_vsa_90",
                consumer_backend=(
                    "metal_indexed" if profile != "fasth3_vsa_90" else "grouped_sdpa"
                ),
                qkv_prep_backend=(
                    "metal_fused"
                    if profile == "fasth3_vsa_90_metal_fused_qkv"
                    else "mlx"
                ),
            )
            config.validate()
            info = {
                "profile": profile,
                "backend": (
                    "mlx_grouped_sdpa_vsa_h3"
                    if config.consumer_backend == "grouped_sdpa"
                    else "metal_indexed_vsa_h3"
                ),
                "qkv_prep_backend": config.qkv_prep_backend,
                "block_stack_preorder": config.block_stack_preorder,
                "storage_layout": (
                    "compact_preordered"
                    if config.consumer_backend == "metal_indexed"
                    else "padded_tiles"
                ),
                "sparsity": config.sparsity,
                "tile_shape": [4, 4, 4],
                "tile_tokens": 64,
                "query_tile_batch": config.query_tile_batch,
                "prefix_policy": "segment-pure dense queries and exempt keys",
                "requires": "FastH3 checkpoint with trained VSA gate_compress weights",
                "status": (
                    "numerically_approximate_research"
                    if config.qkv_prep_backend == "metal_fused"
                    else (
                        "validated_numerically_approximate"
                        if config.consumer_backend == "metal_indexed"
                        else "experimental_control"
                    )
                ),
            }
            return config, json.dumps(info, indent=2, sort_keys=True)

        from minimax_h3_mlx.sol_attention import SolAttentionConfig

        presets = {
            "quality": (0.75, 0.30, 1.0, 4),
            "balanced": (1.00, 0.25, 1.0, 3),
            "speed": (1.25, 0.20, 1.0, 2),
        }
        if profile in presets:
            tau, start_percent, end_percent, dense_blocks = presets[profile]
        config = SolAttentionConfig(
            enabled=True,
            tau=float(tau),
            min_tokens=int(min_tokens),
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            dense_blocks=int(dense_blocks),
        )
        config.validate()
        info = {
            "profile": profile,
            "tau": config.tau,
            "start_percent": config.start_percent,
            "end_percent": config.end_percent,
            "dense_blocks": config.dense_blocks,
            "min_tokens": config.min_tokens,
            "exact_prefix": "automatic: text + visual references + audio",
            "status": "experimental",
        }
        return config, json.dumps(info, indent=2, sort_keys=True)


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
            },
            "optional": {
                "reuse_strategy": (
                    ["output_residual", "core_residual_fresh_heads"],
                    {"default": "output_residual"},
                ),
                "allow_turbo_experimental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Allow Turbo LoRA with EasyCache for controlled comparisons. "
                            "This combination can change video and audio."
                        ),
                    },
                ),
            },
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
        reuse_strategy="output_residual",
        allow_turbo_experimental=False,
    ):
        from minimax_h3_mlx.easycache import H3EasyCacheConfig

        config = H3EasyCacheConfig(
            mode=mode,
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            auto_multiplier=auto_multiplier,
            max_skip_fraction=max_skip_fraction,
            reuse_strategy=reuse_strategy,
            allow_turbo_experimental=allow_turbo_experimental,
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
            },
            "optional": {
                "video_tile_mode": (["fixed", "geometry_experimental"], {"default": "fixed"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "decode_info")
    FUNCTION = "decode"
    CATEGORY = "WeeTodd/H3/decoding"
    DESCRIPTION = (
        "Decode the video stream from synchronized H3 latents with the final video VAE. "
        "The audio latent stream remains available on the original latent output."
    )

    def decode(self, components, latents, unload_after_decode, video_tile_mode="fixed"):
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
            H3VideoVAESpec.from_components(components, tile_mode=video_tile_mode),
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
            "phase_memory": getattr(latents, "phase_memory", None),
            "video_vae_quantization": result.quantization,
            "memory_mode": latents.generation_config.memory_mode,
            "tile_decode_batch": result.decode_batch,
            "tile_plan": getattr(result, "tile_plan", None),
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
            "phase_memory": getattr(latents, "phase_memory", None),
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
                "video_tile_mode": (
                    ["fixed", "geometry_experimental"],
                    {
                        "default": "fixed",
                        "tooltip": (
                            "Experimental geometry-aware VAE tiling changes decode context, "
                            "not output resolution."
                        ),
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
        video_tile_mode="fixed",
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
            video_tile_mode=video_tile_mode,
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


class WeeToddH3DirectPublishChain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "chain": ("WEETODD_H3_LATENT_CHAIN",),
                "filename_prefix": ("STRING", {"default": "WeeTodd/H3_chain"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "max_av_drift_seconds": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.001},
                ),
            },
            "optional": {
                "generation_metadata": ("STRING", {"default": "{}", "multiline": True}),
                "ffmpeg_path": ("STRING", {"default": "", "advanced": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "publish"
    CATEGORY = "WeeTodd/H3/output"
    DESCRIPTION = (
        "Decode an H3 latent chain by VAE stage, remove duplicated joins, force exact 24 fps / "
        "32 kHz duration, and atomically publish one MP4."
    )

    def publish(
        self,
        components,
        chain,
        filename_prefix,
        crf,
        max_av_drift_seconds,
        generation_metadata="{}",
        ffmpeg_path="",
    ):
        chain.validate_complete()
        first = chain.windows[0]
        try:
            supplied = json.loads(generation_metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Generation metadata must be a valid JSON object.") from exc
        if not isinstance(supplied, dict):
            raise ValueError("Generation metadata must be a JSON object.")
        publication = _publication_environment(ffmpeg_path)
        supplied.update(
            {
                "timeline": chain.timeline.metadata(),
                "components": {
                    "checkpoint": Path(components.checkpoint).name,
                    "task": components.task,
                    "transformer": Path(components.resolved_paths()["transformer"]).name,
                },
                "windows": [
                    {
                        "index": index,
                        "seed": window.generation_config.seed,
                        "steps": window.generation_config.steps,
                        "transformer_evaluations": window.transformer_evaluations,
                        "generation_seconds": window.total_seconds,
                        "loras": list(window.lora_report),
                    }
                    for index, window in enumerate(chain.windows, start=1)
                ],
                "publication": publication,
            }
        )
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
            video_releases = prepare_low_memory_stage(
                "video_vae", first.generation_config.memory_mode
            )

        def prepare_audio_stage():
            nonlocal audio_releases
            audio_releases = prepare_low_memory_stage(
                "audio_vae", first.generation_config.memory_mode
            )

        output_root = Path(publication["output_directory"])
        target = _safe_output_target(output_root, filename_prefix, first.generation_config.seed)
        result = publish_latent_chain_direct(
            target,
            components,
            chain,
            crf=crf,
            max_av_drift_seconds=max_av_drift_seconds,
            generation_metadata=json.dumps(supplied),
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
        return {"ui": {"gifs": [preview]}, "result": (str(result.video_path), info)}


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
                    {
                        "default": 5.0,
                        "min": 2.5,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": (
                            "Durations below 5 seconds are experimental and snap upward to the "
                            "H3 17n+5 frame grid."
                        ),
                    },
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
                    [
                        "auto",
                        "mlx",
                        "mpp_experimental",
                        "m5_low_bit_experimental",
                        "mpp_resident_expanded_experimental",
                    ],
                    {
                        "default": "auto",
                        "advanced": True,
                        "tooltip": (
                            "Auto uses bitwise-verified Metal Performance Primitives acceleration "
                            "for eligible BF16 transformer projections. The M5 low-bit option "
                            "requires an M5 g17 GPU and a quantized checkpoint; every unsupported "
                            "case falls back to standard MLX."
                            " Resident expanded mode trades RAM for selective Q8-to-BF16 expansion."
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
                "inference_optimization": (
                    ["off", "transient_q8", "compiled_adaln", "combined"],
                    {
                        "default": "off", "advanced": True,
                        "tooltip": "Experimental H3 hot paths. Benchmark on your GPU before use.",
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
        "dimensions. The canvas stays on H3's 32-pixel grid. "
        "Optional hot-path experiments default off."
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
        projection_backend="auto",
        short_edge=None,
        sampling_method="euler",
        inference_optimization="off",
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
            inference_optimization=inference_optimization,
        )
        config.validate()
        if ratio_mode:
            short_edge_label = f"{min(width, height)} px short edge"
            return (
                config,
                f"{width} × {height} pixels — {normalized_ratio} — {short_edge_label}",
            )
        return config, f"{width} × {height} pixels — custom"


class WeeToddH3LowMemoryTuning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("WEETODD_H3_CONFIG",),
                "attention_head_chunk_size": (
                    ["automatic", "2", "4", "8", "disabled"],
                    {
                        "default": "automatic",
                        "tooltip": (
                            "Chunk independent attention heads to bound the SDPA workspace. "
                            "Automatic selects four heads in low-memory BF16 mode."
                        ),
                    },
                ),
                "ffn_row_chunk_size": (
                    ["automatic", "128", "256", "512", "disabled"],
                    {
                        "default": "automatic",
                        "tooltip": (
                            "Chunk packed rows to bound the SwiGLU intermediate. Automatic "
                            "selects 256 rows in low-memory BF16 mode."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "apply"
    CATEGORY = "WeeTodd/H3/sampling"
    DESCRIPTION = (
        "Apply optional MLX attention-head and feed-forward row chunking without invalidating "
        "older Generation Config workflows."
    )

    def apply(self, config, attention_head_chunk_size, ffn_row_chunk_size):
        tuned = replace(
            config,
            attention_head_chunk_size=attention_head_chunk_size,
            ffn_row_chunk_size=ffn_row_chunk_size,
        )
        tuned.validate()
        return (tuned,)


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
    "WeeToddH3PreviewOverride": WeeToddH3PreviewOverride,
    "WeeToddH3QuantizedTransformerLoader": WeeToddH3QuantizedTransformerLoader,
    "WeeToddH3Preflight": WeeToddH3Preflight,
    "WeeToddH3TokenBudget": WeeToddH3TokenBudget,
    "WeeToddH3FirstFrame": WeeToddH3FirstFrame,
    "WeeToddH3LastFrame": WeeToddH3LastFrame,
    "WeeToddH3FirstLastFrame": WeeToddH3FirstLastFrame,
    "WeeToddH3ChainedTimeline": WeeToddH3ChainedTimeline,
    "WeeToddH3Frames": WeeToddH3Frames,
    "WeeToddH3TimedKeyframe": WeeToddH3TimedKeyframe,
    "WeeToddH3ReferenceImage": WeeToddH3ReferenceImage,
    "WeeToddH3ReferenceVideo": WeeToddH3ReferenceVideo,
    "WeeToddH3ReferenceAudio": WeeToddH3ReferenceAudio,
    "WeeToddH3TimelineVisualGuide": WeeToddH3TimelineVisualGuide,
    "WeeToddH3TimelineAudioGuide": WeeToddH3TimelineAudioGuide,
    "WeeToddH3KeyframeEncode": WeeToddH3KeyframeEncode,
    "WeeToddH3TimedKeyframeEncode": WeeToddH3TimedKeyframeEncode,
    "WeeToddH3ReferenceEncode": WeeToddH3ReferenceEncode,
    "WeeToddH3ReferenceStrength": WeeToddH3ReferenceStrength,
    "WeeToddH3TextEncode": WeeToddH3TextEncode,
    "WeeToddH3UnloadTextEncoder": WeeToddH3UnloadTextEncoder,
    "WeeToddH3ContinuationContext": WeeToddH3ContinuationContext,
    "WeeToddH3ChainAppend": WeeToddH3ChainAppend,
    "WeeToddH3Sample": WeeToddH3Sample,
    "WeeToddH3LearnedLatentUpscalerLoader": WeeToddH3LearnedLatentUpscalerLoader,
    "WeeToddH3LatentHiresFix": WeeToddH3LatentHiresFix,
    "WeeToddH3LoRALoader": WeeToddH3LoRALoader,
    "WeeToddH3VDNCheckpoint": WeeToddH3VDNCheckpoint,
    "WeeToddH3ValidatedSamplingPreset": WeeToddH3ValidatedSamplingPreset,
    "WeeToddH3FastH3ProductionProfile": WeeToddH3FastH3ProductionProfile,
    "WeeToddH3FastVideoApproximation": WeeToddH3FastVideoApproximation,
    "WeeToddH3SolAttention": WeeToddH3SolAttention,
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
    "WeeToddH3DirectPublishChain": WeeToddH3DirectPublishChain,
    "WeeToddH3ModelLoader": WeeToddH3ModelLoader,
    "WeeToddH3GenerationConfig": WeeToddH3GenerationConfig,
    "WeeToddH3LowMemoryTuning": WeeToddH3LowMemoryTuning,
    "WeeToddH3Generate": WeeToddH3Generate,
    "WeeToddH3Unload": WeeToddH3Unload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddH3ComponentLoader": "WeeTodd H3 Component Loader",
    "WeeToddH3PreviewOverride": "WeeTodd H3 Model Preview Override",
    "WeeToddH3QuantizedTransformerLoader": "WeeTodd H3 Quantized Transformer Loader",
    "WeeToddH3Preflight": "WeeTodd H3 Component Preflight",
    "WeeToddH3TokenBudget": "WeeTodd H3 Token + Memory Budget",
    "WeeToddH3FirstFrame": "WeeTodd H3 First Frame",
    "WeeToddH3LastFrame": "WeeTodd H3 Last Frame",
    "WeeToddH3FirstLastFrame": "WeeTodd H3 First + Last Frame",
    "WeeToddH3ChainedTimeline": "WeeTodd H3 Chained Timeline",
    "WeeToddH3Frames": "WeeTodd H3 Frames",
    "WeeToddH3TimedKeyframe": "WeeTodd H3 Timed Keyframe",
    "WeeToddH3ReferenceImage": "WeeTodd H3 Reference Image",
    "WeeToddH3ReferenceVideo": "WeeTodd H3 Reference Video",
    "WeeToddH3ReferenceAudio": "WeeTodd H3 Reference Audio",
    "WeeToddH3TimelineVisualGuide": "WeeTodd H3 Timeline Visual Guide",
    "WeeToddH3TimelineAudioGuide": "WeeTodd H3 Timeline Audio Guide",
    "WeeToddH3KeyframeEncode": "WeeTodd H3 Encode First / Last Frames",
    "WeeToddH3TimedKeyframeEncode": "WeeTodd H3 Encode Timed Keyframes",
    "WeeToddH3ReferenceEncode": "WeeTodd H3 Encode References",
    "WeeToddH3ReferenceStrength": "WeeTodd H3 Reference Strength",
    "WeeToddH3TextEncode": "WeeTodd H3 Text Encode (Qwen3-VL)",
    "WeeToddH3UnloadTextEncoder": "WeeTodd H3 Unload Qwen3-VL",
    "WeeToddH3ContinuationContext": "WeeTodd H3 Motion Continuation Context",
    "WeeToddH3ChainAppend": "WeeTodd H3 Append Latent Chain Window",
    "WeeToddH3Sample": "WeeTodd H3 Sample Video + Audio Latents",
    "WeeToddH3LearnedLatentUpscalerLoader": (
        "WeeTodd H3 Learned Latent Upscaler Loader (MLX)"
    ),
    "WeeToddH3LatentHiresFix": "WeeTodd H3 Latent Hi Res Fix",
    "WeeToddH3LoRALoader": "WeeTodd H3 LoRA Loader (MLX)",
    "WeeToddH3VDNCheckpoint": "WeeTodd H3 VDN Checkpoint (MLX)",
    "WeeToddH3ValidatedSamplingPreset": "WeeTodd H3 Validated Sampling Preset",
    "WeeToddH3FastH3ProductionProfile": "WeeTodd H3 FastH3 Production Profile",
    "WeeToddH3FastVideoApproximation": (
        "WeeTodd H3 FastVideo Approximation (MLX Experimental)"
    ),
    "WeeToddH3SolAttention": "WeeTodd H3 Sparse Attention (MLX Experimental)",
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
    "WeeToddH3DirectPublishChain": "WeeTodd H3 Direct Publish Chained Timeline (MLX)",
    "WeeToddH3ModelLoader": "WeeTodd H3 Model Loader (MLX)",
    "WeeToddH3GenerationConfig": "WeeTodd H3 Generation Config",
    "WeeToddH3LowMemoryTuning": "WeeTodd H3 Low-Memory Tuning (MLX)",
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

# LTX 2.5 uses its own split-component contract and never aliases LTX 2.3
# checkpoints or Gemma 3 state. Imports remain lightweight until execution.
from .ltx25_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as LTX25_NODE_CLASS_MAPPINGS,
)
from .ltx25_nodes import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as LTX25_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(LTX25_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LTX25_NODE_DISPLAY_NAME_MAPPINGS)

# Control preprocessors use a separate MLX execution layer so their models and
# intermediate state never become part of either video-generation runtime.
from .control_preprocessors import (  # noqa: E402
    NODE_CLASS_MAPPINGS as CONTROL_PREPROCESSOR_NODE_CLASS_MAPPINGS,
)
from .control_preprocessors import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as CONTROL_PREPROCESSOR_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(CONTROL_PREPROCESSOR_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROL_PREPROCESSOR_NODE_DISPLAY_NAME_MAPPINGS)

from .h3_controlnet import (  # noqa: E402
    NODE_CLASS_MAPPINGS as H3_CONTROLNET_NODE_CLASS_MAPPINGS,
)
from .h3_controlnet import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as H3_CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(H3_CONTROLNET_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(H3_CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS)

# CorridorKey is a separately licensed, optional MLX media engine. This import
# registers only the lightweight Apache-2.0 adapter and mask utilities.
from .corridorkey_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as CORRIDORKEY_NODE_CLASS_MAPPINGS,
)
from .corridorkey_nodes import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as CORRIDORKEY_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(CORRIDORKEY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CORRIDORKEY_NODE_DISPLAY_NAME_MAPPINGS)

# Florence-2 uses the existing MLX-VLM dependency as an optional, independently
# unloadable text-grounding provider for standard ComfyUI masks.
from .florence_mask_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as FLORENCE_MASK_NODE_CLASS_MAPPINGS,
)
from .florence_mask_nodes import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as FLORENCE_MASK_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(FLORENCE_MASK_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FLORENCE_MASK_NODE_DISPLAY_NAME_MAPPINGS)

# Motion refinement runs the same isolated renderer used by Studio and headless jobs.
from .motion_nodes import (  # noqa: E402
    NODE_CLASS_MAPPINGS as MOTION_NODE_CLASS_MAPPINGS,
)
from .motion_nodes import (  # noqa: E402
    NODE_DISPLAY_NAME_MAPPINGS as MOTION_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(MOTION_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MOTION_NODE_DISPLAY_NAME_MAPPINGS)
