"""ComfyUI adapters for the native MLX MiniMax-H3 Fun ControlNet-Union path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conditioning_inputs import _comfy_frames_to_uint8
from .decoding import VIDEO_VAE_RUNTIME, H3VideoVAESpec
from .residency import prepare_low_memory_stage


@dataclass(frozen=True)
class H3FunControlSpec:
    checkpoint: str
    strength: float = 1.0

    def validate(self) -> Path:
        path = Path(self.checkpoint).expanduser()
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise FileNotFoundError(f"H3 Fun ControlNet checkpoint not found: {path}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("H3 Fun ControlNet strength must be between 0 and 1")
        return path


@dataclass(frozen=True)
class H3PreparedFunControl:
    spec: H3FunControlSpec
    latent: Any
    source_frames: int
    target_frames: int


def inspect_h3_fun_control_header(path: str | Path) -> dict[str, object]:
    """Validate the published five-block architecture without mapping its 6.8 GB payload."""
    from .preflight import read_safetensors_header

    header = read_safetensors_header(path)
    prefixes = ("model.diffusion_model.", "diffusion_model.", "controlnet.")

    def normalized(name: str) -> str:
        for prefix in prefixes:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    normalized_to_source = {normalized(name): name for name in header.tensor_names}
    if len(normalized_to_source) != len(header.tensor_names):
        raise ValueError("H3 Fun ControlNet tensor names collide after prefix normalization")
    names = set(normalized_to_source)
    required = {
        "control_proj_in.weight",
        "control_proj_in.bias",
        "control_blocks.0.before_proj.weight",
        "control_blocks.0.adaln_proj.linear.weight",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"H3 Fun ControlNet header is missing required tensors: {missing}")
    blocks = sorted(
        {
            int(name.split(".", 2)[1])
            for name in names
            if name.startswith("control_blocks.") and name.split(".", 2)[1].isdigit()
        }
    )
    if blocks != [0, 1, 2, 3, 4]:
        raise ValueError(f"H3 Fun ControlNet must contain blocks 0..4, found {blocks}")
    for index in blocks:
        if f"control_blocks.{index}.after_proj.weight" not in names:
            raise ValueError(f"H3 Fun ControlNet block {index} has no after projection")
        raw_q = f"control_blocks.{index}.attn.to_q.weight"
        native_qkv = f"control_blocks.{index}.attn.qkv_proj.weight"
        if raw_q not in names and native_qkv not in names:
            raise ValueError(f"H3 Fun ControlNet block {index} has no attention projection")
    proj_shape = header.tensor_shapes[normalized_to_source["control_proj_in.weight"]]
    if len(proj_shape) != 2 or proj_shape[1] % 4 or proj_shape[1] // 4 != 49:
        raise ValueError(
            "H3 Fun ControlNet input projection must declare control_in_dim=49 at patch 1x2x2"
        )
    return {
        "tensor_count": header.tensor_count,
        "tensor_bytes": header.tensor_bytes,
        "dtypes": list(header.dtypes),
        "control_in_dim": 49,
        "hidden_size": int(proj_shape[0]),
        "injection_layers": [0, 10, 20, 30, 40],
        "checkpoint_layout": (
            "videox_fun_split_qkv"
            if "control_blocks.0.attn.to_q.weight" in names
            else "weetodd_native_qkv"
        ),
    }


def _controlnet_names():
    try:
        import folder_paths

        names = [
            name
            for name in folder_paths.get_filename_list("controlnet")
            if name.lower().endswith(".safetensors")
            and "minimax" in name.lower()
            and ("fun" in name.lower() or "h3" in name.lower())
        ]
        if names:
            return names
    except (ImportError, KeyError):
        pass
    return ["MiniMax-H3-Fun-Controlnet-Union.safetensors"]


def _resolve_controlnet(name: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    try:
        import folder_paths

        resolved = folder_paths.get_full_path("controlnet", name)
        if resolved:
            return Path(resolved).resolve()
    except (ImportError, KeyError):
        pass
    raise FileNotFoundError(
        f"MiniMax-H3 Fun ControlNet {name!r} was not found in a configured controlnet folder"
    )


def prepare_h3_control_frames(frames, *, target_frames: int, height: int, width: int):
    """Hold/trim time and cover-crop control pixels to the exact H3 generation canvas."""
    import numpy as np
    from PIL import Image

    source = np.asarray(frames)
    if source.ndim != 4 or source.shape[-1] < 3 or source.shape[0] < 1:
        raise ValueError("H3 control video must have shape (frames, height, width, channels)")
    if source.dtype != np.uint8:
        finite = np.nan_to_num(source, nan=0.0, posinf=1.0, neginf=0.0)
        if float(finite.max(initial=0.0)) <= 1.0:
            finite = finite * 255.0
        source = np.rint(np.clip(finite, 0.0, 255.0)).astype(np.uint8)
    indices = np.minimum(np.arange(target_frames), source.shape[0] - 1)
    selected = source[indices, ..., :3]
    prepared = []
    for frame in selected:
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (max(width, round(image.width * scale)), max(height, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        prepared.append(np.asarray(resized.crop((left, top, left + width, top + height))))
    return np.ascontiguousarray(prepared, dtype=np.uint8)


class WeeToddH3FunControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_controlnet_names(),),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "1.0 is the publisher's recommended strongest guidance.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_FUN_CONTROL_SPEC", "STRING")
    RETURN_NAMES = ("controlnet", "controlnet_info")
    FUNCTION = "load"
    CATEGORY = "WeeTodd/H3/control"
    DESCRIPTION = (
        "Select the 5-block MiniMax-H3 Fun ControlNet-Union branch. Weights remain deferred "
        "until sampling. One checkpoint supports Canny, depth, HED, MLSD, and pose guides."
    )

    def load(self, checkpoint, strength):
        spec = H3FunControlSpec(str(_resolve_controlnet(checkpoint)), float(strength))
        path = spec.validate()
        header = inspect_h3_fun_control_header(path)
        return spec, json.dumps(
            {
                "checkpoint": path.name,
                "strength": spec.strength,
                **header,
                "guidance_scale": 1.0,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddH3FunControlEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("WEETODD_H3_FUN_CONTROL_SPEC",),
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "control_video": ("IMAGE",),
                "unload_after_encode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WEETODD_H3_FUN_CONTROL", "STRING")
    RETURN_NAMES = ("control", "control_info")
    FUNCTION = "encode"
    CATEGORY = "WeeTodd/H3/control"
    DESCRIPTION = (
        "Fit a preprocessed control video to the H3 canvas and encode it with the selected H3 "
        "video VAE. Short guides hold their final frame; long guides are trimmed."
    )

    def encode(self, controlnet, components, config, control_video, unload_after_encode):
        from minimax_h3_mlx.packing import align_num_frames

        if components.task != "t2va":
            raise ValueError("H3 Fun ControlNet requires components.task=t2va")
        config.validate()
        source = _comfy_frames_to_uint8(control_video, "H3 Fun control video")
        target_frames = align_num_frames(round(config.duration_seconds * 24))
        prepared = prepare_h3_control_frames(
            source,
            target_frames=target_frames,
            height=config.height,
            width=config.width,
        )
        staged_releases = ()

        def prepare_stage():
            nonlocal staged_releases
            staged_releases = prepare_low_memory_stage("video_vae", config.memory_mode)

        latent = VIDEO_VAE_RUNTIME.encode_continuation(
            H3VideoVAESpec.from_components(components),
            prepared,
            unload_after=unload_after_encode or config.memory_mode == "low_memory_bf16",
            prepare_stage=prepare_stage,
        )
        control = H3PreparedFunControl(
            spec=controlnet,
            latent=latent,
            source_frames=int(source.shape[0]),
            target_frames=target_frames,
        )
        return control, json.dumps(
            {
                "source_frames": control.source_frames,
                "target_frames": control.target_frames,
                "canvas": [config.width, config.height],
                "latent_shape": list(latent.shape),
                "staged_releases": list(staged_releases),
            },
            indent=2,
            sort_keys=True,
        )


NODE_CLASS_MAPPINGS = {
    "WeeToddH3FunControlNetLoader": WeeToddH3FunControlNetLoader,
    "WeeToddH3FunControlEncode": WeeToddH3FunControlEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddH3FunControlNetLoader": "WeeTodd H3 Fun ControlNet-Union Loader (MLX)",
    "WeeToddH3FunControlEncode": "WeeTodd H3 Encode Fun Control Video (MLX)",
}
