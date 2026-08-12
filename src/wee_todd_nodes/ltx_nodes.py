"""ComfyUI adapters for optional standalone LTX 2.3 MLX pipelines."""

from __future__ import annotations

import json
import os
import platform
import secrets
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ltx23_mlx.runtime import RUNTIME, LTX23GenerationConfig, LTX23ModelSpec
from ltx23_mlx.upscale import LTX23UpscalerSpec, upscale_video_to_file

from .publishing import _available_target


def _register_ltx_model_folder() -> None:
    try:
        import folder_paths

        path = Path(folder_paths.models_dir) / "LTX-2.3"
        add_path = getattr(folder_paths, "add_model_folder_path", None)
        if add_path is not None:
            add_path("ltx2", str(path), is_default=True)
    except ImportError:
        pass


def _resolve_ltx_model_root(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    if ".." in path.parts:
        raise ValueError("Relative LTX model paths cannot contain '..'.")
    try:
        import folder_paths

        _register_ltx_model_folder()
        roots = []
        seen = set()

        def add_root(root_value) -> None:
            root = Path(root_value).expanduser()
            if str(root) not in seen:
                seen.add(str(root))
                roots.append(root)

        for category in ("ltx2", "diffusers", "checkpoints", "diffusion_models"):
            try:
                for root in folder_paths.get_folder_paths(category):
                    add_root(root)
            except KeyError:
                continue
        models_dir = Path(folder_paths.models_dir)
        add_root(models_dir)
        candidates = [root / path for root in roots]
        if path.parts and path.parts[0].lower() in {"ltx-2.3", "ltx2.3", "ltx2"}:
            tail = Path(*path.parts[1:])
            candidates.extend(root / tail for root in roots)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return models_dir / path
    except ImportError:
        return path


def _resolve_gemma_root(value: str) -> str:
    """Resolve explicit Comfy model paths while leaving cached repo IDs intact."""
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)
    if ".." in path.parts:
        raise ValueError("Relative Gemma model paths cannot contain '..'.")
    try:
        import folder_paths

        roots = []
        for category in ("text_encoders", "clip", "LLM", "llm"):
            try:
                roots.extend(Path(root) for root in folder_paths.get_folder_paths(category))
            except KeyError:
                continue
        roots.append(Path(folder_paths.models_dir))
        for root in roots:
            candidate = root / path
            if candidate.is_dir():
                return str(candidate)
    except ImportError:
        pass
    return value


def _output_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path.cwd() / "output"


def _safe_target(prefix: str, seed: int) -> Path:
    relative = Path(prefix.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ValueError("filename_prefix must stay inside ComfyUI's output directory.")
    root = _output_directory().resolve()
    target = (root / relative.parent / f"{relative.name}_{seed}.mp4").resolve()
    if root not in target.parents:
        raise ValueError("filename_prefix resolves outside ComfyUI's output directory.")
    return _available_target(target)


def _preview(path: Path) -> dict[str, str]:
    relative = path.resolve().relative_to(_output_directory().resolve())
    return {
        "filename": relative.name,
        "subfolder": str(relative.parent) if str(relative.parent) != "." else "",
        "type": "output",
        "format": "video/mp4",
    }


def _check_interrupted():
    try:
        import comfy.model_management

        return comfy.model_management.throw_exception_if_processing_interrupted
    except ImportError:
        return None


def _comfy_progress(total: int):
    try:
        import comfy.utils

        progress = comfy.utils.ProgressBar(total)
    except ImportError:
        return None

    def update(completed, reported_total):
        progress.update_absolute(completed, reported_total)

    return update


def _release_h3_stages() -> list[str]:
    from .conditioning import TEXT_ENCODER_RUNTIME
    from .decoding import AUDIO_VAE_RUNTIME, VIDEO_VAE_RUNTIME
    from .sampling import TRANSFORMER_RUNTIME

    released = []
    for name, cache in (
        ("text_encoder", TEXT_ENCODER_RUNTIME),
        ("transformer", TRANSFORMER_RUNTIME),
        ("video_vae", VIDEO_VAE_RUNTIME),
        ("audio_vae", AUDIO_VAE_RUNTIME),
    ):
        if getattr(cache, "loaded", False):
            cache.unload()
            released.append(name)
    try:
        from .runtime import RUNTIME as h3_runtime

        if h3_runtime.loaded:
            h3_runtime.unload()
            released.append("legacy_h3_pipeline")
    except ImportError:
        pass
    return released


def _software_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package, label in (
        ("mlx", "mlx"),
        ("ltx-pipelines-mlx", "ltx_pipelines_mlx"),
        ("comfyui-weetodd-nodes", "weetodd_nodes"),
    ):
        try:
            values[label] = version(package)
        except PackageNotFoundError:
            values[label] = "uninstalled"
    return values


class WeeToddLTX23ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        _register_ltx_model_folder()
        return {
            "required": {
                "model_directory": ("STRING", {"default": "LTX-2.3/q8"}),
                "gemma_model": (
                    "STRING",
                    {"default": "mlx-community/gemma-3-12b-it-4bit"},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "specify"
    CATEGORY = "WeeTodd/LTX 2.3/loaders"
    DESCRIPTION = "Select a local LTX 2.3 MLX bundle. No weights load in this node."

    def specify(self, model_directory, gemma_model):
        return (
            LTX23ModelSpec(
                str(_resolve_ltx_model_root(model_directory)),
                _resolve_gemma_root(gemma_model),
            ),
        )


class WeeToddLTX23GenerationConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline_mode": (
                    ["two_stage", "two_stage_hq", "distilled", "one_stage"],
                    {"default": "two_stage"},
                ),
                "width": ("INT", {"default": 704, "min": 64, "max": 1920, "step": 32}),
                "height": ("INT", {"default": 448, "min": 64, "max": 1920, "step": 32}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.25, "max": 30.0, "step": 0.25},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0},
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "stage1_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Zero selects the recommended count for the pipeline mode.",
                    },
                ),
                "stage2_steps": (
                    "INT",
                    {"default": 0, "min": 0, "max": 20, "tooltip": "Zero selects 3 steps."},
                ),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "low_memory": ("BOOLEAN", {"default": True}),
                "low_ram_streaming": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_CONFIG", "STRING")
    RETURN_NAMES = ("config", "resolved_settings")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/LTX 2.3"

    def configure(
        self,
        pipeline_mode,
        width,
        height,
        duration_seconds,
        frame_rate,
        seed,
        stage1_steps,
        stage2_steps,
        cfg_scale,
        stg_scale,
        low_memory,
        low_ram_streaming,
    ):
        recommended = {"two_stage": 30, "two_stage_hq": 15, "distilled": 8, "one_stage": 30}
        resolved_seed = secrets.randbelow(0x80000000) if seed < 0 else seed
        config = LTX23GenerationConfig(
            pipeline_mode=pipeline_mode,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            frame_rate=frame_rate,
            seed=resolved_seed,
            stage1_steps=stage1_steps or recommended[pipeline_mode],
            stage2_steps=stage2_steps or 3,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )
        config.validate()
        info = {
            **asdict(config),
            "num_frames": config.num_frames,
            "delivered_duration_seconds": config.delivered_duration_seconds,
        }
        return config, json.dumps(info, indent=2, sort_keys=True)


class WeeToddLTX23Preflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX23_MODEL",),
                "config": ("WEETODD_LTX23_CONFIG",),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_MODEL", "STRING")
    RETURN_NAMES = ("model", "preflight_report")
    FUNCTION = "check"
    CATEGORY = "WeeTodd/LTX 2.3/loaders"

    def check(self, model, config):
        config.validate()
        model.validate(config.pipeline_mode)
        inventory = model.inventory(config.pipeline_mode)
        report = {
            "ok": True,
            "pipeline_mode": config.pipeline_mode,
            "canvas": [config.width, config.height],
            "num_frames": config.num_frames,
            "model_directory": model.root().name,
            **inventory,
        }
        return model, json.dumps(report, indent=2, sort_keys=True)


class WeeToddLTX23Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX23_MODEL",),
                "config": ("WEETODD_LTX23_CONFIG",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "filename_prefix": ("STRING", {"default": "WeeTodd/LTX23"}),
                "unload_after_generate": ("BOOLEAN", {"default": True}),
            },
            "optional": {"first_frame": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/LTX 2.3"
    DESCRIPTION = "Generate synchronized LTX 2.3 video and 48 kHz stereo audio through MLX."

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

        config.validate()
        model.validate(config.pipeline_mode)
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
                    raise ValueError("LTX first frame must be an RGB ComfyUI IMAGE.")
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
                step_callback=_comfy_progress(
                    config.stage1_steps
                    + (0 if config.pipeline_mode == "one_stage" else config.stage2_steps)
                ),
            )
            if not partial.is_file() or partial.stat().st_size == 0:
                raise RuntimeError("LTX 2.3 pipeline did not produce a video file.")
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


class WeeToddLTX23UpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX23_MODEL",),
                "upscaler_name": (
                    [
                        "spatial_upscaler_x2_v1_1",
                        "spatial_upscaler_x1_5_v1_0",
                    ],
                    {"default": "spatial_upscaler_x2_v1_1"},
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_UPSCALER", "STRING")
    RETURN_NAMES = ("upscaler", "upscaler_info")
    FUNCTION = "load"
    CATEGORY = "WeeTodd/LTX 2.3/loaders"
    DESCRIPTION = "Select and preflight a learned LTX 2.3 spatial latent upscaler."

    def load(self, model, upscaler_name):
        spec = LTX23UpscalerSpec(str(model.root()), upscaler_name)
        config = spec.validate()
        info = {
            "upscaler": upscaler_name,
            "spatial_scale": float(config.get("spatial_scale", 2.0)),
            "weights_bytes": spec.weights_path.stat().st_size,
            "loads_at_execution": True,
        }
        return spec, json.dumps(info, indent=2, sort_keys=True)


class WeeToddLTX23UpscalePublish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaler": ("WEETODD_LTX23_UPSCALER",),
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "WeeTodd/LTX23_upscaled"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "max_av_drift_seconds": (
                    "FLOAT",
                    {"default": 0.025, "min": 0.0, "max": 0.25, "step": 0.001},
                ),
            },
            "optional": {"generation_metadata": ("STRING", {"default": "{}", "multiline": True})},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "upscale_info")
    OUTPUT_NODE = True
    FUNCTION = "upscale"
    CATEGORY = "WeeTodd/LTX 2.3/upscale"
    DESCRIPTION = (
        "Upscale decoded H3 or other ComfyUI video frames with the LTX latent "
        "upscaler and preserve the supplied audio."
    )

    def upscale(
        self,
        upscaler,
        images,
        audio,
        filename_prefix,
        fps,
        max_av_drift_seconds,
        generation_metadata="{}",
    ):
        try:
            supplied = json.loads(generation_metadata or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Generation metadata must be a JSON object.") from exc
        if not isinstance(supplied, dict):
            raise ValueError("Generation metadata must be a JSON object.")
        _release_h3_stages()
        target = _safe_target(filename_prefix, 0)
        result = upscale_video_to_file(
            upscaler,
            images,
            audio,
            target,
            fps=fps,
            max_av_drift_seconds=max_av_drift_seconds,
            generation_metadata={**supplied, "software": _software_versions()},
            check_interrupted=_check_interrupted(),
        )
        return {
            "ui": {"gifs": [_preview(result.video_path)]},
            "result": (
                str(result.video_path),
                json.dumps(result.metadata, indent=2, sort_keys=True),
            ),
        }


class WeeToddLTX23Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/LTX 2.3"

    def release(self, unload):
        if unload:
            RUNTIME.unload()
            return ("LTX 2.3 MLX runtime unloaded",)
        return ("LTX 2.3 MLX runtime kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddLTX23ModelLoader": WeeToddLTX23ModelLoader,
    "WeeToddLTX23GenerationConfig": WeeToddLTX23GenerationConfig,
    "WeeToddLTX23Preflight": WeeToddLTX23Preflight,
    "WeeToddLTX23Generate": WeeToddLTX23Generate,
    "WeeToddLTX23UpscalerLoader": WeeToddLTX23UpscalerLoader,
    "WeeToddLTX23UpscalePublish": WeeToddLTX23UpscalePublish,
    "WeeToddLTX23Unload": WeeToddLTX23Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddLTX23ModelLoader": "WeeTodd LTX 2.3 Model Loader (MLX)",
    "WeeToddLTX23GenerationConfig": "WeeTodd LTX 2.3 Generation Config",
    "WeeToddLTX23Preflight": "WeeTodd LTX 2.3 Preflight",
    "WeeToddLTX23Generate": "WeeTodd LTX 2.3 Generate Video + Audio",
    "WeeToddLTX23UpscalerLoader": "WeeTodd LTX 2.3 Upscaler Loader",
    "WeeToddLTX23UpscalePublish": "WeeTodd LTX 2.3 Upscale + Publish",
    "WeeToddLTX23Unload": "WeeTodd LTX 2.3 Unload MLX Runtime",
}
