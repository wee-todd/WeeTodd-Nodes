"""ComfyUI adapters for MLX-native control-guide preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _ensure_annotators_category(folder_paths):
    category = "annotators"
    if category not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(category, str(Path(folder_paths.models_dir) / category))
    return category


def _model_filenames():
    try:
        import folder_paths

        category = _ensure_annotators_category(folder_paths)
        names = [
            name
            for name in folder_paths.get_filename_list(category)
            if name.endswith(".safetensors") and "video_depth_anything" in name.lower()
        ]
        return names or ["video_depth_anything/video_depth_anything_vits_mlx.safetensors"]
    except (ImportError, KeyError):
        return ["video_depth_anything/video_depth_anything_vits_mlx.safetensors"]


def _fast_depth_model_names():
    return _model_safetensor_names(
        "depth_anything_v2", "depth_anything_v2/Depth-Anything-V2-Small-hf/model.safetensors"
    )


def _resolve_model(name: str) -> Path:
    try:
        import folder_paths

        category = _ensure_annotators_category(folder_paths)
        value = folder_paths.get_full_path(category, name)
        if value is not None:
            return Path(value)
    except (ImportError, KeyError):
        pass
    candidate = Path(name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"MLX preprocessor checkpoint {name!r} was not found under a configured annotators "
        "model folder. Place or convert the required checkpoint, then refresh ComfyUI."
    )


def _model_bundle_names(kind: str):
    try:
        import folder_paths

        category = _ensure_annotators_category(folder_paths)
        names = []
        for root in folder_paths.get_folder_paths(category):
            root = Path(root)
            for graph in root.rglob("graph.json"):
                relative = graph.parent.relative_to(root).as_posix()
                if kind in relative.lower():
                    names.append(relative)
        if names:
            return sorted(set(names))
    except (ImportError, KeyError, OSError, ValueError):
        pass
    fallback = "yolox_l" if kind == "yolox" else "dw-ll_ucoco_384"
    return [f"dwpose_mlx/{fallback}"]


def _model_safetensor_names(kind: str, fallback: str):
    try:
        import folder_paths

        category = _ensure_annotators_category(folder_paths)
        names = [
            name
            for name in folder_paths.get_filename_list(category)
            if name.endswith(".safetensors") and kind in name.lower()
        ]
        if names:
            return names
    except (ImportError, KeyError):
        pass
    return [fallback]


def _resolve_model_bundle(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() and (candidate / "graph.json").is_file():
        return candidate
    try:
        import folder_paths

        category = _ensure_annotators_category(folder_paths)
        for root in folder_paths.get_folder_paths(category):
            candidate = Path(root) / name
            if (candidate / "graph.json").is_file() and (
                candidate / "weights.safetensors"
            ).is_file():
                return candidate
    except (ImportError, KeyError):
        pass
    raise FileNotFoundError(
        f"Converted MLX preprocessor bundle {name!r} was not found under a configured "
        "annotators model folder."
    )


@dataclass(frozen=True)
class MLXVideoDepthModelSpec:
    checkpoint_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class MLXDWPoseModelSpec:
    detector_name: str
    detector_path: Path
    pose_name: str
    pose_path: Path


@dataclass(frozen=True)
class MLXTEEDModelSpec:
    checkpoint_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class MLXFastDepthModelSpec:
    checkpoint_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class MLXLineArtModelSpec:
    checkpoint_name: str
    checkpoint_path: Path


@dataclass(frozen=True)
class LTX25CrossViewCameraConfig:
    """Serializable camera controls shared by the UI and headless workflows."""

    azimuth: float
    elevation: float
    distance: float
    horizontal_fov: float
    vertical_shift: float
    depth_ratio: float
    invert_depth: bool
    keep_source_aim: bool
    camera_keyframes: str
    interpolation: str
    pivot_x: float
    pivot_y: float
    pivot_z: float
    splat_radius: int
    path_id: str

    def warp_kwargs(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "azimuth",
                "elevation",
                "distance",
                "horizontal_fov",
                "vertical_shift",
                "depth_ratio",
                "invert_depth",
                "keep_source_aim",
                "camera_keyframes",
                "interpolation",
                "pivot_x",
                "pivot_y",
                "pivot_z",
                "splat_radius",
            )
        }


class _VideoDepthRuntime:
    def __init__(self):
        self.model = None
        self.checkpoint_path = None
        self.precision = None

    def load(self, spec: MLXVideoDepthModelSpec, precision: str):
        if (
            self.model is not None
            and self.checkpoint_path == spec.checkpoint_path
            and self.precision == precision
        ):
            return self.model, False
        self.unload()
        from mlx_preprocessors.video_depth import load_video_depth_anything_small

        self.model = load_video_depth_anything_small(spec.checkpoint_path)
        if precision == "bfloat16 balanced":
            import mlx.core as mx

            self.model.set_dtype(mx.bfloat16)
        self.checkpoint_path = spec.checkpoint_path
        self.precision = precision
        return self.model, True

    def unload(self):
        import gc

        self.model = None
        self.checkpoint_path = None
        self.precision = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


VIDEO_DEPTH_RUNTIME = _VideoDepthRuntime()


class _DWPoseRuntime:
    def __init__(self):
        self.model = None
        self.identity = None

    def load(self, spec: MLXDWPoseModelSpec):
        identity = (spec.detector_path, spec.pose_path)
        if self.model is not None and self.identity == identity:
            return self.model, False
        self.unload()
        from mlx_preprocessors.dwpose import DWPoseMLX

        self.model = DWPoseMLX(spec.detector_path, spec.pose_path)
        self.identity = identity
        return self.model, True

    def unload(self):
        import gc

        self.model = None
        self.identity = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


DWPOSE_RUNTIME = _DWPoseRuntime()


class _TEEDRuntime:
    def __init__(self):
        self.model = None
        self.checkpoint_path = None

    def load(self, spec: MLXTEEDModelSpec):
        if self.model is not None and self.checkpoint_path == spec.checkpoint_path:
            return self.model, False
        self.unload()
        from mlx_preprocessors.teed import load_teed

        self.model = load_teed(spec.checkpoint_path)
        self.checkpoint_path = spec.checkpoint_path
        return self.model, True

    def unload(self):
        import gc

        self.model = None
        self.checkpoint_path = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


TEED_RUNTIME = _TEEDRuntime()


class _FastDepthRuntime:
    def __init__(self):
        self.model = None
        self.checkpoint_path = None
        self.precision = None

    def load(self, spec: MLXFastDepthModelSpec, precision: str):
        if (
            self.model is not None
            and self.checkpoint_path == spec.checkpoint_path
            and self.precision == precision
        ):
            return self.model, False
        self.unload()
        from mlx_preprocessors.fast_depth import load_depth_anything_v2_small

        self.model = load_depth_anything_v2_small(spec.checkpoint_path)
        if precision == "bfloat16 speed":
            import mlx.core as mx

            self.model.set_dtype(mx.bfloat16)
        self.checkpoint_path = spec.checkpoint_path
        self.precision = precision
        return self.model, True

    def unload(self):
        import gc

        self.model = None
        self.checkpoint_path = None
        self.precision = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


FAST_DEPTH_RUNTIME = _FastDepthRuntime()


class _LineArtRuntime:
    def __init__(self):
        self.model = None
        self.checkpoint_path = None

    def load(self, spec: MLXLineArtModelSpec):
        if self.model is not None and self.checkpoint_path == spec.checkpoint_path:
            return self.model, False
        self.unload()
        from mlx_preprocessors.lineart import load_realistic_lineart

        self.model = load_realistic_lineart(spec.checkpoint_path)
        self.checkpoint_path = spec.checkpoint_path
        return self.model, True

    def unload(self):
        import gc

        self.model = None
        self.checkpoint_path = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass


LINEART_RUNTIME = _LineArtRuntime()


class WeeToddMLXCannyPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "low_threshold": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "high_threshold": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "gaussian_kernel_size": ([3, 5, 7], {"default": 5}),
                "gaussian_sigma": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1},
                ),
                "hysteresis": ("BOOLEAN", {"default": True}),
                "frame_chunk_size": ([4, 8, 16, 32, 64], {"default": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edge_images", "preprocessor_info")
    FUNCTION = "detect"
    CATEGORY = "WeeTodd/MLX preprocessors/edges"
    DESCRIPTION = (
        "Create temporally aligned Canny control frames with MLX. The defaults match "
        "ComfyUI's current normalized-threshold Canny contract."
    )

    def detect(
        self,
        images,
        low_threshold,
        high_threshold,
        gaussian_kernel_size,
        gaussian_sigma,
        hysteresis,
        frame_chunk_size,
    ):
        from mlx_preprocessors import CannyConfig, canny_edges

        output, report = canny_edges(
            images,
            CannyConfig(
                low_threshold=float(low_threshold),
                high_threshold=float(high_threshold),
                gaussian_kernel_size=int(gaussian_kernel_size),
                gaussian_sigma=float(gaussian_sigma),
                hysteresis=bool(hysteresis),
                frame_chunk_size=int(frame_chunk_size),
            ),
        )
        try:
            import torch

            if isinstance(images, torch.Tensor):
                output = torch.from_numpy(output).to(
                    device=images.device,
                    dtype=images.dtype,
                )
        except ImportError:
            pass
        return output, json.dumps(report, indent=2, sort_keys=True)


class WeeToddMLXVideoDepthLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint": (_model_filenames(),)}}

    RETURN_TYPES = ("WEETODD_MLX_VIDEO_DEPTH_MODEL", "STRING")
    RETURN_NAMES = ("depth_model", "model_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/MLX preprocessors/depth"
    DESCRIPTION = (
        "Select a converted Apache-2.0 Video Depth Anything Small checkpoint. "
        "This node does not load weights."
    )

    def select(self, checkpoint):
        path = _resolve_model(str(checkpoint))
        spec = MLXVideoDepthModelSpec(str(checkpoint), path)
        return spec, json.dumps(
            {
                "architecture": "video_depth_anything_vits",
                "backend": "mlx",
                "checkpoint": str(checkpoint),
                "loaded": False,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddMLXVideoDepthPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_model": ("WEETODD_MLX_VIDEO_DEPTH_MODEL",),
                "input_size": ([392, 448, 518, 560, 644], {"default": 518}),
                "precision": (
                    ["float32 quality", "bfloat16 balanced"],
                    {"default": "float32 quality"},
                ),
                "encoder_chunk_size": ([1, 2, 4, 8, 16, 32], {"default": 4}),
                "decoder_chunk_size": ([1, 2, 4, 8, 16, 32], {"default": 4}),
                "depth_polarity": (
                    ["near white (recommended)", "near black"],
                    {"default": "near white (recommended)"},
                ),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_images", "preprocessor_info")
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/MLX preprocessors/depth"
    DESCRIPTION = (
        "Estimate temporally consistent relative depth with Video Depth Anything Small on MLX. "
        "The default unloads the model after preprocessing."
    )

    def estimate(
        self,
        images,
        depth_model,
        input_size,
        precision,
        encoder_chunk_size,
        decoder_chunk_size,
        depth_polarity,
        keep_model_warm,
    ):
        from mlx_preprocessors.video_depth import VideoDepthConfig, infer_video_depth

        def interruption():
            try:
                import comfy.model_management

                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def progress(current, total):
                progress_bar.update_absolute(current, total)

        except ImportError:
            pass

        model, loaded_now = VIDEO_DEPTH_RUNTIME.load(depth_model, str(precision))
        try:
            output, report = infer_video_depth(
                images,
                model,
                VideoDepthConfig(
                    input_size=int(input_size),
                    output_invert=str(depth_polarity) == "near black",
                    encoder_chunk_size=int(encoder_chunk_size),
                    decoder_chunk_size=int(decoder_chunk_size),
                ),
                progress_callback=progress,
                interruption_callback=interruption,
            )
            report.update(
                {
                    "checkpoint": depth_model.checkpoint_name,
                    "precision": str(precision),
                    "loaded_now": loaded_now,
                    "resident_after": bool(keep_model_warm),
                }
            )
            try:
                import torch

                if isinstance(images, torch.Tensor):
                    output = torch.from_numpy(output).to(device=images.device, dtype=images.dtype)
            except ImportError:
                pass
            return output, json.dumps(report, indent=2, sort_keys=True)
        finally:
            if not keep_model_warm:
                VIDEO_DEPTH_RUNTIME.unload()


class WeeToddMLXDWPoseLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detector_bundle": (_model_bundle_names("yolox"),),
                "pose_bundle": (_model_bundle_names("dw-ll"),),
            }
        }

    RETURN_TYPES = ("WEETODD_MLX_DWPOSE_MODEL", "STRING")
    RETURN_NAMES = ("pose_model", "model_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/MLX preprocessors/pose"
    DESCRIPTION = "Select converted YOLOX-L and DWPose whole-body MLX bundles without loading them."

    def select(self, detector_bundle, pose_bundle):
        spec = MLXDWPoseModelSpec(
            str(detector_bundle),
            _resolve_model_bundle(str(detector_bundle)),
            str(pose_bundle),
            _resolve_model_bundle(str(pose_bundle)),
        )
        return spec, json.dumps(
            {
                "backend": "mlx",
                "detector": spec.detector_name,
                "pose": spec.pose_name,
                "loaded": False,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddMLXDWPosePreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_model": ("WEETODD_MLX_DWPOSE_MODEL",),
                "render_mode": (
                    ["whole body", "body and hands", "body only"],
                    {"default": "whole body"},
                ),
                "detection_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "keypoint_threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_images", "preprocessor_info")
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/MLX preprocessors/pose"
    DESCRIPTION = (
        "Estimate whole-body pose with MLX YOLOX-L and DWPose. "
        "The default includes body, face, and hands, then unloads both models."
    )

    def estimate(
        self,
        images,
        pose_model,
        render_mode,
        detection_threshold,
        keypoint_threshold,
        keep_model_warm,
    ):
        from mlx_preprocessors.dwpose import DWPoseConfig, infer_dwpose

        def interruption():
            try:
                import comfy.model_management

                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def progress(current, total):
                progress_bar.update_absolute(current, total)

        except ImportError:
            pass

        model, loaded_now = DWPOSE_RUNTIME.load(pose_model)
        try:
            output, report = infer_dwpose(
                images,
                model,
                DWPoseConfig(
                    detection_threshold=float(detection_threshold),
                    keypoint_threshold=float(keypoint_threshold),
                    render_mode=str(render_mode),
                ),
                progress_callback=progress,
                interruption_callback=interruption,
            )
            report.update(
                {
                    "detector_bundle": pose_model.detector_name,
                    "pose_bundle": pose_model.pose_name,
                    "loaded_now": loaded_now,
                    "resident_after": bool(keep_model_warm),
                }
            )
            try:
                import torch

                if isinstance(images, torch.Tensor):
                    output = torch.from_numpy(output).to(device=images.device, dtype=images.dtype)
            except ImportError:
                pass
            return output, json.dumps(report, indent=2, sort_keys=True)
        finally:
            if not keep_model_warm:
                DWPOSE_RUNTIME.unload()


class WeeToddMLXTEEDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_model_safetensor_names("teed", "teed/7_model_mlx.safetensors"),)
            }
        }

    RETURN_TYPES = ("WEETODD_MLX_TEED_MODEL", "STRING")
    RETURN_NAMES = ("teed_model", "model_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/MLX preprocessors/edges"
    DESCRIPTION = "Select a converted MIT-licensed TEED checkpoint without loading it."

    def select(self, checkpoint):
        path = _resolve_model(str(checkpoint))
        spec = MLXTEEDModelSpec(str(checkpoint), path)
        return spec, json.dumps(
            {
                "architecture": "teed",
                "backend": "mlx",
                "checkpoint": spec.checkpoint_name,
                "loaded": False,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddMLXTEEDPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "teed_model": ("WEETODD_MLX_TEED_MODEL",),
                "safe_steps": ("INT", {"default": 2, "min": 0, "max": 10}),
                "frame_chunk_size": ([1, 2, 4, 8, 16, 32], {"default": 8}),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edge_images", "preprocessor_info")
    FUNCTION = "detect"
    CATEGORY = "WeeTodd/MLX preprocessors/edges"
    DESCRIPTION = (
        "Create learned soft-edge guides with the tiny TEED model on MLX. "
        "The default unloads the model after preprocessing."
    )

    def detect(self, images, teed_model, safe_steps, frame_chunk_size, keep_model_warm):
        from mlx_preprocessors.teed import TEEDConfig, infer_teed

        def interruption():
            try:
                import comfy.model_management

                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def progress(current, total):
                progress_bar.update_absolute(current, total)

        except ImportError:
            pass

        model, loaded_now = TEED_RUNTIME.load(teed_model)
        try:
            output, report = infer_teed(
                images,
                model,
                TEEDConfig(
                    safe_steps=int(safe_steps),
                    frame_chunk_size=int(frame_chunk_size),
                ),
                progress_callback=progress,
                interruption_callback=interruption,
            )
            report.update(
                {
                    "checkpoint": teed_model.checkpoint_name,
                    "loaded_now": loaded_now,
                    "resident_after": bool(keep_model_warm),
                }
            )
            try:
                import torch

                if isinstance(images, torch.Tensor):
                    output = torch.from_numpy(output).to(device=images.device, dtype=images.dtype)
            except ImportError:
                pass
            return output, json.dumps(report, indent=2, sort_keys=True)
        finally:
            if not keep_model_warm:
                TEED_RUNTIME.unload()


class WeeToddMLXFastDepthLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"checkpoint": (_fast_depth_model_names(),)}}

    RETURN_TYPES = ("WEETODD_MLX_FAST_DEPTH_MODEL", "STRING")
    RETURN_NAMES = ("depth_model", "model_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/MLX preprocessors/depth"
    DESCRIPTION = (
        "Select the standard Apache-2.0 Depth Anything V2 Small safetensors checkpoint. "
        "Weights are loaded directly into MLX only when preprocessing runs."
    )

    def select(self, checkpoint):
        path = _resolve_model(str(checkpoint))
        spec = MLXFastDepthModelSpec(str(checkpoint), path)
        return spec, json.dumps(
            {
                "architecture": "depth_anything_v2_small",
                "backend": "mlx",
                "checkpoint": spec.checkpoint_name,
                "loaded": False,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddMLXFastDepthPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "depth_model": ("WEETODD_MLX_FAST_DEPTH_MODEL",),
                "input_size": ([280, 336, 392, 448, 518], {"default": 392}),
                "precision": (
                    ["float32 quality", "bfloat16 speed"],
                    {"default": "bfloat16 speed"},
                ),
                "frame_chunk_size": ([1, 2, 4, 8, 16], {"default": 2}),
                "normalization": (["per clip", "per frame"], {"default": "per clip"}),
                "depth_polarity": (
                    ["near white (recommended)", "near black"],
                    {"default": "near white (recommended)"},
                ),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_images", "preprocessor_info")
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/MLX preprocessors/depth"
    DESCRIPTION = (
        "Estimate fast per-frame relative depth with Depth Anything V2 Small on MLX. "
        "Use Video Depth Anything when maximum temporal consistency matters."
    )

    def estimate(
        self,
        images,
        depth_model,
        input_size,
        precision,
        frame_chunk_size,
        normalization,
        depth_polarity,
        keep_model_warm,
    ):
        from mlx_preprocessors.fast_depth import FastDepthConfig, infer_fast_depth

        def interruption():
            try:
                import comfy.model_management

                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def progress(current, total):
                progress_bar.update_absolute(current, total)

        except ImportError:
            pass

        model, loaded_now = FAST_DEPTH_RUNTIME.load(depth_model, str(precision))
        try:
            output, report = infer_fast_depth(
                images,
                model,
                FastDepthConfig(
                    input_size=int(input_size),
                    frame_chunk_size=int(frame_chunk_size),
                    output_invert=str(depth_polarity) == "near black",
                    normalize=str(normalization),
                ),
                progress_callback=progress,
                interruption_callback=interruption,
            )
            report.update(
                {
                    "checkpoint": depth_model.checkpoint_name,
                    "precision": str(precision),
                    "loaded_now": loaded_now,
                    "resident_after": bool(keep_model_warm),
                }
            )
            try:
                import torch

                if isinstance(images, torch.Tensor):
                    output = torch.from_numpy(output).to(device=images.device, dtype=images.dtype)
            except ImportError:
                pass
            return output, json.dumps(report, indent=2, sort_keys=True)
        finally:
            if not keep_model_warm:
                FAST_DEPTH_RUNTIME.unload()


class WeeToddMLXNormalMapPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_images": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 40.0, "min": 0.1, "max": 400.0, "step": 1.0},
                ),
                "gradient_method": (["sobel", "central"], {"default": "sobel"}),
                "depth_polarity": (["near white", "near black"], {"default": "near white"}),
                "discontinuity_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "flip_y": ("BOOLEAN", {"default": False}),
                "frame_chunk_size": ([1, 2, 4, 8, 16, 32, 64], {"default": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("normal_images", "preprocessor_info")
    FUNCTION = "estimate"
    CATEGORY = "WeeTodd/MLX preprocessors/normals"
    DESCRIPTION = (
        "Convert a relative-depth IMAGE batch into standard RGB +Z-blue surface normals on MLX. "
        "Strength compensates for the small slopes in normalized depth. The node is weightless "
        "and preserves the input frame count and dimensions."
    )

    def estimate(
        self,
        depth_images,
        strength,
        gradient_method,
        depth_polarity,
        discontinuity_threshold,
        flip_y,
        frame_chunk_size,
    ):
        from mlx_preprocessors.normals import NormalMapConfig, depth_to_normals

        output, report = depth_to_normals(
            depth_images,
            NormalMapConfig(
                strength=float(strength),
                method=str(gradient_method),
                depth_polarity=str(depth_polarity),
                discontinuity_threshold=float(discontinuity_threshold),
                flip_y=bool(flip_y),
                frame_chunk_size=int(frame_chunk_size),
            ),
        )
        try:
            import torch

            if isinstance(depth_images, torch.Tensor):
                output = torch.from_numpy(output).to(
                    device=depth_images.device, dtype=depth_images.dtype
                )
        except ImportError:
            pass
        return output, json.dumps(report, indent=2, sort_keys=True)


class WeeToddMLXLineArtLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (
                    _model_safetensor_names(
                        "lineart", "lineart/realistic_lineart_fine_mlx.safetensors"
                    ),
                )
            }
        }

    RETURN_TYPES = ("WEETODD_MLX_LINEART_MODEL", "STRING")
    RETURN_NAMES = ("lineart_model", "model_info")
    FUNCTION = "select"
    CATEGORY = "WeeTodd/MLX preprocessors/line art"
    DESCRIPTION = (
        "Select a converted realistic fine or coarse line-art checkpoint without loading it."
    )

    def select(self, checkpoint):
        path = _resolve_model(str(checkpoint))
        spec = MLXLineArtModelSpec(str(checkpoint), path)
        return spec, json.dumps(
            {
                "architecture": "realistic_lineart",
                "backend": "mlx",
                "checkpoint": spec.checkpoint_name,
                "loaded": False,
            },
            indent=2,
            sort_keys=True,
        )


class WeeToddMLXLineArtPreprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lineart_model": ("WEETODD_MLX_LINEART_MODEL",),
                "detect_resolution": ([256, 384, 512, 640, 768, 1024], {"default": 512}),
                "output_mode": (["white lines", "black lines"], {"default": "white lines"}),
                "frame_chunk_size": ([1, 2, 4, 8, 16], {"default": 2}),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("lineart_images", "preprocessor_info")
    FUNCTION = "detect"
    CATEGORY = "WeeTodd/MLX preprocessors/line art"
    DESCRIPTION = (
        "Extract realistic fine or coarse line art with a compact MLX residual generator. "
        "The default matches ComfyUI's conventional white-line guide on black."
    )

    def detect(
        self,
        images,
        lineart_model,
        detect_resolution,
        output_mode,
        frame_chunk_size,
        keep_model_warm,
    ):
        from mlx_preprocessors.lineart import LineArtConfig, infer_realistic_lineart

        def interruption():
            try:
                import comfy.model_management

                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            import comfy.utils

            progress_bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def progress(current, total):
                progress_bar.update_absolute(current, total)

        except ImportError:
            pass

        model, loaded_now = LINEART_RUNTIME.load(lineart_model)
        try:
            output, report = infer_realistic_lineart(
                images,
                model,
                LineArtConfig(
                    detect_resolution=int(detect_resolution),
                    frame_chunk_size=int(frame_chunk_size),
                    output_mode=str(output_mode),
                ),
                progress_callback=progress,
                interruption_callback=interruption,
            )
            report.update(
                {
                    "checkpoint": lineart_model.checkpoint_name,
                    "loaded_now": loaded_now,
                    "resident_after": bool(keep_model_warm),
                }
            )
            try:
                import torch

                if isinstance(images, torch.Tensor):
                    output = torch.from_numpy(output).to(device=images.device, dtype=images.dtype)
            except ImportError:
                pass
            return output, json.dumps(report, indent=2, sort_keys=True)
        finally:
            if not keep_model_warm:
                LINEART_RUNTIME.unload()


class WeeToddOpticalFlowMotionTracks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_tracks": ("INT", {"default": 8, "min": 1, "max": 16, "step": 1}),
                "quality_level": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001},
                ),
                "minimum_distance": (
                    "INT",
                    {"default": 32, "min": 2, "max": 512, "step": 1},
                ),
                "window_size": ([15, 21, 31, 41, 51, 61], {"default": 21}),
                "pyramid_levels": ([1, 2, 3, 4, 5, 6], {"default": 3}),
                "forward_backward_limit": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.1, "max": 20.0, "step": 0.1},
                ),
                "trail_frames": (
                    "INT",
                    {"default": 50, "min": 1, "max": 200, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("motion_guide", "tracks_json", "tracking_info")
    FUNCTION = "track"
    CATEGORY = "WeeTodd/MLX preprocessors/motion"
    DESCRIPTION = (
        "Extract reliable sparse trajectories from an IMAGE batch with forward/backward "
        "optical flow, then render the LTX Motion Track training-color guide."
    )

    def track(
        self,
        images,
        max_tracks,
        quality_level,
        minimum_distance,
        window_size,
        pyramid_levels,
        forward_backward_limit,
        trail_frames,
    ):
        from mlx_preprocessors.motion_tracks import MotionTrackConfig, render_motion_tracks
        from mlx_preprocessors.optical_flow_tracks import (
            OpticalFlowTrackConfig,
            extract_optical_flow_tracks,
        )

        def interruption_callback():
            try:
                import comfy.model_management as model_management

                model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(max(1, int(images.shape[0]) - 1))
        except ImportError:
            pass
        tracks_json, tracking = extract_optical_flow_tracks(
            images,
            OpticalFlowTrackConfig(
                max_tracks=int(max_tracks),
                quality_level=float(quality_level),
                minimum_distance=int(minimum_distance),
                window_size=int(window_size),
                pyramid_levels=int(pyramid_levels),
                forward_backward_limit=float(forward_backward_limit),
            ),
            progress_callback=(
                (lambda current, total: progress.update_absolute(current, total))
                if progress is not None
                else None
            ),
            interruption_callback=interruption_callback,
        )
        guide, rendering = render_motion_tracks(
            tracks_json,
            MotionTrackConfig(
                width=int(images.shape[2]),
                height=int(images.shape[1]),
                num_frames=int(images.shape[0]),
                coordinate_space="normalized",
                track_format="per-frame coordinates",
                trail_frames=int(trail_frames),
            ),
            interruption_callback=interruption_callback,
        )
        try:
            import torch

            if isinstance(images, torch.Tensor):
                guide = torch.from_numpy(guide).to(device=images.device, dtype=images.dtype)
        except ImportError:
            pass
        return guide, tracks_json, json.dumps(
            {"tracking": tracking, "rendering": rendering}, indent=2, sort_keys=True
        )


class WeeToddMLXMotionTrackGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tracks_json": (
                    "STRING",
                    {
                        "default": (
                            '[[{"x":0.30,"y":0.55},{"x":0.50,"y":0.45},{"x":0.70,"y":0.55}]]'
                        ),
                        "multiline": True,
                        "tooltip": (
                            "A JSON list of tracks. Each track is a list of {x,y} points. "
                            "Normalized coordinates are portable across output sizes."
                        ),
                    },
                ),
                "width": ("INT", {"default": 768, "min": 64, "max": 1920, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1920, "step": 8}),
                "num_frames": (
                    "INT",
                    {"default": 121, "min": 2, "max": 1000, "step": 1},
                ),
                "coordinate_space": (
                    ["normalized", "pixels"],
                    {"default": "normalized"},
                ),
                "track_format": (
                    ["spline control points", "per-frame coordinates"],
                    {"default": "spline control points"},
                ),
                "trail_frames": (
                    "INT",
                    {"default": 50, "min": 1, "max": 200, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("motion_guide", "guide_info")
    FUNCTION = "render"
    CATEGORY = "WeeTodd/MLX preprocessors/motion"
    DESCRIPTION = (
        "Render sparse colored point trajectories into the guide-video representation expected "
        "by the LTX Motion Track IC-LoRA. This node uses MLX and does not require a checkpoint."
    )

    def render(
        self,
        tracks_json,
        width,
        height,
        num_frames,
        coordinate_space,
        track_format,
        trail_frames,
    ):
        from mlx_preprocessors.motion_tracks import MotionTrackConfig, render_motion_tracks

        def interruption_callback():
            try:
                import comfy.model_management as model_management

                model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(int(num_frames))
        except ImportError:
            pass

        output, report = render_motion_tracks(
            tracks_json,
            MotionTrackConfig(
                width=int(width),
                height=int(height),
                num_frames=int(num_frames),
                coordinate_space=str(coordinate_space),
                track_format=str(track_format),
                trail_frames=int(trail_frames),
            ),
            progress_callback=(
                (lambda current, _total: progress.update_absolute(current))
                if progress is not None
                else None
            ),
            interruption_callback=interruption_callback,
        )
        try:
            import torch

            output = torch.from_numpy(output)
        except ImportError:
            pass
        return output, json.dumps(report, indent=2, sort_keys=True)


class WeeToddLTX25CrossViewCameraOrbit:
    MATURITY = "Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "azimuth": (
                    "FLOAT",
                    {"default": -30.0, "min": -180.0, "max": 180.0, "step": 1.0},
                ),
                "elevation": (
                    "FLOAT",
                    {"default": 15.0, "min": -90.0, "max": 90.0, "step": 1.0},
                ),
                "distance": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05},
                ),
                "horizontal_fov": (
                    "FLOAT",
                    {"default": 50.0, "min": 20.0, "max": 120.0, "step": 1.0},
                ),
                "vertical_shift": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02},
                ),
                "depth_ratio": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.5, "max": 100.0, "step": 0.5},
                ),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "keep_source_aim": ("BOOLEAN", {"default": True}),
                "camera_keyframes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional 1-based JSON camera path. Use the visual timeline to select "
                            "a frame, position the sphere, and add or update its camera pose."
                        ),
                    },
                ),
                "interpolation": (
                    ["linear", "ease_in", "ease_out", "ease_in_out", "smooth"],
                    {"default": "smooth"},
                ),
            },
            "optional": {
                "pivot_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "pivot_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "step": 0.01}),
                "splat_radius": ("INT", {"default": 2, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("WEETODD_LTX25_CROSSVIEW_CAMERA", "STRING")
    RETURN_NAMES = ("camera", "camera_info")
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/LTX 2.5/preprocessors/camera"
    DESCRIPTION = (
        "Build a multi-point CrossView camera path with a stock-Comfy visual sphere and frame "
        "timeline. The path preview follows the selected interpolation, and the sphere view can "
        "rotate independently without changing camera poses. Numeric widgets and camera_keyframes "
        "remain authoritative for API workflows."
    )

    @staticmethod
    def _validate_keyframes(raw: str) -> str:
        text = str(raw).strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("CrossView camera keyframes must be valid JSON.") from exc
        if not isinstance(payload, list):
            raise ValueError("CrossView camera keyframes must be a JSON list.")
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"CrossView keyframe {index} must be a JSON object.")
            frame = item.get("frame", item.get("f"))
            if frame is None or isinstance(frame, bool):
                raise ValueError(f"CrossView keyframe {index} needs a positive frame number.")
            try:
                if int(frame) < 1 or float(frame) != int(frame):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CrossView keyframe {index} needs a positive integer frame number."
                ) from exc
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def configure(
        self,
        azimuth,
        elevation,
        distance,
        horizontal_fov,
        vertical_shift,
        depth_ratio,
        invert_depth,
        keep_source_aim,
        camera_keyframes,
        interpolation,
        pivot_x=0.0,
        pivot_y=0.0,
        pivot_z=1.05,
        splat_radius=2,
    ):
        normalized_keyframes = self._validate_keyframes(camera_keyframes)
        values = {
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "distance": float(distance),
            "horizontal_fov": float(horizontal_fov),
            "vertical_shift": float(vertical_shift),
            "depth_ratio": float(depth_ratio),
            "invert_depth": bool(invert_depth),
            "keep_source_aim": bool(keep_source_aim),
            "camera_keyframes": normalized_keyframes,
            "interpolation": str(interpolation),
            "pivot_x": float(pivot_x),
            "pivot_y": float(pivot_y),
            "pivot_z": float(pivot_z),
            "splat_radius": int(splat_radius),
        }
        identity = hashlib.sha256(
            json.dumps(values, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        camera = LTX25CrossViewCameraConfig(**values, path_id=identity)
        in_reliable_range = (
            -45.0 <= camera.azimuth <= 45.0
            and -20.0 <= camera.elevation <= 30.0
            and abs(camera.distance - 1.0) <= 0.05
        )
        report = {
            **values,
            "camera_contract": "weetodd-ltx25-crossview-camera-v1",
            "path_id": identity,
            "reliable_adapter_range": in_reliable_range,
            "warning": (
                None
                if in_reliable_range
                else "Pose is outside the adapter's best-observed training range."
            ),
        }
        return camera, json.dumps(report, indent=2, sort_keys=True)


class WeeToddLTX25CrossViewWarp:
    MATURITY = "Experimental"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "depth_video": ("IMAGE",),
                "azimuth": (
                    "FLOAT",
                    {
                        "default": -30.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": (
                            "Use -45 to +45 degrees first. The adapter has weaker support "
                            "outside this range."
                        ),
                    },
                ),
                "elevation": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": -90.0,
                        "max": 90.0,
                        "step": 1.0,
                        "tooltip": (
                            "Use -20 to +30 degrees first. Low upward views have limited "
                            "training coverage."
                        ),
                    },
                ),
                "distance": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "Keep 1.0 for reliable control. Distance control is weak in "
                            "the current adapter."
                        ),
                    },
                ),
                "horizontal_fov": (
                    "FLOAT",
                    {"default": 50.0, "min": 20.0, "max": 120.0, "step": 1.0},
                ),
                "vertical_shift": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.02},
                ),
                "depth_ratio": (
                    "FLOAT",
                    {"default": 6.0, "min": 1.5, "max": 100.0, "step": 0.5},
                ),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "keep_source_aim": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep the original optical-axis aim. This matches the adapter "
                            "training warp."
                        ),
                    },
                ),
                "camera_keyframes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional 1-based JSON camera path. Example: "
                            '[{"f":1,"az":-30,"el":10,"dist":1.0},'
                            '{"f":121,"az":30,"el":10,"dist":1.0}]'
                        ),
                    },
                ),
                "interpolation": (
                    ["linear", "ease_in", "ease_out", "ease_in_out", "smooth"],
                    {"default": "smooth"},
                ),
            },
            "optional": {
                "camera": ("WEETODD_LTX25_CROSSVIEW_CAMERA",),
                "pivot_x": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "pivot_y": ("FLOAT", {"default": 0.0, "step": 0.01}),
                "pivot_z": ("FLOAT", {"default": 1.05, "min": 0.01, "step": 0.01}),
                "splat_radius": ("INT", {"default": 2, "min": 0, "max": 3, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("warp_video", "warp_info")
    FUNCTION = "warp"
    CATEGORY = "WeeTodd/LTX 2.5/preprocessors/camera"
    DESCRIPTION = (
        "Build the full-resolution magenta-hole camera warp expected by the CrossView Warp "
        "IC-LoRA. Connect the result and the same source video to CrossView Dual Reference Guide."
    )

    def warp(
        self,
        source_video,
        depth_video,
        azimuth,
        elevation,
        distance,
        horizontal_fov,
        vertical_shift,
        depth_ratio,
        invert_depth,
        keep_source_aim,
        camera_keyframes,
        interpolation,
        pivot_x=0.0,
        pivot_y=0.0,
        pivot_z=1.05,
        splat_radius=2,
        camera=None,
    ):
        from ltx25_mlx.crossview import build_crossview_warp

        if camera is not None:
            if not isinstance(camera, LTX25CrossViewCameraConfig):
                raise TypeError("CrossView Warp received an incompatible camera contract.")
            controls = camera.warp_kwargs()
            azimuth = controls["azimuth"]
            elevation = controls["elevation"]
            distance = controls["distance"]
            horizontal_fov = controls["horizontal_fov"]
            vertical_shift = controls["vertical_shift"]
            depth_ratio = controls["depth_ratio"]
            invert_depth = controls["invert_depth"]
            keep_source_aim = controls["keep_source_aim"]
            camera_keyframes = controls["camera_keyframes"]
            interpolation = controls["interpolation"]
            pivot_x = controls["pivot_x"]
            pivot_y = controls["pivot_y"]
            pivot_z = controls["pivot_z"]
            splat_radius = controls["splat_radius"]

        def interruption_callback():
            try:
                import comfy.model_management as model_management

                model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                return

        progress = None
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(int(source_video.shape[0]))
        except ImportError:
            pass
        output, report = build_crossview_warp(
            source_video,
            depth_video,
            azimuth=float(azimuth),
            elevation=float(elevation),
            distance=float(distance),
            horizontal_fov=float(horizontal_fov),
            vertical_shift=float(vertical_shift),
            depth_ratio=float(depth_ratio),
            invert_depth=bool(invert_depth),
            pivot_x=float(pivot_x),
            pivot_y=float(pivot_y),
            pivot_z=float(pivot_z),
            keep_source_aim=bool(keep_source_aim),
            keyframes=str(camera_keyframes),
            interpolation=str(interpolation),
            splat_radius=int(splat_radius),
            progress_callback=(
                (lambda current, total: progress.update_absolute(current, total))
                if progress is not None
                else None
            ),
            interruption_callback=interruption_callback,
        )
        try:
            import torch

            output = torch.from_numpy(output)
        except ImportError:
            pass
        report["camera_path_id"] = camera.path_id if camera is not None else None
        report["camera_source"] = "orbit node" if camera is not None else "warp widgets"
        return output, json.dumps(report, indent=2, sort_keys=True)


class WeeToddMLXPreprocessorUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/MLX preprocessors"
    DESCRIPTION = "Release weighted MLX preprocessor state without changing H3 or LTX residency."

    def release(self, unload):
        if unload:
            from ltx25_mlx.crossview import clear_crossview_geometry_cache

            VIDEO_DEPTH_RUNTIME.unload()
            DWPOSE_RUNTIME.unload()
            TEED_RUNTIME.unload()
            FAST_DEPTH_RUNTIME.unload()
            LINEART_RUNTIME.unload()
            geometry_entries = clear_crossview_geometry_cache()
            return (
                "MLX preprocessor models unloaded; "
                f"cleared {geometry_entries} cached CrossView projection grid(s)",
            )
        return ("MLX preprocessor models kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddMLXCannyPreprocessor": WeeToddMLXCannyPreprocessor,
    "WeeToddMLXVideoDepthLoader": WeeToddMLXVideoDepthLoader,
    "WeeToddMLXVideoDepthPreprocessor": WeeToddMLXVideoDepthPreprocessor,
    "WeeToddMLXDWPoseLoader": WeeToddMLXDWPoseLoader,
    "WeeToddMLXDWPosePreprocessor": WeeToddMLXDWPosePreprocessor,
    "WeeToddMLXTEEDLoader": WeeToddMLXTEEDLoader,
    "WeeToddMLXTEEDPreprocessor": WeeToddMLXTEEDPreprocessor,
    "WeeToddMLXFastDepthLoader": WeeToddMLXFastDepthLoader,
    "WeeToddMLXFastDepthPreprocessor": WeeToddMLXFastDepthPreprocessor,
    "WeeToddMLXNormalMapPreprocessor": WeeToddMLXNormalMapPreprocessor,
    "WeeToddMLXLineArtLoader": WeeToddMLXLineArtLoader,
    "WeeToddMLXLineArtPreprocessor": WeeToddMLXLineArtPreprocessor,
    "WeeToddOpticalFlowMotionTracks": WeeToddOpticalFlowMotionTracks,
    "WeeToddMLXMotionTrackGuide": WeeToddMLXMotionTrackGuide,
    "WeeToddLTX25CrossViewCameraOrbit": WeeToddLTX25CrossViewCameraOrbit,
    "WeeToddLTX25CrossViewWarp": WeeToddLTX25CrossViewWarp,
    "WeeToddMLXPreprocessorUnload": WeeToddMLXPreprocessorUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddMLXCannyPreprocessor": "WeeTodd Canny Preprocessor (MLX)",
    "WeeToddMLXVideoDepthLoader": "WeeTodd Video Depth Model Loader (MLX)",
    "WeeToddMLXVideoDepthPreprocessor": "WeeTodd Video Depth Preprocessor (MLX)",
    "WeeToddMLXDWPoseLoader": "WeeTodd DWPose Model Loader (MLX)",
    "WeeToddMLXDWPosePreprocessor": "WeeTodd DWPose Preprocessor (MLX)",
    "WeeToddMLXTEEDLoader": "WeeTodd TEED Model Loader (MLX)",
    "WeeToddMLXTEEDPreprocessor": "WeeTodd TEED Soft-Edge Preprocessor (MLX)",
    "WeeToddMLXFastDepthLoader": "WeeTodd Fast Depth Model Loader (MLX)",
    "WeeToddMLXFastDepthPreprocessor": "WeeTodd Fast Depth Preprocessor (MLX)",
    "WeeToddMLXNormalMapPreprocessor": "WeeTodd Depth to Normal Map (MLX)",
    "WeeToddMLXLineArtLoader": "WeeTodd Line Art Model Loader (MLX)",
    "WeeToddMLXLineArtPreprocessor": "WeeTodd Realistic Line Art Preprocessor (MLX)",
    "WeeToddOpticalFlowMotionTracks": "WeeTodd Optical Flow Motion Tracks",
    "WeeToddMLXMotionTrackGuide": "WeeTodd Motion Track Guide (MLX)",
    "WeeToddLTX25CrossViewCameraOrbit": "WeeTodd LTX 2.5 CrossView Camera Orbit",
    "WeeToddLTX25CrossViewWarp": "WeeTodd LTX 2.5 CrossView Warp",
    "WeeToddMLXPreprocessorUnload": "WeeTodd Unload MLX Preprocessors",
}
