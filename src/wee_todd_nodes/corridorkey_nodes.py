"""ComfyUI adapters for the separately installed CorridorKey MLX runtime.

This module contains only integration and general-purpose mask operations. CorridorKey's
implementation and checkpoint are separately licensed and are not vendored by WeeTodd Nodes.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CORRIDORKEY_LICENSE = "CorridorKey Licence (CC BY-NC-SA 4.0 plus additional terms)"
CORRIDORKEY_SOURCE = "https://github.com/nikopueringer/CorridorKey"
CORRIDORKEY_MLX_SOURCE = "https://github.com/nikopueringer/corridorkey-mlx"


def _ensure_model_category(folder_paths: Any) -> str:
    category = "corridorkey"
    if category not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            category, str(Path(folder_paths.models_dir) / "corridorkey")
        )
    return category


def _checkpoint_names() -> list[str]:
    try:
        import folder_paths

        category = _ensure_model_category(folder_paths)
        names = [
            name
            for name in folder_paths.get_filename_list(category)
            if name.lower().endswith(".safetensors")
        ]
        if names:
            return sorted(set(names))
    except (ImportError, KeyError, OSError):
        pass
    return ["corridorkey_mlx.safetensors"]


def _resolve_checkpoint(name: str) -> Path:
    try:
        import folder_paths

        category = _ensure_model_category(folder_paths)
        value = folder_paths.get_full_path(category, name)
        if value:
            return Path(value).resolve()
    except (ImportError, KeyError, OSError):
        pass

    candidate = Path(name).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"CorridorKey MLX checkpoint {name!r} was not found. Place "
        "corridorkey_mlx.safetensors under ComfyUI/models/corridorkey or a configured shared "
        "corridorkey model folder, then refresh ComfyUI."
    )


def _check_interrupted():
    try:
        import comfy.model_management

        return comfy.model_management.throw_exception_if_processing_interrupted
    except ImportError:
        return None


def _progress_bar(total: int):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except ImportError:
        return None


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CorridorKey ComfyUI nodes require ComfyUI's PyTorch runtime.") from exc
    return torch


@dataclass(frozen=True)
class CorridorKeyModelSpec:
    checkpoint_name: str
    checkpoint_path: Path
    profile: str
    inference_resolution: int
    compile_model: bool
    tile_size: int | None
    overlap: int
    keep_loaded: bool


class _CorridorKeyRuntime:
    def __init__(self) -> None:
        self.engine = None
        self.signature: tuple[Any, ...] | None = None

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load(self, spec: CorridorKeyModelSpec):
        signature = (
            spec.checkpoint_path,
            spec.inference_resolution,
            spec.compile_model,
            spec.tile_size,
            spec.overlap,
        )
        if self.engine is not None and self.signature == signature:
            return self.engine, False
        self.unload()
        try:
            from corridorkey_mlx import CorridorKeyMLXEngine
        except ImportError as exc:
            raise RuntimeError(
                "The separately licensed corridorkey-mlx runtime is not installed in ComfyUI's "
                "Python environment. Install it from the official CorridorKey MLX repository, "
                "then restart ComfyUI."
            ) from exc
        self.engine = CorridorKeyMLXEngine(
            checkpoint_path=spec.checkpoint_path,
            img_size=spec.inference_resolution,
            compile=spec.compile_model,
            tile_size=spec.tile_size,
            overlap=spec.overlap,
        )
        self.signature = signature
        return self.engine, True

    def unload(self) -> bool:
        was_loaded = self.engine is not None
        self.engine = None
        self.signature = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
        return was_loaded


CORRIDORKEY_RUNTIME = _CorridorKeyRuntime()


def _profile_values(profile: str) -> tuple[int, bool, int | None, int]:
    profiles = {
        "speed — compiled 512": (512, True, None, 64),
        "balanced — compiled 1024": (1024, True, None, 64),
        "quality — compiled 2048": (2048, True, None, 64),
        "low memory — tiled 512": (512, False, 512, 64),
        "custom": (1024, True, None, 64),
    }
    return profiles[profile]


class WeeToddCorridorKeyModelLoader:
    """Build a lazy model specification without loading weights during graph validation."""

    DESCRIPTION = (
        "Select a separately installed CorridorKey MLX checkpoint and a measured speed, quality, "
        "or low-memory profile. Weights load only when the keyer executes."
    )
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "configure"
    RETURN_TYPES = ("WEETODD_CORRIDORKEY_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (_checkpoint_names(),),
                "profile": (
                    [
                        "speed — compiled 512",
                        "balanced — compiled 1024",
                        "quality — compiled 2048",
                        "low memory — tiled 512",
                        "custom",
                    ],
                    {"default": "balanced — compiled 1024"},
                ),
                "keep_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "inference_resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 128},
                ),
                "compile_model": ("BOOLEAN", {"default": True}),
                "tile_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2048, "step": 128},
                ),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 256, "step": 16}),
            },
        }

    def configure(
        self,
        checkpoint: str,
        profile: str,
        keep_loaded: bool,
        inference_resolution: int = 1024,
        compile_model: bool = True,
        tile_size: int = 0,
        overlap: int = 64,
    ):
        checkpoint_path = _resolve_checkpoint(checkpoint)
        if checkpoint_path.suffix.lower() != ".safetensors":
            raise ValueError("CorridorKey MLX requires a converted .safetensors checkpoint.")
        if profile != "custom":
            inference_resolution, compile_model, resolved_tile_size, overlap = _profile_values(
                profile
            )
        else:
            resolved_tile_size = tile_size or None
            if resolved_tile_size and overlap >= resolved_tile_size:
                raise ValueError("CorridorKey tile overlap must be smaller than the tile size.")
            if resolved_tile_size:
                compile_model = False
                inference_resolution = resolved_tile_size
        spec = CorridorKeyModelSpec(
            checkpoint_name=checkpoint,
            checkpoint_path=checkpoint_path,
            profile=profile,
            inference_resolution=int(inference_resolution),
            compile_model=bool(compile_model),
            tile_size=resolved_tile_size,
            overlap=int(overlap),
            keep_loaded=bool(keep_loaded),
        )
        stat = checkpoint_path.stat()
        info = {
            "adapter": "WeeTodd CorridorKey MLX",
            "checkpoint": checkpoint_path.name,
            "checkpoint_bytes": stat.st_size,
            "profile": profile,
            "inference_resolution": spec.inference_resolution,
            "compiled": spec.compile_model,
            "tile_size": spec.tile_size,
            "overlap": spec.overlap,
            "keep_loaded": spec.keep_loaded,
            "runtime_source": CORRIDORKEY_MLX_SOURCE,
            "model_source": CORRIDORKEY_SOURCE,
            "external_license": CORRIDORKEY_LICENSE,
        }
        return spec, json.dumps(info, indent=2)


def _border_pixels(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    band = max(1, min(height, width) // 20)
    return np.concatenate(
        (
            frame[:band].reshape(-1, 3),
            frame[-band:].reshape(-1, 3),
            frame[band:-band, :band].reshape(-1, 3),
            frame[band:-band, -band:].reshape(-1, 3),
        ),
        axis=0,
    )


def _estimate_screen_color(frame: np.ndarray, mode: str) -> np.ndarray:
    pixels = _border_pixels(frame)
    if mode == "green":
        selected = pixels[(pixels[:, 1] >= pixels[:, 0]) & (pixels[:, 1] >= pixels[:, 2])]
    elif mode == "blue":
        selected = pixels[(pixels[:, 2] >= pixels[:, 0]) & (pixels[:, 2] >= pixels[:, 1])]
    else:
        median = np.median(pixels, axis=0)
        dominant = 1 if median[1] >= median[2] else 2
        selected = pixels[pixels[:, dominant] >= np.max(pixels[:, [0, 3 - dominant]], axis=1)]
    if selected.size < 30:
        selected = pixels
    color = np.median(selected, axis=0).astype(np.float32)
    return color / max(float(color.sum()), 1e-6)


def _morph_and_blur(mask: np.ndarray, erode: int, blur: int) -> np.ndarray:
    result = mask.astype(np.float32)
    if erode:
        size = abs(int(erode)) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        operation = cv2.erode if erode > 0 else cv2.dilate
        result = operation(result, kernel)
    if blur > 0:
        kernel_size = int(blur) * 2 + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    return np.clip(result, 0.0, 1.0)


class WeeToddCorridorKeyAutoHint:
    """Create the coarse, eroded alpha hint that CorridorKey expects from screen color."""

    DESCRIPTION = (
        "Create a coarse green-screen alpha hint from border chromaticity. The default erosion "
        "and blur match the hint style that CorridorKey expects."
    )
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "build"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("alpha_hint", "mask_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "screen_color": (
                    ["auto from first frame", "green"],
                    {"default": "auto from first frame"},
                ),
                "screen_threshold": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
                "edge_softness": (
                    "FLOAT",
                    {"default": 0.12, "min": 0.005, "max": 0.5, "step": 0.005},
                ),
                "erode_pixels": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "blur_pixels": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "temporal_smoothing": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 0.95, "step": 0.05},
                ),
            }
        }

    def build(
        self,
        images,
        screen_color: str,
        screen_threshold: float,
        edge_softness: float,
        erode_pixels: int,
        blur_pixels: int,
        temporal_smoothing: float,
    ):
        torch = _torch_module()
        frames = np.clip(images.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if frames.ndim != 4 or frames.shape[-1] < 3:
            raise ValueError("CorridorKey Auto Hint requires ComfyUI IMAGE data in BHWC layout.")
        mode = "auto" if screen_color.startswith("auto") else screen_color
        color = _estimate_screen_color(frames[0, :, :, :3], mode)
        if mode == "auto" and color[2] > color[1]:
            raise ValueError(
                "The plate appears to use a blue screen. The official CorridorKey MLX runtime "
                "currently supports only the green checkpoint."
            )
        masks = []
        previous = None
        interruption = _check_interrupted()
        progress = _progress_bar(len(frames))
        for frame in frames:
            if interruption:
                interruption()
            rgb = frame[:, :, :3]
            chroma = rgb / np.maximum(rgb.sum(axis=2, keepdims=True), 1e-6)
            distance = np.linalg.norm(chroma - color.reshape(1, 1, 3), axis=2)
            mask = np.clip(
                (distance - float(screen_threshold)) / float(edge_softness), 0.0, 1.0
            )
            mask = _morph_and_blur(mask, int(erode_pixels), int(blur_pixels))
            if previous is not None and temporal_smoothing > 0:
                amount = float(temporal_smoothing)
                mask = mask * (1.0 - amount) + previous * amount
            previous = mask
            masks.append(mask.astype(np.float32))
            if progress:
                progress.update(1)
        info = {
            "method": "border-sampled chromaticity distance",
            "screen_mode": mode,
            "estimated_screen_rgb_chromaticity": color.tolist(),
            "screen_threshold": screen_threshold,
            "edge_softness": edge_softness,
            "erode_pixels": erode_pixels,
            "blur_pixels": blur_pixels,
            "temporal_smoothing": temporal_smoothing,
            "frames": len(masks),
        }
        return torch.from_numpy(np.stack(masks)), json.dumps(info, indent=2)


def _remove_small_regions(mask: np.ndarray, minimum_area: int, fill_holes: bool) -> np.ndarray:
    if minimum_area <= 0 and not fill_holes:
        return mask
    hard = (mask >= 0.5).astype(np.uint8)
    if minimum_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
        keep = np.zeros_like(hard)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area:
                keep[labels == index] = 1
        hard = keep
    if fill_holes:
        inverse = 1 - hard
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
        border_labels = set(np.unique(labels[0])) | set(np.unique(labels[-1]))
        border_labels |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
        for index in range(1, count):
            if index not in border_labels and stats[index, cv2.CC_STAT_AREA] > 0:
                hard[labels == index] = 1
    return np.where(hard > 0, np.maximum(mask, 0.5), 0.0).astype(np.float32)


class WeeToddCorridorKeyMaskRefine:
    """Prepare any ComfyUI mask, including SAM or Florence-derived masks, as an alpha hint."""

    DESCRIPTION = (
        "Shrink or grow, blur, fill, and clean any standard ComfyUI MASK before CorridorKey. "
        "Use this node with SAM, Florence-derived, Impact Pack, or manual masks."
    )
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "refine"
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "grow_shrink_pixels": (
                    "INT",
                    {"default": -4, "min": -64, "max": 64, "step": 1},
                ),
                "blur_pixels": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "remove_islands_below": (
                    "INT",
                    {"default": 0, "min": 0, "max": 65536, "step": 16},
                ),
                "fill_holes": ("BOOLEAN", {"default": False}),
            }
        }

    def refine(
        self,
        mask,
        grow_shrink_pixels: int,
        blur_pixels: int,
        remove_islands_below: int,
        fill_holes: bool,
    ):
        torch = _torch_module()
        masks = np.clip(mask.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if masks.ndim == 2:
            masks = masks[np.newaxis]
        refined = []
        for current in masks:
            current = _remove_small_regions(current, int(remove_islands_below), fill_holes)
            # Positive UI values grow the foreground; the shared helper uses negative erosion.
            current = _morph_and_blur(current, -int(grow_shrink_pixels), int(blur_pixels))
            refined.append(current.astype(np.float32))
        return (torch.from_numpy(np.stack(refined)),)


class WeeToddCorridorKeyKeyer:
    """Predict straight foreground color and a linear alpha matte with CorridorKey MLX."""

    DESCRIPTION = (
        "Run the optional CorridorKey MLX engine on an IMAGE batch and coarse MASK. Return "
        "straight foreground, alpha, premultiplied color, preview, and provenance metadata."
    )
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "key"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("foreground", "matte", "premultiplied", "checkerboard", "metadata")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_CORRIDORKEY_MODEL",),
                "images": ("IMAGE",),
                "alpha_hint": ("MASK",),
                "refiner_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    def key(self, model: CorridorKeyModelSpec, images, alpha_hint, refiner_strength: float):
        torch = _torch_module()
        frames = np.clip(images.detach().cpu().numpy(), 0.0, 1.0)
        masks = np.clip(alpha_hint.detach().cpu().numpy(), 0.0, 1.0)
        if frames.ndim != 4 or frames.shape[-1] < 3:
            raise ValueError("CorridorKey Keyer requires ComfyUI IMAGE data in BHWC layout.")
        if masks.ndim == 2:
            masks = masks[np.newaxis]
        if len(masks) == 1 and len(frames) > 1:
            masks = np.repeat(masks, len(frames), axis=0)
        if len(frames) != len(masks):
            raise ValueError(
                "CorridorKey image and alpha-hint batches must match, or the hint batch must "
                "contain one mask for explicit broadcast."
            )
        if masks.shape[1:3] != frames.shape[1:3]:
            raise ValueError("CorridorKey alpha-hint dimensions must match the input images.")

        interruption = _check_interrupted()
        progress = _progress_bar(len(frames))
        engine = None
        loaded_now = False
        foregrounds: list[np.ndarray] = []
        mattes: list[np.ndarray] = []
        premultiplied: list[np.ndarray] = []
        checkerboards: list[np.ndarray] = []
        failed = True
        try:
            if interruption:
                interruption()
            engine, loaded_now = CORRIDORKEY_RUNTIME.load(model)
            for frame, hint in zip(frames, masks, strict=True):
                if interruption:
                    interruption()
                result = engine.process_frame(
                    np.rint(frame[:, :, :3] * 255.0).astype(np.uint8),
                    np.rint(hint * 255.0).astype(np.uint8),
                    refiner_scale=float(refiner_strength),
                    despill_strength=0.0,
                    auto_despeckle=False,
                )
                fg = np.asarray(result["fg"], dtype=np.float32) / 255.0
                matte = np.asarray(result["alpha"], dtype=np.float32) / 255.0
                alpha = matte[:, :, np.newaxis]
                premul = fg * alpha
                grid_y, grid_x = np.indices(matte.shape)
                checker = (((grid_x // 16) + (grid_y // 16)) % 2).astype(np.float32)
                checker = (checker * 0.18 + 0.35)[:, :, np.newaxis]
                preview = premul + checker * (1.0 - alpha)
                foregrounds.append(np.clip(fg, 0.0, 1.0))
                mattes.append(np.clip(matte, 0.0, 1.0))
                premultiplied.append(np.clip(premul, 0.0, 1.0))
                checkerboards.append(np.clip(preview, 0.0, 1.0))
                if progress:
                    progress.update(1)
            failed = False
        finally:
            if failed or not model.keep_loaded:
                CORRIDORKEY_RUNTIME.unload()

        metadata = {
            "adapter": "WeeTodd CorridorKey MLX",
            "frames": len(frames),
            "input_width": int(frames.shape[2]),
            "input_height": int(frames.shape[1]),
            "refiner_strength": refiner_strength,
            "model_loaded_for_run": loaded_now,
            "model_resident_after_run": CORRIDORKEY_RUNTIME.loaded,
            "model": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(model).items()
                if key != "checkpoint_path"
            },
            "checkpoint": model.checkpoint_path.name,
            "external_license": CORRIDORKEY_LICENSE,
            "runtime_source": CORRIDORKEY_MLX_SOURCE,
        }
        return (
            torch.from_numpy(np.stack(foregrounds).astype(np.float32)),
            torch.from_numpy(np.stack(mattes).astype(np.float32)),
            torch.from_numpy(np.stack(premultiplied).astype(np.float32)),
            torch.from_numpy(np.stack(checkerboards).astype(np.float32)),
            json.dumps(metadata, indent=2),
        )


class WeeToddCorridorKeyComposite:
    """Composite straight or premultiplied foreground color over a background image."""

    DESCRIPTION = (
        "Composite CorridorKey foreground and matte outputs over a matching ComfyUI IMAGE batch."
    )
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                "matte": ("MASK",),
                "background": ("IMAGE",),
                "foreground_format": (
                    ["straight color", "premultiplied"],
                    {"default": "straight color"},
                ),
            }
        }

    def composite(self, foreground, matte, background, foreground_format: str):
        torch = _torch_module()
        fg = foreground.float()
        bg = background.float()
        alpha = matte.float()
        if alpha.ndim == 3:
            alpha = alpha.unsqueeze(-1)
        if bg.shape[0] == 1 and fg.shape[0] > 1:
            bg = bg.repeat(fg.shape[0], 1, 1, 1)
        if fg.shape != bg.shape or fg.shape[:3] != alpha.shape[:3]:
            raise ValueError(
                "CorridorKey composite inputs must have matching batch and dimensions."
            )
        if foreground_format == "premultiplied":
            result = fg + bg * (1.0 - alpha)
        else:
            result = fg * alpha + bg * (1.0 - alpha)
        return (torch.clamp(result, 0.0, 1.0),)


class WeeToddCorridorKeyUnload:
    """Release the process-local CorridorKey MLX engine and reclaim its allocator cache."""

    DESCRIPTION = "Release the process-local CorridorKey MLX engine and allocator cache."
    CATEGORY = "WeeTodd/CorridorKey"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("STRING", {"default": "unload"})}}

    def unload(self, trigger: str):
        unloaded = CORRIDORKEY_RUNTIME.unload()
        return (json.dumps({"unloaded": unloaded, "trigger": trigger}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "WeeToddCorridorKeyModelLoader": WeeToddCorridorKeyModelLoader,
    "WeeToddCorridorKeyAutoHint": WeeToddCorridorKeyAutoHint,
    "WeeToddCorridorKeyMaskRefine": WeeToddCorridorKeyMaskRefine,
    "WeeToddCorridorKeyKeyer": WeeToddCorridorKeyKeyer,
    "WeeToddCorridorKeyComposite": WeeToddCorridorKeyComposite,
    "WeeToddCorridorKeyUnload": WeeToddCorridorKeyUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddCorridorKeyModelLoader": "WeeTodd CorridorKey Model Loader (MLX)",
    "WeeToddCorridorKeyAutoHint": "WeeTodd CorridorKey Auto Chroma Hint",
    "WeeToddCorridorKeyMaskRefine": "WeeTodd CorridorKey Mask Refine",
    "WeeToddCorridorKeyKeyer": "WeeTodd CorridorKey Keyer (MLX)",
    "WeeToddCorridorKeyComposite": "WeeTodd CorridorKey Composite",
    "WeeToddCorridorKeyUnload": "WeeTodd CorridorKey Unload",
}
