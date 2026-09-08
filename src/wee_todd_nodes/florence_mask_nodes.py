"""MLX-native Florence-2 text grounding for ComfyUI masks.

The detector is intentionally separate from CorridorKey. It emits ordinary ComfyUI MASK and
IMAGE values that can feed CorridorKey, inpainting, compositing, or any other mask consumer.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

FLORENCE2_MODEL_SOURCE = "https://huggingface.co/mlx-community/Florence-2-base-ft-8bit"
FLORENCE2_UPSTREAM_SOURCE = "https://huggingface.co/microsoft/Florence-2-base-ft"
FLORENCE2_LICENSE = "MIT"


def _ensure_florence_category(folder_paths: Any) -> str:
    category = "florence2"
    if category not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            category, str(Path(folder_paths.models_dir) / "florence2")
        )
    return category


def _is_florence_bundle(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file() or not (path / "model.safetensors").is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_type") == "florence2"


def _bundle_config(path: Path) -> dict[str, Any]:
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Florence-2 bundle has an unreadable config.json: {path}") from exc


def _validate_tokenizer_contract(path: Path) -> str:
    config_path = path / "tokenizer_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Florence-2 bundle has an unreadable tokenizer_config.json: {path}"
        ) from exc
    tokenizer_class = str(config.get("tokenizer_class", ""))
    if tokenizer_class not in {"BartTokenizer", "BartTokenizerFast"}:
        raise ValueError(
            "Florence-2 requires the checkpoint's BART tokenizer contract; "
            f"the selected bundle declares {tokenizer_class or 'no tokenizer class'!r}."
        )
    return tokenizer_class


def _model_bundle_names() -> list[str]:
    try:
        import folder_paths

        category = _ensure_florence_category(folder_paths)
        names: list[str] = []
        for root_string in folder_paths.get_folder_paths(category):
            root = Path(root_string)
            if not root.is_dir():
                continue
            if _is_florence_bundle(root):
                names.append(".")
            for config_path in root.rglob("config.json"):
                if _is_florence_bundle(config_path.parent):
                    names.append(config_path.parent.relative_to(root).as_posix())
        if names:
            return sorted(set(names))
    except (ImportError, KeyError, OSError, ValueError):
        pass
    return ["Florence-2-base-ft-8bit"]


def _resolve_model_bundle(name: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_absolute() and _is_florence_bundle(candidate):
        return candidate.resolve()
    try:
        import folder_paths

        category = _ensure_florence_category(folder_paths)
        for root_string in folder_paths.get_folder_paths(category):
            root = Path(root_string)
            candidate = root if name == "." else root / name
            if _is_florence_bundle(candidate):
                return candidate.resolve()
    except (ImportError, KeyError, OSError):
        pass
    raise FileNotFoundError(
        f"Florence-2 MLX bundle {name!r} was not found. Download an MLX Florence-2 bundle "
        "under ComfyUI/models/florence2, keep its config, tokenizer, processor, and safetensors "
        "files together, then refresh ComfyUI."
    )


@dataclass(frozen=True)
class Florence2ModelSpec:
    model_name: str
    model_path: Path


class _Florence2Runtime:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.model_path: Path | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self, spec: Florence2ModelSpec):
        if self.loaded and self.model_path == spec.model_path:
            return self.model, self.processor, False
        self.unload()
        try:
            from mlx_vlm.tokenizer_utils import load_tokenizer
            from mlx_vlm.utils import StoppingCriteria, load_model
            from transformers import AutoImageProcessor, BartTokenizerFast
            from transformers.models.florence2.processing_florence2 import Florence2Processor
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 auto masking requires the project's mlx-vlm dependency."
            ) from exc
        # Historical MLX Community Florence bundles contain remote-code auto_map entries that
        # predate current Transformers. Load the MLX architecture directly and construct the
        # current native processor from local files. This never executes checkpoint-supplied code.
        model = load_model(spec.model_path, lazy=True)
        _validate_tokenizer_contract(spec.model_path)
        tokenizer = BartTokenizerFast.from_pretrained(spec.model_path)
        image_processor = AutoImageProcessor.from_pretrained(spec.model_path, use_fast=False)
        processor = Florence2Processor(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        detokenizer_class = load_tokenizer(spec.model_path, return_tokenizer=False)
        processor.detokenizer = detokenizer_class(processor.tokenizer)
        processor.tokenizer.stopping_criteria = StoppingCriteria(
            model.config.eos_token_id,
            processor.tokenizer,
        )
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type != "florence2":
            del model, processor
            raise ValueError(
                f"Selected bundle reports model_type={model_type!r}; Florence-2 is required."
            )
        self.model = model
        self.processor = processor
        self.model_path = spec.model_path
        return self.model, self.processor, True

    def unload(self) -> bool:
        was_loaded = self.loaded
        self.model = None
        self.processor = None
        self.model_path = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass
        return was_loaded


FLORENCE2_RUNTIME = _Florence2Runtime()


_BOX_PATTERN = re.compile(r"([^<]*?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")


def _parse_florence_boxes(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """Parse Florence location tokens without importing PyTorch post-processing code."""
    boxes: list[dict[str, Any]] = []
    clean = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    for match in _BOX_PATTERN.finditer(clean):
        label = match.group(1).replace("<od>", "").replace("</od>", "").strip()
        bins = [min(999, max(0, int(match.group(index)))) for index in range(2, 6)]
        x1 = (bins[0] + 0.5) * width / 1000.0
        y1 = (bins[1] + 0.5) * height / 1000.0
        x2 = (bins[2] + 0.5) * width / 1000.0
        y2 = (bins[3] + 0.5) * height / 1000.0
        if x2 > x1 and y2 > y1:
            boxes.append({"label": label, "bbox": [x1, y1, x2, y2]})
    return boxes


def _union_box(boxes: list[dict[str, Any]]) -> np.ndarray | None:
    if not boxes:
        return None
    values = np.asarray([box["bbox"] for box in boxes], dtype=np.float32)
    return np.array(
        [values[:, 0].min(), values[:, 1].min(), values[:, 2].max(), values[:, 3].max()],
        dtype=np.float32,
    )


def _sample_indices(frame_count: int, stride: int) -> list[int]:
    if frame_count < 1:
        return []
    indices = list(range(0, frame_count, max(1, int(stride))))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _interpolate_boxes(
    sampled: dict[int, np.ndarray | None], frame_count: int
) -> list[np.ndarray | None]:
    """Interpolate one union box between successful sparse detections."""
    valid = sorted(index for index, box in sampled.items() if box is not None)
    if not valid:
        return [None] * frame_count
    result: list[np.ndarray | None] = []
    for frame_index in range(frame_count):
        left_candidates = [index for index in valid if index <= frame_index]
        right_candidates = [index for index in valid if index >= frame_index]
        left = left_candidates[-1] if left_candidates else valid[0]
        right = right_candidates[0] if right_candidates else valid[-1]
        if left == right:
            result.append(sampled[left].copy())
            continue
        amount = (frame_index - left) / (right - left)
        result.append(sampled[left] * (1.0 - amount) + sampled[right] * amount)
    return result


def _box_mask(box: np.ndarray | None, width: int, height: int, expand: float, blur: int):
    mask = np.zeros((height, width), dtype=np.float32)
    rect = _expanded_rect(box, width, height, expand)
    if rect is None:
        return mask
    left, top, right, bottom = rect
    mask[top:bottom, left:right] = 1.0
    if blur > 0:
        kernel = int(blur) * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return np.clip(mask, 0.0, 1.0)


def _expanded_rect(
    box: np.ndarray | None, width: int, height: int, expand: float
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * expand / 100.0
    pad_y = (y2 - y1) * expand / 100.0
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return (left, top, right, bottom) if right > left and bottom > top else None


def _guided_silhouette_mask(
    frame: np.ndarray,
    box: np.ndarray | None,
    expand: float,
    blur: int,
    *,
    max_edge: int = 1024,
    iterations: int = 5,
) -> tuple[np.ndarray, bool]:
    """Refine a Florence box into a silhouette; fall back safely if refinement fails."""
    height, width = frame.shape[:2]
    rect = _expanded_rect(box, width, height, expand)
    if rect is None:
        return np.zeros((height, width), dtype=np.float32), False

    scale = min(1.0, float(max_edge) / max(width, height))
    work_width = max(2, int(round(width * scale)))
    work_height = max(2, int(round(height * scale)))
    image = np.rint(np.clip(frame[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    if (work_width, work_height) != (width, height):
        image = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)

    left, top, right, bottom = rect
    scaled_left = min(work_width - 1, max(0, int(np.floor(left * scale))))
    scaled_top = min(work_height - 1, max(0, int(np.floor(top * scale))))
    scaled_right = min(work_width, max(scaled_left + 1, int(np.ceil(right * scale))))
    scaled_bottom = min(work_height, max(scaled_top + 1, int(np.ceil(bottom * scale))))
    grabcut_rect = (
        scaled_left,
        scaled_top,
        scaled_right - scaled_left,
        scaled_bottom - scaled_top,
    )
    labels = np.full((work_height, work_width), cv2.GC_BGD, dtype=np.uint8)
    labels[scaled_top:scaled_bottom, scaled_left:scaled_right] = cv2.GC_PR_BGD
    # Florence already localized the subject. A conservative central seed gives GrabCut enough
    # foreground evidence without declaring the entire rectangular search region foreground.
    seed_margin_x = max(1, int(round(grabcut_rect[2] * 0.15)))
    seed_margin_y = max(1, int(round(grabcut_rect[3] * 0.15)))
    seed_left = min(scaled_right - 1, scaled_left + seed_margin_x)
    seed_top = min(scaled_bottom - 1, scaled_top + seed_margin_y)
    seed_right = max(seed_left + 1, scaled_right - seed_margin_x)
    seed_bottom = max(seed_top + 1, scaled_bottom - seed_margin_y)
    labels[seed_top:seed_bottom, seed_left:seed_right] = cv2.GC_PR_FGD
    core_margin_x = max(1, int(round(grabcut_rect[2] * 0.45)))
    core_margin_y = max(1, int(round(grabcut_rect[3] * 0.45)))
    core_left = min(scaled_right - 1, scaled_left + core_margin_x)
    core_top = min(scaled_bottom - 1, scaled_top + core_margin_y)
    core_right = max(core_left + 1, scaled_right - core_margin_x)
    core_bottom = max(core_top + 1, scaled_bottom - core_margin_y)
    labels[core_top:core_bottom, core_left:core_right] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image,
            labels,
            None,
            background_model,
            foreground_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
        mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 1.0, 0.0).astype(
            np.float32
        )
        # A blank or nearly full result is not a trustworthy refinement.
        box_area = max(1, grabcut_rect[2] * grabcut_rect[3])
        fill_ratio = float(mask.sum()) / box_area
        if not np.isfinite(fill_ratio) or fill_ratio < 0.005 or fill_ratio > 0.995:
            raise ValueError("degenerate GrabCut result")
        if (work_width, work_height) != (width, height):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        if blur > 0:
            kernel = int(blur) * 2 + 1
            mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        return np.clip(mask, 0.0, 1.0), False
    except (cv2.error, ValueError):
        return _box_mask(box, width, height, expand, blur), True


def _preview_masks(
    frames: np.ndarray, masks: np.ndarray, boxes: list[np.ndarray | None]
) -> np.ndarray:
    previews = np.clip(frames[:, :, :, :3], 0.0, 1.0).copy()
    for frame, mask, box in zip(previews, masks, boxes, strict=True):
        overlay = np.zeros_like(frame)
        overlay[:, :, 1] = 1.0
        alpha = np.clip(mask[:, :, None], 0.0, 1.0) * 0.24
        frame[:] = frame * (1.0 - alpha) + overlay * alpha
        contours, _ = cv2.findContours(
            (mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(
                frame,
                contours,
                -1,
                (0.1, 1.0, 0.25),
                max(1, min(frame.shape[:2]) // 256),
            )
        if box is None:
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(
            frame,
            (max(0, x1), max(0, y1)),
            (min(width - 1, x2), min(height - 1, y2)),
            (1.0, 0.25, 0.0),
            max(1, min(width, height) // 256),
        )
    return previews.astype(np.float32)


def _prepare_detection_image(frame: np.ndarray, max_edge: int = 768) -> Image.Image:
    image = Image.fromarray(np.rint(frame[:, :, :3] * 255.0).astype(np.uint8), mode="RGB")
    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Florence-2 ComfyUI nodes require ComfyUI's tensor runtime.") from exc
    return torch


def _check_interrupted() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _progress_bar(total: int):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(total)
    except ImportError:
        return None


class WeeToddFlorence2ModelLoader:
    DESCRIPTION = (
        "Select a local MLX Florence-2 bundle for text-guided auto masking. The node validates "
        "the bundle but does not load weights until detection executes."
    )
    CATEGORY = "WeeTodd/MLX preprocessors/segmentation"
    FUNCTION = "select"
    RETURN_TYPES = ("WEETODD_FLORENCE2_MODEL", "STRING")
    RETURN_NAMES = ("florence_model", "model_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_bundle": (_model_bundle_names(),)}}

    def select(self, model_bundle: str):
        path = _resolve_model_bundle(str(model_bundle))
        bits = (_bundle_config(path).get("quantization") or {}).get("bits")
        if bits is not None and int(bits) < 8:
            raise ValueError(
                "Florence-2 4-bit coordinate decoding is not reliable in the currently "
                "published MLX conversion. Use the tested 8-bit bundle or BF16 weights."
            )
        spec = Florence2ModelSpec(str(model_bundle), path)
        info = {
            "architecture": "florence2",
            "backend": "mlx-vlm",
            "bundle": spec.model_name,
            "loaded": False,
            "weight_bits": bits or "bf16/fp16",
            "license": FLORENCE2_LICENSE,
            "model_source": FLORENCE2_MODEL_SOURCE,
            "upstream_source": FLORENCE2_UPSTREAM_SOURCE,
        }
        return spec, json.dumps(info, indent=2, sort_keys=True)


class WeeToddFlorence2TextMask:
    DESCRIPTION = (
        "Ground a text description with Florence-2 on sparse video frames, interpolate its "
        "location, and emit either a guided subject silhouette or a fast rectangular mask."
    )
    CATEGORY = "WeeTodd/MLX preprocessors/segmentation"
    FUNCTION = "detect"
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("subject_mask", "mask_preview", "detections_json", "detection_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "florence_model": ("WEETODD_FLORENCE2_MODEL",),
                "subject": (
                    "STRING",
                    {
                        "default": "the foreground person",
                        "multiline": False,
                        "tooltip": "Describe the object or subject that should become foreground.",
                    },
                ),
                "mask_refinement": (
                    ["guided silhouette (recommended)", "bounding box (fast)"],
                    {
                        "default": "guided silhouette (recommended)",
                        "tooltip": (
                            "Refine Florence boxes into contours, or emit fast coarse rectangles."
                        ),
                    },
                ),
                "detect_every_n_frames": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 240,
                        "step": 1,
                        "tooltip": "Florence runs on frame 0, this stride, and the final frame.",
                    },
                ),
                "box_expand_percent": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
                "edge_blur_pixels": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "max_tokens": ("INT", {"default": 128, "min": 32, "max": 512, "step": 16}),
                "keep_model_warm": ("BOOLEAN", {"default": False}),
            }
        }

    def detect(
        self,
        images,
        florence_model: Florence2ModelSpec,
        subject: str,
        mask_refinement: str,
        detect_every_n_frames: int,
        box_expand_percent: float,
        edge_blur_pixels: int,
        max_tokens: int,
        keep_model_warm: bool,
    ):
        if not str(subject).strip():
            raise ValueError("Florence-2 auto masking requires a non-empty subject description.")
        torch = _torch_module()
        frames = np.clip(images.detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
        if frames.ndim != 4 or frames.shape[-1] < 3 or len(frames) < 1:
            raise ValueError("Florence-2 auto masking requires ComfyUI IMAGE data in BHWC layout.")
        frame_count, height, width = frames.shape[:3]
        indices = _sample_indices(frame_count, int(detect_every_n_frames))
        prompt = f"<OPEN_VOCABULARY_DETECTION>{str(subject).strip()}"
        sampled: dict[int, np.ndarray | None] = {}
        raw_detections: dict[str, Any] = {}
        progress = _progress_bar(len(indices))
        loaded_now = False
        failed = True
        try:
            _check_interrupted()
            try:
                import mlx.core as mx

                mx.reset_peak_memory()
            except (ImportError, AttributeError):
                mx = None
            model, processor, loaded_now = FLORENCE2_RUNTIME.load(florence_model)
            from mlx_vlm import generate

            for index in indices:
                _check_interrupted()
                pil_image = _prepare_detection_image(frames[index])
                result = generate(
                    model,
                    processor,
                    prompt,
                    image=pil_image,
                    max_tokens=int(max_tokens),
                    temperature=0.0,
                    skip_special_tokens=False,
                    verbose=False,
                )
                parsed = _parse_florence_boxes(result.text, width, height)
                for detection in parsed:
                    detection["label"] = str(subject).strip()
                sampled[index] = _union_box(parsed)
                raw_detections[str(index)] = {
                    "boxes": parsed,
                    "generation_tokens": result.generation_tokens,
                    "finish_reason": result.finish_reason,
                }
                if progress:
                    progress.update(1)
            boxes = _interpolate_boxes(sampled, frame_count)
            if not any(box is not None for box in boxes):
                raise RuntimeError(
                    f"Florence-2 did not locate {subject!r}. Try a simpler noun phrase or a "
                    "smaller detection stride."
                )
            fallback_frames: list[int] = []
            if mask_refinement == "guided silhouette (recommended)":
                refined = []
                for frame_index, (frame, box) in enumerate(zip(frames, boxes, strict=True)):
                    mask, used_fallback = _guided_silhouette_mask(
                        frame,
                        box,
                        float(box_expand_percent),
                        int(edge_blur_pixels),
                    )
                    refined.append(mask)
                    if used_fallback:
                        fallback_frames.append(frame_index)
                masks = np.stack(refined).astype(np.float32)
                mask_method = "Florence text box + bounded GrabCut silhouette"
            elif mask_refinement == "bounding box (fast)":
                masks = np.stack(
                    [
                        _box_mask(
                            box,
                            width,
                            height,
                            float(box_expand_percent),
                            int(edge_blur_pixels),
                        )
                        for box in boxes
                    ]
                ).astype(np.float32)
                mask_method = "Florence text box"
            else:
                raise ValueError(f"Unknown Florence mask refinement mode: {mask_refinement!r}")
            previews = _preview_masks(frames, masks, boxes)
            peak_gib = float(mx.get_peak_memory() / (1024**3)) if mx is not None else None
            failed = False
            detection_info = {
                "backend": "mlx-vlm",
                "model": florence_model.model_name,
                "task": "open-vocabulary-detection",
                "subject": str(subject).strip(),
                "frames": frame_count,
                "evaluated_frames": indices,
                "evaluations": len(indices),
                "detection_max_edge": 768,
                "box_interpolation": "linear union envelope",
                "mask_method": mask_method,
                "refinement_fallback_frames": fallback_frames,
                "refinement_max_edge": 1024 if "GrabCut" in mask_method else None,
                "box_expand_percent": float(box_expand_percent),
                "edge_blur_pixels": int(edge_blur_pixels),
                "loaded_now": loaded_now,
                "resident_after_run": bool(keep_model_warm),
                "mlx_peak_gib": peak_gib,
            }
            return (
                torch.from_numpy(masks),
                torch.from_numpy(previews),
                json.dumps(raw_detections, indent=2, sort_keys=True),
                json.dumps(detection_info, indent=2, sort_keys=True),
            )
        finally:
            if failed or not keep_model_warm:
                FLORENCE2_RUNTIME.unload()


class WeeToddFlorence2Unload:
    DESCRIPTION = "Release Florence-2 MLX state without changing CorridorKey, H3, or LTX state."
    CATEGORY = "WeeTodd/MLX preprocessors/segmentation"
    FUNCTION = "release"
    RETURN_TYPES = ("STRING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    def release(self, unload: bool):
        if unload:
            released = FLORENCE2_RUNTIME.unload()
            return ("Florence-2 MLX runtime unloaded" if released else "Florence-2 was not loaded",)
        return ("Florence-2 MLX runtime kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddFlorence2ModelLoader": WeeToddFlorence2ModelLoader,
    "WeeToddFlorence2TextMask": WeeToddFlorence2TextMask,
    "WeeToddFlorence2Unload": WeeToddFlorence2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddFlorence2ModelLoader": "WeeTodd Florence-2 Model Loader (MLX)",
    "WeeToddFlorence2TextMask": "WeeTodd Florence-2 Text Auto Mask (MLX)",
    "WeeToddFlorence2Unload": "WeeTodd Unload Florence-2 (MLX)",
}
