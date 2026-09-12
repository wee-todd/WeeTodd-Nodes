"""Canonical verified conditioning for Draw Things requests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_inputs(
    attachments: Any, assets: Any, *, model_family: str = "", num_frames: int | None = None
) -> list[dict[str, Any]]:
    if not isinstance(attachments, list):
        raise ValueError("Draw Things attachments must be an array")
    if not isinstance(assets, list):
        raise ValueError("Draw Things assets must be an array")
    if not attachments:
        return []
    if any(isinstance(item, dict) and item.get("role") == "last" for item in attachments):
        if model_family.lower() != "minimaxh3":
            raise ValueError("Draw Things last-frame attachments require an H3 FL2VA model")
        roles = [item.get("role") if isinstance(item, dict) else None for item in attachments]
        if sorted(str(role) for role in roles) != ["first", "last"]:
            raise ValueError("Draw Things requires exactly one first and one last attachment")
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 2:
            raise ValueError("Draw Things last-frame input requires a resolved video frame count")
        result = []
        for role in ("first", "last"):
            attachment = next(item for item in attachments if item["role"] == role)
            item = canonical_inputs([{**attachment, "role": "first"}], assets)[0]
            item.update(role=role, frameIndex=0 if role == "first" else num_frames - 1)
            result.append(item)
        return result
    unsupported = [
        item.get("role")
        for item in attachments
        if not isinstance(item, dict) or item.get("role") != "first"
    ]
    if unsupported:
        role = unsupported[0] if isinstance(unsupported[0], str) else "unknown"
        raise ValueError(
            f"Draw Things attachment role {role} is not supported; use one first-frame image"
        )
    if len(attachments) != 1:
        raise ValueError("Draw Things supports only one first-frame attachment")
    attachment = attachments[0]
    strength = attachment.get("strength", 1)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or strength != 1:
        raise ValueError("Draw Things first-frame attachment must use strength 1")
    asset_id = attachment.get("assetID")
    matches = [
        item for item in assets if isinstance(item, dict) and str(item.get("id")) == str(asset_id)
    ]
    if len(matches) != 1:
        raise ValueError("Relink the Draw Things first-frame attachment to exactly one asset")
    asset = matches[0]
    if asset.get("kind") != "image":
        raise ValueError("Draw Things first-frame attachment must use an image asset")
    raw_path = asset.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Relink the Draw Things first-frame image")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Relink the Draw Things first-frame image; its file is missing")
    return [
        {
            "role": "first",
            "path": str(path),
            "sha256": _sha256(path),
            "frameIndex": 0,
            "strength": 1,
        }
    ]


def canonical_loras(loras: Any) -> list[dict[str, Any]]:
    if loras is None:
        return []
    if not isinstance(loras, list):
        raise ValueError("Draw Things loras must be an array")
    if len(loras) > 16:
        raise ValueError("Draw Things supports at most 16 LoRAs")
    result = []
    seen: set[str] = set()
    for item in loras:
        if not isinstance(item, dict) or set(item) != {"modelID", "weight"}:
            raise ValueError(
                "Draw Things LoRAs must be server-resident modelID and weight pairs; "
                "local LoRA upload has no verified converter"
            )
        model_id = item["modelID"]
        weight = item["weight"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Draw Things LoRA modelID must be an exact server-resident ID")
        if model_id in seen:
            raise ValueError(f"Draw Things LoRA modelID is duplicated: {model_id}")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or not 0 <= weight <= 2
        ):
            raise ValueError(f"Draw Things LoRA weight for {model_id} must be finite and in [0, 2]")
        seen.add(model_id)
        result.append({"modelID": model_id, "weight": float(weight)})
    return result


def validate_canonical_inputs(request: dict[str, Any]) -> None:
    inputs = request.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) > 2:
        raise ValueError("Draw Things supports one first-frame input and an optional H3 last frame")
    if not inputs:
        return
    if len(inputs) == 2:
        frames = request.get("configuration", {}).get("numFrames")
        if (
            isinstance(frames, bool) or not isinstance(frames, int) or frames < 2
            or not all(isinstance(item, dict) for item in inputs)
            or [item.get("role") for item in inputs] != ["first", "last"]
            or isinstance(inputs[1].get("frameIndex"), bool)
            or inputs[1].get("frameIndex") != frames - 1
        ):
            raise ValueError(
                "Draw Things first/last inputs require the resolved final frame"
            )
        for index, item in enumerate(inputs):
            validate_canonical_inputs({**request, "inputs": [
                item if index == 0 else {**item, "role": "first", "frameIndex": 0}
            ]})
        return
    if request.get("operation") != "video":
        raise ValueError("Draw Things first-frame input is supported only for video")
    item = inputs[0]
    required = {"role", "path", "sha256", "frameIndex", "strength"}
    if (
        not isinstance(item, dict)
        or set(item) != required
        or item.get("role") != "first"
        or item.get("frameIndex") != 0
        or item.get("strength") != 1
    ):
        raise ValueError("Draw Things input must be the canonical first-frame contract")
    raw_path, expected = item.get("path"), item.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError("Draw Things first-frame path must be absolute")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError("Draw Things first-frame file is missing")
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise ValueError("Draw Things first-frame hash does not match the exact file")


def validate_discovered_loras(request: dict[str, Any], discovery: dict[str, Any]) -> None:
    loras = canonical_loras(request.get("loras", []))
    if not loras:
        return
    files = discovery.get("files")
    file_ids = (
        set(files)
        if isinstance(files, list) and all(isinstance(item, str) for item in files)
        else set()
    )
    catalog = discovery.get("loras")
    entries = (
        {
            item.get("id"): item
            for item in catalog
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if isinstance(catalog, list)
        else {}
    )
    target = request.get("modelID")
    for lora in loras:
        model_id = lora["modelID"]
        if model_id not in file_ids or model_id not in entries:
            raise ValueError(
                f"Draw Things LoRA {model_id} is not present in verified "
                "discovery files and catalog"
            )
        compatible = entries[model_id].get("compatibleModelIDs")
        if not isinstance(compatible, list) or target not in compatible:
            raise ValueError(f"Draw Things LoRA {model_id} is not compatible with model {target}")
