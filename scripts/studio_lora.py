"""Payload-free LoRA catalog inspection and Studio's clip model boundary.

Training provenance filters the library, never replaces the renderer's complete
projection, shape, scaling, and specialized-pipeline checks.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from wee_todd_mlx.adapter_contract import inspect_adapter


def inspect_lora(source):
    source = Path(source).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Choose a SafeTensors LoRA file.")
    report = inspect_adapter(source)
    metadata = report["metadata"]
    models = set()
    for key in (
        "model_version",
        "base_model",
        "base_model_name_or_path",
        "ss_base_model_version",
        "modelspec.architecture",
    ):
        value = str(metadata.get(key, "")).lower()
        if "minimax" in value and "h3" in value:
            models.add("h3")
        if key == "model_version" or "ltx" in value:
            version = re.search(r"(?:^|[^0-9])(2)[._-]([0-9]+)(?:[^0-9]|$)", value)
            if version:
                minor = version.group(2)
                if minor not in {"3", "5"}:
                    raise ValueError(f"Unsupported LoRA training version: {value}.")
                models.add("ltx2" + minor)
    if len(models) > 1:
        raise ValueError("LoRA has conflicting training-model metadata.")
    if any(key.startswith("reference_") for key in metadata) or metadata.get(
        "adapter_role", "standard"
    ) not in {"standard", "transformer_lora", "style", "character"}:
        raise ValueError(
            "This specialized adapter belongs in a model/task recipe, not a style LoRA group."
        )
    return {"kind": "lora", "path": str(source), "loraModel": next(iter(models), None)}


def clip_lora(asset, attachment, engine):
    if asset.get("kind") != "lora":
        raise ValueError("A LoRA attachment must reference a SafeTensors adapter.")
    model = asset.get("loraModel")
    if model not in {"h3", "ltx23", "ltx25"} or not (
        model == engine or (model == "ltx23" and engine == "ltx25")
    ):
        raise ValueError("Choose a compatible trained model for this LoRA in the LoRA library.")
    strength = attachment.get("strength", 1)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 2
    ):
        raise ValueError("LoRA strength must be a finite number from 0 to 2.")
    inspected = inspect_lora(asset["path"])
    if inspected["loraModel"] and inspected["loraModel"] != model:
        raise ValueError(
            "The LoRA file declares a different trained model. Reimport it in the LoRA library."
        )
    return {"path": inspected["path"], "strength": strength}
