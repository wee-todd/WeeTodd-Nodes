"""Header-only validation for official LTX 2.5 Gemma 4 text-encoder packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

GEMMA_CONFIG_METADATA_KEY = "gemma_config"
TOKENIZER_JSON_TENSOR_KEY = "tokenizer_json"
HF_ASSET_TENSOR_PREFIX = "hf_asset__"
REQUIRED_SIDECARS = ("tokenizer_config.json", "processor_config.json")
SUPPORTED_MODEL_TYPES = ("gemma4_unified",)


def _parse_json_object(raw: str | None, *, label: str, source: Path) -> dict[str, Any]:
    if raw is None:
        raise ValueError(f"LTX 2.5 Gemma pack {source} is missing {label!r} metadata.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LTX 2.5 Gemma pack {source} contains invalid JSON in {label!r}."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"LTX 2.5 Gemma pack {source} {label!r} must be a JSON object.")
    return value


def inspect_gemma4_pack(path: str | Path) -> dict[str, object]:
    """Validate an LTX 2.5 single-file Gemma pack without loading tensor data.

    Official packs embed the Hugging Face config as metadata and tokenizer/
    processor assets as byte tensors. The inspection deliberately uses only
    the safetensors header so a bad multi-gigabyte checkpoint fails before MLX
    allocates or materializes any weights.
    """
    source = Path(path).expanduser()
    if not source.is_file() or source.suffix != ".safetensors":
        raise FileNotFoundError(f"LTX 2.5 Gemma pack is not a safetensors file: {source}")

    with safe_open(source, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())

    config = _parse_json_object(
        metadata.get(GEMMA_CONFIG_METADATA_KEY),
        label=GEMMA_CONFIG_METADATA_KEY,
        source=source,
    )
    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            "LTX 2.5 generation requires the encode-capable Gemma 4 unified pack; "
            f"got model_type={model_type!r}."
        )

    if TOKENIZER_JSON_TENSOR_KEY not in keys:
        raise ValueError(
            f"LTX 2.5 Gemma pack {source} is missing tensor {TOKENIZER_JSON_TENSOR_KEY!r}."
        )
    sidecar_tensors = {
        key.removeprefix(HF_ASSET_TENSOR_PREFIX)
        for key in keys
        if key.startswith(HF_ASSET_TENSOR_PREFIX)
    }
    missing_sidecars = [
        name for name in REQUIRED_SIDECARS if name not in sidecar_tensors and name not in metadata
    ]
    if missing_sidecars:
        raise ValueError(
            f"LTX 2.5 Gemma pack {source} is missing required embedded assets: "
            + ", ".join(missing_sidecars)
        )

    if "model.layers.0.post_feedforward_layernorm.weight" in keys:
        weight_layout = "comfy_flat"
    elif any(key.startswith("model.language_model.") for key in keys):
        weight_layout = "huggingface_unified"
    else:
        raise ValueError(
            f"LTX 2.5 Gemma pack {source} has no recognized Gemma 4 language-model weights."
        )

    required_projection_prefixes = (
        "text_embedding_projection.video_aggregate_embed.",
        "text_embedding_projection.audio_aggregate_embed.",
    )
    missing_projections = [
        prefix
        for prefix in required_projection_prefixes
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing_projections:
        raise ValueError(
            f"LTX 2.5 Gemma pack {source} is missing trained LTX projection weights: "
            + ", ".join(missing_projections)
        )

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"LTX 2.5 Gemma pack {source} has no text_config object.")
    return {
        "model_type": model_type,
        "gemma_version": config.get("gemma_version"),
        "weight_layout": weight_layout,
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "embedded_assets": sorted(sidecar_tensors),
        "tensor_count": len(keys),
        "connectors_embedded": all(
            any(key.startswith(prefix) for key in keys)
            for prefix in (
                "model.diffusion_model.video_embeddings_connector.",
                "model.diffusion_model.audio_embeddings_connector.",
            )
        ),
    }


def gemma4_mlx_model_config(gemma_config: dict[str, Any]) -> dict[str, Any]:
    """Translate a unified Gemma 4 config into mlx-lm's compatible text model.

    mlx-lm's Gemma 4 implementation can express the unified text backbone, but
    its defaults describe the dense instruct family. In particular, leaving
    per-layer input embeddings enabled would construct parameters that do not
    exist in the unified checkpoint.
    """
    if gemma_config.get("model_type") != "gemma4_unified":
        raise ValueError("Only model_type='gemma4_unified' can be used for LTX 2.5 encoding.")
    text_config = gemma_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Gemma 4 unified config is missing text_config.")
    required = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")
    missing = [name for name in required if name not in text_config]
    if missing:
        raise ValueError("Gemma 4 unified text_config is missing: " + ", ".join(missing))

    translated = dict(text_config)
    translated.update(
        {
            "model_type": "gemma4_text",
            "hidden_size_per_layer_input": 0,
            "vocab_size_per_layer_input": int(text_config.get("vocab_size", 262144)),
            "enable_moe_block": False,
        }
    )
    return {
        "model_type": "gemma4",
        "vocab_size": int(text_config.get("vocab_size", gemma_config.get("vocab_size", 262144))),
        "text_config": translated,
    }


def remap_gemma4_weight_key(key: str, *, layout: str) -> str | None:
    """Map supported pack language-model keys to mlx-lm Gemma 4 names."""
    if layout == "huggingface_unified":
        if not key.startswith("model.language_model."):
            return None
        return "language_model.model." + key.removeprefix("model.language_model.")
    if layout == "comfy_flat":
        for source_prefix in ("model.layers.", "model.embed_tokens.", "model.norm."):
            if key.startswith(source_prefix):
                return "language_model.model." + key.removeprefix("model.")
        return None
    raise ValueError(f"Unsupported Gemma 4 pack weight layout: {layout!r}.")
