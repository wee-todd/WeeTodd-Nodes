"""Small metadata references to original DT checkpoints; never converted weights."""

from __future__ import annotations

import json
import math
from pathlib import Path

DT_SOURCE_FILE = "draw_things_source.json"


def _read_object(file, limit, label):
    if file.stat().st_size > limit:
        raise ValueError(f"{label} is too large.")
    try:
        raw = json.loads(file.read_text())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be a JSON object.") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return raw


def dt_source(directory, component):
    manifest = Path(directory) / DT_SOURCE_FILE
    if not manifest.is_file():
        return None
    raw = _read_object(manifest, 65536, "DT source manifest")
    if raw.get("format") != "weetodd-h3-dt-source-v1" or raw.get("component") != component:
        raise ValueError("Unsupported DT source manifest or component.")
    checkpoint = raw.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("DT source checkpoint must be a nonempty path.")
    source = Path(checkpoint).expanduser()
    if not source.is_absolute():
        source = manifest.parent / source
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("DT source checkpoint reference is not a file.")
    return source


def validate_dt_reference(directory, component):
    raw = _read_object(Path(directory) / "config.json", 1048576, "DT architecture config")
    if component == "text_encoder":
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 25600,
            "num_hidden_layers": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rope_theta": 5000000,
        }
        text = raw.get("text_config", {})
        if not isinstance(text, dict) or any(text.get(k) != v for k, v in expected.items()):
            raise ValueError("DT Qwen architecture must match the H3 32B conditioner.")
    else:
        size = {"video_vae": 24, "audio_vae": 32}[component]
        for key in ("latents_mean", "latents_std"):
            values = raw.get(key)
            if (
                not isinstance(values, list)
                or len(values) != size
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in values
                )
            ):
                raise ValueError(f"DT {component} requires {size} finite {key} values.")
            if key == "latents_std" and any(v <= 0 for v in values):
                raise ValueError("DT latent standard deviations must be positive.")
    return raw


def describe_dt_reference(directory, component):
    """Validate stored tensor metadata without importing MLX or reading weights."""
    from .dt_tensor_store import DTTensorStore

    validate_dt_reference(directory, component)
    source = dt_source(directory, component)
    if source is None:
        raise ValueError("DT source manifest is missing.")
    with DTTensorStore(source) as store:
        validate_inventory(store.records, component)
        records = [
            store.validate_tensor(name, row_access=name == "__text_model__[t-tok_embeddings-0-0]")
            for name in expected_inventory(component)
        ]
        for record in records:
            if record.codec & 0x10000000:
                store._span(record, store._inline(record))
        total = sum(r.elements * 2 for r in records)
        # One language layer plus bounded embedding lookups; VAEs execute in FP32.
        window = 1100000000 if component == "text_encoder" else total * 2
        return {
            "source": source,
            "tensor_count": len(records),
            "tensor_bytes": total if component == "text_encoder" else total * 2,
            "window_bytes": window,
        }


def dt_source_files(directory):
    manifest = Path(directory) / DT_SOURCE_FILE
    if not manifest.is_file():
        return []
    raw = _read_object(manifest, 65536, "DT source manifest")
    component = raw.get("component")
    if component not in {"text_encoder", "video_vae", "audio_vae"}:
        raise ValueError("Unsupported DT source component.")
    source = dt_source(directory, component)
    return [source] + [
        p for suffix in ("-tensordata", "-wal") if (p := Path(str(source) + suffix)).exists()
    ]


def expected_inventory(component):
    return _read_object(
        Path(__file__).with_name("dt_h3_tensor_layout.json"), 1048576, "DT supported tensor layout"
    )[component]


def validate_inventory(records, component):
    expected = expected_inventory(component)
    prefixes = {
        "transformer": ("__dit__",),
        "text_encoder": ("__text_model__",),
        "video_vae": ("__video_decoder__", "__video_encoder__"),
        "audio_vae": ("__audio_decoder__",),
    }[component]
    actual = {n: list(r.shape) for n, r in records.items() if n.startswith(prefixes)}
    if actual != expected:
        raise ValueError(f"DT {component} tensor names or shapes differ from the supported layout.")
