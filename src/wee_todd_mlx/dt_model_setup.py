"""Metadata-only setup for native H3 reads from a user's Draw Things store."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path


def validate_selection(key, path):
    from minimax_h3_mlx.dt_h3_checkpoint import describe_dt_h3
    from minimax_h3_mlx.dt_tensor_store import DTTensorStore

    from .model_setup import _h3_candidate

    if key == "tokenizer":
        return _h3_candidate(key, path, "t2va")
    if key == "dt_transformer":
        return describe_dt_h3(path)
    from minimax_h3_mlx.dt_source import validate_inventory

    with DTTensorStore(path) as store:
        validate_inventory(store.records, "text_encoder" if key == "dt_qwen" else "video_vae")
        if key == "dt_vae":
            validate_inventory(store.records, "audio_vae")
        prefix, count = {"dt_qwen": ("__text_model__", 551), "dt_vae": ("__video_decoder__", 657)}[
            key
        ]
        records = [
            store.validate_tensor(n, row_access=n == "__text_model__[t-tok_embeddings-0-0]")
            for n in store.records
            if n.startswith(prefix)
        ]
        if len(records) != count:
            raise ValueError(f"{key}: not a supported Draw Things H3 component.")
        for r in records:
            if r.codec & 0x10000000:
                store._span(r, store._inline(r))
        if key == "dt_vae" and sum(n.startswith("__audio_decoder__") for n in store.records) != 779:
            raise ValueError("Draw Things H3 VAE must include its audio decoder.")


def write_reference_bundle(selected, destination):
    """Write a new private metadata bundle, never copy or alter model weights."""
    from minimax_h3_mlx import dt_source

    root = Path(destination)
    config = json.loads(Path(dt_source.__file__).with_name("dt_h3_config.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        key: ["weetodd", key]
        for key in (
            "transformer",
            "text_encoder",
            "video_vae",
            "audio_vae",
            "processor",
            "tokenizer",
        )
    }
    manifest["_minimax_h3"] = {
        "partition": "fl2va",
        "tasks": ["t2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    }
    manifest["_class_name"] = "MiniMaxH3Pipeline"
    (root / "model_index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, key in [
        ("text_encoder", "dt_qwen"),
        ("video_vae", "dt_vae"),
        ("audio_vae", "dt_vae"),
    ]:
        directory = root / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config[name], indent=2) + "\n")
        (directory / dt_source.DT_SOURCE_FILE).write_text(
            json.dumps(
                {
                    "format": "weetodd-h3-dt-source-v1",
                    "component": name,
                    "checkpoint": selected[key],
                },
                indent=2,
            )
            + "\n"
        )
    return {
        "checkpoint": str(root),
        "transformer": selected["dt_transformer"],
        "text_encoder": str(root / "text_encoder"),
        "video_vae": str(root / "video_vae"),
        "audio_vae": str(root / "audio_vae"),
        "tokenizer": selected["tokenizer"],
        "processor": selected["tokenizer"],
    }


def prepare_dt_recipe(components, profiles_directory, memory_mode, memory_gb):
    from .model_setup import _preset, _publish_recipe, _recipe

    required = {"dt_transformer", "dt_qwen", "dt_vae", "tokenizer"}
    if not isinstance(components, dict) or set(components) != required:
        raise ValueError("Select the DT H3 transformer, Qwen, VAE and H3 tokenizer.")
    selected = {}
    for key, value in components.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Select a local path for {key}.")
        path = Path(value).expanduser().resolve(strict=True)
        validate_selection(key, path)
        selected[key] = str(path)
    root = Path(profiles_directory).expanduser().resolve()
    bundle = root / ("h3-dt-components-" + uuid.uuid4().hex)
    refs = write_reference_bundle(selected, bundle)
    try:
        preset = _preset("h3-draw-things-text")
        recipe, warnings = _recipe(preset, refs, memory_mode, memory_gb)
        result = _publish_recipe(preset, recipe, warnings, root)
    except BaseException:
        shutil.rmtree(bundle)
        raise
    result["profile"]["name"] = "H3 · Draw Things models · Text to video"
    result["warnings"].append(
        "Experimental DT model reuse: original weights remain read-only. "
        "Text to video only. This runs the WeeTodd renderer; "
        "Draw Things speed and memory do not transfer."
    )
    return result
