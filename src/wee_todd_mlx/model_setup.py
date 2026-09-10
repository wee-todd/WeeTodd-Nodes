"""Explicit, weight-free model setup shared by Studio and the command line.

Discovery proposes candidates from bounded structural evidence. Existing engine
preflight remains authoritative; neither step qualifies the generated media.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from .asset_registry import MAX_CONFIG_BYTES, describe_asset
from .model_library import inspect_safetensors_header

MAX_SCAN_ENTRIES = 20000
LTX25_DISTILLED_NOTICE = (
    "This preset requires distilled transformer weights; structural preflight cannot "
    "distinguish Dev from distilled or verify trained payload identity. Choose the distilled "
    "component or use the curated preconverted/download package."
)


def setup_catalog() -> list[dict]:
    """Return fresh descriptors, usable even with no installed models."""
    groups = {
        "h3": [
            ("checkpoint", "H3 task manifest", "directory"),
            ("transformer", "H3 transformer", "directory"),
            ("text_encoder", "Qwen3-VL encoder", "directory"),
            ("processor", "Qwen3-VL processor", "directory"),
            ("tokenizer", "Qwen tokenizer", "directory"),
            ("video_vae", "H3 video VAE", "file"),
            ("audio_vae", "H3 audio VAE", "file"),
        ],
        "ltx23": [
            ("model_dir", "LTX 2.3 MLX distilled bundle", "directory"),
            ("gemma_model", "Gemma 3 12B encoder", "directory"),
        ],
        "ltx25": [
            ("transformer_path", "LTX 2.5 distilled transformer", "file"),
            ("text_encoder_path", "LTX 2.5 Gemma 4 pack", "file"),
            ("video_vae_path", "LTX 2.5 video VAE", "file"),
            ("audio_vae_path", "LTX 2.5 audio VAE and vocoder", "file"),
            ("spatial_upscaler_path", "LTX spatial latent upscaler", "file"),
        ],
    }
    result = []
    for engine, title in (("h3", "MiniMax H3"), ("ltx23", "LTX 2.3"), ("ltx25", "LTX 2.5")):
        for suffix, task, label in (
            ("text", "t2v", "Text to video"),
            ("image", "fflf", "Image to video"),
            ("reference", "ref2va", "Reference to video"),
        ):
            if suffix == "reference" and engine != "h3":
                continue
            components = []
            for key, name, kind in groups[engine]:
                if engine == "ltx23" and task == "fflf" and key == "model_dir":
                    name = "LTX 2.3 MLX Dev bundle with refinement LoRA"
                accepts = [kind]
                if (engine == "h3" and key in {"transformer", "video_vae", "audio_vae"}) or (
                    engine == "ltx25" and key in {"transformer_path", "text_encoder_path"}
                ):
                    accepts = ["file", "directory"]
                components.append(dict(key=key, label=name, kind=kind, accepts=accepts))
            result.append(
                dict(
                    id=f"{engine}-{suffix}",
                    name=f"{title} · {label}",
                    engine=engine,
                    task=task,
                    description="Synchronized audio and video with staged model unloading. "
                    + (
                        "Attach clip media after setup."
                        if task != "t2v"
                        else "Describe a scene after setup."
                    ),
                    components=components,
                )
            )
    for suffix, task, label, key, guidance in (
        (
            "control",
            "control",
            "IC-LoRA control",
            "control_lora_path",
            "Attach a preprocessed guide video as Control, then choose its matching guide type.",
        ),
        (
            "ingredients",
            "control",
            "Ingredients reference sheet",
            "ingredients_lora_path",
            "Use an image asset as Ingredients reference sheet. Choose at least 121 output frames "
            "(5 seconds at 24 fps) and describe the sheet and generated scene in the prompt.",
        ),
        (
            "msr",
            "ref2va",
            "MSR image references",
            "msr_lora_path",
            "Use one to five image assets as MSR references. Describe each image and choose its "
            "subject, object, clothing or background role in Conditioning. "
            "Only one background is allowed.",
        ),
    ):
        base = next(p for p in result if p["id"] == "ltx25-text")
        result.append(
            dict(
                id=f"ltx25-{suffix}",
                name=f"LTX 2.5 · {label}",
                engine="ltx25",
                task=task,
                description=f"Distilled full-resolution single-stage generation. {guidance} "
                "Select the dedicated compatible adapter below; "
                "ordinary style LoRAs cannot replace it.",
                components=[
                    dict(c) for c in base["components"] if c["key"] != "spatial_upscaler_path"
                ]
                + [dict(key=key, label=f"LTX 2.5 {label} adapter", kind="file", accepts=["file"])],
            )
        )
    result.append(
        dict(
            id="h3-draw-things-text",
            name="MiniMax H3 · Draw Things models · Text to video",
            engine="h3",
            task="t2v",
            description="Experimental native generation from your existing Draw Things H3 files. "
            "No converted weight copies. Text to video only; "
            "speed and memory differ from Draw Things.",
            components=[
                dict(key=key, label=label, kind=kind, accepts=[kind])
                for key, label, kind in (
                    ("dt_transformer", "Draw Things H3 transformer (.ckpt)", "file"),
                    ("dt_qwen", "Draw Things H3 Qwen encoder (.ckpt)", "file"),
                    ("dt_vae", "Draw Things H3 VAE (.ckpt)", "file"),
                    ("tokenizer", "H3 tokenizer (small support files)", "directory"),
                )
            ],
        )
    )
    for preset in result:
        if preset["engine"] == "ltx25":
            preset["description"] += " " + LTX25_DISTILLED_NOTICE
            if preset["task"] == "fflf":
                preset["description"] += (
                    " In Media & Assets, import an image, select it, then choose Use in clip → "
                    "First frame. Select this recipe and prepare the clip. "
                    "Reference is the separate MSR route."
                )
    return result


def _preset(preset_id):
    for preset in setup_catalog():
        if preset["id"] == preset_id:
            return preset
    raise ValueError(f"Unknown setup preset: {preset_id!r}")


def _json(path):
    if path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError(f"Manifest exceeds bounded inspection size: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _object(value, label):
    """Reject malformed external objects before passing them to engine inspectors."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_h3_transformer_identity(root, task):
    # Common H3 tensor shapes cannot distinguish the learned task partitions.
    # Respect explicit provenance, without guessing from filenames or paths.
    expected = "ref2va" if task == "ref2va" else "fl2va"
    for filename in (
        "config.json",
        "paged_manifest.json",
        "model_identity.json",
        "conversion_provenance.json",
        "setup_provenance.json",
    ):
        if not (root / filename).is_file():
            continue
        document = _json(root / filename)
        records = [document]
        if "_minimax_h3" in document:
            records.append(_object(document["_minimax_h3"], "_minimax_h3"))
        for record in records:
            if "partition" in record and record["partition"] != expected:
                raise ValueError(
                    f"H3 transformer partition must be {expected!r}; "
                    f"{filename} declares {record['partition']!r}"
                )


def _h3_candidate(key, path, task):
    from wee_todd_nodes.preflight import _component_report, validate_task_partition

    if key == "checkpoint":
        manifest = _json(path / "model_index.json")
        info = _object(manifest.get("_minimax_h3", {}), "_minimax_h3")
        tasks = info.get("tasks", [])
        if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
            raise ValueError("_minimax_h3.tasks must be an array of task names")
        if not isinstance(info.get("partition", ""), str):
            raise ValueError("_minimax_h3.partition must be a string")
        validate_task_partition(
            task=task,
            tasks=tasks,
            partition=info.get("partition", ""),
            allow_fl2va_weights_for_ref2va=False,
        )
        return
    from minimax_h3_mlx.dt_h3_checkpoint import is_dt_checkpoint
    from minimax_h3_mlx.dt_source import dt_source

    if (key == "transformer" and path.is_file() and is_dt_checkpoint(path)) or (
        key in {"text_encoder", "video_vae", "audio_vae"} and dt_source(path, key)
    ):
        if task != "t2va":
            raise ValueError("DT direct models currently support text-to-video only.")
        _component_report(key, path)
        return
    if path.is_dir():
        if key == "transformer":
            _validate_h3_transformer_identity(path, task)
        for name in (
            "paged_manifest.json",
            "paged_text_encoder_manifest.json",
            "architecture_config.json",
        ):
            if (path / name).is_file():
                support = _json(path / name)
                if name == "paged_text_encoder_manifest.json" and support.get("vision") is not None:
                    _object(support["vision"], "paged text encoder vision record")
        if key in {"processor", "tokenizer"}:
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ):
                if not (path / name).is_file():
                    continue
                support = _json(path / name)
                for field, allowed in (
                    ("processor_class", {"Qwen3VLProcessor"}),
                    ("tokenizer_class", {"Qwen2Tokenizer", "Qwen2TokenizerFast"}),
                ):
                    if field in support and support[field] not in allowed:
                        raise ValueError(f"{key}: incompatible {field} {support[field]!r}")
            _component_report(key, path, allow_text_only_processor=task == "t2va")
            return
        config_path = path / "config.json"
        if key == "video_vae" and (path / "source/config.json").is_file():
            config_path = path / "source/config.json"
        config = _json(config_path)
        if key == "transformer":
            recognized = (
                config.get("latents_dim") == 24
                and config.get("audio_latents_dim") == 32
                and config.get("text_dim") == 5120
            )
        elif key == "text_encoder":
            text_config = _object(config.get("text_config", {}), "text_config")
            compact_config = _object(config.get("text_encoder", {}), "text_encoder")
            recognized = (
                config.get(
                    "hidden_size",
                    text_config.get("hidden_size", compact_config.get("hidden")),
                )
                == 5120
            )
        elif key == "video_vae":
            recognized = config.get("z_channels", config.get("latent_channels")) == 24
        else:
            audio = _object(config.get("kwargs", config), "audio_vae kwargs")
            recognized = audio.get("vae_latent_channels", audio.get("latent_channels")) == 32
        if not recognized:
            raise ValueError(f"{key}: configuration does not identify the required H3 architecture")
    report = _component_report(key, path, allow_text_only_processor=task == "t2va")
    if path.is_file() and key == "transformer":
        names = inspect_safetensors_header(path, include_tensors=True)["tensors"]
        if not any(name.startswith("blocks.") for name in names):
            raise ValueError("transformer: an internal fixed page is not a complete checkpoint")
    if (
        key == "text_encoder"
        and task != "t2va"
        and report.paging_format
        and not report.paging_vision_bytes
    ):
        raise ValueError(
            "text_encoder is text-only; select vision-capable paged v2 or resident Qwen3-VL"
        )


def _validate_candidate(preset, key, path):
    if preset["id"] == "h3-draw-things-text":
        from .dt_model_setup import validate_selection

        return validate_selection(key, path)
    if preset["engine"] == "h3":
        task = {"t2v": "t2va", "fflf": "fl2va", "ref2va": "ref2va"}[preset["task"]]
        _h3_candidate(key, path, task)
    elif preset["engine"] == "ltx23":
        _ltx23_candidate(key, path, mode="two_stage" if preset["task"] == "fflf" else "distilled")
    else:
        _ltx25_candidate(key, path)


def _decoded_metadata(header):
    result = {}
    for key, value in header["metadata"].items():
        try:
            result[key] = json.loads(value)
        except (ValueError, TypeError):
            result[key] = value
    return result


def _ltx23_candidate(key, path, *, mode="distilled"):
    if key == "gemma_model":
        config = _json(path / "config.json")
        if config.get("model_type") not in {"gemma3", "gemma3_text"}:
            raise ValueError("gemma_model requires a Gemma 3 configuration")
        text_config = _object(config.get("text_config", config), "Gemma text_config")
        if text_config.get("hidden_size") != 3840:
            raise ValueError("gemma_model requires the Gemma 3 12B text architecture")
        if not (path / "tokenizer.json").is_file():
            raise ValueError("Gemma tokenizer.json is missing")
        weights = sorted(path.glob("*.safetensors"))
        if not weights:
            raise ValueError("Gemma weights are missing")
        for file in weights:
            inspect_safetensors_header(file)
        return
    from ltx23_mlx.runtime import _required_files

    for name in _required_files(mode):
        if name.endswith(".json"):
            _json(path / name)
            continue
        weights = (
            [path / name]
            if name.endswith(".safetensors")
            else (
                [path / f"{name}.safetensors"]
                if (path / f"{name}.safetensors").is_file()
                else sorted(path.glob(f"{name}-*-of-*.safetensors"))
            )
        )
        if not weights:
            raise ValueError(f"LTX 2.3 bundle is missing {name}")
        for weight in weights:
            header = inspect_safetensors_header(weight, include_tensors=True)
            if name in {"transformer-distilled", "transformer-dev"}:
                metadata = _decoded_metadata(header)
                version = str(metadata.get("model_version", ""))
                if not version and (path / "split_model.json").is_file():
                    manifest = _json(path / "split_model.json")
                    config = _json(path / "config.json")
                    if (
                        manifest.get("format") == "split"
                        and name.removeprefix("transformer-")
                        in manifest.get("transformer_variants", [])
                        and config.get("model_type") == "AudioVideo"
                        and config.get("in_channels") == 128
                        and config.get("num_layers") == 48
                        and manifest.get("model_version") == config.get("model_version")
                        and any(n.startswith("transformer.") for n in header["tensors"])
                    ):
                        version = str(config["model_version"])
                if version != "2.3" and not version.startswith("2.3."):
                    raise ValueError(
                        "LTX 2.3 transformer requires preserved model_version 2.3 metadata; "
                        "the filename alone cannot establish compatibility"
                    )


def _ltx25_candidate(key, path):
    if key in {"control_lora_path", "ingredients_lora_path", "msr_lora_path"}:
        from ltx25_mlx.transformer import inspect_ltx25_ic_lora, inspect_ltx25_msr_lora

        if key == "msr_lora_path":
            inspect_ltx25_msr_lora(path)
            return
        report = inspect_ltx25_ic_lora(path)
        family = report.get("adapter_family")
        allowed = (
            {"ingredients_reference_sheet"}
            if key == "ingredients_lora_path"
            else {"union_control", "motion_track", "crossview_warp"}
        )
        if report.get("adapter_role") != "ic_lora" or family not in allowed:
            raise ValueError(f"{key}: select a matching IC-LoRA adapter; found {family!r}")
        return
    if path.is_dir():
        # Validate bounded support data before the runtime parses its manifest.
        _json(path / "paged_manifest.json")
        describe_asset(path)
        from ltx25_mlx.paged_checkpoint import LTX25PagedManifest

        manifest = LTX25PagedManifest.load(path)
        expected_kind = {"transformer_path": "transformer", "text_encoder_path": "gemma"}.get(key)
        if manifest.kind != expected_kind:
            raise ValueError(f"{key}: wrong paged component kind")
        metadata = manifest.metadata
        names = set()
        for file in (manifest.fixed_path, *manifest.layer_paths):
            names.update(inspect_safetensors_header(file, include_tensors=True)["tensors"])
    else:
        header = inspect_safetensors_header(path, include_tensors=True)
        metadata = _decoded_metadata(header)
        names = set(header["tensors"])
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{key}: expected object metadata config")
    if key == "text_encoder_path":
        from ltx25_mlx.gemma_pack import inspect_gemma4_pack

        inspect_gemma4_pack(path)
        return
    if key == "transformer_path":
        valid = (
            str(metadata.get("model_version", "")).startswith("2.5")
            and isinstance(config.get("transformer"), dict)
            and any(n.endswith("patchify_proj.weight") for n in names)
        )
    elif key == "video_vae_path":
        valid = isinstance(config.get("vae"), dict) and any(
            n.startswith(("decoder.", "encoder.")) for n in names
        )
    elif key == "audio_vae_path":
        valid = any(n.startswith("audio_vae.decoder.") for n in names) and any(
            n.startswith("vocoder.") for n in names
        )
    else:
        valid = (
            config.get("_class_name") == "LatentUpsampler"
            and config.get("in_channels") == 128
            and config.get("dims") == 3
            and config.get("spatial_upsample") is True
            and not config.get("temporal_upsample", False)
        )
    if not valid:
        raise ValueError(f"{key}: header does not identify a compatible LTX 2.5 component")


def ltx25_recipe_control_families(recipe_path) -> set[str]:
    """Inspect task adapters for automatic routing; full render preflight still follows.

    Neither profile names nor user-editable labels establish adapter compatibility.
    Bound JSON and safetensors reads before invoking native header inspectors.
    """
    recipe = _json(Path(recipe_path))
    if recipe.get("engine") != "ltx25":
        return set()
    config = _object(recipe.get("config", {}), "config")
    if config.get("pipeline_mode", "distilled") != "distilled":
        return set()
    components = _object(recipe.get("components", {}), "components")
    adapters = components.get("ic_loras", [])
    if not isinstance(adapters, list):
        raise ValueError("ic_loras must be an array")
    families = set()
    if adapters:
        from ltx25_mlx.transformer import inspect_ltx25_ic_lora

        for adapter in adapters:
            if not isinstance(adapter, (list, tuple)) or len(adapter) != 2:
                raise ValueError("Each IC-LoRA needs a path and strength")
            if (
                type(adapter[1]) not in {int, float}
                or not math.isfinite(adapter[1])
                or adapter[1] <= 0
            ):
                raise ValueError("IC-LoRA strength must be finite and positive")
            source = Path(adapter[0]).expanduser()
            inspect_safetensors_header(source)
            report = inspect_ltx25_ic_lora(source)
            if report.get("adapter_role") == "ic_lora":
                families.add(report.get("adapter_family"))
        return families
    transformer = components.get("transformer_path")
    if not transformer:
        return families
    source = Path(transformer).expanduser()
    if source.is_dir():
        from ltx25_mlx.paged_checkpoint import LTX25PagedManifest

        _json(source / "paged_manifest.json")
        metadata = LTX25PagedManifest.load(source).metadata
    else:
        metadata = _decoded_metadata(inspect_safetensors_header(source))
    baked = metadata.get("weetodd_baked_loras", [])
    if not isinstance(baked, list) or not all(isinstance(item, dict) for item in baked):
        raise ValueError("Malformed baked adapter metadata")
    return {
        item["adapter_family"]
        for item in baked
        if item.get("adapter_role") == "ic_lora" and isinstance(item.get("adapter_family"), str)
    }


def _ltx_recipe_config(engine, components, memory_mode, task):
    if engine == "ltx23":
        from ltx23_mlx.runtime import LTX23GenerationConfig, LTX23ModelSpec

        config = LTX23GenerationConfig(
            pipeline_mode="two_stage" if task == "fflf" else "distilled",
            stage1_steps=30 if task == "fflf" else 8,
            low_memory=True,
            low_ram_streaming=memory_mode == "lower_memory",
        )
        config.validate()
        spec = LTX23ModelSpec(**components)
        spec.validate(config.pipeline_mode)
        report = spec.inventory(config.pipeline_mode)
    else:
        from ltx25_mlx.runtime import LTX25ComponentSpec, LTX25GenerationConfig

        single_stage = bool(components.get("ic_loras"))
        config = LTX25GenerationConfig(
            low_memory=True,
            low_ram_streaming=memory_mode == "lower_memory",
            ic_lora_single_stage=single_stage,
            stage2_steps=0 if single_stage else 3,
        )
        report = LTX25ComponentSpec(**components).validate(
            config.pipeline_mode, require_spatial_upscaler=not single_stage
        )
        config.validate(scale_factors=tuple(report["video_scale_factors"]))
    return asdict(config), report


def scan_models(preset_id, roots) -> dict:
    """Inspect user-selected roots, deduplicating physical aliases and cycles."""
    preset = _preset(preset_id)
    if not isinstance(roots, (list, tuple)) or not roots:
        raise ValueError("Select at least one model root")
    candidates = {c["key"]: [] for c in preset["components"]}
    warnings, seen, pending = [], set(), [Path(r).expanduser() for r in reversed(roots)]
    count = 0
    while pending:
        path = pending.pop()
        count += 1
        if count > MAX_SCAN_ENTRIES:
            warnings.append("Scan entry limit reached; scan a smaller model folder.")
            break
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            path = path.resolve()
            if path.is_dir():
                pending.extend(
                    reversed(
                        sorted(
                            (
                                p
                                for p in path.iterdir()
                                if p.name not in {".git", ".cache", "pages"}
                                and not p.name.startswith("._")
                            ),
                            key=str,
                        )
                    )
                )
                kind = "directory"
            elif path.suffix == ".ckpt" and preset["id"] == "h3-draw-things-text":
                kind = "file"
            elif path.suffix == ".safetensors":
                kind = "file"
                inspect_safetensors_header(path)
            else:
                continue
            for component in preset["components"]:
                if kind not in component["accepts"]:
                    continue
                try:
                    _validate_candidate(preset, component["key"], path)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                candidates[component["key"]].append(str(path))
        except (OSError, ValueError) as exc:
            warnings.append(f"Could not inspect {path}: {exc}")
    for key, values in candidates.items():
        values.sort()
        if len(values) > 1:
            warnings.append(
                f"Multiple candidates for {key}; explicitly choose the component to reuse."
            )
        elif not values:
            warnings.append(
                f"No compatible {key} found. Select a component directly or prepare it."
            )
    warnings.append(
        "Candidates use manifest/header evidence; setup validates the complete selected stack."
    )
    if preset["engine"] == "ltx25":
        warnings.append(LTX25_DISTILLED_NOTICE)
    return dict(candidates=candidates, warnings=warnings)


def _memory(memory_mode, memory_gb):
    if memory_mode not in {"automatic", "lower_memory", "custom"}:
        raise ValueError("Memory mode must be automatic, lower_memory, or custom")
    if memory_gb is not None and (
        type(memory_gb) not in {int, float} or not math.isfinite(memory_gb) or memory_gb <= 0
    ):
        raise ValueError("Available memory GB must be a finite positive number")


def _recipe(preset, components, memory_mode, memory_gb):
    engine = preset["engine"]
    warnings = [
        "Setup validates components without loading weights; a render has not been qualified."
    ]
    if engine == "ltx25":
        warnings.append(LTX25_DISTILLED_NOTICE)
        components = dict(components)
        for key in ("control_lora_path", "ingredients_lora_path", "msr_lora_path"):
            if key not in components:
                continue
            adapter = components[key]
            if key != "msr_lora_path":
                components.pop(key)
            components["ic_loras"] = [[adapter, 1.0]]
            components.setdefault("spatial_upscaler_path", "")
    if memory_mode == "automatic" and memory_gb is not None and memory_gb <= 64:
        memory_mode = "lower_memory"
        warnings.append(
            "Automatic selected Lower Memory for the supplied memory of 64 GB or less; "
            "this is not a fit guarantee."
        )
    if memory_gb is not None:
        warnings.append(
            f"Available memory input: {memory_gb:g} GB; this is an estimate comparison, "
            "not a hard runtime cap. Use clip settings to adjust resolution and duration."
        )
    if engine == "h3":
        from wee_todd_nodes.preflight import (
            H3ComponentSetSpec,
            H3PreflightRequest,
            preflight_components,
        )
        from wee_todd_nodes.runtime import H3GenerationConfig

        components = dict(
            components, task={"t2v": "t2va", "fflf": "fl2va", "ref2va": "ref2va"}[preset["task"]]
        )
        config = H3GenerationConfig(memory_mode="low_memory_bf16", projection_backend="mlx")
        from minimax_h3_mlx.dt_h3_checkpoint import is_dt_checkpoint

        if memory_mode != "lower_memory" and is_dt_checkpoint(components["transformer"]):
            # Runtime hardware and numerical gates decide whether MPP can replace dense MLX.
            config = H3GenerationConfig(memory_mode="normal", projection_backend="auto")
        fields = asdict(config)
        if memory_mode == "lower_memory":
            fields.update(attention_head_chunk_size="2", ffn_row_chunk_size="128")
        config = H3GenerationConfig(**fields)
        config.validate()
        report = preflight_components(
            H3ComponentSetSpec(**components),
            H3PreflightRequest(
                width=config.width,
                height=config.height,
                steps=config.steps,
                duration_seconds=config.duration_seconds,
                available_memory_gb=memory_gb or 0,
            ),
        ).to_dict()
        warnings.extend(report["warnings"])
    else:
        fields, report = _ltx_recipe_config(engine, components, memory_mode, preset["task"])
        warnings.append(
            "LTX memory depends on live activation and cache sizes; no measured peak "
            "or memory-fit guarantee is available during setup."
        )
    recipe = dict(
        format="weetodd-headless-v2",
        engine=engine,
        candidate=preset["id"],
        components=components,
        config=fields,
        prompt="Describe a continuous scene, its action, camera movement, and synchronized sound.",
        conditioning=dict(version=1, task=preset["task"], inputs=[], audio_policy="generated"),
    )
    if preset["task"] == "t2v":
        from .headless_preflight import preflight_recipe

        preflight_recipe(recipe)
    else:
        warnings.append(
            "Component and generation settings passed preflight. Attach image/reference "
            "media to the clip; task and media preflight run before rendering."
        )
    return recipe, warnings


def prepare_recipe(
    preset_id, components, profiles_directory, memory_mode="automatic", memory_gb=None
) -> dict:
    """Validate selected local assets, then atomically publish a new unique profile."""
    preset = _preset(preset_id)
    _memory(memory_mode, memory_gb)
    if preset_id == "h3-draw-things-text":
        from .dt_model_setup import prepare_dt_recipe

        return prepare_dt_recipe(components, profiles_directory, memory_mode, memory_gb)
    if not isinstance(components, dict):
        raise ValueError("Components must be an object of selected local paths")
    required = {c["key"] for c in preset["components"]}
    if required - components.keys():
        raise ValueError("Missing components: " + ", ".join(sorted(required - components.keys())))
    if components.keys() - required:
        raise ValueError(
            "Unexpected components: " + ", ".join(sorted(components.keys() - required))
        )
    selected = {}
    for component in preset["components"]:
        key = component["key"]
        value = components[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Select a local path for {key}")
        path = Path(value).expanduser().resolve(strict=True)
        kind = "directory" if path.is_dir() else "file"
        if kind not in component["accepts"]:
            raise ValueError(f"{key} requires {' or '.join(component['accepts'])}")
        # Reuse the asset registry's bounded inspection and shard completeness rules.
        describe_asset(path, manifest_only=key == "checkpoint")
        _validate_candidate(preset, key, path)
        selected[key] = str(path)
    recipe, warnings = _recipe(preset, selected, memory_mode, memory_gb)
    return _publish_recipe(preset, recipe, warnings, profiles_directory)


def _publish_recipe(preset, recipe, warnings, profiles_directory):
    preset_id = preset["id"]
    root = Path(profiles_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{preset_id}-{uuid.uuid4().hex}.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".setup-", suffix=".tmp", dir=root, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(recipe, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link() publishes a fully flushed file and fails if any target exists.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    profile = dict(
        id=str(target),
        name=target.stem.replace("_", " "),
        engine=preset["engine"],
        task=preset["task"],
        width=recipe["config"]["width"],
        height=recipe["config"]["height"],
    )
    return dict(recipePath=str(target), profile=profile, warnings=warnings)
