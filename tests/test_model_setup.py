"""Setup must reject incompatible assets before publishing a reusable profile."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from test_preflight import _component_tree, _json, _portable_optimized_spec, _safetensors


def service():
    from wee_todd_mlx import model_setup

    return model_setup


def h3_components(root):
    return {
        "checkpoint": str(root),
        **{
            key: str(root / key)
            for key in (
                "transformer",
                "text_encoder",
                "processor",
                "tokenizer",
                "video_vae",
                "audio_vae",
            )
        },
    }


def test_catalog_covers_shared_studio_tasks_without_loading_models():
    catalog = service().setup_catalog()
    assert {(p["engine"], p["task"]) for p in catalog} == {
        ("h3", "t2v"),
        ("h3", "fflf"),
        ("h3", "ref2va"),
        ("ltx23", "t2v"),
        ("ltx23", "fflf"),
        ("ltx25", "t2v"),
        ("ltx25", "fflf"),
    }
    assert all(p["components"] and p["description"] for p in catalog)
    assert all(c["kind"] in {"file", "directory"} for p in catalog for c in p["components"])


@pytest.mark.parametrize(
    "preset,task,partition",
    [
        ("h3-text", "t2va", "fl2va"),
        ("h3-image", "fl2va", "fl2va"),
        ("h3-reference", "ref2va", "ref2va"),
    ],
)
def test_h3_creates_unique_validated_recipe(tmp_path, preset, task, partition):
    root = _component_tree(tmp_path)
    manifest = json.loads((root / "model_index.json").read_text())
    manifest["_minimax_h3"] = {"partition": partition, "tasks": [task]}
    _json(root / "model_index.json", manifest)
    output = tmp_path / "profiles"
    result = service().prepare_recipe(preset, h3_components(root), output)
    second = service().prepare_recipe(preset, h3_components(root), output)
    assert result["recipePath"] != second["recipePath"]
    recipe = json.loads(Path(result["recipePath"]).read_text())
    assert recipe["format"] == "weetodd-headless-v2"
    assert recipe["components"]["task"] == task
    assert recipe["config"]["memory_mode"] == "low_memory_bf16"
    assert result["profile"]["id"] == result["recipePath"]
    if task == "t2va":
        from wee_todd_mlx.headless_preflight import preflight_recipe

        assert preflight_recipe(recipe)["weights_loaded"] is False
    else:
        assert any("media" in w.lower() for w in result["warnings"])


def test_invalid_selection_never_creates_or_replaces_recipe(tmp_path):
    root = _component_tree(tmp_path)
    output = tmp_path / "profiles"
    output.mkdir()
    existing = output / "h3-text.json"
    existing.write_text("keep me")
    components = h3_components(root)
    components["transformer"] = components["audio_vae"]
    with pytest.raises(ValueError, match="transformer"):
        service().prepare_recipe("h3-text", components, output)
    assert list(output.iterdir()) == [existing]
    assert existing.read_text() == "keep me"
    with pytest.raises(ValueError, match="Missing"):
        service().prepare_recipe("h3-text", {}, output)


def test_reference_rejects_wrong_partition_and_text_only_paged_encoder(tmp_path):
    spec = _portable_optimized_spec(tmp_path)
    components = {
        k: v for k, v in asdict(spec).items() if k in h3_components(Path(spec.checkpoint))
    }
    with pytest.raises(ValueError, match="task|partition"):
        service().prepare_recipe("h3-reference", components, tmp_path / "profiles")
    with pytest.raises(ValueError, match="vision|text-only"):
        service().prepare_recipe("h3-image", components, tmp_path / "profiles")
    assert not (tmp_path / "profiles").exists()


@pytest.mark.parametrize("filename", ["model_identity.json", "setup_provenance.json"])
@pytest.mark.parametrize(
    "preset,partition,wrong", [("h3-image", "fl2va", "ref2va"), ("h3-reference", "ref2va", "fl2va")]
)
def test_h3_transformer_identity_filters_scanning_and_rejects_wrong_selection(
    tmp_path, filename, preset, partition, wrong
):
    root = _component_tree(tmp_path)
    transformer = root / "transformer"
    manifest = json.loads((root / "model_index.json").read_text())
    manifest["_minimax_h3"] = {"partition": partition, "tasks": [partition]}
    _json(root / "model_index.json", manifest)
    _json(transformer / filename, {"engine": "h3", "partition": wrong})
    report = service().scan_models(preset, [str(transformer)])
    assert report["candidates"]["transformer"] == []
    with pytest.raises(ValueError, match="partition"):
        service().prepare_recipe(preset, h3_components(root), tmp_path / "profiles")
    assert not (tmp_path / "profiles").exists()
    _json(transformer / filename, {"engine": "h3", "partition": partition})
    report = service().scan_models(preset, [str(transformer)])
    assert report["candidates"]["transformer"] == [str(transformer.resolve())]


def test_discovery_uses_headers_not_names_and_deduplicates_cycles(tmp_path):
    root = _component_tree(tmp_path)
    renamed = root / "misleading"
    (root / "transformer").rename(renamed)
    (root / "alias").symlink_to(renamed, target_is_directory=True)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    fake = root / "qwen-transformer.safetensors"
    fake.write_text("not a checkpoint")
    report = service().scan_models("h3-text", [str(root), str(root / "alias")])
    assert report["candidates"]["transformer"] == [str(renamed.resolve())]
    assert str(fake) not in {p for paths in report["candidates"].values() for p in paths}
    assert report["warnings"]


@pytest.mark.parametrize(
    "mode,gb",
    [
        ("invalid", None),
        ("custom", -1),
        ("custom", float("nan")),
        ("custom", True),
    ],
)
def test_memory_input_is_validated_before_output(tmp_path, mode, gb):
    with pytest.raises(ValueError, match="memory|Memory"):
        service().prepare_recipe(
            "h3-text", {}, tmp_path / "profiles", memory_mode=mode, memory_gb=gb
        )
    assert not (tmp_path / "profiles").exists()


def test_custom_memory_reports_estimate_and_lower_memory_uses_real_controls(tmp_path):
    components = h3_components(_component_tree(tmp_path))
    low = service().prepare_recipe("h3-text", components, tmp_path / "profiles", "lower_memory")
    config = json.loads(Path(low["recipePath"]).read_text())["config"]
    assert config["attention_head_chunk_size"] == "2"
    assert config["ffn_row_chunk_size"] == "128"
    custom = service().prepare_recipe("h3-text", components, tmp_path / "profiles", "custom", 1)
    assert any("not a hard" in w for w in custom["warnings"])
    assert any("exceeds" in w for w in custom["warnings"])


def test_cli_and_bridge_use_same_catalog(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cli = subprocess.run(
        [sys.executable, str(root / "scripts/setup_models.py"), "catalog"],
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr
    request = tmp_path / "request.json"
    request.write_text("{}")
    bridge = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/studio_bridge.py"),
            "setup-catalog",
            "--request",
            str(request),
        ],
        capture_output=True,
        text=True,
    )
    assert bridge.returncode == 0, bridge.stdout + bridge.stderr
    assert json.loads(cli.stdout)["presets"] == json.loads(bridge.stdout)["result"]["presets"]


def ltx25_components(root):
    def weight(name, tensors, **metadata):
        path = root / (name + ".safetensors")
        _safetensors(
            path,
            {key: ("F32", [1], 4) for key in tensors},
            {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()},
        )
        return str(path)

    return {
        "transformer_path": weight(
            "arbitrary-a",
            ["model.diffusion_model.patchify_proj.weight"],
            model_version="2.5.0",
            config={
                "transformer": {
                    "caption_proj_before_connector": True,
                    "cross_attention_adaln": True,
                    "ff_bias": False,
                    "audio_ff_bias": True,
                    "use_prompt_adaln_single": True,
                    "use_keyframes_abs_pos_embedding": True,
                }
            },
        ),
        "text_encoder_path": weight(
            "arbitrary-b",
            [
                "tokenizer_json",
                "hf_asset__tokenizer_config.json",
                "hf_asset__processor_config.json",
                "model.language_model.layers.0.weight",
                "text_embedding_projection.video_aggregate_embed.weight",
                "text_embedding_projection.audio_aggregate_embed.weight",
                "model.diffusion_model.video_embeddings_connector.learnable_registers",
                "model.diffusion_model.audio_embeddings_connector.learnable_registers",
            ],
            gemma_config={
                "model_type": "gemma4_unified",
                "gemma_version": "gemma4-12b-ltx-v1",
                "text_config": {"hidden_size": 3840, "num_hidden_layers": 48},
            },
        ),
        "video_vae_path": weight(
            "arbitrary-c",
            ["decoder.conv_out.weight", "encoder.conv_in.weight"],
            model_version="2.5.0",
            config={"vae": {"_class_name": "ConvVideoDecoder", "patch_size": 4}},
        ),
        "audio_vae_path": weight(
            "arbitrary-d",
            ["audio_vae.decoder.conv.weight", "vocoder.weight"],
            model_version="2.5.0",
        ),
        "spatial_upscaler_path": weight(
            "arbitrary-e",
            ["initial_conv.weight"],
            config={
                "_class_name": "LatentUpsampler",
                "in_channels": 128,
                "dims": 3,
                "spatial_upsample": True,
                "temporal_upsample": False,
            },
        ),
    }


def ltx23_components(root):
    bundle = root / "bundle"
    for name in (
        "connector",
        "vae_encoder",
        "vae_decoder",
        "audio_vae",
        "vocoder",
        "transformer-distilled",
        "spatial_upscaler_x2_v1_1",
    ):
        _safetensors(
            bundle / (name + ".safetensors"),
            {"weight": ("F32", [1], 4)},
            {"model_version": "2.3.0"},
        )
    _json(bundle / "spatial_upscaler_x2_v1_1_config.json", {"in_channels": 128})
    gemma = root / "gemma"
    _json(gemma / "config.json", {"model_type": "gemma3", "text_config": {"hidden_size": 3840}})
    _json(gemma / "tokenizer.json", {"version": "1.0"})
    _safetensors(
        gemma / "model.safetensors", {"language_model.model.layers.0.weight": ("F32", [1], 4)}
    )
    return {"model_dir": str(bundle), "gemma_model": str(gemma)}


@pytest.mark.parametrize(
    "engine,factory", [("ltx23", ltx23_components), ("ltx25", ltx25_components)]
)
@pytest.mark.parametrize("suffix", ["text", "image"])
def test_ltx_profiles_validate_complete_stack_and_discover_renamed_components(
    tmp_path, engine, factory, suffix
):
    components = factory(tmp_path)
    if engine == "ltx23" and suffix == "image":
        for name in ("transformer-dev", "ltx-2.3-22b-distilled-lora-384"):
            _safetensors(
                Path(components["model_dir"]) / (name + ".safetensors"),
                {"weight": ("F32", [1], 4)},
                {"model_version": "2.3.0"},
            )
    result = service().prepare_recipe(engine + "-" + suffix, components, tmp_path / "profiles")
    recipe = json.loads(Path(result["recipePath"]).read_text())
    assert recipe["config"]["pipeline_mode"] == (
        "two_stage" if engine == "ltx23" and suffix == "image" else "distilled"
    )
    assert recipe["config"]["low_memory"] is True
    assert result["profile"]["task"] == ("t2v" if suffix == "text" else "fflf")
    scan = service().scan_models(engine + "-" + suffix, [str(tmp_path)])
    for key, path in components.items():
        assert str(Path(path).resolve()) in scan["candidates"][key]


def test_ltx25_rejects_foreign_and_swapped_components_before_publication(tmp_path):
    components = ltx25_components(tmp_path)
    components["audio_vae_path"] = components["video_vae_path"]
    with pytest.raises(ValueError, match="audio_vae"):
        service().prepare_recipe("ltx25-text", components, tmp_path / "profiles")
    assert not (tmp_path / "profiles").exists()


def test_ltx23_rejects_foreign_transformer_header(tmp_path):
    components = ltx23_components(tmp_path)
    _safetensors(
        Path(components["model_dir"]) / "transformer-distilled.safetensors",
        {"weight": ("F32", [1], 4)},
        {"model_version": "2.5.0"},
    )
    with pytest.raises(ValueError, match="2.3"):
        service().prepare_recipe("ltx23-text", components, tmp_path / "profiles")


def test_atomic_publish_failure_cleans_temporary_file(tmp_path, monkeypatch):
    import os

    components = h3_components(_component_tree(tmp_path))

    def fail_link(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="disk failure"):
        service().prepare_recipe("h3-text", components, tmp_path / "profiles")
    assert list((tmp_path / "profiles").iterdir()) == []


def test_h3_compact_encoder_config_is_recognized(tmp_path):
    spec = _portable_optimized_spec(tmp_path)
    _json(
        Path(spec.text_encoder) / "config.json",
        {"model_type": "minimax_h3", "text_encoder": {"hidden": 5120, "layers": 50}},
    )
    report = service().scan_models("h3-text", [spec.text_encoder])
    assert report["candidates"]["text_encoder"] == [str(Path(spec.text_encoder).resolve())]


def test_ltx23_split_manifest_identifies_metadata_free_transformer(tmp_path):
    components = ltx23_components(tmp_path)
    root = Path(components["model_dir"])
    _safetensors(
        root / "transformer-distilled.safetensors",
        {"transformer.patchify_proj.weight": ("F32", [1], 4)},
    )
    _json(
        root / "config.json",
        {
            "model_version": "2.3.0",
            "model_type": "AudioVideo",
            "in_channels": 128,
            "num_layers": 48,
        },
    )
    _json(
        root / "split_model.json",
        {"format": "split", "model_version": "2.3.0", "transformer_variants": ["distilled"]},
    )
    result = service().prepare_recipe("ltx23-text", components, tmp_path / "profiles")
    assert Path(result["recipePath"]).is_file()


def test_ltx25_duration_head_is_not_a_transformer_candidate(tmp_path):
    components = ltx25_components(tmp_path)
    header = service().inspect_safetensors_header(components["transformer_path"])
    _safetensors(
        Path(components["transformer_path"]),
        {"duration_head.weight": ("F32", [1], 4)},
        header["metadata"],
    )
    report = service().scan_models("ltx25-text", [components["transformer_path"]])
    assert report["candidates"]["transformer_path"] == []


def test_ltx23_image_requires_dev_and_refinement_weights(tmp_path):
    components = ltx23_components(tmp_path)
    with pytest.raises((ValueError, OSError), match="transformer-dev|refinement|Dev"):
        service().prepare_recipe("ltx23-image", components, tmp_path / "profiles")


@pytest.mark.parametrize(
    "engine,factory",
    [("h3", lambda root: h3_components(_component_tree(root))), ("ltx25", ltx25_components)],
)
def test_automatic_selects_actual_lower_memory_controls_for_64gb(tmp_path, engine, factory):
    result = service().prepare_recipe(
        engine + "-text", factory(tmp_path), tmp_path / "profiles", memory_gb=64
    )
    config = json.loads(Path(result["recipePath"]).read_text())["config"]
    if engine == "h3":
        assert config["attention_head_chunk_size"] == "2"
    else:
        assert config["low_ram_streaming"] is True


def test_h3_discovery_excludes_foreign_processor_and_internal_weight_pages(tmp_path):
    _portable_optimized_spec(tmp_path)
    foreign = tmp_path / "processor"
    _json(
        foreign / "preprocessor_config.json",
        {"processor_class": "Florence2Processor", "patch_size": 16},
    )
    _json(foreign / "tokenizer_config.json", {"tokenizer_class": "BartTokenizer"})
    _json(foreign / "tokenizer.json", {"version": "1.0"})
    report = service().scan_models("h3-text", [str(tmp_path)])
    assert str(foreign.resolve()) not in report["candidates"]["processor"]
    assert str(foreign.resolve()) not in report["candidates"]["tokenizer"]
    assert not any("/pages/" in p for paths in report["candidates"].values() for p in paths)


def test_h3_discovery_bounded_support_json(tmp_path, monkeypatch):
    root = _component_tree(tmp_path)
    # With a tiny bound, ordinary support configs exceed the inspection budget.
    monkeypatch.setattr(service(), "MAX_CONFIG_BYTES", 4)
    result = service().scan_models("h3-text", [str(root / "processor")])
    assert result["candidates"]["processor"] == []


def test_custom_preserves_preset_controls_without_a_memory_cap(tmp_path):
    components = h3_components(_component_tree(tmp_path))
    result = service().prepare_recipe("h3-text", components, tmp_path / "profiles", "custom")
    config = json.loads(Path(result["recipePath"]).read_text())["config"]
    assert config["attention_head_chunk_size"] == "automatic"


@pytest.mark.parametrize(
    "filename,contents",
    [
        ("config.json", {"hidden_size": 5120, "text_config": None}),
        ("config.json", {"text_config": []}),
        ("config.json", {"text_encoder": "foreign"}),
        ("config.json", {"kwargs": None}),
        ("model_index.json", {"_minimax_h3": None}),
        ("model_index.json", {"_minimax_h3": ["foreign"]}),
        (
            "paged_text_encoder_manifest.json",
            {"format": "weetodd-h3-qwen-paged-v2", "vision": "foreign"},
        ),
    ],
)
def test_h3_scan_continues_past_malformed_foreign_nested_objects(tmp_path, filename, contents):
    good = _portable_optimized_spec(tmp_path / "good")
    foreign = tmp_path / "foreign"
    _json(foreign / "config.json", {"hidden_size": 5120})
    _safetensors(foreign / "model.safetensors", {"model.layers.0.weight": ("F32", [1], 4)})
    _json(foreign / filename, contents)
    result = service().scan_models("h3-text", [str(foreign), str(tmp_path / "good")])
    assert str(Path(good.text_encoder).resolve()) in result["candidates"]["text_encoder"]
    assert str(Path(good.checkpoint).resolve()) in result["candidates"]["checkpoint"]


def test_ltx23_scan_continues_past_null_gemma_text_config(tmp_path):
    components = ltx23_components(tmp_path / "good")
    foreign = tmp_path / "foreign"
    _json(foreign / "config.json", {"model_type": "gemma3", "text_config": None})
    result = service().scan_models("ltx23-text", [str(foreign), str(tmp_path / "good")])
    assert result["candidates"]["gemma_model"] == [str(Path(components["gemma_model"]).resolve())]
