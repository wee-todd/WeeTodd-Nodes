import json
from pathlib import Path

import pytest
from test_studio_lora import recipe_request

from wee_todd_mlx import model_setup


def test_ltx25_guided_catalog_requires_dedicated_adapters():
    presets = {p["id"]: p for p in model_setup.setup_catalog()}
    for suffix, key in [
        ("ingredients", "ingredients_lora_path"),
        ("msr", "msr_lora_path"),
        ("control", "control_lora_path"),
    ]:
        preset = presets["ltx25-" + suffix]
        assert key in {c["key"] for c in preset["components"]}
        assert "spatial_upscaler_path" not in {c["key"] for c in preset["components"]}
    assert "First frame" in presets["ltx25-image"]["description"]


def test_msr_editor_maps_all_controls_and_preserves_recipe_options(tmp_path, monkeypatch):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["conditioning"] = dict(
        version=1, task="ref2va", inputs=[{"id": "obsolete"}], audio_policy="generated"
    )
    profile.write_text(json.dumps(recipe))
    clip["attachments"] = [
        dict(
            id="ref",
            assetID="image",
            role="reference",
            description="Red coat",
            referenceRole="clothing",
            referencePriority="supporting",
            referenceFrames="33",
            referenceSizePolicy="quality",
            attentionStrength=0.4,
        )
    ]
    request["project"]["assets"] = [
        dict(id="image", kind="image", path="/tmp/coat.png", name="Coat")
    ]
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    recipe, _ = compose(request)
    contract = recipe["conditioning"]
    assert contract["audio_policy"] == "generated"
    assert len(contract["inputs"]) == 1
    item = contract["inputs"][0]
    assert item["reference_role"] == "clothing"
    assert item["reference_priority"] == "supporting"
    assert item["reference_frames"] == "33"
    assert item["reference_size_policy"] == "quality"
    assert item["attention_strength"] == 0.4


@pytest.mark.parametrize("task", ["a2v", "ref2va"])
def test_removing_clip_attachments_respects_selected_recipe_tasks(tmp_path, monkeypatch, task):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    clip["attachments"] = []
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["conditioning"] = dict(version=1, task=task, inputs=[{"id": "obsolete"}])
    profile.write_text(json.dumps(recipe))
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    if task == "ref2va":
        # A reference-only recipe cannot silently become T2V when attachments disappear.
        with pytest.raises(ValueError, match="ltx25 t2v.*selected recipe"):
            compose(request)
        assert json.loads(profile.read_text()) == recipe
        return
    recipe, _ = compose(request)
    assert recipe["conditioning"]["inputs"] == []
    assert recipe["conditioning"]["task"] == "t2v"


@pytest.mark.parametrize("same_image,expected_role", [(True, "clothing"), (False, "subject")])
def test_unset_msr_options_inherit_only_matching_attached_image(
    tmp_path, monkeypatch, same_image, expected_role
):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    clip["attachments"] = [dict(id="new-id", assetID="image", role="reference")]
    request["project"]["assets"] = [
        dict(id="image", kind="image", name="Coat", path="/tmp/coat.png")
    ]
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["conditioning"] = dict(
        version=1,
        task="ref2va",
        inputs=[
            dict(
                id="old-id",
                role="reference",
                path="/tmp/coat.png" if same_image else "/tmp/other.png",
                reference_role="clothing",
                attention_strength=0.25,
            )
        ],
    )
    profile.write_text(json.dumps(recipe))
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    recipe, _ = compose(request)
    item = recipe["conditioning"]["inputs"][0]
    assert item["id"] == "new-id"
    assert item["reference_role"] == expected_role
    assert item["attention_strength"] == (0.25 if same_image else 1)


@pytest.mark.parametrize(
    "preset,key", [("ltx25-msr", "msr_lora_path"), ("ltx25-ingredients", "ingredients_lora_path")]
)
def test_adapter_setup_rejects_regular_model_weights(tmp_path, preset, key):
    from test_model_setup import ltx25_components

    components = ltx25_components(tmp_path)
    components[key] = components.pop("spatial_upscaler_path")
    with pytest.raises(ValueError):
        model_setup.prepare_recipe(preset, components, tmp_path / "profiles")
    assert not (tmp_path / "profiles").exists()


@pytest.mark.parametrize(
    "suffix,family", [("ingredients", "ingredients_reference_sheet"), ("control", "union_control")]
)
def test_guided_ic_setup_validates_and_publishes_single_stage_recipe(tmp_path, suffix, family):
    from test_model_setup import ltx25_components
    from test_preflight import _safetensors

    components = ltx25_components(tmp_path)
    components.pop("spatial_upscaler_path")
    adapter = tmp_path / "adapter.safetensors"
    _safetensors(
        adapter,
        {
            "transformer_blocks.0.attn1.to_q.lora_A.weight": ("F32", [2, 4], 32),
            "transformer_blocks.0.attn1.to_q.lora_B.weight": ("F32", [4, 2], 32),
        },
        {
            "model_version": "2.5.0",
            "adapter_family": family,
            "reference_downscale_factor": "2" if family == "union_control" else "1",
            "reference_temporal_scale_factor": "1",
        },
    )
    components[suffix + "_lora_path"] = str(adapter)
    result = model_setup.prepare_recipe("ltx25-" + suffix, components, tmp_path / "profiles")
    recipe = json.loads(Path(result["recipePath"]).read_text())
    assert recipe["components"]["ic_loras"] == [[str(adapter), 1.0]]
    assert recipe["components"]["spatial_upscaler_path"] == ""
    assert recipe["config"]["ic_lora_single_stage"] is True
    assert recipe["config"]["stage2_steps"] == 0


@pytest.mark.parametrize(
    "engine,value,expected", [("h3", None, 4), ("h3", 0, 0), ("h3", 8, 8), ("ltx25", 8, 4)]
)
def test_h3_clip_cache_override_preserves_recipe_default(
    tmp_path, monkeypatch, engine, value, expected
):
    request, compose = recipe_request(tmp_path, engine=engine)
    clip = request["project"]["clips"][0]
    clip["attachments"] = []
    clip["h3PagingCacheGB"] = value
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["config"]["paging_cache_gb"] = 4
    profile.write_text(json.dumps(recipe))
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    recipe, _ = compose(request)
    assert recipe["config"]["paging_cache_gb"] == expected


def test_msr_setup_validates_learned_slots_and_publishes_matching_adapter(tmp_path):
    import math

    from test_model_setup import ltx25_components
    from test_preflight import _safetensors

    from ltx25_mlx.transformer import _LTX25_MSR_SLOT_SHAPES

    components = ltx25_components(tmp_path)
    components.pop("spatial_upscaler_path")
    adapter = tmp_path / "msr.safetensors"
    tensors = {
        "reference_slot_embedding." + name: ("F32", list(shape), math.prod(shape) * 4)
        for name, shape in _LTX25_MSR_SLOT_SHAPES.items()
    }
    for block in range(48):
        for target in [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out",
            "audio_attn1.to_q",
            "audio_attn1.to_k",
            "audio_attn1.to_v",
            "audio_attn1.to_out",
            "attn2.to_q",
            "attn2.to_k",
        ]:
            prefix = f"transformer_blocks.{block}.{target}"
            tensors[prefix + ".lora_A.weight"] = ("F32", [128, 4], 2048)
            tensors[prefix + ".lora_B.weight"] = ("F32", [4, 128], 2048)
    _safetensors(
        adapter,
        tensors,
        {
            "model_version": "2.5.0",
            "reference_slot_embedding_type": "fourier_mlp",
            "reference_token_order": "prepend",
            "reference_slot_time_offsets": "pic1_based_negative_time",
        },
    )
    components["msr_lora_path"] = str(adapter)
    result = model_setup.prepare_recipe("ltx25-msr", components, tmp_path / "profiles")
    recipe = json.loads(Path(result["recipePath"]).read_text())
    assert recipe["components"]["msr_lora_path"] == str(adapter)
    assert recipe["components"]["ic_loras"] == [[str(adapter), 1.0]]
    assert recipe["config"]["ic_lora_single_stage"] is True
    assert recipe["conditioning"]["task"] == "ref2va"


@pytest.mark.parametrize(
    "control_type,expected",
    [
        ("ingredients_reference_sheet", "z-renamed-sheet"),
        ("motion_track", "ltx25-control-b-motion"),
        ("canny_edges", "ltx25-control-a-union"),
    ],
)
def test_automatic_control_matches_adapter_family_not_recipe_filename(
    tmp_path, monkeypatch, control_type, expected
):
    from test_preflight import _safetensors

    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    base = json.loads(Path(clip["profileID"]).read_text())
    for name, family in [
        ("ltx25-control-a-union", "union_control"),
        ("ltx25-control-b-motion", "motion_track"),
        ("z-renamed-sheet", "ingredients_reference_sheet"),
    ]:
        adapter = tmp_path / (name + ".safetensors")
        _safetensors(
            adapter,
            {
                "transformer_blocks.0.attn1.to_q.lora_A.weight": ("F32", [2, 4], 32),
                "transformer_blocks.0.attn1.to_q.lora_B.weight": ("F32", [4, 2], 32),
            },
            {
                "model_version": "2.5.0",
                "adapter_family": family,
                "reference_downscale_factor": "1"
                if family == "ingredients_reference_sheet"
                else "2",
                "reference_temporal_scale_factor": "1",
            },
        )
        recipe = dict(
            base,
            components={"ic_loras": [[str(adapter), 1]]},
            conditioning={"version": 1, "task": "control", "inputs": []},
        )
        (tmp_path / (name + ".json")).write_text(json.dumps(recipe))
    clip["profileID"] = "auto"
    clip["attachments"] = [
        dict(id="guide", assetID="guide", role="control", controlType=control_type)
    ]
    request["project"]["assets"] = [
        dict(
            id="guide",
            kind="image" if control_type == "ingredients_reference_sheet" else "video",
            path="/tmp/guide",
            name="Guide",
        )
    ]
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    _, report = compose(request)
    assert report["profile"] == expected


def test_automatic_control_refuses_recipe_with_no_matching_adapter(tmp_path, monkeypatch):
    request, compose = recipe_request(tmp_path)
    clip = request["project"]["clips"][0]
    profile = Path(clip["profileID"])
    recipe = json.loads(profile.read_text())
    recipe["conditioning"] = dict(version=1, task="control", inputs=[])
    profile.write_text(json.dumps(recipe))
    clip["profileID"] = "auto"
    clip["attachments"] = [
        dict(id="guide", assetID="guide", role="control", controlType="motion_track")
    ]
    request["project"]["assets"] = [dict(id="guide", kind="video", path="/tmp/guide", name="Guide")]
    monkeypatch.setattr("wee_todd_mlx.task_conditioning.validate_conditioning", lambda r: {})
    with pytest.raises(ValueError, match="ltx25 control"):
        compose(request)


def test_control_routing_reads_baked_checkpoint_metadata_and_rejects_oversized_header(tmp_path):
    import struct

    from test_preflight import _safetensors

    from wee_todd_mlx.model_library import MAX_HEADER_BYTES

    transformer = tmp_path / "renamed-transformer.safetensors"
    _safetensors(
        transformer,
        {"weight": ("F32", [1], 4)},
        {
            "weetodd_baked_loras": json.dumps(
                [{"adapter_role": "ic_lora", "adapter_family": "ingredients_reference_sheet"}]
            )
        },
    )
    profile = tmp_path / "renamed-profile.json"
    profile.write_text(
        json.dumps(dict(engine="ltx25", components={"transformer_path": str(transformer)}))
    )
    assert model_setup.ltx25_recipe_control_families(profile) == {"ingredients_reference_sheet"}
    transformer.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1))
    with pytest.raises(ValueError, match="header length"):
        model_setup.ltx25_recipe_control_families(profile)
