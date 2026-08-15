import hashlib
import json
from pathlib import Path

import pytest

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parents[1]
CORE_NODES = {
    "GetVideoComponents",
    "LoadImage",
    "LoadVideo",
    "MarkdownNote",
    "Note",
    "Video Slice",
}
PROFILE_POLICIES = {
    "speed": {
        "steps": 5,
        "size": (640, 384),
        "memory": "low_memory_bf16",
        "preset": "Turbo — drbaph v4 step-600 — 384p low-memory — 5 points / 4 evaluations",
    },
    "balance": {
        "steps": 7,
        "size": (896, 512),
        "memory": "normal",
        "preset": "Staged Turbo — drbaph v4 step-600 — 2 base + 4 Turbo evaluations",
    },
    "performance": {
        "steps": 20,
        "size": (896, 512),
        "memory": "normal",
        "preset": "Dense baseline — 20 points / 19 evaluations",
    },
}
PORTABLE_T2VA_COMPONENTS = [
    "MiniMax-H3/FL2VA",
    "t2va",
    "MiniMax-H3/transformers/q8_extended_paged",
    "MiniMax-H3/text_encoders/q8-paged",
    "MiniMax-H3/FL2VA/processor",
    "MiniMax-H3/FL2VA/tokenizer",
    "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
    "MiniMax-H3/FL2VA/audio_vae",
    False,
]
PORTABLE_T2VA_INPUTS = dict(
    zip(
        (
            "checkpoint",
            "task",
            "transformer",
            "text_encoder",
            "processor",
            "tokenizer",
            "video_vae",
            "audio_vae",
            "allow_fl2va_weights_for_ref2va",
        ),
        PORTABLE_T2VA_COMPONENTS,
        strict=True,
    )
)

UI_WORKFLOW_PATHS = sorted((ROOT / "workflows").rglob("*.json"))
API_WORKFLOW_PATHS = sorted((ROOT / "examples").glob("*_api.json"))
PRIMITIVE_WIDGET_TYPES = {"BOOLEAN", "FLOAT", "INT", "STRING"}
NOTE_NODE_TYPES = {"MarkdownNote", "Note"}


def _execution_node_map(workflow):
    return {
        node["id"]: node for node in workflow["nodes"] if node.get("type") not in NOTE_NODE_TYPES
    }


@pytest.mark.parametrize("path", UI_WORKFLOW_PATHS, ids=lambda path: path.name)
def test_shipped_ui_workflows_include_setup_note(path):
    workflow = json.loads(path.read_text())
    notes = [
        node
        for node in workflow["nodes"]
        if node.get("type") == "MarkdownNote" and node.get("title") == "Setup and model downloads"
    ]

    assert len(notes) == 1
    text = notes[0]["widgets_values"][0]
    assert "https://huggingface.co/" in text
    assert "Queue the workflow" in text
    assert "ComfyUI/models/" in text


def _input_schema(node_type):
    schema = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
    return [
        (name, specification)
        for group in ("required", "optional")
        for name, specification in schema.get(group, {}).items()
    ]


def _widget_schema(node):
    connected = {item["name"] for item in node.get("inputs", []) if item.get("link") is not None}
    widgets = []
    for name, specification in _input_schema(node["type"]):
        input_type = specification[0]
        options = specification[1] if len(specification) > 1 else {}
        if name in connected or options.get("forceInput"):
            continue
        if isinstance(input_type, list) or input_type in PRIMITIVE_WIDGET_TYPES:
            widgets.append((name, input_type))

    # The H3 resolution extension deliberately renders the optional size slider beside
    # the aspect-ratio selector, rather than after every required widget.
    if node["type"] == "WeeToddH3GenerationConfig":
        short_edge_index = next(i for i, item in enumerate(widgets) if item[0] == "short_edge")
        short_edge = widgets.pop(short_edge_index)
        aspect_ratio_index = next(i for i, item in enumerate(widgets) if item[0] == "aspect_ratio")
        widgets.insert(aspect_ratio_index + 1, short_edge)
    return widgets


def _assert_widget_value_matches(value, input_type):
    if isinstance(input_type, list):
        assert value in input_type
    elif input_type == "BOOLEAN":
        assert isinstance(value, bool)
    elif input_type == "FLOAT":
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
    elif input_type == "INT":
        assert isinstance(value, int) and not isinstance(value, bool)
    elif input_type == "STRING":
        assert isinstance(value, str)


@pytest.mark.parametrize("path", UI_WORKFLOW_PATHS, ids=lambda path: path.name)
def test_shipped_ui_workflows_match_current_node_contracts(path):
    workflow = json.loads(path.read_text())
    nodes = _execution_node_map(workflow)

    for node in nodes.values():
        if node["type"] in CORE_NODES:
            continue
        assert node["type"] in NODE_CLASS_MAPPINGS
        schema = dict(_input_schema(node["type"]))
        input_names = [item["name"] for item in node.get("inputs", [])]
        expected_input_names = [
            name for name, _ in _input_schema(node["type"]) if name in input_names
        ]
        assert input_names == expected_input_names
        for item in node.get("inputs", []):
            assert item["type"] == schema[item["name"]][0]

        widget_schema = _widget_schema(node)
        widget_values = node.get("widgets_values", [])
        assert len(widget_values) == len(widget_schema), (
            f"{path.name}: {node['type']} {node['id']} has stale widget count"
        )
        for value, (_, input_type) in zip(widget_values, widget_schema, strict=True):
            _assert_widget_value_matches(value, input_type)

        node_class = NODE_CLASS_MAPPINGS[node["type"]]
        return_types = list(getattr(node_class, "RETURN_TYPES", ()))
        return_names = list(getattr(node_class, "RETURN_NAMES", ()))
        outputs = node.get("outputs", [])
        assert [item["type"] for item in outputs] == return_types
        if return_names:
            assert [item["name"] for item in outputs] == return_names

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        origin = nodes[origin_id]["outputs"][origin_slot]
        target = nodes[target_id]["inputs"][target_slot]
        assert origin["type"] == target["type"] == link_type
        assert link_id in origin["links"]
        assert target["link"] == link_id


@pytest.mark.parametrize("path", API_WORKFLOW_PATHS, ids=lambda path: path.name)
def test_shipped_api_workflows_match_current_node_contracts(path):
    prompt = json.loads(path.read_text())
    for node in prompt.values():
        node_type = node["class_type"]
        if node_type in CORE_NODES:
            continue
        assert node_type in NODE_CLASS_MAPPINGS
        input_schema = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
        required = input_schema.get("required", {})
        allowed = set(required) | set(input_schema.get("optional", {}))
        assert set(required) <= set(node["inputs"]), (
            f"{path.name}: {node_type} missing required input"
        )
        assert set(node["inputs"]) <= allowed, f"{path.name}: {node_type} has unknown input"
        for name, value in node["inputs"].items():
            input_type = (required | input_schema.get("optional", {}))[name][0]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                continue
            if isinstance(input_type, list):
                assert value in input_type


def test_ltx23_standalone_api_is_registered_and_preflighted():
    prompt = json.loads((ROOT / "examples" / "ltx23_t2va_two_stage_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["1"]["inputs"]["model_directory"] == "LTX-2.3/q8"
    assert prompt["2"]["inputs"]["pipeline_mode"] == "two_stage"
    assert prompt["2"]["inputs"]["stage1_steps"] == 0
    assert prompt["3"]["inputs"] == {"model": ["1", 0], "config": ["2", 0]}
    assert prompt["4"]["inputs"]["model"] == ["3", 0]
    assert prompt["4"]["inputs"]["unload_after_generate"] is True


def test_ltx23_standalone_ui_workflow_links_are_consistent():
    workflow = json.loads(
        (ROOT / "workflows/balance/t2v/ltx23_two_stage.json").read_text()
    )
    nodes = _execution_node_map(workflow)

    assert len(nodes) == 4
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS)
    assert nodes[2]["widgets_values"] == [
        "two_stage",
        704,
        448,
        5.0,
        24.0,
        0,
        0,
        0,
        3.0,
        1.0,
        True,
        False,
    ]
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


def test_ltx25_high_quality_workflow_uses_verified_preset_and_prompt():
    workflow = json.loads(
        (ROOT / "workflows/performance/t2v/ltx25_1920x1088_two_stage.json").read_text()
    )
    nodes = _execution_node_map(workflow)

    assert len(nodes) == 4
    assert nodes[2]["widgets_values"] == [
        "High quality — 1920×1088, 5 s, reference FP32, 8 ancestral + 3 deterministic",
        1920,
        1088,
        5.0,
        24.0,
        584293325,
        True,
        False,
        "official_1024",
        "reference_fp32",
    ]
    assert nodes[4]["widgets_values"][1] == "WeeTodd/LTX25_1920x1088_high_quality"
    prompt = nodes[4]["widgets_values"][0]
    assert "quiet neighborhood bakery at dawn" in prompt
    assert "Well, you are ambitious." in prompt
    assert "clean synchronized speech" in prompt


def test_ltx25_guided_hq_workflow_pins_dev_model_and_res2s_recipe():
    workflow = json.loads(
        (ROOT / "workflows/performance/t2v/ltx25_768x512_guided_hq.json").read_text()
    )
    nodes = _execution_node_map(workflow)
    guided_loader = next(
        node for node in nodes.values() if node["type"] == "WeeToddLTX25GuidedModelLoader"
    )
    quality = next(
        node for node in nodes.values() if node["type"] == "WeeToddLTX25QualityMode"
    )

    assert guided_loader["widgets_values"] == [
        "ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    ]
    assert quality["widgets_values"][0] == "HQ guided — 15 res_2s + 3 deterministic"


@pytest.mark.parametrize(
    "filename",
    (
        "../../balance/t2v/ltx25_768x512_dfr_conv_vae.json",
        "ltx25_768x512_dfr_diffusion_vae.json",
        "ltx25_768x512_dfr_diffusion_vae_metal_tiled.json",
        "ltx25_768x512_dfr_temporal_48fps.json",
    ),
)
def test_ltx25_prebaked_dfr_workflows_enable_required_page_streaming(filename):
    workflow = json.loads((ROOT / "workflows/performance/t2v" / filename).resolve().read_text())
    nodes = _execution_node_map(workflow)
    loader = next(node for node in nodes.values() if node["type"] == "WeeToddLTX25ComponentLoader")
    config = next(node for node in nodes.values() if node["type"] == "WeeToddLTX25GenerationConfig")
    detailing = next(node for node in nodes.values() if node["type"] == "WeeToddLTX25DFRDetailing")

    assert loader["widgets_values"][0].endswith("-q8-paged")
    assert config["widgets_values"][7] is True
    assert detailing["widgets_values"][2].endswith("-q8-paged")


def test_ltx25_accelerated_dfr_workflow_uses_measured_metal_query_tile():
    workflow = json.loads(
        (
            ROOT
            / "workflows/performance/t2v/ltx25_768x512_dfr_diffusion_vae_metal_tiled.json"
        ).read_text()
    )
    optimization = next(
        node
        for node in _execution_node_map(workflow).values()
        if node["type"] == "WeeToddLTX25DiffVAEOptimization"
    )
    assert optimization["widgets_values"] == [
        "metal_na3d_query_tiled_experimental",
        65536,
        4,
        32,
    ]


def test_ltx25_any_video_upscale_workflow_uses_native_movie_components():
    workflow = json.loads(
        (ROOT / "workflows/balance/video-upscale/ltx25_any_video_pixel_spatial_2x.json").read_text()
    )
    nodes = _execution_node_map(workflow)

    assert nodes[1]["type"] == "LoadVideo"
    assert nodes[2]["type"] == "GetVideoComponents"
    assert nodes[4]["type"] == "WeeToddLTX25VideoUpscale"
    assert nodes[4]["inputs"][1] == {"name": "images", "type": "IMAGE", "link": 2}
    assert nodes[4]["inputs"][2] == {"name": "fps", "type": "FLOAT", "link": 4}
    assert nodes[4]["inputs"][5] == {"name": "audio", "type": "AUDIO", "link": 3}
    assert nodes[4]["widgets_values"][0] == "pixel spatial IC-LoRA 2x (recommended)"
    assert nodes[4]["widgets_values"][3] == "center crop to 32px grid (recommended)"
    assert nodes[4]["widgets_values"][4] == 0.35


def test_h3_to_ltx23_upscale_api_preserves_comfy_image_and_audio_contracts():
    prompt = json.loads((ROOT / "examples" / "h3_to_ltx23_2x_upscale_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["6"]["class_type"] == "WeeToddH3VideoVAEDecode"
    assert prompt["7"]["class_type"] == "WeeToddH3AudioVAEDecode"
    assert prompt["9"]["inputs"]["upscaler_name"] == "spatial_upscaler_x2_v1_1"
    assert prompt["10"]["inputs"]["images"] == ["6", 0]
    assert prompt["10"]["inputs"]["audio"] == ["7", 0]
    assert prompt["10"]["inputs"]["fps"] == 24.0
    assert prompt["1"]["inputs"] == PORTABLE_T2VA_INPUTS


def test_h3_native_hires_fix_api_preserves_audio_and_publishes_refined_latents():
    prompt = json.loads((ROOT / "examples" / "h3_native_hires_fix_1p5x_api.json").read_text())

    assert prompt["3"]["inputs"]["steps"] == 8
    assert prompt["5"]["inputs"]["unload_after_sample"] is False
    assert prompt["6"]["class_type"] == "WeeToddH3LatentHiresFix"
    assert prompt["6"]["inputs"]["source_latents"] == ["5", 0]
    assert prompt["6"]["inputs"]["scale"] == "1.5x — balanced"
    assert prompt["6"]["inputs"]["latent_resize_method"] == "bilinear"
    assert prompt["7"]["inputs"]["latents"] == ["6", 0]
    assert prompt["7"]["inputs"]["sampling_info"] == ["6", 2]


@pytest.mark.parametrize("profile", PROFILE_POLICIES)
@pytest.mark.parametrize("task", ["t2v", "i2v", "fflf2va", "ref2va"])
def test_profiled_h3_workflow_matrix_is_complete(profile, task):
    path = ROOT / "workflows" / profile / task / f"h3_{task}_{profile}.json"
    workflow = json.loads(path.read_text())
    nodes = list(_execution_node_map(workflow).values())
    policy = PROFILE_POLICIES[profile]

    config = next(node for node in nodes if node["type"] == "WeeToddH3GenerationConfig")
    preset = next(node for node in nodes if node["type"] == "WeeToddH3ValidatedSamplingPreset")
    previews = [node for node in nodes if node["type"] == "WeeToddH3PreviewOverride"]
    loaders = [node for node in nodes if node["type"] in {"LoadImage", "LoadVideo"}]

    assert config["widgets_values"][1] == policy["steps"]
    assert tuple(config["widgets_values"][7:9]) == policy["size"]
    assert config["widgets_values"][10] == policy["memory"]
    assert preset["widgets_values"] == [policy["preset"]]
    assert len(previews) == 1
    assert all(node["widgets_values"][0] == "" for node in loaders)

    component = next(node for node in nodes if node["type"] == "WeeToddH3ComponentLoader")
    if task == "t2v":
        assert component["widgets_values"][1] == "t2va"
    elif task == "ref2va":
        assert component["widgets_values"][1] == "ref2va"
        assert component["widgets_values"][8] is False
    else:
        assert component["widgets_values"][1] == "fl2va"

    if task == "fflf2va":
        assert any(node["type"] == "WeeToddH3FirstLastFrame" for node in nodes)


def test_h3_15_second_one_shot_workflow_is_the_portable_quality_control():
    prompt = json.loads(
        (ROOT / "examples" / "h3_768p_15s_one_clip_staged_turbo_api.json").read_text()
    )

    assert prompt["1"]["inputs"] == PORTABLE_T2VA_INPUTS
    assert prompt["2"]["inputs"]["duration_seconds"] == 15.0
    assert prompt["2"]["inputs"]["seed"] == 20260811
    assert prompt["2"]["inputs"]["resolution_mode"] == "exact dimensions"
    assert prompt["2"]["inputs"]["custom_width"] == 1344
    assert prompt["2"]["inputs"]["custom_height"] == 768
    assert prompt["3"]["inputs"]["preset"] == (
        "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline"
    )
    assert prompt["6"]["inputs"]["loras"] == ["3", 1]
    assert prompt["6"]["inputs"]["unload_after_sample"] is True
    assert prompt["7"]["class_type"] == "WeeToddH3DirectPublishLatents"
    assert prompt["7"]["inputs"]["sampling_info"] == ["6", 1]
    assert hashlib.sha256(prompt["5"]["inputs"]["prompt"].encode()).hexdigest() == (
        "056473f39220c73477b8a9ef6d0cb5f322c93ccf508dee509657da86948f50c9"
    )


def test_h3_15_second_chain_uses_four_windows_and_direct_join_repair():
    prompt = json.loads(
        (ROOT / "examples" / "h3_768p_15s_four_window_join_repair_api.json").read_text()
    )

    assert prompt["1"]["inputs"] == PORTABLE_T2VA_INPUTS
    assert prompt["2"]["inputs"]["duration_seconds"] == 4.0
    assert prompt["2"]["inputs"]["seed"] == 20260812
    assert prompt["3"]["inputs"]["preset"] == (
        "Chained staged Turbo — drbaph v4 step-600 — 4 windows / 22-frame context"
    )
    assert prompt["5"]["inputs"] == {
        "window_duration_seconds": 4.0,
        "window_count": 4,
        "context_frames": "22",
        "target_duration_seconds": 15.0,
    }
    samples = [node for node in prompt.values() if node["class_type"] == "WeeToddH3Sample"]
    contexts = [
        node for node in prompt.values() if node["class_type"] == "WeeToddH3ContinuationContext"
    ]
    appends = [node for node in prompt.values() if node["class_type"] == "WeeToddH3ChainAppend"]
    assert len(samples) == len(appends) == 4
    assert len(contexts) == 3
    assert all(node["inputs"]["context_frames"] == "22" for node in contexts)
    assert [node["inputs"]["unload_after_sample"] for node in samples] == [
        False,
        False,
        False,
        True,
    ]
    assert prompt["21"]["class_type"] == "WeeToddH3DirectPublishChain"
    metadata = json.loads(prompt["21"]["inputs"]["generation_metadata"])
    assert metadata["join_policy"].startswith("motion-matched overlap")


def test_h3_15_second_alien_ref2va_workflow_uses_strict_two_reference_order():
    prompt = json.loads(
        (ROOT / "examples" / "h3_768p_15s_ref2va_aliens_staged_turbo_api.json").read_text()
    )

    assert prompt["1"]["inputs"] == {
        "checkpoint": "MiniMax-H3/Ref2VA",
        "task": "ref2va",
        "transformer": "MiniMax-H3/Ref2VA/transformer",
        "text_encoder": "MiniMax-H3/Ref2VA/text_encoder",
        "processor": "MiniMax-H3/Ref2VA/processor",
        "tokenizer": "MiniMax-H3/Ref2VA/tokenizer",
        "video_vae": "MiniMax-H3/Ref2VA/video_vae",
        "audio_vae": "MiniMax-H3/Ref2VA/audio_vae",
        "allow_fl2va_weights_for_ref2va": False,
    }
    assert prompt["2"]["inputs"]["seed"] == 20260811
    assert prompt["2"]["inputs"]["duration_seconds"] == 15.0
    assert prompt["3"]["inputs"]["preset"] == (
        "One-shot staged Turbo — drbaph v4 step-600 — 15-second quality baseline"
    )
    assert prompt["5"]["inputs"]["image"] == ""
    assert prompt["7"]["inputs"]["image"] == ""
    assert prompt["8"]["inputs"]["previous_references"] == ["6", 0]
    reference_prompt = prompt["9"]["inputs"]["prompt"]
    assert reference_prompt.startswith("<Picture 1> defines the single tall white")
    assert "<Picture 2> defines the single short grey" in reference_prompt
    assert "Dwight" not in reference_prompt
    assert "Jesse" not in reference_prompt
    assert prompt["10"]["inputs"]["visual_strength"] == 0.999
    assert prompt["10"]["inputs"]["audio_strength"] == 1.0
    assert prompt["11"]["inputs"]["loras"] == ["3", 1]
    assert prompt["12"]["class_type"] == "WeeToddH3DirectPublishLatents"


def test_h3_15_second_alien_ref2va_chain_keeps_media_and_latent_context():
    prompt = json.loads(
        (ROOT / "examples" / "h3_768p_15s_ref2va_aliens_chained_staged_turbo_api.json").read_text()
    )

    assert prompt["1"]["inputs"]["task"] == "ref2va"
    assert prompt["1"]["inputs"]["allow_fl2va_weights_for_ref2va"] is False
    assert prompt["10"]["inputs"]["file"] == ""
    assert [prompt[node]["inputs"]["start_time"] for node in ("11", "20", "29", "38")] == [
        0.0,
        85 / 24,
        170 / 24,
        255 / 24,
    ]
    assert all(
        prompt[node]["inputs"]["previous_references"] == ["9", 0]
        for node in ("13", "22", "31", "40")
    )
    assert all(
        prompt[node]["inputs"]["soundtrack"] == [component, 1]
        for node, component in zip(
            ("13", "22", "31", "40"),
            ("12", "21", "30", "39"),
            strict=True,
        )
    )
    assert [prompt[node]["inputs"].get("continuation") for node in ("16", "25", "34", "43")] == [
        None,
        ["18", 0],
        ["27", 0],
        ["36", 0],
    ]
    assert all(
        "<Picture 1> defines the single tall white" in prompt[node]["inputs"]["prompt"]
        and "<Picture 2> defines the single short grey" in prompt[node]["inputs"]["prompt"]
        and "<Video 1> supplies" in prompt[node]["inputs"]["prompt"]
        for node in ("14", "23", "32", "41")
    )
    assert prompt["45"]["class_type"] == "WeeToddH3DirectPublishChain"


def test_t2va_api_prompt_uses_registered_nodes_and_staged_unloading():
    prompt = json.loads((ROOT / "examples" / "t2va_smoke_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["4"]["inputs"]["unload_after_encode"] is True
    assert prompt["5"]["inputs"]["unload_after_sample"] is True
    assert prompt["5"]["inputs"]["easycache"] == ["9", 0]
    assert prompt["2"]["inputs"]["components"] == ["10", 0]
    assert prompt["10"]["inputs"] == {
        "components": ["1", 0],
        "tae_model": "taeh3.safetensors",
        "preview_backend": "auto",
        "coreml_model": "taeh3_coreml_256.mlpackage",
        "preview_every": 1,
        "preview_frames": 6,
        "max_preview_edge": 256,
        "safety_guard": "conservative collapse guard",
    }
    assert prompt["2"]["inputs"]["config"] == ["3", 0]
    assert "width" not in prompt["2"]["inputs"]
    assert prompt["3"]["inputs"]["resolution_mode"] == "ratio + size"
    assert prompt["3"]["inputs"]["resolution_tier"] == "384 px short edge — fast smoke"
    assert prompt["3"]["inputs"]["aspect_ratio"] == "16:9 — widescreen landscape"
    assert prompt["3"]["inputs"]["short_edge"] == 384
    assert prompt["3"]["inputs"]["memory_mode"] == "normal"
    assert prompt["3"]["inputs"]["attention_chunk_size"] == "automatic"
    assert prompt["3"]["inputs"]["projection_backend"] == "mlx"
    assert prompt["3"]["inputs"]["sampling_method"] == "euler"
    assert prompt["9"]["inputs"] == {
        "mode": "manual",
        "reuse_threshold": 0.2,
        "start_percent": 0.15,
        "end_percent": 0.95,
        "auto_multiplier": 1.15,
        "max_skip_fraction": 0.25,
    }
    assert prompt["6"]["inputs"]["unload_after_decode"] is True
    assert prompt["7"]["inputs"]["unload_after_decode"] is True
    assert prompt["8"]["inputs"]["images"] == ["6", 0]
    assert prompt["8"]["inputs"]["audio"] == ["7", 0]
    assert prompt["8"]["inputs"]["sampling_info"] == ["5", 1]
    assert prompt["1"]["inputs"] == PORTABLE_T2VA_INPUTS


def test_low_memory_paged_api_uses_dual_paging_and_direct_publication():
    prompt = json.loads((ROOT / "examples" / "t2va_low_memory_paged_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["1"]["inputs"]["transformer"].endswith("q8_extended_paged")
    assert prompt["1"]["inputs"]["text_encoder"].endswith("q8-paged")
    assert prompt["1"]["inputs"]["processor"].endswith("FL2VA/processor")
    assert prompt["1"]["inputs"]["tokenizer"].endswith("FL2VA/tokenizer")
    assert prompt["1"]["inputs"]["video_vae"].endswith("q8/video_vae_affine_q8.safetensors")
    assert prompt["1"]["inputs"]["audio_vae"].endswith("FL2VA/audio_vae")
    assert prompt["3"]["inputs"] == {
        "aspect_ratio": "16:9 — widescreen landscape",
        "attention_chunk_size": "automatic",
        "custom_height": 384,
        "custom_width": 640,
        "drop_adaln": True,
        "duration_seconds": 5.0,
        "memory_mode": "low_memory_bf16",
        "projection_backend": "mlx",
        "resolution_mode": "ratio + size",
        "resolution_tier": "384 px short edge — fast smoke",
        "sampling_method": "euler",
        "short_edge": 384,
        "seed": 0,
        "steps": 5,
    }
    assert prompt["4"]["inputs"]["unload_after_encode"] is True
    assert prompt["5"]["inputs"]["unload_after_sample"] is True
    assert "easycache" not in prompt["5"]["inputs"]
    assert "blockcache" not in prompt["5"]["inputs"]
    assert prompt["6"]["class_type"] == "WeeToddH3DirectPublishLatents"
    assert prompt["6"]["inputs"]["latents"] == ["5", 0]
    assert prompt["6"]["inputs"]["sampling_info"] == ["5", 1]


@pytest.mark.parametrize(
    "name, conditioning_node",
    [
        ("fl2va_first_frame", "WeeToddH3KeyframeEncode"),
        ("ref2va_image", "WeeToddH3ReferenceEncode"),
    ],
)
def test_conditioning_api_examples_use_current_nodes(name, conditioning_node):
    prompt = json.loads((ROOT / "examples" / f"{name}_api.json").read_text())
    classes = {node["class_type"] for node in prompt.values()}
    assert classes <= set(NODE_CLASS_MAPPINGS) | CORE_NODES
    assert conditioning_node in classes
    assert prompt["6"]["inputs"]["unload_after_sample"] is True
    assert prompt["7"]["class_type"] == "WeeToddH3DirectPublishLatents"
    loader = prompt["1"]["inputs"]
    assert loader["transformer"]
    assert loader["text_encoder"]
    assert loader["processor"]
    assert loader["tokenizer"]
    assert loader["video_vae"]
    assert loader["audio_vae"]
    assert "q8-paged" not in loader["text_encoder"]
