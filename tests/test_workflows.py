import json
from pathlib import Path

import pytest

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parents[1]
CORE_NODES = {"LoadImage"}
VALIDATED_WORKFLOW_PRESETS = {
    "h3_512p_dense_baseline.json": "Dense baseline — 20 points / 19 evaluations",
    "h3_512p_trajectory_replay.json": (
        "Trajectory speed + offline replay — 20 points / up to 11 evaluations"
    ),
    "h3_512p_turbo_larry_ema850.json": ("Turbo — Larry EMA-850 — 5 points / 4 evaluations"),
    "h3_512p_turbo_larry_v4.json": ("Turbo — Larry v4 step-600 — 5 points / 4 evaluations"),
    "h3_512p_turbo_lightx2v_full.json": ("Turbo — LightX2V full rank — 5 points / 4 evaluations"),
    "h3_512p_turbo_lightx2v_dynamic_rank21.json": (
        "Turbo — LightX2V dynamic rank 21 — 5 points / 4 evaluations"
    ),
}
CHAINED_WORKFLOW_PRESETS = {
    "h3_544p_chained_dense_turbo_rank21.json": (
        "Chained context — Dense Turbo LightX2V rank 21 — 5 points / 4 evaluations"
    ),
    "h3_544p_chained_trajectory_replay.json": (
        "Chained context — Trajectory target-only replay — "
        "20 points / up to 11 evaluations"
    ),
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

UI_WORKFLOW_PATHS = sorted((ROOT / "workflows").glob("*.json")) + sorted(
    (ROOT / "examples").glob("*_workflow.json")
)
API_WORKFLOW_PATHS = sorted((ROOT / "examples").glob("*_api.json"))
PRIMITIVE_WIDGET_TYPES = {"BOOLEAN", "FLOAT", "INT", "STRING"}


def _input_schema(node_type):
    schema = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
    return [
        (name, specification)
        for group in ("required", "optional")
        for name, specification in schema.get(group, {}).items()
    ]


def _widget_schema(node):
    connected = {
        item["name"] for item in node.get("inputs", []) if item.get("link") is not None
    }
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
        short_edge_index = next(
            i for i, item in enumerate(widgets) if item[0] == "short_edge"
        )
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
    nodes = {node["id"]: node for node in workflow["nodes"]}

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
    workflow = json.loads((ROOT / "workflows" / "ltx23_t2va_two_stage.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}

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
    prompt = json.loads(
        (ROOT / "examples" / "h3_native_hires_fix_1p5x_api.json").read_text()
    )

    assert prompt["3"]["inputs"]["steps"] == 8
    assert prompt["5"]["inputs"]["unload_after_sample"] is False
    assert prompt["6"]["class_type"] == "WeeToddH3LatentHiresFix"
    assert prompt["6"]["inputs"]["source_latents"] == ["5", 0]
    assert prompt["6"]["inputs"]["scale"] == "1.5x — balanced"
    assert prompt["6"]["inputs"]["latent_resize_method"] == "bilinear"
    assert prompt["7"]["inputs"]["latents"] == ["6", 0]
    assert prompt["7"]["inputs"]["sampling_info"] == ["6", 2]


def test_h3_768p_staged_turbo_workflow_uses_one_continuous_schedule():
    prompt = json.loads(
        (ROOT / "examples" / "h3_768p_fl2va_staged_turbo_drbaph_v4_api.json").read_text()
    )
    workflow = json.loads(
        (ROOT / "workflows" / "h3_768p_fl2va_staged_turbo_drbaph_v4.json").read_text()
    )
    ui_nodes = {node["id"]: node for node in workflow["nodes"]}

    assert prompt["1"]["inputs"]["task"] == "fl2va"
    assert prompt["2"]["inputs"]["steps"] == 7
    assert prompt["2"]["inputs"]["custom_width"] == 1344
    assert prompt["2"]["inputs"]["custom_height"] == 768
    assert prompt["2"]["inputs"]["sampling_method"] == "euler"
    assert prompt["5"]["inputs"]["config"] == ["6", 0]
    assert prompt["6"]["class_type"] == "WeeToddH3ValidatedSamplingPreset"
    assert prompt["6"]["inputs"] == {
        "config": ["2", 0],
        "preset": (
            "Staged Turbo — drbaph v4 step-600 — "
            "2 base + 4 Turbo evaluations"
        ),
    }
    assert prompt["7"]["inputs"]["config"] == ["6", 0]
    assert prompt["7"]["inputs"]["loras"] == ["6", 1]
    assert prompt["7"]["inputs"]["trajectory_forecast"] == ["6", 2]
    assert "easycache" not in prompt["7"]["inputs"]
    assert "blockcache" not in prompt["7"]["inputs"]
    assert prompt["7"]["inputs"]["unload_after_sample"] is True
    assert ui_nodes[8]["type"] == "WeeToddH3ValidatedSamplingPreset"
    assert ui_nodes[8]["widgets_values"] == [prompt["6"]["inputs"]["preset"]]
    assert ui_nodes[6]["inputs"][3]["name"] == "trajectory_forecast"
    assert ui_nodes[6]["inputs"][4]["name"] == "loras"


def test_t2va_api_prompt_uses_registered_nodes_and_staged_unloading():
    prompt = json.loads((ROOT / "examples" / "t2va_smoke_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["4"]["inputs"]["unload_after_encode"] is True
    assert prompt["5"]["inputs"]["unload_after_sample"] is True
    assert prompt["5"]["inputs"]["easycache"] == ["9", 0]
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


def test_t2va_ui_workflow_links_are_consistent():
    workflow = json.loads((ROOT / "examples" / "t2va_smoke_workflow.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert len(nodes) == 9
    assert set(links) == set(range(1, 17))
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS)
    assert nodes[3]["widgets_values"] == [
        5.0,
        8,
        0,
        "ratio + size",
        "384 px short edge — fast smoke",
        "16:9 — widescreen landscape",
        384,
        640,
        384,
        True,
        "normal",
        "automatic",
        "mlx",
        "euler",
    ]
    assert nodes[1]["widgets_values"] == PORTABLE_T2VA_COMPONENTS
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


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


def test_low_memory_paged_ui_workflow_links_are_consistent():
    workflow = json.loads((ROOT / "examples" / "t2va_low_memory_paged_workflow.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert len(nodes) == 6
    assert set(links) == set(range(1, 10))
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS)
    assert nodes[1]["widgets_values"][2:4] == [
        "MiniMax-H3/transformers/q8_extended_paged",
        "MiniMax-H3/text_encoders/q8-paged",
    ]
    assert nodes[1]["widgets_values"][6] == ("MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors")
    assert nodes[1]["widgets_values"] == PORTABLE_T2VA_COMPONENTS
    assert nodes[3]["widgets_values"] == [
        5.0,
        5,
        0,
        "ratio + size",
        "384 px short edge — fast smoke",
        "16:9 — widescreen landscape",
        384,
        640,
        384,
        True,
        "low_memory_bf16",
        "automatic",
        "mlx",
        "euler",
    ]
    assert nodes[6]["type"] == "WeeToddH3DirectPublishLatents"
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


@pytest.mark.parametrize("filename,preset", VALIDATED_WORKFLOW_PRESETS.items())
def test_validated_sampling_workflows_are_loadable_and_preconfigured(filename, preset):
    workflow = json.loads((ROOT / "workflows" / filename).read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert len(nodes) == 7
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS)
    assert nodes[2]["widgets_values"][:3] == [5.17, 20, 246813579]
    assert nodes[2]["widgets_values"][4:9] == [
        "512 px short edge — balanced preview",
        "16:9 — widescreen landscape",
        512,
        896,
        512,
    ]
    assert nodes[3]["type"] == "WeeToddH3ValidatedSamplingPreset"
    assert nodes[3]["widgets_values"] == [preset]
    assert nodes[1]["widgets_values"] == PORTABLE_T2VA_COMPONENTS
    assert [item["name"] for item in nodes[6]["inputs"]] == [
        "components",
        "conditioning",
        "config",
        "trajectory_forecast",
        "loras",
    ]

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


@pytest.mark.parametrize("filename,preset", CHAINED_WORKFLOW_PRESETS.items())
def test_chained_workflows_are_portable_linked_and_preconfigured(filename, preset):
    raw = (ROOT / "workflows" / filename).read_text()
    workflow = json.loads(raw)
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert len(nodes) == 30
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS)
    assert "/Volumes/" not in raw
    assert "/Users/" not in raw
    assert nodes[2]["widgets_values"] == [
        5.17,
        20,
        54420260810,
        "ratio + size",
        "Use size slider — 32 px steps",
        "16:9 — widescreen landscape",
        544,
        960,
        544,
        False,
        "normal",
        "automatic",
        "mlx",
        "euler",
    ]
    assert nodes[3]["widgets_values"] == [preset]
    assert nodes[1]["widgets_values"] == PORTABLE_T2VA_COMPONENTS

    by_type = {}
    for node in nodes.values():
        by_type.setdefault(node["type"], []).append(node)
    assert len(by_type["WeeToddH3TextEncode"]) == 4
    assert len(by_type["WeeToddH3Sample"]) == 4
    assert len(by_type["WeeToddH3ContinuationContext"]) == 3
    assert len(by_type["WeeToddH3VideoVAEDecode"]) == 4
    assert len(by_type["WeeToddH3AudioVAEDecode"]) == 4
    assert len(by_type["WeeToddH3TrimContinuation"]) == 3
    assert len(by_type["WeeToddH3PublishVideoAudio"]) == 4
    assert [node["widgets_values"] for node in by_type["WeeToddH3ContinuationContext"]] == [
        ["22"],
        ["22"],
        ["22"],
    ]
    assert [node["widgets_values"] for node in by_type["WeeToddH3Sample"]] == [
        [False],
        [False],
        [False],
        [True],
    ]
    assert [item["name"] for item in nodes[9]["inputs"]] == [
        "components",
        "conditioning",
        "config",
        "trajectory_forecast",
        "loras",
    ]
    for sample_id in (10, 11, 12):
        assert [item["name"] for item in nodes[sample_id]["inputs"]] == [
            "components",
            "conditioning",
            "config",
            "trajectory_forecast",
            "continuation",
            "loras",
        ]

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


def test_four_reference_ref2va_forward_attention_workflow_is_portable_and_linked():
    path = ROOT / "workflows" / "h3_512p_ref2va_four_reference_forward_attention.json"
    raw = path.read_text()
    workflow = json.loads(raw)
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert len(nodes) == 15
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS) | CORE_NODES
    assert "/Volumes/" not in raw
    assert "/Users/" not in raw
    assert nodes[1]["widgets_values"] == [
        "MiniMax-H3/FL2VA",
        "ref2va",
        "MiniMax-H3/transformers/q8_extended_paged",
        "MiniMax-H3/FL2VA/text_encoder",
        "MiniMax-H3/FL2VA/processor",
        "MiniMax-H3/FL2VA/tokenizer",
        "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
        "MiniMax-H3/FL2VA/audio_vae",
        True,
    ]
    assert "q8-paged" not in nodes[1]["widgets_values"][3]
    assert nodes[2]["widgets_values"] == [
        5.0,
        20,
        842731905,
        "ratio + size",
        "512 px short edge — balanced preview",
        "16:9 — widescreen landscape",
        512,
        896,
        512,
        True,
        "normal",
        "automatic",
        "mlx",
        "euler",
    ]
    assert nodes[3]["widgets_values"] == [
        ("Ref2VA four-reference BF16 — Forward Attention replay — 20 points / up to 11 evaluations")
    ]
    load_images = [node for node in nodes.values() if node["type"] == "LoadImage"]
    reference_nodes = [node for node in nodes.values() if node["type"] == "WeeToddH3ReferenceImage"]
    assert len(load_images) == 4
    assert len(reference_nodes) == 4
    assert [node["widgets_values"][0] for node in load_images] == [
        "select_little_red_reference.png",
        "select_wolf_reference.png",
        "select_granny_reference.png",
        "select_woodsman_reference.png",
    ]
    assert all(node["widgets_values"] == [100] for node in reference_nodes)
    assert all(
        label in nodes[13]["widgets_values"][0]
        for label in (
            "<Picture 1>",
            "<Picture 2>",
            "<Picture 3>",
            "<Picture 4>",
        )
    )
    assert [item["name"] for item in nodes[14]["inputs"]] == [
        "components",
        "conditioning",
        "config",
        "trajectory_forecast",
        "loras",
    ]

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type


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


@pytest.mark.parametrize("name", ["fl2va_first_frame", "ref2va_image"])
def test_conditioning_ui_examples_have_consistent_links(name):
    workflow = json.loads((ROOT / "examples" / f"{name}_workflow.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS) | CORE_NODES
    assert all(nodes[1]["widgets_values"][index] for index in range(2, 8))
    assert "q8-paged" not in nodes[1]["widgets_values"][3]
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type
