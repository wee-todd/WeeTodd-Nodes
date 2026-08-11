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
    ]
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
    assert prompt["1"]["inputs"]["video_vae"].endswith("q8/video_vae_affine_q8.safetensors")
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
    assert [item["name"] for item in nodes[6]["inputs"]] == [
        "components",
        "conditioning",
        "config",
        "loras",
        "trajectory_forecast",
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
    ]
    assert nodes[3]["widgets_values"] == [preset]

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
        "loras",
        "trajectory_forecast",
    ]
    for sample_id in (10, 11, 12):
        assert [item["name"] for item in nodes[sample_id]["inputs"]][-1] == "continuation"

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
        "",
        "MiniMax-H3/text_encoders/q8-paged",
        "",
        "",
        "",
        "",
        True,
    ]
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
        "loras",
        "trajectory_forecast",
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


@pytest.mark.parametrize("name", ["fl2va_first_frame", "ref2va_image"])
def test_conditioning_ui_examples_have_consistent_links(name):
    workflow = json.loads((ROOT / "examples" / f"{name}_workflow.json").read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert {node["type"] for node in nodes.values()} <= set(NODE_CLASS_MAPPINGS) | CORE_NODES
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type
