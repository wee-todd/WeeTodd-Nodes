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
    "h3_512p_turbo_larry_ema850.json": (
        "Turbo — Larry EMA-850 — 5 points / 4 evaluations"
    ),
    "h3_512p_turbo_larry_v4.json": (
        "Turbo — Larry v4 step-600 — 5 points / 4 evaluations"
    ),
    "h3_512p_turbo_lightx2v_full.json": (
        "Turbo — LightX2V full rank — 5 points / 4 evaluations"
    ),
    "h3_512p_turbo_lightx2v_dynamic_rank21.json": (
        "Turbo — LightX2V dynamic rank 21 — 5 points / 4 evaluations"
    ),
}


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
