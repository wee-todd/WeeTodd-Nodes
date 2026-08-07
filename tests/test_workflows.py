import json
from pathlib import Path

from wee_todd_nodes.nodes import NODE_CLASS_MAPPINGS

ROOT = Path(__file__).parents[1]


def test_t2va_api_prompt_uses_registered_nodes_and_staged_unloading():
    prompt = json.loads((ROOT / "examples" / "t2va_smoke_api.json").read_text())

    assert {node["class_type"] for node in prompt.values()} <= set(NODE_CLASS_MAPPINGS)
    assert prompt["4"]["inputs"]["unload_after_encode"] is True
    assert prompt["5"]["inputs"]["unload_after_sample"] is True
    assert prompt["5"]["inputs"]["easycache"] == ["9", 0]
    assert prompt["2"]["inputs"]["config"] == ["3", 0]
    assert "width" not in prompt["2"]["inputs"]
    assert prompt["3"]["inputs"]["resolution_mode"] == "preset"
    assert prompt["3"]["inputs"]["resolution_tier"] == "384P (fast smoke)"
    assert prompt["3"]["inputs"]["aspect_ratio"] == "16:9"
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
        "preset",
        "384P (fast smoke)",
        "16:9",
        640,
        384,
        True,
        "normal",
        "automatic",
    ]
    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links.values():
        assert link_id in nodes[origin_id]["outputs"][origin_slot]["links"]
        target_input = nodes[target_id]["inputs"][target_slot]
        assert target_input["link"] == link_id
        assert target_input["type"] == link_type
