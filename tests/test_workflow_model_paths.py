import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "preflight_h3_workflow.py"
SPEC = importlib.util.spec_from_file_location("preflight_h3_workflow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def graph(**overrides):
    inputs = {
        "checkpoint": "MiniMax-H3/FL2VA",
        "task": "t2va",
        "transformer": "MiniMax-H3/transformers/q8_extended_paged",
        "text_encoder": "MiniMax-H3/text_encoders/q8-paged",
        "processor": "MiniMax-H3/FL2VA/processor",
        "tokenizer": "MiniMax-H3/FL2VA/tokenizer",
        "video_vae": "MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
        "audio_vae": "MiniMax-H3/FL2VA/audio_vae",
    }
    inputs.update(overrides)
    return {"1": {"class_type": MODULE.COMPONENT_NODE, "inputs": inputs}}


def test_portable_component_paths_accept_relative_comfy_model_names():
    values = MODULE.portable_component_paths(graph())

    assert values["checkpoint"] == "MiniMax-H3/FL2VA"
    assert values["audio_vae"] == "MiniMax-H3/FL2VA/audio_vae"


@pytest.mark.parametrize("value", ["/private/models/FL2VA", "../shared/FL2VA"])
def test_portable_component_paths_reject_machine_or_parent_paths(value):
    with pytest.raises(ValueError, match="relative|cannot contain"):
        MODULE.portable_component_paths(graph(checkpoint=value))


def test_load_api_workflow_rejects_ui_workflow(tmp_path):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({"nodes": []}))

    with pytest.raises(ValueError, match="class_type"):
        MODULE.load_api_workflow(path)


def test_portable_media_inputs_accepts_comfy_input_names():
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": "refs/character.png"}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
        }
    )

    assert MODULE.portable_media_inputs(document) == {
        "2": "refs/character.png",
        "3": "source.mp4",
    }


def test_portable_media_inputs_accepts_unselected_portable_fields():
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": ""}},
            "4": {"class_type": "LoadAudio", "inputs": {"audio": ""}},
        }
    )

    assert MODULE.portable_media_inputs(document) == {}
    assert MODULE.unselected_media_inputs(document) == {
        "2": "LoadImage",
        "3": "LoadVideo",
        "4": "LoadAudio",
    }


@pytest.mark.parametrize("value", ["/tmp/private.png", "../outside.png"])
def test_portable_media_inputs_rejects_machine_or_parent_paths(value):
    document = graph()
    document["2"] = {"class_type": "LoadImage", "inputs": {"image": value}}
    with pytest.raises(ValueError, match="relative to ComfyUI"):
        MODULE.portable_media_inputs(document)


def test_missing_media_inputs_reports_every_missing_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "present.png").write_bytes(b"present")
    document = graph()
    document.update(
        {
            "2": {"class_type": "LoadImage", "inputs": {"image": "present.png"}},
            "3": {"class_type": "LoadVideo", "inputs": {"file": "missing.mp4"}},
        }
    )
    folder_paths = SimpleNamespace(
        get_annotated_filepath=lambda name: str(input_dir / name),
        get_input_directory=lambda: str(input_dir),
    )

    assert MODULE.missing_media_inputs(document, folder_paths) == {
        "3": str(input_dir / "missing.mp4")
    }


def test_media_preflight_does_not_bypass_host_path_rejection(tmp_path):
    (tmp_path / "existing.png").write_bytes(b"existing")
    document = {"1": {"class_type": "LoadImage", "inputs": {"image": "existing.png"}}}

    def reject(name):
        raise ValueError("resolved path is outside the input folder")

    folder_paths = SimpleNamespace(
        get_annotated_filepath=reject,
        get_input_directory=lambda: str(tmp_path),
    )
    missing = MODULE.missing_media_inputs(document, folder_paths)
    assert "host rejected path" in missing["1"]


def test_missing_component_paths_reports_all_missing_values(tmp_path):
    existing = tmp_path / "transformer"
    existing.mkdir()
    components = SimpleNamespace(
        checkpoint=str(tmp_path / "missing-checkpoint"),
        resolved_paths=lambda: {
            "transformer": existing,
            "processor": tmp_path / "missing-processor",
            "audio_vae": tmp_path / "missing-audio-vae",
        },
    )

    assert MODULE.missing_component_paths(components) == {
        "checkpoint": str(tmp_path / "missing-checkpoint"),
        "processor": str(tmp_path / "missing-processor"),
        "audio_vae": str(tmp_path / "missing-audio-vae"),
    }


def test_every_shipped_h3_api_workflow_has_portable_component_paths():
    for path in sorted((ROOT / "examples").glob("h3_*_api.json")):
        document = MODULE.load_api_workflow(path)
        MODULE.portable_component_paths(document)
        MODULE.validate_fasth3_profile_wiring(document)


@pytest.mark.parametrize(
    "field", ["config", "sol_attention", "production_profile_info", "fastvideo"]
)
def test_speed_workflow_preflight_rejects_disconnected_profile_outputs(field):
    document = MODULE.load_api_workflow(ROOT / "examples/h3_fasth3_40layer_768x448_api.json")
    del document["7"]["inputs"][field]
    with pytest.raises(ValueError, match=field):
        MODULE.validate_fasth3_profile_wiring(document)


def test_speed_workflow_preflight_rejects_competing_modifier():
    document = MODULE.load_api_workflow(ROOT / "examples/h3_fasth3_40layer_768x448_api.json")
    document["7"]["inputs"]["loras"] = ["10", 0]
    with pytest.raises(ValueError, match="cannot be combined with loras"):
        MODULE.validate_fasth3_profile_wiring(document)


def test_resident_vdn_preflight_discloses_memory_estimate_exclusions():
    document = graph()
    document["2"] = {"class_type": "WeeToddH3Sample", "inputs": {"block_residency": "resident"}}
    document["3"] = {"class_type": MODULE.VDN_NODE, "inputs": {}}
    policy = MODULE.sampling_memory_policy(document)
    assert policy["sample_block_residency"] == ["resident"]
    assert len(policy["memory_estimate_notes"]) == 2
    document["2"]["inputs"]["block_residency"] = "unknown"
    with pytest.raises(ValueError, match="residency"):
        MODULE.sampling_memory_policy(document)
