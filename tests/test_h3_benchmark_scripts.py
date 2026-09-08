import importlib.util
from pathlib import Path

import pytest


def _load(name):
    location = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_workflow_timing_uses_server_events_not_polling_interval():
    module = _load("benchmark_saved_h3_workflow")
    history = {
        "status": {
            "messages": [
                ["execution_start", {"timestamp": 1000}],
                ["execution_cached", {"nodes": []}],
                ["execution_success", {"timestamp": 125123}],
            ]
        }
    }
    assert module.elapsed_seconds(history) == 124.123
    history["status"]["messages"].pop()
    with pytest.raises(ValueError, match="unambiguous"):
        module.elapsed_seconds(history)


def test_speed_artifact_gate_requires_real_dispatch_not_only_healthy_media():
    module = _load("verify_fasth3_speed_artifact")
    metadata = {
        "frames": 107,
        "fps": 24,
        "sample_rate": 32000,
        "av_drift_seconds": 1 / 120,
        "sampling": {
            "transformer_evaluations": 4,
            "transformer_resident": False,
            "fastvideo": {
                "executed_layers": 40,
                "skipped_layers": 10,
                "active_layer_indices": [*range(38), 48, 49],
                "skipped_layer_indices": list(range(38, 48)),
            },
            "sol_attention": {
                "executed_calls": 160,
                "fallback_calls": 0,
                "storage_layout": "compact_preordered",
            },
        },
    }
    video = {
        "frames": 107,
        "near_frozen_frame_pair_fraction": 0.0,
        "black_pixel_fraction": 0.01,
        "white_pixel_fraction": 0.01,
    }
    audio = {"finite": True, "channels": 2, "rms": 0.03, "clipped_sample_fraction": 0.0}
    assert all(module.technical_checks(metadata, video, audio).values())
    metadata["sampling"]["sol_attention"]["fallback_calls"] = 1
    assert not module.technical_checks(metadata, video, audio)["compact_metal_dispatch"]


def test_speed_artifact_checks_actual_container_drift_and_missing_audio():
    module = _load("verify_fasth3_speed_artifact")
    metadata = {"width": 768, "height": 448, "frames": 107}
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 768,
                "height": 448,
                "nb_read_frames": "107",
                "r_frame_rate": "24/1",
                "start_time": "0.0",
                "duration": "4.458333",
            },
            {
                "codec_type": "audio",
                "channels": 2,
                "sample_rate": "32000",
                "start_time": "0.0",
                "duration": "4.45",
            },
        ]
    }
    assert all(module.stream_checks(metadata, probe).values())
    probe["streams"][1]["duration"] = "3.5"
    assert not module.stream_checks(metadata, probe)["container_av_synchronization"]
    probe["streams"].pop()
    assert not any(module.stream_checks(metadata, probe).values())


def test_workflow_note_generator_preserves_native_fasth3_speed_contract():
    module = _load("add_workflow_notes")
    workflow = {
        "nodes": [
            {
                "type": "WeeToddH3FastH3ProductionProfile",
                "widgets_values": [
                    "Speed candidate — compact Metal + 40 layers",
                ],
            }
        ]
    }
    note = module._model_note(Path("h3_fasth3_40layer_candidate.json"), workflow)
    assert "weetodd-fasth3-vsa-datafree-q8-paged" in note
    assert "fastvideo from profile output 4" in note
    assert "listening acceptance" in note
    assert "MiniMax-H3-MLX-q8-extended-paged" not in note
