import importlib
import json
from pathlib import Path

import pytest

from wee_todd_mlx.progress import render_progress


def test_native_progress_is_a_flushed_stage_event(capsys):
    render_progress("encoding", "Encoding prompt")
    render_progress("sampling", "Sampling", completed=3, total=16)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0] == {
        "event": "progress", "stage": "encoding", "message": "Encoding prompt", "fraction": 0,
    }
    assert events[1]["message"] == "Sampling 3/16"
    assert events[1]["fraction"] == 3 / 16


def test_progress_completion_moves_to_finishing(capsys):
    render_progress("sampling", "Sampling", completed=11, total=11)
    event = json.loads(capsys.readouterr().out)
    assert event["stage"] == "finishing"
    assert event["message"] == "Decoding and publishing"
    assert event["fraction"] == 0


@pytest.mark.parametrize("engine", ["ltx23", "ltx25"])
def test_native_entrypoint_delivers_live_sampling_and_finishing_events(
    engine, capsys, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    runner = importlib.import_module("render_headless")
    runtime = importlib.import_module(f"{engine}_mlx.runtime")
    released = []

    def generate(spec, config, prompt, target, **options):
        options["step_callback"](1, 11)
        options["step_callback"](11, 11)
        return {"video_path": str(target)}

    monkeypatch.setattr(runtime.RUNTIME, "generate_to_file", generate)
    monkeypatch.setattr(runtime.RUNTIME, "unload", lambda: released.append(True))
    components = {"model_dir": str(tmp_path)} if engine == "ltx23" else {
        "transformer_path": "a", "text_encoder_path": "b", "video_vae_path": "c",
        "audio_vae_path": "d", "spatial_upscaler_path": "e",
    }
    runner.render_ltx({"engine": engine, "components": components, "config": {}, "prompt": "test"},
                      tmp_path / "out.mp4")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["stage"] for e in events] == ["encoding", "sampling", "finishing"]
    assert events[1]["message"] == "Sampling 1/11"
    assert released == [True]
