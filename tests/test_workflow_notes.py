"""Setup notes must describe the encoder selected by the executable graph."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
NOTES = runpy.run_path(str(ROOT / "scripts" / "add_workflow_notes.py"))


@pytest.mark.parametrize("profile", ["speed", "balance", "performance"])
@pytest.mark.parametrize("task", ["i2v", "fflf2va"])
def test_visual_h3_notes_identify_selected_encoder_and_optional_paged_switch(profile, task):
    path = ROOT / "workflows" / profile / task / f"h3_{task}_{profile}.json"
    workflow = json.loads(path.read_text())
    saved = next(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node.get("title") == "Setup and model downloads"
    )
    generated = NOTES["_model_note"](path, workflow)
    for note in (generated, saved):
        assert "vision-capable resident Qwen3-VL encoder" in note
        assert "ComfyUI/models/MiniMax-H3/FL2VA/text_encoder" in note
        assert "https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged" in note
        assert "change **Component Loader → text_encoder**" in note
        assert "MiniMax-H3/text_encoders/q8-vision-paged" in note
        assert NOTES["H3_Q8_QWEN"] + ")" not in note


def test_text_only_h3_note_keeps_its_selected_paged_encoder():
    path = ROOT / "workflows" / "speed" / "t2v" / "h3_t2v_speed.json"
    note = NOTES["_model_note"](path, json.loads(path.read_text()))
    assert NOTES["H3_Q8_QWEN"] + ")" in note
    assert "ComfyUI/models/MiniMax-H3/text_encoders/q8-paged" in note
    assert "vision-capable resident" not in note
