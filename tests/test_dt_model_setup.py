"""DT model reuse is an explicit T2V choice and produces metadata, never weights."""

import json
from pathlib import Path

import pytest


def test_dt_setup_is_native_h3_t2v_with_three_original_checkpoints():
    from wee_todd_mlx.model_setup import setup_catalog

    preset = next(p for p in setup_catalog() if p["id"] == "h3-draw-things-text")
    assert preset["engine"] == "h3" and preset["task"] == "t2v"
    assert {c["key"] for c in preset["components"]} == {
        "dt_transformer",
        "dt_qwen",
        "dt_vae",
        "tokenizer",
    }
    assert "experimental" in preset["description"].lower()


def test_dt_bundle_contains_only_small_metadata_and_original_references(tmp_path):
    from wee_todd_mlx.dt_model_setup import write_reference_bundle

    files = {}
    for key in ("dt_transformer", "dt_qwen", "dt_vae"):
        f = tmp_path / (key + ".ckpt")
        f.write_bytes(b"original")
        files[key] = str(f)
    files["tokenizer"] = str(tmp_path / "tokenizer")
    out = tmp_path / "bundle"
    components = write_reference_bundle(files, out)
    assert components["transformer"] == files["dt_transformer"]
    assert components["processor"] == components["tokenizer"] == files["tokenizer"]
    assert json.loads((out / "model_index.json").read_text())["_minimax_h3"]["tasks"] == ["t2va"]
    assert all(p.suffix == ".json" for p in out.rglob("*") if p.is_file())
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) < 30000
    for name, key in [
        ("text_encoder", "dt_qwen"),
        ("video_vae", "dt_vae"),
        ("audio_vae", "dt_vae"),
    ]:
        raw = json.loads((Path(components[name]) / "draw_things_source.json").read_text())
        assert raw["checkpoint"] == files[key]
    with pytest.raises(FileExistsError):
        write_reference_bundle(files, out)


def test_dt_tokenizer_download_has_no_model_weights():
    from wee_todd_mlx.model_downloads import PRECONVERTED

    item = next(p for p in PRECONVERTED if p["descriptor"]["id"] == "h3-dt-tokenizer")
    assert sum(f["size"] for f in item["files"]) < 12000000
    assert not any(f["filename"].endswith((".safetensors", ".ckpt")) for f in item["files"])
