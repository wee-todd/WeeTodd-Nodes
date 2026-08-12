import importlib.util
import json
from pathlib import Path

import pytest


def _script_module():
    path = Path("scripts/create_h3_model_index.py")
    spec = importlib.util.spec_from_file_location("create_h3_model_index_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("partition", "tasks"),
    [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])],
)
def test_create_model_index_uses_partition_template(tmp_path, partition, tasks):
    module = _script_module()
    checkpoint = tmp_path / partition.upper()
    checkpoint.mkdir()

    target = module.create_model_index(checkpoint, partition)
    manifest = json.loads(target.read_text())

    assert target == checkpoint / "model_index.json"
    assert manifest["_minimax_h3"]["partition"] == partition
    assert manifest["_minimax_h3"]["tasks"] == tasks
    assert manifest["_minimax_h3"]["sigma_shift_scales"] == {
        "video": 12.0,
        "audio": 3.0,
    }
    assert {
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    } <= manifest.keys()


def test_create_model_index_does_not_replace_existing_manifest(tmp_path):
    module = _script_module()
    checkpoint = tmp_path / "FL2VA"
    checkpoint.mkdir()
    target = checkpoint / "model_index.json"
    target.write_text('{"keep": true}\n')

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        module.create_model_index(checkpoint, "fl2va")

    assert json.loads(target.read_text()) == {"keep": True}


def test_create_model_index_rejects_unknown_partition(tmp_path):
    module = _script_module()

    with pytest.raises(ValueError, match="Partition must be"):
        module.create_model_index(tmp_path, "unknown")
