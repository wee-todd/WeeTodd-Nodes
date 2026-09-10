import json

import pytest

from minimax_h3_mlx.dt_source import dt_source, validate_dt_reference


def test_source_reference_resolves_relative_without_copy(tmp_path):
    source = tmp_path / "original.ckpt"
    source.write_bytes(b"not decoded here")
    (tmp_path / "draw_things_source.json").write_text(
        json.dumps(
            {
                "format": "weetodd-h3-dt-source-v1",
                "component": "video_vae",
                "checkpoint": source.name,
            }
        )
    )
    assert dt_source(tmp_path, "video_vae") == source
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "draw_things_source.json",
        "original.ckpt",
    ]


@pytest.mark.parametrize("raw", [[], None, {"checkpoint": 1}, {"format": "unknown"}])
def test_malformed_source_is_actionable_value_error(tmp_path, raw):
    (tmp_path / "draw_things_source.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="DT source"):
        dt_source(tmp_path, "audio_vae")


def test_source_reference_is_bounded(tmp_path):
    (tmp_path / "draw_things_source.json").write_text(" " * 65537)
    with pytest.raises(ValueError, match="too large"):
        dt_source(tmp_path, "text_encoder")


def test_vae_reference_rejects_invalid_normalization_before_loading(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"latents_mean": [0] * 24, "latents_std": [0] * 24})
    )
    with pytest.raises(ValueError, match="positive"):
        validate_dt_reference(tmp_path, "video_vae")


@pytest.mark.parametrize("component,size", [("video_vae", 24), ("audio_vae", 32)])
def test_decoder_spec_accepts_metadata_reference_without_weight_copy(tmp_path, component, size):
    from wee_todd_nodes.decoding import H3AudioVAESpec, H3VideoVAESpec

    (tmp_path / "config.json").write_text(
        json.dumps({"latents_mean": [0] * size, "latents_std": [1] * size})
    )
    (tmp_path / "original.ckpt").write_bytes(b"SQLite format 3\x00")
    (tmp_path / "draw_things_source.json").write_text(
        json.dumps(
            {
                "format": "weetodd-h3-dt-source-v1",
                "component": component,
                "checkpoint": "original.ckpt",
            }
        )
    )
    cls = H3VideoVAESpec if component == "video_vae" else H3AudioVAESpec
    cls(str(tmp_path)).validate()


def test_reference_asset_revision_and_prompt_cache_follow_original_model(tmp_path):
    from wee_todd_mlx.asset_registry import describe_asset
    from wee_todd_nodes.conditioning import H3TextEncoderSpec
    from wee_todd_nodes.conditioning_cache import cache_key

    model = tmp_path / "qwen.ckpt"
    model.write_bytes(b"weights1")
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "draw_things_source.json").write_text(
        json.dumps(
            {
                "format": "weetodd-h3-dt-source-v1",
                "component": "text_encoder",
                "checkpoint": str(model),
            }
        )
    )
    spec = H3TextEncoderSpec(str(meta), str(meta), str(meta))
    before = describe_asset(meta)["revision"]
    key = cache_key(spec, "scene", "t2va")
    model.write_bytes(b"updated weights2")
    assert describe_asset(meta)["revision"] != before
    assert cache_key(spec, "scene", "t2va") != key


def test_dt_checkpoint_asset_inspection_is_metadata_only(tmp_path):
    from test_dt_tensor_store import checkpoint

    from wee_todd_mlx.asset_registry import describe_asset

    model = checkpoint(tmp_path, 0, (1,), b"\x00\x00", trailer=True)
    assert describe_asset(model)["members"][0]["tensor_count"] == 1


def test_component_inventory_rejects_wrong_shape_before_weight_execution():
    from minimax_h3_mlx.dt_source import expected_inventory, validate_inventory
    from minimax_h3_mlx.dt_tensor_store import TensorRecord

    expected = expected_inventory("text_encoder")
    records = {n: TensorRecord(n, tuple(shape), 0, 0x20000, 0) for n, shape in expected.items()}
    validate_inventory(records, "text_encoder")
    key = "__text_model__[t-q_proj-49-0]"
    records[key] = TensorRecord(key, (8191, 5120), 0, 0x20000, 0)
    with pytest.raises(ValueError, match="shape"):
        validate_inventory(records, "text_encoder")


def test_qwen_config_failure_precedes_opening_direct_weight_store(tmp_path, monkeypatch):
    pytest.importorskip('mlx.core')
    from pathlib import Path

    import minimax_h3_mlx.dt_qwen as qwen
    from minimax_h3_mlx import dt_source as source_module
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
    raw=json.loads(Path(source_module.__file__).with_name('dt_h3_config.json').read_text())['text_encoder']
    raw.pop('image_token_id')
    (tmp_path/'config.json').write_text(json.dumps(raw))
    (tmp_path/'model.ckpt').touch()
    (tmp_path/'draw_things_source.json').write_text(json.dumps({
        'format':'weetodd-h3-dt-source-v1','component':'text_encoder','checkpoint':'model.ckpt'}))
    def unopened(*args,**kwargs):
        raise AssertionError('Weight store opened before config initialization completed')
    monkeypatch.setattr(qwen,'DTTensorStore',unopened)
    with pytest.raises(KeyError,match='image_token_id'):
        MiniMaxH3TextEncoder(tmp_path,load_vision=False)
