from __future__ import annotations

import json

import mlx.core as mx
import pytest

from minimax_h3_mlx.paged_checkpoint import PagedTensorStore
from minimax_h3_mlx.paged_text_encoder import (
    PAGED_QWEN_MANIFEST,
    PagedTextEncoderManifest,
    convert_to_paged_text_encoder,
)


def _source(tmp_path):
    source = tmp_path / "compact"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"text_config": {}}))
    mx.save_safetensors(
        str(source / "text_encoder.safetensors"),
        {
            "model.embed_tokens.weight": mx.ones((4, 4)),
            "model.layers.0.self_attn.q_proj.weight": mx.full((4, 4), 2),
            "model.layers.1.self_attn.q_proj.weight": mx.full((4, 4), 3),
            "visual.patch_embed.proj.weight": mx.full((4, 4), 4),
        },
    )
    return source


def test_converter_writes_text_only_fixed_and_layer_pages(tmp_path):
    destination = tmp_path / "paged"
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), destination, num_layers=2
    )

    assert manifest.num_blocks == 2
    assert manifest.skipped_visual_bytes == 64
    assert (destination / PAGED_QWEN_MANIFEST).is_file()
    assert set(mx.load(str(destination / manifest.fixed.file))) == {
        "model.embed_tokens.weight"
    }
    assert set(mx.load(str(destination / manifest.layers[1].file))) == {
        "model.layers.1.self_attn.q_proj.weight"
    }


def test_converter_packages_full_architecture_config(tmp_path):
    architecture = tmp_path / "architecture.json"
    architecture.write_text(json.dumps({"text_config": {"num_hidden_layers": 64}}))
    destination = tmp_path / "paged"

    convert_to_paged_text_encoder(
        _source(tmp_path),
        destination,
        num_layers=2,
        architecture_config=architecture,
    )

    assert json.loads((destination / "architecture_config.json").read_text()) == {
        "text_config": {"num_hidden_layers": 64}
    }


def test_text_layer_pages_follow_generic_store_lifetime(tmp_path):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )
    store = PagedTensorStore(manifest)

    values = store.load_block(0)
    mx.eval(values)
    with pytest.raises(RuntimeError, match="still active"):
        store.load_block(1)
    values.clear()
    store.release()

    assert store.active_page is None
    assert store.pages_loaded == 1


def test_text_manifest_rejects_modified_layer(tmp_path):
    destination = tmp_path / "paged"
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), destination, num_layers=2
    )
    page = destination / manifest.layers[0].file
    page.write_bytes(page.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="hash differs"):
        PagedTextEncoderManifest.load(destination, verify_hashes=True)
