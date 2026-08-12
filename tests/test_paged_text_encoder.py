from __future__ import annotations

import json

import mlx.core as mx
import pytest

from minimax_h3_mlx.page_prefetch import PagePrefetchResult, SequentialPagePrefetch
from minimax_h3_mlx.paged_checkpoint import PagedTensorStore
from minimax_h3_mlx.paged_text_encoder import (
    PAGED_QWEN_MANIFEST,
    PagedTextEncoderManifest,
    PagedTextLayerExecutor,
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


def test_sequential_page_prefetch_reads_one_bounded_future(tmp_path):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )
    prefetch = SequentialPagePrefetch(manifest.root, manifest.layers)

    prefetch.start(1)
    assert prefetch.wait(1) is True
    report = prefetch.report()
    prefetch.close()

    assert report["prefetch_enabled"] is True
    assert report["prefetch_depth"] == 1
    assert report["prefetch_requests"] == 1
    assert report["prefetch_hits"] == 1
    assert report["prefetch_failures"] == 0
    assert report["prefetch_bytes"] == (manifest.root / manifest.layers[1].file).stat().st_size
    assert report["prefetch_buffer_bytes"] == 8 * 1024 * 1024
    assert 0 <= report["prefetch_wait_seconds"] <= report["prefetch_read_seconds"] + 0.1


def test_sequential_page_prefetch_failure_uses_serial_fallback(tmp_path):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )

    def fail_reader(_index, _paths):
        raise OSError("injected speculative read failure")

    prefetch = SequentialPagePrefetch(
        manifest.root, manifest.layers, reader=fail_reader
    )
    prefetch.start(1)

    assert prefetch.wait(1) is False
    assert set(mx.load(str(manifest.root / manifest.layers[1].file))) == {
        "model.layers.1.self_attn.q_proj.weight"
    }
    assert prefetch.report()["prefetch_failures"] == 1
    prefetch.close()


def test_sequential_page_prefetch_can_be_disabled(tmp_path):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )
    prefetch = SequentialPagePrefetch(
        manifest.root, manifest.layers, enabled=False
    )

    prefetch.start(1)

    assert prefetch.wait(1) is False
    assert prefetch.report()["prefetch_requests"] == 0
    assert prefetch.report()["prefetch_buffer_bytes"] == 0
    prefetch.close()


def test_paged_text_layer_prefetch_defaults_off_and_can_be_enabled(tmp_path, monkeypatch):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )
    monkeypatch.delenv("WEETODD_H3_QWEN_PREFETCH", raising=False)
    default = PagedTextLayerExecutor(manifest, object())
    assert default.report()["prefetch_enabled"] is False
    default.close()

    monkeypatch.setenv("WEETODD_H3_QWEN_PREFETCH", "1")
    enabled = PagedTextLayerExecutor(manifest, object())
    assert enabled.report()["prefetch_enabled"] is True
    enabled.close()


def test_sequential_page_prefetch_records_reader_metrics(tmp_path):
    manifest = convert_to_paged_text_encoder(
        _source(tmp_path), tmp_path / "paged", num_layers=2
    )

    def measured_reader(index, paths):
        return PagePrefetchResult(
            index, len(paths), sum(path.stat().st_size for path in paths), 0.25
        )

    prefetch = SequentialPagePrefetch(
        manifest.root, manifest.layers, reader=measured_reader
    )
    prefetch.start(1)
    assert prefetch.wait(1) is True
    report = prefetch.report()
    prefetch.close()

    assert report["prefetch_read_seconds"] == 0.25
    assert 0 <= report["prefetch_hidden_seconds"] <= 0.25
