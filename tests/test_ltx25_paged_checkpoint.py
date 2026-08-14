import json

import mlx.core as mx
import pytest

from ltx25_mlx.page_prefetch import LTX25PagePrefetch, PagePrefetchResult
from ltx25_mlx.paged_checkpoint import (
    PAGED_GEMMA_FORMAT,
    PAGED_TRANSFORMER_FORMAT,
    LTX25PagedManifest,
    convert_to_paged_q8,
)
from ltx25_mlx.transformer import transformer_metadata


@pytest.mark.parametrize(
    ("kind", "prefix", "expected_format"),
    [
        (
            "transformer",
            "model.diffusion_model.transformer_blocks",
            PAGED_TRANSFORMER_FORMAT,
        ),
        ("gemma", "model.layers", PAGED_GEMMA_FORMAT),
    ],
)
def test_streamed_q8_converter_writes_directly_loadable_pages(
    tmp_path, kind, prefix, expected_format
):
    source = tmp_path / "source.safetensors"
    metadata = (
        {"model_version": "2.5.0", "config": json.dumps({"transformer": {}})}
        if kind == "transformer"
        else {
            "gemma_config": json.dumps(
                {
                    "model_type": "gemma4_unified",
                    "text_config": {
                        "hidden_size": 64,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 1,
                        "num_key_value_heads": 1,
                    },
                }
            )
        }
    )
    mx.save_safetensors(
        str(source),
        {
            "fixed.weight": mx.ones((64, 64), dtype=mx.bfloat16),
            f"{prefix}.0.proj.weight": mx.ones((64, 64), dtype=mx.bfloat16),
            f"{prefix}.0.norm.weight": mx.ones((64,), dtype=mx.bfloat16),
            f"{prefix}.1.proj.weight": mx.ones((64, 64), dtype=mx.bfloat16),
        },
        metadata=metadata,
    )
    destination = tmp_path / "paged"
    manifest = convert_to_paged_q8(source, destination, kind=kind)
    assert manifest.format == expected_format
    assert manifest.num_layers == 2
    assert manifest.output_tensor_bytes < manifest.source_tensor_bytes
    assert not any(key.endswith(".scales") for key in mx.load(str(manifest.fixed_path)))
    layer = mx.load(str(manifest.layer_paths[0]))
    assert f"{prefix}.0.proj.scales" in layer
    assert f"{prefix}.0.proj.biases" in layer
    assert LTX25PagedManifest.load(destination, verify_hashes=True) == manifest
    if kind == "transformer":
        assert transformer_metadata(destination)["model_version"] == "2.5.0"


def test_streamed_q8_converter_refuses_overwrite(tmp_path):
    source = tmp_path / "source.safetensors"
    mx.save_safetensors(
        str(source),
        {"model.layers.0.proj.weight": mx.ones((64, 64), dtype=mx.bfloat16)},
    )
    destination = tmp_path / "paged"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        convert_to_paged_q8(source, destination, kind="gemma")


def test_ltx25_page_prefetch_is_bounded_and_reports_metrics(tmp_path):
    from types import SimpleNamespace

    page = tmp_path / "layer.safetensors"
    page.write_bytes(b"page")

    def reader(index, path):
        assert path == page
        return PagePrefetchResult(index, path.stat().st_size, 0.25)

    prefetch = LTX25PagePrefetch(
        tmp_path,
        (SimpleNamespace(file=page.name),),
        enabled=True,
        reader=reader,
    )
    # The production backend is Darwin-only. Force the executor path in this
    # platform-independent unit test without changing the production default.
    if not prefetch.enabled:
        from concurrent.futures import ThreadPoolExecutor

        prefetch.enabled = True
        prefetch._executor = ThreadPoolExecutor(max_workers=1)
    prefetch.start(0)
    assert prefetch.wait(0) is True
    report = prefetch.report()
    prefetch.close()

    assert report["prefetch_depth"] == 1
    assert report["prefetch_requests"] == 1
    assert report["prefetch_hits"] == 1
    assert report["prefetch_failures"] == 0
    assert report["prefetch_bytes"] == 4
    assert report["prefetch_buffer_bytes"] == 0


def test_ltx25_page_prefetch_defaults_off_and_has_opt_in(monkeypatch):
    monkeypatch.delenv("WEETODD_LTX25_PAGE_PREFETCH", raising=False)
    assert LTX25PagePrefetch.default_enabled() is False
    monkeypatch.setenv("WEETODD_LTX25_PAGE_PREFETCH", "1")
    assert LTX25PagePrefetch.default_enabled() is True
