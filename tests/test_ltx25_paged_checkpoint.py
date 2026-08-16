import json

import mlx.core as mx
import pytest

from ltx25_mlx.page_prefetch import LTX25PagePrefetch, PagePrefetchResult
from ltx25_mlx.paged_checkpoint import (
    PAGED_GEMMA_FORMAT,
    PAGED_TRANSFORMER_FORMAT,
    LTX25PagedManifest,
    convert_to_paged_q8,
    fuse_paged_transformer_loras,
)
from ltx25_mlx.runtime import LTX25GenerationConfig, validate_ltx25_dfr_prebaked_pair
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


def test_paged_lora_fusion_bakes_block_and_non_block_targets(tmp_path):
    source = tmp_path / "source.safetensors"
    block_key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    fixed_key = "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
    connector_key = "model.diffusion_model.video_embeddings_connector.0.weight"
    mx.save_safetensors(
        str(source),
        {
            block_key: mx.zeros((64, 64), dtype=mx.bfloat16),
            fixed_key: mx.zeros((64, 64), dtype=mx.bfloat16),
            connector_key: mx.ones((3, 3), dtype=mx.bfloat16),
        },
        metadata={"model_version": "2.5.0", "config": json.dumps({"transformer": {}})},
    )
    paged = convert_to_paged_q8(source, tmp_path / "paged", kind="transformer")
    adapter = tmp_path / "adapter.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones(
                (2, 64), dtype=mx.bfloat16
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones(
                (64, 2), dtype=mx.bfloat16
            ),
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight": mx.ones(
                (2, 64), dtype=mx.bfloat16
            ),
            "diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_B.weight": mx.ones(
                (64, 2), dtype=mx.bfloat16
            ),
        },
        metadata={"model_version": "2.5.0", "lora_rank": "2", "lora_alpha": "2"},
    )
    fused = fuse_paged_transformer_loras(
        paged.root, tmp_path / "fused", ((adapter, 0.5),)
    )
    assert LTX25PagedManifest.load(fused.root, verify_hashes=True) == fused
    assert fused.metadata["weetodd_baked_loras"][0]["lora_rank"] == 2
    fixed = mx.load(str(fused.fixed_path))
    layer = mx.load(str(fused.layer_paths[0]))
    assert mx.any(fixed[fixed_key] != 0).item()
    assert mx.array_equal(fixed[connector_key], mx.ones((3, 3), dtype=mx.bfloat16))
    module = block_key.removesuffix(".weight")
    restored = mx.dequantize(
        layer[block_key],
        layer[f"{module}.scales"],
        layer[f"{module}.biases"],
        group_size=64,
        bits=8,
        mode="affine",
    )
    assert mx.any(restored != 0).item()
    with pytest.raises(ValueError, match="original Q8 pages"):
        fuse_paged_transformer_loras(
            fused.root, tmp_path / "sequential", ((adapter, 0.5),)
        )


def test_paged_ic_lora_fusion_preserves_reference_contract(tmp_path):
    source = tmp_path / "source.safetensors"
    block_key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    mx.save_safetensors(
        str(source),
        {block_key: mx.zeros((64, 64), dtype=mx.bfloat16)},
        metadata={"model_version": "2.5.0", "config": json.dumps({"transformer": {}})},
    )
    paged = convert_to_paged_q8(source, tmp_path / "paged", kind="transformer")
    adapter = tmp_path / "ingredients-reference.safetensors"
    mx.save_safetensors(
        str(adapter),
        {
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones(
                (2, 64), dtype=mx.bfloat16
            ),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones(
                (64, 2), dtype=mx.bfloat16
            ),
        },
        metadata={
            "model_version": "2.5.0",
            "reference_downscale_factor": "2",
            "reference_temporal_scale_factor": "3",
        },
    )

    fused = fuse_paged_transformer_loras(
        paged.root, tmp_path / "fused", ((adapter, 1.2),)
    )

    baked = fused.metadata["weetodd_baked_loras"][0]
    assert baked["adapter_role"] == "ic_lora"
    assert baked["adapter_family"] == "ingredients_reference_sheet"
    assert baked["ic_lora_task"] == "reference_conditioning"
    assert baked["reference_downscale_factor"] == 2
    assert baked["reference_temporal_scale_factor"] == 3
    assert baked["strength"] == pytest.approx(1.2)


def test_dfr_prebaked_pair_validates_adapter_provenance(tmp_path):
    source = tmp_path / "source.safetensors"
    block_key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    mx.save_safetensors(
        str(source),
        {block_key: mx.zeros((64, 64), dtype=mx.bfloat16)},
        metadata={"model_version": "2.5.0", "config": json.dumps({"transformer": {}})},
    )
    paged = convert_to_paged_q8(source, tmp_path / "paged", kind="transformer")

    base = tmp_path / "base.safetensors"
    detail = tmp_path / "detail.safetensors"
    adapter_values = {
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": mx.ones(
            (2, 64), dtype=mx.bfloat16
        ),
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": mx.ones(
            (64, 2), dtype=mx.bfloat16
        ),
    }
    mx.save_safetensors(
        str(base),
        adapter_values,
        metadata={"model_version": "2.5.0", "lora_rank": "450", "lora_alpha": "450"},
    )
    mx.save_safetensors(
        str(detail),
        adapter_values,
        metadata={"model_version": "2.5.0", "reference_downscale_factor": "2"},
    )
    combined = fuse_paged_transformer_loras(
        paged.root, tmp_path / "combined", ((base, 1.0), (detail, 1.0))
    )
    baked = combined.metadata["weetodd_baked_loras"]
    config = LTX25GenerationConfig(
        dfr_prebaked_transformer_path=str(combined.root),
        dfr_detailing_lora_path=str(detail),
    )
    validate_ltx25_dfr_prebaked_pair(
        config,
        {"transformer_baked_loras": [baked[0]]},
    )
    changed_detail = tmp_path / "changed-detail.safetensors"
    changed_detail.write_bytes(detail.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="selected Pixel-Spatial"):
        validate_ltx25_dfr_prebaked_pair(
            LTX25GenerationConfig(
                dfr_prebaked_transformer_path=str(combined.root),
                dfr_detailing_lora_path=str(changed_detail),
            ),
            {"transformer_baked_loras": [baked[0]]},
        )


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
