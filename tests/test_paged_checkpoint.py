from __future__ import annotations

import gc
import json
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.lora import LoRARequest, apply_lora, prepare_lora_timesteps
from minimax_h3_mlx.paged_checkpoint import (
    PAGED_MANIFEST,
    PagedCheckpointManifest,
    PagedTensorStore,
    convert_to_paged_checkpoint,
    load_paged_dit,
)
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"num_layers": 2}))
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            "condition_proj.weight": mx.ones((2, 2)),
            "blocks.0.attn.qkv_proj.weight": mx.full((3, 2), 2),
            "blocks.1.attn.qkv_proj.weight": mx.full((3, 2), 3),
        },
    )
    return source


def test_converter_writes_fixed_and_contiguous_block_pages(tmp_path):
    destination = tmp_path / "paged"
    manifest = convert_to_paged_checkpoint(_source(tmp_path), destination)

    assert manifest.num_blocks == 2
    assert (destination / PAGED_MANIFEST).is_file()
    assert (destination / "config.json").is_file()
    assert set(mx.load(str(destination / manifest.fixed.file))) == {"condition_proj.weight"}
    assert set(mx.load(str(destination / manifest.blocks[0].file))) == {
        "blocks.0.attn.qkv_proj.weight"
    }


def test_store_requires_release_between_pages_and_tracks_bytes(tmp_path):
    manifest = convert_to_paged_checkpoint(_source(tmp_path), tmp_path / "paged")
    store = PagedTensorStore(manifest)

    fixed = store.load_fixed()
    mx.eval(fixed)
    with pytest.raises(RuntimeError, match="still active"):
        store.load_block(0)
    store.release()

    block = store.load_block(0)
    mx.eval(block)
    assert store.active_page == "pages/block-000.safetensors"
    assert store.pages_loaded == 2
    assert store.peak_page_bytes >= sum(value.nbytes for value in block.values())
    del block, fixed
    store.release()
    gc.collect()
    assert store.active_page is None


def test_block_window_releases_after_cancellation_exception(tmp_path):
    manifest = convert_to_paged_checkpoint(_source(tmp_path), tmp_path / "paged")
    store = PagedTensorStore(manifest)

    with pytest.raises(RuntimeError, match="cancelled"):
        try:
            store.load_block_window(0, 2)
            raise RuntimeError("cancelled")
        finally:
            store.release()

    assert store.active_page is None


def test_manifest_rejects_a_modified_page(tmp_path):
    destination = tmp_path / "paged"
    manifest = convert_to_paged_checkpoint(_source(tmp_path), destination)
    page = destination / manifest.blocks[0].file
    page.write_bytes(page.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="hash differs"):
        PagedCheckpointManifest.load(destination, verify_hashes=True)


def test_manifest_rejects_a_page_outside_the_checkpoint(tmp_path):
    destination = tmp_path / "paged"
    convert_to_paged_checkpoint(_source(tmp_path), destination)
    manifest_path = destination / PAGED_MANIFEST
    raw = json.loads(manifest_path.read_text())
    raw["fixed"]["file"] = "../outside.safetensors"
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="escapes the checkpoint root"):
        PagedCheckpointManifest.load(destination)


def _tiny_dit_config():
    hidden = 64
    return DiTConfig(
        hidden_size=hidden,
        num_layers=3,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=32,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )


def _tiny_inputs(config):
    text_rows, video_rows, audio_rows = 3, 5, 2
    rows = text_rows + video_rows + audio_rows
    generator = np.random.default_rng(9)
    tags = np.concatenate(
        [
            np.full(text_rows, TAG_TEXT),
            np.full(video_rows, TAG_VIDEO),
            np.full(audio_rows, TAG_AUDIO),
        ]
    ).astype(np.int32)
    timestep_indices = np.concatenate(
        [np.zeros(text_rows), np.ones(video_rows), np.zeros(audio_rows)]
    ).astype(np.int32)
    positions = np.stack(
        [np.arange(rows) % 3, np.arange(rows) % 5, np.arange(rows) % 7], axis=-1
    ).astype(np.float32)
    return (
        mx.array(generator.standard_normal((1, video_rows, config.video_patch_dim))),
        mx.array(generator.standard_normal((1, audio_rows, config.audio_latents_dim))),
        mx.array(generator.standard_normal((1, text_rows, config.text_dim))),
        mx.array([0.0, 0.6]),
        mx.array(timestep_indices),
        mx.array(tags),
        mx.array(positions),
        mx.array(np.arange(text_rows, text_rows + video_rows, dtype=np.int32)),
        mx.array(np.arange(text_rows + video_rows, rows, dtype=np.int32)),
        mx.array(np.arange(text_rows, dtype=np.int32)),
    )


def test_paged_forward_and_modulation_cache_match_resident_model(tmp_path):
    config = _tiny_dit_config()
    mx.random.seed(4)
    resident = MiniMaxH3DiT(config)
    mx.eval(resident.parameters())
    source = tmp_path / "full"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    paged_dir = tmp_path / "paged"
    convert_to_paged_checkpoint(source, paged_dir)
    paged = load_paged_dit(
        paged_dir, window_size=2, verify_hashes=True, prefetch=True
    )

    args = _tiny_inputs(config)
    resident_cache = ModulationCache.build(resident, args[3], dtype=mx.float32)
    paged_cache = ModulationCache.build(paged, args[3], dtype=mx.float32)
    expected_video, expected_audio = resident(*args, modulation_cache=resident_cache)
    actual_video, actual_audio = paged(*args, modulation_cache=paged_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    np.testing.assert_array_equal(np.asarray(actual_video), np.asarray(expected_video))
    np.testing.assert_array_equal(np.asarray(actual_audio), np.asarray(expected_audio))
    assert paged.paged_blocks.store.active_page is None
    assert paged.paged_blocks.store.peak_page_bytes > 0
    report = paged.paged_blocks.report()
    assert report["prefetch_enabled"] is True
    # Modulation precomputation and the transformer forward each traverse both windows.
    assert report["prefetch_requests"] == 2
    assert report["prefetch_hits"] == 2
    assert report["prefetch_failures"] == 0
    assert report["windows_materialized"] == 4
    assert report["prefetch_backend"] == "darwin_advisory"
    assert report["prefetch_buffer_bytes"] == 0
    paged.paged_blocks.close()


def test_paged_transformer_prefetch_defaults_off_and_can_use_environment(
    tmp_path, monkeypatch
):
    config = _tiny_dit_config()
    mx.random.seed(14)
    resident = MiniMaxH3DiT(config)
    mx.eval(resident.parameters())
    source = tmp_path / "full"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    paged_dir = tmp_path / "paged"
    convert_to_paged_checkpoint(source, paged_dir)

    monkeypatch.delenv("WEETODD_H3_TRANSFORMER_PREFETCH", raising=False)
    default = load_paged_dit(paged_dir, window_size=2)
    assert default.paged_blocks.report()["prefetch_enabled"] is False
    default.paged_blocks.close()

    monkeypatch.setenv("WEETODD_H3_TRANSFORMER_PREFETCH", "1")
    enabled = load_paged_dit(paged_dir, window_size=2)
    assert enabled.paged_blocks.report()["prefetch_enabled"] is True
    enabled.paged_blocks.close()


def test_quantized_paged_forward_matches_resident_model(tmp_path):
    config = _tiny_dit_config()
    mx.random.seed(5)
    resident = MiniMaxH3DiT(config)
    mx.eval(resident.parameters())
    recipe = QuantConfig(bits=8, group_size=32, quantize_adaln=True, adaln_bits=8)
    quantize_dit(resident, recipe)
    source = tmp_path / "quantized"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    (source / "quant_config.json").write_text(
        json.dumps(
            {
                "bits": recipe.bits,
                "group_size": recipe.group_size,
                "quantize_adaln": recipe.quantize_adaln,
                "adaln_bits": recipe.adaln_bits,
                "quantize_core": recipe.quantize_core,
                "overrides": recipe.overrides,
            }
        )
    )
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    paged_dir = tmp_path / "paged-quantized"
    convert_to_paged_checkpoint(source, paged_dir)
    paged = load_paged_dit(paged_dir, window_size=2)

    args = _tiny_inputs(config)
    resident_cache = ModulationCache.build(resident, args[3], dtype=mx.float32)
    paged_cache = ModulationCache.build(paged, args[3], dtype=mx.float32)
    expected_video, expected_audio = resident(*args, modulation_cache=resident_cache)
    actual_video, actual_audio = paged(*args, modulation_cache=paged_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    np.testing.assert_array_equal(np.asarray(actual_video), np.asarray(expected_video))
    np.testing.assert_array_equal(np.asarray(actual_audio), np.asarray(expected_audio))


def test_paged_block_lora_matches_resident_adapter(tmp_path):
    config = _tiny_dit_config()
    mx.random.seed(6)
    original = MiniMaxH3DiT(config)
    mx.eval(original.parameters())
    source = tmp_path / "lora-base"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(original.parameters()))
    )
    resident = load_dit(source)
    paged_dir = tmp_path / "lora-paged"
    convert_to_paged_checkpoint(source, paged_dir)
    paged = load_paged_dit(paged_dir, window_size=2)
    adapter_path = tmp_path / "adapter.safetensors"
    output_width = 3 * config.num_attention_heads * config.attention_head_dim
    mx.save_safetensors(
        str(adapter_path),
        {
            "blocks.0.attn.qkv_proj.lora_A.weight": mx.full((2, 64), 0.01),
            "blocks.0.attn.qkv_proj.lora_B.weight": mx.arange(
                output_width * 2, dtype=mx.float32
            ).reshape(output_width, 2)
            / output_width,
        },
    )
    request = LoRARequest(str(adapter_path), strength=0.75, qkv_layout="contiguous_qkv")
    apply_lora(resident, request)
    apply_lora(paged, request)

    args = _tiny_inputs(config)
    prepare_lora_timesteps(resident, args[3])
    prepare_lora_timesteps(paged, args[3])
    resident_cache = ModulationCache.build(resident, args[3], dtype=mx.float32)
    paged_cache = ModulationCache.build(paged, args[3], dtype=mx.float32)
    expected_video, expected_audio = resident(*args, modulation_cache=resident_cache)
    actual_video, actual_audio = paged(*args, modulation_cache=paged_cache)
    mx.eval(expected_video, expected_audio, actual_video, actual_audio)

    np.testing.assert_array_equal(np.asarray(actual_video), np.asarray(expected_video))
    np.testing.assert_array_equal(np.asarray(actual_audio), np.asarray(expected_audio))
    assert paged.paged_blocks.report()["lora_count"] == 1
