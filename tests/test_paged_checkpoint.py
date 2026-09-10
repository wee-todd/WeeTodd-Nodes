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
from minimax_h3_mlx.projection import MPPLinear, configure_projection_backend
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


def test_raw_page_cache_pins_within_budget_and_avoids_real_disk_loads(tmp_path, monkeypatch):
    manifest = convert_to_paged_checkpoint(_source(tmp_path), tmp_path / "paged")
    store = PagedTensorStore(manifest)
    disk_loads = []
    original_load = mx.load

    def tracked_load(filename, *args, **kwargs):
        disk_loads.append(filename)
        return original_load(filename, *args, **kwargs)

    monkeypatch.setattr(mx, "load", tracked_load)
    store.configure_cache(24)
    # Schedule preparation must not populate the pending cache.
    store.load_block(0)
    store.release()
    assert store.retained_bytes == 0
    store.begin_cache()
    for _ in range(3):
        for index in (0, 1):
            values = store.load_block(index)
            mx.eval(values)
            np.testing.assert_array_equal(np.asarray(next(iter(values.values()))), index + 2)
            values.clear()
            store.release()
            assert store.retained_bytes <= 24
    assert len(disk_loads) == 5  # preparation + first traversal + uncached page each repeat
    assert store.disk_page_loads == 5
    assert store.raw_cache_hits == 2
    assert store.retained_bytes == 24
    assert store.peak_retained_bytes == 24
    store.clear_retained_cache()
    assert store.retained_bytes == 0


def test_raw_page_cache_zero_or_undersized_budget_never_retains(tmp_path):
    manifest = convert_to_paged_checkpoint(_source(tmp_path), tmp_path / "paged")
    store = PagedTensorStore(manifest)
    for budget in (0, 23):
        store.configure_cache(budget)
        store.begin_cache()
        for _ in range(2):
            values = store.load_block(0)
            mx.eval(values)
            store.release()
        assert store.retained_bytes == 0
        assert store.raw_cache_hits == 0


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


def test_store_loads_only_selected_block_pages(tmp_path):
    manifest = convert_to_paged_checkpoint(_source(tmp_path), tmp_path / "paged")
    store = PagedTensorStore(manifest)

    values = store.load_blocks((1,))
    mx.eval(values)

    assert set(values) == {"blocks.1.attn.qkv_proj.weight"}
    assert store.pages_loaded == 1
    store.release()


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
    paged = load_paged_dit(paged_dir, window_size=2, verify_hashes=True, prefetch=True)

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


@pytest.mark.parametrize(
    ("setter", "component", "attribute"),
    [
        ("set_attention_head_chunk_size", "attn", "head_chunk_size"),
        ("set_ffn_row_chunk_size", "mlp", "row_chunk_size"),
    ],
)
def test_paged_chunk_controls_reach_new_windows_and_can_be_reset(
    tmp_path, setter, component, attribute
):
    config = _tiny_dit_config()
    mx.random.seed(49)
    resident = MiniMaxH3DiT(config)
    source = tmp_path / "full"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    convert_to_paged_checkpoint(source, tmp_path / "paged")
    paged = load_paged_dit(tmp_path / "paged", window_size=2, prefetch=False)
    pager = paged.paged_blocks
    args = _tiny_inputs(config)
    try:
        # Defaults, changes after prior windows, and reset must all reach future blocks.
        for chunk_size in (None, 1, 2, None):
            getattr(resident, setter)(chunk_size)
            getattr(paged, setter)(chunk_size)
            for start in (0, 2):
                with pager.window(start) as blocks:
                    for block in blocks:
                        assert getattr(getattr(block, component), attribute) == chunk_size
            with pager.selected_window((0, 2)) as blocks:
                for block in blocks:
                    assert getattr(getattr(block, component), attribute) == chunk_size
            expected = resident(*args)
            actual = paged(*args)
            mx.eval(expected, actual)
            for observed, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(
                    np.asarray(observed), np.asarray(reference), rtol=1e-5, atol=1e-5
                )
            assert pager.store.active_page is None
    finally:
        pager.close()


def test_paged_mpp_backend_wraps_each_materialized_bf16_block(tmp_path, monkeypatch):
    config = _tiny_dit_config()
    mx.random.seed(44)
    resident = MiniMaxH3DiT(config)
    resident.set_dtype(mx.bfloat16)
    mx.eval(resident.parameters())
    source = tmp_path / "full"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    paged_dir = tmp_path / "paged"
    convert_to_paged_checkpoint(source, paged_dir)
    paged = load_paged_dit(paged_dir, window_size=2)
    monkeypatch.setattr("minimax_h3_mlx.projection.mpp_capability", lambda: (True, None))

    report = configure_projection_backend(paged, "mpp_experimental")
    with paged.paged_blocks.window(0) as blocks:
        assert all(isinstance(block.attn.qkv_proj, MPPLinear) for block in blocks)
        assert all(isinstance(block.mlp.fc2, MPPLinear) for block in blocks)

    runtime = paged.paged_blocks.report()
    assert report.resolved == "mpp_experimental"
    assert runtime["projection_backend"] == "mpp_experimental"
    assert runtime["projection_wrapped"] == 8
    assert runtime["projection_skipped"] == 0
    paged.paged_blocks.close()


def test_paged_transformer_prefetch_defaults_off_and_can_use_environment(tmp_path, monkeypatch):
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


@pytest.mark.parametrize("quantized", [False, True])
def test_retained_pages_match_uncached_forward_exclude_adaln_and_release(tmp_path, quantized):
    from minimax_h3_mlx.adaln import drop_adaln_weights

    config = _tiny_dit_config()
    mx.random.seed(51)
    resident = MiniMaxH3DiT(config)
    source = tmp_path / "full"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(asdict(config)))
    if quantized:
        recipe = QuantConfig(bits=8, group_size=32, quantize_adaln=True, adaln_bits=8)
        quantize_dit(resident, recipe)
        (source / "quant_config.json").write_text(json.dumps(asdict(recipe)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    convert_to_paged_checkpoint(source, tmp_path / "paged")
    paged = load_paged_dit(tmp_path / "paged", window_size=2, prefetch=False)
    pager = paged.paged_blocks
    args = _tiny_inputs(config)
    cache = ModulationCache.build(paged, args[3], dtype=mx.float32)
    drop_adaln_weights(paged)
    expected = paged(*args, modulation_cache=cache)
    mx.eval(expected)
    page_zero = mx.load(str(tmp_path / "paged" / pager.manifest.blocks[0].file))
    budget = sum(v.nbytes for k, v in page_zero.items() if ".adaln_proj." not in k)
    pager.store.configure_cache(budget)
    pager.store.begin_cache()
    for _ in range(3):
        actual = paged(*args, modulation_cache=cache)
        mx.eval(actual)
        for observed, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(reference))
    report = pager.report()
    assert report["raw_cache_hits"] == 2
    assert report["raw_cache_retained_bytes"] == budget
    assert report["raw_cache_retained_bytes"] < pager.manifest.blocks[0].tensor_bytes
    assert report["raw_cache_budget_bytes"] == budget
    # Rebuilding a schedule must restore AdaLN weights and invalidate inference-only pages.
    rebuilt = ModulationCache.build(paged, args[3], dtype=mx.float32)
    for previous, current in zip(cache.tables, rebuilt.tables, strict=True):
        for a, b in zip(previous, current, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert pager.store.retained_bytes == 0
    drop_adaln_weights(paged)
    pager.store.begin_cache()
    with pytest.raises(KeyboardInterrupt):
        with pager.window(0):
            assert pager.store.retained_bytes == budget
            raise KeyboardInterrupt
    assert pager.store.retained_bytes == 0
    assert pager.store.active_page is None
    pager.store.begin_cache()
    with pager.window(0):
        pass
    pager.close()
    assert pager.store.retained_bytes == 0


@pytest.mark.parametrize("drop_adaln", [False, True])
def test_pipeline_starts_raw_cache_after_modulation_preparation(tmp_path, drop_adaln):
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    config = _tiny_dit_config()
    source = tmp_path / "full"
    source.mkdir()
    resident = MiniMaxH3DiT(config)
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(
        str(source / "model.safetensors"), dict(tree_flatten(resident.parameters()))
    )
    convert_to_paged_checkpoint(source, tmp_path / "paged")
    paged = load_paged_dit(tmp_path / "paged", window_size=2, prefetch=False)
    pager = paged.paged_blocks
    pager.store.configure_cache(1_000_000)
    pipeline = MiniMaxH3Pipeline(paged, None, None, None)
    try:
        result = pipeline.sample_latents(
            mx.zeros((1, 3, config.text_dim)), np.full(3, TAG_TEXT, dtype=np.int32),
            duration_seconds=2.5, num_inference_steps=3, width=32, height=32, verbose=False,
            drop_adaln=drop_adaln,
        )
        mx.eval(result.video_latents, result.audio_latents)
        assert pager.store.raw_cache_hits >= config.num_layers
        assert pager.store.raw_cache_misses == config.num_layers
        assert pager.store.retained_bytes > 0
        assert pager.skip_adaln == drop_adaln
    finally:
        pager.close()


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

    from minimax_h3_mlx.adaln import drop_adaln_weights

    drop_adaln_weights(paged)
    assert paged.paged_blocks.skip_adaln
    opens = paged.paged_blocks.adapter_file_opens
    skipped = paged(*args, modulation_cache=paged_cache)
    mx.eval(skipped)
    np.testing.assert_array_equal(np.asarray(skipped[0]), np.asarray(expected_video))
    np.testing.assert_array_equal(np.asarray(skipped[1]), np.asarray(expected_audio))
    assert paged.paged_blocks.adaln_bytes_avoided > 0
    assert paged.paged_blocks.adapter_file_opens - opens == (config.num_layers + 1) // 2
    paged.paged_blocks.store.configure_cache(1_000_000)
    paged.paged_blocks.store.begin_cache()
    for _ in range(2):
        retained = paged(*args, modulation_cache=paged_cache)
        mx.eval(retained)
        np.testing.assert_array_equal(np.asarray(retained[0]), np.asarray(expected_video))
        np.testing.assert_array_equal(np.asarray(retained[1]), np.asarray(expected_audio))
    assert paged.paged_blocks.store.raw_cache_hits > 0
    rebuilt = ModulationCache.build(paged, args[3], dtype=mx.float32)
    assert not paged.paged_blocks.skip_adaln
    for old, new in zip(paged_cache.tables, rebuilt.tables, strict=True):
        for a, b in zip(old, new, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_explicit_resident_materialization_preserves_paged_weights_and_forward(tmp_path):
    from minimax_h3_mlx.paged_checkpoint import materialize_paged_blocks

    config = _tiny_dit_config()
    source = tmp_path / "resident-source"
    source.mkdir()
    model = MiniMaxH3DiT(config)
    (source / "config.json").write_text(json.dumps(asdict(config)))
    mx.save_safetensors(str(source / "model.safetensors"), dict(tree_flatten(model.parameters())))
    convert_to_paged_checkpoint(source, tmp_path / "resident-paged")
    paged = load_paged_dit(tmp_path / "resident-paged", window_size=2)
    args = _tiny_inputs(config)
    expected = paged(*args)
    mx.eval(expected)
    pager = paged.paged_blocks
    report = materialize_paged_blocks(paged)
    assert paged.paged_blocks is None
    assert len(paged.blocks) == config.num_layers
    assert report["materialized_blocks"] == config.num_layers
    assert report["weight_bytes"] > 0
    loaded = pager.store.pages_loaded
    actual = paged(*args)
    for a, b in zip(actual, expected, strict=True):
        assert mx.array_equal(a, b).item()
    assert pager.store.pages_loaded == loaded
