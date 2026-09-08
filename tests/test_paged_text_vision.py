"""Real tiny Qwen fixtures exercise paging without checkpoint downloads."""

from __future__ import annotations

import json
import weakref
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_vl.config import TextConfig, VisionConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
from mlx_vlm.models.qwen3_vl.vision import VisionModel

from minimax_h3_mlx.paged_text_encoder import (
    PAGED_QWEN_MANIFEST,
    PagedTextEncoderManifest,
    convert_to_paged_text_encoder,
)
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder


def _tiny_source(tmp_path):
    mx.random.seed(37)
    text = TextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10000,
        max_position_embeddings=128,
        rope_scaling={"mrope_section": [2, 3, 3], "rope_type": "default"},
    )
    vision = VisionConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        out_hidden_size=64,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    language_model, vision_model = Qwen3VLModel(text), VisionModel(vision)
    for model in (language_model, vision_model):
        nn.quantize(model, group_size=64, bits=8)
    weights = {f"model.{k}": v for k, v in tree_flatten(language_model.parameters())}
    weights.update({f"visual.{k}": v for k, v in tree_flatten(vision_model.parameters())})
    # The compact H3 export omits the unused final norm.
    weights.pop("model.norm.weight")
    source = tmp_path / "compact"
    source.mkdir()
    mx.save_safetensors(str(source / "text_encoder.safetensors"), weights)
    raw = {
        "model_type": "qwen3_vl",
        "text_config": {**asdict(text), "num_hidden_layers": 3},
        "vision_config": asdict(vision),
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 62,
        "vision_end_token_id": 63,
    }
    (source / "config.json").write_text(json.dumps(raw))
    return source


def _request(kind):
    rng = np.random.default_rng(83)
    if kind == "mixed":
        ids = mx.array([[3, 62, 60, 63, 4, 62, 61, 61, 63, 5]])
        units = [
            (60, rng.normal(size=(4, 24)).astype(np.float32), np.array([[1, 2, 2]])),
            (61, rng.normal(size=(8, 24)).astype(np.float32), np.array([[2, 2, 2]])),
        ]
    elif kind == "image":
        ids = mx.array([[3, 62, 60, 63, 5]])
        units = (rng.normal(size=(4, 24)).astype(np.float32), np.array([[1, 2, 2]]))
    else:
        ids, units = mx.array([[3, 4, 5]]), None
    return ids, np.arange(ids.shape[1], dtype=np.int32) % 2, units


def _encoder(source, dtype=mx.float32):
    return MiniMaxH3TextEncoder(source, num_layers=2, dtype=dtype)


def test_converter_preserves_vision_quantization_and_bfloat_storage(tmp_path):
    source = _tiny_source(tmp_path)
    weights = mx.load(str(source / "text_encoder.safetensors"))
    weights["visual.patch_embed.proj.weight"] = weights["visual.patch_embed.proj.weight"].astype(
        mx.bfloat16
    )
    mx.eval(weights)
    mx.save_safetensors(str(source / "text_encoder.safetensors"), weights)
    manifest = convert_to_paged_text_encoder(
        source, tmp_path / "paged", num_layers=2, include_vision=True
    )
    assert manifest.supports_vision
    assert manifest.skipped_visual_bytes == 0
    actual = mx.load(str(manifest.root / manifest.vision.file))
    expected = {k: v for k, v in weights.items() if k.startswith("visual.")}
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert mx.array_equal(actual[key], expected[key]).item()


@pytest.mark.parametrize("damage", ["missing_record", "empty", "missing_file", "escape", "corrupt"])
def test_manifest_rejects_invalid_vision_page(tmp_path, damage):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2, include_vision=True
    )
    file = manifest.root / PAGED_QWEN_MANIFEST
    raw = json.loads(file.read_text())
    if damage == "missing_record":
        raw.pop("vision")
    elif damage == "empty":
        raw["vision"]["tensor_count"] = 0
    elif damage == "missing_file":
        (manifest.root / manifest.vision.file).unlink()
    elif damage == "escape":
        raw["vision"]["file"] = "../outside.safetensors"
    else:
        page = manifest.root / manifest.vision.file
        page.write_bytes(page.read_bytes() + b"corrupt")
    file.write_text(json.dumps(raw))
    with pytest.raises((ValueError, FileNotFoundError), match="vision|page"):
        PagedTextEncoderManifest.load(manifest.root, verify_hashes=True)


@pytest.mark.parametrize("kind", ["text", "image", "mixed"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_tiny_quantized_multimodal_parity_and_reload(tmp_path, monkeypatch, kind, dtype):
    source = _tiny_source(tmp_path)
    manifest = convert_to_paged_text_encoder(
        source, tmp_path / "paged", num_layers=2, include_vision=True
    )
    resident, paged = _encoder(source, dtype), _encoder(manifest.root, dtype)
    request = _request(kind)
    for encoder in (resident, paged):
        monkeypatch.setattr(encoder, "build_request", lambda *_: request)
    args = {"references": [object()]} if kind == "mixed" else {}
    expected, tags = resident.encode("fixture", **args)
    weak_visions = []
    import minimax_h3_mlx.vision_forward as vf

    actual_vision = vf.encode_vision

    def observed(vision, pixels, grid):
        weak_visions.append(weakref.ref(vision))
        return actual_vision(vision, pixels, grid)

    monkeypatch.setattr(vf, "encode_vision", observed)
    real_hidden = paged._hidden_states

    def after_vision(*args, **kwargs):
        assert paged.vision is None
        assert all(ref() is None for ref in weak_visions)
        return real_hidden(*args, **kwargs)

    monkeypatch.setattr(paged, "_hidden_states", after_vision)
    for _ in range(2):
        actual, actual_tags = paged.encode("fixture", **args)
        assert actual.dtype == expected.dtype == dtype
        np.testing.assert_allclose(
            np.asarray(actual.astype(mx.float32)),
            np.asarray(expected.astype(mx.float32)),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_array_equal(actual_tags, tags)
        assert paged.vision is None
        assert paged.paged_layers.store.active_page is None
    assert len(weak_visions) == {"text": 0, "image": 2, "mixed": 4}[kind]
    paged.paged_layers.close()


def test_paged_constructor_never_constructs_language_layers(tmp_path, monkeypatch):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2
    )
    import mlx_vlm.models.qwen3_vl.language as language

    def forbidden(*args, **kwargs):
        raise AssertionError("language layer constructed before encode")

    monkeypatch.setattr(language, "Qwen3VLDecoderLayer", forbidden)
    encoder = MiniMaxH3TextEncoder(manifest.root, num_layers=2, dtype=mx.float32, load_vision=False)
    assert encoder.language.layers == []
    encoder.paged_layers.close()


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_vision_failure_releases_weights_and_encoder_can_retry(tmp_path, monkeypatch, error):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2, include_vision=True
    )
    encoder = _encoder(manifest.root)
    monkeypatch.setattr(encoder, "build_request", lambda *_: _request("mixed"))
    import minimax_h3_mlx.vision_forward as vf

    real = vf.encode_vision
    refs = []

    def fail(vision, *_):
        refs.append(weakref.ref(vision))
        raise error("injected vision failure")

    monkeypatch.setattr(vf, "encode_vision", fail)
    with pytest.raises(error, match="injected"):
        encoder.encode("fixture", references=[object()])
    assert encoder.vision is None
    assert all(ref() is None for ref in refs)
    assert encoder.paged_layers.store.pages_loaded == 0
    monkeypatch.setattr(vf, "encode_vision", real)
    actual, _ = encoder.encode("fixture", references=[object()])
    assert mx.all(mx.isfinite(actual)).item()
    assert encoder.vision is None
    encoder.paged_layers.close()


@pytest.mark.parametrize(
    "damage", ["missing", "unexpected", "position_ids", "shape", "quant_dtype", "quant_scales"]
)
def test_invalid_vision_weights_fail_before_visual_inference(tmp_path, monkeypatch, damage):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2, include_vision=True
    )
    page = manifest.root / manifest.vision.file
    weights = mx.load(str(page))
    if damage == "missing":
        weights.pop("visual.patch_embed.proj.weight")
    elif damage == "unexpected":
        weights["visual.unused.weight"] = mx.ones((2,))
    elif damage == "position_ids":
        weights["visual.invalid.position_ids"] = mx.ones((2,))
    elif damage == "shape":
        weights["visual.patch_embed.proj.weight"] = mx.ones((1, 2, 2, 2, 3))
    else:
        key = next(k for k in weights if k.endswith(".scales"))
        if damage == "quant_scales":
            weights.pop(key)
        else:
            key = key.removesuffix(".scales") + ".weight"
            weights[key] = weights[key].astype(mx.float32)
    mx.eval(weights)
    mx.save_safetensors(str(page), weights)
    encoder = _encoder(manifest.root)
    monkeypatch.setattr(encoder, "build_request", lambda *_: _request("image"))
    with pytest.raises((ValueError, KeyError), match="vision|Vision"):
        encoder.encode("fixture")
    assert encoder.vision is None
    assert encoder.paged_layers.store.pages_loaded == 0
    encoder.paged_layers.close()


def test_v1_text_parity_and_visual_request_rejected(tmp_path, monkeypatch):
    source = _tiny_source(tmp_path)
    manifest = convert_to_paged_text_encoder(source, tmp_path / "paged", num_layers=2)
    assert not manifest.supports_vision
    assert manifest.vision is None
    resident, encoder = _encoder(source), _encoder(manifest.root)
    for model in (resident, encoder):
        monkeypatch.setattr(model, "build_request", lambda *_: _request("text"))
    expected, _ = resident.encode("fixture")
    actual, _ = encoder.encode("fixture")
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)
    monkeypatch.setattr(encoder, "build_request", lambda *_: _request("image"))
    with pytest.raises(ValueError, match="vision page"):
        encoder.encode("fixture")
    encoder.paged_layers.close()


def test_conversion_does_not_materialize_source_checkpoint(tmp_path, monkeypatch):
    source = _tiny_source(tmp_path)
    original_load = mx.load

    def no_source_load(file, *args, **kwargs):
        assert str(file) != str(source / "text_encoder.safetensors")
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(mx, "load", no_source_load)
    manifest = convert_to_paged_text_encoder(
        source, tmp_path / "paged", num_layers=2, include_vision=True
    )
    assert manifest.supports_vision


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_layer_construction_failure_releases_page_for_retry(tmp_path, monkeypatch, error):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2, include_vision=True
    )
    encoder = _encoder(manifest.root)
    monkeypatch.setattr(encoder, "build_request", lambda *_: _request("text"))
    import mlx_vlm.models.qwen3_vl.language as language

    real = language.Qwen3VLDecoderLayer

    def fail(*args, **kwargs):
        raise error("injected constructor failure")

    monkeypatch.setattr(language, "Qwen3VLDecoderLayer", fail)
    with pytest.raises(error, match="injected"):
        encoder.encode("fixture")
    assert encoder.paged_layers.store.active_page is None
    monkeypatch.setattr(language, "Qwen3VLDecoderLayer", real)
    actual, _ = encoder.encode("fixture")
    assert mx.all(mx.isfinite(actual)).item()
    encoder.paged_layers.close()


def test_previous_language_layer_is_released_before_next_construction(tmp_path, monkeypatch):
    manifest = convert_to_paged_text_encoder(
        _tiny_source(tmp_path), tmp_path / "paged", num_layers=2
    )
    encoder = _encoder(manifest.root)
    monkeypatch.setattr(encoder, "build_request", lambda *_: _request("text"))
    import mlx_vlm.models.qwen3_vl.language as language

    real = language.Qwen3VLDecoderLayer
    refs = []

    def observed(*args, **kwargs):
        assert all(ref() is None for ref in refs), "previous layer remains resident"
        layer = real(*args, **kwargs)
        refs.append(weakref.ref(layer))
        return layer

    monkeypatch.setattr(language, "Qwen3VLDecoderLayer", observed)
    actual, _ = encoder.encode("fixture")
    assert mx.all(mx.isfinite(actual)).item()
    assert all(ref() is None for ref in refs)
    encoder.paged_layers.close()
