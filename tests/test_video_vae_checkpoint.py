from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from minimax_h3_mlx.load import load_compact_video_vae, safetensor_metadata
from minimax_h3_mlx.video_vae_checkpoint import (
    VIDEO_VAE_METADATA_KEY,
    VIDEO_VAE_MLX_LAYOUT,
    VIDEO_VAE_NATIVE_FORMAT,
    VIDEO_VAE_NATIVE_FORMAT_VERSION,
    VIDEO_VAE_QUANTIZATION_FORMAT,
    VIDEO_VAE_QUANTIZATION_SCOPE,
    VIDEO_VAE_SOURCE_LAYOUT,
    apply_video_vae_quantization_structure,
    convert_video_vae_checkpoint,
    prepare_video_vae_tensor,
    quantize_video_vae_checkpoint,
    validate_video_vae_quantization,
    validate_video_vae_wrapper,
)


def _wrapper(**updates) -> dict:
    value = {
        "source_config": {"z_channels": 24},
        "vae_clip_length": 17,
        "vae_token_drop": 3,
        "latents_mean": [0.0] * 24,
        "latents_std": [1.0] * 24,
    }
    value.update(updates)
    return value


def _loadable_wrapper(**updates) -> dict:
    value = _wrapper(
        source_config={
            "ch": 1,
            "in_channels": 3,
            "out_ch": 3,
            "z_channels": 24,
            "ch_mult": [1],
            "num_res_blocks": 1,
            "space_down": [],
            "time_down": [],
            "vit_decoder_kwargs": {
                "num_layers": 1,
                "heads": 1,
                "dim_head": 1,
                "rope_theta": 10_000.0,
                "rope_dim_ratio": 1.0,
            },
        }
    )
    value.update(updates)
    return value


def _write_compact(path: Path, tensors: dict[str, mx.array], wrapper: dict) -> None:
    mx.save_safetensors(
        str(path),
        tensors,
        metadata={VIDEO_VAE_METADATA_KEY: json.dumps(wrapper)},
    )


def _q8_recipe(layers: int = 4) -> dict:
    return {
        "format": VIDEO_VAE_QUANTIZATION_FORMAT,
        "bits": 8,
        "group_size": 64,
        "scope": VIDEO_VAE_QUANTIZATION_SCOPE,
        "quantized_layers": layers,
    }


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(64, 192)
        self.to_out = nn.Linear(64, 64)


class _FakeFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(64, 128)
        self.w2 = nn.Linear(64, 64)


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttention()
        self.ff = _FakeFeedForward()


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = [_FakeBlock()]
        self.proj_out = nn.Linear(64, 64)


class _FakeVideoVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.decoder = _FakeDecoder()


def test_source_layout_conversion_is_bit_exact() -> None:
    source = mx.array(np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6))

    converted = prepare_video_vae_tensor(source, VIDEO_VAE_SOURCE_LAYOUT)

    expected = np.asarray(source).transpose(0, 2, 3, 4, 1)
    np.testing.assert_array_equal(np.asarray(converted), expected)
    assert converted.shape == (2, 4, 5, 6, 3)


def test_native_layout_is_not_transposed_again() -> None:
    native = mx.arange(24).reshape(1, 2, 3, 4, 1)

    assert prepare_video_vae_tensor(native, VIDEO_VAE_MLX_LAYOUT) is native


def test_converter_writes_versioned_native_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "native.safetensors"
    conv = mx.array(np.arange(2 * 3 * 2 * 2 * 2).reshape(2, 3, 2, 2, 2))
    bias = mx.arange(2)
    _write_compact(source, {"conv.weight": conv, "conv.bias": bias}, _wrapper())
    expected_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    report = convert_video_vae_checkpoint(source, output)

    converted = mx.load(str(output))
    np.testing.assert_array_equal(
        np.asarray(converted["conv.weight"]), np.asarray(conv).transpose(0, 2, 3, 4, 1)
    )
    np.testing.assert_array_equal(np.asarray(converted["conv.bias"]), np.asarray(bias))
    native_wrapper = json.loads(safetensor_metadata(output)[VIDEO_VAE_METADATA_KEY])
    assert native_wrapper["format"] == VIDEO_VAE_NATIVE_FORMAT
    assert native_wrapper["format_version"] == VIDEO_VAE_NATIVE_FORMAT_VERSION
    assert native_wrapper["tensor_layout"] == VIDEO_VAE_MLX_LAYOUT
    assert native_wrapper["source_sha256"] == expected_source_hash
    assert report["transposed_tensors"] == 1
    assert report["source_layout"] == VIDEO_VAE_SOURCE_LAYOUT
    assert report["output_layout"] == VIDEO_VAE_MLX_LAYOUT


def test_converter_is_layout_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "native-source.safetensors"
    output = tmp_path / "native-copy.safetensors"
    native = mx.arange(24).reshape(1, 2, 3, 4, 1)
    wrapper = _wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
    )
    _write_compact(source, {"conv.weight": native}, wrapper)

    report = convert_video_vae_checkpoint(source, output)

    np.testing.assert_array_equal(
        np.asarray(mx.load(str(output))["conv.weight"]), np.asarray(native)
    )
    assert report["source_layout"] == VIDEO_VAE_MLX_LAYOUT


def test_converter_preserves_existing_output_without_permission(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "native.safetensors"
    _write_compact(source, {"bias": mx.arange(2)}, _wrapper())
    output.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="Output already exists"):
        convert_video_vae_checkpoint(source, output)

    assert output.read_bytes() == b"keep"


def test_converter_accepts_released_directory_layout(tmp_path: Path) -> None:
    source = tmp_path / "video_vae"
    output = tmp_path / "native.safetensors"
    source_config = {"z_channels": 24, "ch": 128}
    (source / "source").mkdir(parents=True)
    (source / "config.json").write_text(json.dumps({"vae_clip_length": 17, "vae_token_drop": 3}))
    (source / "source" / "config.json").write_text(json.dumps(source_config))
    conv = mx.arange(24).reshape(1, 1, 2, 3, 4)
    mx.save_safetensors(str(source / "source" / "model.safetensors"), {"weight": conv})

    convert_video_vae_checkpoint(source, output)

    wrapper = json.loads(safetensor_metadata(output)[VIDEO_VAE_METADATA_KEY])
    assert wrapper["source_config"] == source_config
    np.testing.assert_array_equal(
        np.asarray(mx.load(str(output))["weight"]), np.asarray(conv).transpose(0, 2, 3, 4, 1)
    )


def test_converter_cleans_temporary_file_after_writer_failure(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.safetensors"
    output = tmp_path / "native.safetensors"
    _write_compact(source, {"bias": mx.arange(2)}, _wrapper())

    def fail_writer(*args, **kwargs):
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(mx, "save_safetensors", fail_writer)

    with pytest.raises(RuntimeError, match="simulated writer failure"):
        convert_video_vae_checkpoint(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp.safetensors")) == []


def test_unknown_native_format_version_is_rejected() -> None:
    wrapper = _wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION + 1,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
    )

    with pytest.raises(ValueError, match="format version"):
        validate_video_vae_wrapper(wrapper)


def test_q8_recipe_rejects_unsupported_width() -> None:
    recipe = _q8_recipe()
    recipe["bits"] = 4

    with pytest.raises(ValueError, match="quantization bits"):
        validate_video_vae_quantization({"quantization": recipe})


def test_q8_structure_quantizes_only_decoder_transformer_core() -> None:
    model = _FakeVideoVAE()
    before = sum(value.nbytes for _, value in tree_flatten(model.parameters()))

    count = apply_video_vae_quantization_structure(model, _q8_recipe())
    mx.eval(model.parameters())
    after = sum(value.nbytes for _, value in tree_flatten(model.parameters()))

    assert count == 4
    assert isinstance(model.decoder.transformer_blocks[0].attn.to_qkv, nn.QuantizedLinear)
    assert isinstance(model.decoder.transformer_blocks[0].ff.w2, nn.QuantizedLinear)
    assert isinstance(model.decoder.proj_out, nn.Linear)
    assert after < before


def test_q8_converter_writes_direct_load_metadata(tmp_path: Path, monkeypatch) -> None:
    import minimax_h3_mlx.load as load_module

    source = tmp_path / "native.safetensors"
    output = tmp_path / "q8.safetensors"
    wrapper = _loadable_wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
    )
    _write_compact(source, {"placeholder": mx.zeros((1,))}, wrapper)
    model = _FakeVideoVAE()
    monkeypatch.setattr(load_module, "load_compact_video_vae", lambda path: model)

    report = quantize_video_vae_checkpoint(source, output)

    saved_wrapper = json.loads(safetensor_metadata(output)[VIDEO_VAE_METADATA_KEY])
    assert saved_wrapper["quantization"] == _q8_recipe()
    assert report["quantized_layers"] == 4
    assert report["resident_after_bytes"] < report["resident_before_bytes"]
    assert any(key.endswith(".scales") for key in mx.load(str(output)))


def test_q8_converter_cleans_temporary_file_after_writer_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import minimax_h3_mlx.load as load_module

    source = tmp_path / "native.safetensors"
    output = tmp_path / "q8.safetensors"
    wrapper = _loadable_wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
    )
    _write_compact(source, {"placeholder": mx.zeros((1,))}, wrapper)
    monkeypatch.setattr(load_module, "load_compact_video_vae", lambda path: _FakeVideoVAE())

    def fail_writer(*args, **kwargs):
        raise RuntimeError("simulated q8 writer failure")

    monkeypatch.setattr(mx, "save_safetensors", fail_writer)

    with pytest.raises(RuntimeError, match="simulated q8 writer failure"):
        quantize_video_vae_checkpoint(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp.safetensors")) == []


def test_compact_q8_loader_builds_quantized_tree(tmp_path: Path, monkeypatch) -> None:
    import minimax_h3_mlx.video_vae as video_vae_module

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    source_model = _FakeVideoVAE(FakeConfig())
    apply_video_vae_quantization_structure(source_model, _q8_recipe())
    mx.eval(source_model.parameters())
    path = tmp_path / "q8.safetensors"
    wrapper = _loadable_wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
        quantization=_q8_recipe(),
    )
    _write_compact(path, dict(tree_flatten(source_model.parameters())), wrapper)
    monkeypatch.setattr(video_vae_module, "VideoVAEConfig", FakeConfig)
    monkeypatch.setattr(video_vae_module, "VideoVAE", _FakeVideoVAE)

    loaded = load_compact_video_vae(path)

    assert isinstance(loaded.decoder.transformer_blocks[0].attn.to_qkv, nn.QuantizedLinear)
    assert isinstance(loaded.decoder.transformer_blocks[0].ff.w2, nn.QuantizedLinear)
    assert isinstance(loaded.decoder.proj_out, nn.Linear)


def test_compact_loader_does_not_transpose_native_weights(tmp_path: Path, monkeypatch) -> None:
    import minimax_h3_mlx.video_vae as video_vae_module

    native = mx.arange(24).reshape(1, 2, 3, 4, 1)
    path = tmp_path / "native.safetensors"
    wrapper = _loadable_wrapper(
        format=VIDEO_VAE_NATIVE_FORMAT,
        format_version=VIDEO_VAE_NATIVE_FORMAT_VERSION,
        tensor_layout=VIDEO_VAE_MLX_LAYOUT,
    )
    _write_compact(path, {"weight": native}, wrapper)

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVideoVAE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = mx.zeros(native.shape, dtype=native.dtype)

    monkeypatch.setattr(video_vae_module, "VideoVAEConfig", FakeConfig)
    monkeypatch.setattr(video_vae_module, "VideoVAE", FakeVideoVAE)

    model = load_compact_video_vae(path)

    np.testing.assert_array_equal(np.asarray(model.weight), np.asarray(native))
