from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.tae_preview import H3TinyPreviewDecoder, preview_statistics
from wee_todd_nodes.nodes import WeeToddH3PreviewOverride
from wee_todd_nodes.preflight import H3ComponentSetSpec
from wee_todd_nodes.preview import H3PreviewConfig, H3PreviewSession


def _zero_decoder_weights():
    weights = {
        "decoder.1.weight": mx.zeros((256, 24, 3, 3), dtype=mx.float16),
        "decoder.1.bias": mx.zeros((256,), dtype=mx.float16),
        "decoder.7.conv.weight": mx.zeros((256, 256, 1, 1), dtype=mx.float16),
        "decoder.8.weight": mx.zeros((128, 256, 3, 3), dtype=mx.float16),
        "decoder.13.conv.weight": mx.zeros((256, 128, 1, 1), dtype=mx.float16),
        "decoder.14.weight": mx.zeros((64, 128, 3, 3), dtype=mx.float16),
        "decoder.19.conv.weight": mx.zeros((128, 64, 1, 1), dtype=mx.float16),
        "decoder.20.weight": mx.zeros((64, 64, 3, 3), dtype=mx.float16),
        "decoder.22.weight": mx.zeros((12, 64, 3, 3), dtype=mx.float16),
        "decoder.22.bias": mx.zeros((12,), dtype=mx.float16),
    }
    blocks = (
        (3, 256),
        (4, 256),
        (5, 256),
        (9, 128),
        (10, 128),
        (11, 128),
        (15, 64),
        (16, 64),
        (17, 64),
    )
    for index, channels in blocks:
        weights[f"decoder.{index}.conv.0.weight"] = mx.zeros(
            (channels, channels * 2, 3, 3), dtype=mx.float16
        )
        weights[f"decoder.{index}.conv.0.bias"] = mx.zeros((channels,), dtype=mx.float16)
        for layer in (2, 4):
            weights[f"decoder.{index}.conv.{layer}.weight"] = mx.zeros(
                (channels, channels, 3, 3), dtype=mx.float16
            )
            weights[f"decoder.{index}.conv.{layer}.bias"] = mx.zeros((channels,), dtype=mx.float16)
    return weights


def test_mlx_preview_decoder_shape_and_range():
    decoder = H3TinyPreviewDecoder(_zero_decoder_weights())
    result = decoder.decode(mx.zeros((1, 24, 10, 4, 6)), max_edge=96)

    assert result.shape == (22, 64, 96, 3)
    assert result.dtype == np.float32
    assert np.all(result == 0)
    decoder.release()


def test_preview_statistics_identify_featureless_output():
    stats = preview_statistics(np.full((6, 64, 96, 3), 0.4, dtype=np.float32))
    assert stats.finite is True
    assert stats.collapsed is True

    varied = np.zeros((6, 64, 96, 3), dtype=np.float32)
    varied[:, :, 48:] = 1.0
    assert preview_statistics(varied).collapsed is False

    # A smooth brightness ramp resembles the observed muddy H3 checkpoint failure: its total
    # luminance range is non-trivial, but it has no useful local structure.
    ramp = np.linspace(0.3, 0.55, 96, dtype=np.float32)[None, None, :, None]
    ramp = np.broadcast_to(ramp, (6, 64, 96, 3))
    assert preview_statistics(ramp).collapsed is True


def test_preview_override_preserves_component_contract(tmp_path: Path):
    tae = tmp_path / "taeh3.safetensors"
    tae.write_bytes(b"header-only test placeholder")
    components = H3ComponentSetSpec(checkpoint=str(tmp_path), task="t2va")

    (previewed,) = WeeToddH3PreviewOverride().apply(
        components,
        str(tae),
        "mlx",
        "",
        2,
        6,
        384,
        "preview only",
    )

    assert components.preview_override is None
    assert previewed.preview_override == H3PreviewConfig(
        tae_path=str(tae),
        backend="mlx",
        every_n_evaluations=2,
        preview_frames=6,
        max_edge=384,
        guard_mode="preview only",
    )


def test_conservative_guard_requires_repeated_late_collapse(monkeypatch, tmp_path: Path):
    tae = tmp_path / "taeh3.safetensors"
    tae.write_bytes(b"placeholder")

    class FakeDecoder:
        @classmethod
        def from_safetensors(cls, _path):
            return cls()

        def decode(self, _latents, *, max_edge):
            assert max_edge == 128
            return np.full((6, 32, 48, 3), 0.4, dtype=np.float32)

        def release(self):
            pass

    monkeypatch.setattr("minimax_h3_mlx.tae_preview.H3TinyPreviewDecoder", FakeDecoder)
    session = H3PreviewSession(
        H3PreviewConfig(
            tae_path=str(tae),
            preview_frames=6,
            max_edge=128,
            guard_mode="conservative collapse guard",
        )
    )
    latents = mx.zeros((1, 24, 10, 4, 6))
    assert session.update(latents, 1, 4).reject_reason is None
    assert session.update(latents, 2, 4).reject_reason is None
    assert "featureless" in session.update(latents, 3, 4).reject_reason
    session.release()


def test_preview_config_rejects_missing_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="preview TAE not found"):
        H3PreviewConfig(tae_path=str(tmp_path / "missing.safetensors")).validate()


def test_auto_preview_prefers_coreml_and_records_backend(monkeypatch, tmp_path: Path):
    tae = tmp_path / "taeh3.safetensors"
    tae.write_bytes(b"placeholder")
    coreml = tmp_path / "taeh3.mlmodelc"
    coreml.mkdir()

    class FakeCoreMLDecoder:
        def __init__(self, path):
            assert Path(path) == coreml

        def decode(self, _latents, *, max_edge):
            assert max_edge == 128
            return np.zeros((6, 64, 96, 3), dtype=np.float32)

        def release(self):
            pass

    monkeypatch.setattr("minimax_h3_mlx.coreml_preview.H3CoreMLPreviewDecoder", FakeCoreMLDecoder)
    session = H3PreviewSession(
        H3PreviewConfig(
            tae_path=str(tae),
            backend="auto",
            coreml_model_path=str(coreml),
            max_edge=128,
        )
    )

    assert session.backend == "neural engine"
    assert session.fallback_reason is None
    session.release()
