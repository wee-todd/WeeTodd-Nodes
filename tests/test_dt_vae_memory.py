"""DT decoding must retain source half weights without changing FP32 execution."""

import contextlib
import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402

from minimax_h3_mlx import dt_vae  # noqa: E402
from minimax_h3_mlx.video_vae import ViTBlock  # noqa: E402


def test_dt_video_load_omits_encoder_and_preserves_fp32_output(tmp_path, monkeypatch):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = [ViTBlock(8, 1, 8), ViTBlock(8, 1, 8)]

        def __call__(self, x):
            for block in self.transformer_blocks:
                x = block(x, None)
            return x

    class SmallVAE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = nn.Linear(8, 8)
            self.quant_conv = nn.Linear(8, 8)
            self.decoder = Decoder()
            self.post_quant_conv = nn.Linear(8, 8)

    mx.random.seed(14)
    reference = SmallVAE(None)
    values = [
        (k, np.asarray(mx.random.uniform(-0.2, 0.2, v.shape).astype(mx.float16)))
        for k, v in tree_flatten(reference.parameters())
    ]
    dt_vae._assign(reference, iter(values))
    read_names = []

    def source_values(store, *, decode_only=False):
        for key, value in values:
            if decode_only and key.startswith(("encoder.", "quant_conv.")):
                continue
            read_names.append(key)
            yield key, value

    (tmp_path / "config.json").write_text(
        json.dumps({"latents_mean": [0] * 24, "latents_std": [1] * 24})
    )
    monkeypatch.setattr("minimax_h3_mlx.video_vae.VideoVAE", SmallVAE)
    monkeypatch.setattr(dt_vae, "dt_source", lambda *args: tmp_path / "unused.ckpt")
    monkeypatch.setattr(dt_vae, "DTTensorStore", lambda *args: contextlib.nullcontext(None))
    monkeypatch.setattr(dt_vae, "video_values", source_values)
    candidate = dt_vae.load_dt_video_vae(tmp_path)
    assert not any(k.startswith(("encoder.", "quant_conv.")) for k in read_names)
    assert not hasattr(candidate, "encoder") and not hasattr(candidate, "quant_conv")
    assert all(v.dtype == mx.float16 for _, v in tree_flatten(candidate.parameters()))
    for rows in (3, 7):
        x = mx.random.uniform(-0.5, 0.5, (2, rows, 8))
        expected = reference.decoder(x)
        actual = candidate.decoder(x)
        mx.eval(expected, actual)
        assert actual.dtype == mx.float32
        assert bool(mx.array_equal(actual, expected))
    with pytest.raises(ValueError, match="encoder.*not.*qualified"):
        candidate.encode(mx.zeros((1,)))


def test_decoder_values_never_read_encoder_payloads():
    class Store:
        def read(self, name):
            assert name.startswith("__video_decoder__")
            if name == "__video_decoder__[t-post_quant_conv-0-0]":
                return np.zeros((24 * 24,), np.float16)
            return np.zeros((2048,), np.float16)

    values = list(dt_vae.video_values(Store(), decode_only=True))
    assert len(values) == 9 + 36 * 12
    assert all(k.startswith(("decoder.", "post_quant_conv.")) for k, _ in values)


def test_preflight_counts_compact_decoder_storage_and_promoted_block(tmp_path, monkeypatch):
    import math
    from types import SimpleNamespace

    from minimax_h3_mlx.dt_source import describe_dt_reference, expected_inventory

    records = {
        k: SimpleNamespace(shape=tuple(shape), elements=math.prod(shape), codec=0)
        for k, shape in expected_inventory("video_vae").items()
    }

    class Store:
        def __init__(self, *_):
            self.records = records

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def validate_tensor(self, name, **kwargs):
            return records[name]

    (tmp_path / "config.json").write_text(
        json.dumps({"latents_mean": [0] * 24, "latents_std": [1] * 24})
    )
    (tmp_path / "source.ckpt").touch()
    (tmp_path / "draw_things_source.json").write_text(
        json.dumps(
            {
                "format": "weetodd-h3-dt-source-v1",
                "component": "video_vae",
                "checkpoint": "source.ckpt",
            }
        )
    )
    monkeypatch.setattr("minimax_h3_mlx.dt_tensor_store.DTTensorStore", Store)
    report = describe_dt_reference(tmp_path, "video_vae")
    assert report["tensor_count"] == 657
    assert report["tensor_bytes"] == 4847062192
    assert report["tensor_bytes"] < report["window_bytes"] < 6 * 1024**3
