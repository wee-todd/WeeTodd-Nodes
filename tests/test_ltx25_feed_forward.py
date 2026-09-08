import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx25_mlx.feed_forward import (
    BF16MPPLinear,
    configure_feed_forward_backend,
    feed_forward_runtime_status,
    mpp_bf16_linear,
    mpp_capability,
    reset_feed_forward_runtime_status,
    set_mpp_feed_forward_enabled,
)


def test_feed_forward_runtime_status_can_be_scoped_to_one_generation():
    reset_feed_forward_runtime_status()
    assert feed_forward_runtime_status() == {
        "mpp_calls": 0,
        "bf16_cast_elements": 0,
        "fused_calls": 0,
    }


def test_experimental_backend_wraps_bias_free_video_ff_and_casts_input(monkeypatch):
    monkeypatch.setattr("ltx25_mlx.feed_forward.mpp_capability", lambda: (True, None))
    observed = []

    def project(source, weight):
        observed.append(source.dtype)
        return source @ weight.T

    monkeypatch.setattr("ltx25_mlx.feed_forward.mpp_bf16_linear", project)
    feed_forward = SimpleNamespace(
        proj_in=nn.Linear(8, 16, bias=False),
        proj_out=nn.Linear(16, 8, bias=False),
    )
    feed_forward.proj_in.weight = feed_forward.proj_in.weight.astype(mx.bfloat16)
    feed_forward.proj_out.weight = feed_forward.proj_out.weight.astype(mx.bfloat16)
    model = SimpleNamespace(transformer_blocks=[SimpleNamespace(ff=feed_forward)])
    report = configure_feed_forward_backend(model, "bf16_mpp_experimental")
    assert report.approximate is True
    assert report.wrapped_projections == 2
    assert isinstance(feed_forward.proj_in, BF16MPPLinear)

    result = feed_forward.proj_in(mx.ones((1, 4, 8), dtype=mx.float32))
    mx.eval(result)
    assert observed == [mx.bfloat16]
    assert feed_forward_runtime_status()["mpp_calls"] == 1


def test_reference_backend_does_not_modify_video_ff():
    feed_forward = SimpleNamespace(
        proj_in=nn.Linear(8, 16, bias=False),
        proj_out=nn.Linear(16, 8, bias=False),
    )
    original = feed_forward.proj_in
    model = SimpleNamespace(transformer_blocks=[SimpleNamespace(ff=feed_forward)])
    report = configure_feed_forward_backend(model, "reference_fp32")
    assert report.approximate is False
    assert feed_forward.proj_in is original


def test_compiled_rms_adaln_feed_forward_matches_complete_block_exactly():
    from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

    mx.random.seed(20260814)
    reference = BasicAVTransformerBlock(
        video_dim=8,
        audio_dim=4,
        video_num_heads=1,
        audio_num_heads=1,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=1,
        av_cross_head_dim=4,
        ff_mult=2,
    )
    mx.eval(reference.parameters())
    candidate = copy.deepcopy(reference)
    model = SimpleNamespace(transformer_blocks=[candidate])
    report = configure_feed_forward_backend(model, "mlx_fused_experimental")
    assert report.approximate is False
    assert report.wrapped_projections == 2

    args = (
        mx.random.normal((1, 5, 8)),
        mx.random.normal((1, 3, 4)),
        mx.random.normal((1, 72)),
        mx.random.normal((1, 36)),
        mx.random.normal((1, 16)),
        mx.random.normal((1, 8)),
        mx.random.normal((1, 32)),
        mx.random.normal((1, 16)),
        mx.random.normal((1, 8)),
        mx.random.normal((1, 4)),
    )
    expected = reference(*args)
    actual = candidate(*args)
    mx.eval(*expected, *actual)
    assert mx.array_equal(actual[0], expected[0])
    assert mx.array_equal(actual[1], expected[1])
    assert feed_forward_runtime_status()["fused_calls"] == 2


def test_compiled_backend_can_bypass_and_reenable_without_reloading():
    from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

    block = BasicAVTransformerBlock(
        video_dim=8,
        audio_dim=4,
        video_num_heads=1,
        audio_num_heads=1,
        video_head_dim=8,
        audio_head_dim=4,
        av_cross_num_heads=1,
        av_cross_head_dim=4,
        ff_mult=2,
    )
    original_video_ff = block.ff
    original_audio_ff = block.audio_ff
    model = SimpleNamespace(transformer_blocks=[block])
    configure_feed_forward_backend(model, "mlx_fused_experimental")

    assert set_mpp_feed_forward_enabled(model, True) == 2
    assert block.ff is not original_video_ff
    assert block.audio_ff is not original_audio_ff
    assert set_mpp_feed_forward_enabled(model, False) == 2
    assert block.ff is original_video_ff
    assert block.audio_ff is original_audio_ff
    assert set_mpp_feed_forward_enabled(model, True) == 2
    assert block.ff is not original_video_ff
    assert block.audio_ff is not original_audio_ff


def test_experimental_backend_can_bypass_and_reenable_without_reloading(monkeypatch):
    monkeypatch.setattr("ltx25_mlx.feed_forward.mpp_capability", lambda: (True, None))
    calls = []

    def project(source, weight):
        calls.append(source.dtype)
        return source @ weight.T

    monkeypatch.setattr("ltx25_mlx.feed_forward.mpp_bf16_linear", project)
    feed_forward = SimpleNamespace(
        proj_in=nn.Linear(8, 16, bias=False),
        proj_out=nn.Linear(16, 8, bias=False),
    )
    feed_forward.proj_in.weight = feed_forward.proj_in.weight.astype(mx.bfloat16)
    feed_forward.proj_out.weight = feed_forward.proj_out.weight.astype(mx.bfloat16)
    model = SimpleNamespace(transformer_blocks=[SimpleNamespace(ff=feed_forward)])
    configure_feed_forward_backend(model, "bf16_mpp_experimental")
    source = mx.ones((1, 4, 8), dtype=mx.float32)

    assert set_mpp_feed_forward_enabled(model, False) == 2
    reference = feed_forward.proj_in(source)
    mx.eval(reference)
    assert calls == []
    assert reference.dtype == mx.float32

    assert set_mpp_feed_forward_enabled(model, True) == 2
    approximate = feed_forward.proj_in(source)
    mx.eval(approximate)
    assert calls == [mx.bfloat16]


@pytest.mark.skipif(not mpp_capability()[0], reason="Metal 4 MPP runtime is unavailable")
def test_mpp_projection_handles_partial_row_tile():
    source = mx.random.normal((1, 33, 64)).astype(mx.bfloat16)
    weight = mx.random.normal((128, 64)).astype(mx.bfloat16)
    expected = source @ weight.T
    actual = mpp_bf16_linear(source, weight)
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected)
