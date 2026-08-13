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
    set_mpp_feed_forward_enabled,
)


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
