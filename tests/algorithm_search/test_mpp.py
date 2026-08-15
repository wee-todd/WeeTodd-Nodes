from __future__ import annotations

import platform
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from minimax_h3_mlx.projection import (
    MPPLinear,
    MPPTile,
    configure_projection_backend,
    mpp_bf16_linear,
    mpp_runtime_status,
    reset_mpp_runtime_status,
)


def _has_mpp_runtime() -> bool:
    release = platform.mac_ver()[0]
    return sys.platform == "darwin" and bool(release) and int(release.split(".", 1)[0]) >= 26


def test_mpp_projection_validates_contract_before_dispatch() -> None:
    weight = mx.ones((8, 16), dtype=mx.bfloat16)

    with pytest.raises(TypeError, match="requires BF16"):
        mpp_bf16_linear(mx.ones((2, 16)), weight)
    with pytest.raises(ValueError, match="input width"):
        mpp_bf16_linear(mx.ones((2, 15), dtype=mx.bfloat16), weight)
    with pytest.raises(ValueError, match="must be positive"):
        MPPTile(rows=0)


def test_mpp_backend_wraps_only_eligible_bf16_core_projections(monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_capability", lambda: (True, None)
    )

    def linear(input_dims: int, output_dims: int, dtype) -> nn.Linear:
        layer = nn.Linear(input_dims, output_dims, bias=False)
        layer.weight = layer.weight.astype(dtype)
        return layer

    eligible = SimpleNamespace(
        attn=SimpleNamespace(
            qkv_proj=linear(16, 48, mx.bfloat16),
            out_proj=linear(16, 16, mx.bfloat16),
        ),
        mlp=SimpleNamespace(
            fc1=linear(16, 32, mx.bfloat16),
            fc2=linear(16, 16, mx.bfloat16),
        ),
    )
    mixed = SimpleNamespace(
        attn=SimpleNamespace(
            qkv_proj=linear(16, 48, mx.float32),
            out_proj=linear(16, 16, mx.bfloat16),
        ),
        mlp=SimpleNamespace(
            fc1=linear(16, 32, mx.bfloat16),
            fc2=linear(16, 16, mx.bfloat16),
        ),
    )
    dit = SimpleNamespace(blocks=[eligible, mixed])

    report = configure_projection_backend(dit, "mpp_experimental")

    assert report.resolved == "mpp_experimental"
    assert report.wrapped_projections == 7
    assert report.skipped_projections == 1
    assert isinstance(eligible.attn.qkv_proj, MPPLinear)
    assert isinstance(mixed.attn.qkv_proj, nn.Linear)


def test_automatic_backend_uses_mpp_when_capable(monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_capability", lambda: (True, None)
    )
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_auto_capability", lambda: (True, None)
    )
    layer = nn.Linear(16, 48, bias=False)
    layer.weight = layer.weight.astype(mx.bfloat16)
    block = SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=layer, out_proj=nn.Identity()),
        mlp=SimpleNamespace(fc1=nn.Identity(), fc2=nn.Identity()),
    )

    report = configure_projection_backend(SimpleNamespace(blocks=[block]), "auto")

    assert report.requested == "auto"
    assert report.resolved == "mpp_experimental"
    assert report.wrapped_projections == 1
    assert report.skipped_projections == 3
    assert isinstance(block.attn.qkv_proj, MPPLinear)


def test_mpp_backend_is_deferred_to_paged_block_materialization(monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_capability", lambda: (True, None)
    )
    paged = SimpleNamespace(projection_backend="mlx")

    report = configure_projection_backend(
        SimpleNamespace(blocks=[], paged_blocks=paged), "mpp_experimental"
    )

    assert report.resolved == "mpp_experimental"
    assert "paged block window" in report.reason
    assert paged.projection_backend == "mpp_experimental"


def test_automatic_backend_falls_back_when_mpp_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_capability",
        lambda: (False, "test runtime has no MPP"),
    )

    report = configure_projection_backend(SimpleNamespace(blocks=[]), "auto")

    assert report.requested == "auto"
    assert report.resolved == "mlx"
    assert report.wrapped_projections == 0
    assert report.reason == "test runtime has no MPP"


def test_automatic_backend_requires_a_measured_architecture(monkeypatch) -> None:
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_capability", lambda: (True, None)
    )
    monkeypatch.setattr(
        "minimax_h3_mlx.projection.mpp_auto_capability",
        lambda: (False, "unmeasured test architecture"),
    )

    report = configure_projection_backend(SimpleNamespace(blocks=[]), "auto")

    assert report.resolved == "mlx"
    assert report.reason == "unmeasured test architecture"


def test_mpp_linear_failure_returns_one_standard_mlx_projection(monkeypatch) -> None:
    reset_mpp_runtime_status()

    class CountingLinear(nn.Linear):
        def __init__(self) -> None:
            super().__init__(128, 256, bias=False)
            self.weight = self.weight.astype(mx.bfloat16)
            self.calls = 0

        def __call__(self, source):
            self.calls += 1
            return super().__call__(source)

    base = CountingLinear()
    layer = MPPLinear(base)
    source = mx.ones((2, 64, 128), dtype=mx.bfloat16)

    def fail(*args, **kwargs):
        raise RuntimeError("forced test failure")

    monkeypatch.setattr("minimax_h3_mlx.projection.mpp_bf16_linear", fail)
    actual = layer(source)
    expected = source @ base.weight.T
    mx.eval(actual, expected)

    assert base.calls == 1
    assert bool(mx.array_equal(actual, expected))
    assert mpp_runtime_status() == {
        "verified_signatures": 0,
        "fallback_signatures": 1,
        "fallback_reasons": ["RuntimeError"],
    }


@pytest.mark.skipif(not _has_mpp_runtime(), reason="Metal 4 MPP runtime is unavailable")
def test_mpp_projection_matches_mlx_bf16() -> None:
    mx.random.seed(20260808)
    source = mx.random.uniform(low=-0.125, high=0.125, shape=(2, 64, 128)).astype(
        mx.bfloat16
    )
    weight = mx.random.uniform(low=-0.125, high=0.125, shape=(256, 128)).astype(
        mx.bfloat16
    )

    actual = mpp_bf16_linear(source, weight)
    expected = source @ weight.T
    mx.eval(actual, expected)

    assert actual.shape == (2, 64, 256)
    assert bool(mx.array_equal(actual, expected))


@pytest.mark.skipif(not _has_mpp_runtime(), reason="Metal 4 MPP runtime is unavailable")
def test_mpp_linear_verifies_once_and_returns_exact_output() -> None:
    reset_mpp_runtime_status()
    base = nn.Linear(128, 256, bias=False)
    base.weight = mx.random.uniform(low=-0.125, high=0.125, shape=(256, 128)).astype(
        mx.bfloat16
    )
    layer = MPPLinear(base)
    source = mx.random.uniform(low=-0.125, high=0.125, shape=(2, 64, 128)).astype(
        mx.bfloat16
    )

    first = layer(source)
    second = layer(source)
    expected = base(source)
    mx.eval(first, second, expected)

    assert bool(mx.array_equal(first, expected))
    assert bool(mx.array_equal(second, expected))
    assert mpp_runtime_status() == {
        "verified_signatures": 1,
        "fallback_signatures": 0,
        "fallback_reasons": [],
    }
