from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from ltx25_mlx.conv_vae_acceleration import (
    ConvVAEAccelerationReport,
    StagedSmallDepthConv3d,
    bounded_conv_workspace,
    install_staged_conv3d,
)


def test_staged_small_depth_conv3d_matches_reference_with_bounded_error() -> None:
    mx.random.seed(7)
    source = nn.Conv3d(16, 32, 3, padding=0, bias=True)
    staged = StagedSmallDepthConv3d(source)
    values = mx.random.normal((1, 5, 8, 8, 16)).astype(mx.bfloat16)

    reference = source(values)
    candidate = staged(values)
    mx.eval(reference, candidate)
    reference_np = np.asarray(reference.astype(mx.float32))
    candidate_np = np.asarray(candidate.astype(mx.float32))
    relative_l2 = np.linalg.norm(candidate_np - reference_np) / np.linalg.norm(reference_np)

    assert relative_l2 < 0.02


def test_install_staged_conv3d_changes_only_eligible_conv_leaves() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(16, 32, 3, padding=0)
            self.unsupported = nn.Conv3d(3, 5, 3, padding=0)

    block = Block()
    unsupported = block.unsupported
    report = install_staged_conv3d(block)

    assert isinstance(block.conv, StagedSmallDepthConv3d)
    assert block.unsupported is unsupported
    assert report.backend == "staged_depth_conv2d"
    assert report.replaced_convolutions == 1
    assert report.winograd_working_set_bytes == 1024**3


def test_bounded_conv_workspace_restores_process_environment(monkeypatch) -> None:
    monkeypatch.setenv("MLX_CONV_WINOGRAD_WORKING_SET", "123")
    report = ConvVAEAccelerationReport("staged_depth_conv2d", "test", 1, 456)

    with pytest.raises(RuntimeError, match="synthetic"):
        with bounded_conv_workspace(report):
            assert os.environ["MLX_CONV_WINOGRAD_WORKING_SET"] == "456"
            raise RuntimeError("synthetic")

    assert os.environ["MLX_CONV_WINOGRAD_WORKING_SET"] == "123"


def test_bounded_conv_workspace_accepts_unconfigured_fake_decoder(monkeypatch) -> None:
    monkeypatch.delenv("MLX_CONV_WINOGRAD_WORKING_SET", raising=False)

    with bounded_conv_workspace(None):
        assert "MLX_CONV_WINOGRAD_WORKING_SET" not in os.environ
