"""Bounded MLX 0.32.2 Conv3D execution for the LTX convolutional VAE."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version

import mlx.core as mx
import mlx.nn as nn

_DEFAULT_WINOGRAD_WORKING_SET_BYTES = 1024**3
_DEFAULT_DEPTH_WINDOW_ELEMENTS = 64 << 20
_WORKSPACE_LOCK = threading.RLock()


def _triple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (value, value, value) if isinstance(value, int) else tuple(value)


class StagedSmallDepthConv3d(nn.Module):
    """Run an eligible depth kernel as evaluated 2D slices with bounded residency."""

    def __init__(
        self, source: nn.Conv3d, *, max_window_elements: int = _DEFAULT_DEPTH_WINDOW_ELEMENTS
    ) -> None:
        super().__init__()
        if max_window_elements < 1:
            raise ValueError("Conv3D depth window budget must be positive.")
        self.max_window_elements = int(max_window_elements)
        self.weight = source.weight
        if "bias" in source:
            self.bias = source.bias
        self.stride = _triple(source.stride)
        self.padding = _triple(source.padding)
        self.dilation = _triple(source.dilation)

    @classmethod
    def supports(cls, source: nn.Conv3d) -> bool:
        stride = _triple(source.stride)
        padding = _triple(source.padding)
        dilation = _triple(source.dilation)
        out_channels, depth, _height, _width, in_channels = source.weight.shape
        return (
            depth <= 7
            and stride[0] == 1
            and padding[0] == 0
            and dilation == (1, 1, 1)
            and in_channels % 16 == 0
            and (out_channels <= 16 or out_channels % 16 == 0)
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch, depth, height, width, channels = x.shape
        out_channels, kernel_depth, _kernel_height, _kernel_width, _ = self.weight.shape
        out_depth = depth - kernel_depth + 1
        if out_depth <= 0:
            raise ValueError(
                f"Conv3D input depth {depth} is smaller than kernel depth {kernel_depth}."
            )

        out_h = (height + 2 * self.padding[1] - _kernel_height) // self.stride[1] + 1
        out_w = (width + 2 * self.padding[2] - _kernel_width) // self.stride[2] + 1
        elements_per_frame = batch * max(
            height * width * channels, out_h * out_w * out_channels
        )
        window_depth = max(1, self.max_window_elements // elements_per_frame)
        windows = []
        for start in range(0, out_depth, window_depth):
            count = min(window_depth, out_depth - start)
            accumulated = None
            # Each output window reads kernel_depth-1 overlapping input frames.
            # No output overlap, averaging, or changed temporal padding.
            for depth_index in range(kernel_depth):
                input_2d = x[:, start + depth_index : start + depth_index + count].reshape(
                    batch * count, height, width, channels
                )
                contribution = mx.conv2d(
                    input_2d, self.weight[:, depth_index], self.stride[1:],
                    self.padding[1:], self.dilation[1:],
                ).reshape(batch, count, out_h, out_w, out_channels)
                mx.eval(contribution)
                accumulated = contribution if accumulated is None else accumulated + contribution
                mx.eval(accumulated)
            if "bias" in self:
                accumulated = accumulated + self.bias
            mx.eval(accumulated)
            windows.append(accumulated)
        return windows[0] if len(windows) == 1 else mx.concatenate(windows, axis=1)


@dataclass(frozen=True)
class ConvVAEAccelerationReport:
    backend: str
    mlx_version: str
    replaced_convolutions: int
    winograd_working_set_bytes: int | None
    depth_window_elements: int = _DEFAULT_DEPTH_WINDOW_ELEMENTS

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def install_staged_conv3d(decoder: nn.Module) -> ConvVAEAccelerationReport:
    """Replace eligible Conv3D leaves in one loaded decoder only."""
    mlx_version = version("mlx")
    replaced = 0
    for _name, module in decoder.named_modules():
        source = getattr(module, "conv", None)
        if isinstance(source, nn.Conv3d) and StagedSmallDepthConv3d.supports(source):
            module.conv = StagedSmallDepthConv3d(source)
            replaced += 1

    return ConvVAEAccelerationReport(
        "staged_depth_conv2d",
        mlx_version,
        replaced,
        _DEFAULT_WINOGRAD_WORKING_SET_BYTES if replaced else None,
    )


@contextmanager
def bounded_conv_workspace(report: ConvVAEAccelerationReport | None) -> Iterator[None]:
    """Scope MLX's process-global Winograd workspace cap to one synchronous decode."""
    cap = report.winograd_working_set_bytes if report is not None else None
    if cap is None:
        yield
        return

    key = "MLX_CONV_WINOGRAD_WORKING_SET"
    with _WORKSPACE_LOCK:
        previous = os.environ.get(key)
        os.environ[key] = str(cap)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


__all__ = [
    "ConvVAEAccelerationReport",
    "StagedSmallDepthConv3d",
    "bounded_conv_workspace",
    "install_staged_conv3d",
]
