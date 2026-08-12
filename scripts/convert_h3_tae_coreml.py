#!/usr/bin/env python3
"""Convert the MiniMax H3 tiny preview decoder to a fixed-shape Core ML model.

This is an optional, one-time conversion utility. The generated model is intended for
CPU-and-Neural-Engine execution while the H3 transformer remains on MLX/Metal.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="H3 TAE safetensors file")
    parser.add_argument("output", type=Path, help="Destination .mlpackage")
    parser.add_argument(
        "--edge",
        type=int,
        choices=(256, 320, 384),
        default=384,
        help="Fixed square preview edge",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.output.suffix != ".mlpackage":
        raise ValueError("Core ML output must use the .mlpackage suffix.")

    try:
        import coremltools as ct
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError(
            "Conversion requires coremltools, torch, numpy, and safetensors. "
            "Install the optional preview conversion dependencies in a compatible arm64 venv."
        ) from exc

    class Clamp(nn.Module):
        def forward(self, values):
            return torch.tanh(values / 3.0) * 3.0

    latent_edge = args.edge // 16

    class MemoryBlock(nn.Module):
        def __init__(self, channels: int, frames: int, edge: int):
            super().__init__()
            self.channels = channels
            self.frames = frames
            self.edge = edge
            self.conv = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

        def forward(self, values):
            past = torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)
            merged = torch.cat([values, past], dim=2).reshape(
                self.frames, self.channels * 2, self.edge, self.edge
            )
            result = self.conv(merged).reshape(1, self.frames, self.channels, self.edge, self.edge)
            return torch.relu(result + values)

    class TemporalGrow(nn.Module):
        def __init__(self, channels: int, stride: int, frames: int, edge: int):
            super().__init__()
            self.channels = channels
            self.stride = stride
            self.frames = frames
            self.edge = edge
            self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

        def forward(self, values):
            result = self.conv(values.reshape(self.frames, self.channels, self.edge, self.edge))
            return result.reshape(
                1,
                self.frames * self.stride,
                self.channels,
                self.edge,
                self.edge,
            )

    class SpatialGrow(nn.Module):
        def __init__(self, channels: int, frames: int, edge: int):
            super().__init__()
            self.channels = channels
            self.frames = frames
            self.edge = edge

        def forward(self, values):
            result = F.interpolate(
                values.reshape(self.frames, self.channels, self.edge, self.edge),
                scale_factor=2.0,
                mode="nearest",
            )
            return result.reshape(1, self.frames, self.channels, self.edge * 2, self.edge * 2)

    class H3PreviewDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Indices deliberately match the public taeh3 safetensors tensor names.
            self.decoder = nn.ModuleList(
                [
                    Clamp(),
                    nn.Conv2d(24, 256, 3, padding=1),
                    nn.ReLU(),
                    MemoryBlock(256, 10, latent_edge),
                    MemoryBlock(256, 10, latent_edge),
                    MemoryBlock(256, 10, latent_edge),
                    SpatialGrow(256, 10, latent_edge),
                    TemporalGrow(256, 1, 10, latent_edge * 2),
                    nn.Conv2d(256, 128, 3, padding=1, bias=False),
                    MemoryBlock(128, 10, latent_edge * 2),
                    MemoryBlock(128, 10, latent_edge * 2),
                    MemoryBlock(128, 10, latent_edge * 2),
                    SpatialGrow(128, 10, latent_edge * 2),
                    TemporalGrow(128, 2, 10, latent_edge * 4),
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    MemoryBlock(64, 20, latent_edge * 4),
                    MemoryBlock(64, 20, latent_edge * 4),
                    MemoryBlock(64, 20, latent_edge * 4),
                    SpatialGrow(64, 20, latent_edge * 4),
                    TemporalGrow(64, 2, 20, latent_edge * 8),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(64, 12, 3, padding=1),
                ]
            )

        @staticmethod
        def _frame_conv(module, values, frames, input_channels, output_channels, edge):
            result = module(values.reshape(frames, input_channels, edge, edge))
            return result.reshape(1, frames, output_channels, edge, edge)

        def forward(self, latents):
            values = latents.permute(0, 2, 1, 3, 4)
            for index, module in enumerate(self.decoder):
                if isinstance(module, nn.Conv2d):
                    if index == 1:
                        values = self._frame_conv(module, values, 10, 24, 256, latent_edge)
                    elif index == 8:
                        values = self._frame_conv(module, values, 10, 256, 128, latent_edge * 2)
                    elif index == 14:
                        values = self._frame_conv(module, values, 20, 128, 64, latent_edge * 4)
                    elif index in (20, 22):
                        output_channels = 12 if index == 22 else 64
                        values = self._frame_conv(
                            module,
                            values,
                            40,
                            64,
                            output_channels,
                            latent_edge * 8,
                        )
                else:
                    values = module(values)

            values = F.pixel_shuffle(
                values.reshape(40, 12, latent_edge * 8, latent_edge * 8), 2
            ).reshape(1, 40, 3, args.edge, args.edge)
            # Ten latent frames produce two twenty-frame chunks. Drop each three-frame causal
            # prefix and the final twelve frames corresponding to H3's latent tail padding.
            values = torch.cat([values[:, 3:20], values[:, 23:40]], dim=1)
            values = values[:, :22].clamp(0.0, 1.0)
            return values.permute(0, 1, 3, 4, 2)

    model = H3PreviewDecoder().eval()
    tensors = load_file(str(args.source), device="cpu")
    decoder_tensors = {
        name: value.float() for name, value in tensors.items() if name.startswith("decoder.")
    }
    missing, unexpected = model.load_state_dict(decoder_tensors, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"H3 TAE tensor mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    example = torch.zeros((1, 24, 10, latent_edge, latent_edge), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=True)
        reference = model(example).numpy()

    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="latents",
                shape=example.shape,
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name="frames", dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))

    # Load and execute once so conversion does not report success for a model that Core ML rejects.
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.CPU_AND_NE)
    prediction = runtime.predict({"latents": example.numpy().astype(np.float16)})["frames"]
    if prediction.shape != reference.shape:
        raise RuntimeError(
            f"Core ML output shape {prediction.shape} did not match reference {reference.shape}."
        )
    error = float(np.max(np.abs(prediction.astype(np.float32) - reference)))
    print(f"saved={args.output}")
    print(f"edge={args.edge} output_shape={prediction.shape} max_abs_error={error:.7f}")


if __name__ == "__main__":
    main()
