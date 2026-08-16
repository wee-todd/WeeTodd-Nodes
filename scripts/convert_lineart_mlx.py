#!/usr/bin/env python3
"""Convert an official ControlNet realistic line-art checkpoint for MLX loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

SOURCE_KEYS = {
    "input_conv": "model0.1",
    "down_convs.0": "model1.0",
    "down_convs.1": "model1.3",
    "residuals.0.conv1": "model2.0.conv_block.1",
    "residuals.0.conv2": "model2.0.conv_block.5",
    "residuals.1.conv1": "model2.1.conv_block.1",
    "residuals.1.conv2": "model2.1.conv_block.5",
    "residuals.2.conv1": "model2.2.conv_block.1",
    "residuals.2.conv2": "model2.2.conv_block.5",
    "up_convs.0": "model3.0",
    "up_convs.1": "model3.3",
    "output_conv": "model4.1",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Conversion requires PyTorch, but runtime inference does not.") from error
    state = torch.load(arguments.source, map_location="cpu", weights_only=True)
    output = {}
    for target, source in SOURCE_KEYS.items():
        weight = state[f"{source}.weight"].detach().cpu().float().numpy()
        if target.startswith("up_convs"):
            weight = np.transpose(weight, (1, 2, 3, 0))
        else:
            weight = np.transpose(weight, (0, 2, 3, 1))
        output[f"{target}.weight"] = np.ascontiguousarray(weight)
        output[f"{target}.bias"] = np.ascontiguousarray(
            state[f"{source}.bias"].detach().cpu().float().numpy()
        )
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, arguments.destination)
    print(f"Wrote {arguments.destination} with {len(output)} tensors")


if __name__ == "__main__":
    main()
