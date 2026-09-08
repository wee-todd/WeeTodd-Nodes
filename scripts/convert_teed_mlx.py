#!/usr/bin/env python3
"""Convert the official TEED checkpoint into runtime-only MLX safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

TRANSPOSED = {
    "up_block_1.features.2.weight",
    "up_block_2.features.2.weight",
    "up_block_3.features.2.weight",
    "up_block_3.features.5.weight",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    import torch

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    if "state_dict" in source:
        source = source["state_dict"]
    weights = {}
    for name, tensor in source.items():
        value = tensor.detach().cpu().float().numpy()
        if value.ndim == 4:
            axes = (1, 2, 3, 0) if name in TRANSPOSED else (0, 2, 3, 1)
            value = value.transpose(axes)
        weights[name] = np.ascontiguousarray(value)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        weights,
        args.destination,
        metadata={"architecture": "teed", "source": args.source.name},
    )
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "tensors": len(weights),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
