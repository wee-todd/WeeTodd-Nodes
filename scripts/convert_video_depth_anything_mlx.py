#!/usr/bin/env python3
"""Convert the official Video Depth Anything Small PyTorch checkpoint for MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlx.utils import tree_flatten  # noqa: E402

from mlx_preprocessors.video_depth import VideoDepthAnythingSmall, _convert_weight  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required only for this one-time conversion. Run the script with the "
            "active ComfyUI Python interpreter."
        ) from error

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise SystemExit("The source checkpoint does not contain a tensor mapping.")
    expected = dict(tree_flatten(VideoDepthAnythingSmall().parameters()))
    missing = sorted(set(expected) - set(source))
    extra = sorted(set(source) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"Checkpoint mismatch: {len(missing)} missing and {len(extra)} extra tensors."
        )
    converted = {
        name: _convert_weight(name, source[name].detach().cpu().numpy(), target.shape)
        for name, target in expected.items()
    }
    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        args.destination,
        metadata={
            "format": "weetodd-mlx-video-depth-anything-v1",
            "source_sha256": source_digest,
            "architecture": "video_depth_anything_vits",
            "license": "Apache-2.0",
        },
    )
    print(
        json.dumps(
            {
                "destination": str(args.destination),
                "source_sha256": source_digest,
                "tensor_count": len(converted),
                "bytes": args.destination.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
