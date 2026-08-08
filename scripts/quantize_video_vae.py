#!/usr/bin/env python3
"""Convert a MiniMax H3 video VAE to a directly loadable MLX affine-Q8 artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minimax_h3_mlx.video_vae_checkpoint import quantize_video_vae_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a released directory or self-describing MiniMax H3 video VAE to an "
            "MLX-native affine-Q8 checkpoint. No model is downloaded."
        )
    )
    parser.add_argument("source", help="video_vae directory or native safetensors file")
    parser.add_argument("output", help="new affine-Q8 .safetensors artifact")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output after conversion"
    )
    args = parser.parse_args()
    result = quantize_video_vae_checkpoint(
        args.source,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
