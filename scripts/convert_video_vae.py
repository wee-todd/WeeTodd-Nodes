#!/usr/bin/env python3
"""Convert a MiniMax H3 video VAE to its versioned MLX-native tensor layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minimax_h3_mlx.video_vae_checkpoint import convert_video_vae_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a released directory or self-describing MiniMax H3 video VAE safetensors "
            "file to a versioned MLX-native ODHWI artifact. No model is downloaded."
        )
    )
    parser.add_argument("source", help="video_vae directory or self-describing safetensors file")
    parser.add_argument("output", help="new MLX-native .safetensors artifact")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output after conversion"
    )
    args = parser.parse_args()

    result = convert_video_vae_checkpoint(args.source, args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
