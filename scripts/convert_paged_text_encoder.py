#!/usr/bin/env python3
"""Convert the H3 Qwen3-VL text subset into sequential MLX layer pages."""

from __future__ import annotations

import argparse
import json

from minimax_h3_mlx.paged_text_encoder import convert_to_paged_text_encoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--architecture-config")
    parser.add_argument("--skip-output-hashes", action="store_true")
    args = parser.parse_args()
    manifest = convert_to_paged_text_encoder(
        args.source,
        args.destination,
        num_layers=args.layers,
        verify_output=not args.skip_output_hashes,
        architecture_config=args.architecture_config,
    )
    print(
        json.dumps(
            {
                "destination": str(manifest.root),
                "layers": manifest.num_blocks,
                "source_tensor_bytes": manifest.source_tensor_bytes,
                "fixed_bytes": manifest.fixed.tensor_bytes,
                "largest_layer_bytes": max(page.tensor_bytes for page in manifest.layers),
                "skipped_visual_bytes": manifest.skipped_visual_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
