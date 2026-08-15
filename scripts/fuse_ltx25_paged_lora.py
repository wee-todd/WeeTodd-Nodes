#!/usr/bin/env python3
"""Bake one or more LTX 2.5 LoRAs into directly streamable transformer pages."""

from __future__ import annotations

import argparse
import json

from ltx25_mlx.paged_checkpoint import fuse_paged_transformer_loras


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("lora")
    parser.add_argument("destination")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument(
        "--extra-lora",
        action="append",
        default=[],
        help="Additional adapter to fuse from the same original Q8 base. Repeat as needed.",
    )
    parser.add_argument(
        "--extra-strength",
        action="append",
        type=float,
        default=[],
        help="Strength for the matching --extra-lora. Defaults to 1.0.",
    )
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args()
    if len(args.extra_strength) > len(args.extra_lora):
        parser.error("Each --extra-strength requires a matching --extra-lora.")
    extras = [
        (path, args.extra_strength[index] if index < len(args.extra_strength) else 1.0)
        for index, path in enumerate(args.extra_lora)
    ]
    manifest = fuse_paged_transformer_loras(
        args.source,
        args.destination,
        ((args.lora, args.strength), *extras),
        verify_output=not args.skip_hash_verification,
    )
    print(
        json.dumps(
            {
                "destination": str(manifest.root),
                "format": manifest.format,
                "layers": manifest.num_layers,
                "output_tensor_bytes": manifest.output_tensor_bytes,
                "baked_loras": manifest.metadata.get("weetodd_baked_loras", []),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
