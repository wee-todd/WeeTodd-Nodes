#!/usr/bin/env python3
"""Convert an official LTX 2.5 component to WeeTodd's directly paged Q8 layout."""

from __future__ import annotations

import argparse
import json

from ltx25_mlx.paged_checkpoint import convert_to_paged_q8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("transformer", "gemma"))
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args()
    manifest = convert_to_paged_q8(
        args.source,
        args.destination,
        kind=args.kind,
        group_size=args.group_size,
        verify_output=not args.skip_hash_verification,
    )
    print(
        json.dumps(
            {
                "destination": str(manifest.root),
                "format": manifest.format,
                "layers": manifest.num_layers,
                "source_tensor_bytes": manifest.source_tensor_bytes,
                "output_tensor_bytes": manifest.output_tensor_bytes,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
