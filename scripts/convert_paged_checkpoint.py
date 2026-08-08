#!/usr/bin/env python3
"""Convert an H3 transformer into a block-aligned paged checkpoint."""

from __future__ import annotations

import argparse
import json

from minimax_h3_mlx.paged_checkpoint import convert_to_paged_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--skip-output-hashes", action="store_true")
    args = parser.parse_args()
    manifest = convert_to_paged_checkpoint(
        args.source,
        args.destination,
        verify_output=not args.skip_output_hashes,
    )
    print(
        json.dumps(
            {
                "destination": str(manifest.root),
                "blocks": manifest.num_blocks,
                "source_tensor_bytes": manifest.source_tensor_bytes,
                "largest_page_bytes": max(
                    record.tensor_bytes for record in (manifest.fixed, *manifest.blocks)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
