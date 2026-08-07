#!/usr/bin/env python3
"""Build the accepted experimental H3 mixed-precision checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimax_h3_mlx.mixed_checkpoint import (
    DEFAULT_MAX_SHARD_BYTES,
    Q8_CONSERVATIVE_PROFILE,
    Q8_PROFILE_NAMES,
    convert_mixed_checkpoint,
    named_q8_recipe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="BF16 transformer file or directory")
    parser.add_argument("output", type=Path, help="new mixed-precision checkpoint directory")
    parser.add_argument(
        "--profile",
        choices=Q8_PROFILE_NAMES,
        default=Q8_CONSERVATIVE_PROFILE,
        help="validated mixed-precision recipe",
    )
    parser.add_argument(
        "--max-shard-mib",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES // 1024**2,
        help="maximum buffered output shard size in MiB",
    )
    args = parser.parse_args()
    if args.max_shard_mib < 1:
        parser.error("--max-shard-mib must be positive")
    result = convert_mixed_checkpoint(
        args.source,
        args.output,
        named_q8_recipe(args.profile),
        max_shard_bytes=args.max_shard_mib * 1024**2,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
