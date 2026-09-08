"""Convert the official dense FastH3 checkpoint directly into WeeTodd MLX pages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minimax_h3_mlx.fastvideo_checkpoint import convert_fastvideo_fasth3_to_paged
from minimax_h3_mlx.quantize import QuantConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="downloaded FastH3 transformer directory")
    parser.add_argument("--out", required=True, help="new WeeTodd paged checkpoint directory")
    parser.add_argument("--bits", type=int, choices=(4, 5, 6, 8), default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--quantize-adaln", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaln-bits", type=int, choices=(6, 8), default=8)
    args = parser.parse_args()
    result = convert_fastvideo_fasth3_to_paged(
        args.source,
        args.out,
        quant_config=QuantConfig(
            bits=args.bits,
            group_size=args.group_size,
            quantize_adaln=args.quantize_adaln,
            adaln_bits=args.adaln_bits,
        ),
    )
    print(json.dumps({
        "root": str(result.root),
        "blocks": result.num_blocks,
        "source_tensor_bytes": result.source_tensor_bytes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
