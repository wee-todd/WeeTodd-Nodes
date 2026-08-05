"""Build a quantized MiniMax-H3 DiT for publication.

    ./.venv/bin/python scripts/build_quant.py --bits 4 --out models/h3-mlx-4bit

Writes an MLX-native directory: `config.json`, sharded safetensors and a `quant_config.json`
recording exactly which layers were quantized at which width. The video/audio VAEs and the text
encoder are copied or referenced unchanged — only the DiT is quantized, because it is the only
component whose size is worth attacking and the only one whose sensitivity has been characterized.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit, resident_footprint

MAX_SHARD_BYTES = 5 * 1024**3


def save_sharded(model, out_dir: Path, metadata: dict) -> list[str]:
    """Write the parameter tree as sharded safetensors with an index, like the release."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = dict(tree_flatten(model.parameters()))

    shards: list[dict[str, mx.array]] = [{}]
    sizes = [0]
    for key in sorted(tensors):
        value = tensors[key]
        if sizes[-1] and sizes[-1] + value.nbytes > MAX_SHARD_BYTES:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = value
        sizes[-1] += value.nbytes

    total = len(shards)
    weight_map: dict[str, str] = {}
    names = []
    for index, shard in enumerate(shards, start=1):
        name = f"model-{index:05d}-of-{total:05d}.safetensors"
        names.append(name)
        mx.save_safetensors(str(out_dir / name), shard, metadata={"format": "mlx"})
        for key in shard:
            weight_map[key] = name

    with open(out_dir / "model.safetensors.index.json", "w") as fh:
        json.dump({"metadata": {"total_size": sum(sizes), **metadata}, "weight_map": weight_map}, fh, indent=2)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/MiniMax-H3/FL2VA")
    parser.add_argument("--out", required=True,
                        help="output directory; with several --bits it is the parent, "
                             "and each width lands in <out>/MiniMax-H3-MLX-<n>bit")
    parser.add_argument("--bits", type=int, nargs="+", default=[4], choices=[2, 3, 4, 6, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--quantize-adaln", action="store_true",
                        help="also quantize the 13B adaln_proj (off by default; it is dropped at runtime)")
    parser.add_argument("--adaln-bits", type=int, default=8)
    args = parser.parse_args()

    source = Path(args.checkpoint)
    out_root = Path(args.out)

    print(f"loading {source / 'transformer'}", flush=True)
    started = time.perf_counter()
    reference = load_dit(source / "transformer")
    print(f"  loaded in {time.perf_counter() - started:.1f}s", flush=True)

    # Quantization is destructive, so keep the bfloat16 weights and rebuild per width rather than
    # re-reading 62 GB from disk for every one.
    base_weights = dict(tree_flatten(reference.parameters()))
    cfg = reference.config
    del reference

    for bits in args.bits:
        out_dir = out_root / f"MiniMax-H3-MLX-{bits}bit" if len(args.bits) > 1 else out_root
        print(f"\n=== {bits}-bit -> {out_dir} ===", flush=True)

        model = MiniMaxH3DiT(cfg)
        model.update(tree_unflatten(list(base_weights.items())))
        mx.eval(model.parameters())

        config = QuantConfig(
            bits=bits,
            group_size=args.group_size,
            quantize_adaln=args.quantize_adaln,
            adaln_bits=args.adaln_bits,
        )
        print(f"quantizing at {bits}-bit (group {args.group_size})"
              f"{', adaln at %d-bit' % args.adaln_bits if args.quantize_adaln else ', adaln left in bf16'}",
              flush=True)
        summary = quantize_dit(model, config, verbose=True)

        footprint = resident_footprint(model)
        print(f"  on disk {footprint['total_gb']:.1f} GB; resident after the adaln drop "
              f"{footprint['resident_gb']:.1f} GB", flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / "transformer" / "config.json", out_dir / "config.json")

        quant_meta = {
            "bits": bits,
            "group_size": args.group_size,
            "quantize_adaln": args.quantize_adaln,
            "adaln_bits": args.adaln_bits if args.quantize_adaln else None,
            "quantized_layers": {str(k): v for k, v in summary["quantized_layers"].items()},
            "gb_on_disk": round(footprint["total_gb"], 2),
            "gb_resident_after_adaln_drop": round(footprint["resident_gb"], 2),
        }
        with open(out_dir / "quant_config.json", "w") as fh:
            json.dump(quant_meta, fh, indent=2)

        started = time.perf_counter()
        names = save_sharded(model, out_dir, {"quantization": json.dumps(quant_meta)})
        print(f"  wrote {len(names)} shards in {time.perf_counter() - started:.1f}s", flush=True)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
