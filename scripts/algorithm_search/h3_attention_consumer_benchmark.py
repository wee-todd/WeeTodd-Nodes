#!/usr/bin/env python3
"""Benchmark native MLX against direct-token-major H3 attention output."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.nax_attention import (
    NAXAttentionTile,
    nax_attention,
    nax_attention_available,
)
from minimax_h3_mlx.algorithm_search.steel_attention import (
    SteelAttentionTile,
    steel_attention,
)


def _expand(value: mx.array, *, heads: int, rows: int) -> mx.array:
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 128:
        raise ValueError("QKV capture tensors must have shape [1, heads, rows, 128]")
    if heads % value.shape[1]:
        raise ValueError("requested heads must be divisible by captured heads")
    repeats = math.ceil(rows / value.shape[2])
    expanded = mx.tile(value, (1, heads // value.shape[1], repeats, 1))
    return expanded[:, :, :rows, :]


def _measure(operation, *, warmups: int, repetitions: int) -> tuple[float, list[float]]:
    for _ in range(warmups):
        mx.eval(operation())
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(operation())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("qkv_capture", type=Path)
    parser.add_argument("block_page", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, nargs="+", default=[9477, 25138])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.qkv_capture.is_file():
        parser.error(f"QKV capture is not a file: {args.qkv_capture}")
    if not args.block_page.is_file():
        parser.error(f"block page is not a file: {args.block_page}")
    if min(args.rows) < 1 or min(args.heads, args.warmups, args.repetitions) < 1:
        parser.error("heads, rows, warmups, and repetitions must be positive")

    capture = mx.load(str(args.qkv_capture))
    required = {"tensor_0", "tensor_1", "tensor_2"}
    if not required.issubset(capture):
        parser.error("QKV capture must contain tensor_0, tensor_1, and tensor_2")
    page = mx.load(str(args.block_page))
    weight_key = f"blocks.{args.block}.attn.out_proj.weight"
    if weight_key not in page:
        parser.error(f"block page does not contain {weight_key}")
    weight = page[weight_key]
    if weight.dtype != mx.bfloat16 or weight.shape[1] != args.heads * 128:
        parser.error("attention output projection is not compatible BF16 H3 weight")

    backend = "nax_64x32" if nax_attention_available() else "steel_32x16"
    records = []
    for rows in args.rows:
        query = _expand(capture["tensor_0"], heads=args.heads, rows=rows)
        key = _expand(capture["tensor_1"], heads=args.heads, rows=rows)
        value = _expand(capture["tensor_2"], heads=args.heads, rows=rows)
        mx.eval(query, key, value, weight)
        scale = 128**-0.5

        def native(query=query, key=key, value=value, scale=scale, rows=rows):
            attended = mx.fast.scaled_dot_product_attention(
                query, key, value, scale=scale
            )
            packed = attended.transpose(0, 2, 1, 3).reshape(
                1, rows, args.heads * 128
            )
            return packed @ weight.T

        def candidate(query=query, key=key, value=value, scale=scale, rows=rows):
            if backend.startswith("nax"):
                attended = nax_attention(
                    query, key, value, scale=scale, tile=NAXAttentionTile()
                )
            else:
                attended = steel_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    tile=SteelAttentionTile(32, 16, 4),
                )
            return attended.reshape(1, rows, args.heads * 128) @ weight.T

        reference = native()
        actual = candidate()
        mx.eval(reference, actual)
        exact = bool(mx.array_equal(reference, actual))
        baseline_median, baseline_samples = _measure(
            native, warmups=args.warmups, repetitions=args.repetitions
        )
        candidate_median, candidate_samples = _measure(
            candidate, warmups=args.warmups, repetitions=args.repetitions
        )
        records.append(
            {
                "rows": rows,
                "bit_exact": exact,
                "native_median_seconds": baseline_median,
                "native_samples_seconds": baseline_samples,
                "candidate_median_seconds": candidate_median,
                "candidate_samples_seconds": candidate_samples,
                "speedup": baseline_median / candidate_median,
                "runtime_reduction_percent": (
                    1.0 - candidate_median / baseline_median
                )
                * 100.0,
            }
        )
        del reference, actual, query, key, value
        gc.collect()
        mx.clear_cache()

    payload = {
        "operator": "h3_attention_plus_output_projection",
        "candidate_backend": backend,
        "mlx_version": version("mlx"),
        "device": mx.device_info(),
        "qkv_capture": args.qkv_capture.name,
        "block_page": args.block_page.name,
        "block": args.block,
        "heads": args.heads,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cases": records,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
