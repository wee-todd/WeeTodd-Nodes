#!/usr/bin/env python3
"""Benchmark H3 output projection directly from head-major attention values."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.head_major_mpp_output import head_major_mpp_output
from minimax_h3_mlx.projection import mpp_bf16_linear


def _expand(value: mx.array, *, heads: int, rows: int) -> mx.array:
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] != 128:
        raise ValueError("QKV capture tensors must have shape [1, heads, rows, 128].")
    if heads % value.shape[1]:
        raise ValueError("Requested heads must be divisible by captured heads.")
    repeats = math.ceil(rows / value.shape[2])
    return mx.tile(value, (1, heads // value.shape[1], repeats, 1))[:, :, :rows]


def _measure(operations, schedule):
    samples = {name: [] for name in operations}
    for name in schedule:
        started = time.perf_counter()
        mx.eval(operations[name]())
        samples[name].append(time.perf_counter() - started)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qkv_capture", type=Path)
    parser.add_argument("block_page", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--rows", type=int, nargs="+", default=[9_482, 25_138])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows) < 1 or min(args.heads, args.pairs) < 1:
        parser.error("Heads, rows, and pairs must be positive.")

    capture = mx.load(str(args.qkv_capture))
    page = mx.load(str(args.block_page))
    weight_key = f"blocks.{args.block}.attn.out_proj.weight"
    weight = page.get(weight_key)
    if weight is None or weight.dtype != mx.bfloat16:
        parser.error(f"Block page has no BF16 {weight_key} tensor.")

    cases = []
    for rows in args.rows:
        query = _expand(capture["tensor_0"], heads=args.heads, rows=rows)
        key = _expand(capture["tensor_1"], heads=args.heads, rows=rows)
        value = _expand(capture["tensor_2"], heads=args.heads, rows=rows)
        attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=128**-0.5)
        mx.eval(attended, weight)
        packed = attended.transpose(0, 2, 1, 3).reshape(1, rows, args.heads * 128)

        operations = {
            "mlx": lambda packed=packed: packed @ weight.T,
            "packed_mpp": lambda packed=packed: mpp_bf16_linear(packed, weight),
            "head_major_mpp": lambda attended=attended: head_major_mpp_output(attended, weight),
        }
        outputs = {name: operation() for name, operation in operations.items()}
        mx.eval(*outputs.values())
        schedule = []
        forward = tuple(operations)
        reverse = tuple(reversed(forward))
        for _ in range(args.pairs):
            schedule.extend(forward)
            schedule.extend(reverse)
        samples = _measure(operations, schedule)
        peaks = {}
        for name, operation in operations.items():
            mx.clear_cache()
            mx.reset_peak_memory()
            mx.eval(operation())
            peaks[name] = int(mx.get_peak_memory())
        reference = outputs["mlx"].astype(mx.float32)
        candidate = outputs["head_major_mpp"].astype(mx.float32)
        difference = candidate - reference
        cases.append(
            {
                "rows": rows,
                "median_seconds": {
                    name: statistics.median(values) for name, values in samples.items()
                },
                "samples_seconds": samples,
                "peak_bytes": peaks,
                "head_major_maximum_absolute_difference": float(
                    mx.max(mx.abs(difference)).item()
                ),
                "head_major_relative_l2_error": float(
                    (mx.sqrt(mx.sum(difference**2)) / mx.sqrt(mx.sum(reference**2))).item()
                ),
                "head_major_exact": bool(
                    mx.array_equal(outputs["mlx"], outputs["head_major_mpp"]).item()
                ),
            }
        )

    payload = {
        "operator": "h3_head_major_attention_output_projection",
        "block": args.block,
        "heads": args.heads,
        "pairs": args.pairs,
        "cases": cases,
        "device": mx.device_info(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
