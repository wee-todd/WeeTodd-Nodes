#!/usr/bin/env python3
"""Benchmark an MLX-owned MPP BF16 projection at exact MiniMax H3 shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

from minimax_h3_mlx.projection import MPPTile, mpp_bf16_linear


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7_689)
    parser.add_argument("--input-dim", type=int, default=5_376)
    parser.add_argument("--output-dim", type=int, default=21_504)
    parser.add_argument("--tile-m", type=int, default=32)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--simdgroups", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    positive = (
        args.rows,
        args.input_dim,
        args.output_dim,
        args.tile_m,
        args.tile_n,
        args.simdgroups,
        args.iterations,
    )
    if min(positive) < 1:
        parser.error("matrix, tile, simdgroup, and iteration values must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    source = mx.full((args.rows, args.input_dim), 0.015625, dtype=mx.bfloat16)
    weight = mx.full((args.output_dim, args.input_dim), 0.03125, dtype=mx.bfloat16)
    mx.eval(source, weight)
    tile = MPPTile(args.tile_m, args.tile_n, args.simdgroups)

    def projection() -> mx.array:
        return mpp_bf16_linear(source, weight, tile=tile)

    mx.reset_peak_memory()

    def execute() -> tuple[float, mx.array]:
        started = time.perf_counter()
        result = projection()
        mx.eval(result)
        return time.perf_counter() - started, result

    first_seconds, result = execute()
    for _ in range(args.warmup):
        _, result = execute()
    samples: list[float] = []
    for _ in range(args.iterations):
        seconds, result = execute()
        samples.append(seconds)

    median_seconds = statistics.median(samples)
    operation_count = 2 * args.rows * args.input_dim * args.output_dim
    output_sample = float(result[0, 0])
    expected_sample = args.input_dim * 0.015625 * 0.03125
    print(
        json.dumps(
            {
                "backend": "mlx_mpp",
                "dtype": "bf16",
                "rows": args.rows,
                "input_dim": args.input_dim,
                "output_dim": args.output_dim,
                "tile_m": args.tile_m,
                "tile_n": args.tile_n,
                "simdgroups": args.simdgroups,
                "compile_plus_first_execution_seconds": first_seconds,
                "warm_median_seconds": median_seconds,
                "warm_samples_seconds": samples,
                "warm_tflops": operation_count / median_seconds / 1e12,
                "output_sample": output_sample,
                "expected_sample": expected_sample,
                "sample_absolute_error": abs(output_sample - expected_sample),
                "active_memory_bytes": mx.get_active_memory(),
                "peak_memory_bytes": mx.get_peak_memory(),
                "cache_memory_bytes": mx.get_cache_memory(),
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
