#!/usr/bin/env python3
"""Benchmark MLX projection execution at exact MiniMax H3 matrix shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7_689)
    parser.add_argument("--input-dim", type=int, default=5_376)
    parser.add_argument("--output-dim", type=int, default=21_504)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.rows, args.input_dim, args.output_dim, args.iterations) < 1:
        parser.error("matrix dimensions and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    source = mx.full((args.rows, args.input_dim), 0.015625, dtype=dtype)
    weight = mx.full((args.output_dim, args.input_dim), 0.03125, dtype=dtype)
    mx.eval(source, weight)

    def projection(x: mx.array, w: mx.array) -> mx.array:
        return x @ w.T

    operation = mx.compile(projection) if args.compiled else projection
    mx.reset_peak_memory()

    def execute() -> tuple[float, mx.array]:
        started = time.perf_counter()
        result = operation(source, weight)
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
                "backend": "mlx_compiled" if args.compiled else "mlx_eager",
                "dtype": args.dtype,
                "rows": args.rows,
                "input_dim": args.input_dim,
                "output_dim": args.output_dim,
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
