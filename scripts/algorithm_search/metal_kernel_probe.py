#!/usr/bin/env python3
"""Run a bounded H3-shaped operator loop for Metal System Trace inspection."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx


def _measure(operation, *, warmups: int, repetitions: int) -> dict[str, object]:
    for _ in range(warmups):
        mx.eval(operation())
    samples = []
    mx.reset_peak_memory()
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(operation())
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("sdpa", "qkv_projection"), required=True)
    parser.add_argument("--rows", type=int, default=9_477)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=5_376)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gputrace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--start-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if min(args.rows, args.heads, args.head_dim, args.repetitions) < 1 or args.warmups < 0:
        raise ValueError("geometry and repetitions must be positive; warmups cannot be negative")

    mx.random.seed(args.seed)
    if args.operator == "sdpa":
        shape = (1, args.heads, args.rows, args.head_dim)
        query = mx.random.normal(shape).astype(mx.bfloat16)
        key = mx.random.normal(shape).astype(mx.bfloat16)
        value = mx.random.normal(shape).astype(mx.bfloat16)
        mx.eval(query, key, value)

        def operation():
            return mx.fast.scaled_dot_product_attention(
                query, key, value, scale=args.head_dim**-0.5
            )

        input_shapes = [list(shape)] * 3
    else:
        inner = args.heads * args.head_dim
        source = mx.random.normal((1, args.rows, args.hidden_size)).astype(mx.bfloat16)
        weight = mx.random.normal((3 * inner, args.hidden_size)).astype(mx.bfloat16)
        mx.eval(source, weight)

        def operation():
            return (source @ weight.T).reshape(1, args.rows, args.heads, 3, args.head_dim)

        input_shapes = [list(source.shape), list(weight.shape)]

    for _ in range(args.warmups):
        mx.eval(operation())
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.ready_file.touch(exist_ok=False)
    if args.start_file is not None:
        deadline = time.monotonic() + args.start_timeout
        while not args.start_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"start file did not appear: {args.start_file}")
            time.sleep(0.05)
    if args.gputrace is not None:
        if args.gputrace.exists():
            raise FileExistsError(f"refusing to overwrite capture: {args.gputrace}")
        args.gputrace.parent.mkdir(parents=True, exist_ok=True)
        mx.metal.start_capture(str(args.gputrace))
    try:
        result = _measure(operation, warmups=0, repetitions=args.repetitions)
    finally:
        if args.gputrace is not None:
            mx.metal.stop_capture()

    payload = {
        "operator": args.operator,
        "dtype": "bfloat16",
        "mlx_version": version("mlx"),
        "device": mx.device_info(),
        "geometry": {
            "rows": args.rows,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "hidden_size": args.hidden_size,
            "input_shapes": input_shapes,
        },
        "warmups_before_capture": args.warmups,
        "repetitions": args.repetitions,
        "measurement": result,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
