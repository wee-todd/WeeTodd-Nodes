#!/usr/bin/env python3
"""Evaluate low-cost corrections between two captured H3 evaluations."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics


def _load(path: Path, key: str = "tensor_0") -> mx.array:
    tensors = mx.load(str(path))
    if key not in tensors:
        raise KeyError(f"tensor key not found: {key}")
    value = tensors[key]
    return value[0] if value.ndim == 3 and value.shape[0] == 1 else value


def _timing(fn, repetitions: int) -> dict[str, float | list[float]]:
    for _ in range(2):
        mx.eval(fn())
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return {
        "median_seconds": ordered[len(ordered) // 2],
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_previous", type=Path)
    parser.add_argument("output_previous", type=Path)
    parser.add_argument("input_current", type=Path)
    parser.add_argument("output_current", type=Path)
    parser.add_argument("--fit-rows", type=int, default=4096)
    parser.add_argument("--test-rows", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/algorithm_search/cross_timestep.local.json"),
    )
    args = parser.parse_args()

    x0 = _load(args.input_previous).astype(mx.float32)
    y0 = _load(args.output_previous).astype(mx.float32)
    x1 = _load(args.input_current).astype(mx.float32)
    y1 = _load(args.output_current).astype(mx.float32)
    if not (x0.shape == y0.shape == x1.shape == y1.shape):
        raise ValueError("paired capture tensors must have identical shapes")
    required = args.fit_rows + args.test_rows
    if required > x0.shape[0]:
        raise ValueError(f"capture needs at least {required} rows, got {x0.shape[0]}")
    if min(args.fit_rows, args.test_rows, args.repetitions) < 1:
        raise ValueError("fit rows, test rows, and repetitions must be positive")

    dx_fit = x1[: args.fit_rows] - x0[: args.fit_rows]
    dy_fit = y1[: args.fit_rows] - y0[: args.fit_rows]
    epsilon = mx.array(1e-12, dtype=mx.float32)
    scalar = mx.sum(dx_fit * dy_fit) / mx.maximum(mx.sum(dx_fit * dx_fit), epsilon)
    channel = mx.sum(dx_fit * dy_fit, axis=0) / mx.maximum(
        mx.sum(dx_fit * dx_fit, axis=0), epsilon
    )
    channel_bias = mx.mean(dy_fit - channel * dx_fit, axis=0)
    mx.eval(scalar, channel, channel_bias)

    test = slice(args.fit_rows, required)
    dx_test = x1[test] - x0[test]
    candidates = {
        "reuse_previous": y0[test],
        "scalar_input_delta": y0[test] + scalar * dx_test,
        "channel_input_delta": y0[test] + channel * dx_test,
        "channel_affine_input_delta": y0[test] + channel * dx_test + channel_bias,
    }
    metrics = {
        name: asdict(numerical_metrics(y1[test], value))
        for name, value in candidates.items()
    }

    dx_full = x1 - x0
    timings = {
        "scalar_input_delta": _timing(
            lambda: (y0 + scalar * dx_full).astype(mx.bfloat16), args.repetitions
        ),
        "channel_input_delta": _timing(
            lambda: (y0 + channel * dx_full).astype(mx.bfloat16), args.repetitions
        ),
        "channel_affine_input_delta": _timing(
            lambda: (y0 + channel * dx_full + channel_bias).astype(mx.bfloat16),
            args.repetitions,
        ),
    }
    result = {
        "algorithm_class": "generatively_approximate",
        "shape": list(x0.shape),
        "fit_rows": args.fit_rows,
        "test_rows": args.test_rows,
        "scalar": float(scalar.item()),
        "metrics": metrics,
        "timings": timings,
        "notes": (
            "Offline block-24 output prediction from the previous evaluation and current input "
            "delta; timing excludes the transformer baseline."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
