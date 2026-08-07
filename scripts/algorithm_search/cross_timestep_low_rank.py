#!/usr/bin/env python3
"""Fit a low-rank map to cross-timestep block-correction residuals."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics
from minimax_h3_mlx.algorithm_search.low_rank import (
    fit_reduced_map,
    randomized_right_basis,
    supervised_input_basis,
)


def _load(path: Path, key: str = "tensor_0") -> mx.array:
    tensors = mx.load(str(path))
    if key not in tensors:
        raise KeyError(f"tensor key not found: {key}")
    value = tensors[key]
    return value[0] if value.ndim == 3 and value.shape[0] == 1 else value


def _timing(fn, repetitions: int) -> dict[str, float]:
    for _ in range(2):
        mx.eval(fn())
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - started)
    return {
        "median_seconds": float(np.median(samples)),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_previous", type=Path)
    parser.add_argument("output_previous", type=Path)
    parser.add_argument("input_current", type=Path)
    parser.add_argument("output_current", type=Path)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--fit-rows", type=int, default=4096)
    parser.add_argument("--test-rows", type=int, default=2048)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/algorithm_search/cross_timestep_low_rank.local.json"),
    )
    args = parser.parse_args()
    if not args.ranks or min(args.ranks) < 1:
        raise ValueError("ranks must be positive")

    x0 = _load(args.input_previous).astype(mx.float32)
    y0 = _load(args.output_previous).astype(mx.float32)
    x1 = _load(args.input_current).astype(mx.float32)
    y1 = _load(args.output_current).astype(mx.float32)
    if not (x0.shape == y0.shape == x1.shape == y1.shape):
        raise ValueError("paired capture tensors must have identical shapes")
    required = args.fit_rows + args.test_rows
    if required > x0.shape[0]:
        raise ValueError(f"capture needs at least {required} rows, got {x0.shape[0]}")

    dx_fit_mx = x1[: args.fit_rows] - x0[: args.fit_rows]
    dy_fit_mx = y1[: args.fit_rows] - y0[: args.fit_rows]
    epsilon = mx.array(1e-12, dtype=mx.float32)
    channel = mx.sum(dx_fit_mx * dy_fit_mx, axis=0) / mx.maximum(
        mx.sum(dx_fit_mx * dx_fit_mx, axis=0), epsilon
    )
    bias = mx.mean(dy_fit_mx - channel * dx_fit_mx, axis=0)
    residual_fit_mx = dy_fit_mx - channel * dx_fit_mx - bias
    mx.eval(channel, bias, residual_fit_mx)
    dx_fit = np.asarray(dx_fit_mx)
    residual_fit = np.asarray(residual_fit_mx)

    max_rank = max(args.ranks)
    target_basis_full = randomized_right_basis(
        residual_fit, max_rank, oversample=args.oversample, seed=args.seed + 1
    )
    input_basis_full = supervised_input_basis(dx_fit, residual_fit, target_basis_full)
    test = slice(args.fit_rows, required)
    dx_test = x1[test] - x0[test]
    affine_test = y0[test] + channel * dx_test + bias
    dx_full = x1 - x0
    results = []

    for rank in args.ranks:
        input_basis_np = input_basis_full[:, :rank]
        target_basis_np = target_basis_full[:, :rank]
        mapping_np = fit_reduced_map(
            dx_fit, residual_fit, input_basis_np, target_basis_np
        )
        input_basis = mx.array(input_basis_np).astype(mx.bfloat16)
        target_basis = mx.array(target_basis_np).astype(mx.bfloat16)
        mapping = mx.array(mapping_np).astype(mx.bfloat16)
        predicted = affine_test + (
            (dx_test.astype(mx.bfloat16) @ input_basis) @ mapping
        ) @ target_basis.T
        metrics = asdict(numerical_metrics(y1[test], predicted))

        def apply_full(
            input_basis=input_basis,
            mapping=mapping,
            target_basis=target_basis,
        ):
            correction = ((dx_full.astype(mx.bfloat16) @ input_basis) @ mapping) @ target_basis.T
            return (y0 + channel * dx_full + bias + correction).astype(mx.bfloat16)

        timing = _timing(apply_full, args.repetitions)
        results.append({"rank": rank, "metrics": metrics, "timing": timing})

    payload = {
        "algorithm_class": "generatively_approximate",
        "shape": list(x0.shape),
        "fit_rows": args.fit_rows,
        "test_rows": args.test_rows,
        "oversample": args.oversample,
        "results": results,
        "notes": "Low-rank map predicts residual after per-channel affine input-delta correction.",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
