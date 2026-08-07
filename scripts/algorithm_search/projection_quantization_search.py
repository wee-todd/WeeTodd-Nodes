#!/usr/bin/env python3
"""Benchmark MLX packed-weight kernels on one real H3 projection and activation capture."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics
from minimax_h3_mlx.algorithm_search.projection_quantization import (
    PROJECTION_QUANTIZATION_SPECS,
    dynamic_input_probe,
    packed_nbytes,
    packed_projection_matmul,
    quantize_projection,
)


def _load_key(path: Path, key: str) -> tuple[mx.array, str]:
    tensors = mx.load(str(path))
    if key in tensors:
        value, resolved = tensors[key], key
    else:
        matches = [name for name in tensors if name.endswith(key)]
        if len(matches) != 1:
            raise KeyError(f"tensor key not found or ambiguous: {key}")
        resolved = matches[0]
        value = tensors[resolved]
    del tensors
    return value, resolved


def _timed(fn, warmups: int, repetitions: int) -> dict[str, object]:
    for _ in range(warmups):
        output = fn()
        mx.eval(output)
        del output
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        output = fn()
        mx.eval(output)
        samples.append(time.perf_counter() - started)
        del output
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
    }


def _peak_delta(fn) -> dict[str, int]:
    mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    output = fn()
    mx.eval(output)
    peak = int(mx.get_peak_memory())
    del output
    return {
        "active_before_bytes": active_before,
        "peak_bytes": peak,
        "peak_delta_bytes": max(0, peak - active_before),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("activation_capture", type=Path)
    parser.add_argument("--weight-key", default="blocks.24.mlp.fc1.weight")
    parser.add_argument("--activation-key", default="tensor_0")
    parser.add_argument(
        "--format", choices=sorted(PROJECTION_QUANTIZATION_SPECS), default="affine4"
    )
    parser.add_argument("--timing-rows", type=int, default=9477)
    parser.add_argument("--error-rows", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.timing_rows, args.error_rows, args.repetitions) < 1 or args.warmups < 0:
        raise ValueError("rows/repetitions must be positive and warmups must be non-negative")

    inputs, activation_key = _load_key(args.activation_capture, args.activation_key)
    if inputs.ndim == 3:
        if inputs.shape[0] != 1:
            raise ValueError(f"expected capture batch 1, got {inputs.shape}")
        inputs = inputs[0]
    if inputs.ndim != 2:
        raise ValueError(f"expected rank-2 activation capture, got {inputs.shape}")
    if args.timing_rows > inputs.shape[0] or args.error_rows > inputs.shape[0]:
        raise ValueError(f"requested rows exceed the {inputs.shape[0]} captured rows")
    inputs = inputs[: args.timing_rows].astype(mx.bfloat16)
    error_inputs = inputs[: args.error_rows]
    weight, weight_key = _load_key(args.checkpoint, args.weight_key)
    weight = weight.astype(mx.bfloat16)
    if inputs.shape[-1] != weight.shape[-1]:
        raise ValueError(f"activation width {inputs.shape[-1]} does not match {weight.shape}")
    mx.eval(inputs, error_inputs, weight)

    def baseline(weight=weight):
        return inputs @ weight.T

    baseline_timing = _timed(baseline, args.warmups, args.repetitions)
    baseline_peak = _peak_delta(baseline)
    reference = error_inputs @ weight.T
    mx.eval(reference)
    baseline_weight_bytes = int(weight.nbytes)

    spec = PROJECTION_QUANTIZATION_SPECS[args.format]
    packed = quantize_projection(weight, spec)
    mx.eval(*packed)
    candidate_weight_bytes = packed_nbytes(packed)
    del baseline
    del weight
    mx.clear_cache()

    def candidate():
        return packed_projection_matmul(inputs, packed, spec)

    candidate_timing = _timed(candidate, args.warmups, args.repetitions)
    candidate_peak = _peak_delta(candidate)
    candidate_error_output = packed_projection_matmul(error_inputs, packed, spec)
    mx.eval(candidate_error_output)
    errors = numerical_metrics(reference, candidate_error_output)
    dynamic_supported, dynamic_error = dynamic_input_probe(error_inputs, packed, spec)

    payload = {
        "experiment": "h3_projection_packed_weight",
        "operator": weight_key,
        "activation": activation_key,
        "format": {
            "name": spec.name,
            "mode": spec.mode,
            "group_size": spec.group_size,
            "bits": spec.bits,
        },
        "geometry": {
            "timing_rows": args.timing_rows,
            "error_rows": args.error_rows,
            "input_width": int(inputs.shape[-1]),
            "output_width": int(reference.shape[-1]),
        },
        "storage": {
            "baseline_weight_bytes": baseline_weight_bytes,
            "candidate_weight_bytes": candidate_weight_bytes,
            "fraction_of_bf16": candidate_weight_bytes / baseline_weight_bytes,
            "bytes_saved": baseline_weight_bytes - candidate_weight_bytes,
        },
        "baseline": {**baseline_timing, **baseline_peak},
        "candidate": {**candidate_timing, **candidate_peak},
        "speedup": baseline_timing["median_seconds"] / candidate_timing["median_seconds"],
        "errors": {
            "relative_l2_error": errors.relative_l2_error,
            "cosine_similarity": errors.cosine_similarity,
            "max_absolute_error": errors.max_absolute_error,
            "mean_absolute_error": errors.mean_absolute_error,
            "rmse": errors.rmse,
        },
        "dynamic_quantized_input": {
            "supported": dynamic_supported,
            "error": dynamic_error,
        },
        "environment": {
            "mlx": mx.__version__,
            "device": mx.device_info(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
