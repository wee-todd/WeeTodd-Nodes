#!/usr/bin/env python3
"""Benchmark fused H3 QKV preparation through SDPA and output projection."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.fused_qkv_prep import fused_qkv_prep
from minimax_h3_mlx.dit import apply_rotary

HIDDEN = 5_376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
ROTARY_DIM = 96


def _rms_norm(value: mx.array, weight: mx.array) -> mx.array:
    fp32 = value.astype(mx.float32)
    normalized = fp32 * mx.rsqrt(mx.mean(mx.square(fp32), axis=-1, keepdims=True) + 1e-5)
    return normalized.astype(mx.bfloat16) * weight


def _measure(operation, schedule: list[str]) -> dict[str, list[float]]:
    samples = {"reference": [], "candidate": []}
    for name in schedule:
        started = time.perf_counter()
        mx.eval(operation[name]())
        samples[name].append(time.perf_counter() - started)
    return samples


def _reference_pipeline(x, qkv_weight, out_weight, q_weight, k_weight, cosine, sine, rows):
    qkv = (x @ qkv_weight.T).reshape(1, rows, HEADS, 3, HEAD_DIM)
    q = _rms_norm(qkv[:, :, :, 0], q_weight).transpose(0, 2, 1, 3)
    k = _rms_norm(qkv[:, :, :, 1], k_weight).transpose(0, 2, 1, 3)
    v = qkv[:, :, :, 2].transpose(0, 2, 1, 3)
    q = apply_rotary(q, cosine, sine)
    k = apply_rotary(k, cosine, sine)
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
    packed = attended.transpose(0, 2, 1, 3).reshape(1, rows, INNER)
    return packed @ out_weight.T


def _candidate_pipeline(x, qkv_weight, out_weight, q_weight, k_weight, cosine, sine, rows):
    qkv = (x @ qkv_weight.T).reshape(1, rows, HEADS, 3, HEAD_DIM)
    q, k, v = fused_qkv_prep(qkv, q_weight, k_weight, cosine, sine)
    attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
    packed = attended.transpose(0, 2, 1, 3).reshape(1, rows, INNER)
    return packed @ out_weight.T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[9_482, 25_138])
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.rows) < 1 or args.pairs < 1:
        parser.error("rows and pairs must be positive")

    records = []
    for rows in args.rows:
        dtype = mx.bfloat16
        row_scale = mx.linspace(0.75, 1.25, rows, dtype=dtype).reshape(1, rows, 1)
        x = mx.full((1, rows, HIDDEN), 0.015625, dtype=dtype) * row_scale
        qkv_scale = mx.linspace(0.75, 1.25, 3 * INNER, dtype=dtype).reshape(-1, 1)
        qkv_weight = mx.full((3 * INNER, HIDDEN), 1 / 8192, dtype=dtype) * qkv_scale
        out_scale = mx.linspace(0.75, 1.25, HIDDEN, dtype=dtype).reshape(-1, 1)
        out_weight = mx.full((HIDDEN, INNER), 1 / 8192, dtype=dtype) * out_scale
        q_weight = mx.linspace(0.75, 1.25, HEAD_DIM, dtype=dtype)
        k_weight = mx.linspace(1.25, 0.75, HEAD_DIM, dtype=dtype)
        frequencies = mx.exp(
            -9.210340371976184
            * mx.arange(ROTARY_DIM // 2, dtype=mx.float32)
            / (ROTARY_DIM // 2)
        )
        angles = mx.arange(rows, dtype=mx.float32)[:, None] * frequencies[None]
        cosine = mx.concatenate((mx.cos(angles), mx.cos(angles)), axis=-1)
        sine = mx.concatenate((mx.sin(angles), mx.sin(angles)), axis=-1)
        mx.eval(x, qkv_weight, out_weight, q_weight, k_weight, cosine, sine)

        arguments = (x, qkv_weight, out_weight, q_weight, k_weight, cosine, sine, rows)
        reference = partial(_reference_pipeline, *arguments)
        candidate = partial(_candidate_pipeline, *arguments)

        expected = reference()
        actual = candidate()
        mx.eval(expected, actual)
        maximum_difference = float(mx.max(mx.abs(expected - actual)).item())
        cosine_similarity = float(
            mx.sum(expected.astype(mx.float32) * actual.astype(mx.float32)).item()
            / (
                mx.sqrt(mx.sum(expected.astype(mx.float32) ** 2)).item()
                * mx.sqrt(mx.sum(actual.astype(mx.float32) ** 2)).item()
            )
        )
        operations = {"reference": reference, "candidate": candidate}
        mx.eval(reference(), candidate())
        schedule = [name for _ in range(args.pairs) for name in ("reference", "candidate")]
        schedule += [name for _ in range(args.pairs) for name in ("candidate", "reference")]
        samples = _measure(operations, schedule)
        reference_median = statistics.median(samples["reference"])
        candidate_median = statistics.median(samples["candidate"])

        mx.clear_cache()
        mx.reset_peak_memory()
        mx.eval(reference())
        reference_peak = int(mx.get_peak_memory())
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.eval(candidate())
        candidate_peak = int(mx.get_peak_memory())
        records.append(
            {
                "rows": rows,
                "reference_median_seconds": reference_median,
                "candidate_median_seconds": candidate_median,
                "runtime_reduction_percent": 100.0 * (1.0 - candidate_median / reference_median),
                "reference_peak_bytes": reference_peak,
                "candidate_peak_bytes": candidate_peak,
                "peak_reduction_bytes": reference_peak - candidate_peak,
                "maximum_absolute_difference": maximum_difference,
                "cosine_similarity": cosine_similarity,
                "samples_seconds": samples,
            }
        )

    payload = {
        "operator": "h3_qkv_projection_through_output_projection",
        "candidate": "fused_qk_rmsnorm_rotary_qkv_layout",
        "pairs": args.pairs,
        "cases": records,
        "device": mx.device_info(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        from pathlib import Path

        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
