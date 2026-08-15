#!/usr/bin/env python3
"""Benchmark fused H3 MPP QKV preparation against the exact packed MPP path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.fused_mpp_qkv import fused_mpp_qkv
from minimax_h3_mlx.dit import apply_rotary
from minimax_h3_mlx.projection import MPPTile, mpp_bf16_linear


def _expand(value: mx.array, rows: int) -> mx.array:
    repeats = math.ceil(rows / value.shape[1])
    return mx.tile(value, (1, repeats, 1))[:, :rows]


def _rms_norm(value: mx.array, weight: mx.array) -> mx.array:
    fp32 = value.astype(mx.float32)
    normalized = fp32 * mx.rsqrt(mx.mean(fp32**2, axis=-1, keepdims=True) + 1e-5)
    return normalized.astype(mx.bfloat16) * weight


def _reference(source, weight, q_weight, k_weight, cosine, sine):
    rows = source.shape[1]
    heads = weight.shape[0] // (3 * 128)
    projected = mpp_bf16_linear(
        source, weight, tile=MPPTile(rows=32, columns=128, simdgroups=4)
    ).reshape(1, rows, heads, 3, 128)
    query = _rms_norm(projected[:, :, :, 0], q_weight).transpose(0, 2, 1, 3)
    key = _rms_norm(projected[:, :, :, 1], k_weight).transpose(0, 2, 1, 3)
    value = projected[:, :, :, 2].transpose(0, 2, 1, 3)
    return apply_rotary(query, cosine, sine), apply_rotary(key, cosine, sine), value


def _attention_pipeline(prepare, out_weight):
    query, key, value = prepare()
    attended = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=128**-0.5
    )
    packed = attended.transpose(0, 2, 1, 3).reshape(
        1, query.shape[2], query.shape[1] * query.shape[3]
    )
    return mpp_bf16_linear(packed, out_weight)


def _measure(operations, schedule):
    samples = {name: [] for name in operations}
    for name in schedule:
        started = time.perf_counter()
        mx.eval(*operations[name]())
        samples[name].append(time.perf_counter() - started)
    return samples


def _relative_l2(candidate, reference) -> float:
    difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
    return float(
        mx.sqrt(mx.sum(difference**2) / mx.sum(reference.astype(mx.float32) ** 2)).item()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_capture", type=Path)
    parser.add_argument("block_page", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--rows", type=int, nargs="+", default=[9_482, 25_138])
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source_capture = mx.load(str(args.source_capture))
    page = mx.load(str(args.block_page))
    source_seed = source_capture["tensor_0"]
    prefix = f"blocks.{args.block}.attn"
    weight = page[f"{prefix}.qkv_proj.weight"]
    q_weight = page[f"{prefix}.q_norm.weight"]
    k_weight = page[f"{prefix}.k_norm.weight"]
    out_weight = page[f"{prefix}.out_proj.weight"]
    heads = weight.shape[0] // (3 * 128)

    cases = []
    for rows in args.rows:
        source = _expand(source_seed, rows)
        angles = mx.arange(rows * 48, dtype=mx.float32).reshape(rows, 48) / 10_000
        cosine = mx.concatenate((mx.cos(angles), mx.cos(angles)), axis=-1)
        sine = mx.concatenate((mx.sin(angles), mx.sin(angles)), axis=-1)
        mx.eval(source, weight, q_weight, k_weight, cosine, sine)

        operations = {
            "exact_mpp_prepare": lambda source=source, cosine=cosine, sine=sine: _reference(
                source, weight, q_weight, k_weight, cosine, sine
            ),
            "fused_mpp_prepare": lambda source=source, cosine=cosine, sine=sine: fused_mpp_qkv(
                source, weight, q_weight, k_weight, cosine, sine
            ),
        }
        outputs = {name: operation() for name, operation in operations.items()}
        mx.eval(*(item for result in outputs.values() for item in result))
        schedule = []
        for _ in range(args.pairs):
            schedule.extend(operations)
            schedule.extend(reversed(operations))
        samples = _measure(operations, schedule)

        peaks = {}
        for name, operation in operations.items():
            mx.clear_cache()
            mx.reset_peak_memory()
            mx.eval(*operation())
            peaks[name] = int(mx.get_peak_memory())

        reference = outputs["exact_mpp_prepare"]
        candidate = outputs["fused_mpp_prepare"]
        pipelines = {
            "exact_mpp_attention": lambda operation=operations[
                "exact_mpp_prepare"
            ]: (_attention_pipeline(operation, out_weight),),
            "fused_mpp_attention": lambda operation=operations[
                "fused_mpp_prepare"
            ]: (_attention_pipeline(operation, out_weight),),
        }
        pipeline_outputs = {name: operation()[0] for name, operation in pipelines.items()}
        mx.eval(*pipeline_outputs.values())
        pipeline_schedule = []
        for _ in range(args.pairs):
            pipeline_schedule.extend(pipelines)
            pipeline_schedule.extend(reversed(pipelines))
        pipeline_samples = _measure(pipelines, pipeline_schedule)
        pipeline_peaks = {}
        for name, operation in pipelines.items():
            mx.clear_cache()
            mx.reset_peak_memory()
            mx.eval(*operation())
            pipeline_peaks[name] = int(mx.get_peak_memory())
        cases.append(
            {
                "rows": rows,
                "heads": heads,
                "median_seconds": {
                    name: statistics.median(values) for name, values in samples.items()
                },
                "samples_seconds": samples,
                "peak_bytes": peaks,
                "relative_l2_error": {
                    label: _relative_l2(candidate[index], reference[index])
                    for index, label in enumerate(("query", "key", "value"))
                },
                "value_exact": bool(mx.array_equal(candidate[2], reference[2]).item()),
                "attention_pipeline_median_seconds": {
                    name: statistics.median(values)
                    for name, values in pipeline_samples.items()
                },
                "attention_pipeline_samples_seconds": pipeline_samples,
                "attention_pipeline_peak_bytes": pipeline_peaks,
                "attention_pipeline_relative_l2_error": _relative_l2(
                    pipeline_outputs["fused_mpp_attention"],
                    pipeline_outputs["exact_mpp_attention"],
                ),
            }
        )

    payload = {
        "operator": "h3_fused_mpp_qkv_prepare",
        "block": args.block,
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
