#!/usr/bin/env python3
"""Measure isolated production-shape FastH3 VSA attention time and MLX peak memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

import minimax_h3_mlx.vsa_h3 as vsa_module
from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig, vsa_h3_attention

PREFIX_SEGMENTS = (133, 414)
VIDEO_GRID = (37, 12, 20)
ROWS = sum(PREFIX_SEGMENTS) + 37 * 12 * 20
HEADS = 56
HEAD_DIM = 128


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("grouped", "indexed"), required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")

    shape = (1, HEADS, ROWS, HEAD_DIM)
    q = mx.full(shape, 0.015625, dtype=mx.bfloat16)
    k = mx.full(shape, 0.03125, dtype=mx.bfloat16)
    v = mx.full(shape, 0.0625, dtype=mx.bfloat16)
    gate = mx.zeros(shape, dtype=mx.bfloat16)
    mx.eval(q, k, v, gate)
    mx.clear_cache()
    input_active = int(mx.get_active_memory())

    indexed = args.profile == "indexed"
    config = FastH3VSAConfig(
        sparsity=0.9,
        min_tokens=8192,
        query_tile_batch=8,
        prefix_segments=PREFIX_SEGMENTS,
        video_grid=VIDEO_GRID,
        block_stack_preorder=indexed,
        consumer_backend="metal_indexed" if indexed else "grouped_sdpa",
    )
    samples = []
    peaks = []
    output_active = []
    counts_value = None
    for _ in range(args.iterations):
        mx.reset_peak_memory()
        started = time.perf_counter()
        output, counts = vsa_h3_attention(
            q,
            k,
            v,
            gate,
            scale=HEAD_DIM**-0.5,
            config=config,
        )
        mx.eval(output, counts)
        samples.append(time.perf_counter() - started)
        peaks.append(int(mx.get_peak_memory()))
        output_active.append(int(mx.get_active_memory()))
        counts_value = counts.reshape(-1).tolist()
        del output, counts
        mx.clear_cache()

    report = json.dumps(
        {
            "implementation": str(vsa_module.__file__),
            "profile": args.profile,
            "shape": list(shape),
            "iterations": args.iterations,
            "seconds": samples,
            "mean_seconds": sum(samples) / len(samples),
            "mean_warm_seconds": (
                sum(samples[1:]) / len(samples[1:]) if len(samples) > 1 else samples[0]
            ),
            "input_active_memory_bytes": input_active,
            "peak_memory_bytes": peaks,
            "peak_increment_bytes": [peak - input_active for peak in peaks],
            "output_active_memory_bytes": output_active,
            "route_counts": counts_value,
        },
        indent=2,
        sort_keys=True,
    )
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
