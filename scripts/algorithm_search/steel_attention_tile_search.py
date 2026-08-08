#!/usr/bin/env python3
"""Sweep classic MLX Steel attention tiles at H3 checkpoint-derived shapes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.steel_attention import (
    SteelAttentionTile,
    steel_attention,
)
from minimax_h3_mlx.load import load_dit

DEFAULT_TILES = (
    SteelAttentionTile(16, 16, 2),
    SteelAttentionTile(16, 32, 2),
    SteelAttentionTile(16, 64, 2),
    SteelAttentionTile(32, 16, 4),
    SteelAttentionTile(32, 32, 4),
    SteelAttentionTile(32, 64, 4),
    SteelAttentionTile(64, 16, 8),
    SteelAttentionTile(64, 32, 8),
    SteelAttentionTile(64, 64, 8),
)


def _capture(path: Path) -> mx.array:
    value = mx.load(str(path)).get("tensor_0")
    if value is None:
        raise ValueError(f"capture has no tensor_0: {path}")
    return value.astype(mx.bfloat16)


def _expanded_rows(source: mx.array, rows: int) -> mx.array:
    if source.shape[1] == rows:
        return source
    repeats = math.ceil(rows / source.shape[1])
    return mx.tile(source, (1, repeats, 1))[:, :rows, :]


def _checkpoint_qkv(checkpoint: Path, capture: Path, block_index: int, rows: int):
    dit = load_dit(checkpoint)
    block = dit.blocks[block_index]
    normalized = _expanded_rows(_capture(capture), rows)
    qkv = block.attn.qkv_proj(normalized).reshape(1, rows, block.attn.heads, 3, block.attn.head_dim)
    query, key, value = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
    query = block.attn.q_norm(query).transpose(0, 2, 1, 3)
    key = block.attn.k_norm(key).transpose(0, 2, 1, 3)
    value = value.transpose(0, 2, 1, 3)
    mx.eval(query, key, value)
    scale = block.attn.scale
    heads = block.attn.heads
    del qkv, normalized, block, dit
    gc.collect()
    mx.clear_cache()
    return query, key, value, scale, heads


def _measure(operation, repetitions: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(operation())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _quality(candidate: mx.array, reference: mx.array) -> dict[str, float | bool]:
    delta = candidate.astype(mx.float32) - reference.astype(mx.float32)
    numerator = mx.sum(delta * delta)
    denominator = mx.sum(reference.astype(mx.float32) ** 2)
    exact = mx.array_equal(candidate, reference)
    maximum = mx.max(mx.abs(delta))
    mx.eval(numerator, denominator, exact, maximum)
    return {
        "bit_exact": bool(exact.item()),
        "max_abs_error": float(maximum.item()),
        "relative_l2": math.sqrt(float(numerator.item()) / float(denominator.item())),
    }


def _mlx_attention(query, key, value, *, scale: float):
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale).transpose(
        0, 2, 1, 3
    )


def _steel_attention(query, key, value, *, scale: float, tile: SteelAttentionTile):
    return steel_attention(query, key, value, scale=scale, tile=tile)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--rows", type=int, nargs="+", default=[9477, 25138])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint is not a file: {args.checkpoint}")
    if not args.capture.is_file():
        parser.error(f"capture is not a file: {args.capture}")
    if min(args.rows) < 1 or min(args.warmups, args.repetitions) < 1:
        parser.error("rows, warmups, and repetitions must be positive")

    cases = []
    for rows in args.rows:
        query, key, value, scale, heads = _checkpoint_qkv(
            args.checkpoint, args.capture, args.block, rows
        )

        baseline = partial(_mlx_attention, query, key, value, scale=scale)

        reference = baseline()
        mx.eval(reference)
        for _ in range(args.warmups):
            mx.eval(baseline())
        baseline_median, baseline_samples = _measure(baseline, args.repetitions)
        baseline_peak = None
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
            mx.eval(baseline())
            baseline_peak = int(mx.get_peak_memory())

        for tile in DEFAULT_TILES:
            record = {
                "rows": rows,
                "heads": heads,
                "head_dim": 128,
                "tile": {
                    "query_rows": tile.query_rows,
                    "key_rows": tile.key_rows,
                    "simdgroups_m": tile.simdgroups_m,
                    "simdgroups_n": tile.simdgroups_n,
                },
                "baseline_median_seconds": baseline_median,
                "baseline_samples_seconds": baseline_samples,
                "baseline_peak_bytes": baseline_peak,
            }
            try:
                started = time.perf_counter()
                candidate = steel_attention(query, key, value, scale=scale, tile=tile)
                mx.eval(candidate)
                record["compile_and_first_seconds"] = time.perf_counter() - started
                record.update(_quality(candidate, reference))
                del candidate
                gc.collect()
                for _ in range(args.warmups):
                    mx.eval(steel_attention(query, key, value, scale=scale, tile=tile))

                operation = partial(_steel_attention, query, key, value, scale=scale, tile=tile)

                median, samples = _measure(operation, args.repetitions)
                record["candidate_median_seconds"] = median
                record["candidate_samples_seconds"] = samples
                record["speedup"] = baseline_median / median
                record["runtime_reduction_percent"] = (1.0 - median / baseline_median) * 100.0
                if hasattr(mx, "reset_peak_memory"):
                    mx.reset_peak_memory()
                    mx.eval(operation())
                    record["candidate_peak_bytes"] = int(mx.get_peak_memory())
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                mx.clear_cache()
            cases.append(record)

        del reference, query, key, value
        gc.collect()
        mx.clear_cache()

    payload = {
        "operator": "classic_steel_full_self_attention",
        "mlx_version": version("mlx"),
        "device": mx.device_info(),
        "checkpoint": args.checkpoint.name,
        "capture": args.capture.name,
        "block": args.block,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
