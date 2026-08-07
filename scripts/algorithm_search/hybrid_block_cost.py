#!/usr/bin/env python3
"""Measure the compute boundary for exact-video/predicted-text-audio H3 blocks."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.hybrid import (
    hybrid_theoretical_fraction,
    trajectory_state_bytes,
)


def _load_tensor(path: Path, key: str = "tensor_0") -> mx.array:
    tensors = mx.load(str(path))
    if key not in tensors:
        raise KeyError(f"tensor key not found: {key}")
    return tensors[key]


def _load_weight(path: Path, key: str) -> mx.array:
    tensors = mx.load(str(path))
    if key not in tensors:
        matches = [name for name in tensors if name.endswith(key)]
        if len(matches) != 1:
            raise KeyError(f"weight key not found or ambiguous: {key}")
        key = matches[0]
    weight = tensors[key].astype(mx.bfloat16)
    del tensors
    mx.eval(weight)
    return weight


def _leaves(value):
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in _leaves(child)]
    return [value]


def _measure(fn, *, warmups: int, repetitions: int) -> dict:
    for _ in range(warmups):
        mx.eval(*_leaves(fn()))
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        mx.eval(*_leaves(fn()))
        samples.append(time.perf_counter() - started)
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(*_leaves(fn()))
    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "samples_seconds": samples,
    }


def _comparison(name: str, baseline, candidate, warmups: int, repetitions: int) -> dict:
    baseline_stats = _measure(baseline, warmups=warmups, repetitions=repetitions)
    candidate_stats = _measure(candidate, warmups=warmups, repetitions=repetitions)
    return {
        "name": name,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "speedup": baseline_stats["median_seconds"]
        / max(candidate_stats["median_seconds"], 1e-12),
        "peak_delta_bytes": (
            candidate_stats["peak_memory_bytes"] - baseline_stats["peak_memory_bytes"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalized_input", type=Path)
    parser.add_argument("mlp_input", type=Path)
    parser.add_argument("--total-rows", type=int, default=9477)
    parser.add_argument("--predicted-rows", type=int, default=597)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=5376)
    parser.add_argument("--ffn-size", type=int, default=14336)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/algorithm_search/hybrid_block_cost.local.json"),
    )
    args = parser.parse_args()
    exact_rows = args.total_rows - args.predicted_rows
    if min(exact_rows, args.predicted_rows, args.repetitions) < 1:
        raise ValueError("row counts and repetitions must be positive")
    mx.random.seed(args.seed)
    results = []

    normalized = _load_tensor(args.normalized_input).astype(mx.bfloat16)
    if normalized.shape[-2:] != (args.total_rows, args.hidden_size):
        raise ValueError(f"unexpected normalized input shape: {normalized.shape}")
    qkv_weight = _load_weight(args.checkpoint, "blocks.24.attn.qkv_proj.weight")
    qkv_heads = qkv_weight.reshape(args.heads, 3, args.head_dim, args.hidden_size)
    q_weight = qkv_heads[:, 0].reshape(-1, args.hidden_size)
    kv_weight = qkv_heads[:, 1:].reshape(-1, args.hidden_size)
    mx.eval(q_weight, kv_weight)
    results.append(
        _comparison(
            "qkv_projection",
            lambda: normalized @ qkv_weight.T,
            lambda: (
                normalized @ kv_weight.T,
                normalized[:, args.predicted_rows :] @ q_weight.T,
            ),
            args.warmups,
            args.repetitions,
        )
    )
    normalized = qkv_weight = qkv_heads = q_weight = kv_weight = None
    gc.collect()
    mx.clear_cache()

    shape = (1, args.heads, args.total_rows, args.head_dim)
    q = mx.random.normal(shape).astype(mx.bfloat16)
    k = mx.random.normal(shape).astype(mx.bfloat16)
    v = mx.random.normal(shape).astype(mx.bfloat16)
    scale = args.head_dim**-0.5
    mx.eval(q, k, v)
    results.append(
        _comparison(
            "sdpa_queries",
            lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale),
            lambda: mx.fast.scaled_dot_product_attention(
                q[..., args.predicted_rows :, :], k, v, scale=scale
            ),
            args.warmups,
            args.repetitions,
        )
    )
    q = k = v = None
    gc.collect()
    mx.clear_cache()

    out_input = mx.random.normal(
        (1, args.total_rows, args.heads * args.head_dim)
    ).astype(mx.bfloat16)
    out_weight = _load_weight(args.checkpoint, "blocks.24.attn.out_proj.weight")
    mx.eval(out_input)
    results.append(
        _comparison(
            "attention_output_projection",
            lambda: out_input @ out_weight.T,
            lambda: out_input[:, args.predicted_rows :] @ out_weight.T,
            args.warmups,
            args.repetitions,
        )
    )
    out_input = out_weight = None
    gc.collect()
    mx.clear_cache()

    mlp_input = _load_tensor(args.mlp_input).astype(mx.bfloat16)
    fc1_weight = _load_weight(args.checkpoint, "blocks.24.mlp.fc1.weight")
    mx.eval(mlp_input)
    results.append(
        _comparison(
            "mlp_fc1",
            lambda: mlp_input @ fc1_weight.T,
            lambda: mlp_input[:, args.predicted_rows :] @ fc1_weight.T,
            args.warmups,
            args.repetitions,
        )
    )
    mlp_input = fc1_weight = None
    gc.collect()
    mx.clear_cache()

    fc2_input = mx.random.normal((1, args.total_rows, args.ffn_size)).astype(mx.bfloat16)
    fc2_weight = _load_weight(args.checkpoint, "blocks.24.mlp.fc2.weight")
    mx.eval(fc2_input)
    results.append(
        _comparison(
            "mlp_fc2",
            lambda: fc2_input @ fc2_weight.T,
            lambda: fc2_input[:, args.predicted_rows :] @ fc2_weight.T,
            args.warmups,
            args.repetitions,
        )
    )
    fc2_input = fc2_weight = None
    gc.collect()
    mx.clear_cache()

    predicted = mx.zeros((1, args.predicted_rows, args.hidden_size), dtype=mx.bfloat16)
    exact = mx.zeros((1, exact_rows, args.hidden_size), dtype=mx.bfloat16)
    mx.eval(predicted, exact)
    reconstruction = _measure(
        lambda: mx.concatenate([predicted, exact], axis=1),
        warmups=args.warmups,
        repetitions=args.repetitions,
    )

    steady_state = trajectory_state_bytes(
        predicted_rows=args.predicted_rows,
        hidden_size=args.hidden_size,
        blocks=args.blocks,
    )
    baseline_total = sum(item["baseline"]["median_seconds"] for item in results)
    candidate_total = sum(item["candidate"]["median_seconds"] for item in results)
    candidate_with_reconstruction = candidate_total + reconstruction["median_seconds"]
    payload = {
        "algorithm_class": "generatively_approximate",
        "geometry": {
            "total_rows": args.total_rows,
            "predicted_rows": args.predicted_rows,
            "exact_rows": exact_rows,
            "predicted_fraction": args.predicted_rows / args.total_rows,
            "qkv_theoretical_work_fraction": hybrid_theoretical_fraction(
                total_rows=args.total_rows,
                exact_rows=exact_rows,
                shared_projections=2,
            ),
        },
        "operations": results,
        "reconstruction": reconstruction,
        "aggregate": {
            "baseline_selected_seconds": baseline_total,
            "candidate_selected_seconds": candidate_total,
            "candidate_with_reconstruction_seconds": candidate_with_reconstruction,
            "speedup_with_reconstruction": baseline_total
            / max(candidate_with_reconstruction, 1e-12),
            "fraction_saved": 1.0 - candidate_with_reconstruction / baseline_total,
        },
        "trajectory_state": {
            "steady_bytes": steady_state,
            "fit_peak_bytes": 2 * steady_state,
            "parameter_bytes_bf16": 4 * args.hidden_size * args.blocks * 2,
        },
        "notes": (
            "Actual block-24 weights for projections; synthetic BF16 inputs for uncaptured "
            "SDPA, attention output, and FC2 shapes. Candidate outputs only exact video rows."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
