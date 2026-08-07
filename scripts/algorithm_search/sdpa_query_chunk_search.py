#!/usr/bin/env python3
"""Benchmark exact query-chunked SDPA at representative H3 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass
from minimax_h3_mlx.algorithm_search.sdpa import query_chunked_sdpa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=37_893)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--error-query-rows", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-budget", type=float, default=1e-4)
    parser.add_argument(
        "--defer-chunk-sync",
        action="store_true",
        help="Let MLX schedule every chunk lazily until the concatenated output is evaluated.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/algorithm_search/results.local.jsonl")
    )
    args = parser.parse_args()
    if min(args.rows, args.heads, args.head_dim, args.error_query_rows) < 1:
        raise ValueError("geometry values must be positive")

    mx.random.seed(args.seed)
    shape = (1, args.heads, args.rows, args.head_dim)
    q = mx.random.normal(shape).astype(mx.bfloat16)
    k = mx.random.normal(shape).astype(mx.bfloat16)
    v = mx.random.normal(shape).astype(mx.bfloat16)
    scale = args.head_dim**-0.5
    mx.eval(q, k, v)
    store = ExperimentStore(args.output)

    def baseline(query, keys, values):
        return mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale)

    for chunk_size in args.chunks:
        if chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        comparison_rows = min(args.rows, max(args.error_query_rows, chunk_size + 1))
        error_q = q[..., :comparison_rows, :]
        mx.eval(error_q)

        def candidate(query, keys, values, chunk_size=chunk_size):
            return query_chunked_sdpa(
                query,
                keys,
                values,
                scale=scale,
                chunk_size=chunk_size,
                synchronize_chunks=not args.defer_chunk_sync,
            )

        result = benchmark_candidate(
            baseline,
            candidate,
            (q, k, v),
            error_inputs=(error_q, k, v),
            candidate_id=(
                f"sdpa:query_chunk_{chunk_size}:rows_{args.rows}:"
                f"{'lazy' if args.defer_chunk_sync else 'synchronized'}"
            ),
            operator="mx.fast.scaled_dot_product_attention",
            algorithm_class=AlgorithmClass.EXACT,
            parameters={
                "query_chunk_size": chunk_size,
                "rows": args.rows,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "error_query_rows": comparison_rows,
                "synchronize_chunks": not args.defer_chunk_sync,
            },
            transformation="query_axis_partition",
            error_budget=args.error_budget,
            warmups=args.warmups,
            repetitions=args.repetitions,
            notes="Synthetic BF16 values at measured native H3 geometry; query-only partition.",
        )
        store.append(result, context={"seed": args.seed, "geometry": "native_1344x768_5s"})
        print(json.dumps(result.to_dict(), sort_keys=True))
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
