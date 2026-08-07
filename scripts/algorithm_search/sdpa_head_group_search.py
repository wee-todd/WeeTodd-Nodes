#!/usr/bin/env python3
"""Benchmark head-grouped SDPA at representative H3 geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass
from minimax_h3_mlx.algorithm_search.sdpa import head_grouped_sdpa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=37_893)
    parser.add_argument(
        "--query-rows",
        type=int,
        help="Query length when it differs from the complete key/value row count.",
    )
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--groups", type=int, nargs="+", default=[7, 14, 28])
    parser.add_argument("--error-query-rows", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--error-budget", type=float, default=1e-4)
    parser.add_argument("--synchronize-groups", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/algorithm_search/results.local.jsonl")
    )
    args = parser.parse_args()
    query_rows = args.rows if args.query_rows is None else args.query_rows
    if min(args.rows, query_rows, args.heads, args.head_dim, args.error_query_rows) < 1:
        raise ValueError("geometry values must be positive")

    mx.random.seed(args.seed)
    q = mx.random.normal((1, args.heads, query_rows, args.head_dim)).astype(mx.bfloat16)
    kv_shape = (1, args.heads, args.rows, args.head_dim)
    k = mx.random.normal(kv_shape).astype(mx.bfloat16)
    v = mx.random.normal(kv_shape).astype(mx.bfloat16)
    error_q = q[..., : min(query_rows, args.error_query_rows), :]
    scale = args.head_dim**-0.5
    mx.eval(q, k, v, error_q)
    store = ExperimentStore(args.output)

    def baseline(query, keys, values):
        return mx.fast.scaled_dot_product_attention(query, keys, values, scale=scale)

    for heads_per_group in args.groups:
        if heads_per_group < 1 or heads_per_group > args.heads:
            raise ValueError("heads per group must be between one and the total head count")

        def candidate(query, keys, values, heads_per_group=heads_per_group):
            return head_grouped_sdpa(
                query,
                keys,
                values,
                scale=scale,
                heads_per_group=heads_per_group,
                synchronize_groups=args.synchronize_groups,
            )

        mode = "synchronized" if args.synchronize_groups else "lazy"
        result = benchmark_candidate(
            baseline,
            candidate,
            (q, k, v),
            error_inputs=(error_q, k, v),
            candidate_id=(
                f"sdpa:head_group_{heads_per_group}:q_{query_rows}:kv_{args.rows}:{mode}"
            ),
            operator="mx.fast.scaled_dot_product_attention",
            algorithm_class=AlgorithmClass.EXACT,
            parameters={
                "heads_per_group": heads_per_group,
                "rows": args.rows,
                "query_rows": query_rows,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "error_query_rows": int(error_q.shape[-2]),
                "synchronize_groups": args.synchronize_groups,
            },
            transformation="head_axis_partition",
            error_budget=args.error_budget,
            warmups=args.warmups,
            repetitions=args.repetitions,
            notes="Synthetic BF16 values at measured H3 geometry; head-only partition.",
        )
        store.append(
            result,
            context={"seed": args.seed, "geometry": f"q_{query_rows}_kv_{args.rows}"},
        )
        print(json.dumps(result.to_dict(), sort_keys=True))
        mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
