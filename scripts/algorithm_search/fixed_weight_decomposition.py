#!/usr/bin/env python3
"""Explore low-rank application of one fixed H3 weight tensor without loading the full model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weight-key", default="final_layer.video_out.weight")
    parser.add_argument("--rows", type=int, default=9806)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/algorithm_search/results.local.jsonl")
    )
    args = parser.parse_args()
    tensors = mx.load(str(args.checkpoint))
    if args.weight_key not in tensors:
        matches = [key for key in tensors if key.endswith(args.weight_key)]
        if len(matches) != 1:
            raise KeyError(f"weight key not found or ambiguous: {args.weight_key}")
        weight_key = matches[0]
    else:
        weight_key = args.weight_key
    weight = tensors[weight_key].astype(mx.float32)
    mx.eval(weight)
    weight_np = np.asarray(weight)
    u, singular, vh = np.linalg.svd(weight_np, full_matrices=False)
    rng = np.random.default_rng(0)
    inputs = mx.array(rng.standard_normal((1, args.rows, weight.shape[1])).astype(np.float32))
    store = ExperimentStore(args.output)
    for rank in args.ranks:
        if rank < 1 or rank > min(weight.shape):
            raise ValueError(f"rank must be between 1 and {min(weight.shape)}")
        first = mx.array(vh[:rank].T.astype(np.float32))
        second = mx.array((u[:, :rank] * singular[:rank]).T.astype(np.float32))

        def candidate(x, first=first, second=second):
            return (x @ first) @ second

        result = benchmark_candidate(
            lambda x: x @ weight.T,
            candidate,
            (inputs,),
            candidate_id=f"{weight_key}:svd_rank_{rank}",
            operator=weight_key,
            algorithm_class=AlgorithmClass.NUMERICALLY_APPROXIMATE,
            parameters={"family": "truncated_svd", "rank": rank, "rows": args.rows},
            transformation="truncated_svd",
            error_budget=0.05,
            warmups=2,
            repetitions=args.repetitions,
            notes=(
                "Fixed-weight two-factor application; optimize measured latency, not compression."
            ),
        )
        store.append(result, context={"checkpoint_file": args.checkpoint.name})
        print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
