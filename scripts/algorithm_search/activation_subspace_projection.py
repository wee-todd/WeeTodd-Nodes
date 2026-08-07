#!/usr/bin/env python3
"""Test an activation-aware input basis for one fixed H3 projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass


def _load_only(path: Path, key: str):
    tensors = mx.load(str(path))
    if key in tensors:
        return tensors[key], key
    matches = [name for name in tensors if name.endswith(key)]
    if len(matches) != 1:
        raise KeyError(f"tensor key not found or ambiguous: {key}")
    return tensors[matches[0]], matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("activation_capture", type=Path)
    parser.add_argument("--activation-key", default="tensor_0")
    parser.add_argument("--weight-key", default="blocks.24.mlp.fc1.weight")
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--fit-rows", type=int, default=1024)
    parser.add_argument("--error-rows", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--error-budget", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/algorithm_search/results.local.jsonl")
    )
    args = parser.parse_args()

    activation_map = mx.load(str(args.activation_capture))
    if args.activation_key not in activation_map:
        raise KeyError(f"activation key not found: {args.activation_key}")
    activations = activation_map[args.activation_key]
    if activations.ndim == 3:
        activations = activations[0]
    weight, weight_key = _load_only(args.checkpoint, args.weight_key)
    if activations.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"activation width {activations.shape[-1]} does not match weight {weight.shape}"
        )
    required = args.fit_rows + args.error_rows
    if required > activations.shape[0]:
        raise ValueError(f"capture needs at least {required} rows, got {activations.shape[0]}")

    fit = np.asarray(activations[: args.fit_rows].astype(mx.float32))
    fit -= fit.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(fit, full_matrices=False)
    timing_input = activations[None] if activations.ndim == 2 else activations
    error_input = activations[args.fit_rows : required][None]
    weight = weight.astype(mx.bfloat16)
    mx.eval(weight, timing_input, error_input)
    store = ExperimentStore(args.output)

    for rank in args.ranks:
        if rank < 1 or rank > right.shape[0]:
            raise ValueError(f"rank must be between 1 and {right.shape[0]}")
        basis = mx.array(right[:rank].T).astype(mx.bfloat16)
        reduced_weight = (
            basis.T.astype(mx.float32) @ weight.T.astype(mx.float32)
        ).astype(mx.bfloat16)
        mx.eval(basis, reduced_weight)

        def baseline(x):
            return x @ weight.T

        def candidate(x, basis=basis, reduced_weight=reduced_weight):
            return (x @ basis) @ reduced_weight

        result = benchmark_candidate(
            baseline,
            candidate,
            (timing_input,),
            error_inputs=(error_input,),
            candidate_id=f"{weight_key}:activation_pca_rank_{rank}",
            operator=weight_key,
            algorithm_class=AlgorithmClass.NUMERICALLY_APPROXIMATE,
            parameters={
                "family": "activation_pca_input_basis",
                "rank": rank,
                "fit_rows": args.fit_rows,
                "error_rows": args.error_rows,
                "timing_rows": int(activations.shape[0]),
            },
            transformation="activation_pca_input_basis",
            error_budget=args.error_budget,
            warmups=2,
            repetitions=args.repetitions,
            notes="Basis fitted on real block input; numerical metrics use held-out rows.",
        )
        store.append(
            result,
            context={
                "checkpoint_file": args.checkpoint.name,
                "capture_file": args.activation_capture.name,
            },
        )
        print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
