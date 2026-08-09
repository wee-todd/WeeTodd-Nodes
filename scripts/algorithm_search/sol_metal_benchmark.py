#!/usr/bin/env python3
"""Benchmark the independent Sol-style Metal prototype against dense MLX SDPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass
from minimax_h3_mlx.algorithm_search.sol_metal import (
    SolMetalConfig,
    sol_metal_attention,
    sol_metal_block_attention,
)
from minimax_h3_mlx.algorithm_search.sol_metal_global import (
    sol_metal_global_pool_block_attention,
)


def _capture(metadata: dict, block: int, evaluation: int) -> dict:
    matches = [
        item
        for item in metadata.get("captures", [])
        if item.get("name") == "attention_qkv"
        and item.get("block") == block
        and item.get("evaluation_index") == evaluation
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matching attention_qkv capture, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--evaluation", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument(
        "--layout", choices=("row", "block", "global_block"), default="global_block"
    )
    parser.add_argument("--simdgroups", type=int, choices=(8, 16), default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-reduction", type=float, default=0.15)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.capture_directory / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"capture metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    record = _capture(metadata, args.block, args.evaluation)
    layout = record.get("metadata", {})
    prefix_rows = int(layout.get("prefix_rows", 0))
    tensors = mx.load(str(args.capture_directory / record["path"]))
    query, key, value = (tensors[f"tensor_{index}"] for index in range(3))
    mx.eval(query, key, value)
    scale = int(query.shape[-1]) ** -0.5
    config = SolMetalConfig(prefix_rows=prefix_rows, beta=args.beta)

    def dense(q, k, v):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

    def candidate(q, k, v):
        if args.layout == "row":
            return sol_metal_attention(q, k, v, scale=scale, config=config)
        if args.layout == "block":
            return sol_metal_block_attention(q, k, v, scale=scale, config=config)
        return sol_metal_global_pool_block_attention(
            q,
            k,
            v,
            scale=scale,
            config=config,
            simdgroups=args.simdgroups,
        )

    transformation = (
        "exact_multimodal_prefix_plus_row_routed_corrected_target_video"
        if args.layout == "row"
        else (
            "exact_multimodal_prefix_plus_fused_pool_block_routed_corrected_target_video"
            if args.layout == "block"
            else "exact_multimodal_prefix_plus_global_pool_block_routed_corrected_target_video"
        )
    )

    result = benchmark_candidate(
        dense,
        candidate,
        (query, key, value),
        candidate_id=(
            f"sol-metal-{args.layout}-route-{int(query.shape[-2])}-rows-"
            f"block-{args.block}-eval-{args.evaluation}"
        ),
        operator="complete_attention_with_pooling",
        algorithm_class=AlgorithmClass.NUMERICALLY_APPROXIMATE,
        parameters={
            **config.to_dict(),
            "layout": args.layout,
            "simdgroups": args.simdgroups,
        },
        transformation=transformation,
        error_budget=args.maximum_relative_l2,
        warmups=args.warmups,
        repetitions=args.repetitions,
        notes=(
            "Research-only Metal prototype. Includes target K/V pooling and dense prefix-query "
            f"work. Selected layout: {args.layout}."
        ),
    )
    routing = None
    if args.layout in {"block", "global_block"}:
        if args.layout == "block":
            _, route_counts = sol_metal_block_attention(
                query,
                key,
                value,
                scale=scale,
                config=config,
                return_route_counts=True,
            )
        else:
            _, route_counts = sol_metal_global_pool_block_attention(
                query,
                key,
                value,
                scale=scale,
                config=config,
                simdgroups=args.simdgroups,
                return_route_counts=True,
            )
        mx.eval(route_counts)
        key_blocks = (int(query.shape[-2]) - prefix_rows + 63) // 64
        total_routes = int(route_counts.size) * key_blocks
        exact_routes = int(mx.sum(route_counts).item())
        routing = {
            "exact_route_blocks": exact_routes,
            "total_route_blocks": total_routes,
            "exact_route_density": exact_routes / total_routes,
            "route_count_shape": list(route_counts.shape),
            "route_tensor_materialized": False,
        }
    reduction = 1.0 - 1.0 / result.speedup
    performance_pass = reduction >= args.minimum_reduction
    quality_pass = result.errors.relative_l2_error <= args.maximum_relative_l2
    payload = {
        "source": {
            "capture_directory": args.capture_directory.name,
            "capture_file": record["path"],
            "block": args.block,
            "evaluation": args.evaluation,
        },
        "layout": layout,
        "routing": routing,
        "candidate": result.to_dict(),
        "gate": {
            "minimum_attention_runtime_reduction": args.minimum_reduction,
            "measured_attention_runtime_reduction": reduction,
            "maximum_relative_l2": args.maximum_relative_l2,
            "performance_pass": performance_pass,
            "quality_pass": quality_pass,
            "end_to_end_generation_approved": performance_pass and quality_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
