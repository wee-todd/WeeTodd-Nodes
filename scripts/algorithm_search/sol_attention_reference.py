#!/usr/bin/env python3
"""Evaluate the slow Sol-style MLX reference on one bounded real-QKV capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics
from minimax_h3_mlx.algorithm_search.sol_attention import (
    SolReferenceConfig,
    dense_attention_reference,
    sol_reference_attention,
)


def _capture_record(metadata: dict, block: int, evaluation: int) -> dict:
    matches = [
        item
        for item in metadata.get("captures", [])
        if item.get("name") == "attention_qkv"
        and item.get("block") == block
        and item.get("evaluation_index") == evaluation
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one attention_qkv capture for "
            f"block {block}, evaluation {evaluation}; found {len(matches)}"
        )
    return matches[0]


def _even_query_blocks(total: int, count: int) -> tuple[int, ...]:
    if min(total, count) < 1:
        raise ValueError("query block counts must be positive")
    if count >= total:
        return tuple(range(total))
    return tuple(sorted(set(int(round(value)) for value in np.linspace(0, total - 1, count))))


def _query_rows(
    query: mx.array,
    *,
    prefix_rows: int,
    block_size: int,
    blocks: tuple[int, ...],
) -> mx.array:
    pieces = []
    for block in blocks:
        start = prefix_rows + block * block_size
        stop = min(start + block_size, int(query.shape[-2]))
        pieces.append(query[..., start:stop, :])
    return mx.concatenate(pieces, axis=-2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_directory", type=Path)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--evaluation", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--query-blocks", type=int, default=4)
    parser.add_argument("--beta", type=float, action="append", default=[])
    parser.add_argument(
        "--threshold-mode",
        choices=("diagonal", "exact"),
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.capture_directory / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"capture metadata does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    record = _capture_record(metadata, args.block, args.evaluation)
    layout = record.get("metadata", {})
    prefix_rows = int(layout.get("prefix_rows", 0))
    capture_path = args.capture_directory / record["path"]
    tensors = mx.load(str(capture_path))
    query, key, value = (tensors[f"tensor_{index}"] for index in range(3))
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("captured attention Q/K/V shapes do not match")
    if prefix_rows <= 0:
        raise ValueError("attention capture has no exact-prefix metadata")
    target_rows = int(query.shape[-2]) - prefix_rows
    target_blocks = (target_rows + args.block_size - 1) // args.block_size
    selected_query_blocks = _even_query_blocks(target_blocks, args.query_blocks)
    query_rows = _query_rows(
        query,
        prefix_rows=prefix_rows,
        block_size=args.block_size,
        blocks=selected_query_blocks,
    )
    scale = int(query.shape[-1]) ** -0.5
    dense = dense_attention_reference(query_rows, key, value, scale=scale)
    mx.eval(dense)

    betas = args.beta or [0.5, 0.75, 1.0]
    threshold_modes = args.threshold_mode or ["diagonal"]
    experiments = []
    for threshold_mode in threshold_modes:
        for beta in betas:
            variants = {}
            for corrected in (False, True):
                candidate, telemetry = sol_reference_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    config=SolReferenceConfig(
                        prefix_rows=prefix_rows,
                        block_size=args.block_size,
                        beta=beta,
                        threshold_mode=threshold_mode,
                        approximate_correction=corrected,
                        target_query_blocks=selected_query_blocks,
                    ),
                )
                mx.eval(candidate)
                variants["corrected" if corrected else "exact_only_drop"] = {
                    "telemetry": telemetry.to_dict(),
                    "errors": numerical_metrics(dense, candidate).__dict__,
                }
            experiments.append(
                {
                    "beta": beta,
                    "threshold_mode": threshold_mode,
                    "variants": variants,
                    "correction_improves_relative_l2": (
                        variants["corrected"]["errors"]["relative_l2_error"]
                        < variants["exact_only_drop"]["errors"]["relative_l2_error"]
                    ),
                }
            )

    result = {
        "source": {
            "capture_directory": args.capture_directory.name,
            "capture_file": capture_path.name,
            "block": args.block,
            "evaluation": args.evaluation,
            "captured_heads": layout.get("attention_heads", []),
        },
        "layout": layout,
        "geometry": {
            "shape": list(query.shape),
            "block_size": args.block_size,
            "target_key_blocks": target_blocks,
            "sampled_target_query_blocks": list(selected_query_blocks),
            "sampled_target_query_rows": int(query_rows.shape[-2]),
        },
        "audiovisual_contract": {
            "complete_multimodal_prefix_exact": True,
            "prefix_queries_dense": True,
            "audio_rows_exact": int(layout.get("audio_rows", 0)),
            "target_video_only_approximation": True,
        },
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
