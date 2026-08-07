#!/usr/bin/env python3
"""Evaluate cross-timestep corrections independently by packed modality."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics
from minimax_h3_mlx.algorithm_search.low_rank import (
    fit_reduced_map,
    randomized_right_basis,
    supervised_input_basis,
)
from minimax_h3_mlx.algorithm_search.modalities import t2va_modality_rows


def _load(path: Path, key: str = "tensor_0") -> mx.array:
    tensors = mx.load(str(path))
    if key not in tensors:
        raise KeyError(f"tensor key not found: {key}")
    value = tensors[key]
    return value[0] if value.ndim == 3 and value.shape[0] == 1 else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_previous", type=Path)
    parser.add_argument("output_previous", type=Path)
    parser.add_argument("input_current", type=Path)
    parser.add_argument("output_current", type=Path)
    parser.add_argument("--transfer-input-previous", type=Path)
    parser.add_argument("--transfer-output-previous", type=Path)
    parser.add_argument("--transfer-input-current", type=Path)
    parser.add_argument("--transfer-output-current", type=Path)
    parser.add_argument("--text-rows", type=int, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/algorithm_search/cross_timestep_modalities.local.json"),
    )
    args = parser.parse_args()
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("fit fraction must be between zero and one")

    x0 = _load(args.input_previous).astype(mx.float32)
    y0 = _load(args.output_previous).astype(mx.float32)
    x1 = _load(args.input_current).astype(mx.float32)
    y1 = _load(args.output_current).astype(mx.float32)
    if not (x0.shape == y0.shape == x1.shape == y1.shape):
        raise ValueError("paired capture tensors must have identical shapes")
    transfer_paths = (
        args.transfer_input_previous,
        args.transfer_output_previous,
        args.transfer_input_current,
        args.transfer_output_current,
    )
    if any(path is not None for path in transfer_paths) and not all(
        path is not None for path in transfer_paths
    ):
        raise ValueError("all four transfer capture paths must be supplied together")
    transfer = None
    if all(path is not None for path in transfer_paths):
        transfer = tuple(_load(path).astype(mx.float32) for path in transfer_paths)
        if any(value.shape != x0.shape for value in transfer):
            raise ValueError("transfer capture tensors must match the fitting pair shape")
    layout = t2va_modality_rows(
        total_rows=int(x0.shape[0]),
        text_rows=args.text_rows,
        duration_seconds=args.duration,
        height=args.height,
        width=args.width,
    )
    payload = {
        "algorithm_class": "generatively_approximate",
        "shape": list(x0.shape),
        "row_counts": layout.counts,
        "modalities": {},
    }
    transfer_hybrid_targets = []
    transfer_hybrid_predictions = []
    full_hybrid_targets = []
    full_hybrid_predictions = []

    for modality, region in (
        ("text", layout.text),
        ("audio", layout.audio),
        ("video", layout.video),
    ):
        region_x0 = x0[region]
        region_y0 = y0[region]
        region_x1 = x1[region]
        region_y1 = y1[region]
        transfer_region = (
            None
            if transfer is None
            else tuple(value[region] for value in transfer)
        )
        row_count = int(region_x0.shape[0])
        fit_rows = max(1, min(row_count - 1, int(row_count * args.fit_fraction)))
        fit = slice(0, fit_rows)
        test = slice(fit_rows, row_count)
        dx_fit_mx = region_x1[fit] - region_x0[fit]
        dy_fit_mx = region_y1[fit] - region_y0[fit]
        epsilon = mx.array(1e-12, dtype=mx.float32)
        channel = mx.sum(dx_fit_mx * dy_fit_mx, axis=0) / mx.maximum(
            mx.sum(dx_fit_mx * dx_fit_mx, axis=0), epsilon
        )
        bias = mx.mean(dy_fit_mx - channel * dx_fit_mx, axis=0)
        residual_fit_mx = dy_fit_mx - channel * dx_fit_mx - bias
        mx.eval(channel, bias, residual_fit_mx)
        dx_test = region_x1[test] - region_x0[test]
        affine_test = region_y0[test] + channel * dx_test + bias
        affine_metrics = asdict(numerical_metrics(region_y1[test], affine_test))
        transfer_affine_metrics = None
        transfer_dx_test = None
        transfer_affine_test = None
        transfer_target = None
        if transfer_region is not None:
            transfer_x0, transfer_y0, transfer_x1, transfer_y1 = transfer_region
            transfer_dx_test = transfer_x1[test] - transfer_x0[test]
            transfer_affine_test = (
                transfer_y0[test] + channel * transfer_dx_test + bias
            )
            transfer_target = transfer_y1[test]
            transfer_affine_metrics = asdict(
                numerical_metrics(transfer_target, transfer_affine_test)
            )
            if modality in {"text", "audio"}:
                transfer_hybrid_targets.append(transfer_target)
                transfer_hybrid_predictions.append(transfer_affine_test)
                full_hybrid_targets.append(transfer_target)
                full_hybrid_predictions.append(transfer_affine_test)
            else:
                full_hybrid_targets.append(transfer_target)
                full_hybrid_predictions.append(transfer_target)
        dx_fit = np.asarray(dx_fit_mx)
        residual_fit = np.asarray(residual_fit_mx)
        valid_ranks = [rank for rank in args.ranks if rank <= min(residual_fit.shape)]
        rank_results = []
        if valid_ranks:
            max_rank = max(valid_ranks)
            target_basis_full = randomized_right_basis(
                residual_fit,
                max_rank,
                oversample=args.oversample,
                seed=args.seed + {"text": 1, "audio": 2, "video": 3}[modality],
            )
            input_basis_full = supervised_input_basis(
                dx_fit, residual_fit, target_basis_full
            )
            for rank in valid_ranks:
                input_basis_np = input_basis_full[:, :rank]
                target_basis_np = target_basis_full[:, :rank]
                mapping_np = fit_reduced_map(
                    dx_fit, residual_fit, input_basis_np, target_basis_np
                )
                input_basis = mx.array(input_basis_np).astype(mx.bfloat16)
                target_basis = mx.array(target_basis_np).astype(mx.bfloat16)
                mapping = mx.array(mapping_np).astype(mx.bfloat16)
                correction = ((dx_test.astype(mx.bfloat16) @ input_basis) @ mapping) @ (
                    target_basis.T
                )
                predicted = affine_test + correction
                rank_result = {
                    "rank": rank,
                    "metrics": asdict(numerical_metrics(region_y1[test], predicted)),
                }
                if transfer_dx_test is not None:
                    transfer_correction = (
                        (transfer_dx_test.astype(mx.bfloat16) @ input_basis) @ mapping
                    ) @ target_basis.T
                    transfer_predicted = transfer_affine_test + transfer_correction
                    rank_result["transfer_metrics"] = asdict(
                        numerical_metrics(transfer_target, transfer_predicted)
                    )
                rank_results.append(rank_result)
        payload["modalities"][modality] = {
            "rows": row_count,
            "fit_rows": fit_rows,
            "test_rows": row_count - fit_rows,
            "affine_metrics": affine_metrics,
            "transfer_affine_metrics": transfer_affine_metrics,
            "low_rank": rank_results,
        }

    if transfer_hybrid_targets:
        payload["text_audio_affine_transfer_metrics"] = asdict(
            numerical_metrics(transfer_hybrid_targets, transfer_hybrid_predictions)
        )
        payload["full_hybrid_transfer_metrics"] = asdict(
            numerical_metrics(full_hybrid_targets, full_hybrid_predictions)
        )

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
