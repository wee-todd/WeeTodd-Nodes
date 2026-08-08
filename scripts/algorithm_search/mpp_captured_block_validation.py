#!/usr/bin/env python3
"""Validate MPP projections with captured H3 activations and real checkpoint weights."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate, numerical_metrics
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass
from minimax_h3_mlx.dit import apply_rotary
from minimax_h3_mlx.projection import MPPTile, mpp_bf16_linear


def _captured_tensor(path: Path) -> mx.array:
    tensors = mx.load(str(path))
    value = tensors.get("tensor_0")
    if value is None:
        raise ValueError(f"capture has no tensor_0: {path}")
    return value.astype(mx.bfloat16)


def _weight(tensors: dict[str, mx.array], key: str) -> mx.array:
    value = tensors.get(key)
    if value is None:
        raise KeyError(f"checkpoint has no {key}")
    if value.dtype != mx.bfloat16:
        raise TypeError(f"checkpoint weight {key} must be BF16, got {value.dtype}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=96)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/algorithm_search/mpp_captured_block.local.jsonl"),
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint is not a file: {args.checkpoint}")
    if not args.capture_root.is_dir():
        parser.error(f"capture root is not a directory: {args.capture_root}")

    normalized_input = _captured_tensor(
        args.capture_root / f"0001_normalized_block_input_block_{args.block}.safetensors"
    )
    captured_q = _captured_tensor(
        args.capture_root / f"0002_q_output_block_{args.block}.safetensors"
    )
    mlp_input = _captured_tensor(
        args.capture_root / f"0004_mlp_input_block_{args.block}.safetensors"
    )
    batch, rows, hidden = normalized_input.shape
    inner = args.heads * args.head_dim
    if mlp_input.shape != normalized_input.shape:
        raise ValueError("captured normalized and MLP inputs must have matching shapes")
    if captured_q.shape != (batch, rows, args.heads, args.head_dim):
        raise ValueError("captured Q tensor does not match the requested attention dimensions")

    checkpoint = mx.load(str(args.checkpoint))
    prefix = f"blocks.{args.block}"
    qkv_weight = _weight(checkpoint, f"{prefix}.attn.qkv_proj.weight")
    out_weight = _weight(checkpoint, f"{prefix}.attn.out_proj.weight")
    q_norm_weight = _weight(checkpoint, f"{prefix}.attn.q_norm.weight")
    k_norm_weight = _weight(checkpoint, f"{prefix}.attn.k_norm.weight")
    fc1_weight = _weight(checkpoint, f"{prefix}.mlp.fc1.weight")
    fc2_weight = _weight(checkpoint, f"{prefix}.mlp.fc2.weight")
    del checkpoint
    gc.collect()
    mx.eval(
        normalized_input,
        captured_q,
        mlp_input,
        qkv_weight,
        out_weight,
        q_norm_weight,
        k_norm_weight,
        fc1_weight,
        fc2_weight,
    )

    expected_shapes = {
        "qkv": (3 * inner, hidden),
        "out": (hidden, inner),
        "fc1": (2 * fc2_weight.shape[1], hidden),
        "fc2": (hidden, fc2_weight.shape[1]),
    }
    actual_shapes = {
        "qkv": qkv_weight.shape,
        "out": out_weight.shape,
        "fc1": fc1_weight.shape,
        "fc2": fc2_weight.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"unexpected block projection shapes: {actual_shapes}")

    baseline_qkv = (normalized_input @ qkv_weight.T).reshape(
        batch, rows, args.heads, 3, args.head_dim
    )
    mpp_qkv = mpp_bf16_linear(normalized_input, qkv_weight).reshape(
        batch, rows, args.heads, 3, args.head_dim
    )
    mx.eval(baseline_qkv, mpp_qkv)
    checkpoint_capture_metrics = numerical_metrics(captured_q, baseline_qkv[:, :, :, 0])
    mpp_capture_metrics = numerical_metrics(captured_q, mpp_qkv[:, :, :, 0])
    if checkpoint_capture_metrics.max_absolute_error != 0.0:
        raise RuntimeError("checkpoint does not reproduce the captured Q tensor")
    if mpp_capture_metrics.max_absolute_error != 0.0:
        raise RuntimeError("MPP does not reproduce the captured Q tensor")
    del baseline_qkv, mpp_qkv
    mx.clear_cache()

    mx.random.seed(args.seed)
    angles = mx.random.normal((rows, args.rotary_dim)).astype(mx.float32)
    rotary = (mx.cos(angles), mx.sin(angles))
    mx.eval(*rotary)

    def rms_norm(value: mx.array, weight: mx.array) -> mx.array:
        scale = mx.rsqrt(
            mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6
        )
        return (value * scale).astype(value.dtype) * weight

    def attention_with(
        value: mx.array,
        projection,
    ) -> mx.array:
        packed = projection(value, qkv_weight).reshape(
            batch, rows, args.heads, 3, args.head_dim
        )
        query, key, val = (packed[:, :, :, index] for index in range(3))
        query = apply_rotary(rms_norm(query, q_norm_weight).transpose(0, 2, 1, 3), *rotary)
        key = apply_rotary(rms_norm(key, k_norm_weight).transpose(0, 2, 1, 3), *rotary)
        val = val.transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            val,
            scale=args.head_dim**-0.5,
        )
        projected_input = attended.transpose(0, 2, 1, 3).reshape(batch, rows, inner)
        return projection(projected_input.astype(value.dtype), out_weight)

    def mlx_projection(value: mx.array, weight: mx.array) -> mx.array:
        return value @ weight.T

    def mpp_projection(value: mx.array, weight: mx.array) -> mx.array:
        tile = MPPTile(64, 128, 8) if weight.shape == fc2_weight.shape else MPPTile()
        return mpp_bf16_linear(value, weight, tile=tile)

    def baseline_attention(value: mx.array) -> mx.array:
        return attention_with(value, mlx_projection)

    def candidate_attention(value: mx.array) -> mx.array:
        return attention_with(value, mpp_projection)

    ffn = fc2_weight.shape[1]

    def feed_forward_with(value: mx.array, projection) -> mx.array:
        fused = projection(value, fc1_weight)
        activated = nn.silu(fused[..., :ffn]) * fused[..., ffn:]
        return projection(activated, fc2_weight)

    def baseline_feed_forward(value: mx.array) -> mx.array:
        return feed_forward_with(value, mlx_projection)

    def candidate_feed_forward(value: mx.array) -> mx.array:
        return feed_forward_with(value, mpp_projection)

    attention_result = benchmark_candidate(
        baseline_attention,
        candidate_attention,
        (normalized_input,),
        candidate_id=f"mpp-captured-attention-block-{args.block}-{rows}-rows",
        operator="complete_attention",
        algorithm_class=AlgorithmClass.EXACT,
        parameters={"block": args.block, "rows": rows, "projection_backend": "mpp"},
        transformation="mpp_bf16_qkv_and_output_projection",
        error_budget=0.0,
        warmups=args.warmups,
        repetitions=args.repetitions,
        notes="Real captured input and real BF16 weights; deterministic production-shape rotary.",
    )
    feed_forward_result = benchmark_candidate(
        baseline_feed_forward,
        candidate_feed_forward,
        (mlp_input,),
        candidate_id=f"mpp-captured-mlp-block-{args.block}-{rows}-rows",
        operator="complete_feed_forward",
        algorithm_class=AlgorithmClass.EXACT,
        parameters={"block": args.block, "rows": rows, "projection_backend": "mpp"},
        transformation="mpp_bf16_fc1_and_fc2_projection",
        error_budget=0.0,
        warmups=args.warmups,
        repetitions=args.repetitions,
        notes="Real captured MLP input and real BF16 weights.",
    )

    store = ExperimentStore(args.results)
    context = {
        "checkpoint_file": args.checkpoint.name,
        "capture_directory": args.capture_root.name,
        "seed": args.seed,
    }
    store.append(attention_result, context=context)
    store.append(feed_forward_result, context=context)
    report = {
        "checkpoint_capture_q": checkpoint_capture_metrics.__dict__,
        "mpp_capture_q": mpp_capture_metrics.__dict__,
        "attention": attention_result.to_dict(),
        "feed_forward": feed_forward_result.to_dict(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
