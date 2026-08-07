#!/usr/bin/env python3
"""Compare eager and compiled exact H3 attention at a captured sequence shape."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate
from minimax_h3_mlx.algorithm_search.results import ExperimentStore
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass


def _load_weight(tensors: dict[str, mx.array], suffix: str) -> mx.array:
    matches = [name for name in tensors if name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one weight ending in {suffix!r}, found {len(matches)}")
    return tensors[matches[0]].astype(mx.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("normalized_input", type=Path)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/algorithm_search/results.local.jsonl"),
    )
    args = parser.parse_args()

    captured = mx.load(str(args.normalized_input))
    x = captured.get("tensor_0")
    if x is None or x.ndim != 3:
        raise ValueError("normalized input must contain rank-3 tensor_0")
    x = x.astype(mx.bfloat16)
    batch, sequence, _ = x.shape

    tensors = mx.load(str(args.checkpoint))
    prefix = f"blocks.{args.block}.attn"
    qkv_weight = _load_weight(tensors, f"{prefix}.qkv_proj.weight")
    out_weight = _load_weight(tensors, f"{prefix}.out_proj.weight")
    q_norm_weight = _load_weight(tensors, f"{prefix}.q_norm.weight")
    k_norm_weight = _load_weight(tensors, f"{prefix}.k_norm.weight")
    del tensors, captured
    gc.collect()

    inner = args.heads * args.head_dim
    if qkv_weight.shape != (3 * inner, x.shape[-1]):
        raise ValueError(f"unexpected QKV weight shape: {qkv_weight.shape}")
    if out_weight.shape != (x.shape[-1], inner):
        raise ValueError(f"unexpected output weight shape: {out_weight.shape}")

    mx.random.seed(args.seed)
    angles = mx.random.normal((sequence, args.rotary_dim)).astype(mx.float32)
    rotary_cos = mx.cos(angles)
    rotary_sin = mx.sin(angles)
    mx.eval(x, qkv_weight, out_weight, q_norm_weight, k_norm_weight, rotary_cos, rotary_sin)

    def rms_norm(value: mx.array, weight: mx.array) -> mx.array:
        scale = mx.rsqrt(
            mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6
        )
        return (value * scale).astype(value.dtype) * weight

    def rotary(value: mx.array) -> mx.array:
        rotated, passthrough = value[..., : args.rotary_dim], value[..., args.rotary_dim :]
        half = args.rotary_dim // 2
        first, second = rotated[..., :half], rotated[..., half:]
        quarter_turn = mx.concatenate([-second, first], axis=-1)
        cos = rotary_cos.astype(value.dtype)[None, None]
        sin = rotary_sin.astype(value.dtype)[None, None]
        return mx.concatenate([rotated * cos + quarter_turn * sin, passthrough], axis=-1)

    def attention(
        value: mx.array,
        qkv: mx.array,
        output: mx.array,
        q_weight: mx.array,
        k_weight: mx.array,
    ) -> mx.array:
        packed = (value @ qkv.T).reshape(
            batch, sequence, args.heads, 3, args.head_dim
        )
        query, key, val = (packed[:, :, :, index] for index in range(3))
        query = rotary(rms_norm(query, q_weight).transpose(0, 2, 1, 3))
        key = rotary(rms_norm(key, k_weight).transpose(0, 2, 1, 3))
        val = val.transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            query, key, val, scale=args.head_dim**-0.5
        )
        rows = attended.transpose(0, 2, 1, 3).reshape(batch, sequence, inner)
        return rows.astype(value.dtype) @ output.T

    compiled_attention = mx.compile(attention)
    result = benchmark_candidate(
        attention,
        compiled_attention,
        (x, qkv_weight, out_weight, q_norm_weight, k_norm_weight),
        candidate_id=f"compiled-attention-block-{args.block}-{sequence}-rows",
        operator="complete_attention",
        algorithm_class=AlgorithmClass.EXACT,
        parameters={
            "block": args.block,
            "rows": sequence,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "rotary_dim": args.rotary_dim,
        },
        transformation="mlx_compile_complete_attention",
        error_budget=5e-4,
        warmups=args.warmups,
        repetitions=args.repetitions,
        notes=(
            "Uses real captured BF16 input and real projection/norm weights. Rotary angles are "
            "deterministic synthetic values with the production shape."
        ),
    )
    ExperimentStore(args.results).append(
        result,
        context={"seed": args.seed, "geometry": f"rows_{sequence}"},
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
