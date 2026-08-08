#!/usr/bin/env python3
"""Benchmark one production-shaped MiniMax H3 transformer block in MLX."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx

HIDDEN = 5_376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14_336
ROTARY_DIM = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7_689)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.rows < 1 or args.iterations < 1:
        parser.error("rows and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    return args


def apply_rotary(value: mx.array, cosine: mx.array, sine: mx.array) -> mx.array:
    rotated, passed = value[..., :ROTARY_DIM], value[..., ROTARY_DIM:]
    first, second = rotated[..., : ROTARY_DIM // 2], rotated[..., ROTARY_DIM // 2 :]
    rotate_half = mx.concatenate([-second, first], axis=-1)
    mixed = rotated * cosine[None, None] + rotate_half * sine[None, None]
    return mx.concatenate([mixed, passed], axis=-1)


def rms_norm(value: mx.array, weight: mx.array) -> mx.array:
    fp32 = value.astype(mx.float32)
    normalized = fp32 * mx.rsqrt(mx.mean(mx.square(fp32), axis=-1, keepdims=True) + 1e-5)
    return normalized.astype(mx.bfloat16) * weight


def main() -> None:
    args = parse_args()
    dtype = mx.bfloat16
    x = mx.full((1, args.rows, HIDDEN), 0.015625, dtype=dtype)
    norm1_weight = mx.ones((HIDDEN,), dtype=dtype)
    norm2_weight = mx.ones((HIDDEN,), dtype=dtype)
    q_norm_weight = mx.ones((HEAD_DIM,), dtype=dtype)
    k_norm_weight = mx.ones((HEAD_DIM,), dtype=dtype)
    qkv_weight = mx.full((3 * INNER, HIDDEN), 1 / 8192, dtype=dtype)
    out_weight = mx.full((HIDDEN, INNER), 1 / 8192, dtype=dtype)
    fc1_weight = mx.full((2 * FFN, HIDDEN), 1 / 8192, dtype=dtype)
    fc2_weight = mx.full((HIDDEN, FFN), 1 / 8192, dtype=dtype)
    indices = mx.arange(args.rows, dtype=mx.int32) % 3
    zeros = mx.zeros((3, HIDDEN), dtype=dtype)
    ones = mx.ones((3, HIDDEN), dtype=dtype)
    cosine = mx.ones((args.rows, ROTARY_DIM), dtype=dtype)
    sine = mx.zeros((args.rows, ROTARY_DIM), dtype=dtype)

    def block(
        value: mx.array,
        norm1: mx.array,
        norm2: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        qkv_w: mx.array,
        out_w: mx.array,
        fc1_w: mx.array,
        fc2_w: mx.array,
        row_indices: mx.array,
        shift_msa: mx.array,
        scale_msa: mx.array,
        gate_msa: mx.array,
        shift_mlp: mx.array,
        scale_mlp: mx.array,
        gate_mlp: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        normalized = rms_norm(value, norm1)
        attention_input = normalized * (1 + scale_msa[row_indices][None])
        attention_input = attention_input + shift_msa[row_indices][None]
        qkv = (attention_input @ qkv_w.T).reshape(1, args.rows, HEADS, 3, HEAD_DIM)
        q = rms_norm(qkv[:, :, :, 0], q_norm).transpose(0, 2, 1, 3)
        k = rms_norm(qkv[:, :, :, 1], k_norm).transpose(0, 2, 1, 3)
        v = qkv[:, :, :, 2].transpose(0, 2, 1, 3)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=HEAD_DIM**-0.5
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(1, args.rows, INNER)
        attention_output = attended @ out_w.T
        residual = value + gate_msa[row_indices][None] * attention_output
        normalized = rms_norm(residual, norm2)
        mlp_input = normalized * (1 + scale_mlp[row_indices][None])
        mlp_input = mlp_input + shift_mlp[row_indices][None]
        fused = mlp_input @ fc1_w.T
        gate, linear_value = fused[..., :FFN], fused[..., FFN:]
        hidden = mx.sigmoid(gate) * gate * linear_value
        return residual + gate_mlp[row_indices][None] * (hidden @ fc2_w.T)

    inputs = (
        x,
        norm1_weight,
        norm2_weight,
        q_norm_weight,
        k_norm_weight,
        qkv_weight,
        out_weight,
        fc1_weight,
        fc2_weight,
        indices,
        zeros,
        zeros,
        ones,
        zeros,
        zeros,
        ones,
        cosine,
        sine,
    )
    mx.eval(*inputs)
    operation = mx.compile(block) if args.compiled else block
    mx.reset_peak_memory()

    def execute() -> tuple[float, mx.array]:
        started = time.perf_counter()
        output = operation(*inputs)
        mx.eval(output)
        return time.perf_counter() - started, output

    first_seconds, output = execute()
    for _ in range(args.warmup):
        _, output = execute()
    samples: list[float] = []
    for _ in range(args.iterations):
        seconds, output = execute()
        samples.append(seconds)

    print(
        json.dumps(
            {
                "backend": "mlx_compiled_block" if args.compiled else "mlx_eager_block",
                "dtype": "bf16",
                "rows": args.rows,
                "compile_plus_first_execution_seconds": first_seconds,
                "warm_median_seconds": statistics.median(samples),
                "warm_samples_seconds": samples,
                "output_sample": float(output[0, 0, 0]),
                "active_memory_bytes": mx.get_active_memory(),
                "peak_memory_bytes": mx.get_peak_memory(),
                "cache_memory_bytes": mx.get_cache_memory(),
                "warmup": args.warmup,
                "iterations": args.iterations,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
