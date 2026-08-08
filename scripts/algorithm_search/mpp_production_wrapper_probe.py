#!/usr/bin/env python3
"""Load a real H3 transformer and exercise the production MPP wrapper path."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.projection import (
    MPPLinear,
    configure_projection_backend,
    mpp_runtime_status,
    reset_mpp_runtime_status,
)


def _capture(path: Path) -> mx.array:
    value = mx.load(str(path)).get("tensor_0")
    if value is None:
        raise ValueError(f"capture has no tensor_0: {path}")
    return value.astype(mx.bfloat16)


def _check_layer(layer: MPPLinear, source: mx.array) -> bool:
    first = layer(source)
    mx.eval(first)
    candidate = layer(source)
    reference = layer.base(source)
    mx.eval(candidate, reference)
    return bool(mx.array_equal(candidate, reference).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--block", type=int, default=24)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint is not a file: {args.checkpoint}")
    if not args.capture_root.is_dir():
        parser.error(f"capture root is not a directory: {args.capture_root}")

    reset_mpp_runtime_status()
    load_started = time.perf_counter()
    dit = load_dit(args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    report = configure_projection_backend(dit, "mpp_experimental")
    block = dit.blocks[args.block]
    layers = (
        block.attn.qkv_proj,
        block.attn.out_proj,
        block.mlp.fc1,
        block.mlp.fc2,
    )
    if not all(isinstance(layer, MPPLinear) for layer in layers):
        raise RuntimeError("production backend did not wrap all four BF16 block projections")

    normalized = _capture(
        args.capture_root / f"0001_normalized_block_input_block_{args.block}.safetensors"
    )
    mlp_input = _capture(
        args.capture_root / f"0004_mlp_input_block_{args.block}.safetensors"
    )
    qkv = block.attn.qkv_proj(normalized)
    mx.eval(qkv)
    attention_projection_input = qkv[..., : block.attn.heads * block.attn.head_dim]
    fc1 = block.mlp.fc1(mlp_input)
    mx.eval(fc1)
    ffn = block.mlp._ffn
    fc2_input = nn.silu(fc1[..., :ffn]) * fc1[..., ffn:]
    mx.eval(attention_projection_input, fc2_input)

    exact = {
        "qkv": _check_layer(block.attn.qkv_proj, normalized),
        "attention_output": _check_layer(
            block.attn.out_proj, attention_projection_input
        ),
        "fc1": _check_layer(block.mlp.fc1, mlp_input),
        "fc2": _check_layer(block.mlp.fc2, fc2_input),
    }
    if not all(exact.values()):
        raise RuntimeError(f"production MPP wrapper mismatch: {exact}")

    result = {
        "load_seconds": load_seconds,
        "backend": report.to_dict(),
        "runtime": mpp_runtime_status(),
        "block": args.block,
        "rows": int(normalized.shape[-2]),
        "exact": exact,
        "active_memory_bytes_before_release": int(mx.get_active_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
    del qkv, attention_projection_input, fc1, fc2_input, normalized, mlp_input, layers, block, dit
    gc.collect()
    mx.clear_cache()
    result["active_memory_bytes_after_release"] = int(mx.get_active_memory())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
