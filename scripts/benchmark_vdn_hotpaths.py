"""Paired, full-shape VDN operator benchmarks using real selected-block checkpoint weights.

Inputs are deterministic synthetic activations, not a rendered clip. This isolates
projection, adapter, and feature costs; whole-workflow timings remain authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.lora import LoRALinear, LoRARequest, apply_paged_loras_to_block
from minimax_h3_mlx.paged_checkpoint import load_paged_dit
from minimax_h3_mlx.projection import mpp_bf16_linear, select_mpp_tile
from minimax_h3_mlx.vdn import VDNLayout, _linear_branch, _temporal_reference
from minimax_h3_mlx.vdn_metal import VDNFeatureKernels, VDNMatrixSolver, temporal_five_tap


def paired(cases, repeats):
    timings = {name: [] for name in cases}
    outputs = {}
    for iteration in range(repeats + 1):
        # Alternate order to reduce warm-up/thermal ordering bias.
        names = list(cases)
        if iteration % 2:
            names.reverse()
        for name in names:
            started = time.perf_counter()
            result = cases[name]()
            mx.eval(result)
            timings[name].append(time.perf_counter() - started)
            outputs[name] = result
    reference = outputs[next(iter(cases))].astype(mx.float32)
    metrics = {}
    for name, output in outputs.items():
        difference = output.astype(mx.float32) - reference
        metrics[name] = {
            "max_absolute_difference": mx.max(mx.abs(difference)).item(),
            "relative_rms_difference": mx.sqrt(
                mx.mean(difference**2) / mx.maximum(mx.mean(reference**2), 1e-20)
            ).item(),
        }
    return {
        "seconds": timings,
        "warm_median_seconds": {name: statistics.median(v[1:]) for name, v in timings.items()},
        "difference_from_first_case": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--vdn-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10250)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 2:
        parser.error("rows must be positive and repeats must be at least two")
    if args.output.exists():
        parser.error("output already exists; use a unique benchmark filename")
    for path in (
        args.transformer / "paged_manifest.json",
        args.vdn_stage / "linear_branch/model.safetensors",
    ):
        if not path.is_file():
            parser.error(f"missing checkpoint input: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "device": mx.device_info(),
        "rows": args.rows,
        "block": args.block,
        "input_policy": "synthetic BF16 activations; real checkpoint weights",
        "lora_candidate": "batched_ab_ordered_updates_benchmark_only",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "projections": {},
    }
    mx.random.seed(42)
    dit = load_paged_dit(args.transformer, window_size=1, prefetch=False)
    pager = dit.paged_blocks
    if not 0 <= args.block < pager.num_blocks:
        parser.error("block index is outside the checkpoint")
    requests = [
        (
            LoRARequest(
                str(args.vdn_stage / f"adapters/{name}/adapter_model.safetensors"),
                qkv_layout="contiguous_qkv",
            ),
            None,
        )
        for name in ("default", "turbo")
    ]
    try:
        with pager.window(args.block) as blocks:
            block = blocks[0]
            apply_paged_loras_to_block(block, args.block, requests, None)
            for label, layer in (
                ("qkv", block.attn.qkv_proj),
                ("attention_output", block.attn.out_proj),
                ("ff1", block.mlp.fc1),
                ("ff2", block.mlp.fc2),
            ):
                base = layer.base if isinstance(layer, LoRALinear) else layer
                bits = getattr(base, "bits", None)
                width = base.weight.shape[1] * 32 // bits if bits else base.weight.shape[1]
                value = mx.random.normal((1, args.rows, width)).astype(mx.bfloat16)
                dense_weight = (
                    mx.dequantize(
                        base.weight,
                        base.scales,
                        base.biases,
                        group_size=base.group_size,
                        bits=base.bits,
                    ).astype(mx.bfloat16)
                    if bits
                    else base.weight
                )
                mx.eval(value, base.parameters(), dense_weight)
                cases = {
                    "current_base": lambda base=base, value=value: base(value),
                    "bf16_mlx": lambda v=value, w=dense_weight: v @ w.T,
                    "bf16_mpp": lambda v=value, w=dense_weight: mpp_bf16_linear(
                        v, w, tile=select_mpp_tile(w)
                    ),
                }
                entry = {
                    "input_shape": list(value.shape),
                    "weight_shape": list(dense_weight.shape),
                    "base_bits": bits or 16,
                    "base": paired(cases, args.repeats),
                }
                if isinstance(layer, LoRALinear):

                    def original(layer=layer, value=value):
                        layer.batched_inputs = False
                        return layer(value)

                    def batched(layer=layer, value=value):
                        layer.batched_inputs = True
                        return layer(value)

                    layer.configure_batched_inputs()
                    mx.eval(layer.parameters())
                    entry["adapters"] = len(layer.adapters)
                    entry["lora"] = paired({"original": original, "batched": batched}, args.repeats)
                report["projections"][label] = entry
                print(label, json.dumps(entry), flush=True)
                del value, dense_weight

            if args.rows == 10250:
                layout = VDNLayout(10250, 926, 37, 252, 12, 21, 0, 512)
                weights = dict(mx.load(str(args.vdn_stage / "linear_branch/model.safetensors")))
                prefix = f"transformer_blocks.{args.block}.attn."
                weights = {
                    name[len(prefix) :]: v for name, v in weights.items() if name.startswith(prefix)
                }
                x = mx.random.normal((1, layout.sequence, dit.config.hidden_size)).astype(
                    mx.bfloat16
                )
                raw = tuple(
                    mx.random.normal((1, layout.sequence, 56, 128)).astype(mx.bfloat16)
                    for _ in range(3)
                )
                spatial = mx.random.normal((35, 12, 21, 7168)).astype(mx.bfloat16)
                temporal_weight = weights["linear_attention.short_conv.k_tm.weight"]
                mx.eval(x, raw, spatial, weights)
                report["temporal"] = paired(
                    {
                        "original": lambda: _temporal_reference(spatial, temporal_weight),
                        "fused": lambda: temporal_five_tap(spatial, temporal_weight),
                    },
                    args.repeats,
                )
                solver, features = VDNMatrixSolver(), VDNFeatureKernels()
                report["linear_branch"] = paired(
                    {
                        "original": lambda: _linear_branch(x, raw, weights, layout, solver=solver),
                        "fused": lambda: _linear_branch(
                            x, raw, weights, layout, solver=solver, feature_kernels=features
                        ),
                    },
                    args.repeats,
                )
                report["feature_kernels"] = features.report()
                print(
                    "VDN",
                    json.dumps({k: report[k] for k in ("temporal", "linear_branch")}),
                    flush=True,
                )
    finally:
        pager.close()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
