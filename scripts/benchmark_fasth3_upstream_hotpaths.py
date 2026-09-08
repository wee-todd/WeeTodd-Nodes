"""Alternate Q8/GEMM and AdaLN A/Bs using real FastH3 weights, synthetic BF16 inputs."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.inference_optimizations import TransientQ8Linear, compiled_scale_add
from minimax_h3_mlx.paged_checkpoint import load_paged_dit


def paired(cases, repeats):
    seconds = {name: [] for name in cases}
    outputs = {}
    peaks = {name: [] for name in cases}
    for iteration in range(repeats + 1):
        for name in list(cases)[:: (-1 if iteration % 2 else 1)]:
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            output = cases[name]()
            mx.eval(output)
            seconds[name].append(time.perf_counter() - started)
            peaks[name].append(mx.get_peak_memory())
            outputs[name] = output
    reference = outputs[next(iter(cases))]
    return {
        "seconds": seconds,
        "warm_median_seconds": {key: statistics.median(v[1:]) for key, v in seconds.items()},
        "phase_peak_bytes": peaks,
        "parity": {
            key: {
                "exact": bool(mx.array_equal(reference, value).item()),
                "max_abs": mx.max(
                    mx.abs(reference.astype(mx.float32) - value.astype(mx.float32))
                ).item(),
            }
            for key, value in outputs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=11594)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.rows < 768 or args.repeats < 2:
        parser.error("Choose a new output, >=768 rows, and >=2 repeats")
    if not (args.transformer / "paged_manifest.json").is_file():
        parser.error("Missing paged checkpoint")
    report = {
        "device": mx.device_info(),
        "input_policy": "synthetic BF16; real block 0 weights",
        "rows": args.rows,
        "cases": {},
    }
    mx.random.seed(19)
    dit = load_paged_dit(args.transformer, window_size=1, prefetch=False)
    try:
        with dit.paged_blocks.window(0) as blocks:
            block = blocks[0]
            for name, layer, rows in (
                ("qkv", block.attn.qkv_proj, args.rows),
                ("out", block.attn.out_proj, args.rows),
                ("ff1_chunk", block.mlp.fc1, 256),
                ("ff2_chunk", block.mlp.fc2, 256),
            ):
                value = mx.random.normal((1, rows, layer.weight.shape[1] * 32 // layer.bits))
                value = value.astype(mx.bfloat16)
                mx.eval(value)
                candidate = TransientQ8Linear(layer)
                report["cases"][name] = paired(
                    {
                        "baseline": lambda layer=layer, value=value: layer(value),
                        "candidate": lambda candidate=candidate, value=value: candidate(value),
                    },
                    args.repeats,
                )
                print(name, json.dumps(report["cases"][name]), flush=True)
            values = [
                mx.random.normal((1, args.rows, dit.config.hidden_size)).astype(mx.bfloat16)
                for _ in range(3)
            ]
            mx.eval(values)
            report["cases"]["adaln"] = paired(
                {
                    "baseline": lambda: values[0] * values[1] + values[2],
                    "candidate": lambda: compiled_scale_add(*values),
                },
                args.repeats,
            )
    finally:
        dit.paged_blocks.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
