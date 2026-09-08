"""Screen VDN inference kernels using full-shape synthetic activations and real weights."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from minimax_h3_mlx.vdn import VDNLayout, _linear_branch
from minimax_h3_mlx.vdn_inference import VDNInferenceKernels
from minimax_h3_mlx.vdn_metal import VDNFeatureKernels, VDNMatrixSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    checkpoint = args.stage / "linear_branch/model.safetensors"
    if not checkpoint.is_file() or args.output.exists() or args.repeats < 2:
        parser.error("Checkpoint must exist, output must be new, and repeats must be at least two.")
    layout = VDNLayout(10250, 926, 37, 252, 12, 21, 0, 512)
    prefix = "transformer_blocks.0.attn."
    weights = {
        k[len(prefix) :]: v for k, v in mx.load(str(checkpoint)).items() if k.startswith(prefix)
    }
    mx.random.seed(20260903)
    x = mx.random.normal((1, 10250, 5376)).astype(mx.bfloat16)
    raw = tuple(mx.random.normal((1, 10250, 56, 128)).astype(mx.bfloat16) for _ in range(3))
    mx.eval(x, raw, weights)
    solver, features = VDNMatrixSolver(), VDNFeatureKernels()
    inference = VDNInferenceKernels(mpp=True)
    cases = {"reference": None, "verified": inference}
    timings = {k: [] for k in cases}
    outputs = {}
    for trial in range(args.repeats + 1):
        for name in list(cases)[:: 1 if trial % 2 == 0 else -1]:
            start = time.perf_counter()
            out = _linear_branch(
                x,
                raw,
                weights,
                layout,
                solver=solver,
                feature_kernels=features,
                inference=cases[name],
            )
            mx.eval(out)
            elapsed = time.perf_counter() - start
            if trial:
                timings[name].append(elapsed)
            outputs[name] = out
            print(f"{trial} {name}: {elapsed:.6f}s", flush=True)
    report = {
        "scope": "synthetic activations; real block-0 weights; not a full render",
        "warm_median_seconds": {k: statistics.median(v) for k, v in timings.items()},
        "bitwise_equal": bool(mx.array_equal(outputs["reference"], outputs["verified"]).item()),
        "inference": inference.report(),
        "solver": solver.report(),
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
