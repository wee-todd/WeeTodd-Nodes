"""Compare bounded versus full-depth staged convolution on one saved LTX latent."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx

from ltx25_mlx.components import LTX25VideoDecoder
from ltx25_mlx.conv_vae_acceleration import StagedSmallDepthConv3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24)
    args = parser.parse_args()
    if not args.vae.is_file() or not args.latent.is_file():
        parser.error("Both checkpoint and saved latent must exist")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if any(args.output_directory.iterdir()):
        parser.error("Use an empty output directory")
    latent = mx.load(str(args.latent))["video_latent"]
    mx.eval(latent)
    decoder = LTX25VideoDecoder(args.vae, verbose=False)
    loaded = decoder.load()
    layers = [
        module for _, module in loaded.named_modules() if isinstance(module, StagedSmallDepthConv3d)
    ]
    if not layers:
        raise RuntimeError("No staged Conv3D modules on the final decoder path")
    report = {"latent": str(args.latent), "latent_shape": list(latent.shape), "runs": []}
    try:
        for index, policy in enumerate(("full_depth", "windowed", "windowed", "full_depth")):
            budget = (64 << 20) if policy == "windowed" else (1 << 50)
            for layer in layers:
                layer.max_window_elements = budget
            decoder._conv_acceleration = replace(
                decoder._conv_acceleration, depth_window_elements=budget
            )
            path = args.output_directory / f"{index}_{policy}.mp4"
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            started = time.perf_counter()
            decoder.decode_and_stream(latent, str(path), frame_rate=args.fps)
            mx.synchronize()
            record = {
                "policy": policy,
                "seconds": time.perf_counter() - started,
                "peak_bytes": mx.get_peak_memory(),
                "video": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runtime": decoder.last_decode_report,
            }
            report["runs"].append(record)
            print(json.dumps(record), flush=True)
    finally:
        decoder.free()
    (args.output_directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
