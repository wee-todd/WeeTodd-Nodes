#!/usr/bin/env python3
"""Benchmark exact H3 video-VAE decode batching with synthetic five-second latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.load import load_compact_video_vae
from minimax_h3_mlx.packing import FPS, align_num_frames, video_latent_num_frames


def _digest(value: mx.array) -> str:
    host = np.asarray(value)
    return hashlib.sha256(memoryview(np.ascontiguousarray(host))).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.width % 32 or args.height % 32:
        raise ValueError("width and height must be divisible by 32")
    if min(args.width, args.height, args.repetitions, *args.batches) < 1:
        raise ValueError("dimensions, batches, and repetitions must be positive")

    frames = align_num_frames(round(args.seconds * FPS))
    latent_frames = video_latent_num_frames(frames)
    model = load_compact_video_vae(args.checkpoint)
    ratio = model.config.spatial_compression_ratio
    mx.random.seed(args.seed)
    latent = mx.random.normal(
        (
            1,
            model.config.latent_channels,
            latent_frames,
            args.height // ratio,
            args.width // ratio,
        )
    ).astype(mx.float32)
    mx.eval(latent)

    measurements = []
    reference_digest = None
    for batch in args.batches:
        model.decode_batch = batch
        samples = []
        peaks = []
        digest = None
        for _ in range(args.repetitions):
            mx.clear_cache()
            mx.reset_peak_memory()
            started = time.perf_counter()
            decoded = model.decode(latent)
            mx.eval(decoded)
            samples.append(time.perf_counter() - started)
            peaks.append(int(mx.get_peak_memory()))
            digest = _digest(decoded)
            del decoded
            mx.clear_cache()
        if reference_digest is None:
            reference_digest = digest
        measurements.append(
            {
                "decode_batch": batch,
                "samples_seconds": samples,
                "median_seconds": statistics.median(samples),
                "peak_memory_bytes": max(peaks),
                "sha256": digest,
                "bit_exact_to_first": digest == reference_digest,
            }
        )

    payload = {
        "checkpoint_file": args.checkpoint.name,
        "geometry": {
            "width": args.width,
            "height": args.height,
            "frames": frames,
            "latent_frames": latent_frames,
            "latent_shape": list(latent.shape),
        },
        "dtype": str(latent.dtype),
        "mlx_device": mx.device_info(),
        "measurements": measurements,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
