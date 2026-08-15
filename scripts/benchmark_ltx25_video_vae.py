#!/usr/bin/env python3
"""Generate one LTX 2.5 latent and compare convolutional and diffusion VAE decoding."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--conv-vae", type=Path, required=True)
    parser.add_argument("--diffusion-vae", type=Path, required=True)
    parser.add_argument("--audio-vae", type=Path, required=True)
    parser.add_argument("--spatial-upscaler", type=Path, required=True)
    parser.add_argument("--dfr-detailing-lora", type=Path)
    parser.add_argument(
        "--reuse-latent",
        type=Path,
        help="Decode a previously saved benchmark latent instead of sampling again.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--diffvae-optimization",
        choices=(
            "combined",
            "deferred_stage4",
            "stage4_width_tiles",
            "metal_na3d_experimental",
            "metal_na3d_query_tiled_experimental",
        ),
        default="combined",
    )
    parser.add_argument("--diffvae-query-chunk-size", type=int, default=512)
    parser.add_argument(
        "--decode-mode",
        choices=("both", "conv", "diffusion"),
        default="both",
        help="Decode only the requested VAE path when reusing a latent.",
    )
    return parser


def _require_paths(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if (
            isinstance(value, Path)
            and name != "output_prefix"
            and value is not None
            and not value.exists()
        ):
            raise FileNotFoundError(f"Missing --{name.replace('_', '-')}: {value}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)


def _decode_targets(mode: str, conv_vae: Path, diffusion_vae: Path) -> tuple[tuple[str, Path], ...]:
    targets = {
        "both": (("conv", conv_vae), ("diffusion", diffusion_vae)),
        "conv": (("conv", conv_vae),),
        "diffusion": (("diffusion", diffusion_vae),),
    }
    try:
        return targets[mode]
    except KeyError as error:
        raise ValueError(f"Unknown decode mode: {mode}") from error


def main() -> int:
    args = _parser().parse_args()
    _require_paths(args)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt file is empty.")

    import mlx.core as mx
    from ltx_pipelines_mlx.utils._orchestration import save_waveform

    from ltx25_mlx.components import LTX25VideoDecoder
    from ltx25_mlx.pipeline import LTX25DistilledPipeline

    outputs = {
        "conv": args.output_prefix.with_name(f"{args.output_prefix.name}_Conv.mp4"),
        "diffusion": args.output_prefix.with_name(f"{args.output_prefix.name}_Diffusion.mp4"),
        "latent": args.output_prefix.with_name(f"{args.output_prefix.name}_Latents.safetensors"),
        "audio": args.output_prefix.with_name(f".{args.output_prefix.name}_audio.wav"),
        "report": args.output_prefix.with_name(f"{args.output_prefix.name}_Report.json"),
    }
    total_started = time.perf_counter()
    if args.reuse_latent is None:
        pipeline = LTX25DistilledPipeline(
            transformer_path=str(args.transformer),
            text_encoder_path=str(args.text_encoder),
            video_vae_path=str(args.conv_vae),
            audio_vae_path=str(args.audio_vae),
            spatial_upscaler_path=str(args.spatial_upscaler),
            low_memory=True,
            low_ram_streaming=True,
            feed_forward_backend="mpp_bf16",
            verbose=True,
        )
        mx.reset_peak_memory()
        latent_started = time.perf_counter()
        video_latent, audio_latent = pipeline.generate_two_stage(
            prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.fps,
            seed=args.seed,
            stage1_steps=8,
            stage2_steps=3,
            prompt_context="official_1024",
            dfr_enabled=args.dfr_detailing_lora is not None,
            dfr_detailing_lora=(
                (str(args.dfr_detailing_lora), 1.0) if args.dfr_detailing_lora is not None else None
            ),
        )
        mx.eval(video_latent, audio_latent)
        latent_seconds = time.perf_counter() - latent_started
        sampling_peak = mx.get_peak_memory()
        mx.save_safetensors(
            str(outputs["latent"]),
            {"video_latent": video_latent, "audio_latent": audio_latent},
            metadata={
                "prompt": prompt,
                "seed": str(args.seed),
                "width": str(args.width),
                "height": str(args.height),
                "frames": str(args.frames),
                "fps": str(args.fps),
            },
        )
        pipeline._release_sampling()

        audio_started = time.perf_counter()
        waveform = pipeline.audio_decoder_block(audio_latent)
        mx.eval(waveform)
        save_waveform(waveform, str(outputs["audio"]), sample_rate=48000)
        audio_seconds = time.perf_counter() - audio_started
        pipeline.audio_decoder_block.free()
        del waveform, audio_latent
        mx.clear_cache()
    else:
        reused = mx.load(str(args.reuse_latent))
        video_latent = reused["video_latent"]
        latent_seconds = 0.0
        sampling_peak = 0
        audio_seconds = 0.0

    decode_reports: dict[str, dict[str, float | int]] = {}
    decode_targets = _decode_targets(args.decode_mode, args.conv_vae, args.diffusion_vae)
    for name, vae_path in decode_targets:
        decoder = LTX25VideoDecoder(
            vae_path,
            verbose=True,
            diffvae_optimization=(
                args.diffvae_optimization if name == "diffusion" else "combined"
            ),
            diffvae_query_chunk_size=args.diffvae_query_chunk_size,
        )
        mx.reset_peak_memory()
        started = time.perf_counter()
        decoder.decode_and_stream(
            video_latent,
            str(outputs[name]),
            frame_rate=args.fps,
            audio_path=str(outputs["audio"]) if outputs["audio"].exists() else None,
        )
        elapsed = time.perf_counter() - started
        decode_reports[name] = {
            "seconds": elapsed,
            "peak_bytes": int(mx.get_peak_memory()),
        }
        decoder.free()
        mx.clear_cache()

    outputs["audio"].unlink(missing_ok=True)
    reported_outputs = {name: str(outputs[name]) for name, _ in decode_targets}
    reported_outputs["report"] = str(outputs["report"])
    if args.reuse_latent is None:
        reported_outputs["latent"] = str(outputs["latent"])
    else:
        reported_outputs["reused_latent"] = str(args.reuse_latent)
    report = {
        "prompt": prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "latent_seconds": latent_seconds,
        "sampling_peak_bytes": int(sampling_peak),
        "audio_decode_seconds": audio_seconds,
        "dfr_enabled": args.dfr_detailing_lora is not None,
        "diffvae_optimization": args.diffvae_optimization,
        "diffvae_query_chunk_size": args.diffvae_query_chunk_size,
        "decode_mode": args.decode_mode,
        "decode": decode_reports,
        "total_seconds": time.perf_counter() - total_started,
        "outputs": reported_outputs,
    }
    outputs["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
