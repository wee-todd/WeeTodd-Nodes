"""Real-checkpoint joint DFR forward smoke test; synthetic tokens, not a quality render."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from ltx25_mlx.frozen_audio import DFRFrozenAudioX0Model
from ltx25_mlx.transformer import load_ltx25_transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.transformer / "paged_manifest.json").is_file() or args.output.exists():
        parser.error("Need a paged checkpoint and a new output path")
    started = time.perf_counter()
    transformer = load_ltx25_transformer(args.transformer, low_ram_streaming=True)
    model = DFRFrozenAudioX0Model(transformer)
    owner = transformer
    while getattr(owner, "inner", None) is not None:
        owner = owner.inner
    original = owner.audio_prompt_adaln_single
    mx.random.seed(35)
    video = mx.random.normal((1, 3, 128)).astype(mx.bfloat16)
    audio = mx.random.normal((1, 5, 128)).astype(mx.bfloat16)
    video_result, audio_result = model(
        video_latent=video,
        audio_latent=audio,
        sigma=mx.array([0.5], dtype=mx.bfloat16),
        video_timesteps=mx.array([[0, 0.5, 0.5]], dtype=mx.bfloat16),
        audio_timesteps=mx.zeros((1, 5), dtype=mx.bfloat16),
        video_text_embeds=mx.ones((1, 2, owner.config.video_dim), dtype=mx.bfloat16),
        audio_text_embeds=mx.ones((1, 2, owner.config.audio_dim), dtype=mx.bfloat16),
    )
    mx.eval(video_result, audio_result)
    checks = {
        "video_finite": bool(mx.all(mx.isfinite(video_result)).item()),
        "audio_unchanged": bool(mx.array_equal(audio_result, audio).item()),
        "preserved_video_row": bool(mx.array_equal(video_result[:, :1], video[:, :1]).item()),
        "modulation_restored": owner.audio_prompt_adaln_single is original,
    }
    report = {
        "checks": checks,
        "layers": owner.config.num_layers,
        "scope": "synthetic-token full joint transformer; no image/video quality claim",
        "seconds": time.perf_counter() - started,
        "peak_bytes": mx.get_peak_memory(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise RuntimeError("DFR joint frozen-audio checkpoint smoke failed")


if __name__ == "__main__":
    main()
