#!/usr/bin/env python3
"""Transcribe FastH3 acceptance audio with a local MLX Whisper checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mlx_whisper

VARIANTS = (
    "grouped_vsa",
    "indexed_metal",
    "indexed_metal_fused_qkv",
    "indexed_metal_layer40",
    "dense_control",
    "dense_token_pair",
)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _word_error_rate(reference: str, candidate: str) -> float:
    expected = _words(reference)
    actual = _words(candidate)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    media_root = args.media_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (model / "config.json").is_file() or not (model / "weights.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete MLX Whisper checkpoint: {model}")

    results = {}
    for name in VARIANTS:
        audio_path = media_root / f"{args.prefix}-{name}-audio.mp3"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Acceptance audio is missing: {audio_path}")
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(model),
            language="en",
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )
        transcript = result["text"].strip()
        results[name] = {
            "audio": str(audio_path),
            "transcript": transcript,
            "expected": args.expected,
            "word_error_rate": _word_error_rate(args.expected, transcript),
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "avg_logprob": segment["avg_logprob"],
                    "no_speech_prob": segment["no_speech_prob"],
                }
                for segment in result["segments"]
            ],
        }
        print(f"{name}: {transcript}", flush=True)

    output.write_text(
        json.dumps(
            {
                "model": str(model),
                "expected": args.expected,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
