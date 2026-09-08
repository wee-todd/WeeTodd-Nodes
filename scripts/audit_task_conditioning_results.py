#!/usr/bin/env python3
"""Recheck saved task diagnostic outputs without running the model again."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from validate_headless_matrix import validate_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, action="append", required=True)
    parser.add_argument("--baseline-candidates", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-controls", type=Path)
    args = parser.parse_args()
    rows = []
    for matrix_path in args.matrix:
        for entry in json.loads(matrix_path.read_text())["renders"]:
            if entry["status"] != "render_completed":
                continue
            result = entry["result"]
            recipe = json.loads(Path(entry["recipe"]).read_text())
            video = Path(result["video"])
            probe = subprocess.run(
                [
                    str(args.ffmpeg.with_name("ffprobe")),
                    "-v",
                    "error",
                    "-count_frames",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            row = {
                "selection": entry["selection"],
                "video": str(video),
                "media": validate_media(recipe, json.loads(probe.stdout)),
                "seconds": result["seconds"],
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "input_ids": result["consumed_input_ids"],
            }
            if any(result["runtime_loaded"]):
                raise AssertionError("Diagnostic left a weighted runtime loaded")
            baseline = args.baseline_candidates / recipe["candidate"] / "headless" / "result.json"
            if baseline.is_file():
                old = json.loads(baseline.read_text())
                row["same_mp4_as_prior_candidate_baseline"] = old["mp4_sha256"] == row["sha256"]
                row["comparison_note"] = (
                    "Only an identical recipe/settings comparison establishes parity"
                )
            if args.paired_controls and recipe["conditioning"]["task"] != "t2v":
                control_task = "ref2va" if recipe["candidate"] == "h3-ref2va" else "t2v"
                name = f"{recipe['candidate']}-{control_task}"
                paired = args.paired_controls / name / "result.json"
                if paired.is_file() and paired.parent != video.parent:
                    control_recipe = json.loads((args.paired_controls / f"{name}.json").read_text())
                    control = json.loads(paired.read_text())
                    if (
                        recipe["config"] != control_recipe["config"]
                        or recipe["prompt"] != control_recipe["prompt"]
                    ):
                        raise AssertionError(
                            "Paired control has different prompt or sampler settings"
                        )

                    def components(value):
                        return {k: v for k, v in value["components"].items() if k != "task"}

                    if components(recipe) != components(control_recipe):
                        raise AssertionError("Paired control changed the model components")
                    if row["sha256"] == control["mp4_sha256"]:
                        raise AssertionError(
                            "Conditioning did not change the final paired-control MP4"
                        )
                    row["paired_conditioning_effect"] = {
                        "control_video": control["video"],
                        "same_prompt_config_components": True,
                        "final_mp4_changed": True,
                    }
            if recipe["conditioning"]["task"] == "a2v":
                import numpy as np

                audio = subprocess.run(
                    [
                        str(args.ffmpeg),
                        "-v",
                        "error",
                        "-nostdin",
                        "-i",
                        str(video),
                        "-map",
                        "0:a:0",
                        "-ac",
                        "1",
                        "-ar",
                        "32000",
                        "-f",
                        "f32le",
                        "pipe:1",
                    ],
                    check=True,
                    capture_output=True,
                ).stdout
                signal = np.frombuffer(audio, dtype="<f4")
                spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
                peak_hz = float(np.argmax(spectrum) * 32000 / len(signal))
                if abs(peak_hz - 440) > 2:
                    raise AssertionError(
                        f"A2V did not retain the 440 Hz diagnostic source: {peak_hz}"
                    )
                row["retained_diagnostic_audio_peak_hz"] = peak_hz
                row["audio_note"] = (
                    "Tone retention only; not speech/lip-sync or exact PCM qualification"
                )
            rows.append(row)
    with args.output.open("x") as handle:
        json.dump(
            {"renders": rows, "count": len(rows), "qualification": "diagnostic_only"},
            handle,
            indent=2,
        )
        handle.write("\n")
    print(json.dumps({"verified_renders": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
