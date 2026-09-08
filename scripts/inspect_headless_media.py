"""Inspect completed matrix media for technical validity, not aesthetic quality."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from validate_headless_matrix import validate_media


def inspect_video(filename, recipe):
    media = json.loads(
        subprocess.check_output(
            [
                "/opt/homebrew/bin/ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(filename),
            ]
        )
    )
    result = validate_media(recipe, media)
    video = np.frombuffer(
        subprocess.check_output(
            [
                "/opt/homebrew/bin/ffmpeg",
                "-v",
                "error",
                "-i",
                str(filename),
                "-map",
                "0:v:0",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        ),
        dtype=np.uint8,
    )
    audio = np.frombuffer(
        subprocess.check_output(
            [
                "/opt/homebrew/bin/ffmpeg",
                "-v",
                "error",
                "-i",
                str(filename),
                "-map",
                "0:a:0",
                "-acodec",
                "pcm_f32le",
                "-f",
                "f32le",
                "pipe:1",
            ]
        ),
        dtype=np.float32,
    )
    result.update(
        audio_finite=bool(np.isfinite(audio).all()),
        audio_rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        audio_peak=float(np.max(np.abs(audio))),
        black_fraction=float(np.mean(video <= 1)),
        white_fraction=float(np.mean(video >= 254)),
    )
    result["technical_pass"] = (
        result["audio_finite"]
        and result["audio_rms"] > 1e-5
        and result["black_fraction"] < 0.99
        and result["white_fraction"] < 0.99
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for row in json.loads(args.matrix.read_text()):
        result = Path(row["recipe"]).parent / "headless/result.json"
        report = {"candidate": row["candidate"], "technical_pass": False}
        if result.exists():
            render = json.loads(result.read_text())
            if render["status"] == "success":
                report.update(
                    inspect_video(render["video"], json.loads(Path(row["recipe"]).read_text()))
                )
                if row["engine"] == "h3":
                    from safetensors.numpy import load_file

                    captures = list((result.parent / "raw").glob("*.safetensors"))
                    if len(captures) != 1:
                        raise ValueError("Expected exactly one final H3 latent capture")
                    latents = load_file(captures[0])
                    report["latent_finite"] = {
                        name: bool(np.isfinite(value).all()) for name, value in latents.items()
                    }
                    report["technical_pass"] = (
                        report["technical_pass"]
                        and set(latents) == {"video_latents", "audio_latents"}
                        and all(report["latent_finite"].values())
                    )
        reports.append(report)
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
