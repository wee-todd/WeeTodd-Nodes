"""Run candidate recipes sequentially and persist honest per-candidate validation status."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def validate_media(recipe, media):
    """Check delivered media against the recipe, not merely the writer's own metadata."""
    videos = [stream for stream in media["streams"] if stream["codec_type"] == "video"]
    audios = [stream for stream in media["streams"] if stream["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ValueError("Expected exactly one video and one audio stream")
    video, audio = videos[0], audios[0]
    config = recipe["config"]
    fps = 24.0 if recipe["engine"] == "h3" else config["frame_rate"]
    if recipe["engine"] == "h3":
        frames = round(config["duration_seconds"] * fps)
        frames += (5 - frames) % 17
    else:
        frames = max(1, round(config["duration_seconds"] * fps / 8)) * 8 + 1
    expected = (config["width"], config["height"], frames)
    actual = (video["width"], video["height"], int(video["nb_read_frames"]))
    if actual != expected:
        raise ValueError(f"Media dimensions/frame count {actual} do not match {expected}")
    if abs(float(Fraction(video["avg_frame_rate"])) - fps) > 1e-6:
        raise ValueError("Media frame rate does not match recipe")
    drift = abs(float(video["duration"]) - float(audio["duration"]))
    # AAC padding and the LTX 8n+1 endpoint frame can add up to one frame.
    tolerance = 0.025 if recipe["engine"] == "h3" else 1 / fps + 0.025
    if drift > tolerance:
        raise ValueError(f"Audio/video duration drift {drift:.6f}s exceeds {tolerance:.6f}s")
    return {"expected_frames": frames, "fps": fps, "av_drift_seconds": drift}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    rows = json.loads(args.matrix.read_text())
    project = Path(__file__).resolve().parents[1]
    results = []

    def persist():
        (args.matrix.parent / "headless-validation.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )

    for row in rows:
        if args.only and row["candidate"] not in args.only:
            continue
        output = Path(row["recipe"]).parent / "headless"
        record = dict(row)
        if row["status"] == "ready":
            existing = output / "result.json"
            if not existing.exists():
                if output.exists():
                    record.update(
                        status="running_or_incomplete",
                        error="Output exists without a result; not starting a duplicate",
                    )
                    results.append(record)
                    persist()
                    break
                print(f"Starting {row['candidate']}", flush=True)
                if row["engine"] == "h3":
                    check = subprocess.run(
                        [
                            sys.executable,
                            str(project / "scripts/preflight_h3_workflow.py"),
                            "--project",
                            str(project),
                            "--workflow",
                            row["api"],
                            "--comfy-root",
                            str(args.comfy_root),
                        ],
                        cwd="/",
                        capture_output=True,
                        text=True,
                    )
                    (output.parent / "preflight.log").write_text(check.stdout + check.stderr)
                    if check.returncode:
                        record.update(status="blocked_preflight", error=check.stderr)
                        results.append(record)
                        persist()
                        continue
                with (output.parent / "headless.log").open("x") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(project / "scripts/render_headless.py"),
                            "--recipe",
                            row["recipe"],
                            "--output-directory",
                            str(output),
                        ],
                        cwd="/",
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            if existing.exists():
                rendered = json.loads(existing.read_text())
                record["render"] = rendered
                record["status"] = (
                    "render_failed" if rendered["status"] != "success" else "headless_render_pass"
                )
                if rendered["status"] == "success":
                    probe = subprocess.run(
                        [
                            "/opt/homebrew/bin/ffprobe",
                            "-v",
                            "error",
                            "-count_frames",
                            "-show_streams",
                            "-show_format",
                            "-of",
                            "json",
                            rendered["video"],
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    media = json.loads(probe.stdout)
                    record["probe"] = media
                    try:
                        record["media_validation"] = validate_media(
                            json.loads(Path(row["recipe"]).read_text()), media
                        )
                    except (ValueError, KeyError) as exc:
                        record["status"] = "media_failed"
                        record["error"] = str(exc)
            else:
                record.update(status="process_failed", error="Runner did not write result.json")
        results.append(record)
        persist()
        print(f"{row['candidate']}: {record['status']}", flush=True)


if __name__ == "__main__":
    main()
