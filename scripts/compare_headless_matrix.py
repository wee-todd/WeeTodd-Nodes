"""Compare complete media from each headless candidate and its saved Comfy control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_digest(video, kind):
    args = ["/opt/homebrew/bin/ffmpeg", "-v", "error", "-i", str(video), "-map", f"0:{kind}:0"]
    if kind == "v":
        args += ["-pix_fmt", "rgb24", "-f", "rawvideo"]
    else:
        args += ["-acodec", "pcm_f32le", "-f", "f32le"]
    args += ["pipe:1"]
    run = subprocess.run(args, check=True, capture_output=True)
    return hashlib.sha256(run.stdout).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--control-traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.matrix.read_text())
    traces = {}
    if args.control_traces:
        for filename in args.control_traces.glob("*.json"):
            trace = json.loads(filename.read_text())
            if trace.get("prompt_id"):
                traces[trace["prompt_id"]] = trace
    results = []
    for row in rows:
        directory = Path(row["recipe"]).parent
        result_path, history_path = directory / "headless/result.json", directory / "control.json"
        result = {"candidate": row["candidate"], "status": "not_validated"}
        if result_path.exists():
            headless = json.loads(result_path.read_text())
            result["headless_status"] = headless["status"]
            result["error"] = headless.get("error")
        if result_path.exists() and history_path.exists() and headless["status"] == "success":
            control_report = json.loads(history_path.read_text())
            if control_report["workflow_sha256"] != digest(Path(row["api"])):
                raise ValueError(f"Control graph changed after rendering: {row['candidate']}")
            if headless["recipe_sha256"] != digest(Path(row["recipe"])):
                raise ValueError(f"Headless recipe changed after rendering: {row['candidate']}")
            history = control_report["runs"][0]
            if history["history"]["status"]["status_str"] == "success":
                outputs = history["history"]["outputs"]
                media = [item for value in outputs.values() for item in value.get("gifs", [])]
                if len(media) != 1:
                    raise ValueError(f"Expected one final media item for {row['candidate']}")
                item = media[0]
                control = args.control_output / item["subfolder"] / item["filename"]
                video = Path(headless["video"])
                exact_file = digest(control) == digest(video)
                exact_video = stream_digest(control, "v") == stream_digest(video, "v")
                exact_audio = stream_digest(control, "a") == stream_digest(video, "a")
                result.update(
                    status="parity_pass" if exact_video and exact_audio else "parity_failed",
                    byte_identical_mp4=exact_file,
                    exact_decoded_video=exact_video,
                    exact_decoded_audio=exact_audio,
                    control=str(control),
                    headless=str(video),
                    control_seconds=history["server_seconds"],
                    headless_seconds=headless["seconds"],
                    timing_qualification="single smoke pair, not an overhead benchmark",
                )
                if row["engine"] == "h3" and args.control_traces:
                    raw = list((directory / "headless/raw").glob("*.json"))
                    if len(raw) != 1:
                        raise ValueError("Expected exactly one headless H3 trace")
                    headless_trace = json.loads(raw[0].read_text())
                    control_trace = traces[history["prompt_id"]]
                    result["exact_latents"] = digest(Path(headless_trace["latents"])) == digest(
                        Path(control_trace["latents"])
                    )
                    if not result["exact_latents"]:
                        result["status"] = "parity_failed"
        results.append(result)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
