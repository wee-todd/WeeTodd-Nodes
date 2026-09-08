#!/usr/bin/env python3
"""Audit every saved candidate against all tasks; optionally run diagnostic renders.

The matrix distinguishes rejection, preflight acceptance, and real render completion.
It never labels preflight acceptance as render or visual qualification.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import wave
from pathlib import Path

from validate_headless_matrix import validate_media

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wee_todd_mlx.headless_preflight import preflight_recipe  # noqa: E402
from wee_todd_mlx.task_conditioning import TASKS  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixtures(output, ffmpeg):
    import numpy as np
    from PIL import Image, ImageDraw

    for name, color, x in (("first", "red", 64), ("last", "blue", 256)):
        image = Image.new("RGB", (384, 256), (40, 40, 40))
        ImageDraw.Draw(image).rectangle((x, 80, x + 64, 160), fill=color)
        image.save(output / f"{name}.png")
    samples = np.arange(32000 * 3) / 32000
    waveform = (np.sin(2 * np.pi * 440 * samples) * 0.1 * 32767).astype("<i2")
    with wave.open(str(output / "audio.wav"), "wb") as handle:
        handle.setparams((1, 2, 32000, len(waveform), "NONE", "not compressed"))
        handle.writeframes(waveform.tobytes())
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-nostdin",
            "-loop",
            "1",
            "-i",
            str(output / "first.png"),
            "-t",
            "3",
            "-r",
            "24",
            "-vf",
            "edgedetect",
            "-pix_fmt",
            "yuv420p",
            str(output / "control.mp4"),
        ],
        check=True,
    )


def task_recipe(original, task, output):
    r = copy.deepcopy(original)
    r.pop("reference_images", None)
    inputs = []

    def item(identity, kind, role, filename, **kwargs):
        return {
            "id": identity,
            "kind": kind,
            "role": role,
            "path": str(output / filename),
            **kwargs,
        }

    if task == "fflf":
        inputs = [
            item("first", "image", "keyframe", "first.png", frame_index=0),
            item("last", "image", "keyframe", "last.png", frame_index="last"),
        ]
    elif task == "ref2va":
        inputs = [item("reference", "image", "reference", "first.png")]
        if r["engine"] == "ltx25":
            inputs[0].update(role="control", control_type="ingredients_reference_sheet")
    elif task == "a2v":
        inputs = [item("audio", "audio", "audio_driver", "audio.wav")]
    elif task == "control":
        inputs = [item("control", "video", "control", "control.mp4", control_type="canny_edges")]
    if r["engine"] == "h3":
        r["components"]["task"] = {"t2v": "t2va", "fflf": "fl2va", "ref2va": "ref2va"}.get(
            task, "t2va"
        )
    r["conditioning"] = {"version": 1, "task": task, "inputs": inputs}
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--render", action="append", default=[], help="candidate:task")
    parser.add_argument("--control-adapter", type=Path)
    parser.add_argument("--h3-vision-encoder", type=Path)
    parser.add_argument("--h3-multimodal-references", action="store_true")
    parser.add_argument("--extra-recipe", type=Path, action="append", default=[])
    parser.add_argument("--ltx23-two-stage-diagnostic", action="store_true")
    args = parser.parse_args()
    sources = sorted(args.candidates.glob("*/recipe.json")) + args.extra_recipe
    if not sources or not args.ffmpeg.is_file() or not args.python.is_file():
        parser.error("Candidate recipes, ffmpeg, and Python must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures(args.output, args.ffmpeg)
    matrix = {"format": "weetodd-task-matrix-v1", "baseline": [], "tasks": [], "renders": []}
    recipes = {}
    for source in sources:
        original = json.loads(source.read_text())
        if original.get("format") == "weetodd-h3-headless-recipe-v1":
            # Explicit adapter for the earlier pinned benchmark transport. Its production
            # profile certificate is not transferred to the broader runner by this audit.
            original = {
                k: v
                for k, v in original.items()
                if k
                in {
                    "components",
                    "config",
                    "prompt",
                    "attention",
                    "fastvideo",
                    "publication",
                    "cache_directory",
                }
            }
            active_layers = original.get("fastvideo", {}).get("active_layers", 50)
            original.update(
                format="weetodd-headless-v2",
                engine="h3",
                candidate=f"h3-benchmark-{active_layers}layer",
                ffmpeg=str(args.ffmpeg),
            )
        if original["candidate"] in recipes:
            raise ValueError(f"Duplicate candidate identifier: {original['candidate']}")
        candidate = original["candidate"]
        recipes[candidate] = original
        try:
            report = preflight_recipe(original)
            matrix["baseline"].append({"candidate": candidate, "status": "preflight_passed"})
        except Exception as exc:
            matrix["baseline"].append(
                {"candidate": candidate, "status": "failed", "error": str(exc)}
            )
        for task in TASKS:
            entry = {"candidate": candidate, "task": task, "render_qualification": "not_evaluated"}
            try:
                report = preflight_recipe(task_recipe(original, task, args.output))
                entry.update(status="preflight_passed", conditioning=report["conditioning"])
            except Exception as exc:
                entry.update(status="rejected", error=f"{type(exc).__name__}: {exc}")
            matrix["tasks"].append(entry)
    save(args.output / "matrix.json", matrix)
    print(
        json.dumps({"baselines": matrix["baseline"], "task_cells": len(matrix["tasks"])}),
        flush=True,
    )
    for selection in args.render:
        candidate, task = selection.split(":", 1)
        r = task_recipe(recipes[candidate], task, args.output)
        # Explicit diagnostic recipe adaptation, recorded separately from unchanged matrix.
        if r["engine"] == "ltx23" and task in {"fflf", "a2v"}:
            r["config"].update(pipeline_mode="two_stage", stage1_steps=2, stage2_steps=1)
        elif r["engine"] == "h3" and not (r.get("vdn") or r.get("fastvideo")):
            r["config"]["steps"] = 3
        if r["engine"] == "ltx23" and args.ltx23_two_stage_diagnostic:
            r["config"].update(pipeline_mode="two_stage", stage1_steps=2, stage2_steps=1)
        if r["engine"] == "h3" and args.h3_vision_encoder:
            r["components"]["text_encoder"] = str(args.h3_vision_encoder)
        if r["engine"] == "h3" and task == "ref2va" and args.h3_multimodal_references:
            r["conditioning"]["inputs"].extend(
                [
                    {
                        "id": "video",
                        "kind": "video",
                        "role": "reference",
                        "path": str(args.output / "control.mp4"),
                    },
                    {
                        "id": "audio",
                        "kind": "audio",
                        "role": "reference",
                        "path": str(args.output / "audio.wav"),
                    },
                ]
            )
        if task == "control" and args.control_adapter:
            r["components"]["ic_loras"] = [[str(args.control_adapter), 1.0]]
        name = selection.replace(":", "-")
        filename = args.output / f"{name}.json"
        save(filename, r)
        entry = {
            "selection": selection,
            "recipe": str(filename),
            "qualification": "diagnostic_only",
        }
        try:
            preflight_recipe(r)
            with (args.output / f"{name}.log").open("w") as log:
                run = subprocess.run(
                    [
                        str(args.python),
                        str(ROOT / "scripts/render_headless.py"),
                        "--recipe",
                        str(filename),
                        "--output-directory",
                        str(args.output / name),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if run.returncode:
                raise RuntimeError(f"Render exited {run.returncode}; see {name}.log")
            result = json.loads((args.output / name / "result.json").read_text())
            probe = subprocess.run(
                [
                    str(args.ffmpeg.with_name("ffprobe")),
                    "-v",
                    "error",
                    "-show_streams",
                    "-count_frames",
                    "-of",
                    "json",
                    result["video"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            if not {"video", "audio"} <= {s["codec_type"] for s in streams}:
                raise RuntimeError("Expected synchronized audio and video streams")
            entry.update(
                status="render_completed",
                result=result,
                streams=streams,
                media_validation=validate_media(r, {"streams": streams}),
            )
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        matrix["renders"].append(entry)
        save(args.output / "matrix.json", matrix)
        print(
            json.dumps(
                {"selection": selection, "status": entry["status"], "error": entry.get("error")}
            ),
            flush=True,
        )
    if any(i["status"] == "failed" for i in matrix["baseline"] + matrix["renders"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
