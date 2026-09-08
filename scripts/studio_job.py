#!/usr/bin/env python3
"""Export and execute resumable, sequential WeeTodd movie/clip jobs without a graphical host."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import studio_bridge as bridge


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def artifact_hash(path):
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    if path.is_dir():
        return digest(
            [
                [str(p.relative_to(path)), file_hash(p)]
                for p in sorted(path.rglob("*"))
                if p.is_file()
            ]
        )
    return None


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    bridge.write_json(temporary, value)
    os.replace(temporary, path)


def clip_project(project, clip_id):
    """Keep title/audio intersections at clip-local times; retain linked assets and settings."""
    p = copy.deepcopy(project)
    elapsed = 0.0
    selected = None
    for i, clip in enumerate(p["clips"]):
        overlap = (
            0
            if i == 0 or clip.get("transition", "cut") == "cut"
            else min(
                clip["transitionDuration"], clip["duration"] / 2, p["clips"][i - 1]["duration"] / 2
            )
        )
        elapsed -= overlap
        if clip["id"] == clip_id:
            selected = clip
            break
        elapsed += clip["duration"]
    if selected is None:
        raise ValueError("The exported clip no longer exists.")
    end = elapsed + selected["duration"]
    for key in ("titles", "audio"):
        kept = []
        for item in p.get(key, []):
            begin = max(elapsed, item["start"])
            finish = min(end, item["start"] + item["duration"])
            if finish > begin:
                item = copy.deepcopy(item)
                if key == "audio":
                    item["sourceIn"] = item.get("sourceIn", 0) + begin - item["start"]
                item["start"] = begin - elapsed
                item["duration"] = finish - begin
                kept.append(item)
        p[key] = kept
    selected["transition"] = "cut"
    p["clips"] = [selected]
    p["name"] = selected["name"]
    return p


def renderer_fingerprint():
    files = sorted((bridge.ROOT / "src").rglob("*.py"))
    files += [
        bridge.ROOT / "scripts" / name
        for name in (
            "render_headless.py",
            "studio_bridge.py",
            "studio_job.py",
            "motion_fidelity.py",
        )
    ]
    return digest([[str(p.relative_to(bridge.ROOT)), file_hash(p)] for p in files])


def export_job(request, target):
    if target.exists():
        raise ValueError("Choose a new job filename. Existing jobs are not overwritten.")
    request = copy.deepcopy(request)
    if request.get("clipOnly"):
        request["project"] = clip_project(request["project"], request["clipID"])
    if not request["project"]["clips"]:
        raise ValueError("Add a clip before exporting a job.")
    recipes = {}
    for clip in request["project"]["clips"]:
        if clip["engine"] == "movie" or clip["id"] not in request.get("generateIDs", []):
            continue
        current = dict(request, clipID=clip["id"])
        recipe, report = bridge.compose_recipe(current)
        recipes[clip["id"]] = {"recipe": recipe, "report": report}
    motion_recipes = {}
    for clip in request["project"]["clips"]:
        if (clip.get("motionFidelity") or {}).get("enabled"):
            motion_recipes[clip["id"]] = bridge.motion_request(dict(request, clipID=clip["id"]))[
                "recipe"
            ]
    job = {
        "format": "weetodd-studio-job-v2" if motion_recipes else "weetodd-studio-job-v1",
        "scope": "clip" if request.get("clipOnly") else "movie",
        "project": request["project"],
        "globalAssets": request.get("globalAssets", []),
        "runtime": request["runtime"],
        "recipes": recipes,
        **({"motionRecipes": motion_recipes} if motion_recipes else {}),
        "execution": {
            "parallelGenerations": 1,
            "rendererSHA256": renderer_fingerprint(),
            "unloadBetweenStages": True,
            "finishingOrder": ["upscale", "interpolate", "assemble"],
        },
    }
    job["manifestSHA256"] = digest(job)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, job)
    instructions = target.with_name(target.name + ".txt")
    instructions.write_text(
        "WeeTodd sequential headless job\n\n"
        "The app includes WeeToddCLI in Contents/MacOS. "
        "It finds this job's native Python automatically.\n"
        "WeeToddCLI --job JOB.json --output-directory OUTPUT --resume\n\n"
        "Developer Python alternative:\n"
        "You can close WeeTodd Studio before running this job. Use the "
        "existing MLX Python environment.\n"
        "Run from WeeTodd-Nodes (replace JOB.json and OUTPUT with this "
        "job and a new output folder):\n\n"
        "python scripts/render_headless.py --job JOB.json "
        "--output-directory OUTPUT --preflight-only\n"
        "python scripts/render_headless.py --job JOB.json --output-directory OUTPUT --resume\n\n"
        "Jobs embed recipes and reference existing media/models. They do "
        "not include model weights.\n"
        "Use --resume after interruption. Completed renders and finishing "
        "stages are reused only when\n"
        "their recorded output hashes still match. Changed inputs invalidate resume.\n"
        "Generation and clip finishing run serially. Individual "
        "model-stage memory needs still apply.\n"
        "Movie dimensions/frame rate are applied per clip, then clips, "
        "titles, transitions and audio are assembled.\n"
    )
    return {
        "job": str(target),
        "instructions": str(instructions),
        "generations": len(recipes),
        "clips": len(job["project"]["clips"]),
    }


def inputs_fingerprint(job):
    """Stat selected input roots and files without loading weight payloads."""
    paths = set()
    path_keys = {
        "path",
        "sourcePath",
        "extensionSource",
        "transformer",
        "checkpoint",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
        "model_dir",
        "gemma_model",
        "loras",
        "ic_loras",
        "depthDirectory",
        "motionDirectory",
        "rifeWeights",
        "rifePath",
        "metalPath",
    }

    def walk(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                walk(child, name)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        elif isinstance(value, str) and value and (key in path_keys or key.endswith("_path")):
            p = Path(value)
            if p.exists():
                paths.add(p.resolve())

    walk(job["project"])
    walk(job["recipes"])
    walk(job.get("motionRecipes", {}))
    walk(job.get("globalAssets", []))
    walk(job["runtime"])
    observations = []
    for p in sorted(paths):
        members = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
        for member in members:
            s = member.stat()
            observations.append([str(member), s.st_size, s.st_mtime_ns, s.st_ino])
    return digest(observations)


def preflight(job, output):
    expected = job.get("execution", {}).get("rendererSHA256")
    if expected and expected != renderer_fingerprint():
        raise ValueError(
            "The renderer version changed. Run this job with its original runtime or re-export it."
        )
    output.mkdir(parents=True, exist_ok=True)
    for clip in job["project"]["clips"]:
        bridge.preflight_finishing(job["project"], clip, job["runtime"])
        record = job["recipes"].get(clip["id"])
        if record:
            root = output / clip["id"]
            # Preflight outputs are disposable evidence; give each attempt its own directory.
            import uuid

            attempt = root / uuid.uuid4().hex
            attempt.mkdir(parents=True)
            recipe_path = attempt / "recipe.json"
            bridge.write_json(recipe_path, record["recipe"])
            bridge.run(
                [
                    sys.executable,
                    str(bridge.ROOT / "scripts/render_headless.py"),
                    "--recipe",
                    str(recipe_path),
                    "--output-directory",
                    str(attempt / "result"),
                    "--preflight-only",
                ]
            )
        else:
            media = bridge.inspect_media(clip["sourcePath"], job["runtime"])
            if (
                media["kind"] != "image"
                and clip.get("sourceIn", 0) + clip["duration"] > media["duration"] + 0.08
            ):
                raise ValueError(f"{clip['name']}: trim extends beyond its source movie.")
    for clip in job["project"]["clips"]:
        if (clip.get("motionFidelity") or {}).get("enabled"):
            from wee_todd_mlx.motion_fidelity import MotionSettings, validate_recipe

            MotionSettings(**clip["motionFidelity"]).validate()
            validate_recipe(job.get("motionRecipes", {}).get(clip["id"], {}))
            if clip["id"] not in job["recipes"]:
                from motion_fidelity import preflight as motion_preflight

                motion_preflight(
                    {
                        "clip": clip,
                        "recipe": job["motionRecipes"][clip["id"]],
                        "settings": clip["motionFidelity"],
                        "runtime": job["runtime"],
                    }
                )
    for region in job["project"].get("audio", []):
        media = bridge.inspect_media(region["path"], job["runtime"])
        if not media.get("hasAudio"):
            raise ValueError("An audio region has no audio stream.")
        if region.get("sourceIn", 0) + region["duration"] > media["duration"] + 0.08:
            raise ValueError("An audio region extends beyond its source. Adjust the trim.")
    return {
        "status": "preflight_passed",
        "clips": len(job["project"]["clips"]),
        "generations": len(job["recipes"]),
    }


def execute(job, output, resume):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".job.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is executing this output folder.") from error
        return _execute_locked(job, output, resume)


def _execute_locked(job, output, resume):
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "job-state.json"
    identity = job["manifestSHA256"]
    inputs = inputs_fingerprint(job)
    state = {
        "manifestSHA256": identity,
        "inputsFingerprint": inputs,
        "completed": {},
        "status": "running",
    }
    if state_path.exists():
        if not resume:
            raise ValueError("This output folder has a job. Use --resume or choose a new folder.")
        state = json.loads(state_path.read_text())
        if state["manifestSHA256"] != identity or state["inputsFingerprint"] != inputs:
            raise ValueError(
                "Job or source inputs changed. Choose a new output folder; old "
                "outputs are preserved."
            )
    project = copy.deepcopy(job["project"])
    atomic_json(state_path, state)
    try:
        preflight(job, output / "preflight")
        for i, clip in enumerate(project["clips"]):
            record = job["recipes"].get(clip["id"])
            if not record:
                continue
            key = "generate-" + clip["id"]
            previous = state["completed"].get(key)
            if (
                previous
                and Path(previous["video"]).is_file()
                and file_hash(previous["video"]) == previous["sha256"]
            ):
                result = previous
                bridge.emit(event="resume", message=f"Reusing {clip['name']}")
            else:
                import uuid

                folder = output / "renders" / clip["id"] / uuid.uuid4().hex
                folder.mkdir(parents=True)
                recipe = copy.deepcopy(record["recipe"])
                recipe_path = folder / "recipe.json"
                bridge.write_json(recipe_path, recipe)
                bridge.emit(
                    event="progress",
                    message=f"Generating clip {i + 1}/{len(project['clips'])}: {clip['name']}",
                )
                result = bridge.render(str(recipe_path), folder / "result")
                result = {"video": result["video"], "sha256": file_hash(result["video"])}
                state["completed"][key] = result
                atomic_json(state_path, state)
            clip["sourcePath"] = result["video"]
            clip["sourceIn"] = 0
            info = bridge.inspect_media(result["video"], job["runtime"])
            requested_duration = clip["duration"]
            clip["duration"] = info["duration"]
            if clip.get("extensionSource"):
                source = bridge.inspect_media(clip["extensionSource"], job["runtime"])
                if clip.get("extensionDirection") == "after":
                    clip["sourceIn"] = source["duration"]
                clip["duration"] -= source["duration"]
            clip["duration"] = min(requested_duration, clip["duration"])
        motion_outputs = {}
        for clip in project["clips"]:
            if not (clip.get("motionFidelity") or {}).get("enabled"):
                continue
            key = "motion-" + clip["id"]
            previous = state["completed"].get(key)
            if (
                previous
                and Path(previous["video"]).is_file()
                and file_hash(previous["video"]) == previous["sha256"]
                and previous.get("sourceSHA256") == file_hash(clip["sourcePath"])
            ):
                result = previous
            else:
                import uuid

                folder = output / key / uuid.uuid4().hex
                folder.parent.mkdir(parents=True, exist_ok=True)
                request_path = folder.with_suffix(".json")
                atomic_json(
                    request_path,
                    {
                        "clip": clip,
                        "recipe": job["motionRecipes"][clip["id"]],
                        "settings": clip["motionFidelity"],
                        "runtime": job["runtime"],
                    },
                )
                bridge.run(
                    [
                        sys.executable,
                        str(bridge.ROOT / "scripts/motion_fidelity.py"),
                        "--request",
                        str(request_path),
                        "--output-directory",
                        str(folder),
                    ]
                )
                result = json.loads((folder / "result.json").read_text())
                result["sha256"] = file_hash(result["video"])
                state["completed"][key] = result
                atomic_json(state_path, state)
            motion_outputs[clip["id"]] = result["sha256"]
            clip["sourcePath"] = result["video"]
            clip["sourceIn"] = result.get("sourceIn", 0)
            clip["motionFidelity"]["enabled"] = False  # already resolved for finishing
        format_name = project["settings"].get("format", "mp4")
        name = (
            "movie-frames"
            if format_name == "pngSequence"
            else "movie.mov"
            if format_name in {"mov", "proRes"}
            else "movie.mp4"
        )
        final = output / name
        previous = state["completed"].get("export")
        if previous and motion_outputs and previous.get("motionOutputs") != motion_outputs:
            raise ValueError(
                "An enhancement changed since the saved final export. "
                "Choose a new output folder; existing movies are preserved."
            )
        if previous and final.exists() and artifact_hash(final) == previous.get("sha256"):
            result = previous
        else:
            if final.exists():
                raise ValueError(
                    "An unverified final export already exists. Preserve it elsewhere "
                    "before resuming."
                )
            result = bridge.export_movie(
                {"project": project, "runtime": job["runtime"]},
                final,
                cache_directory=output / "finished-clips",
            )
            result["sha256"] = artifact_hash(final)
            if motion_outputs:
                result["motionOutputs"] = motion_outputs
            state["completed"]["export"] = result
        state["status"] = "success"
        state["resolvedProject"] = project
        atomic_json(state_path, state)
        return result
    except BaseException as error:
        state["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = str(error)
        atomic_json(state_path, state)
        raise


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job.get("format") not in {"weetodd-studio-job-v1", "weetodd-studio-job-v2"}:
        parser.error("Unsupported job format")
    body = dict(job)
    expected = body.pop("manifestSHA256", None)
    if not expected or digest(body) != expected:
        parser.error(
            "Job manifest changed. Re-export it from Studio so recipes and settings agree."
        )
    result = (
        preflight(job, args.output_directory / "preflight")
        if args.preflight_only
        else execute(job, args.output_directory, args.resume)
    )
    bridge.emit(status="success", result=result)


def cli():
    try:
        main()
    except KeyboardInterrupt:
        bridge.emit(status="cancelled")
        sys.exit(130)
    except Exception as error:
        bridge.emit(status="failed", error=str(error))
        sys.exit(1)


if __name__ == "__main__":
    cli()
