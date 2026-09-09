#!/usr/bin/env python3
"""Local JSON bridge for the native Swift editor. No server, shell, or weight imports at startup."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def emit(**value):
    print(json.dumps(value), flush=True)


def run(command, *, capture=False):
    """Keep cancellation attached to this job's child process, including its cleanup."""
    child = subprocess.Popen(
        command, stdout=subprocess.PIPE if capture else sys.stderr, stderr=sys.stderr, text=True
    )
    try:
        output, _ = child.communicate()
        if child.returncode:
            raise RuntimeError(
                f"{Path(str(command[0])).name} exited with status {child.returncode}"
            )
        return output or ""
    except BaseException:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        raise


def executable(name, settings):
    key = {"ffmpeg": "ffmpegPath", "ffprobe": "ffprobePath"}.get(name, name)
    configured = settings.get(key, "")
    resolved = configured or shutil.which(name)
    if not resolved:
        for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
            if (Path(prefix) / name).is_file():
                resolved = str(Path(prefix) / name)
                break
    if not resolved or not os.access(resolved, os.X_OK):
        raise ValueError(f"Select an installed {name} executable in Runtime Settings.")
    return str(Path(resolved).resolve())


def inspect_media(value, settings):
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Media is missing: {source.name}. Relink it in Assets.")
    suffix = source.suffix.lower()
    if suffix in {".txt", ".md"}:
        return {"kind": "text", "text": source.read_text()[:100000], "path": str(source)}
    if suffix == ".safetensors":
        from studio_lora import inspect_lora

        return inspect_lora(source)
    probe = json.loads(
        run(
            [
                executable("ffprobe", settings),
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            capture=True,
        )
    )
    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), {})
    audio = any(s["codec_type"] == "audio" for s in probe["streams"])
    kind = (
        "image"
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".heic"}
        else "video"
        if video
        else "audio"
    )
    if not video and not audio:
        raise ValueError("The selected file contains no supported media stream.")
    num, den = video.get("avg_frame_rate", "0/1").split("/")
    return {
        "kind": kind,
        "path": str(source),
        "duration": float(probe.get("format", {}).get("duration", 0)),
        "width": video.get("width", 0),
        "height": video.get("height", 0),
        "fps": float(num) / float(den) if float(den) else 0,
        "hasAudio": audio,
    }


def profiles(directory):
    result = []
    root = Path(directory).expanduser()
    if not root.is_dir():
        return result
    for p in sorted(root.glob("*.json")):
        try:
            recipe = json.loads(p.read_text())
            if recipe.get("format") != "weetodd-headless-v2" or recipe.get("engine") not in {
                "h3",
                "ltx23",
                "ltx25",
            }:
                continue
            task = recipe.get("conditioning", {}).get("task", "t2v")
            if (
                recipe["engine"] == "h3"
                and recipe.get("components", {}).get("task") == "ref2va"
                and task == "t2v"
            ):
                task = "ref2va"
            result.append(
                {
                    "id": str(p.resolve()),
                    "name": p.stem.replace("_", " "),
                    "engine": recipe["engine"],
                    "task": task,
                    "width": recipe["config"].get("width", 768),
                    "height": recipe["config"].get("height", 448),
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    return result


def infer_task(clip):
    if clip.get("extensionDirection"):
        return "extension"
    roles = {a["role"] for a in clip.get("attachments", [])}
    for role, task in (("control", "control"), ("audioDriver", "a2v"), ("reference", "ref2va")):
        if role in roles:
            return task
    return "fflf" if roles & {"first", "last", "keyframe"} else "t2v"


def compose_recipe(request):
    """Map explicit editorial roles to the existing fail-closed renderer contract."""
    project, settings = request["project"], request["runtime"]
    clip = next(c for c in project["clips"] if c["id"] == request["clipID"])
    engine, task = clip["engine"], infer_task(clip)
    if engine == "movie":
        raise ValueError("Imported movies need finishing/export, not model generation.")
    available = profiles(settings["profilesDirectory"])
    selected = clip.get("profileID", "auto")
    if selected == "auto":
        candidates = [p for p in available if p["engine"] == engine and p["task"] == task]
        # LTX 2.5's basic distilled pipeline can transport keyframes and one frozen audio input.
        if not candidates and engine == "ltx25" and task in {"fflf", "a2v"}:
            candidates = [
                p
                for p in available
                if p["engine"] == engine and p["task"] == "t2v" and "distilled" in p["name"]
            ]
        if not candidates:
            raise ValueError(
                f"Import a compatible {engine} {task} recipe in Runtime Settings. "
                "No inputs were discarded."
            )
        selected = candidates[0]["id"]
    if selected not in {p["id"] for p in available}:
        raise ValueError("The selected model recipe is missing. Reimport or select Automatic.")
    recipe = json.loads(Path(selected).read_text())
    if recipe["engine"] != engine:
        raise ValueError("Selected recipe belongs to another model family.")
    recipe = copy.deepcopy(recipe)
    for key in ("reference_images", "conditioning"):
        recipe.pop(key, None)
    config = recipe["config"]
    for key, value in (
        ("width", clip["generationWidth"]),
        ("height", clip["generationHeight"]),
        ("seed", clip["seed"]),
        ("duration_seconds", clip["duration"]),
    ):
        config[key] = value
    if clip.get("negativePrompt") and engine != "h3":
        config["negative_prompt"] = clip["negativePrompt"]
    assets = {a["id"]: a for a in project["assets"] + request.get("globalAssets", [])}
    inputs, loras = [], []
    fps = 24 if engine == "h3" else config.get("frame_rate", 24)
    for attachment in clip.get("attachments", []):
        asset = assets.get(attachment["assetID"])
        if asset is None:
            raise ValueError("A clip attachment is missing from its asset store.")
        role = attachment["role"]
        if role == "lora":
            from studio_lora import clip_lora

            item = clip_lora(asset, attachment, engine)
            if any(previous["path"] == item["path"] for previous in loras):
                raise ValueError("Each LoRA file can be applied only once per clip.")
            loras.append(item)
            continue
        item = {
            "id": attachment["id"],
            "kind": "video" if asset["kind"] == "sequence" else asset["kind"],
            "role": "reference",
            "path": str(Path(asset["path"]).expanduser().resolve()),
            "strength": attachment.get("strength", 1),
        }
        if role in {"first", "last", "keyframe"}:
            item["role"] = "keyframe"
            item["frame_index"] = (
                "last"
                if role == "last"
                else 0
                if role == "first"
                else round(attachment["time"] * fps)
            )
        elif role == "audioDriver":
            item["role"] = "audio_driver"
        elif role == "control":
            item.update(role="control", control_type=attachment.get("controlType", "canny_edges"))
        elif role == "reference" and engine == "ltx25":
            item.update(
                reference_role="subject", description=attachment.get("description") or asset["name"]
            )
        elif role == "reference" and engine == "ltx23":
            item.update(role="control", control_type="ingredients_reference_sheet")
        inputs.append(item)
    contract = {"version": 1, "task": task, "inputs": inputs}
    if task == "extension":
        source = clip.get("extensionSource") or clip.get("sourcePath")
        if not source:
            raise ValueError("Select the movie to extend first.")
        media = inspect_media(source, settings)
        contract["inputs"] = [
            {"id": "extension-source", "kind": "video", "role": "reference", "path": source}
        ]
        if inputs:
            raise ValueError(
                "External extension does not accept extra media attachments in this editor."
            )
        config.update(width=media["width"], height=media["height"])
        if engine == "ltx23":
            config["duration_seconds"] = (round(media["duration"] * fps) - 1) / fps
        additional = max(8, round(clip["duration"] * fps / 8) * 8)
        if engine == "h3":
            from wee_todd_mlx.task_conditioning import frame_geometry

            additional = frame_geometry(recipe)[0]
        contract["extension"] = {
            "direction": clip["extensionDirection"],
            "additional_frames": additional,
        }
        if engine == "ltx25":
            frames = round(media["duration"] * fps)
            contract["extension"]["context_frames"] = min(25, (frames - 1) // 8 * 8 + 1)
    if loras:
        if engine == "ltx25":
            recipe["components"].setdefault("loras", []).extend(
                [[a["path"], a["strength"]] for a in loras]
            )
        elif engine == "ltx23" and recipe["components"].get("loras"):
            recipe["components"]["loras"].extend(loras)
        else:
            recipe.setdefault("loras", {}).setdefault("adapters", []).extend(loras)
    recipe["conditioning"] = contract
    prompt = clip["prompt"].strip()
    if not prompt:
        raise ValueError("Write a prompt before preparing the render.")
    if (
        engine == "h3"
        and not prompt.startswith("integrated_multimodal_description:")
        and "[video continuation" not in prompt
    ):
        if task in {"ref2va", "a2v", "extension"}:
            # Native reference prompts are semantic contracts; do not fabricate descriptions.
            if (
                "reference" not in prompt.lower()
                or "integrated_multimodal_description:" not in prompt
            ):
                raise ValueError(
                    "H3 Ref2VA needs its native reference prompt. "
                    "Paste the complete six-section prompt in the full-window editor."
                )
        else:
            prompt = (
                f"integrated_multimodal_description: [Shot 1] {prompt}\n\n"
                f"overall_soundscape: {clip.get('soundscape', 'Natural location sound.')}\n\n"
                f"non_diegetic_music: {clip.get('music', 'N/A')}"
            )
    if engine == "ltx23" and task == "ref2va" and not prompt.startswith("Reference sheet:"):
        descriptions = [
            a.get("description", "") for a in clip["attachments"] if a["role"] == "reference"
        ]
        if not all(descriptions):
            raise ValueError(
                "Describe the Ingredients reference sheet in the attachment description "
                "before generation."
            )
        prompt = "Reference sheet: " + "; ".join(descriptions) + "\n\nGenerated video: " + prompt
    recipe.update(
        prompt=prompt,
        ffmpeg=executable("ffmpeg", settings),
        ffprobe=executable("ffprobe", settings),
    )
    from wee_todd_mlx.task_conditioning import validate_conditioning

    report = validate_conditioning(recipe)
    return recipe, {
        "profile": Path(selected).stem,
        "task": task,
        "conditioning": report,
        "nativeFPS": fps,
        "movieSettings": clip.get("settingsOverride") or project["settings"],
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(request, destination):
    destination.mkdir(parents=True, exist_ok=False)
    recipe, report = compose_recipe(request)
    recipe_path = destination / "recipe.json"
    write_json(recipe_path, recipe)
    write_json(destination / "editor-request.json", request)
    run(
        [
            sys.executable,
            str(ROOT / "scripts/render_headless.py"),
            "--recipe",
            str(recipe_path),
            "--output-directory",
            str(destination / "preflight"),
            "--preflight-only",
        ]
    )
    return {"recipePath": str(recipe_path), "prompt": recipe["prompt"], "report": report}


def render(prepared, destination):
    if not Path(prepared).is_file():
        raise ValueError("Prepare and review this clip before rendering.")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/render_headless.py"),
            "--recipe",
            prepared,
            "--output-directory",
            str(destination),
        ]
    )
    result = json.loads((destination / "result.json").read_text())
    if result.get("status") != "success":
        raise RuntimeError(result.get("error", "Renderer did not complete."))
    return result


def motion_request(request):
    """Resolve an explicit repair recipe or the recipe of the selected base render."""
    from wee_todd_mlx.motion_fidelity import MotionSettings, validate_recipe

    clip = next(c for c in request["project"]["clips"] if c["id"] == request["clipID"])
    if clip["engine"] != "h3":
        raise ValueError("Motion Fidelity is currently available for H3 clips only.")
    settings = clip.get("motionFidelity") or {}
    MotionSettings(**settings).validate()
    recipe_path = clip.get("motionRecipeID")
    if not recipe_path:
        version = next(
            (v for v in clip.get("versions", []) if v["path"] == clip.get("sourcePath")), None
        )
        recipe_path = version.get("recipePath") if version else None
    recipe_evidence = {}
    if recipe_path:
        from studio_job import file_hash

        recipe_file = Path(recipe_path)
        recipe_evidence = {
            "recipeSHA256": file_hash(recipe_file),
            "recipeSize": recipe_file.stat().st_size,
            "recipeModified": recipe_file.stat().st_mtime,
        }
        recipe = json.loads(recipe_file.read_text())
    else:
        recipe, _ = compose_recipe(request)
    recipe_prompt = recipe["prompt"]
    override = clip.get("motionPrompt")
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("The repair prompt override must be a nonblank string.")
        recipe = copy.deepcopy(recipe)
        recipe["prompt"] = override
    validate_recipe(recipe)
    return {
        "recipe": recipe,
        "recipePrompt": recipe_prompt,
        "clip": clip,
        "settings": settings,
        "runtime": request["runtime"],
        "recipePath": recipe_path or "",
        "recipeEvidence": recipe_evidence,
    }


def motion_prepare(request):
    """Resolve and validate the complete repair prompt without writing or loading weights."""
    current = motion_request(request)
    return {
        "prompt": current["recipe"]["prompt"],
        "recipePrompt": current["recipePrompt"],
        "usingOverride": current["clip"].get("motionPrompt") is not None,
    }


def motion_enhance(request, destination, *, analyze=False):
    current = motion_request(request)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_path = destination.with_name(destination.name + "-request.json")
    write_json(request_path, current)
    command = [
        sys.executable,
        str(ROOT / "scripts/motion_fidelity.py"),
        "--request",
        str(request_path),
        "--output-directory",
        str(destination),
    ]
    if analyze:
        command.append("--analyze-only")
    run(command)
    result = json.loads((destination / "result.json").read_text())
    if not analyze:
        from studio_job import file_hash

        clip = current["clip"]
        source, output = Path(clip["sourcePath"]), Path(result["video"])
        recipe_path = Path(current["recipePath"]) if current["recipePath"] else None
        result["motionResult"] = {
            "path": str(output),
            "sourcePath": str(source),
            "sourceIn": clip.get("sourceIn", 0),
            "duration": clip["duration"],
            "outputIn": result.get("sourceIn", 0),
            "recipeID": clip.get("motionRecipeID") or "",
            "motionPrompt": clip.get("motionPrompt"),
            "settings": current["settings"],
            "sourceSHA256": result["sourceSHA256"],
            "sha256": file_hash(output),
            "report": result["report"],
            "sourceSize": source.stat().st_size,
            "sourceModified": source.stat().st_mtime,
            "outputSize": output.stat().st_size,
            "outputModified": output.stat().st_mtime,
            "recipePath": str(recipe_path) if recipe_path else None,
            **current["recipeEvidence"],
        }
    return result


def resolve_motion_output(clip):
    """Do not silently export a base movie when an enhancement was requested."""
    if not (clip.get("motionFidelity") or {}).get("enabled"):
        return
    from studio_job import file_hash

    result = clip.get("motionResult") or {}
    if (
        not result
        or result.get("settings") != clip["motionFidelity"]
        or result.get("sourcePath") != clip.get("sourcePath")
        or result.get("sourceIn") != clip.get("sourceIn", 0)
        or result.get("duration") != clip.get("duration")
        or result.get("recipeID", "") != (clip.get("motionRecipeID") or "")
        or result.get("motionPrompt") != clip.get("motionPrompt")
        or not Path(result.get("path", "")).is_file()
        or file_hash(clip["sourcePath"]) != result.get("sourceSHA256")
        or file_hash(result["path"]) != result.get("sha256")
        or (
            result.get("recipePath")
            and (
                not Path(result["recipePath"]).is_file()
                or file_hash(result["recipePath"]) != result.get("recipeSHA256")
            )
        )
    ):
        raise ValueError(
            "Motion Fidelity is pending or stale. Enhance the clip or export a headless job."
        )
    clip["sourcePath"] = result["path"]
    clip["sourceIn"] = result.get("outputIn", 0)


def resolved_settings(project, clip):
    s = dict(clip.get("settingsOverride") or project["settings"])
    for key in ("width", "height", "upscaleWidth", "upscaleHeight"):
        if type(s[key]) is not int or not 64 <= s[key] <= 8192 or s[key] % 2:
            raise ValueError("Movie dimensions must be even values from 64 through 8192.")
    for key in ("fps", "interpolatedFPS"):
        if (
            not isinstance(s[key], (float, int))
            or not math.isfinite(s[key])
            or not 1 <= s[key] <= 240
        ):
            raise ValueError("Movie frame rates must be finite values from 1 through 240.")
    s["outputFPS"] = s["fps"] if s["interpolation"] == "off" else s["interpolatedFPS"]
    s["outputWidth"] = s["width"] if s["upscaling"] == "off" else s["upscaleWidth"]
    s["outputHeight"] = s["height"] if s["upscaling"] == "off" else s["upscaleHeight"]
    if s["interpolation"] != "off":
        ratio = s["interpolatedFPS"] / s["fps"]
        if ratio not in (2, 3, 4) or (s["interpolation"] == "metalFX" and ratio != 2):
            raise ValueError("Choose 2×, 3× or 4× interpolation; MetalFX supports 2×.")
    return s


def preflight_finishing(project, clip, runtime):
    """Check finishing dependencies before loading any generation model."""
    settings = resolved_settings(project, clip)
    executable("ffmpeg", runtime)
    executable("ffprobe", runtime)
    if settings["interpolation"] == "rife":
        cli, weights = runtime.get("rifePath", ""), runtime.get("rifeWeights", "")
        if (
            not cli
            or not os.access(cli, os.X_OK)
            or not (Path(weights) / "model.safetensors").is_file()
        ):
            raise ValueError("Connect the RIFE executable and weights before running this job.")
    if "metalFX" in (settings["interpolation"], settings["upscaling"]):
        helper = runtime.get("metalPath", "")
        if not helper or not os.access(helper, os.X_OK):
            raise ValueError("Build and connect StudioMetal before running this job.")
        capabilities = json.loads(run([helper, "capabilities"], capture=True))
        key = "interpolation" if settings["interpolation"] == "metalFX" else "spatial"
        if not capabilities.get(key):
            raise ValueError(f"This Mac does not support MetalFX {key}.")
    if settings["interpolation"] == "metalFX":
        camera_file = Path(clip.get("depthDirectory", "")) / "camera.json"
        if not camera_file.is_file():
            raise ValueError("MetalFX depth folder needs camera.json with actual camera metadata.")
        camera = json.loads(camera_file.read_text())
        near, far, fov = (camera.get(k, 0) for k in ("nearPlane", "farPlane", "fieldOfView"))
        if (
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (near, far, fov))
            or not 0 < near < far
            or not 0 < fov < 180
            or type(camera.get("depthReversed")) is not bool
        ):
            raise ValueError("MetalFX camera metadata is invalid.")
        frames = max(1, round(clip["duration"] * settings["fps"]))
        pixels = settings["outputWidth"] * settings["outputHeight"]
        for name, channels in (("depthDirectory", 1), ("motionDirectory", 2)):
            folder = Path(clip.get(name, ""))
            if not clip.get(name) or not folder.is_dir():
                raise ValueError(
                    "MetalFX interpolation requires clip depth and motion guide folders."
                )
            for index in range(1, frames):
                guide = folder / f"{index:06d}.f32"
                if not guide.is_file() or guide.stat().st_size != pixels * channels * 4:
                    raise ValueError(
                        f"Missing or incorrectly sized MetalFX guide: {guide.name} ({name})"
                    )
    return settings


def filter_fit(width, height, mode):
    if mode == "fill":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height}"
        )
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
    )


def normalize_clip(clip, project, runtime, destination, work):
    s = resolved_settings(project, clip)
    preflight_finishing(project, clip, runtime)
    source = clip.get("sourcePath", "")
    info = inspect_media(source, runtime)
    duration, start = float(clip["duration"]), float(clip.get("sourceIn", 0))
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start) or start < 0:
        raise ValueError("Clip duration or source in point is invalid.")
    if info["kind"] != "image" and start + duration > info["duration"] + 0.08:
        raise ValueError(
            f"{clip['name']}: trim extends beyond the source movie. Adjust its duration."
        )
    ffmpeg = executable("ffmpeg", runtime)
    args = [ffmpeg, "-v", "error", "-nostdin", "-y"]
    if info["kind"] == "image":
        args += ["-loop", "1"]
    args += ["-ss", str(start), "-i", source]
    if not info.get("hasAudio"):
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    width, height = (
        (s["width"], s["height"])
        if s["upscaling"] == "metalFX"
        else (s["outputWidth"], s["outputHeight"])
    )
    args += [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0" if info.get("hasAudio") else "1:a:0",
        "-vf",
        filter_fit(width, height, s.get("fit", "fit")) + f",fps={s['fps']},setsar=1",
        "-af",
        f"aresample=48000,volume={float(clip.get('volume', 1))},apad",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(s.get("quality", 18)),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-ac",
        "2",
    ]
    base = work / "base.mp4"
    run(args + [str(base)])
    current = base
    if s["upscaling"] == "metalFX":
        helper = runtime.get("metalPath", "")
        if not helper or not Path(helper).is_file():
            raise ValueError("Build the StudioMetal helper before selecting MetalFX.")
        metal = work / "metal.mp4"
        run(
            [
                helper,
                "upscale",
                str(current),
                str(metal),
                str(s["outputWidth"]),
                str(s["outputHeight"]),
            ]
        )
        # The native helper processes video frames; remux the original track explicitly.
        mux = work / "metal-audio.mp4"
        run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(metal),
                "-i",
                str(current),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-t",
                str(duration),
                str(mux),
            ]
        )
        current = mux
    if s["interpolation"] == "rife":
        cli, weights = runtime.get("rifePath", ""), runtime.get("rifeWeights", "")
        if (
            not cli
            or not os.access(cli, os.X_OK)
            or not (Path(weights) / "model.safetensors").is_file()
        ):
            raise ValueError("Select the RIFE executable and weights folder in Runtime Settings.")
        result = work / "rife.mp4"
        run(
            [
                cli,
                "-i",
                str(current),
                "-o",
                str(result),
                "-m",
                str(round(s["outputFPS"] / s["fps"])),
                "-s",
                str(s.get("rifeScale", 1.0)),
                "--weights_dir",
                weights,
            ]
        )
        current = result
    elif s["interpolation"] == "metalFX":
        if not clip.get("depthDirectory") or not clip.get("motionDirectory"):
            raise ValueError(
                "MetalFX interpolation needs per-frame depth and backward-motion guides. "
                "Set clip guide folders or choose RIFE."
            )
        result = work / "metal-interpolated.mp4"
        run(
            [
                runtime["metalPath"],
                "interpolate",
                str(current),
                str(result),
                clip["depthDirectory"],
                clip["motionDirectory"],
            ]
        )
        mux = work / "interpolated-audio.mp4"
        run(
            [
                ffmpeg,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(result),
                "-i",
                str(current),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-t",
                str(duration),
                str(mux),
            ]
        )
        current = mux
    # RIFE produces (N-1)*multiplier+1 frames. Hold the final fraction to preserve edit timing.
    run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(current),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration=0.2,fps={s['outputFPS']}",
            "-af",
            "apad",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-crf",
            str(s.get("quality", 18)),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(destination),
        ]
    )
    return s


def export_movie(request, destination, *, cache_directory=None):
    project, runtime = request["project"], request["runtime"]
    if not project["clips"]:
        raise ValueError("Add a movie or render a clip before exporting.")
    if destination.exists():
        raise ValueError("Choose a new export filename. Existing movies are never overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = executable("ffmpeg", runtime)
    tracks = {t["id"]: t for t in project.get("audioTracks", [])}
    solo = any(t.get("solo") for t in tracks.values())
    project = copy.deepcopy(project)
    for clip in project["clips"]:
        resolve_motion_output(clip)
    project["audio"] = [
        a
        for a in project.get("audio", [])
        if not tracks.get(a.get("trackID"), {}).get("muted")
        and (not solo or tracks.get(a.get("trackID"), {}).get("solo"))
    ]
    if cache_directory:
        cache_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".weetodd-export-", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        rendered, settings_list = [], []
        for i, clip in enumerate(project["clips"]):
            emit(
                event="progress",
                message=f"Finishing clip {i + 1} of {len(project['clips'])}: {clip['name']}",
                fraction=i / (len(project["clips"]) + 1),
            )
            folder = work / str(i)
            folder.mkdir()
            target = folder / "finished.mp4"
            settings = resolved_settings(project, clip)
            if cache_directory:
                from studio_job import digest, file_hash

                source = Path(clip.get("sourcePath", ""))
                stamp = (
                    [source.stat().st_size, source.stat().st_mtime_ns] if source.is_file() else []
                )
                key = digest({"clip": clip, "settings": settings, "source": stamp})
                cached = cache_directory / (key + ".mp4")
                receipt = cache_directory / (key + ".json")
                if (
                    cached.is_file()
                    and receipt.is_file()
                    and file_hash(cached) == json.loads(receipt.read_text()).get("sha256")
                ):
                    target = cached
                else:
                    normalize_clip(clip, project, runtime, target, folder)
                    shutil.copy2(target, cached)
                    write_json(receipt, {"sha256": file_hash(cached)})
                    target = cached
            else:
                normalize_clip(clip, project, runtime, target, folder)
            settings_list.append(settings)
            rendered.append(target)
        movie = resolved_settings(project, {})
        # Finish each clip separately, then conform assembly to the movie canvas and frame rate.
        args = [ffmpeg, "-v", "error", "-nostdin", "-y"]
        for target in rendered:
            args += ["-threads", "1", "-i", str(target)]
        for region in project.get("audio", []):
            if not Path(region["path"]).is_file():
                raise ValueError("An audio-track file is missing. Relink it before export.")
            args += ["-i", region["path"]]
        filters, v, a = [], "v0", "a0"
        width, height, fps = movie["outputWidth"], movie["outputHeight"], movie["outputFPS"]
        for i, _ in enumerate(rendered):
            filters.append(
                f"[{i}:v]{filter_fit(width, height, 'fit')},fps={fps},setsar=1,"
                f"settb=AVTB,setpts=PTS-STARTPTS[v{i}]"
            )
            filters.append(
                f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]"
            )
        duration = project["clips"][0]["duration"]
        for i, clip in enumerate(project["clips"][1:], 1):
            transition = clip.get("transition", "cut")
            overlap = (
                0
                if transition == "cut"
                else max(
                    0,
                    min(
                        clip.get("transitionDuration", 0.5),
                        clip["duration"] / 2,
                        project["clips"][i - 1]["duration"] / 2,
                    ),
                )
            )
            nv, na = f"joinedv{i}", f"joineda{i}"
            if overlap:
                transition = {"dissolve": "fade", "fadeBlack": "fadeblack", "wipe": "wipeleft"}.get(
                    transition
                )
                if not transition:
                    raise ValueError("Unsupported transition.")
                filters += [
                    f"[{v}][v{i}]xfade=transition={transition}:duration={overlap}:"
                    f"offset={duration - overlap}[{nv}]",
                    f"[{a}][a{i}]acrossfade=d={overlap}:c1=tri:c2=tri[{na}]",
                ]
            else:
                filters.append(f"[{v}][{a}][v{i}][a{i}]concat=n=2:v=1:a=1[{nv}][{na}]")
            duration += clip["duration"] - overlap
            v, a = nv, na
        for i, title in enumerate(project.get("titles", [])):
            if not title["text"]:
                continue
            textfile = work / f"title-{i}.txt"
            textfile.write_text(title["text"])
            size = max(8, min(300, int(title.get("fontSize", 56))))
            y = "(h-text_h)/2" if title.get("position") == "center" else "h-text_h-h*0.1"
            start, length = float(title["start"]), float(title["duration"])
            if not math.isfinite(start + length) or start < 0 or length <= 0:
                raise ValueError("Title timing is invalid.")
            # textfile and expansion=none preserve literal punctuation, percent signs, and newlines.
            escaped = str(textfile).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
            nv = f"titlev{i}"
            filters.append(
                f"[{v}]drawtext=textfile='{escaped}':expansion=none:fontsize={size}:"
                f"fontcolor=white:box=1:boxcolor=black@0.35:boxborderw=12:x=(w-text_w)/2:y={y}:"
                f"enable='between(t,{start},{start + length})'[{nv}]"
            )
            v = nv
        if solo:
            filters.append(f"[{a}]volume=0[source-muted]")
            a = "source-muted"
        else:
            for i, region in enumerate(project.get("audio", [])):
                if tracks.get(region.get("trackID"), {}).get("replacesSource"):
                    new_audio = f"replaced{i}"
                    start = float(region["start"])
                    end = start + float(region["duration"])
                    filters.append(f"[{a}]volume=0:enable='between(t,{start},{end})'[{new_audio}]")
                    a = new_audio
        audio_labels = [a]
        for i, region in enumerate(project.get("audio", [])):
            start, source_in, length = (
                float(region["start"]),
                float(region.get("sourceIn", 0)),
                float(region["duration"]),
            )
            if (
                not math.isfinite(start + source_in + length)
                or min(start, source_in) < 0
                or length <= 0
            ):
                raise ValueError("Audio region timing is invalid.")
            fade = min(max(float(region.get("fade", 0.2)), 0), length / 2)
            label = f"music{i}"
            filters.append(
                f"[{len(rendered) + i}:a]atrim=start={source_in}:duration={length},"
                f"asetpts=PTS-STARTPTS,aresample=48000,volume={float(region['volume'])},"
                f"afade=t=in:d={fade},afade=t=out:st={length - fade}:d={fade},"
                f"adelay={round(start * 1000)}:all=1[{label}]"
            )
            audio_labels.append(label)
        if len(audio_labels) > 1:
            filters.append(
                "".join(f"[{label}]" for label in audio_labels)
                + f"amix=inputs={len(audio_labels)}:duration=first:normalize=0,"
                "alimiter=limit=0.95[mixed]"
            )
            a = "mixed"
        filter_file = work / "composition.txt"
        filter_file.write_text(";\n".join(filters))
        args += [
            "-filter_complex_threads",
            "1",
            "-filter_complex_script",
            str(filter_file),
            "-map",
            f"[{v}]",
            "-map",
            f"[{a}]",
            "-t",
            str(duration),
        ]
        format_name = movie.get("format", "mp4")
        final = work / ("movie.mov" if format_name in {"mov", "proRes"} else "movie.mp4")
        if format_name == "proRes":
            args += [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "3",
                "-pix_fmt",
                "yuv422p10le",
                "-c:a",
                "pcm_s16le",
            ]
        else:
            args += [
                "-c:v",
                "libx264",
                "-crf",
                str(movie.get("quality", 18)),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
            ]
        emit(
            event="progress",
            message="Assembling movie, transitions, titles and audio",
            fraction=0.94,
        )
        run(args + [str(final)])
        info = inspect_media(final, runtime)
        if (
            (info["width"], info["height"]) != (width, height)
            or abs(info["fps"] - fps) > 0.01
            or not info["hasAudio"]
        ):
            raise RuntimeError("Export dimensions, frame rate or audio verification failed.")
        if abs(info["duration"] - duration) > max(0.1, 2 / fps):
            raise RuntimeError("Movie duration did not match the edit. Export was not published.")
        if format_name == "pngSequence":
            sequence = work / "sequence"
            sequence.mkdir()
            run([ffmpeg, "-v", "error", "-i", str(final), str(sequence / "frame-%06d.png")])
            run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(final),
                    "-vn",
                    "-c:a",
                    "pcm_s16le",
                    str(sequence / "audio.wav"),
                ]
            )
            write_json(
                sequence / "manifest.json",
                {"fps": fps, "duration": duration, "width": width, "height": height},
            )
            shutil.move(str(sequence), destination)
        else:
            os.replace(final, destination)
        write_json(
            destination.with_name(destination.name + ".json"),
            {
                "format": "weetodd-export-v1",
                "project": project,
                "resolvedClipSettings": settings_list,
                "media": info,
            },
        )
    return {
        "video": str(destination),
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
    }


def preview_movie(request, destination):
    request = copy.deepcopy(request)
    project = request["project"]
    settings = project["settings"]
    ratio = min(1, 960 / max(settings["width"], settings["height"]))
    settings.update(
        width=max(64, round(settings["width"] * ratio / 2) * 2),
        height=max(64, round(settings["height"] * ratio / 2) * 2),
        upscaling="off",
        interpolation="off",
        format="mp4",
        quality=25,
    )
    for clip in project["clips"]:
        if not clip.get("sourcePath"):
            raise ValueError(f"Generate {clip['name']} before building the movie preview.")
        clip.pop("settingsOverride", None)
    for title in project.get("titles", []):
        title["fontSize"] = max(8, title.get("fontSize", 56) * ratio)
    return export_movie(request, destination)


def import_sequence(request, destination):
    """Link naturally sorted frames and build an editable local movie proxy."""
    import re

    folder = Path(request["path"])
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"}

    def natural_key(path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]

    frames = sorted(
        (
            p
            for p in folder.iterdir()
            if p.suffix.lower() in allowed and not p.name.startswith("._")
        ),
        key=natural_key,
    )
    if not frames:
        raise ValueError("Choose a folder containing PNG, JPEG, TIFF or EXR frames.")
    if len({p.suffix.lower() for p in frames}) != 1:
        raise ValueError("Use one image format per sequence folder.")
    fps = float(request.get("fps", 24))
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError("Choose a sequence frame rate from 1 to 240.")
    destination.mkdir(parents=True, exist_ok=False)
    links = destination / "frames"
    links.mkdir()
    suffix = frames[0].suffix
    for i, frame in enumerate(frames):
        (links / f"{i:06d}{suffix}").symlink_to(frame.resolve())
    output = destination / "sequence.mov"
    run(
        [
            executable("ffmpeg", request["runtime"]),
            "-v",
            "error",
            "-nostdin",
            "-framerate",
            str(fps),
            "-i",
            str(links / f"%06d{suffix}"),
            "-c:v",
            "prores_ks",
            "-profile:v",
            "3",
            "-pix_fmt",
            "yuv422p10le",
            str(output),
        ]
    )
    write_json(
        destination / "source-sequence.json",
        {"folder": str(folder), "fps": fps, "frames": [str(p) for p in frames]},
    )
    return dict(video=str(output), frames=len(frames), duration=len(frames) / fps)


def bridge_frames(request, destination):
    project = request["project"]
    index = next(i for i, c in enumerate(project["clips"]) if c["id"] == request["clipID"])
    if index + 1 == len(project["clips"]):
        raise ValueError("A bridge needs a following clip.")
    before, after = project["clips"][index : index + 2]
    destination.mkdir(parents=True, exist_ok=False)
    for name, clip, seconds in (
        (
            "first",
            before,
            before.get("sourceIn", 0) + before["duration"] - 1 / project["settings"]["fps"],
        ),
        ("last", after, after.get("sourceIn", 0)),
    ):
        if not clip.get("sourcePath"):
            raise ValueError("Render both neighboring clips before inserting a bridge.")
        run(
            [
                executable("ffmpeg", request["runtime"]),
                "-v",
                "error",
                "-nostdin",
                "-ss",
                str(max(0, seconds)),
                "-i",
                clip["sourcePath"],
                "-frames:v",
                "1",
                str(destination / (name + ".png")),
            ]
        )
    return {"first": str(destination / "first.png"), "last": str(destination / "last.png")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "catalog",
            "inspect",
            "prepare",
            "render",
            "export",
            "export-job",
            "preview",
            "sequence",
            "bridge-frames",
            "motion-analyze",
            "motion-enhance",
            "motion-prepare",
        ],
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if args.command == "catalog":
        result = {"profiles": profiles(request["runtime"]["profilesDirectory"])}
    elif args.command == "inspect":
        result = inspect_media(request["path"], request["runtime"])
    elif args.command == "export-job":
        from studio_job import export_job

        result = export_job(request, args.output.resolve())
    elif args.command == "preview":
        result = preview_movie(request, args.output.resolve())
    elif args.command == "sequence":
        result = import_sequence(request, args.output.resolve())
    elif args.command == "bridge-frames":
        result = bridge_frames(request, args.output.resolve())
    elif args.command in {"motion-analyze", "motion-enhance"}:
        result = motion_enhance(
            request, args.output.resolve(), analyze=args.command == "motion-analyze"
        )
    elif args.command == "motion-prepare":
        result = motion_prepare(request)
    elif args.command == "prepare":
        result = prepare(request, args.output.resolve())
    elif args.command == "render":
        result = render(request["recipePath"], args.output.resolve())
    else:
        result = export_movie(request, args.output.resolve())
    emit(status="success", result=result)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        main()
    except KeyboardInterrupt:
        emit(status="cancelled")
        sys.exit(130)
    except Exception as error:
        emit(status="failed", error=str(error))
        sys.exit(1)
