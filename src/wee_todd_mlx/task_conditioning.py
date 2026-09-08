"""Versioned, weight-free task contracts shared by headless engine adapters.

Accepted transport is not render qualification. Unsupported combinations fail closed.
Media is local and explicit; this module never downloads or loads model weights.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

TASKS = ("t2v", "fflf", "ref2va", "a2v", "extension", "control")
H3_CONTINUATION_FRAMES = (5, 22, 39, 56)
CONTROL_FAMILIES = {
    "canny_edges": "union_control",
    "depth_map": "union_control",
    "pose_skeleton": "union_control",
    "hed_edges": "h3_fun_union",
    "mlsd_lines": "h3_fun_union",
    "motion_track": "motion_track",
    "ingredients_reference_sheet": "ingredients_reference_sheet",
    "crossview_warp": "crossview_warp",
}


def _fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {label} fields: {', '.join(sorted(unknown))}")


def _number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _extension_audio_policy(engine):
    return "source_reencoded_and_generated_extension"


def _task_audio_policy(engine, task):
    if task == "a2v":
        # H3 Ref2VA uses the waveform as a semantic/timing reference and generates
        # a new synchronized soundtrack. LTX A2V freezes the supplied waveform.
        return "generated" if engine == "h3" else "source"
    if task == "extension":
        return _extension_audio_policy(engine)
    return "generated"


def frame_geometry(recipe):
    config = recipe["config"]
    if recipe["engine"] == "h3":
        # Match align_num_frames without importing an MLX engine.
        fps = 24.0
        frames = round(config.get("duration_seconds", 5.0) * fps)
        frames += (5 - frames) % 17
    else:
        fps = config.get("frame_rate", 24.0)
        frames = max(1, round(config.get("duration_seconds", 5.0) * fps / 8.0)) * 8 + 1
    return frames, fps


def normalize_conditioning(recipe, *, check_files=True):
    """Return an explicit contract; never mutate a legacy recipe or infer from filenames."""
    engine = recipe["engine"]
    if engine not in {"h3", "ltx23", "ltx25"}:
        raise ValueError(f"Unsupported engine: {engine}")
    # These fields were never implemented by the v2 runner. Do not keep ignoring them.
    misplaced = {
        "image",
        "images",
        "image_path",
        "first_frame",
        "last_frame",
        "audio_path",
        "audio_reference",
        "video_references",
        "msr_references",
        "keyframes",
        "task",
        "video_path",
        "controlnet",
        "reference_videos",
        "reference_audio",
    } & recipe.keys()
    if misplaced:
        raise ValueError(
            "Move media/task fields into conditioning: " + ", ".join(sorted(misplaced))
        )
    _fields(
        recipe,
        {
            "format",
            "engine",
            "candidate",
            "components",
            "config",
            "prompt",
            "ffmpeg",
            "ffprobe",
            "conditioning",
            "reference_images",
            "attention",
            "fastvideo",
            "vdn",
            "loras",
            "publication",
            "cache_directory",
        },
        "recipe",
    )
    if "conditioning" in recipe:
        if "reference_images" in recipe:
            raise ValueError("Use conditioning or legacy reference_images, not both")
        result = copy.deepcopy(recipe["conditioning"])
        _fields(
            result,
            {"version", "task", "inputs", "audio_policy", "extension"},
            "conditioning",
        )
        if type(result.get("version")) is not int or result["version"] != 1:
            raise ValueError("conditioning.version must be 1")
    else:
        task = recipe.get("components", {}).get("task", "t2va") if engine == "h3" else "t2va"
        legacy = recipe.get("reference_images", [])
        if not isinstance(legacy, list):
            raise ValueError("reference_images must be a list")
        if legacy and (engine != "h3" or task != "ref2va"):
            raise ValueError("Legacy reference_images require an H3 Ref2VA component set")
        result = {
            "version": 1,
            "task": {"t2va": "t2v", "fl2va": "fflf", "ref2va": "ref2va"}.get(task, task),
            "inputs": [
                {"id": f"reference-{i}", "kind": "image", "role": "reference", "path": p}
                for i, p in enumerate(legacy)
            ],
        }
    if result.get("task") not in TASKS:
        raise ValueError(f"Unsupported conditioning task: {result.get('task')!r}")
    result.setdefault("inputs", [])
    default_audio_policy = _task_audio_policy(engine, result["task"])
    result.setdefault("audio_policy", default_audio_policy)
    if result["audio_policy"] not in {
        "generated",
        "source",
        "source_latent_reconstructed_and_generated_extension",
        "source_reencoded_and_generated_extension",
    }:
        raise ValueError("Unsupported conditioning audio_policy")
    extension = result.get("extension")
    if result["task"] == "extension":
        _fields(extension, {"direction", "additional_frames", "context_frames"}, "extension")
        if extension.get("direction") not in {"before", "after"}:
            raise ValueError("extension.direction must be before or after")
        additional = extension.get("additional_frames")
        if engine == "h3":
            if extension["direction"] != "after":
                raise ValueError("H3 Ref2VA continuation supports extension after the source only")
            if "context_frames" in extension:
                raise ValueError(
                    "H3 external extension no longer accepts context_frames: the released "
                    "Ref2VA checkpoint consumes the full source video and a derived first-frame "
                    "anchor. Latent-overlap continuation remains an experimental internal node."
                )
            generated_frames, _ = frame_geometry(recipe)
            if type(additional) is not int or additional != generated_frames:
                raise ValueError(
                    "H3 extension.additional_frames must equal the aligned Ref2VA output "
                    f"window ({generated_frames} frames for this duration)"
                )
        elif engine == "ltx23":
            if type(additional) is not int or not 8 <= additional <= 720 or additional % 8:
                raise ValueError(
                    "LTX 2.3 extension.additional_frames must be a multiple of 8 in [8, 720]"
                )
            if "context_frames" in extension:
                raise ValueError("LTX 2.3 extension does not accept context_frames")
        else:
            if extension["direction"] != "after":
                raise ValueError("LTX 2.5 continuation supports extension after the source only")
            if type(additional) is not int or not 8 <= additional <= 720 or additional % 8:
                raise ValueError(
                    "LTX 2.5 extension.additional_frames must be a multiple of 8 in [8, 720]"
                )
            context = extension.get("context_frames")
            if type(context) is not int or not 9 <= context <= 241 or (context - 1) % 8:
                raise ValueError(
                    "LTX 2.5 extension.context_frames must equal 8n+1 in [9, 241]"
                )
    elif extension is not None:
        raise ValueError("conditioning.extension is only valid for the extension task")
    if not isinstance(result["inputs"], list) or len(result["inputs"]) > 12:
        raise ValueError("conditioning.inputs must be a list with at most twelve entries")
    num_frames, _fps = frame_geometry(recipe)
    ids = set()
    anchors = set()
    for item in result["inputs"]:
        _fields(
            item,
            {
                "id",
                "kind",
                "role",
                "path",
                "frame_index",
                "strength",
                "control_type",
                "soundtrack_path",
                "description",
                "reference_role",
                "attention_strength",
                "reference_frames",
                "reference_size_policy",
                "reference_priority",
            },
            "input",
        )
        identity = item.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Each conditioning input needs a unique nonempty id")
        ids.add(identity)
        if item.get("kind") not in {"image", "video", "audio"}:
            raise ValueError(f"{identity}: kind must be image, video, or audio")
        if item.get("role") not in {"keyframe", "reference", "audio_driver", "control"}:
            raise ValueError(f"{identity}: unsupported media role")
        path = item.get("path")
        if not isinstance(path, str) or not path or "://" in path:
            raise ValueError(f"{identity}: path must name a local file")
        if check_files and not Path(path).is_file():
            raise FileNotFoundError(f"{identity}: media file not found: {path}")
        if "soundtrack_path" in item:
            soundtrack = item["soundtrack_path"]
            if engine != "h3" or item["kind"] != "video" or item["role"] != "reference":
                raise ValueError("soundtrack_path is only supported for H3 video references")
            if not isinstance(soundtrack, str) or not soundtrack or "://" in soundtrack:
                raise ValueError(f"{identity}: soundtrack_path must name a local audio source")
            if check_files and not Path(soundtrack).is_file():
                raise FileNotFoundError(f"{identity}: soundtrack file not found: {soundtrack}")
        item.setdefault("strength", 1.0)
        _number(item["strength"], f"{identity}.strength", 0.0, 1.0)
        if "frame_index" in item:
            frame = item["frame_index"]
            if frame == "last":
                frame = num_frames - 1
            if type(frame) is not int or not 0 <= frame < num_frames:
                raise ValueError(f"{identity}: frame_index must be in 0..{num_frames - 1} or last")
            item["frame_index"] = frame
        if item["role"] == "keyframe":
            if item["kind"] != "image" or "frame_index" not in item:
                raise ValueError(f"{identity}: keyframes require an image and frame_index")
            if item["frame_index"] in anchors:
                raise ValueError("Two keyframes target the same frame")
            anchors.add(item["frame_index"])
        elif "frame_index" in item and not (engine == "h3" and item["role"] == "reference"):
            raise ValueError(f"{identity}: timed placement is not implemented for this role")
        if item["role"] == "audio_driver" and item["kind"] != "audio":
            raise ValueError(f"{identity}: audio_driver requires audio")
        if item["role"] == "control":
            if item.get("control_type") not in CONTROL_FAMILIES or item["kind"] not in {
                "image",
                "video",
            }:
                raise ValueError(
                    f"{identity}: control requires a supported control_type and visual media"
                )
        elif "control_type" in item:
            raise ValueError(f"{identity}: control_type is only valid for a control input")
    msr_fields = {
        "description",
        "reference_role",
        "attention_strength",
        "reference_frames",
        "reference_size_policy",
        "reference_priority",
    }
    is_ltx25_msr = (
        engine == "ltx25"
        and result["task"] == "ref2va"
        and bool(result["inputs"])
        and all(item["role"] == "reference" for item in result["inputs"])
    )
    if is_ltx25_msr:
        if not 1 <= len(result["inputs"]) <= 5:
            raise ValueError("LTX 2.5 MSR requires one to five reference images")
        if any(item["kind"] != "image" for item in result["inputs"]):
            raise ValueError("LTX 2.5 MSR accepts still-image references only")
        for item in result["inputs"]:
            identity = item["id"]
            description = item.get("description")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"{identity}: MSR requires a nonempty description")
            item["description"] = description.strip()
            if item.get("reference_role") not in {
                "subject",
                "object",
                "clothing",
                "background",
            }:
                raise ValueError(
                    f"{identity}: MSR reference_role must be subject, object, clothing, "
                    "or background"
                )
            item.setdefault("attention_strength", 1.0)
            _number(item["attention_strength"], f"{identity}.attention_strength", 0.0, 1.0)
            item.setdefault("reference_frames", "auto")
            if str(item["reference_frames"]) not in {"auto", "25", "33"}:
                raise ValueError(f"{identity}: MSR reference_frames must be auto, 25, or 33")
            item["reference_frames"] = str(item["reference_frames"])
            item.setdefault("reference_size_policy", "sol_auto")
            if item["reference_size_policy"] not in {
                "sol_auto",
                "quality",
                "balanced",
                "speed",
            }:
                raise ValueError(f"{identity}: unsupported MSR reference_size_policy")
            item.setdefault("reference_priority", "auto")
            if item["reference_priority"] not in {
                "auto",
                "primary",
                "supporting",
                "background",
            }:
                raise ValueError(f"{identity}: unsupported MSR reference_priority")
        backgrounds = [
            item for item in result["inputs"] if item["reference_role"] == "background"
        ]
        if len(backgrounds) > 1:
            raise ValueError("LTX 2.5 MSR accepts at most one background reference")
        ordered = [
            item for item in result["inputs"] if item["reference_role"] != "background"
        ] + backgrounds
        for index, item in enumerate(ordered):
            if item["reference_priority"] == "auto":
                item["reference_priority"] = (
                    "background"
                    if item["reference_role"] == "background" or index >= 4
                    else "supporting"
                    if index >= 2
                    else "primary"
                )
        result["inputs"] = ordered
    elif any(msr_fields & set(item) for item in result["inputs"]):
        raise ValueError("MSR reference fields are only valid for LTX 2.5 Ref2VA image inputs")
    return result


def ltx25_msr_prompt_guide(contract):
    """Build the explicit learned-slot labels used by the LTX 2.5 MSR prompt."""

    if contract["task"] != "ref2va" or not contract["inputs"]:
        return ""
    if not all("reference_role" in item for item in contract["inputs"]):
        return ""
    return "\n".join(
        f"Image {index} provides the {item['reference_role']}: {item['description']}"
        for index, item in enumerate(contract["inputs"], start=1)
    )


def apply_ltx25_msr_prompt_guide(prompt, guide):
    """Prepend the canonical MSR slot guide once; preserve already-prefixed prompts."""

    prompt = str(prompt).strip()
    guide = str(guide).strip()
    if not guide or prompt.startswith(guide):
        return prompt
    return f"{guide}\n{prompt}"


def validate_conditioning(recipe, contract=None, *, check_files=True):
    """Validate task/transport compatibility independently of weighted model validation."""
    c = normalize_conditioning(recipe, check_files=check_files) if contract is None else contract
    engine, task, inputs = recipe["engine"], c["task"], c["inputs"]
    roles = {item["role"] for item in inputs}
    config = recipe["config"]
    allowed = {
        "t2v": set(),
        "fflf": {"keyframe"},
        "ref2va": {"reference", "control"},
        "a2v": {"audio_driver", "keyframe"},
        "extension": {"reference"},
        "control": {"control", "keyframe"},
    }[task]
    if roles - allowed or (task != "t2v" and not inputs):
        raise ValueError(f"{task}: missing required media or incompatible input roles")
    required = {"fflf": "keyframe", "a2v": "audio_driver", "control": "control"}.get(task)
    if required and required not in roles:
        raise ValueError(f"{task} requires a {required} input")
    expected_audio_policy = _task_audio_policy(engine, task)
    if c["audio_policy"] != expected_audio_policy:
        raise ValueError(f"{task} requires audio_policy={expected_audio_policy}")
    if task == "a2v" and sum(i["role"] == "audio_driver" for i in inputs) != 1:
        raise ValueError("A2V requires exactly one audio_driver")
    if task == "extension" and (
        engine not in {"h3", "ltx23", "ltx25"}
        or len(inputs) != 1
        or inputs[0]["role"] != "reference"
        or inputs[0]["kind"] != "video"
    ):
        if engine not in {"h3", "ltx23", "ltx25"}:
            raise ValueError(f"{engine} existing-video extension transport is not implemented")
        raise ValueError(f"{engine} extension requires exactly one source video reference")
    if task == "extension" and inputs[0]["strength"] != 1:
        raise ValueError("Extension source strength is fixed at 1")
    if engine == "h3":
        expected = {
            "t2v": "t2va",
            "fflf": "fl2va",
            "ref2va": "ref2va",
            "a2v": "ref2va",
            "extension": "ref2va",
            "control": "t2va",
        }.get(task)
        if expected is None:
            raise ValueError(f"H3 {task} is not implemented by the selected component path")
        if recipe["components"].get("task") != expected:
            raise ValueError(f"H3 {task} requires components.task={expected}")
        if task == "extension":
            if float(config.get("duration_seconds", 0)) < 4.0:
                raise ValueError("H3 external extension requires a 4-15 second Ref2VA window")
            required_prompt_markers = (
                "subject_definitions:",
                "summary:",
                "[video continuation",
                "retention_analysis:",
                "detailed_description:",
                "overall_soundscape:",
                "non_diegetic_music:",
                "<Video 1>",
                "<Picture 1>",
            )
            prompt = recipe.get("prompt", "")
            missing = [marker for marker in required_prompt_markers if marker not in prompt]
            if missing:
                raise ValueError(
                    "H3 external extension requires the released Ref2VA continuation prompt "
                    "structure; missing: " + ", ".join(missing)
                )
        if "control" in roles:
            control_inputs = [item for item in inputs if item["role"] == "control"]
            if task != "control" or len(control_inputs) != 1:
                raise ValueError("H3 Fun ControlNet requires exactly one control input")
            item = control_inputs[0]
            if item["kind"] != "video" or item["control_type"] not in {
                "canny_edges",
                "depth_map",
                "hed_edges",
                "mlsd_lines",
                "pose_skeleton",
            }:
                raise ValueError(
                    "H3 Fun ControlNet accepts one preprocessed Canny, depth, HED, MLSD, "
                    "or pose video"
                )
            checkpoint = recipe["components"].get("fun_controlnet")
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError("H3 control requires components.fun_controlnet")
            if check_files and not Path(checkpoint).expanduser().is_file():
                raise FileNotFoundError(f"H3 Fun ControlNet checkpoint not found: {checkpoint}")
        if task != "t2v" and (
            recipe.get("attention") or recipe.get("fastvideo") or recipe.get("vdn")
        ):
            raise ValueError("Accelerated H3 conditioning is unqualified; use the native task path")
        if any(i["strength"] != 1.0 for i in inputs if i["role"] != "control"):
            raise ValueError(
                "H3 media strength is fixed at 1; fractional conditioning is not implemented"
            )
        if task == "fflf" and len(inputs) > 8:
            raise ValueError("H3 FL2VA supports at most eight keyframes")
        if task == "ref2va":
            for kind, maximum in (("image", 9), ("video", 3), ("audio", 3)):
                count = sum(i["kind"] == kind for i in inputs)
                if kind == "audio":
                    count += sum("soundtrack_path" in i for i in inputs)
                if count > maximum:
                    raise ValueError(f"H3 Ref2VA supports at most {maximum} {kind} references")
            if not any(i["kind"] != "audio" for i in inputs) and any(
                "frame_index" not in i for i in inputs
            ):
                raise ValueError("Untimed H3 audio references require an image or video reference")
    else:
        if task != "t2v" and (
            config.get("dfr_enabled") or config.get("duration_mode", "manual") != "manual"
        ):
            raise ValueError(
                "Conditioned headless tasks currently require manual duration and DFR disabled"
            )
        if any(i.get("control_type") in {"hed_edges", "mlsd_lines"} for i in inputs):
            raise ValueError("HED and MLSD control types are specific to H3 Fun ControlNet")
        ltx25_msr = (
            engine == "ltx25"
            and task == "ref2va"
            and bool(inputs)
            and all(i["role"] == "reference" for i in inputs)
        )
        if task != "extension" and any(i["role"] == "reference" for i in inputs) and not ltx25_msr:
            raise ValueError(
                "LTX references require an explicit IC-LoRA control role or LTX 2.5 MSR adapter"
            )
        if any(i["role"] == "audio_driver" and i["strength"] != 1 for i in inputs):
            raise ValueError("Frozen audio strength is fixed at 1")
        if engine == "ltx23":
            ic = recipe["components"].get("ic_loras", [])
            if task in {"control", "ref2va"} or ic:
                if task not in {"control", "ref2va"}:
                    raise ValueError("LTX 2.3 IC-LoRA requires a control/reference task")
                if not isinstance(ic, list) or len(ic) != 1 or not isinstance(ic[0], dict):
                    raise ValueError("LTX 2.3 control requires one explicit IC-LoRA adapter")
                if len(inputs) != 1:
                    raise ValueError("LTX 2.3 IC-LoRA requires exactly one guide")
                family = CONTROL_FAMILIES[inputs[0]["control_type"]]
                if family not in set(CONTROL_FAMILIES.values()) or ic[0].get("family") != family:
                    raise ValueError(
                        "LTX 2.3 control guide/IC-LoRA family mismatch or unsupported family"
                    )
                ingredients = family == "ingredients_reference_sheet"
                expected_mode = "two_stage" if ingredients else "distilled"
                if config.get("pipeline_mode") != expected_mode:
                    label = "Dev two_stage" if ingredients else "distilled"
                    raise ValueError(f"LTX 2.3 {family} requires the resident {label} pipeline")
                generic = recipe.get("loras")
                has_generic = bool(recipe["components"].get("loras")) or (
                    isinstance(generic, dict) and bool(generic.get("adapters"))
                )
                if config.get("low_ram_streaming") or has_generic:
                    raise ValueError(
                        "LTX 2.3 IC-LoRA requires resident loading without generic LoRAs"
                    )
                if inputs[0]["kind"] != ("image" if ingredients else "video"):
                    raise ValueError("Ingredients requires an image; other controls require video")
                if (task == "ref2va") != ingredients:
                    raise ValueError("LTX 2.3 Ref2VA is the Ingredients reference-sheet task")
                factor = 1 if ingredients else 2
                if config.get("width", 704) % (64 * factor) or config.get("height", 448) % (
                    64 * factor
                ):
                    raise ValueError("LTX 2.3 IC-LoRA dimensions do not match its reference scale")
                if ingredients and (
                    (config.get("width"), config.get("height")) != (768, 448)
                    or frame_geometry(recipe)[0] < 121
                    or frame_geometry(recipe)[1] != 24
                ):
                    raise ValueError(
                        "LTX 2.3 Ingredients requires 768x448, at least 121 frames, and 24 fps"
                    )
                prompt = recipe.get("prompt", "")
                if ingredients and not (
                    "Reference sheet:" in prompt and "Generated video:" in prompt
                ):
                    raise ValueError(
                        "Ingredients prompt requires Reference sheet: and Generated video:"
                    )
            if task in {"fflf", "a2v"} and config.get("pipeline_mode", "two_stage") != "two_stage":
                raise ValueError(
                    "LTX 2.3 FFLF/A2V require the Dev two_stage pipeline and refinement weights"
                )
            if task == "extension" and config.get("pipeline_mode") not in {
                "one_stage",
                "distilled",
            }:
                raise ValueError(
                    "LTX 2.3 extension requires the resident Dev one_stage or distilled pipeline"
                )
            if (
                task == "extension"
                and config.get("pipeline_mode") == "distilled"
                and config.get("stage1_steps") != 8
            ):
                raise ValueError("LTX 2.3 distilled extension requires exactly eight steps")
            if task == "a2v" and "keyframe" in roles:
                raise ValueError("LTX 2.3 combined A2V/keyframe transport is not qualified yet")
        else:
            if task == "extension":
                if config.get("pipeline_mode") != "distilled":
                    raise ValueError("LTX 2.5 extension currently requires distilled sampling")
                if config.get("ic_lora_single_stage"):
                    raise ValueError("LTX 2.5 extension currently requires the two-stage pipeline")
                if recipe["components"].get("ic_loras") or recipe["components"].get(
                    "msr_lora_path"
                ):
                    raise ValueError("LTX 2.5 extension cannot be combined with IC-LoRA or MSR")
            if ltx25_msr:
                components = recipe["components"]
                path = components.get("msr_lora_path")
                ic_loras = components.get("ic_loras", [])
                if not isinstance(path, str) or not path:
                    raise ValueError("LTX 2.5 MSR requires components.msr_lora_path")
                if (
                    not isinstance(ic_loras, list)
                    or len(ic_loras) != 1
                    or not isinstance(ic_loras[0], (list, tuple))
                    or len(ic_loras[0]) != 2
                    or Path(str(ic_loras[0][0])).expanduser().resolve()
                    != Path(path).expanduser().resolve()
                ):
                    raise ValueError(
                        "LTX 2.5 MSR adapter must be the single components.ic_loras entry"
                    )
                if config.get("pipeline_mode") != "distilled" or not config.get(
                    "ic_lora_single_stage"
                ):
                    raise ValueError(
                        "LTX 2.5 MSR requires distilled full-resolution single-stage mode"
                    )
            for item in inputs:
                if item["role"] != "control":
                    continue
                sheet = item["control_type"] == "ingredients_reference_sheet"
                if item["kind"] != ("image" if sheet else "video"):
                    raise ValueError(
                        "Ingredients requires a reference-sheet image; "
                        "other controls require preprocessed video"
                    )
                if sheet and frame_geometry(recipe)[0] < 121:
                    raise ValueError("Ingredients requires at least 121 output frames")
            if task == "ref2va" and not ltx25_msr and any(
                i.get("control_type") != "ingredients_reference_sheet" for i in inputs
            ):
                raise ValueError(
                    "LTX Ref2VA transport currently supports Ingredients reference sheets"
                )
    result = {
        "contract": c,
        "transport": "implemented",
        "render_qualification": "not_evaluated",
        "input_ids": [i["id"] for i in inputs],
    }
    if engine == "h3" and task == "a2v":
        result["audio_semantics"] = (
            "reference_audio_drives generated video and generated soundtrack; "
            "the source waveform is not copied"
        )
    guide = ltx25_msr_prompt_guide(c) if engine == "ltx25" else ""
    if guide:
        result["prompt_guide"] = guide
    return result


def validate_ltx25_control_families(contract, engine_report):
    families = {
        item.get("adapter_family")
        for item in engine_report.get("components", [])
        if str(item.get("component", "")).startswith("ic_lora_")
        or item.get("component") == "baked_ic_lora"
    }
    for item in contract["inputs"]:
        if item["role"] == "control":
            required = CONTROL_FAMILIES[item["control_type"]]
            if required not in families:
                raise ValueError(
                    f"{item['id']}: guide requires IC-LoRA family {required}; "
                    f"found {sorted(str(f) for f in families)}"
                )
