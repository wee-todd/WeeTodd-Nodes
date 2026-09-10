"""Pure, lightweight generation selection shared by Studio and exported native jobs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path


def infer_task(clip):
    if clip.get("extensionDirection"):
        return "extension"
    roles = {a["role"] for a in clip.get("attachments", [])}
    for role, task in (("control", "control"), ("audioDriver", "a2v"), ("reference", "ref2va")):
        if role in roles:
            return task
    return "fflf" if roles & {"first", "last", "keyframe"} else "t2v"


def supported_tasks(recipe):
    engine, config = recipe["engine"], recipe.get("config", {})
    components = recipe.get("components", {})
    if engine == "h3":
        tasks = {
            "t2va": ["t2v", "i2v", "fflf"],
            "fl2va": ["t2v", "i2v", "fflf"],
            "ref2va": ["ref2va", "a2v", "extension"],
        }.get(components.get("task"), [])
        if recipe.get("conditioning", {}).get("task") == "control":
            tasks = ["control"]
        return tasks
    if engine == "ltx23":
        ic = components.get("ic_loras", [])
        if ic:
            return ["ref2va" if ic[0].get("family") == "ingredients_reference_sheet" else "control"]
        return (
            ["t2v", "i2v", "fflf", "a2v"]
            if config.get("pipeline_mode", "two_stage") == "two_stage"
            else ["t2v", "extension"]
            if config.get("pipeline_mode") in {"one_stage", "distilled"}
            else ["t2v"]
        )
    if engine == "ltx25":
        task = recipe.get("conditioning", {}).get("task", "t2v")
        if (
            components.get("ic_loras")
            or config.get("ic_lora_single_stage")
            or task in {"control", "ref2va"}
        ):
            return [task]
        return ["t2v", "i2v", "fflf", "a2v", "extension"]
    return []


def h3_checkpoint_sampling(recipe):
    """Inspect bounded provenance JSON, never tensor payloads or checkpoint filenames."""
    if recipe.get("engine") != "h3":
        return {}
    component = recipe.get("components", {}).get("transformer")
    if not component:
        return {}
    root = Path(component).expanduser()
    result = {}
    for name in ("paged_manifest.json", "model_identity.json", "conversion_provenance.json"):
        source = root / name
        if not source.is_file():
            continue
        if source.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Checkpoint provenance exceeds the 4 MiB inspection limit.")
        document = json.loads(source.read_text())
        if not isinstance(document, dict):
            raise ValueError("Checkpoint provenance must be a JSON object.")
        records = [document]
        if isinstance(document.get("metadata"), dict):
            records.append(document["metadata"])
        for record in records:
            declared_source = str(record.get("source", "")).lower()
            if any(marker in declared_source for marker in ("fasth3", "vdn-h3", "turbo")):
                result["special"] = True
            sampling = record.get("sampling")
            if isinstance(sampling, dict) and sampling:
                result["special"] = True
                for key in ("schedule_points", "transformer_evaluations"):
                    value = sampling.get(key)
                    if value is not None:
                        if type(value) is not int or value < 1:
                            raise ValueError(
                                "Checkpoint sampling counts must be positive integers."
                            )
                        if key in result and result[key] != value:
                            raise ValueError(
                                "Checkpoint provenance has conflicting sampling counts."
                            )
                        result[key] = value
    return result


def special_h3_sampling(recipe):
    if recipe.get("engine") != "h3":
        return False
    if h3_checkpoint_sampling(recipe).get("special"):
        return True
    if any(recipe.get(key) for key in ("attention", "fastvideo", "vdn")):
        return True
    for adapter in recipe.get("loras", {}).get("adapters", []):
        if adapter.get("profile") == "standard":
            continue
        if adapter.get("profile") == "turbo":
            return True
        try:
            from wee_todd_nodes.lora import H3LoRASpec

            if H3LoRASpec(**adapter).resolved_profile != "standard":
                return True
        except (OSError, ValueError, TypeError, KeyError):
            # Unknown adapters must not make the generic schedule editable.
            return True
    return False


def generation_descriptor(recipe):
    engine, config = recipe["engine"], recipe.get("config", {})
    h3 = engine == "h3"
    mode = config.get("pipeline_mode", "distilled" if engine == "ltx25" else "two_stage")
    ordinary = (
        h3
        and not special_h3_sampling(recipe)
        and config.get("sampling_method", "euler") == "euler"
        and not any(config.get(key) for key in ("sigmas", "custom_sigmas", "turbo"))
    )
    steps = config.get("steps", 16) - 1 if ordinary else None
    checkpoint_sampling = h3_checkpoint_sampling(recipe)
    if h3 and checkpoint_sampling.get("transformer_evaluations"):
        steps = checkpoint_sampling["transformer_evaluations"]
    refinement = None
    if not h3:
        steps = config.get("stage1_steps", 8 if engine == "ltx25" else 30)
        refinement = config.get("stage2_steps", 3) if mode != "one_stage" else None
        if engine == "ltx25" and config.get("stage1_sampler") == "euler_ancestral_cfg_pp":
            from ltx25_mlx.runtime import LTX25_CFG_PP_SCHEDULES

            steps += len(LTX25_CFG_PP_SCHEDULES[config.get("cfg_pp_schedule", "full")])
    ic = recipe.get("components", {}).get("ic_loras", []) if engine == "ltx23" else []
    if engine == "ltx23" and (mode == "distilled" or ic):
        steps = min(steps, 8)
        refinement = min(refinement, 3) if refinement is not None else None
        topology = config.get("ic_lora_topology", "auto")
        if ic and topology == "control_refine":
            refinement = min(config.get("stage2_steps", 3), 8)
        if topology in {"single_stage", "upsample_only"} or (
            topology == "auto" and ic and ic[0].get("family") == "motion_track"
        ):
            refinement = None
    cfg_editable = not h3 and mode != "distilled" and not ic
    controls = {
        "evaluations": steps,
        "refinementSteps": refinement,
        "cfg": None
        if h3 or ic
        else config.get("video_cfg_scale", 1.0)
        if engine == "ltx25"
        else config.get("cfg_scale", 3.0),
        "shift": None,
        "stepsEditable": ordinary or (engine == "ltx23" and mode != "distilled" and not ic),
        "refinementStepsEditable": engine == "ltx23"
        and refinement is not None
        and mode != "distilled"
        and not ic,
        "cfgEditable": cfg_editable,
        "shiftEditable": False,
        "stepsExplanation": "Actual Euler evaluations; the recipe stores one extra schedule point."
        if ordinary
        else "Stage-one schedule steps. Fixed schedules cannot be overridden."
        if not h3
        else "Fixed checkpoint schedule; preserve its declared evaluation count."
        if checkpoint_sampling.get("transformer_evaluations")
        else "Custom schedule: evaluation count is not inferred.",
        "cfgExplanation": "Native H3 uses distilled guidance; CFG overrides are unsupported."
        if h3
        else "Video CFG; audio CFG retains the recipe value."
        if engine == "ltx25" and cfg_editable
        else "Distilled guidance is fixed."
        if not cfg_editable
        else "Guidance scale for the selected native pipeline.",
        "shiftExplanation": "This native adapter does not expose a Shift override.",
    }
    return {
        "supportedTasks": supported_tasks(recipe),
        "controls": controls,
        "presets": [
            {"id": "custom", "name": "Custom", "description": "Preserve imported recipe settings."},
            {
                "id": "balanced",
                "name": "Balanced",
                "description": "Preserve validated recipe sampling settings.",
            },
            {
                "id": "speed",
                "name": "Speed",
                "description": "Keep recipe sampling pending faster-settings qualification.",
            },
            {
                "id": "lowMemory",
                "name": "Low memory",
                "description": "Staged loading; hardware qualification remains separate.",
            },
        ],
    }


def fingerprint(recipe):
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resolve_generation_selection(selection, clip, profiles, capabilities):
    """Resolve recipe content, never filename hints; leave caller-owned objects untouched."""
    selection = dict(selection or {})
    unknown = set(selection) - {
        "task",
        "preset",
        "steps",
        "refinementSteps",
        "cfg",
        "shift",
        "memoryPolicy",
        "projectionBackend",
    }
    if unknown:
        raise ValueError(f"Unsupported generation controls: {sorted(unknown)}")
    explicit = bool(selection)
    acceleration = None
    if explicit and clip["engine"] == "h3" and selection.get("preset", "custom") != "custom":
        from wee_todd_mlx.acceleration import resolve_h3_acceleration

        acceleration = resolve_h3_acceleration(
            capabilities.get("acceleration"), capabilities.get("hardware")
        )
        selection.setdefault("projectionBackend", acceleration["projectionBackend"])
        if selection.get("preset") != "lowMemory":
            selection.setdefault("memoryPolicy", acceleration["memoryPolicy"])
    task = selection.get("task") or infer_task(clip)
    required = {
        "i2v": ["first"],
        "fflf": ["first", "last"],
        "ref2va": ["reference"],
        "a2v": ["audioDriver"],
        "control": ["control"],
        "extension": ["source"],
    }.get(task, [])
    if explicit:
        roles = {a["role"] for a in clip.get("attachments", [])} - {"lora"}
        labels = {
            "i2v": "Image to video",
            "fflf": "First and last frames",
            "ref2va": "Reference to video",
            "a2v": "Audio to video",
            "control": "Control to video",
            "extension": "Video extension",
        }
        input_labels = {
            "first": "First frame image",
            "last": "Last frame image",
            "reference": "Reference media",
            "audioDriver": "Audio driver",
            "control": "Control guide",
            "source": "Source movie",
        }
        missing = []
        for role in required:
            present = (
                bool(clip.get("extensionSource") or clip.get("sourcePath"))
                if role == "source"
                else role in roles
            )
            if not present:
                missing.append(input_labels[role])
        if missing:
            raise ValueError(f"{labels.get(task, task)} requires {' and '.join(missing)}.")
        allowed = {
            "t2v": set(),
            "i2v": {"first"},
            "fflf": {"first", "last", "keyframe"},
            "ref2va": {"reference"},
            "a2v": {"audioDriver", "first", "last", "keyframe"},
            "control": {"control"},
            "extension": set(),
        }.get(task, set())
        conflicts = roles - allowed
        if conflicts:
            raise ValueError(
                f"{task} conflicts with attached {', '.join(sorted(conflicts))}; "
                "attachments are preserved. Remove them or select their task."
            )
    candidates = []
    for profile in profiles:
        recipe = profile.get("recipe")
        if recipe is None:
            recipe = json.loads(Path(profile["id"]).read_text())
        if recipe.get("engine") == clip["engine"]:
            candidates.append((profile, recipe))
    selected = clip.get("profileID") or "auto"
    if selected != "auto":
        candidates = [(p, r) for p, r in candidates if p["id"] == selected]
        if not candidates:
            raise ValueError("The selected model recipe is missing. Reimport or select Automatic.")
    compatible = [(p, r) for p, r in candidates if task in supported_tasks(r)]
    if not explicit and clip["engine"] == "h3":
        native_task = {
            "t2v": "t2va",
            "fflf": "fl2va",
            "ref2va": "ref2va",
            "a2v": "ref2va",
            "extension": "ref2va",
            "control": "t2va",
        }.get(task)
        compatible = [
            (p, r) for p, r in compatible if r.get("components", {}).get("task") == native_task
        ]
    if clip["engine"] == "ltx25" and task == "control":
        from wee_todd_mlx.model_setup import ltx25_recipe_control_families
        from wee_todd_mlx.task_conditioning import CONTROL_FAMILIES

        required_families = {
            CONTROL_FAMILIES.get(a.get("controlType", "canny_edges"))
            for a in clip.get("attachments", [])
            if a["role"] == "control"
        }
        checked = []
        for candidate, content in compatible:
            try:
                families = ltx25_recipe_control_families(candidate["id"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if None not in required_families and required_families <= families:
                checked.append((candidate, content))
        compatible = checked
    if not compatible:
        raise ValueError(
            f"{clip['engine']} {task} conflicts with the selected recipe. "
            "Import a compatible recipe or select Automatic. No inputs were discarded."
        )
    if explicit and selected == "auto":
        compatible.sort(key=lambda pair: special_h3_sampling(pair[1]))
        if selection.get("preset", "custom") != "custom" and special_h3_sampling(compatible[0][1]):
            raise ValueError("Import a standard native recipe for this preset, or select Custom.")
    profile, original = compatible[0]
    recipe = copy.deepcopy(original)
    if explicit and clip["engine"] == "h3" and task in {"t2v", "i2v", "fflf"}:
        recipe["components"]["task"] = "t2va" if task == "t2v" else "fl2va"
    if explicit:
        recipe.setdefault("conditioning", {})["task"] = "fflf" if task == "i2v" else task
    config = recipe.setdefault("config", {})
    checkpoint_sampling = h3_checkpoint_sampling(recipe)
    fixed_points = checkpoint_sampling.get("schedule_points")
    if fixed_points is not None and config.get("steps", 16) != fixed_points:
        raise ValueError(f"This checkpoint requires a fixed schedule of {fixed_points} points.")
    preset = selection.get("preset", "custom")
    if preset not in {"custom", "balanced", "speed", "lowMemory"}:
        raise ValueError(f"Unknown generation preset: {preset}")

    def describe_resolved():
        inspected = copy.deepcopy(recipe)
        if clip["engine"] == "h3" and capabilities.get("attached_loras"):
            inspected.setdefault("loras", {}).setdefault("adapters", []).extend(
                capabilities["attached_loras"]
            )
        return generation_descriptor(inspected)

    descriptor = describe_resolved()
    controls = descriptor["controls"]
    for key in ("steps", "refinementSteps", "cfg", "shift"):
        value = selection.get(key)
        if value is None:
            continue
        if not controls[key + "Editable"]:
            raise ValueError(
                f"{key} override is unsupported. "
                + controls.get(key + "Explanation", "The selected native schedule is fixed.")
            )
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value < (0 if key == "cfg" else 1)
        ):
            raise ValueError(f"{key} must be a finite valid positive value.")
        if key in {"steps", "refinementSteps"} and (type(value) is not int or value > 1000):
            raise ValueError(f"{key} must be an integer between 1 and 1000.")
        target = {
            "steps": "steps" if clip["engine"] == "h3" else "stage1_steps",
            "refinementSteps": "stage2_steps",
            "cfg": "video_cfg_scale" if clip["engine"] == "ltx25" else "cfg_scale",
        }[key]
        config[target] = value + 1 if key == "steps" and clip["engine"] == "h3" else value
    policy = selection.get("memoryPolicy", "paged" if preset == "lowMemory" else "recipe")
    if policy not in {"recipe", "paged", "pagedNormal", "resident"}:
        raise ValueError("Unknown memoryPolicy; use recipe, paged, pagedNormal, or resident.")
    if policy == "pagedNormal" and clip["engine"] != "h3":
        raise ValueError("pagedNormal is supported for native H3 only.")
    if policy != "recipe":
        if clip["engine"] == "h3":
            if policy == "resident" and config.get("paging_cache_gb", 0) > 0:
                raise ValueError("Resident sampling requires paging cache budget zero.")
            config["memory_mode"] = (
                "normal" if policy in {"resident", "pagedNormal"} else "low_memory_bf16"
            )
            recipe["block_residency"] = "resident" if policy == "resident" else "checkpoint_default"
        else:
            config.update(low_memory=True, low_ram_streaming=policy == "paged")
    backend = selection.get("projectionBackend")
    if backend is not None:
        if clip["engine"] != "h3" or backend not in {"mlx", "auto"}:
            raise ValueError("projectionBackend supports mlx or auto for native H3 only.")
        config["projection_backend"] = backend
    descriptor = describe_resolved()
    if acceleration:
        from wee_todd_mlx.acceleration import effective_h3_acceleration_report

        descriptor["acceleration"] = effective_h3_acceleration_report(
            acceleration, policy, config.get("projection_backend", "auto")
        )
    warnings = (
        ["Speed keeps the recipe sampling settings pending measured qualification."]
        if preset == "speed"
        else []
    )
    return {
        "recipe": recipe,
        "resolved_controls": descriptor["controls"],
        "generation": descriptor,
        "required_inputs": required,
        "warnings": warnings,
        "fingerprint": fingerprint(recipe),
        "profileID": profile["id"],
        "task": task,
    }
