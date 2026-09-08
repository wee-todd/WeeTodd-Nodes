#!/usr/bin/env python3
"""Host-independent H3/LTX recipe runner; no ComfyUI graph or node imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from render_h3_headless import NoComfyImports, assert_isolated


def assemble_external_extension(
    source,
    generated,
    target,
    *,
    source_frames,
    context_frames,
    fps,
    sample_rate,
    ffmpeg,
):
    """Append a generated window after removing its repeated continuation context."""

    from wee_todd_mlx.conditioning_media import assemble_external_extension as assemble

    return assemble(
        source,
        generated,
        target,
        source_frames=source_frames,
        context_frames=context_frames,
        fps=fps,
        sample_rate=sample_rate,
        ffmpeg=ffmpeg,
    )


def assemble_h3_extension(
    source, generated, target, *, source_frames, context_frames, fps, ffmpeg
):
    """Backward-compatible H3 assembly entry point."""

    return assemble_external_extension(
        source,
        generated,
        target,
        source_frames=source_frames,
        context_frames=context_frames,
        fps=fps,
        sample_rate=32000,
        ffmpeg=ffmpeg,
    )


def validate_h3_adapter_keys(names):
    """Do not partially apply a foreign adapter while silently discarding extra tensors."""
    endings = (
        ".lora_A.turbo.weight",
        ".lora_B.turbo.weight",
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".alpha",
        ".diff",
        ".diff_b",
    )
    unknown = [name for name in names if not name.endswith(endings)]
    if unknown:
        raise ValueError(
            "This H3 adapter contains unsupported tensor fields; refusing partial application. "
            "A format-specific converter or runtime implementation is required: "
            + ", ".join(unknown[:5])
        )


def render_h3(recipe, target):
    from wee_todd_mlx.conditioning_media import (
        inspect_media,
        read_embedded_video_audio,
        read_media,
    )
    from wee_todd_mlx.task_conditioning import validate_conditioning

    task_report = validate_conditioning(recipe)
    contract = task_report["contract"]
    media_report = inspect_media(recipe, contract)
    from profile_fasth3_server import install_profiling

    from wee_todd_nodes.conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
    from wee_todd_nodes.decoding import (
        AUDIO_VAE_RUNTIME,
        VIDEO_VAE_RUNTIME,
        H3AudioVAESpec,
        H3VideoVAESpec,
    )
    from wee_todd_nodes.direct_publishing import publish_latents_direct
    from wee_todd_nodes.preflight import (
        H3ComponentSetSpec,
        H3PreflightRequest,
        preflight_components,
    )
    from wee_todd_nodes.preview import H3PreviewConfig
    from wee_todd_nodes.residency import prepare_low_memory_stage
    from wee_todd_nodes.runtime import RUNTIME, H3GenerationConfig
    from wee_todd_nodes.sampling import TRANSFORMER_RUNTIME, H3TransformerSpec

    fields = dict(recipe["components"])
    preview = fields.pop("preview_override", None)
    fun_controlnet_path = fields.pop("fun_controlnet", None)
    components = H3ComponentSetSpec(
        **fields, preview_override=H3PreviewConfig(**preview) if preview else None
    )
    config = H3GenerationConfig(**recipe["config"])
    config.validate()
    preflight_components(
        components,
        H3PreflightRequest(
            duration_seconds=config.duration_seconds,
            steps=config.steps,
            width=config.width,
            height=config.height,
        ),
    )
    runtimes = (
        TEXT_ENCODER_RUNTIME,
        TRANSFORMER_RUNTIME,
        VIDEO_VAE_RUNTIME,
        AUDIO_VAE_RUNTIME,
        RUNTIME,
    )

    def prepare(stage):
        return prepare_low_memory_stage(stage, config.memory_mode)

    attention = None
    if recipe.get("attention"):
        from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig

        fields = dict(recipe["attention"])
        for name in ("prefix_segments", "video_grid"):
            fields[name] = tuple(fields.get(name, ()))
        attention = FastH3VSAConfig(**fields)
    fastvideo = None
    if recipe.get("fastvideo"):
        from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig

        fastvideo = FastH3ApproximationConfig(**recipe["fastvideo"])
    vdn = loras = None
    if recipe.get("loras"):
        from wee_todd_nodes.lora import H3LoRASpec, H3LoRAStack
        from wee_todd_nodes.preflight import read_safetensors_header

        loras = H3LoRAStack()
        for value in recipe["loras"]["adapters"]:
            loras = loras.append(H3LoRASpec(**value))
        loras.validate_for_steps(config.steps)
        for adapter in loras.adapters:
            validate_h3_adapter_keys(read_safetensors_header(Path(adapter.path)).tensor_names)
    if recipe.get("vdn"):
        from wee_todd_nodes.vdn import H3VDNSpec

        vdn = H3VDNSpec(**recipe["vdn"])
        vdn.validate_sampling(config, loras)
    try:
        prepared = None
        images = None
        anchors = None
        continuation = None
        fun_control_spec = None
        fun_control_latent = None
        extension_source = None
        extension_report = None
        if contract["task"] == "extension":
            from minimax_h3_mlx.packing import align_num_frames
            from wee_todd_nodes.conditioning_inputs import h3_external_extension_references

            item = contract["inputs"][0]
            extension_report = media_report[0]
            source_video = read_media(recipe, item, extension_report)
            if len(source_video) != extension_report["num_frames"]:
                raise ValueError("H3 extension decoded frame count differs from ffprobe")
            source_audio = read_embedded_video_audio(recipe, item, extension_report)
            stack = h3_external_extension_references(
                source_video, source_audio, extension_report["fps"]
            )
            prepared = stack.prepare(
                target_width=config.width,
                target_height=config.height,
                target_num_frames=align_num_frames(round(config.duration_seconds * 24)),
            )
            extension_source = item["path"]
        if contract["task"] == "control":
            from minimax_h3_mlx.controlnet import H3FunControlSpec
            from minimax_h3_mlx.packing import align_num_frames
            from wee_todd_nodes.h3_controlnet import prepare_h3_control_frames

            item = next(value for value in contract["inputs"] if value["role"] == "control")
            report = media_report[contract["inputs"].index(item)]
            source = read_media(recipe, item, report)
            prepared_control = prepare_h3_control_frames(
                source,
                target_frames=align_num_frames(round(config.duration_seconds * 24)),
                height=config.height,
                width=config.width,
            )
            fun_control_spec = H3FunControlSpec(
                str(fun_controlnet_path), float(item["strength"])
            )
            fun_control_latent = VIDEO_VAE_RUNTIME.encode_continuation(
                H3VideoVAESpec.from_components(components),
                prepared_control,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
        if components.task == "ref2va":
            if prepared is None:
                from minimax_h3_mlx.packing import align_num_frames
                from wee_todd_nodes.conditioning_inputs import H3ReferenceInput, H3ReferenceStack

                stack = H3ReferenceStack()
                for item, report in zip(contract["inputs"], media_report, strict=True):
                    soundtrack = None
                    if item.get("soundtrack_path"):
                        soundtrack = read_media(
                            recipe,
                            {
                                "id": item["id"] + "-soundtrack",
                                "kind": "audio",
                                "path": item["soundtrack_path"],
                            },
                            report["soundtrack"],
                        )
                    stack = stack.append(
                        H3ReferenceInput(
                            kind=item["kind"],
                            media=read_media(recipe, item, report),
                            fps=report.get("fps"),
                            target_frame=(
                                0
                                if contract["task"] == "a2v"
                                and item["role"] == "audio_driver"
                                else item.get("frame_index")
                            ),
                            soundtrack=soundtrack,
                        )
                    )
                prepared = stack.prepare(
                    target_width=config.width,
                    target_height=config.height,
                    target_num_frames=align_num_frames(round(config.duration_seconds * 24)),
                )
        elif components.task == "fl2va":
            from PIL import Image

            from minimax_h3_mlx.packing import prepare_keyframe_image

            ordered = sorted(contract["inputs"], key=lambda item: item["frame_index"])
            anchors = tuple(item["frame_index"] for item in ordered)
            images = []
            for item in ordered:
                with Image.open(item["path"]) as source:
                    images.append(
                        prepare_keyframe_image(
                            source.convert("RGB"),
                            config.height,
                            config.width,
                            stretch=item["frame_index"] == 0,
                        )
                    )
        needs_vision = images is not None or (
            prepared is not None and any(reference.kind != "audio" for reference in prepared)
        )
        conditioning = TEXT_ENCODER_RUNTIME.encode(
            H3TextEncoderSpec.from_components(
                components, load_vision=needs_vision
            ),
            recipe["prompt"],
            images=images,
            references=prepared,
            task=components.task,
            unload_after=True,
            prepare_stage=lambda: prepare("text_encoder"),
            cache_directory=recipe.get("cache_directory")
            if prepared is None and images is None
            else None,
        )
        if prepared is not None:
            rows = VIDEO_VAE_RUNTIME.encode_references(
                H3VideoVAESpec.from_components(components),
                prepared,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
            audio_rows = None
            if any(reference.has_audio for reference in prepared):
                audio_rows = AUDIO_VAE_RUNTIME.encode_references(
                    H3AudioVAESpec.from_components(components),
                    prepared,
                    unload_after=True,
                    prepare_stage=lambda: prepare("audio_vae"),
                )
            conditioning = replace(
                conditioning,
                condition_video_rows=rows,
                condition_audio_rows=audio_rows,
                references=tuple(prepared),
            )
        elif images is not None:
            rows = VIDEO_VAE_RUNTIME.encode_keyframes(
                H3VideoVAESpec.from_components(components),
                images,
                height=config.height,
                width=config.width,
                unload_after=True,
                prepare_stage=lambda: prepare("video_vae"),
            )
            conditioning = replace(
                conditioning, condition_video_rows=rows, keyframe_anchors=anchors
            )
        with install_profiling(target.parent / "raw", "off"):
            latents = TRANSFORMER_RUNTIME.sample(
                H3TransformerSpec.from_components(components),
                conditioning,
                config,
                unload_after=True,
                sol_attention=attention,
                fastvideo=fastvideo,
                vdn=vdn,
                loras=loras,
                continuation=continuation,
                preview_config=components.preview_override,
                prepare_stage=lambda: prepare("transformer"),
                step_callback=lambda done, total: print(f"evaluation {done}/{total}", flush=True),
                fun_control_spec=fun_control_spec,
                fun_control_latent=fun_control_latent,
            )
        if latents.transformer_evaluations != config.steps - 1:
            raise RuntimeError("Unexpected transformer evaluation count")
        if attention and (
            latents.sol_attention_report.get("executed_calls")
            != (config.steps - 1)
            * (fastvideo.active_layers if fastvideo and fastvideo.active_layers else 50)
            or latents.sol_attention_report.get("fallback_calls") != 0
        ):
            raise RuntimeError("Native VSA execution proof failed")
        publish_target = (
            target if extension_source is None else target.with_name("generated-window.mp4")
        )
        result = publish_latents_direct(
            publish_target,
            components,
            latents,
            **recipe.get("publication", {}),
            ffmpeg_path=recipe["ffmpeg"],
            prepare_video_stage=lambda: prepare("video_vae"),
            prepare_audio_stage=lambda: prepare("audio_vae"),
        )
        extension_metadata = None
        video_path = result.video_path
        if extension_source is not None:
            assert extension_source is not None and extension_report is not None
            assemble_h3_extension(
                extension_source,
                result.video_path,
                target,
                source_frames=extension_report["num_frames"],
                context_frames=0,
                fps=24,
                ffmpeg=recipe["ffmpeg"],
            )
            video_path = target
            extension_metadata = {
                "direction": "after",
                "source_frames": extension_report["num_frames"],
                "context_frames": 0,
                "conditioning": "released_ref2va_source_plus_first_frame_anchor",
                "additional_frames": contract["extension"]["additional_frames"],
                "output_frames": extension_report["num_frames"]
                + contract["extension"]["additional_frames"],
                "audio_policy": contract["audio_policy"],
                "generated_window": str(result.video_path),
            }
        return {
            "video": str(video_path),
            "metadata": {
                **result.metadata,
                "conditioning_task": contract["task"],
                "extension": extension_metadata,
            },
            "evaluations": latents.transformer_evaluations,
            "attention": latents.sol_attention_report,
            "vdn": latents.vdn_report,
            "preview": latents.preview_report,
            "conditioning_cache": conditioning.cache_report,
            "conditioning_rows": {
                "video": 0
                if conditioning.condition_video_rows is None
                else int(conditioning.condition_video_rows.shape[0]),
                "audio": 0
                if conditioning.condition_audio_rows is None
                else int(conditioning.condition_audio_rows.shape[0]),
                "keyframe_anchors": list(anchors or ()),
            },
            "conditioning": task_report,
            "consumed_input_ids": task_report["input_ids"],
            "runtime_loaded": [runtime.loaded for runtime in runtimes],
        }
    finally:
        for runtime in runtimes:
            runtime.unload()


def render_ltx(recipe, target):
    if recipe.get("loras") and (
        recipe["engine"] == "ltx25" or not isinstance(recipe["loras"], dict)
    ):
        raise ValueError(
            "Invalid adapter declaration; refusing to render without requested adapters"
        )
    from wee_todd_mlx.conditioning_media import inspect_media, ltx_conditioning_kwargs
    from wee_todd_mlx.task_conditioning import (
        apply_ltx25_msr_prompt_guide,
        validate_conditioning,
        validate_ltx25_control_families,
    )

    task_report = validate_conditioning(recipe)
    contract = task_report["contract"]
    media_report = inspect_media(recipe, contract)
    if recipe["engine"] == "ltx23":
        from ltx23_mlx.ic_lora import recipe_ic_specs
        from ltx23_mlx.lora import recipe_specs
        from ltx23_mlx.runtime import RUNTIME, LTX23GenerationConfig, LTX23ModelSpec

        loras = recipe_specs(recipe)
        fields = dict(recipe["components"])
        fields.pop("loras", None)
        fields.pop("ic_loras", None)
        spec = LTX23ModelSpec(**fields, loras=loras, ic_loras=recipe_ic_specs(recipe))
        config = LTX23GenerationConfig(**recipe["config"])
    else:
        if recipe.get("loras"):
            raise ValueError(
                "LTX 2.5 adapters must be declared under components.loras; "
                "refusing to render without the requested adapters."
            )
        from ltx25_mlx.runtime import RUNTIME, LTX25ComponentSpec, LTX25GenerationConfig

        fields = dict(recipe["components"])
        for name in ("loras", "ic_loras"):
            fields[name] = tuple(tuple(item) for item in fields.get(name, ()))
        spec = LTX25ComponentSpec(**fields)
        if spec.loras:
            from ltx25_mlx.transformer import inspect_ltx25_lora

            for filename, _strength in spec.loras:
                inspect_ltx25_lora(filename)
        fields = dict(recipe["config"])
        fields["stg_blocks"] = tuple(fields.get("stg_blocks", ()))
        config = LTX25GenerationConfig(**fields)
        if any(item["role"] == "control" for item in contract["inputs"]):
            report = spec.validate(
                config.pipeline_mode, require_spatial_upscaler=not config.ic_lora_single_stage
            )
            validate_ltx25_control_families(contract, report)
    conditioning_kwargs = ltx_conditioning_kwargs(recipe, contract, media_report)
    effective_prompt = apply_ltx25_msr_prompt_guide(
        recipe["prompt"], task_report.get("prompt_guide", "")
    )
    ingredients_dir = None
    if recipe["engine"] == "ltx23" and "reference_sheet_input" in conditioning_kwargs:
        from wee_todd_mlx.conditioning_media import materialize_ingredients_video

        ingredients_dir = tempfile.TemporaryDirectory(prefix="weetodd-ltx23-ingredients-")
        item = conditioning_kwargs.pop("reference_sheet_input")
        conditioning_kwargs["control_inputs"] = [
            materialize_ingredients_video(
                item, config, ingredients_dir.name, ffmpeg=recipe.get("ffmpeg")
            )
        ]
    try:
        extension = recipe["engine"] in {"ltx23", "ltx25"} and contract[
            "task"
        ] == "extension"
        generation_target = target.with_name("generated-window.mp4") if extension else target
        result = RUNTIME.generate_to_file(
            spec,
            config,
            effective_prompt,
            generation_target,
            unload_after=True,
            step_callback=lambda done, total: print(f"evaluation {done}/{total}", flush=True),
            **conditioning_kwargs,
        )
        if extension:
            source_report = media_report[0]
            context_frames = (
                contract["extension"]["context_frames"]
                if recipe["engine"] == "ltx25"
                else source_report["num_frames"]
            )
            from wee_todd_mlx.conditioning_media import media_binary

            assemble_external_extension(
                contract["inputs"][0]["path"],
                generation_target,
                target,
                source_frames=source_report["num_frames"],
                context_frames=context_frames,
                fps=source_report["fps"],
                sample_rate=48000,
                ffmpeg=media_binary(recipe, "ffmpeg"),
            )
            result["video_path"] = str(target)
            result["extension"] = {
                "direction": "after",
                "source_frames": source_report["num_frames"],
                "context_frames": context_frames,
                "additional_frames": contract["extension"]["additional_frames"],
                "output_frames": source_report["num_frames"]
                + contract["extension"]["additional_frames"],
                "audio_policy": contract["audio_policy"],
                "generated_window": str(generation_target),
            }
            result["audio_policy"] = contract["audio_policy"]
        return {
            "video": result["video_path"],
            "metadata": result,
            "conditioning": task_report,
            "consumed_input_ids": task_report["input_ids"],
            "runtime_loaded": [RUNTIME.loaded],
        }
    finally:
        RUNTIME.unload()
        if ingredients_dir is not None:
            ingredients_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-library", type=Path, help="Registry for local asset references")
    parser.add_argument("--preflight-only", action="store_true", help="Validate without rendering")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get("format") != "weetodd-headless-v2" or recipe.get("engine") not in {
        "h3",
        "ltx23",
        "ltx25",
    }:
        parser.error("Unsupported recipe format/engine")
    sys.meta_path.insert(0, NoComfyImports())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    assert_isolated()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {
        "candidate": recipe["candidate"],
        "recipe": str(args.recipe.resolve()),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "status": "failed",
    }
    try:
        from wee_todd_mlx.asset_registry import AssetRegistry, resolve_recipe

        recipe, resolution = resolve_recipe(
            recipe, AssetRegistry(args.model_library) if args.model_library else None
        )
        record["asset_resolution"] = resolution
        (output / "resolved-recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        from wee_todd_mlx.headless_preflight import preflight_recipe

        record["preflight"] = preflight_recipe(recipe)
        (output / "effective-conditioning.json").write_text(
            json.dumps(record["preflight"]["conditioning"]["contract"], indent=2) + "\n"
        )
        if args.preflight_only:
            record.update(
                status="preflight_passed",
                seconds=time.perf_counter() - started,
                isolation=assert_isolated(),
            )
            print(json.dumps({"status": record["status"], "output": str(output)}), flush=True)
            return
        record.update(
            (render_h3 if recipe["engine"] == "h3" else render_ltx)(recipe, output / "render.mp4")
        )
        record["seconds"] = time.perf_counter() - started
        if any(record["runtime_loaded"]):
            raise RuntimeError("A weighted runtime was not released")
        record["isolation"] = assert_isolated()
        record["mp4_sha256"] = hashlib.sha256(Path(record["video"]).read_bytes()).hexdigest()
        record["status"] = "success"
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {k: record[k] for k in ("candidate", "status", "seconds", "video", "mp4_sha256")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
