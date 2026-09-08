"""Build matching node graphs and host-independent recipes from existing local assets.

Machine paths are CLI inputs. Bundles are symlink views, never copies of model weights.
This validates ten checkpoint candidates, not every model/task/adapter combination.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


def defaults(cls):
    result = {}
    for name, field in cls.INPUT_TYPES()["required"].items():
        if len(field) > 1 and "default" in field[1]:
            result[name] = field[1]["default"]
        elif isinstance(field[0], list):
            result[name] = field[0][0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--ltx23-dev", type=Path, required=True)
    parser.add_argument("--ltx23-distilled", type=Path, required=True)
    parser.add_argument("--gemma3", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument(
        "--h3-ref2va-transformer",
        default="MiniMax-H3/Ref2VA/transformer-native",
        help="Genuine Ref2VA transformer path (absolute or relative to Comfy models)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.comfy_root.resolve()
    sys.path[:0] = [str(root), str(Path(__file__).resolve().parents[1] / "src")]
    os.chdir(root)
    from wee_todd_nodes import ltx25_nodes as l25
    from wee_todd_nodes import ltx_nodes as l23
    from wee_todd_nodes import nodes as h3
    from wee_todd_nodes.conditioning_cache import default_cache_directory
    from wee_todd_nodes.preflight import H3PreflightRequest, preflight_components

    manifest = []
    prompt = (
        "A small red wind-up robot on a wooden workbench raises one metal hand and waves. "
        "A fixed camera holds a medium shot. Warm workshop light. Soft rain and synchronized "
        "metallic clicks and servo whirs. No speech, no music."
    )
    h3_prompt = (
        "integrated_multimodal_description: [Shot 1] "
        + prompt
        + "\n\noverall_soundscape: Soft rain and metallic clicks.\n\nnon_diegetic_music: N/A"
    )

    def node(kind, inputs):
        return {"class_type": kind, "inputs": inputs}

    def save(candidate, graph, recipe, check):
        recipe.update(
            format="weetodd-headless-v2", candidate=candidate, ffmpeg="/opt/homebrew/bin/ffmpeg"
        )
        directory = output / candidate
        directory.mkdir()
        (directory / "api.json").write_text(json.dumps(graph, indent=2) + "\n")
        (directory / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        try:
            check()
            status, error = "ready", None
        except Exception as exc:
            status, error = "blocked_preflight", f"{type(exc).__name__}: {exc}"
        manifest.append(
            {
                "candidate": candidate,
                "engine": recipe["engine"],
                "status": status,
                "error": error,
                "api": str(directory / "api.json"),
                "recipe": str(directory / "recipe.json"),
            }
        )

    for candidate in ("h3-fl2va", "h3-ref2va", "fasth3-dense", "fasth3-vsa", "vdn-8", "vdn-50"):
        ref = candidate == "h3-ref2va"
        loader = dict(
            checkpoint="MiniMax-H3/Ref2VA" if ref else "MiniMax-H3/FL2VA",
            task="ref2va" if ref else "t2va",
            allow_fl2va_weights_for_ref2va=False,
            transformer="MiniMax-H3/transformers/q8_extended_paged",
            text_encoder="MiniMax-H3/text_encoders/q8-paged",
            processor="MiniMax-H3/FL2VA/processor",
            tokenizer="MiniMax-H3/FL2VA/tokenizer",
            video_vae="MiniMax-H3/vae/q8/video_vae_affine_q8.safetensors",
            audio_vae="MiniMax-H3/FL2VA/audio_vae/audio_vae.safetensors",
        )
        if ref:
            loader.update(
                transformer=args.h3_ref2va_transformer,
                text_encoder="MiniMax-H3/Ref2VA/text_encoder",
                processor="MiniMax-H3/Ref2VA/processor",
                tokenizer="MiniMax-H3/Ref2VA/tokenizer",
            )
        if candidate.startswith("fasth3"):
            folder = (
                "weetodd-fasth3-dense-q8-paged"
                if candidate.endswith("dense")
                else "weetodd-fasth3-vsa-datafree-q8-paged"
            )
            loader["transformer"] = "MiniMax-H3/transformers/" + folder
        cfg = defaults(h3.WeeToddH3GenerationConfig)
        cfg.update(
            duration_seconds=2.5,
            steps=20,
            seed=20260903,
            resolution_mode="exact dimensions",
            custom_width=384,
            custom_height=256,
            memory_mode="low_memory_bf16",
            projection_backend="mlx",
        )
        if candidate.startswith("fasth3"):
            cfg.update(steps=5, duration_seconds=4, custom_width=768, custom_height=448)
        components = h3.WeeToddH3ComponentLoader().specify(**loader)[0]
        config = h3.WeeToddH3GenerationConfig().configure(**cfg)[0]
        graph = {
            "1": node("WeeToddH3ComponentLoader", loader),
            "2": node("WeeToddH3GenerationConfig", cfg),
        }
        comp_link, config_link = ["1", 0], ["2", 0]
        options, extra = {}, {}
        if candidate.startswith("vdn"):
            args_vdn = dict(
                repository="OpenVDN/vdn-minimax-h3",
                stage="VDN-H3 8-step (recommended)" if candidate == "vdn-8" else "VDN-H3 50-step",
                adaln_input_grid="h3_silu_temb_grid.safetensors",
                inference_backend="verified",
            )
            components, config, vdn, loras, _ = h3.WeeToddH3VDNCheckpoint().select(
                components, config, **args_vdn
            )
            graph["3"] = node(
                "WeeToddH3VDNCheckpoint", dict(components=comp_link, config=config_link, **args_vdn)
            )
            comp_link, config_link = ["3", 0], ["3", 1]
            options.update(vdn=["3", 2], loras=["3", 3])
            extra.update(vdn=asdict(vdn), loras=asdict(loras))
        if candidate == "fasth3-vsa":
            settings = defaults(h3.WeeToddH3SolAttention)
            settings["profile"] = "fasth3_vsa_90_metal"
            attention, _ = h3.WeeToddH3SolAttention().configure(**settings)
            graph["3"] = node("WeeToddH3SolAttention", settings)
            options["sol_attention"] = ["3", 0]
            extra["attention"] = asdict(attention)
        graph["4"] = node(
            "WeeToddH3Preflight",
            dict(
                components=comp_link, config=config_link, prompt_tokens=512, available_memory_gb=0
            ),
        )
        comp_link = ["4", 0]
        actual_prompt = h3_prompt
        if ref:
            image = args.reference_image.resolve()
            local = root / "input" / image.name
            if not local.exists():
                local.symlink_to(image)
            graph["10"] = node("LoadImage", {"image": local.name})
            graph["11"] = node(
                "WeeToddH3ReferenceImage", {"image": ["10", 0], "pixel_budget_percent": 100}
            )
            actual_prompt = h3_prompt.replace(
                "A small red", "The robot in <Picture 1>, a small red"
            )
            graph["5"] = node(
                "WeeToddH3ReferenceEncode",
                dict(
                    components=comp_link,
                    config=config_link,
                    references=["11", 0],
                    prompt=actual_prompt,
                ),
            )
            extra["reference_images"] = [str(image)]
        else:
            graph["5"] = node(
                "WeeToddH3TextEncode",
                dict(
                    components=comp_link,
                    config=config_link,
                    prompt=actual_prompt,
                    unload_after_encode=True,
                    persistent_cache=True,
                ),
            )
        graph["6"] = node(
            "WeeToddH3Sample",
            dict(
                components=comp_link,
                config=config_link,
                conditioning=["5", 0],
                unload_after_sample=True,
                **options,
            ),
        )
        publication = dict(crf=18, max_av_drift_seconds=0.025, generation_metadata="{}")
        graph["7"] = node(
            "WeeToddH3DirectPublishLatents",
            dict(
                components=comp_link,
                latents=["6", 0],
                sampling_info=["6", 1],
                filename_prefix="headless-matrix/" + candidate,
                **publication,
            ),
        )
        save(
            candidate,
            graph,
            dict(
                engine="h3",
                components=asdict(components),
                config=asdict(config),
                prompt=actual_prompt,
                cache_directory=str(default_cache_directory()),
                publication=publication,
                **extra,
            ),
            lambda components=components, config=config: preflight_components(
                components,
                H3PreflightRequest(
                    duration_seconds=config.duration_seconds,
                    steps=config.steps,
                    width=config.width,
                    height=config.height,
                ),
            ),
        )

    for candidate in ("ltx23-dev", "ltx23-distilled"):
        original = args.ltx23_dev if candidate.endswith("dev") else args.ltx23_distilled
        # A view shares the available upscaler and existing tensors; no weight duplication.
        bundle = output / (candidate + "-bundle")
        bundle.mkdir()
        for source in original.iterdir():
            if source.is_file():
                (bundle / source.name).symlink_to(source.resolve())
        upscaler = bundle / "spatial_upscaler_x2_v1_1.safetensors"
        if not upscaler.exists():
            upscaler.symlink_to((args.ltx23_dev / upscaler.name).resolve())
        loader = dict(model_directory=str(bundle), gemma_model=str(args.gemma3.resolve()))
        cfg = defaults(l23.WeeToddLTX23GenerationConfig)
        cfg.update(
            pipeline_mode="one_stage" if candidate.endswith("dev") else "distilled",
            width=384,
            height=256,
            duration_seconds=1,
            seed=20260903,
        )
        model = l23.WeeToddLTX23ModelLoader().specify(**loader)[0]
        config = l23.WeeToddLTX23GenerationConfig().configure(**cfg)[0]
        graph = {
            "1": node("WeeToddLTX23ModelLoader", loader),
            "2": node("WeeToddLTX23GenerationConfig", cfg),
            "3": node("WeeToddLTX23Preflight", dict(model=["1", 0], config=["2", 0])),
            "4": node(
                "WeeToddLTX23Generate",
                dict(
                    model=["3", 0],
                    config=["2", 0],
                    prompt=prompt,
                    filename_prefix="headless-matrix/" + candidate,
                    unload_after_generate=True,
                ),
            ),
        }
        save(
            candidate,
            graph,
            dict(engine="ltx23", components=asdict(model), config=asdict(config), prompt=prompt),
            lambda model=model, config=config: model.validate(config.pipeline_mode),
        )

    for candidate in ("ltx25-dev", "ltx25-distilled"):
        loader = defaults(l25.WeeToddLTX25ComponentLoader)
        loader.update(
            transformer="ltx-2.5-22b-distilled-transformer-q8-paged",
            text_encoder="gemma4-12b-with-proj-ltx-2.5-q8-paged",
        )
        cfg = defaults(l25.WeeToddLTX25GenerationConfig)
        cfg.update(preset="Custom", width=384, height=256, duration_seconds=1, seed=20260903)
        model = l25.WeeToddLTX25ComponentLoader().specify(**loader)[0]
        config = l25.WeeToddLTX25GenerationConfig().configure(**cfg)[0]
        graph = {
            "1": node("WeeToddLTX25ComponentLoader", loader),
            "2": node("WeeToddLTX25GenerationConfig", cfg),
        }
        model_link, cfg_link = ["1", 0], ["2", 0]
        if candidate.endswith("dev"):
            guided = defaults(l25.WeeToddLTX25GuidedModelLoader)
            guided["development_transformer"] = "ltx-2.5-22b-dev-transformer-q8-paged"
            model = l25.WeeToddLTX25GuidedModelLoader().attach(model, **guided)[0]
            quality = defaults(l25.WeeToddLTX25QualityMode)
            quality["mode"] = l25.LTX25_QUALITY_MODES[1]
            config = l25.WeeToddLTX25QualityMode().apply(config, **quality)[0]
            graph["5"] = node("WeeToddLTX25GuidedModelLoader", dict(model=model_link, **guided))
            graph["6"] = node("WeeToddLTX25QualityMode", dict(config=cfg_link, **quality))
            model_link, cfg_link = ["5", 0], ["6", 0]
        graph["3"] = node("WeeToddLTX25Preflight", dict(model=model_link, config=cfg_link))
        graph["4"] = node(
            "WeeToddLTX25Generate",
            dict(
                model=["3", 0],
                config=cfg_link,
                prompt=prompt,
                filename_prefix="headless-matrix/" + candidate,
                unload_after_generate=True,
            ),
        )
        save(
            candidate,
            graph,
            dict(engine="ltx25", components=asdict(model), config=asdict(config), prompt=prompt),
            lambda model=model, config=config: model.validate(config.pipeline_mode),
        )
    (output / "matrix.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
