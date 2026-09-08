#!/usr/bin/env python3
"""Render an exported H3 milestone recipe without importing or running ComfyUI.

Uses the same MLX staging, preview guards, sampler and streaming publisher. This
is a benchmark/standalone-engine milestone, not a general Comfy graph executor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import resource
import sys
import time
from pathlib import Path

STARTED = time.perf_counter()
FORBIDDEN = {"comfy", "comfy_execution", "folder_paths", "nodes", "server"}


class NoComfyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ModuleNotFoundError(f"ComfyUI is intentionally unavailable: {fullname}")
        return None


def assert_isolated():
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN)
    if loaded or "wee_todd_nodes.nodes" in sys.modules:
        raise RuntimeError(f"Headless import isolation failed: {loaded}")
    return {
        "comfy_modules_loaded": loaded,
        "node_catalog_loaded": False,
        "sys_path": list(sys.path),
        "executable": sys.executable,
    }


def validate_execution(latents):
    layers = latents.fast_h3_approximation_report or {}
    attention = latents.sol_attention_report or {}
    kept, skipped = layers.get("active_layer_indices", []), layers.get("skipped_layer_indices", [])
    if not (
        latents.transformer_evaluations == 4
        and layers.get("executed_layers") == len(kept) == 40
        and layers.get("skipped_layers") == len(skipped) == 10
        and sorted(kept + skipped) == list(range(50))
        and {0, 1, 48, 49}.issubset(kept)
        and attention.get("executed_calls") == 160
        and attention.get("fallback_calls") == 0
        and attention.get("storage_layout") == "compact_preordered"
    ):
        raise RuntimeError("40-layer execution proof failed; refusing publication")


def load_recipe(file):
    recipe = json.loads(file.read_text())
    if recipe.get("format") != "weetodd-h3-headless-recipe-v1":
        raise ValueError("Unsupported headless recipe")
    for filename, expected in recipe["model_metadata_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model metadata changed: {filename}")
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.meta_path.insert(0, NoComfyImports())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    assert_isolated()
    recipe = load_recipe(args.recipe.resolve())
    # Same off-mode wrappers/latent capture as the Comfy controls, no diagnostic substitution.
    from profile_fasth3_server import install_profiling

    from minimax_h3_mlx.fasth3_approx import FastH3ApproximationConfig
    from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig
    from wee_todd_nodes.conditioning import TEXT_ENCODER_RUNTIME, H3TextEncoderSpec
    from wee_todd_nodes.decoding import AUDIO_VAE_RUNTIME, VIDEO_VAE_RUNTIME
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
    fields["preview_override"] = H3PreviewConfig(**fields["preview_override"])
    components = H3ComponentSetSpec(**fields)
    config = H3GenerationConfig(**recipe["config"])
    attention_fields = dict(recipe["attention"])
    for name in ("prefix_segments", "video_grid"):
        attention_fields[name] = tuple(attention_fields[name])
    attention = FastH3VSAConfig(**attention_fields)
    fastvideo = FastH3ApproximationConfig(**recipe["fastvideo"])
    config.validate()
    attention.validate()
    components.preview_override.validate()
    fastvideo.validate(50)
    if not (
        components.task == "t2va"
        and config.steps == 5
        and config.sampling_method == "euler"
        and fastvideo.active_layers == 40
        and config.inference_optimization == "off"
        and config.memory_mode == "low_memory_bf16"
    ):
        raise ValueError("This milestone requires the staged 40-layer Euler control")
    startup_seconds = time.perf_counter() - STARTED
    records = []
    runtimes = (
        TEXT_ENCODER_RUNTIME,
        TRANSFORMER_RUNTIME,
        VIDEO_VAE_RUNTIME,
        AUDIO_VAE_RUNTIME,
        RUNTIME,
    )
    try:
        with install_profiling(output / "raw", "off"):
            for index in range(args.runs):
                started = time.perf_counter()
                preflight_components(
                    components,
                    H3PreflightRequest(
                        duration_seconds=config.duration_seconds,
                        steps=config.steps,
                        width=config.width,
                        height=config.height,
                        prompt_tokens=512,
                    ),
                )
                conditioning = TEXT_ENCODER_RUNTIME.encode(
                    H3TextEncoderSpec.from_components(components, load_vision=False),
                    recipe["prompt"],
                    task=components.task,
                    unload_after=True,
                    cache_directory=recipe["cache_directory"],
                    prepare_stage=lambda: prepare_low_memory_stage(
                        "text_encoder", config.memory_mode
                    ),
                )
                if conditioning.cache_report.get("status") != "hit":
                    raise RuntimeError("Matched benchmark requires a warm conditioning cache")
                latents = TRANSFORMER_RUNTIME.sample(
                    H3TransformerSpec.from_components(components),
                    conditioning,
                    config,
                    unload_after=True,
                    sol_attention=attention,
                    fastvideo=fastvideo,
                    preview_config=components.preview_override,
                    step_callback=lambda done, total: print(
                        f"evaluation {done}/{total}", flush=True
                    ),
                    prepare_stage=lambda: prepare_low_memory_stage(
                        "transformer", config.memory_mode
                    ),
                )
                validate_execution(latents)
                result = publish_latents_direct(
                    output / f"headless-{index + 1}.mp4",
                    components,
                    latents,
                    **recipe["publication"],
                    ffmpeg_path=args.ffmpeg.resolve(),
                    prepare_video_stage=lambda: prepare_low_memory_stage(
                        "video_vae", config.memory_mode
                    ),
                    prepare_audio_stage=lambda: prepare_low_memory_stage(
                        "audio_vae", config.memory_mode
                    ),
                )
                elapsed = time.perf_counter() - started
                record = {
                    "run": index + 1,
                    "render_seconds": elapsed,
                    "sampler_seconds": latents.total_seconds,
                    "phase_memory": latents.phase_memory,
                    "attention": latents.sol_attention_report,
                    "layers": latents.fast_h3_approximation_report,
                    "preview": latents.preview_report,
                    "conditioning_cache": conditioning.cache_report,
                    "runtime_loaded_after_render": [runtime.loaded for runtime in runtimes],
                    "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "output": str(result.video_path),
                    "output_sha256": hashlib.sha256(result.video_path.read_bytes()).hexdigest(),
                    "isolation": assert_isolated(),
                }
                records.append(record)
                (output / "results.json").write_text(
                    json.dumps(
                        {
                            "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
                            "startup_seconds": startup_seconds,
                            "runs": records,
                            "timing_scope": (
                                "preflight through mux; excludes imports and post-render hashing"
                            ),
                            "preview_ui": (
                                "decoder and safety guard preserved; "
                                "no contact sheet/browser delivery"
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"Run {index + 1}: {elapsed:.3f}s; {record['output_sha256']}", flush=True)
                del latents, conditioning, result
    finally:
        for runtime in runtimes:
            runtime.unload()
        assert_isolated()


if __name__ == "__main__":
    main()
