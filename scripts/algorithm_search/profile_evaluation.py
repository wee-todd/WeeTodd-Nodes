#!/usr/bin/env python3
"""Profile one real H3 transformer evaluation with optional bounded activation capture."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.block_quantization import (
    parse_block_bit_overrides,
    quantize_selected_blocks,
)
from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession
from minimax_h3_mlx.algorithm_search.hybrid_swap import SelectiveHybridBlockController
from minimax_h3_mlx.algorithm_search.preflight import validate_profile_components
from minimax_h3_mlx.config import PipelineConfig
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder


def _save_final_latents(output_directory: Path, result) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory / "final_latents.safetensors"
    mx.save_safetensors(
        str(target),
        {
            "video_latents": result.video_latents,
            "audio_latents": result.audio_latents,
        },
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--text-config", type=Path)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-block", type=int, action="append", default=[])
    parser.add_argument("--profile-block", type=int, action="append", default=[])
    parser.add_argument("--capture-target", action="append", default=[])
    parser.add_argument("--capture-evaluation", type=int, action="append", default=[])
    parser.add_argument("--max-capture-mib", type=int, default=512)
    parser.add_argument("--profile-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-block", type=int)
    parser.add_argument("--hybrid-text-rows", type=int)
    parser.add_argument("--hybrid-audio-rows", type=int)
    parser.add_argument("--hybrid-apply-evaluation", type=int, default=2)
    parser.add_argument("--quantize-block", type=int, action="append", default=[])
    parser.add_argument("--quantize-all-blocks", action="store_true")
    parser.add_argument("--quantize-block-bit", action="append", default=[])
    parser.add_argument("--quantize-bits", type=int, default=5, choices=[4, 5, 6, 8])
    parser.add_argument("--quantize-group-size", type=int, default=64)
    parser.add_argument("--save-final-latents", action="store_true")
    args = parser.parse_args()
    validate_profile_components(
        model_index=args.model_index,
        transformer=args.transformer,
        text_encoder_directory=args.text_encoder,
        text_config=args.text_config,
        processor_directory=args.processor,
        tokenizer_directory=args.tokenizer,
        prompt_file=args.prompt_file,
    )
    prompt = args.prompt_file.read_text().strip()
    encoder = MiniMaxH3TextEncoder(
        args.text_encoder,
        load_vision=False,
        config_path=args.text_config,
        processor_dir=args.processor,
        tokenizer_dir=args.tokenizer,
    )
    embeddings, tags = encoder.encode(prompt)
    mx.eval(embeddings, tags)
    del encoder
    gc.collect()
    mx.clear_cache()
    dit = load_dit(args.transformer)
    quantization = None
    block_overrides = parse_block_bit_overrides(args.quantize_block_bit)
    selected_blocks = set(args.quantize_block)
    if args.quantize_all_blocks:
        selected_blocks.update(range(len(dit.blocks)))
    selected_blocks.update(block_overrides)
    if selected_blocks:
        invalid = sorted(index for index in selected_blocks if index >= len(dit.blocks))
        if invalid:
            raise ValueError(f"quantized block indices exceed model depth: {invalid}")
        groups: dict[int, list[int]] = {}
        for block in sorted(selected_blocks):
            groups.setdefault(block_overrides.get(block, args.quantize_bits), []).append(block)
        summaries = [
            quantize_selected_blocks(
                dit,
                blocks,
                bits=bits,
                group_size=args.quantize_group_size,
            )
            for bits, blocks in sorted(groups.items())
        ]
        quantization = (
            summaries[0]
            if len(summaries) == 1
            else {
                "groups": summaries,
                "parameter_bytes_before": summaries[0]["parameter_bytes_before"],
                "parameter_bytes_after": summaries[-1]["parameter_bytes_after"],
                "parameter_bytes_saved": sum(
                    summary["parameter_bytes_saved"] for summary in summaries
                ),
            }
        )
        gc.collect()
        mx.clear_cache()
    hybrid_values = (args.hybrid_block, args.hybrid_text_rows, args.hybrid_audio_rows)
    if any(value is not None for value in hybrid_values) and not all(
        value is not None for value in hybrid_values
    ):
        raise ValueError("hybrid block, text rows, and audio rows must be supplied together")
    hybrid_controller = (
        None
        if args.hybrid_block is None
        else SelectiveHybridBlockController(
            block_index=args.hybrid_block,
            text_rows=args.hybrid_text_rows,
            audio_rows=args.hybrid_audio_rows,
            apply_evaluation=args.hybrid_apply_evaluation,
        )
    )
    session = DiagnosticSession(
        CaptureConfig(
            enabled=bool(args.capture_target),
            output_directory=str(args.output),
            targets=tuple(args.capture_target),
            blocks=tuple(args.capture_block),
            profile_blocks=tuple(args.profile_block),
            evaluation_indices=tuple(args.capture_evaluation),
            max_total_bytes=args.max_capture_mib * 1024 * 1024,
            profile_regions=args.profile_regions,
        ),
        hybrid_controller=hybrid_controller,
    )
    pipeline = MiniMaxH3Pipeline(
        dit,
        None,
        None,
        None,
        PipelineConfig.from_model_index(args.model_index),
    )
    mx.reset_peak_memory()
    active_before_sample = int(mx.get_active_memory())
    result = pipeline.sample_latents(
        embeddings,
        np.asarray(tags),
        duration_seconds=args.duration,
        num_inference_steps=args.steps,
        seed=0,
        height=args.height,
        width=args.width,
        drop_adaln=True,
        diagnostics=session,
    )
    sample_peak_memory = int(mx.get_peak_memory())
    latent_path = _save_final_latents(args.output, result) if args.save_final_latents else None
    metadata = session.write_metadata()
    run_path = args.output / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "transformer_evaluations": result.transformer_evaluations,
                "seconds_per_evaluation": result.seconds_per_evaluation,
                "total_seconds": result.total_seconds,
                "active_before_sample_bytes": active_before_sample,
                "sample_peak_memory_bytes": sample_peak_memory,
                "quantization": quantization,
                "final_latents": latent_path.name if latent_path is not None else None,
            },
            indent=2,
        )
        + "\n"
    )
    if hybrid_controller is not None:
        hybrid_path = args.output / "hybrid.json"
        hybrid_path.write_text(json.dumps(hybrid_controller.history, indent=2) + "\n")
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
