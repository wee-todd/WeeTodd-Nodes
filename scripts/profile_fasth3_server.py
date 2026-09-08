#!/usr/bin/env python3
"""Launch a dedicated ComfyUI server with opt-in normal-path H3 profiling.

This never supplies the older dense-only diagnostics argument. Wrappers delegate
to the original functions and are restored on exit. Detailed timing synchronizes
real inputs/outputs and therefore perturbs scheduling; compare it with off/coarse
runs of the exact same saved API, not with synthetic operator estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

_ACTIVE = ContextVar("fasth3_native_profiler", default=None)


class NativeProfiler:
    def __init__(self, mode, mx):
        self.mode = mode
        self.mx = mx
        self.events = []
        self.stack = []
        self.evaluation = None
        self.block = None
        self.roles = {}
        self.blocks = []

    def arrays(self, value):
        if isinstance(value, self.mx.array):
            return [value]
        if isinstance(value, (list, tuple)):
            return [array for child in value for array in self.arrays(child)]
        if isinstance(value, dict):
            return [array for child in value.values() for array in self.arrays(child)]
        if hasattr(value, "video_latents") and hasattr(value, "audio_latents"):
            return self.arrays((value.video_latents, value.audio_latents))
        return []

    def measure(self, name, fn, *, inputs=(), sync=True, metadata=None):
        if self.mode == "off":
            return fn()
        event = {
            "name": name,
            "evaluation": self.evaluation,
            "block": self.block,
            "depth": len(self.stack),
            "child_seconds": 0.0,
            "input_ready_seconds": 0.0,
            "metadata": metadata or {},
            "status": "failed",
        }
        self.stack.append(event)
        started = time.perf_counter()
        try:
            if sync:
                arrays = self.arrays(inputs)
                if arrays:
                    self.mx.eval(*arrays)
                self.mx.synchronize()
                event["input_ready_seconds"] = time.perf_counter() - started
            result = fn()
            if sync:
                arrays = self.arrays(result)
                if arrays:
                    self.mx.eval(*arrays)
                self.mx.synchronize()
            event["status"] = "success"
            return result
        finally:
            event["wall_seconds"] = time.perf_counter() - started
            event["exclusive_seconds"] = max(
                0.0,
                event["wall_seconds"] - event["child_seconds"] - event["input_ready_seconds"],
            )
            self.stack.pop()
            if self.stack:
                self.stack[-1]["child_seconds"] += event["wall_seconds"]
            self.events.append(event)

    def summary(self):
        totals = defaultdict(
            lambda: {
                "calls": 0,
                "exclusive_seconds": 0.0,
                "input_ready_seconds": 0.0,
                "inclusive_seconds": 0.0,
            }
        )
        for event in self.events:
            row = totals[event["name"]]
            row["calls"] += 1
            for field in ("exclusive_seconds", "input_ready_seconds"):
                row[field] += event[field]
            row["inclusive_seconds"] += event["wall_seconds"]
        return dict(sorted(totals.items(), key=lambda pair: -pair[1]["exclusive_seconds"]))


@contextmanager
def install_profiling(output_directory, mode):
    """Patch only this dedicated process; inactive calls and engine defaults are unchanged."""
    import gc

    import mlx.core as mx
    import mlx.nn as nn

    import minimax_h3_mlx.dit as dit_module
    import minimax_h3_mlx.paged_checkpoint as paged_module
    import minimax_h3_mlx.pipeline as pipeline_module
    import minimax_h3_mlx.vsa_h3 as vsa_module
    import minimax_h3_mlx.vsa_h3_metal as metal_module

    if mode not in {"off", "coarse", "detailed"}:
        raise ValueError("Unknown profiling mode")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    patches = []

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, replacement)

    def timed(owner, attribute, name, *, detailed=False, sync=True):
        original = getattr(owner, attribute)

        @wraps(original)
        def wrapper(*args, **kwargs):
            profiler = _ACTIVE.get()
            if profiler is None or (detailed and profiler.mode != "detailed"):
                return original(*args, **kwargs)
            return profiler.measure(
                name,
                lambda: original(*args, **kwargs),
                inputs=(args, kwargs),
                sync=sync,
            )

        patch(owner, attribute, wrapper)

    original_sample = pipeline_module.MiniMaxH3Pipeline.sample_latents

    @wraps(original_sample)
    def sample(pipeline, *args, **kwargs):
        if _ACTIVE.get() is not None:
            raise RuntimeError("Nested H3 sampling is not supported by this profiler")
        if kwargs.get("diagnostics") is not None:
            raise ValueError("Native-path profiling cannot use dense diagnostic substitution")
        profiler = NativeProfiler(mode, mx)
        token = _ACTIVE.set(profiler)
        result = None
        identifier = uuid.uuid4().hex
        try:
            result = profiler.measure(
                "sampler",
                lambda: original_sample(pipeline, *args, **kwargs),
                inputs=args,
            )
            return result
        finally:
            _ACTIVE.reset(token)
            record = {
                "format": "weetodd-h3-native-profile-v1",
                "mode": mode,
                "scope": "sample_latents; excludes loader, final decode and publication",
                "timing": "synchronized wall time, not GPU-only; detailed mode perturbs scheduling",
                "peak_policy": "profiler never resets MLX peak memory",
                "implementation": str(Path(dit_module.__file__).resolve()),
                "configuration": {
                    key: value
                    for key, value in kwargs.items()
                    if value is None or isinstance(value, (str, int, float, bool))
                },
                "blocks": profiler.blocks,
                "summary": profiler.summary(),
                "events": profiler.events,
                "status": "success" if result is not None else "failed",
            }
            try:
                from comfy_execution.utils import get_executing_context

                record["prompt_id"] = get_executing_context().prompt_id
            except (ImportError, AttributeError):
                record["prompt_id"] = None
            pager = getattr(pipeline.dit, "paged_blocks", None)
            record["paging"] = pager.report() if pager is not None else None
            if result is not None:
                latent_path = output_directory / f"{identifier}.safetensors"
                mx.save_safetensors(
                    str(latent_path),
                    {
                        "video_latents": result.video_latents,
                        "audio_latents": result.audio_latents,
                    },
                )
                record["latents"] = str(latent_path)
                record["latent_file_sha256"] = hashlib.sha256(latent_path.read_bytes()).hexdigest()
                record["pipeline_seconds"] = result.total_seconds
            target = output_directory / f"{identifier}.json"
            target.write_text(json.dumps(record, indent=2) + "\n")
            print(f"H3 native {mode} profile: {target}", flush=True)

    patch(pipeline_module.MiniMaxH3Pipeline, "sample_latents", sample)
    timed(pipeline_module.MiniMaxH3Pipeline, "_ensure_cache", "adaln.schedule_prepare", sync=False)

    original_evaluation = dit_module.MiniMaxH3DiT.__call__

    @wraps(original_evaluation)
    def evaluation(model, *args, **kwargs):
        profiler = _ACTIVE.get()
        if profiler is None:
            return original_evaluation(model, *args, **kwargs)
        previous = profiler.evaluation
        profiler.evaluation = int(kwargs.get("step_index", 0))
        try:
            return profiler.measure(
                "evaluation",
                lambda: original_evaluation(model, *args, **kwargs),
                inputs=(args, kwargs),
            )
        finally:
            profiler.evaluation = previous

    patch(dit_module.MiniMaxH3DiT, "__call__", evaluation)
    timed(dit_module.MiniMaxH3DiT, "pack_inputs", "evaluation.pack_inputs")
    timed(dit_module.MiniMaxH3DiT, "_project_packed_features", "evaluation.output_heads")
    timed(dit_module.MiniMaxH3DiT, "_run_paged_blocks", "block_stack")

    original_block = dit_module.TransformerBlock.__call__

    @wraps(original_block)
    def block(module, *args, **kwargs):
        profiler = _ACTIVE.get()
        if profiler is None:
            return original_block(module, *args, **kwargs)
        old_block, old_roles = profiler.block, profiler.roles
        profiler.block = kwargs.get("block_index")
        roles = {
            "attention.qkv_projection": module.attn.qkv_proj,
            "attention.gate_projection": module.attn.gate_compress,
            "attention.output_projection": module.attn.out_proj,
            "attention.q_norm": module.attn.q_norm,
            "attention.k_norm": module.attn.k_norm,
            "block.norm1": module.norm1,
            "block.norm2": module.norm2,
            "ffn.fc1": module.mlp.fc1,
            "ffn.fc2": module.mlp.fc2,
        }
        profiler.roles = {id(value): name for name, value in roles.items() if value is not None}
        profiler.blocks.append(
            {
                "evaluation": profiler.evaluation,
                "block": profiler.block,
                "input_shape": list(args[0].shape),
                "ffn_row_chunk_size": module.mlp.row_chunk_size,
                "attention_head_chunk_size": module.attn.head_chunk_size,
                "attention_backend": getattr(module.attn.vsa_h3_config, "consumer_backend", None),
            }
        )
        try:
            if mode != "detailed":
                return original_block(module, *args, **kwargs)
            return profiler.measure(
                "block",
                lambda: original_block(module, *args, **kwargs),
                inputs=(args, kwargs),
            )
        finally:
            profiler.block, profiler.roles = old_block, old_roles

    patch(dit_module.TransformerBlock, "__call__", block)

    def time_module(kind):
        original = kind.__call__

        @wraps(original)
        def wrapper(module, *args, **kwargs):
            profiler = _ACTIVE.get()
            name = profiler.roles.get(id(module)) if profiler is not None else None
            if name is None or mode != "detailed":
                return original(module, *args, **kwargs)
            return profiler.measure(
                name,
                lambda: original(module, *args, **kwargs),
                inputs=(args, kwargs),
                metadata={"input_shape": list(args[0].shape)},
            )

        patch(kind, "__call__", wrapper)

    for kind in (nn.QuantizedLinear, nn.Linear, nn.RMSNorm):
        time_module(kind)
    timed(dit_module.Attention, "_normal", "attention", detailed=True)
    timed(dit_module.FeedForward, "__call__", "ffn", detailed=True)
    timed(dit_module, "apply_rotary", "attention.rotary", detailed=True)
    timed(vsa_module, "vsa_h3_attention", "vsa", detailed=True)
    timed(vsa_module, "_route_indices", "vsa.route_selection", detailed=True)
    timed(metal_module, "vsa_h3_compact_summaries", "vsa.tile_summaries", detailed=True)
    timed(metal_module, "vsa_h3_indexed_attention", "vsa.indexed_video_attention", detailed=True)
    timed(mx.fast, "scaled_dot_product_attention", "attention.sdpa", detailed=True)
    timed(paged_module.PagedTensorStore, "_load_many", "page.lazy_file_load", sync=False)
    timed(gc, "collect", "cleanup.python_gc", sync=False)
    timed(mx, "clear_cache", "cleanup.mlx_cache", sync=False)

    original_window = paged_module.PagedBlockExecutor._selected_window

    @contextmanager
    def window(executor, indices, prefetch_after):
        profiler = _ACTIVE.get()
        if profiler is None:
            with original_window(executor, indices, prefetch_after) as blocks:
                yield blocks
            return
        original = original_window(executor, indices, prefetch_after)
        blocks = profiler.measure(
            "page.setup",
            original.__enter__,
            sync=False,
            metadata={"indices": list(indices)},
        )
        try:
            yield blocks
        except BaseException:
            error = sys.exc_info()
            suppressed = profiler.measure(
                "page.release",
                lambda: original.__exit__(*error),
                sync=False,
            )
            if not suppressed:
                raise
        else:
            profiler.measure(
                "page.release", lambda: original.__exit__(None, None, None), sync=False
            )

    patch(paged_module.PagedBlockExecutor, "_selected_window", window)
    try:
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "coarse", "detailed"), required=True)
    parser.add_argument("comfy_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.comfy_root.resolve()
    if not (root / "main.py").is_file():
        parser.error("ComfyUI main.py is missing")
    remaining = args.comfy_args
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if "--cache-none" not in remaining:
        parser.error("Profiling requires a dedicated --cache-none server")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.argv = [str(root / "main.py"), *remaining]
    os.chdir(root)
    with install_profiling(args.profile_output.resolve(), args.mode):
        runpy.run_path(str(root / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
