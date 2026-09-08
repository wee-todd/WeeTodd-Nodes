import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import pytest


def load_profiler():
    filename = Path(__file__).parents[1] / "scripts/profile_fasth3_server.py"
    spec = importlib.util.spec_from_file_location("fasth3_native_profiler", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_times_are_additive_and_errors_release_stack():
    module = load_profiler()
    profiler = module.NativeProfiler("detailed", mx)
    result = profiler.measure("outer", lambda: profiler.measure("inner", lambda: mx.ones((2,))))
    assert result.tolist() == [1, 1]
    root = profiler.events[-1]
    accounted = sum(e["exclusive_seconds"] + e["input_ready_seconds"] for e in profiler.events)
    assert accounted == pytest.approx(root["wall_seconds"])
    with pytest.raises(ValueError):
        profiler.measure("failed", lambda: (_ for _ in ()).throw(ValueError("test")))
    assert profiler.stack == []
    assert profiler.events[-1]["status"] == "failed"


def test_installed_wrappers_preserve_normal_vsa_and_restore(tmp_path):
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.dit import TransformerBlock
    from minimax_h3_mlx.vsa_h3 import FastH3VSAConfig

    module = load_profiler()
    original = TransformerBlock.__call__
    config = DiTConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=64,
        time_embed_dim=16,
        time_embed_hidden_size=16,
        timestep_input_dim=16,
        rope_inv_freq_len=16,
        vsa_gate=True,
    )
    block = TransformerBlock(config)
    block.attn.vsa_h3_config = FastH3VSAConfig(
        prefix_segments=(8,),
        video_grid=(1, 8, 8),
        min_tokens=64,
        consumer_backend="metal_indexed",
    )
    block.set_dtype(mx.bfloat16)
    x = mx.ones((1, 72, 128), dtype=mx.bfloat16)
    modulation = tuple(mx.full((3, 128), 0.01, dtype=mx.bfloat16) for _ in range(6))
    indices = mx.zeros((72,), dtype=mx.int32)
    rotary = (mx.ones((72, 96)), mx.zeros((72, 96)))
    with pytest.raises(ValueError, match="normal-path FastH3 profiler"):
        block.attn(x, rotary, diagnostics=object())
    expected = block(x, modulation, indices, rotary, block_index=0)
    mx.eval(expected)
    with module.install_profiling(tmp_path, "detailed"):
        profiler = module.NativeProfiler("detailed", mx)
        token = module._ACTIVE.set(profiler)
        try:
            actual = block(x, modulation, indices, rotary, block_index=0)
            assert mx.array_equal(expected, actual).item()
            names = {e["name"] for e in profiler.events}
            assert {"vsa.indexed_video_attention", "vsa.route_selection", "ffn.fc1"} <= names
            assert profiler.blocks[0]["ffn_row_chunk_size"] is None
        finally:
            module._ACTIVE.reset(token)
    assert TransformerBlock.__call__ is original


def test_off_does_not_evaluate_and_failed_context_restores(tmp_path):
    from minimax_h3_mlx.dit import TransformerBlock

    module = load_profiler()
    profiler = module.NativeProfiler("off", object())
    assert profiler.measure("unused", lambda: 123) == 123
    assert not profiler.events
    original = TransformerBlock.__call__
    with pytest.raises(RuntimeError):
        with module.install_profiling(tmp_path, "coarse"):
            raise RuntimeError("cancel")
    assert TransformerBlock.__call__ is original


def test_sample_failure_records_trace_and_restores_active_session(tmp_path, monkeypatch):
    from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

    module = load_profiler()

    def fail(*args, **kwargs):
        raise ValueError("cancelled")

    monkeypatch.setattr(MiniMaxH3Pipeline, "sample_latents", fail)
    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.dit = object()
    with module.install_profiling(tmp_path, "coarse"):
        with pytest.raises(ValueError, match="cancelled"):
            pipeline.sample_latents()
    assert module._ACTIVE.get() is None
    record = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert record["status"] == "failed"


def test_page_wrapper_preserves_exception_cleanup(tmp_path, monkeypatch):
    from minimax_h3_mlx.paged_checkpoint import PagedBlockExecutor

    module = load_profiler()
    actions = []

    @contextmanager
    def original_window(self, indices, prefetch_after):
        actions.append(("enter", indices))
        try:
            yield ["block"]
        finally:
            actions.append("release")

    monkeypatch.setattr(PagedBlockExecutor, "_selected_window", original_window)
    executor = object.__new__(PagedBlockExecutor)
    with module.install_profiling(tmp_path, "coarse"):
        profiler = module.NativeProfiler("coarse", mx)
        token = module._ACTIVE.set(profiler)
        try:
            with pytest.raises(KeyboardInterrupt):
                with executor._selected_window((0,), None) as blocks:
                    assert blocks == ["block"]
                    raise KeyboardInterrupt()
            assert actions == [("enter", (0,)), "release"]
            assert {e["name"] for e in profiler.events} == {"page.setup", "page.release"}
        finally:
            module._ACTIVE.reset(token)


def test_summary_rejects_wrong_path_and_double_counting():
    filename = Path(__file__).parents[1] / "scripts/summarize_fasth3_profile.py"
    spec = importlib.util.spec_from_file_location("fasth3_profile_summary", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = {
        "mode": "coarse",
        "blocks": [
            {"evaluation": evaluation, "block": block, "attention_backend": "metal_indexed"}
            for evaluation in range(4)
            for block in [0, 1, 2, 3, 5, 6, 7, 9, *range(18, 50)]
        ],
        "events": [
            {"name": "sampler", "wall_seconds": 1, "exclusive_seconds": 1, "input_ready_seconds": 0}
        ],
    }
    module.validate_trace(trace)
    trace["events"][0]["exclusive_seconds"] = 2
    with pytest.raises(ValueError, match="double-counted"):
        module.validate_trace(trace)
    trace["blocks"][0]["attention_backend"] = "dense"
    with pytest.raises(ValueError, match="Metal VSA"):
        module.validate_trace(trace)
