"""Tiny-config forward test for the MiniMax-H3 DiT — no weights download required."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.adaln import ModulationCache, drop_adaln_weights, schedule_timesteps
from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession
from minimax_h3_mlx.blockcache import H3BlockCacheConfig
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.easycache import H3EasyCacheConfig
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.ref2va import PreparedReference
from minimax_h3_mlx.trajectory_forecast import H3TrajectoryForecastConfig


def tiny_config() -> DiTConfig:
    hidden = 64
    return DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        ffn_hidden_size=32,
        latents_dim=4,
        audio_latents_dim=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=32,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=2,
    )


def build_packed_layout(n_text: int, n_video: int, n_audio: int):
    """Text rows first, then video, then audio — one contiguous packed sequence."""
    seq = n_text + n_video + n_audio
    text_indices = mx.arange(n_text)
    video_indices = mx.arange(n_text, n_text + n_video)
    audio_indices = mx.arange(n_text + n_video, seq)

    tags = mx.concatenate(
        [
            mx.full((n_text,), TAG_TEXT, dtype=mx.int32),
            mx.full((n_video,), TAG_VIDEO, dtype=mx.int32),
            mx.full((n_audio,), TAG_AUDIO, dtype=mx.int32),
        ]
    )
    # Text and audio sit at the clean level (index 0); video rows at the noisy level (index 1).
    timestep_indices = mx.concatenate(
        [
            mx.zeros((n_text,), dtype=mx.int32),
            mx.ones((n_video,), dtype=mx.int32),
            mx.zeros((n_audio,), dtype=mx.int32),
        ]
    )
    position_ids = mx.stack(
        [mx.arange(seq) % 3, mx.arange(seq) % 5, mx.arange(seq) % 7], axis=-1
    ).astype(mx.int32)
    return text_indices, video_indices, audio_indices, tags, timestep_indices, position_ids


def _forward_fixture():
    cfg = tiny_config()
    mx.random.seed(0)
    dit = MiniMaxH3DiT(cfg)
    mx.eval(dit.parameters())

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, pos = build_packed_layout(n_text, n_video, n_audio)

    video = mx.random.normal((1, n_video, cfg.video_patch_dim))
    audio = mx.random.normal((1, n_audio, cfg.audio_latents_dim))
    text = mx.random.normal((1, n_text, cfg.text_dim))
    timestep = mx.array([0.0, 0.7])

    v_out, a_out = dit(video, audio, text, timestep, ts_i, tags, pos, video_i, audio_i, text_i)
    mx.eval(v_out, a_out)

    assert v_out.shape == (1, n_video, cfg.video_patch_dim), v_out.shape
    assert a_out.shape == (1, n_audio, cfg.audio_latents_dim), a_out.shape
    assert not mx.any(mx.isnan(v_out)).item()
    assert not mx.any(mx.isnan(a_out)).item()
    print(f"forward ok: video {v_out.shape}, audio {a_out.shape}")
    return dit, cfg, (video, audio, text, timestep, ts_i, tags, pos, video_i, audio_i, text_i)


def test_forward_shapes():
    _forward_fixture()


def test_query_chunked_attention_matches_dense_within_bf16_precision():
    dit, _, args = _forward_fixture()
    dense_v, dense_a = dit(*args)
    mx.eval(dense_v, dense_a)

    dit.set_attention_query_chunk_size(4)
    chunked_v, chunked_a = dit(*args)
    mx.eval(chunked_v, chunked_a)

    np.testing.assert_allclose(np.asarray(chunked_v), np.asarray(dense_v), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(chunked_a), np.asarray(dense_a), rtol=2e-2, atol=2e-2)
    assert all(block.attn.query_chunk_size == 4 for block in dit.blocks)


def test_modulation_cache_matches_live_projection():
    """The precomputed AdaLN table must reproduce the live projection bit-for-bit in float32."""
    dit, _, args = _forward_fixture()
    timestep = args[3]

    live_v, live_a = dit(*args)
    mx.eval(live_v, live_a)

    cache = ModulationCache.build(dit, timestep, dtype=mx.float32)
    cached_v, cached_a = dit(*args, modulation_cache=cache)
    mx.eval(cached_v, cached_a)

    dv = float(mx.max(mx.abs(live_v - cached_v)).item())
    da = float(mx.max(mx.abs(live_a - cached_a)).item())
    assert dv == 0.0 and da == 0.0, f"cache mismatch: video {dv}, audio {da}"
    print(f"modulation cache exact: video delta {dv}, audio delta {da}")

    # Once cached, the projections can be dropped and the model still runs.
    freed = drop_adaln_weights(dit)
    after_v, after_a = dit(*args, modulation_cache=cache)
    mx.eval(after_v, after_a)
    assert float(mx.max(mx.abs(after_v - cached_v)).item()) == 0.0
    total = sum(p.size for _, p in _flatten(dit.parameters()))
    print(f"dropped adaln, freeing {freed / 1024:.1f} KB; {total:,} params remain")


def test_schedule_timesteps():
    sigmas = mx.array([1.0, 0.75, 0.5, 0.25])
    ts = schedule_timesteps(sigmas)
    assert ts.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0], ts.tolist()
    print(f"schedule timesteps: {ts.tolist()}")


def test_transformer_only_text_sampling_shapes():
    cfg = tiny_config()
    mx.random.seed(4)
    dit = MiniMaxH3DiT(cfg)
    pipeline = MiniMaxH3Pipeline(dit, None, None, None)
    progress = []

    result = pipeline.sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=3,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        step_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)
    assert result.transformer_evaluations == 2
    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_transformer_only_continuation_sampling_shapes():
    cfg = tiny_config()
    mx.random.seed(5)
    pipeline = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None)

    result = pipeline.sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=3,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        continuation_video_latents=mx.random.normal((1, cfg.latents_dim, 7, 2, 2)),
        continuation_audio_latents=mx.random.normal((2, cfg.audio_latents_dim, 37)),
        continuation_frames=22,
    )

    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)
    assert result.transformer_evaluations == 2


def test_continuation_target_only_trajectory_forecast_excludes_fixed_rows():
    cfg = tiny_config()
    mx.random.seed(6)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        continuation_video_latents=mx.random.normal((1, cfg.latents_dim, 7, 2, 2)),
        continuation_audio_latents=mx.random.normal((2, cfg.audio_latents_dim, 37)),
        continuation_frames=22,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="automatic_speed",
            max_delta_ratio=100.0,
            offline_smoothing_replay=True,
            conditioned_row_policy="target_only",
        ),
    )

    assert result.trajectory_conditioned_row_policy == "target_only"
    assert result.trajectory_excluded_video_rows == 7
    assert result.trajectory_excluded_audio_rows == 74
    assert result.trajectory_forecasts == 1
    assert result.transformer_evaluations == 3
    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)


def test_continuation_hidden_state_becomes_target_dependent_after_block_zero(tmp_path):
    cfg = tiny_config()
    session = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("block_input",),
            blocks=(0, 1),
            evaluation_indices=(0, 1),
            max_total_bytes=32 * 1024 * 1024,
        )
    )
    MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=3,
        seed=19,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        continuation_video_latents=mx.random.normal((1, cfg.latents_dim, 7, 2, 2)),
        continuation_audio_latents=mx.random.normal((2, cfg.audio_latents_dim, 37)),
        continuation_frames=22,
        diagnostics=session,
    )

    captures = {
        (item["block"], item["evaluation_index"]): mx.load(str(tmp_path / item["path"]))[
            "tensor_0"
        ]
        for item in session.captures
    }
    # Three text rows, seven video-context rows, then 74 audio-context rows.
    context_indices = mx.concatenate((mx.arange(3, 10), mx.arange(10, 84)))
    block0_first = captures[(0, 0)][:, context_indices]
    block0_second = captures[(0, 1)][:, context_indices]
    block1_first = captures[(1, 0)][:, context_indices]
    block1_second = captures[(1, 1)][:, context_indices]
    mx.eval(block0_first, block0_second, block1_first, block1_second)

    assert mx.array_equal(block0_first, block0_second).item()
    assert not mx.array_equal(block1_first, block1_second).item()
    assert float(mx.max(mx.abs(block1_first - block1_second)).item()) > 0


def test_terminal_target_only_matches_full_terminal_block_within_bf16_precision():
    cfg = tiny_config()
    mx.random.seed(29)
    pipeline = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None)
    embeddings = mx.random.normal((1, 3, cfg.text_dim))
    video_context = mx.random.normal((1, cfg.latents_dim, 7, 2, 2))
    audio_context = mx.random.normal((2, cfg.audio_latents_dim, 37))
    arguments = {
        "duration_seconds": 5.0,
        "num_inference_steps": 3,
        "seed": 31,
        "height": 32,
        "width": 32,
        "drop_adaln": False,
        "verbose": False,
        "continuation_video_latents": video_context,
        "continuation_audio_latents": audio_context,
        "continuation_frames": 22,
    }
    baseline = pipeline.sample_latents(
        embeddings,
        np.full((3,), TAG_TEXT, dtype=np.int32),
        **arguments,
    )
    candidate = pipeline.sample_latents(
        embeddings,
        np.full((3,), TAG_TEXT, dtype=np.int32),
        terminal_target_only=True,
        **arguments,
    )
    mx.eval(
        baseline.video_latents,
        baseline.audio_latents,
        candidate.video_latents,
        candidate.audio_latents,
    )

    np.testing.assert_allclose(
        np.asarray(candidate.video_latents),
        np.asarray(baseline.video_latents),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(candidate.audio_latents),
        np.asarray(baseline.audio_latents),
        rtol=2e-2,
        atol=2e-2,
    )
    assert mx.array_equal(candidate.video_latents, baseline.video_latents).item()
    assert mx.array_equal(candidate.audio_latents, baseline.audio_latents).item()


def test_terminal_target_only_rejects_trajectory_forecast():
    cfg = tiny_config()
    with pytest.raises(ValueError, match="resident dense transformer"):
        MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
            mx.random.normal((1, 3, cfg.text_dim)),
            np.full((3,), TAG_TEXT, dtype=np.int32),
            duration_seconds=5.0,
            num_inference_steps=3,
            seed=37,
            height=32,
            width=32,
            drop_adaln=False,
            verbose=False,
            terminal_target_only=True,
            trajectory_forecast_config=H3TrajectoryForecastConfig(
                mode="manual",
                max_forecast_fraction=0.0,
            ),
        )


def test_transformer_only_ref2va_sampling_keeps_reference_rows_fixed():
    cfg = tiny_config()
    mx.random.seed(14)
    pipeline = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None)
    reference = PreparedReference(
        "video",
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
    )
    embeddings = mx.random.normal((1, 2, cfg.text_dim))
    video_condition = mx.random.normal((1, cfg.video_patch_dim))
    audio_condition = mx.random.normal((4, cfg.audio_latents_dim))
    arguments = dict(
        duration_seconds=5.0,
        num_inference_steps=3,
        seed=27,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        condition_video_rows=video_condition,
        condition_audio_rows=audio_condition,
        references=(reference,),
    )
    result = pipeline.sample_latents(
        embeddings,
        np.array([TAG_TEXT, TAG_VIDEO], dtype=np.int32),
        **arguments,
    )
    explicit_defaults = pipeline.sample_latents(
        embeddings,
        np.array([TAG_TEXT, TAG_VIDEO], dtype=np.int32),
        visual_condition_strength=0.999,
        audio_condition_strength=1.0,
        **arguments,
    )
    weakened = pipeline.sample_latents(
        embeddings,
        np.array([TAG_TEXT, TAG_VIDEO], dtype=np.int32),
        visual_condition_strength=0.7,
        audio_condition_strength=0.8,
        **arguments,
    )
    mx.eval(
        result.video_latents,
        result.audio_latents,
        explicit_defaults.video_latents,
        explicit_defaults.audio_latents,
        weakened.video_latents,
        weakened.audio_latents,
    )

    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)
    assert result.transformer_evaluations == 2
    assert mx.array_equal(result.video_latents, explicit_defaults.video_latents)
    assert mx.array_equal(result.audio_latents, explicit_defaults.audio_latents)
    assert not mx.array_equal(result.video_latents, weakened.video_latents)
    assert not mx.array_equal(result.audio_latents, weakened.audio_latents)


def test_h3_easycache_skips_joint_video_audio_evaluation():
    cfg = tiny_config()
    mx.random.seed(4)
    pipeline = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None)

    result = pipeline.sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=6,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        easycache_config=H3EasyCacheConfig(
            reuse_threshold=100.0,
            start_percent=0.0,
            end_percent=1.0,
        ),
    )

    assert result.easycache_skipped_steps > 0
    assert result.transformer_evaluations + result.easycache_skipped_steps == 5
    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)


def test_h3_blockcache_reuses_later_blocks_but_runs_each_sampling_step():
    cfg = tiny_config()
    mx.random.seed(4)
    pipeline = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None)

    result = pipeline.sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=8,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        blockcache_config=H3BlockCacheConfig(
            mode="manual",
            reuse_threshold=100.0,
            start_percent=0.0,
            end_percent=1.0,
            max_hit_fraction=0.6,
        ),
    )

    assert result.blockcache_hits > 0
    assert result.transformer_evaluations + result.blockcache_hits == 7
    assert result.blockcache_cache_bytes > 0
    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)


def test_h3_trajectory_forecast_runs_current_heads_on_turbo_length_schedule():
    cfg = tiny_config()
    mx.random.seed(9)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="manual",
            forecast_strength=0.75,
            warmup_steps=2,
            tail_actual_steps=1,
            max_forecast_fraction=0.5,
            max_delta_ratio=100.0,
        ),
    )

    assert result.trajectory_forecasts == 1
    assert result.transformer_evaluations == 3
    assert result.trajectory_history_bytes > 0
    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)


def test_h3_trajectory_bootstrap_runs_current_heads_on_second_step():
    cfg = tiny_config()
    mx.random.seed(9)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="automatic_speed",
            bootstrap_first_forecast=True,
            max_delta_ratio=100.0,
        ),
    )

    assert result.trajectory_forecasts == 1
    assert result.trajectory_bootstrap_forecasts == 1
    assert result.transformer_evaluations == 3
    assert result.video_latents.shape == (1, cfg.latents_dim, 37, 2, 2)
    assert result.audio_latents.shape == (2, cfg.audio_latents_dim, 207)


def test_h3_offline_replay_restarts_and_runs_heads_without_transformer_blocks():
    cfg = tiny_config()
    mx.random.seed(9)
    progress = []
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        step_callback=lambda completed, total: progress.append((completed, total)),
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="automatic_balanced",
            offline_smoothing_replay=True,
        ),
    )

    assert result.transformer_evaluations == 3
    assert result.trajectory_forecasts == 1
    assert result.trajectory_offline_replay is True
    assert result.trajectory_replay_steps == 4
    assert result.trajectory_replay_anchor_steps == 3
    assert result.trajectory_replay_smoothed_steps == 1
    assert result.trajectory_history_bytes > 0
    assert result.trajectory_capture_seconds > 0
    assert result.trajectory_replay_seconds > 0
    assert result.trajectory_replay_fallback_reason is None
    assert progress == [(index, 8) for index in range(9)]
    assert bool(mx.all(mx.isfinite(result.video_latents)).item())
    assert bool(mx.all(mx.isfinite(result.audio_latents)).item())


def test_h3_offline_replay_releases_archive_when_cancelled(monkeypatch):
    import minimax_h3_mlx.trajectory_forecast as trajectory_module

    original = trajectory_module.H3TrajectoryForecastState

    class TrackingState(original):
        instances = []

        def __init__(self, config):
            super().__init__(config)
            self.instances.append(self)

    monkeypatch.setattr(trajectory_module, "H3TrajectoryForecastState", TrackingState)
    cfg = tiny_config()
    mx.random.seed(9)

    def cancel_during_replay(completed, _total):
        if completed == 5:
            raise RuntimeError("synthetic cancellation")

    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
            mx.random.normal((1, 3, cfg.text_dim)),
            np.full((3,), TAG_TEXT, dtype=np.int32),
            duration_seconds=5.0,
            num_inference_steps=5,
            height=32,
            width=32,
            drop_adaln=False,
            verbose=False,
            step_callback=cancel_during_replay,
            trajectory_forecast_config=H3TrajectoryForecastConfig(
                mode="automatic_balanced",
                offline_smoothing_replay=True,
            ),
        )

    assert len(TrackingState.instances) == 1
    assert TrackingState.instances[0].archive_bytes == 0
    assert TrackingState.instances[0]._history == []


def test_h3_offline_replay_failure_returns_valid_capture_result(monkeypatch):
    import minimax_h3_mlx.trajectory_forecast as trajectory_module

    original = trajectory_module.H3TrajectoryForecastState

    class FailingReplayState(original):
        def _replay_prediction(self, coordinate, index, total_steps):
            if index == 2:
                raise trajectory_module.H3OfflineReplayError("synthetic replay failure")
            return super()._replay_prediction(coordinate, index, total_steps)

    monkeypatch.setattr(trajectory_module, "H3TrajectoryForecastState", FailingReplayState)
    cfg = tiny_config()
    mx.random.seed(12)
    failed_replay = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="automatic_balanced",
            offline_smoothing_replay=True,
        ),
    )

    monkeypatch.setattr(trajectory_module, "H3TrajectoryForecastState", original)
    mx.random.seed(12)
    capture_control = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="manual",
            forecast_strength=0.0,
            warmup_steps=2,
            tail_actual_steps=1,
            max_forecast_fraction=0.5,
            max_delta_ratio=100.0,
        ),
    )

    assert failed_replay.trajectory_replay_fallback_reason == "synthetic replay failure"
    assert failed_replay.trajectory_fallbacks == 1
    assert mx.array_equal(failed_replay.video_latents, capture_control.video_latents)
    assert mx.array_equal(failed_replay.audio_latents, capture_control.audio_latents)


def test_h3_offline_anchor_only_replay_matches_dense_sampling_exactly():
    cfg = tiny_config()
    mx.random.seed(21)
    replayed = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        trajectory_forecast_config=H3TrajectoryForecastConfig(
            mode="manual",
            max_forecast_fraction=0.0,
            offline_smoothing_replay=True,
        ),
    )

    mx.random.seed(21)
    dense = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=5,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
    )

    assert replayed.trajectory_forecasts == 0
    assert replayed.trajectory_replay_anchor_steps == 4
    assert replayed.trajectory_replay_smoothed_steps == 0
    assert mx.array_equal(replayed.video_latents, dense.video_latents)
    assert mx.array_equal(replayed.audio_latents, dense.audio_latents)


def test_h3_easycache_conservative_auto_calibrates_and_caps_short_schedule():
    cfg = tiny_config()
    mx.random.seed(5)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=8,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        easycache_config=H3EasyCacheConfig(mode="automatic_conservative"),
    )

    assert result.easycache_skipped_steps == 1
    assert result.transformer_evaluations == 6
    assert result.easycache_resolved_threshold is not None
    assert 0.05 <= result.easycache_resolved_threshold <= 0.5


def test_h3_easycache_speed_auto_is_more_aggressive_but_bounded():
    cfg = tiny_config()
    mx.random.seed(5)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=8,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        easycache_config=H3EasyCacheConfig(mode="automatic_speed"),
    )

    assert 1 < result.easycache_skipped_steps <= 3
    assert result.transformer_evaluations + result.easycache_skipped_steps == 7
    assert result.easycache_resolved_threshold is not None
    assert 0.25 <= result.easycache_resolved_threshold <= 1.25


def test_h3_easycache_balanced_auto_fills_middle_policy():
    cfg = tiny_config()
    mx.random.seed(5)
    result = MiniMaxH3Pipeline(MiniMaxH3DiT(cfg), None, None, None).sample_latents(
        mx.random.normal((1, 3, cfg.text_dim)),
        np.full((3,), TAG_TEXT, dtype=np.int32),
        duration_seconds=5.0,
        num_inference_steps=8,
        height=32,
        width=32,
        drop_adaln=False,
        verbose=False,
        easycache_config=H3EasyCacheConfig(mode="automatic_balanced"),
    )

    assert result.easycache_skipped_steps == 2
    assert result.transformer_evaluations == 5
    assert result.easycache_resolved_threshold is not None
    assert 0.1 <= result.easycache_resolved_threshold <= 0.8


def test_pruned_adaln_curve_forward_and_cache():
    """Pruned checkpoints interpolate the curve and bypass both timestep SiLU stages."""
    cfg = replace(tiny_config(), time_embed_dim=8, adaln_curve_grid=5)
    mx.random.seed(3)
    dit = MiniMaxH3DiT(cfg)
    table = mx.arange(40, dtype=mx.float32).reshape(5, 8)
    dit.adaln_t_table = table
    mx.eval(dit.parameters())

    interpolated = dit.embed_timesteps(mx.array([0.0, 0.125, 1.0]))
    expected = mx.stack([table[0], (table[0] + table[1]) / 2, table[-1]])
    mx.eval(interpolated)
    assert float(mx.max(mx.abs(interpolated - expected)).item()) == 0.0

    n_text, n_video, n_audio = 5, 9, 3
    text_i, video_i, audio_i, tags, ts_i, pos = build_packed_layout(n_text, n_video, n_audio)
    args = (
        mx.random.normal((1, n_video, cfg.video_patch_dim)),
        mx.random.normal((1, n_audio, cfg.audio_latents_dim)),
        mx.random.normal((1, n_text, cfg.text_dim)),
        mx.array([0.0, 0.7]),
        ts_i,
        tags,
        pos,
        video_i,
        audio_i,
        text_i,
    )
    live_v, live_a = dit(*args)
    cache = ModulationCache.build(dit, args[3], dtype=mx.float32)
    cached_v, cached_a = dit(*args, modulation_cache=cache)
    mx.eval(live_v, live_a, cached_v, cached_a)
    assert float(mx.max(mx.abs(live_v - cached_v)).item()) == 0.0
    assert float(mx.max(mx.abs(live_a - cached_a)).item()) == 0.0
    print("pruned AdaLN curve interpolation and cached forward ok")


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flatten(v, f"{prefix}.{i}")
    elif isinstance(tree, mx.array):
        yield prefix, tree


if __name__ == "__main__":
    test_schedule_timesteps()
    test_pruned_adaln_curve_forward_and_cache()
    test_modulation_cache_matches_live_projection()
    print("\nall smoke tests passed")
