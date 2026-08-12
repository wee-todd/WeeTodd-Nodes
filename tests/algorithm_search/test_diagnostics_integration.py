import mlx.core as mx
import numpy as np

from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession
from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT


def test_selected_diagnostics_preserve_tiny_transformer_output(tmp_path):
    config = DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=24,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        adaln_out_features=6 * 3 * 32,
        final_adaln_out_features=2 * 32,
        rope_inv_freq_len=2,
    )
    mx.random.seed(7)
    model = MiniMaxH3DiT(config)
    text_rows, video_rows, audio_rows = 2, 4, 2
    sequence = text_rows + video_rows + audio_rows
    text_indices = mx.arange(text_rows)
    audio_indices = mx.arange(text_rows, text_rows + audio_rows)
    video_indices = mx.arange(text_rows + audio_rows, sequence)
    tags = mx.array(
        [TAG_TEXT] * text_rows + [TAG_AUDIO] * audio_rows + [TAG_VIDEO] * video_rows,
        dtype=mx.int32,
    )
    timestep_indices = mx.array([0] * text_rows + [0] * audio_rows + [1] * video_rows)
    position_ids = mx.array(
        np.stack([np.arange(sequence) % 3, np.arange(sequence) % 5, np.arange(sequence) % 7], -1)
    )
    inputs = (
        mx.random.normal((1, video_rows, config.video_patch_dim)),
        mx.random.normal((1, audio_rows, config.audio_latents_dim)),
        mx.random.normal((1, text_rows, config.text_dim)),
        mx.array([0.0, 0.5]),
        timestep_indices,
        tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
    )
    baseline = model(*inputs)
    mx.eval(*baseline)
    diagnostics = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("q_output", "mlp_input", "attention_qkv"),
            blocks=(0,),
            attention_heads=(0,),
            profile_blocks=(0,),
            max_total_bytes=1024 * 1024,
            profile_regions=True,
        )
    )
    measured = model(*inputs, diagnostics=diagnostics)
    mx.eval(*measured)
    np.testing.assert_array_equal(np.asarray(measured[0]), np.asarray(baseline[0]))
    np.testing.assert_array_equal(np.asarray(measured[1]), np.asarray(baseline[1]))
    assert diagnostics.measurements
    assert {item["name"] for item in diagnostics.captures} == {
        "q_output",
        "mlp_input",
        "attention_qkv",
    }
    assert all(item.block in {0, None} for item in diagnostics.measurements)
    q_only_diagnostics = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("q_output",),
            blocks=(0,),
            max_total_bytes=1024 * 1024,
        )
    )
    q_only_result = model(*inputs, diagnostics=q_only_diagnostics)
    mx.eval(*q_only_result)
    assert q_only_diagnostics.captures
    assert {item["name"] for item in q_only_diagnostics.captures} == {"q_output"}
    names = {item.name for item in diagnostics.measurements}
    assert "input.video_projection" in names
    assert "output.video_projection" in names


def test_external_diagnostic_measurement_is_opt_in(tmp_path):
    inactive = DiagnosticSession(CaptureConfig(output_directory=str(tmp_path / "inactive")))
    inactive.record_external("packing.layout", 0.25, metadata={"sequence_rows": 12})
    assert inactive.measurements == []

    active = DiagnosticSession(
        CaptureConfig(
            output_directory=str(tmp_path / "active"),
            profile_regions=True,
        )
    )
    active.begin_evaluation(2, timestep=0.5)
    active.record_external("scheduler.step_and_repack", 0.125, metadata={"rows": 12})

    assert len(active.measurements) == 1
    measurement = active.measurements[0]
    assert measurement.name == "scheduler.step_and_repack"
    assert measurement.duration_seconds == 0.125
    assert measurement.evaluation_index == 2
    assert measurement.metadata == {"rows": 12}
