import mlx.core as mx

from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession
from minimax_h3_mlx.algorithm_search.hybrid_swap import SelectiveHybridBlockController
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import TransformerBlock


def test_selective_hybrid_block_keeps_video_path_close_to_exact(tmp_path):
    config = DiTConfig(
        hidden_size=32,
        num_attention_heads=2,
        attention_head_dim=16,
        ffn_hidden_size=24,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        adaln_out_features=6 * 3 * 32,
        final_adaln_out_features=64,
        rope_inv_freq_len=2,
    )
    mx.random.seed(23)
    block = TransformerBlock(config)
    controller = SelectiveHybridBlockController(
        block_index=0, text_rows=1, audio_rows=1, apply_evaluation=2
    )
    session = DiagnosticSession(
        CaptureConfig(output_directory=str(tmp_path)), hybrid_controller=controller
    )
    sequence = 7
    indices = mx.zeros((sequence,), dtype=mx.int32)
    modulation = tuple(mx.random.normal((1, config.hidden_size)) for _ in range(6))
    rotary = (
        mx.ones((sequence, config.rotary_dim), dtype=mx.float32),
        mx.zeros((sequence, config.rotary_dim), dtype=mx.float32),
    )
    inputs = [mx.random.normal((1, sequence, config.hidden_size)) for _ in range(3)]

    outputs = []
    for evaluation, value in enumerate(inputs):
        session.begin_evaluation(evaluation, timestep=evaluation / 3)
        output = block(
            value,
            modulation,
            indices,
            rotary,
            diagnostics=session,
            block_index=0,
        )
        mx.eval(output)
        outputs.append(output)

    exact_third = block(inputs[2], modulation, indices, rotary)
    mx.eval(exact_third)
    assert outputs[2].shape == exact_third.shape
    assert mx.allclose(outputs[2][:, 2:], exact_third[:, 2:], rtol=2e-2, atol=2e-2).item()
    assert [item["action"] for item in controller.history] == [
        "observe_exact",
        "observe_exact",
        "apply_hybrid",
    ]


def test_selective_hybrid_controller_rejects_invalid_configuration():
    try:
        SelectiveHybridBlockController(block_index=0, text_rows=0, audio_rows=1)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid hybrid configuration was accepted")
