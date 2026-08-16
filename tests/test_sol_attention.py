import mlx.core as mx

from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import Attention
from minimax_h3_mlx.packing import build_packed_sequence
from minimax_h3_mlx.sol_attention import SolAttentionConfig, sol_attention
from wee_todd_mlx.execution_evidence import ExecutionEvidence, require_executed


def test_sol_attention_schedule_and_prefix_copy():
    config = SolAttentionConfig(enabled=True, dense_blocks=2, start_percent=0.2)
    assert config.active(step_index=0, total_steps=8, block_index=24) is False
    assert config.active(step_index=3, total_steps=8, block_index=1) is False
    assert config.active(step_index=3, total_steps=8, block_index=24) is True
    assert config.with_prefix(602).exact_prefix_rows == 602
    assert config.exact_prefix_rows == 0


def test_h3_latents_can_report_runtime_resolved_sol_prefix():
    from wee_todd_nodes.sampling import H3Latents

    assert "sol_attention_report" in H3Latents.__dataclass_fields__


def test_h3_sol_prefix_keeps_text_conditioning_and_audio_exact():
    layout = build_packed_sequence(
        [1, 1, 1],
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first",),
    )
    prefix_rows = int(
        layout.video_indices[layout.num_condition_video_rows].item()
    )

    assert max(layout.text_indices.tolist()) < prefix_rows
    assert max(layout.audio_indices.tolist()) < prefix_rows
    assert max(
        layout.video_indices[: layout.num_condition_video_rows].tolist()
    ) < prefix_rows
    assert min(
        layout.video_indices[layout.num_condition_video_rows :].tolist()
    ) == prefix_rows


def test_sol_attention_exact_routes_match_dense_bf16():
    mx.random.seed(7)
    query = mx.random.normal((1, 1, 128, 128)).astype(mx.bfloat16)
    key = mx.random.normal((1, 1, 128, 128)).astype(mx.bfloat16)
    value = mx.random.normal((1, 1, 128, 128)).astype(mx.bfloat16)
    scale = 128**-0.5
    config = SolAttentionConfig(
        enabled=True,
        min_tokens=64,
        exact_prefix_rows=128,
        dense_blocks=0,
        start_percent=0.0,
    )

    expected = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)
    actual, route_counts = sol_attention(
        query,
        key,
        value,
        scale=scale,
        config=config,
        return_route_counts=True,
    )
    mx.eval(expected, actual, route_counts)

    delta = actual.astype(mx.float32) - expected.astype(mx.float32)
    relative_l2 = mx.sqrt(mx.sum(delta * delta) / mx.sum(expected.astype(mx.float32) ** 2))
    assert float(relative_l2.item()) < 1.0e-4
    assert mx.sum(route_counts, axis=(0, 1, 2)).tolist() == [8, 0]


def test_execution_evidence_flags_requested_backend_that_did_no_work():
    evidence = ExecutionEvidence("sol_attention", "mlx_fused_sol_bf16", "test")
    evidence.record_call()
    evidence.record_fallback("unsupported_dtype")
    report = evidence.snapshot()

    assert report["requested_but_not_executed"] is True
    assert report["fallback_counts"] == {"unsupported_dtype": 1}
    try:
        require_executed(report)
    except RuntimeError as exc:
        assert "executed zero calls" in str(exc)
    else:
        raise AssertionError("Zero-work benchmark evidence was accepted.")


def test_h3_attention_reports_actual_sol_execution_and_fallback():
    attention = Attention(DiTConfig(num_layers=1, num_attention_heads=1, hidden_size=128))
    config = SolAttentionConfig(
        enabled=True,
        min_tokens=64,
        exact_prefix_rows=64,
        dense_blocks=0,
        start_percent=0.0,
    )
    evidence = ExecutionEvidence(
        "sol_attention", "mlx_fused_sol_bf16", "H3 packed transformer self-attention"
    )
    attention.sol_config = config
    attention.sol_evidence = evidence
    route_records = []
    attention.sol_route_records = route_records
    query = mx.random.normal((1, 1, 64, 128)).astype(mx.bfloat16)
    output = attention._attend(query, query, query, None, 0)
    mx.eval(output)

    report = evidence.snapshot()
    assert report["eligible_calls"] == 1
    assert report["executed_calls"] == 1
    assert report["requested_but_not_executed"] is False
    assert len(route_records) == 1
    assert mx.sum(route_records[0][2], axis=(0, 1, 2)).tolist() == [2, 0]

    float_query = query.astype(mx.float32)
    mx.eval(attention._attend(float_query, float_query, float_query, None, 0))
    report = evidence.snapshot()
    assert report["fallback_counts"] == {"unsupported_dtype": 1}
