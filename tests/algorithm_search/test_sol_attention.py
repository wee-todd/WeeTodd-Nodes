import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics
from minimax_h3_mlx.algorithm_search.sol_attention import (
    SolReferenceConfig,
    dense_attention_reference,
    sol_reference_attention,
)


def _qkv(rows=16, heads=2, dim=8):
    mx.random.seed(19)
    shape = (1, heads, rows, dim)
    values = tuple(mx.random.normal(shape).astype(mx.bfloat16) for _ in range(3))
    mx.eval(*values)
    return values


def test_force_all_exact_matches_dense_float32_reference():
    q, k, v = _qkv()
    scale = q.shape[-1] ** -0.5
    dense = dense_attention_reference(q, k, v, scale=scale)
    candidate, telemetry = sol_reference_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolReferenceConfig(prefix_rows=4, block_size=4, force_all_exact=True),
    )
    mx.eval(dense, candidate)
    np.testing.assert_allclose(np.asarray(candidate), np.asarray(dense), rtol=2e-6, atol=2e-6)
    assert telemetry.exact_route_density == 1.0


def test_selected_query_blocks_bound_output_and_report_density():
    q, k, v = _qkv(rows=20, heads=1)
    candidate, telemetry = sol_reference_attention(
        q,
        k,
        v,
        scale=q.shape[-1] ** -0.5,
        config=SolReferenceConfig(
            prefix_rows=4,
            block_size=4,
            beta=0.0,
            target_query_blocks=(1, 3),
        ),
    )
    mx.eval(candidate)
    assert candidate.shape == (1, 1, 8, 8)
    assert telemetry.evaluated_query_blocks == 2
    assert telemetry.evaluated_query_rows == 8
    assert 0.0 <= telemetry.exact_route_density <= 1.0


def test_approximate_correction_improves_over_dropping_skipped_blocks():
    q, k, v = _qkv(rows=24, heads=1)
    scale = q.shape[-1] ** -0.5
    prefix = 4
    query_blocks = (0, 1, 2, 3, 4)
    query_rows = mx.concatenate(
        [q[..., prefix + index * 4 : prefix + (index + 1) * 4, :] for index in query_blocks],
        axis=-2,
    )
    dense = dense_attention_reference(query_rows, k, v, scale=scale)
    corrected, corrected_telemetry = sol_reference_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolReferenceConfig(
            prefix_rows=prefix,
            block_size=4,
            beta=0.5,
            approximate_correction=True,
            target_query_blocks=query_blocks,
        ),
    )
    dropped, dropped_telemetry = sol_reference_attention(
        q,
        k,
        v,
        scale=scale,
        config=SolReferenceConfig(
            prefix_rows=prefix,
            block_size=4,
            beta=0.5,
            approximate_correction=False,
            target_query_blocks=query_blocks,
        ),
    )
    mx.eval(dense, corrected, dropped)
    corrected_error = numerical_metrics(dense, corrected).relative_l2_error
    dropped_error = numerical_metrics(dense, dropped).relative_l2_error
    assert corrected_telemetry.exact_route_density == dropped_telemetry.exact_route_density
    assert corrected_error < dropped_error


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"prefix_rows": 0}, "prefix_rows"),
        ({"prefix_rows": 4, "block_size": 0}, "block_size"),
        ({"prefix_rows": 4, "threshold_mode": "unknown"}, "threshold_mode"),
        ({"prefix_rows": 4, "target_query_blocks": (99,)}, "out of range"),
    ],
)
def test_reference_rejects_invalid_configuration(kwargs, message):
    q, k, v = _qkv()
    with pytest.raises(ValueError, match=message):
        sol_reference_attention(
            q,
            k,
            v,
            scale=q.shape[-1] ** -0.5,
            config=SolReferenceConfig(**kwargs),
        )
