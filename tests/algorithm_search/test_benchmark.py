import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import benchmark_candidate, numerical_metrics
from minimax_h3_mlx.algorithm_search.schema import AlgorithmClass


def test_numerical_metrics_report_exact_and_approximate_outputs():
    exact = numerical_metrics(mx.array([1.0, 2.0]), mx.array([1.0, 2.0]))
    approximate = numerical_metrics(mx.array([1.0, 2.0]), mx.array([1.0, 2.1]))
    assert exact.max_absolute_error == 0.0
    assert exact.cosine_similarity == 1.0
    assert approximate.rmse > 0.0
    assert approximate.relative_l2_error > 0.0


def test_numerical_metrics_accept_bfloat16_outputs():
    values = mx.array([1.0, 2.0], dtype=mx.bfloat16)
    metrics = numerical_metrics(values, values)
    assert metrics.max_absolute_error == 0.0
    assert metrics.relative_l2_error == 0.0


def test_candidate_benchmark_is_serializable_and_classified():
    inputs = mx.ones((8, 4), dtype=mx.float32)
    result = benchmark_candidate(
        lambda x: x * 2,
        lambda x: x + x,
        (inputs,),
        candidate_id="exact_double",
        operator="test",
        algorithm_class=AlgorithmClass.EXACT,
        warmups=0,
        repetitions=2,
    )
    assert result.passed_error_budget
    assert result.to_dict()["algorithm_class"] == "exact"
    assert result.baseline.repetitions == 2


def test_candidate_benchmark_accepts_smaller_error_inputs():
    timing = mx.ones((128, 8), dtype=mx.float32)
    held_out = mx.ones((4, 8), dtype=mx.float32)
    result = benchmark_candidate(
        lambda x: x * 3,
        lambda x: x + x + x,
        (timing,),
        error_inputs=(held_out,),
        candidate_id="held_out_exact",
        operator="test",
        algorithm_class=AlgorithmClass.EXACT,
        warmups=0,
        repetitions=1,
    )
    assert result.passed_error_budget
    assert result.input_shapes == ((128, 8),)
