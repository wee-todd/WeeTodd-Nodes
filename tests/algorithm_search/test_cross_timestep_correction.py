import mlx.core as mx

from minimax_h3_mlx.algorithm_search.benchmark import numerical_metrics


def test_per_channel_input_delta_can_reconstruct_linear_change():
    x0 = mx.zeros((8, 3), dtype=mx.float32)
    x1 = mx.arange(24, dtype=mx.float32).reshape(8, 3) / 10
    y0 = mx.ones((8, 3), dtype=mx.float32)
    scale = mx.array([0.5, 1.5, -0.25])
    y1 = y0 + scale * (x1 - x0)
    dx = x1[:4] - x0[:4]
    dy = y1[:4] - y0[:4]
    fitted = mx.sum(dx * dy, axis=0) / mx.maximum(
        mx.sum(dx * dx, axis=0), mx.array(1e-12)
    )
    predicted = y0[4:] + fitted * (x1[4:] - x0[4:])
    assert numerical_metrics(y1[4:], predicted).relative_l2_error < 1e-6
