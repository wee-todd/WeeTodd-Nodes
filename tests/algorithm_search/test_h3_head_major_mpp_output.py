import mlx.core as mx

from minimax_h3_mlx.algorithm_search.head_major_mpp_output import head_major_mpp_output


def test_head_major_mpp_output_has_bounded_partial_accumulation_error():
    source = mx.random.normal((1, 2, 7, 128), key=mx.random.key(91), dtype=mx.bfloat16)
    weight = mx.random.normal((64, 256), key=mx.random.key(92), dtype=mx.bfloat16)
    reference = source.transpose(0, 2, 1, 3).reshape(1, 7, 256) @ weight.T
    candidate = head_major_mpp_output(source, weight)
    mx.eval(reference, candidate)

    assert candidate.shape == (1, 7, 64)
    reference_fp32 = reference.astype(mx.float32)
    difference = candidate.astype(mx.float32) - reference_fp32
    relative_l2 = mx.sqrt(mx.sum(difference**2)) / mx.sqrt(mx.sum(reference_fp32**2))
    assert float(relative_l2.item()) < 0.02


def test_head_major_mpp_output_consumes_a_strided_attention_view():
    token_major = mx.random.normal(
        (1, 7, 2, 128), key=mx.random.key(93), dtype=mx.bfloat16
    )
    source = token_major.transpose(0, 2, 1, 3)
    weight = mx.random.normal((64, 256), key=mx.random.key(94), dtype=mx.bfloat16)
    reference = token_major.reshape(1, 7, 256) @ weight.T
    candidate = head_major_mpp_output(source, weight)
    mx.eval(reference, candidate)

    reference_fp32 = reference.astype(mx.float32)
    difference = candidate.astype(mx.float32) - reference_fp32
    relative_l2 = mx.sqrt(mx.sum(difference**2)) / mx.sqrt(mx.sum(reference_fp32**2))
    assert float(relative_l2.item()) < 0.02


def test_head_major_mpp_output_rejects_weight_width_mismatch():
    source = mx.zeros((1, 2, 7, 128), dtype=mx.bfloat16)
    weight = mx.zeros((64, 128), dtype=mx.bfloat16)
    try:
        head_major_mpp_output(source, weight)
    except ValueError as error:
        assert "heads multiplied by 128" in str(error)
    else:
        raise AssertionError("mismatched output weight width must be rejected")
