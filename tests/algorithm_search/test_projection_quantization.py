import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.projection_quantization import (
    PROJECTION_QUANTIZATION_SPECS,
    dynamic_input_probe,
    packed_nbytes,
    packed_projection_matmul,
    quantize_projection,
)


@pytest.mark.parametrize("name", sorted(PROJECTION_QUANTIZATION_SPECS))
def test_packed_projection_contract(name):
    spec = PROJECTION_QUANTIZATION_SPECS[name]
    weight = mx.random.normal((64, 128)).astype(mx.bfloat16)
    inputs = mx.random.normal((8, 128)).astype(mx.bfloat16)
    packed = quantize_projection(weight, spec)
    output = packed_projection_matmul(inputs, packed, spec)
    mx.eval(*packed, output)

    assert output.shape == (8, 64)
    assert packed_nbytes(packed) < weight.nbytes


def test_projection_quantization_rejects_incompatible_group_width():
    spec = PROJECTION_QUANTIZATION_SPECS["affine4"]
    with pytest.raises(ValueError, match="not divisible"):
        quantize_projection(mx.ones((4, 65)), spec)


def test_dynamic_input_probe_rejects_affine_without_dispatching():
    spec = PROJECTION_QUANTIZATION_SPECS["affine4"]
    weight = mx.ones((64, 128), dtype=mx.bfloat16)
    packed = quantize_projection(weight, spec)
    supported, reason = dynamic_input_probe(mx.ones((2, 128)), packed, spec)

    assert not supported
    assert reason == "mlx.qqmm only supports nvfp4 and mxfp8"
