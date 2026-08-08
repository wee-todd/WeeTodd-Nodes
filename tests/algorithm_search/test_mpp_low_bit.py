import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.mpp_low_bit import mpp_low_bit_linear


@pytest.mark.parametrize(
    ("bits", "weight"),
    [
        (8, mx.ones((64, 64), dtype=mx.int8)),
        (4, mx.full((64, 32), 0x11, dtype=mx.uint8)),
    ],
)
def test_mpp_low_bit_constant_projection(bits, weight):
    source = mx.full((1, 32, 64), 0.015625, dtype=mx.bfloat16)
    result = mpp_low_bit_linear(source, weight, bits=bits)
    mx.eval(result)
    assert result.shape == (1, 32, 64)
    assert bool(mx.all(result == 1.0).item())


def test_mpp_low_bit_rejects_incompatible_storage():
    source = mx.ones((32, 64), dtype=mx.bfloat16)
    with pytest.raises(TypeError, match="Int8"):
        mpp_low_bit_linear(source, mx.ones((64, 64)), bits=8)
    with pytest.raises(ValueError, match="shape"):
        mpp_low_bit_linear(source, mx.ones((64, 64), dtype=mx.uint8), bits=4)
