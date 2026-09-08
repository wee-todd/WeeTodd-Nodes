import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

VisionConfig = pytest.importorskip("mlx_vlm.models.qwen3_vl.config").VisionConfig
VisionModel = pytest.importorskip("mlx_vlm.models.qwen3_vl.vision").VisionModel
encode_vision = pytest.importorskip("minimax_h3_mlx.vision_forward").encode_vision


@pytest.mark.parametrize("rows", [[[1, 2, 4]], [[2, 4, 4]], [[1, 2, 2], [1, 4, 4]]])
def test_qwen_vision_matches_upstream_with_scalar_repeat_corrected(monkeypatch, rows):
    mx.random.seed(17)
    config = VisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    vision = VisionModel(config)
    grid = mx.array(rows, dtype=mx.int32)
    pixels = mx.random.normal((sum(t * h * w for t, h, w in rows), 24))
    repeat = mx.repeat
    with monkeypatch.context() as patch:
        # Correct only the upstream scalar argument to establish a numerical oracle.
        patch.setattr(
            mx,
            "repeat",
            lambda array, repeats, *a, **k: repeat(
                array, int(repeats.item()) if isinstance(repeats, mx.array) else repeats, *a, **k
            ),
        )
        expected, expected_deep = vision(pixels, grid)
        mx.eval(expected, *expected_deep)
    actual, actual_deep = encode_vision(vision, pixels, grid)
    assert len(actual_deep) == len(expected_deep) == 2
    for left, right in zip([expected, *expected_deep], [actual, *actual_deep], strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_vision_grid_rejected_before_weighted_work():
    with pytest.raises(ValueError, match="positive"):
        encode_vision(None, None, mx.array([[0, 2, 2]]))
