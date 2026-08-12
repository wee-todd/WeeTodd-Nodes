import mlx.core as mx
import pytest

from minimax_h3_mlx.hires_fix import (
    resize_video_latents,
    resize_video_latents_bicubic,
    resize_video_latents_bilinear,
    resize_video_latents_lanczos,
    resize_video_latents_nearest_exact,
    resolve_hires_canvas,
)


def test_resolve_hires_canvas_rounds_to_h3_patch_geometry():
    assert resolve_hires_canvas(672, 384, 1.5) == (1024, 576)
    assert resolve_hires_canvas(640, 384, 1.5) == (960, 576)


def test_resolve_hires_canvas_enforces_public_limits():
    with pytest.raises(ValueError, match="axis limit"):
        resolve_hires_canvas(1344, 768, 2.0)


def test_resize_video_latents_bilinear_stays_in_mlx_and_preserves_corners():
    source = mx.array([[[[[0.0, 1.0], [2.0, 3.0]]]]])
    resized = resize_video_latents_bilinear(source, 4, 4)
    mx.eval(resized)

    assert tuple(resized.shape) == (1, 1, 1, 4, 4)
    assert type(resized).__module__.startswith("mlx.")
    assert float(resized[0, 0, 0, 0, 0].item()) == 0.0
    assert float(resized[0, 0, 0, -1, -1].item()) == 3.0


def test_resize_video_latents_nearest_exact_repeats_source_vectors():
    source = mx.array([[[[[0.0, 1.0], [2.0, 3.0]]]]])
    resized = resize_video_latents_nearest_exact(source, 4, 4)
    mx.eval(resized)

    assert resized.tolist() == [
        [[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0]]]]
    ]


def test_resize_video_latents_bicubic_preserves_constant_latents():
    source = mx.ones((1, 2, 3, 2, 3), dtype=mx.float32) * 0.375
    resized = resize_video_latents_bicubic(source, 5, 7)
    mx.eval(resized)

    assert tuple(resized.shape) == (1, 2, 3, 5, 7)
    assert bool(mx.allclose(resized, mx.full_like(resized, 0.375), atol=1e-6).item())


def test_resize_video_latents_lanczos_preserves_constant_latents_and_dtype():
    source = mx.ones((1, 2, 3, 2, 3), dtype=mx.bfloat16) * 0.375
    resized = resize_video_latents_lanczos(source, 5, 7)
    mx.eval(resized)

    assert tuple(resized.shape) == (1, 2, 3, 5, 7)
    assert resized.dtype == mx.bfloat16
    assert bool(mx.allclose(resized, mx.full_like(resized, 0.375), atol=1e-3).item())


def test_resize_video_latents_lanczos_is_distinct_and_validates_radius():
    source = mx.array([[[[[0.0, 1.0], [2.0, 3.0]]]]])
    lanczos = resize_video_latents(source, 5, 5, method="lanczos-3")
    bicubic = resize_video_latents(source, 5, 5, method="bicubic")
    mx.eval(lanczos, bicubic)

    assert not bool(mx.allclose(lanczos, bicubic).item())
    with pytest.raises(ValueError, match="radius must be at least 2"):
        resize_video_latents_lanczos(source, 5, 5, radius=1)


def test_resize_video_latents_dispatches_and_rejects_unknown_method():
    source = mx.array([[[[[0.0, 1.0], [2.0, 3.0]]]]])
    bicubic = resize_video_latents(source, 5, 5, method="bicubic")
    bilinear = resize_video_latents(source, 5, 5, method="bilinear")
    mx.eval(bicubic, bilinear)

    assert not bool(mx.allclose(bicubic, bilinear).item())
    with pytest.raises(ValueError, match="Unsupported H3 latent resize method"):
        resize_video_latents(source, 5, 5, method="lanczos")
