import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from minimax_h3_mlx.hires_fix import (
    resize_fl2va_condition_rows,
    resize_video_latents,
    resize_video_latents_bicubic,
    resize_video_latents_bilinear,
    resize_video_latents_lanczos,
    resize_video_latents_nearest_exact,
    resolve_hires_canvas,
    resolve_hires_maximum_canvas,
)


def test_resolve_hires_canvas_rounds_to_h3_patch_geometry():
    assert resolve_hires_canvas(672, 384, 1.5) == (1024, 576)
    assert resolve_hires_canvas(640, 384, 1.5) == (960, 576)


def test_resolve_hires_canvas_enforces_public_limits():
    with pytest.raises(ValueError, match="axis limit"):
        resolve_hires_canvas(1344, 768, 2.0)


def test_resolve_hires_maximum_canvas_obeys_orientation_and_public_limits():
    assert resolve_hires_maximum_canvas(1344, 768) == (1920, 1088)
    assert resolve_hires_maximum_canvas(768, 1344) == (1088, 1920)
    assert resolve_hires_maximum_canvas(768, 768) == (1088, 1088)

    with pytest.raises(ValueError, match="must enlarge"):
        resolve_hires_maximum_canvas(1920, 1088)


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


def _to_reference_upscaler_checkpoint(model):
    weights = {}
    replacements = (
        (".norm1.", ".in_layers.0."),
        (".conv1.", ".in_layers.2."),
        (".emb.", ".emb_layers.1."),
        (".norm2.", ".out_norm."),
        (".conv2.", ".out_layers.2."),
    )
    for key, value in tree_flatten(model.parameters()):
        source_key = key
        for old, new in replacements:
            source_key = source_key.replace(old, new)
        if source_key.endswith(".dwconv.weight"):
            value = value.transpose(1, 0)[:, None, :, None, None]
        elif value.ndim == 5:
            value = value.transpose(0, 4, 1, 2, 3)
        weights[source_key] = value
    return weights


def test_learned_h3_upscaler_loads_reference_layout_and_runs_mlx(tmp_path):
    from minimax_h3_mlx.learned_latent_upscaler import (
        H3LearnedLatentUpscaler,
        upscale_h3_video_latents_learned,
    )

    reference = H3LearnedLatentUpscaler(
        channels=32,
        input_channels=24,
        input_block_kinds=("residual", "temporal"),
        output_block_kinds=("residual",),
        temporal_kernel=5,
    )
    checkpoint = tmp_path / "synthetic_h3_upscaler.safetensors"
    mx.save_safetensors(str(checkpoint), _to_reference_upscaler_checkpoint(reference))
    from wee_todd_nodes.nodes import _is_h3_latent_upscaler_checkpoint

    assert _is_h3_latent_upscaler_checkpoint(checkpoint) is True
    source = mx.zeros((1, 24, 2, 2, 2), dtype=mx.bfloat16)
    progress = []

    result, report = upscale_h3_video_latents_learned(
        source,
        3,
        4,
        checkpoint,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert tuple(result.shape) == (1, 24, 2, 3, 4)
    assert result.dtype == mx.bfloat16
    assert report["input_channels"] == 24
    assert report["unloaded_after_upscale"] is True
    assert progress[-1] == (6, 6)


def test_hires_fix_resizes_fl2va_condition_rows_to_target_layout():
    from minimax_h3_mlx.packing import patchify_video_latents

    first = mx.ones((1, 24, 1, 4, 6), dtype=mx.bfloat16)
    last = mx.ones((1, 24, 1, 4, 6), dtype=mx.bfloat16) * 2
    rows = mx.concatenate(
        [
            patchify_video_latents(first, (1, 2, 2)),
            patchify_video_latents(last, (1, 2, 2)),
        ]
    )

    resized = resize_fl2va_condition_rows(
        rows,
        ("first", "last"),
        source_height=4,
        source_width=6,
        target_height=6,
        target_width=10,
    )

    assert tuple(resized.shape) == (30, 96)
    assert bool(mx.allclose(resized[:15], mx.ones_like(resized[:15]), atol=1e-3).item())
    assert bool(mx.allclose(resized[15:], mx.ones_like(resized[15:]) * 2, atol=1e-3).item())
