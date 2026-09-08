from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from minimax_h3_mlx.vdn import (
    VDNLayout,
    _gather_state_reference,
    _scan,
    _softmax_values,
    _window_bounds,
)
from minimax_h3_mlx.vdn_inference import (
    VDNInferenceKernels,
    gather_indices,
    gather_state,
    scan_addmm,
)


@pytest.mark.parametrize("frames", [1, 9, 35])
def test_scan_and_gather_preserve_reference(frames):
    mx.random.seed(7)
    t = mx.eye(8)[None, None] * 0.8 + mx.random.normal((frames, 2, 8, 8)) * 0.001
    inj = mx.random.normal(t.shape) * 0.01
    start = mx.random.normal((2, 8, 8)) * 0.01
    expected = _scan(t, inj, start)
    actual = scan_addmm(t, inj, start)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    bounds = tuple((lo - 1, hi - 1) for lo, hi in _window_bounds(frames + 2)[1:-1])
    alpha = mx.full((frames, 2, 8), 0.9)
    a = gather_state(*actual, alpha, start, *gather_indices(bounds, frames))
    b = _gather_state_reference(*actual, alpha, start, bounds)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_inference_mismatch_latches_and_reset_keeps_verdict():
    kernels = VDNInferenceKernels()
    x = mx.ones((2, 3))
    for _ in range(2):
        result = kernels.checked("bad", (x,), lambda v: v + 1, lambda v: v)
        assert mx.array_equal(result, x).item()
    assert kernels.report()["reference_calls"] == {"bad": 2}
    kernels.begin_run()
    assert kernels.report()["reference_calls"] == {}
    assert "bad" in kernels.report()["fallback_reasons"]


@pytest.mark.parametrize(
    "frames,per_frame,prefix,suffix", [(1, 7, 5, 0), (2, 9, 0, 5), (12, 7, 5, 3), (37, 63, 11, 7)]
)
def test_indexed_window_plan_and_kernel_ragged_rows(frames, per_frame, prefix, suffix):
    from minimax_h3_mlx.vdn_attention import indexed_attention, window_plan

    layout = VDNLayout(
        prefix + frames * per_frame + suffix,
        prefix,
        frames,
        per_frame,
        1,
        per_frame,
        0,
        max(prefix, 1),
    )
    qs, qn, qg, ks, kn, rs, rn = [v.tolist() for v in window_plan(layout)]
    assert [
        row for start, size in zip(qs, qn, strict=True) for row in range(start, start + size)
    ] == list(range(layout.sequence))
    for start, group in zip(qs, qg, strict=True):
        actual = [
            row
            for offset, size in zip(
                ks[rs[group] : rs[group] + rn[group]],
                kn[rs[group] : rs[group] + rn[group]],
                strict=True,
            )
            for row in range(offset, offset + size)
        ]
        if (
            start < prefix
            or start >= layout.video_end
            or (start - prefix) // per_frame in (0, frames - 1)
        ):
            expected = list(range(layout.sequence))
        else:
            frame = (start - prefix) // per_frame
            lo, hi = _window_bounds(frames)[frame]
            lo, hi = max(lo, 0), min(hi, frames - 1)
            expected = (
                list(range(prefix))
                + list(range(layout.video_end, layout.sequence))
                + list(range(prefix + lo * per_frame, prefix + (hi + 1) * per_frame))
            )
            for anchor in (0, frames - 1):
                if not lo <= anchor <= hi:
                    expected += list(
                        range(prefix + anchor * per_frame, prefix + (anchor + 1) * per_frame)
                    )
        assert actual == expected
    mx.random.seed(9)
    q, k, v = [mx.random.normal((1, 2, layout.sequence, 128)).astype(mx.bfloat16) for _ in range(3)]
    actual = indexed_attention(q, k, v, layout, 128**-0.5)
    expected = _softmax_values(SimpleNamespace(scale=128**-0.5), q, k, v, layout)
    assert mx.allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=0.002, rtol=0.01
    ).item()


def test_geometry_tile_plan_is_decode_only_and_bounded():
    from minimax_h3_mlx.video_vae import VideoVAE

    obj = SimpleNamespace(
        config=SimpleNamespace(spatial_compression_ratio=16),
        decode_tile_mode="fixed",
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_overlap_height=64,
        tile_sample_min_overlap_width=64,
    )
    obj._split_tiles = lambda *args: VideoVAE._split_tiles(obj, *args)
    VideoVAE.decode_tile_plan(obj, 384, 672)
    assert obj.last_decode_tile_plan["spatial_tiles"] == 8
    obj.decode_tile_mode = "geometry_experimental"
    yp, xp = VideoVAE.decode_tile_plan(obj, 384, 672)
    assert obj.last_decode_tile_plan["tile_width"] == 272
    assert obj.last_decode_tile_plan["spatial_tiles"] == 6
    assert obj.tile_sample_min_width == 256
    assert xp[0][-1] + xp[1][-1] == 672
    assert yp[0][-1] + yp[1][-1] == 384
    assert all(o >= 64 for o in xp[2] + yp[2])


def test_coreml_resolver_accepts_compiled_sibling_without_replacing_exact(tmp_path):
    from wee_todd_nodes.nodes import _resolve_h3_coreml_model

    requested = tmp_path / "preview.mlpackage"
    compiled = requested.with_suffix(".mlmodelc")
    compiled.mkdir()
    assert _resolve_h3_coreml_model(str(requested)) == str(compiled)
    requested.mkdir()
    assert _resolve_h3_coreml_model(str(requested)) == str(requested)


def test_expanded_backend_requires_resident_blocks():
    from minimax_h3_mlx.projection import configure_projection_backend

    with pytest.raises(ValueError, match="resident"):
        configure_projection_backend(
            SimpleNamespace(paged_blocks=object()), "mpp_resident_expanded_experimental"
        )


def test_expanded_q8_retains_packed_fallback_and_latches_mismatch(monkeypatch):
    import mlx.nn as nn

    from minimax_h3_mlx import projection

    projection.reset_mpp_runtime_status()
    original = nn.Linear(64, 64, bias=False)
    original.set_dtype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(original, group_size=64, bits=8)
    wrapped = projection.ExpandedQ8Linear(quantized)
    value = mx.ones((3, 64), dtype=mx.bfloat16)
    monkeypatch.setattr(
        projection.MPPLinear, "__call__", lambda self, x: mx.zeros((3, 64), dtype=mx.bfloat16)
    )
    expected = quantized(value)
    assert wrapped.base is quantized
    for _ in range(2):
        assert mx.array_equal(wrapped(value), expected).item()
    assert projection.mpp_runtime_status()["expanded_q8_fallback_signatures"] == 1
    projection.reset_mpp_runtime_status()
