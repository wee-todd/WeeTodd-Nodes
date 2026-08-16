import json

import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_preprocessors import CannyConfig, canny_edges
from mlx_preprocessors.dwpose import DWPoseConfig, _draw_pose, _letterbox, _nms
from mlx_preprocessors.fast_depth import (
    DepthAnythingV2Small,
    FastDepthConfig,
    load_depth_anything_v2_small,
)
from mlx_preprocessors.lineart import (
    LineArtConfig,
    RealisticLineArt,
    infer_realistic_lineart,
    load_realistic_lineart,
)
from mlx_preprocessors.motion_tracks import MotionTrackConfig, render_motion_tracks
from mlx_preprocessors.normals import NormalMapConfig, depth_to_normals
from mlx_preprocessors.teed import TEED, TEEDConfig
from mlx_preprocessors.video_depth import (
    VideoDepthAnythingSmall,
    VideoDepthConfig,
    _convert_weight,
    _network_size,
    _resize,
    load_video_depth_anything_small,
)
from wee_todd_nodes.control_preprocessors import WeeToddMLXCannyPreprocessor


def _square_fixture(frames=1, height=64, width=96):
    images = np.zeros((frames, height, width, 3), dtype=np.float32)
    images[:, 16:48, 24:72] = 1.0
    return images


def test_mlx_canny_preserves_comfy_image_contract_and_finds_thin_edges():
    output, report = canny_edges(
        _square_fixture(frames=2),
        CannyConfig(low_threshold=0.1, high_threshold=0.2, frame_chunk_size=4),
    )

    assert output.shape == (2, 64, 96, 3)
    assert output.dtype == np.float32
    assert set(np.unique(output)) <= {0.0, 1.0}
    assert np.array_equal(output[..., 0], output[..., 1])
    assert np.array_equal(output[..., 1], output[..., 2])
    assert 100 < np.count_nonzero(output[0, ..., 0]) < 500
    assert report["backend"] == "mlx"
    assert report["frames"] == 2
    assert report["hysteresis_iterations"] >= 1


def test_mlx_canny_keeps_frame_alignment_for_a_moving_edge():
    images = np.zeros((3, 64, 96, 3), dtype=np.float32)
    for frame, left in enumerate((12, 24, 36)):
        images[frame, 18:46, left : left + 24] = 1.0

    output, _ = canny_edges(
        images,
        CannyConfig(low_threshold=0.1, high_threshold=0.2, frame_chunk_size=4),
    )
    centers = []
    for frame in output[..., 0]:
        _, x = np.nonzero(frame)
        centers.append(float(x.mean()))
    assert centers[0] < centers[1] < centers[2]


@pytest.mark.parametrize(
    "config, message",
    (
        (CannyConfig(low_threshold=0.8, high_threshold=0.4), "thresholds"),
        (CannyConfig(gaussian_kernel_size=9), "kernel size"),
        (CannyConfig(gaussian_sigma=0.01), "sigma"),
    ),
)
def test_mlx_canny_rejects_invalid_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        canny_edges(_square_fixture(), config)


def test_mlx_canny_node_reports_execution_contract():
    output, raw_report = WeeToddMLXCannyPreprocessor().detect(
        _square_fixture(), 0.1, 0.2, 5, 1.0, True, 16
    )
    report = json.loads(raw_report)
    assert output.shape == (1, 64, 96, 3)
    assert report["algorithm"] == "canny"
    assert report["gaussian_kernel_size"] == 5


def test_video_depth_small_architecture_matches_official_checkpoint_surface():
    parameters = dict(tree_flatten(VideoDepthAnythingSmall().parameters()))
    assert len(parameters) == 351
    assert parameters["pretrained.patch_embed.proj.weight"].shape == (384, 14, 14, 3)
    assert parameters["pretrained.blocks.11.attn.qkv.weight"].shape == (1152, 384)
    assert parameters["head.motion_modules.1.temporal_transformer.proj_out.weight"].shape == (
        384,
        384,
    )


def test_video_depth_weight_conversion_handles_conv_and_transposed_conv_layouts():
    convolution = np.arange(8 * 4 * 3 * 3, dtype=np.float32).reshape(8, 4, 3, 3)
    converted = _convert_weight("head.scratch.layer1_rn.weight", convolution, (8, 3, 3, 4))
    assert np.array_equal(converted, convolution.transpose(0, 2, 3, 1))
    transposed = np.arange(4 * 4 * 2 * 2, dtype=np.float32).reshape(4, 4, 2, 2)
    converted = _convert_weight("head.resize_layers.1.weight", transposed, (4, 2, 2, 4))
    assert np.array_equal(converted, transposed.transpose(1, 2, 3, 0))


def test_video_depth_network_size_preserves_aspect_and_patch_grid():
    assert _network_size(512, 768, 518) == (518, 784)
    height, width = _network_size(320, 1280, 518)
    assert height % 14 == width % 14 == 0
    assert width / height == pytest.approx(4.0, rel=0.05)


def test_video_depth_resize_preserves_requested_comfy_dimensions():
    import mlx.core as mx

    resized = _resize(mx.zeros((1, 392, 588, 1)), 512, 768)
    assert resized.shape == (1, 512, 768, 1)


def test_video_depth_config_and_loader_reject_unsupported_inputs(tmp_path):
    with pytest.raises(ValueError, match="input size"):
        VideoDepthConfig(input_size=500).validate()
    with pytest.raises(ValueError, match="encoder chunk size"):
        VideoDepthConfig(encoder_chunk_size=3).validate()
    with pytest.raises(ValueError, match="decoder chunk size"):
        VideoDepthConfig(decoder_chunk_size=3).validate()
    checkpoint = tmp_path / "video_depth_anything_vits.pth"
    checkpoint.touch()
    with pytest.raises(ValueError, match="converted"):
        load_video_depth_anything_small(checkpoint)


def test_dwpose_letterbox_and_nms_preserve_detector_contract():
    image = np.zeros((512, 768, 3), dtype=np.uint8)
    prepared, ratio = _letterbox(image)
    assert prepared.shape == (1, 3, 640, 640)
    assert ratio == pytest.approx(640 / 768)
    boxes = np.array(((0, 0, 100, 100), (5, 5, 95, 95), (300, 300, 400, 400)))
    keep = _nms(boxes, np.array((0.9, 0.8, 0.7)), 0.45)
    assert keep.tolist() == [0, 2]


def test_dwpose_renderer_preserves_comfy_image_shape():
    points = np.zeros((133, 2), dtype=np.float32)
    points[:17] = np.array((100, 100), dtype=np.float32)
    scores = np.zeros((133,), dtype=np.float32)
    scores[:17] = 1.0
    output = _draw_pose(512, 768, [(points, scores)], 0.3, "body only")
    assert output.shape == (512, 768, 3)
    assert output.dtype == np.float32
    assert output.max() <= 1.0


def test_dwpose_config_rejects_invalid_render_mode():
    with pytest.raises(ValueError, match="render mode"):
        DWPoseConfig(render_mode="face only").validate()


def test_teed_architecture_matches_official_checkpoint_surface():
    parameters = dict(tree_flatten(TEED().parameters()))
    assert len(parameters) == 36
    assert parameters["block_1.conv1.weight"].shape == (16, 3, 3, 3)
    assert parameters["block_cat.DWconv1.weight"].shape == (24, 3, 3, 1)


def test_teed_config_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk size"):
        TEEDConfig(frame_chunk_size=3).validate()


def test_fast_depth_architecture_matches_official_small_checkpoint_surface():
    parameters = dict(tree_flatten(DepthAnythingV2Small().parameters()))
    # MLX fuses each layer's separate query/key/value tensors into one QKV projection.
    assert len(parameters) == 239
    assert parameters["backbone.patch_embed.proj.weight"].shape == (384, 14, 14, 3)
    assert parameters["resize_layers.0.weight"].shape == (48, 4, 4, 48)
    assert parameters["head_conv3.weight"].shape == (1, 1, 1, 32)


def test_fast_depth_loader_rejects_missing_source_mapping(tmp_path):
    from safetensors.numpy import save_file

    checkpoint = tmp_path / "model.safetensors"
    save_file({"unrelated": np.zeros((1,), dtype=np.float32)}, checkpoint)
    with pytest.raises((KeyError, ValueError)):
        load_depth_anything_v2_small(checkpoint)


def test_fast_depth_config_rejects_invalid_normalization():
    with pytest.raises(ValueError, match="normalization"):
        FastDepthConfig(normalize="global maybe").validate()


def test_depth_to_normals_preserves_contract_and_flat_forward_direction():
    depth = np.full((2, 48, 64, 3), 0.5, dtype=np.float32)
    output, report = depth_to_normals(depth, NormalMapConfig(frame_chunk_size=1))
    assert output.shape == depth.shape
    assert output.dtype == np.float32
    assert np.allclose(output[..., 0], 0.5)
    assert np.allclose(output[..., 1], 0.5)
    assert np.allclose(output[..., 2], 1.0)
    assert report["algorithm"] == "depth_derived_normals"


def test_depth_to_normals_encodes_horizontal_ramp_direction():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, None, :, None], (1, 32, 64, 3)).copy()
    output, _ = depth_to_normals(depth, NormalMapConfig(method="central", strength=4.0))
    assert float(output[0, 16, 32, 0]) < 0.5
    assert float(output[0, 16, 32, 1]) == pytest.approx(0.5)
    assert float(output[0, 16, 32, 2]) < 1.0


def test_depth_to_normals_default_scale_exposes_normalized_depth_geometry():
    ramp = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, None, :, None], (1, 32, 64, 3)).copy()
    output, report = depth_to_normals(depth, NormalMapConfig(method="central"))
    center = output[0, 16, 32]
    # A normalized one-pixel slope must remain visibly directional at the default,
    # rather than collapsing into an almost-flat (0.5, 0.5, 1.0) guide.
    assert float(center[0]) < 0.3
    assert float(center[1]) == pytest.approx(0.5)
    assert 0.85 < float(center[2]) < 0.95
    assert report["strength"] == 40.0


def test_depth_to_normals_flip_y_only_changes_green_orientation():
    ramp = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    depth = np.broadcast_to(ramp[None, :, None, None], (1, 32, 48, 3)).copy()
    standard, _ = depth_to_normals(depth, NormalMapConfig(method="central", flip_y=False))
    flipped, _ = depth_to_normals(depth, NormalMapConfig(method="central", flip_y=True))
    standard_center = standard[0, 16, 24]
    flipped_center = flipped[0, 16, 24]
    assert float(standard_center[1]) < 0.5 < float(flipped_center[1])
    assert float(standard_center[0]) == pytest.approx(float(flipped_center[0]))
    assert float(standard_center[2]) == pytest.approx(float(flipped_center[2]))


def test_motion_track_guide_renders_training_colors_and_direction():
    tracks = '[[{"x":0.25,"y":0.5},{"x":0.75,"y":0.5}]]'
    output, report = render_motion_tracks(
        tracks,
        MotionTrackConfig(width=64, height=64, num_frames=51, trail_frames=50),
    )
    assert output.shape == (51, 64, 64, 3)
    assert output.dtype == np.float32
    assert report["backend"] == "mlx_sparse_raster"
    assert report["channel_order"] == "training_bgr"
    # The current marker is blue after the training-format RGB->BGR channel swap.
    assert output[-1, 32, 47, 2] > 0.9
    # The oldest retained marker is red and remains at the start of the trail.
    assert output[-1, 32, 16, 0] > 0.9


def test_motion_track_guide_supports_static_and_per_frame_tracks():
    points = [[{"x": 20, "y": 24}] for _ in range(1)]
    output, report = render_motion_tracks(
        json.dumps(points),
        MotionTrackConfig(
            width=64,
            height=64,
            num_frames=4,
            coordinate_space="pixels",
            track_format="per-frame coordinates",
            trail_frames=2,
        ),
    )
    assert output.shape == (4, 64, 64, 3)
    assert np.count_nonzero(output[-1]) > 0
    assert report["tracks"] == 1


def test_motion_track_guide_rejects_invalid_tracks():
    with pytest.raises(ValueError, match="at least one track"):
        render_motion_tracks("[]", MotionTrackConfig(width=64, height=64, num_frames=4))
    with pytest.raises(ValueError, match="between 0 and 1"):
        render_motion_tracks(
            '[[{"x":1.5,"y":0.5}]]',
            MotionTrackConfig(width=64, height=64, num_frames=4),
        )


def test_lineart_architecture_and_config_match_converted_contract():
    parameters = dict(tree_flatten(RealisticLineArt().parameters()))
    assert len(parameters) == 24
    assert parameters["input_conv.weight"].shape == (64, 7, 7, 3)
    assert parameters["up_convs.0.weight"].shape == (128, 3, 3, 256)
    assert parameters["output_conv.weight"].shape == (1, 7, 7, 64)
    with pytest.raises(ValueError, match="resolution"):
        LineArtConfig(detect_resolution=500).validate()


def test_lineart_loader_rejects_incomplete_checkpoint(tmp_path):
    from safetensors.numpy import save_file

    checkpoint = tmp_path / "lineart.safetensors"
    save_file({"input_conv.bias": np.zeros((64,), dtype=np.float32)}, checkpoint)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        load_realistic_lineart(checkpoint)


def test_lineart_output_mode_names_match_pixel_polarity():
    import mlx.core as mx

    class ConstantModel:
        def __call__(self, value):
            return mx.full((*value.shape[:3], 1), 0.8)

    images = np.zeros((1, 32, 48, 3), dtype=np.float32)
    white, _ = infer_realistic_lineart(
        images,
        ConstantModel(),
        LineArtConfig(detect_resolution=256, output_mode="white lines"),
    )
    black, _ = infer_realistic_lineart(
        images,
        ConstantModel(),
        LineArtConfig(detect_resolution=256, output_mode="black lines"),
    )
    assert white.mean() == pytest.approx(0.2, abs=1e-5)
    assert black.mean() == pytest.approx(0.8, abs=1e-5)
