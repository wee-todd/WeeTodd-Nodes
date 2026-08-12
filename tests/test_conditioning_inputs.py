from types import SimpleNamespace

import pytest

from wee_todd_nodes.conditioning_inputs import (
    H3KeyframeConditioning,
    H3ReferenceInput,
    H3ReferenceStack,
    H3TimedKeyframe,
    H3TimedKeyframeStack,
    resolve_reference_image_canvas,
)


def _tensor(*shape):
    return SimpleNamespace(shape=shape)


def _audio(channels=2, samples=32000, sample_rate=32000):
    return {
        "waveform": _tensor(1, channels, samples),
        "sample_rate": sample_rate,
    }


def test_keyframe_contract_supports_first_last_and_both():
    image = _tensor(1, 384, 640, 3)
    first = H3KeyframeConditioning(first_frame=image)
    last = H3KeyframeConditioning(last_frame=image)
    both = H3KeyframeConditioning(first_frame=image, last_frame=image)

    assert first.anchors == ("first",)
    assert last.anchors == ("last",)
    assert both.anchors == ("first", "last")
    assert both.metadata()["task"] == "fl2va"


def test_keyframe_contract_rejects_empty_or_invalid_image():
    with pytest.raises(ValueError, match="requires"):
        H3KeyframeConditioning().validate()
    with pytest.raises(ValueError, match="ComfyUI IMAGE shape"):
        H3KeyframeConditioning(first_frame=_tensor(384, 640, 3)).validate()


def test_timed_keyframes_sort_and_resolve_to_exact_24fps_frames():
    import numpy as np

    image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    stack = H3TimedKeyframeStack().append(H3TimedKeyframe(image, 2.0))
    stack = stack.append(H3TimedKeyframe(image, 0.5, 4.0416666667))

    anchors, images = stack.resolve(107)

    assert anchors == (12, 48)
    assert len(images) == 2
    assert stack.metadata(107)["rope_times"] == [20.0, 80.0]


def test_timed_keyframes_reject_duplicate_or_out_of_window_frames():
    import numpy as np

    image = np.zeros((1, 32, 32, 3), dtype=np.float32)
    duplicate = H3TimedKeyframeStack(
        (H3TimedKeyframe(image, 1.0), H3TimedKeyframe(image, 1.01))
    )
    with pytest.raises(ValueError, match="same 24 fps frame"):
        duplicate.resolve(107)
    with pytest.raises(ValueError, match="outside"):
        H3TimedKeyframeStack((H3TimedKeyframe(image, 5.0),)).resolve(107)


def test_reference_stack_preserves_semantic_order_and_prompt_labels():
    stack = H3ReferenceStack()
    stack = stack.append(H3ReferenceInput("audio", _audio()))
    stack = stack.append(H3ReferenceInput("image", _tensor(1, 384, 640, 3)))
    stack = stack.append(
        H3ReferenceInput(
            "video",
            _tensor(48, 384, 640, 3),
            fps=24.0,
            soundtrack=_audio(),
        )
    )
    stack.validate_request()

    metadata = stack.metadata()
    assert [item["kind"] for item in metadata["references"]] == ["audio", "image", "video"]
    assert [item["prompt_labels"] for item in metadata["references"]] == [
        ["<Audio 1>"],
        ["<Picture 1>"],
        ["<Audio 2>", "<Video 1>"],
    ]


def test_reference_stack_rejects_audio_only_at_request_boundary():
    stack = H3ReferenceStack().append(H3ReferenceInput("audio", _audio()))
    with pytest.raises(ValueError, match="at least one image or video"):
        stack.validate_request()


def test_reference_video_validates_frames_fps_and_soundtrack():
    with pytest.raises(ValueError, match="at least five frames"):
        H3ReferenceInput("video", _tensor(4, 384, 640, 3), fps=24.0).validate()
    with pytest.raises(ValueError, match="fps must be positive"):
        H3ReferenceInput("video", _tensor(5, 384, 640, 3), fps=0.0).validate()
    with pytest.raises(ValueError, match="mono-or-stereo"):
        H3ReferenceInput(
            "video",
            _tensor(5, 384, 640, 3),
            fps=24.0,
            soundtrack=_audio(channels=3),
        ).validate()


def test_reference_image_pixel_budget_scales_area_and_preserves_aspect():
    assert resolve_reference_image_canvas(1600, 900, 640, 384, 100) == (672, 384)
    assert resolve_reference_image_canvas(1600, 900, 640, 384, 50) == (480, 256)
    assert resolve_reference_image_canvas(1600, 900, 640, 384, 400) == (1312, 736)


def test_reference_image_pixel_budget_is_bounded():
    image = _tensor(1, 900, 1600, 3)
    with pytest.raises(ValueError, match="between 50% and 400%"):
        H3ReferenceInput("image", image, image_pixel_budget_percent=49).validate()
    with pytest.raises(ValueError, match="between 50% and 400%"):
        resolve_reference_image_canvas(1600, 900, 640, 384, 401)
