import json

import mlx.core as mx
import pytest

from minimax_h3_mlx.algorithm_search.capture import CaptureConfig, DiagnosticSession


def test_disabled_capture_does_not_write(tmp_path):
    session = DiagnosticSession(CaptureConfig(output_directory=str(tmp_path)))
    value = session.measure("test", lambda: mx.ones((2, 2)), capture_as="block_input")
    mx.eval(value)
    assert list(tmp_path.iterdir()) == []
    assert session.captures == []
    assert session.measurements == []


def test_capture_is_selected_bounded_and_metadata_is_separate(tmp_path):
    session = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("block_input",),
            blocks=(3,),
            profile_blocks=(3,),
            max_total_bytes=64,
            profile_regions=True,
        )
    )
    session.measure(
        "blocks.3.input",
        lambda: mx.ones((2, 2), dtype=mx.float32),
        block=3,
        capture_as="block_input",
    )
    session.capture("block_input", mx.ones((20,), dtype=mx.float32), block=2)
    metadata = session.write_metadata()
    payload = json.loads(metadata.read_text())
    assert len(payload["captures"]) == 1
    assert payload["captures"][0]["block"] == 3
    assert (tmp_path / payload["captures"][0]["path"]).is_file()
    assert len(payload["measurements"]) == 1

    session.measure("blocks.2.input", lambda: mx.ones((1,)), block=2)
    assert len(session.measurements) == 1
    session.prepare_block(mx.ones((1,)), 3)

    with pytest.raises(MemoryError):
        session.capture("block_input", mx.ones((20,), dtype=mx.float32), block=3)


def test_capture_selects_and_labels_denoising_evaluations(tmp_path):
    session = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("block_input",),
            evaluation_indices=(1,),
        )
    )
    session.begin_evaluation(0, timestep=0.0, audio_timestep=0.0)
    session.capture("block_input", mx.ones((2,)), block=4)
    session.begin_evaluation(1, timestep=0.75, audio_timestep=0.5)
    session.capture("block_input", mx.ones((2,)), block=4)
    payload = json.loads(session.write_metadata().read_text())
    assert len(payload["captures"]) == 1
    assert payload["captures"][0]["evaluation_index"] == 1
    assert payload["captures"][0]["timestep"] == 0.75
    assert payload["captures"][0]["audio_timestep"] == 0.5
    assert "eval_1" in payload["captures"][0]["path"]


def test_capture_rejects_negative_evaluation_indices():
    with pytest.raises(ValueError, match="evaluation indices"):
        DiagnosticSession(CaptureConfig(evaluation_indices=(-1,)))


def test_attention_capture_requires_bounded_head_selection(tmp_path):
    with pytest.raises(ValueError, match="selected head"):
        DiagnosticSession(
            CaptureConfig(
                enabled=True,
                output_directory=str(tmp_path),
                targets=("attention_qkv",),
            )
        )


def test_block_filter_does_not_reject_global_capture(tmp_path):
    session = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("video_output",),
            blocks=(24,),
        )
    )
    session.capture("video_output", mx.ones((2,)), block=None)
    assert len(session.captures) == 1
    assert session.captures[0]["block"] is None


def test_attention_capture_records_exact_prefix_and_selected_heads(tmp_path):
    session = DiagnosticSession(
        CaptureConfig(
            enabled=True,
            output_directory=str(tmp_path),
            targets=("attention_qkv",),
            blocks=(2,),
            attention_heads=(1,),
        )
    )
    session.set_packed_layout(
        sequence_rows=10,
        text_indices=mx.array([0, 1]),
        audio_indices=mx.array([4, 5]),
        video_indices=mx.array([2, 3, 6, 7, 8, 9]),
    )
    q = mx.ones((1, 3, 10, 4), dtype=mx.bfloat16)
    session.capture_attention_qkv(q, q, q, block=2)
    payload = json.loads(session.write_metadata().read_text())
    capture = payload["captures"][0]
    assert capture["shapes"] == [[1, 1, 10, 4]] * 3
    assert capture["metadata"]["prefix_rows"] == 6
    assert capture["metadata"]["target_video_rows"] == 4
    assert capture["metadata"]["condition_video_rows"] == 2
    assert capture["metadata"]["attention_heads"] == [1]


def test_attention_capture_rejects_non_contiguous_target_video_tail(tmp_path):
    session = DiagnosticSession(CaptureConfig(output_directory=str(tmp_path)))
    with pytest.raises(ValueError, match="contiguous packed tail"):
        session.set_packed_layout(
            sequence_rows=8,
            text_indices=mx.array([0]),
            audio_indices=mx.array([2, 3]),
            video_indices=mx.array([1, 4, 6, 7]),
        )
