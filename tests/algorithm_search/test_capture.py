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
