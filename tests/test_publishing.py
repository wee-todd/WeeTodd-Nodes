import json
from pathlib import Path

import numpy as np
import pytest

from wee_todd_nodes.publishing import (
    publish_synchronized_media,
    validate_synchronized_media,
)


def _media():
    video = np.zeros((124, 8, 12, 3), dtype=np.uint8)
    audio = np.zeros((2, 165600), dtype=np.float32)
    return video, audio


def _writer(path, video, fps, audio, sample_rate, crf):
    path = Path(path)
    path.write_bytes(b"synthetic mp4")
    path.with_suffix(".wav").write_bytes(b"temporary wav")
    return path


def test_validate_synchronized_h3_media_accepts_audio_grid_drift():
    video, audio = _media()

    measured = validate_synchronized_media(video, audio, 32000, 24.0, 0.025)

    assert measured["frames"] == 124
    assert measured["audio_samples"] == 165600
    assert measured["av_drift_seconds"] == pytest.approx(1 / 120)


def test_publication_writes_atomic_media_and_metadata(tmp_path):
    video, audio = _media()
    target = tmp_path / "WeeTodd" / "H3_42.mp4"

    result = publish_synchronized_media(
        target,
        video,
        audio,
        generation_metadata='{"prompt": "test"}',
        writer=_writer,
    )

    assert result.video_path.read_bytes() == b"synthetic mp4"
    assert not (target.parent / ".H3_42.partial.wav").exists()
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["prompt"] == "test"
    assert metadata["output_file"] == "H3_42.mp4"
    assert metadata["sample_rate"] == 32000


def test_publication_uses_next_available_filename(tmp_path):
    video, audio = _media()
    target = tmp_path / "H3_42.mp4"
    target.write_bytes(b"existing")

    result = publish_synchronized_media(target, video, audio, writer=_writer)

    assert target.read_bytes() == b"existing"
    assert result.video_path.name == "H3_42_00001.mp4"


def test_publication_passes_explicit_ffmpeg_to_default_writer(monkeypatch, tmp_path):
    video, audio = _media()
    calls = {}

    def save_mp4(path, video, fps, audio, sample_rate, crf, ffmpeg_path=None):
        calls["ffmpeg_path"] = ffmpeg_path
        Path(path).write_bytes(b"synthetic mp4")
        Path(path).with_suffix(".wav").write_bytes(b"temporary wav")
        return Path(path)

    monkeypatch.setattr("minimax_h3_mlx.media.save_mp4", save_mp4)

    publish_synchronized_media(
        tmp_path / "explicit.mp4",
        video,
        audio,
        ffmpeg_path="/portable/ffmpeg",
    )

    assert calls["ffmpeg_path"] == "/portable/ffmpeg"


def test_publication_rejects_timing_before_writer(tmp_path):
    video, audio = _media()
    called = False

    def writer(*args, **kwargs):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="durations differ"):
        publish_synchronized_media(tmp_path / "bad.mp4", video, audio[:, :32000], writer=writer)

    assert called is False
    assert not tmp_path.joinpath("bad.mp4").exists()


def test_publication_failure_removes_partial_files(tmp_path):
    video, audio = _media()
    target = tmp_path / "failed.mp4"

    def failing_writer(path, *args, **kwargs):
        path = Path(path)
        path.write_bytes(b"partial")
        path.with_suffix(".wav").write_bytes(b"temporary")
        raise RuntimeError("synthetic ffmpeg failure")

    with pytest.raises(RuntimeError, match="synthetic ffmpeg failure"):
        publish_synchronized_media(target, video, audio, writer=failing_writer)

    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_publication_cancellation_after_encode_removes_partial_files(tmp_path):
    video, audio = _media()
    checks = 0

    def cancel_after_encode():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("synthetic cancellation")

    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        publish_synchronized_media(
            tmp_path / "cancelled.mp4",
            video,
            audio,
            writer=_writer,
            check_interrupted=cancel_after_encode,
        )

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("sample_rate", "fps", "message"),
    [(44100, 24.0, "32000 Hz"), (32000, 30.0, "24 fps")],
)
def test_publication_rejects_non_h3_timing(sample_rate, fps, message):
    video, audio = _media()

    with pytest.raises(ValueError, match=message):
        validate_synchronized_media(video, audio, sample_rate, fps, 0.025)
