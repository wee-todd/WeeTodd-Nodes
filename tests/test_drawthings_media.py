import json
import os
import struct
import subprocess
import time
from pathlib import Path

import pytest
from PIL import Image

from wee_todd_remote.media import finish_video, validate_media


def _png(path: Path, size=(16, 12), color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _float32_wav(path: Path, *, rate: int, samples: int, channels: int = 1) -> None:
    values = b"".join(struct.pack("<f", 0.125) for _ in range(samples * channels))
    fmt = struct.pack("<HHIIHH", 3, channels, rate, rate * channels * 4, channels * 4, 32)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(values)) + values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body) + 4) + b"WAVE" + body)


def _video_manifest(frame_count=3, *, audio=True):
    value = {
        "schema": "weetodd-drawthings-media-v1",
        "requestID": "request-1",
        "operation": "video",
        "frameCount": frame_count,
        "configuration": {"width": 16, "height": 12},
        "framesDirectory": "frames",
        "fpsNumerator": 2,
        "fpsDenominator": 1,
        "requiresAudio": audio,
    }
    if audio:
        value |= {"audioPath": "audio.wav", "sampleRate": 8000, "sampleCount": 12000, "channels": 1}
    return value


def _write_frames(root: Path, count=3, *, size=(16, 12)) -> None:
    for index in range(count):
        _png(root / "frames" / f"{index:08d}.png", size=size, color=(index * 30, 0, 0))


def _fake_ffmpeg(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return path


def _finish_manifest(root: Path) -> dict:
    (root / "frames").mkdir(exist_ok=True)
    return {
        "operation": "video", "framesDirectory": str((root / "frames").resolve()),
        "fpsNumerator": 24, "fpsDenominator": 1, "frameCount": 2,
        "requiresAudio": False,
    }


def test_image_manifest_measures_and_normalizes_artifact(tmp_path):
    _png(tmp_path / "frames" / "00000000.png")
    manifest = {
        "schema": "weetodd-drawthings-media-v1",
        "requestID": "request-1",
        "operation": "image",
        "frameCount": 1,
        "configuration": {"width": 16, "height": 12},
        "imagePaths": ["frames/00000000.png"],
    }
    result = validate_media(
        manifest,
        root=tmp_path,
        expected_request_id="request-1",
        expected_operation="image",
        expected_frames=1,
    )
    assert result["imagePaths"] == [str((tmp_path / "frames/00000000.png").resolve())]
    assert (result["width"], result["height"]) == (16, 12)


@pytest.mark.parametrize("relative", ["../outside.png", "/tmp/outside.png"])
def test_image_manifest_rejects_paths_outside_request_root(tmp_path, relative):
    manifest = {
        "schema": "weetodd-drawthings-media-v1",
        "requestID": "request-1",
        "operation": "image",
        "frameCount": 1,
        "configuration": {},
        "imagePaths": [relative],
    }
    with pytest.raises(ValueError, match="confined"):
        validate_media(
            manifest, root=tmp_path, expected_request_id="request-1",
            expected_operation="image", expected_frames=1,
        )


def test_image_manifest_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _png(outside)
    (tmp_path / "escape.png").symlink_to(outside)
    manifest = {
        "schema": "weetodd-drawthings-media-v1", "requestID": "request-1",
        "operation": "image", "frameCount": 1, "configuration": {},
        "imagePaths": ["escape.png"],
    }
    try:
        with pytest.raises(ValueError, match="confined"):
            validate_media(
                manifest, root=tmp_path, expected_request_id="request-1",
                expected_operation="image", expected_frames=1,
            )
    finally:
        outside.unlink()


def test_video_manifest_rejects_nonsequential_or_mismatched_frames(tmp_path):
    _png(tmp_path / "frames" / "00000000.png")
    _png(tmp_path / "frames" / "00000002.png", size=(20, 12))
    with pytest.raises(ValueError, match="sequential"):
        validate_media(
            _video_manifest(2, audio=False), root=tmp_path,
            expected_request_id="request-1", expected_operation="video", expected_frames=2,
        )


def test_video_manifest_rejects_frame_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    _png(outside)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/00000000.png").symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="confined"):
            validate_media(
                _video_manifest(1, audio=False), root=tmp_path,
                expected_request_id="request-1", expected_operation="video", expected_frames=1,
            )
    finally:
        outside.unlink()


def test_video_manifest_rejects_mixed_frame_dimensions(tmp_path):
    _write_frames(tmp_path, count=2)
    _png(tmp_path / "frames/00000001.png", size=(20, 12))
    with pytest.raises(ValueError, match="dimensions"):
        validate_media(
            _video_manifest(2, audio=False), root=tmp_path,
            expected_request_id="request-1", expected_operation="video", expected_frames=2,
        )


def test_video_manifest_measures_float_audio_and_timing(tmp_path):
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=12000)
    result = validate_media(
        _video_manifest(), root=tmp_path, expected_request_id="request-1",
        expected_operation="video", expected_frames=3, requires_audio=True,
    )
    assert result["frameCount"] == 3
    assert result["sampleCount"] == 12000
    assert result["audioPath"] == str((tmp_path / "audio.wav").resolve())
    assert (result["width"], result["height"]) == (16, 12)


def test_caller_audio_requirement_is_preserved_in_normalized_manifest(tmp_path):
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=12000)
    result = validate_media(
        _video_manifest() | {"requiresAudio": False}, root=tmp_path,
        expected_request_id="request-1", expected_operation="video", expected_frames=3,
        requires_audio=True,
    )
    assert result["requiresAudio"] is True


def test_required_audio_cannot_fall_back_to_silence(tmp_path):
    _write_frames(tmp_path)
    manifest = _video_manifest(audio=False) | {"requiresAudio": True}
    with pytest.raises(ValueError, match="audio"):
        validate_media(
            manifest, root=tmp_path, expected_request_id="request-1",
            expected_operation="video", expected_frames=3, requires_audio=True,
        )


def test_audio_duration_must_match_video_within_one_frame(tmp_path):
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=4000)
    manifest = _video_manifest() | {"sampleCount": 4000}
    with pytest.raises(ValueError, match="duration"):
        validate_media(
            manifest, root=tmp_path, expected_request_id="request-1",
            expected_operation="video", expected_frames=3, requires_audio=True,
        )


def test_audio_must_be_ieee_float32_wav(tmp_path):
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=12000)
    data = bytearray((tmp_path / "audio.wav").read_bytes())
    struct.pack_into("<H", data, 20, 1)
    (tmp_path / "audio.wav").write_bytes(data)
    with pytest.raises(ValueError, match="Float32"):
        validate_media(
            _video_manifest(), root=tmp_path, expected_request_id="request-1",
            expected_operation="video", expected_frames=3, requires_audio=True,
        )


def test_audio_rejects_nonfinite_float_samples(tmp_path):
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=12000)
    with (tmp_path / "audio.wav").open("r+b") as stream:
        stream.seek(44)
        stream.write(struct.pack("<f", float("nan")))
    with pytest.raises(ValueError, match="finite"):
        validate_media(
            _video_manifest(), root=tmp_path, expected_request_id="request-1",
            expected_operation="video", expected_frames=3, requires_audio=True,
        )


def test_finish_video_muxes_exact_frame_rate_and_audio(tmp_path):
    ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
    ffprobe = Path("/opt/homebrew/bin/ffprobe")
    if not ffmpeg.is_file() or not ffprobe.is_file():
        pytest.skip("FFmpeg tools unavailable")
    _write_frames(tmp_path)
    _float32_wav(tmp_path / "audio.wav", rate=8000, samples=12000)
    manifest = validate_media(
        _video_manifest(), root=tmp_path, expected_request_id="request-1",
        expected_operation="video", expected_frames=3, requires_audio=True,
    )
    output = finish_video(manifest, output=tmp_path / "finished.mov", ffmpeg=ffmpeg)
    probe = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(output)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert video["r_frame_rate"] == "2/1"
    assert int(video["nb_frames"]) == 3
    assert any(stream["codec_type"] == "audio" for stream in streams)


@pytest.mark.parametrize("callback_raises", [False, True])
def test_finish_video_cancels_process_group_and_removes_partial(tmp_path, callback_raises):
    fake = _fake_ffmpeg(
        tmp_path / "fake-ffmpeg",
        """import pathlib, subprocess, sys, time
frames = pathlib.Path(next(value for value in sys.argv if '%08d.png' in value)).parent
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])
(frames / 'child.pid').write_text(str(child.pid))
pathlib.Path(sys.argv[-1]).write_bytes(b'partial')
time.sleep(3)
""",
    )
    def cancelled():
        ready = (tmp_path / "frames/child.pid").exists()
        if ready and callback_raises:
            raise KeyboardInterrupt
        return ready

    started = time.monotonic()
    expected_error = KeyboardInterrupt if callback_raises else InterruptedError
    with pytest.raises(expected_error):
        finish_video(
            _finish_manifest(tmp_path), tmp_path / "finished.mov", fake,
            cancelled=cancelled,
        )
    assert time.monotonic() - started < 1.5
    child_pid = int((tmp_path / "frames/child.pid").read_text())
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.025)
    else:
        pytest.fail("FFmpeg descendant remained alive after cancellation")
    assert not (tmp_path / "finished.mov").exists()
    assert not list(tmp_path.glob(".*.partial*"))


def test_ffmpeg_failure_preserves_raced_destination_and_cleans_partial(tmp_path):
    fake = _fake_ffmpeg(
        tmp_path / "fake-ffmpeg",
        """import pathlib, sys
frames = pathlib.Path(next(value for value in sys.argv if '%08d.png' in value)).parent
(frames.parent / 'finished.mov').write_bytes(b'existing')
pathlib.Path(sys.argv[-1]).write_bytes(b'partial')
raise SystemExit(7)
""",
    )
    output = tmp_path / "finished.mov"
    with pytest.raises(RuntimeError, match="status 7"):
        finish_video(_finish_manifest(tmp_path), output, fake)
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".*.partial*"))


def test_finish_video_refuses_existing_destination_without_overwrite(tmp_path):
    fake = _fake_ffmpeg(
        tmp_path / "fake-ffmpeg",
        "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes(b'new')\n",
    )
    output = tmp_path / "finished.mov"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        finish_video(_finish_manifest(tmp_path), output, fake)
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".*.partial*"))


def test_publication_failure_preserves_destination_and_cleans_partial(tmp_path, monkeypatch):
    fake = _fake_ffmpeg(
        tmp_path / "fake-ffmpeg",
        "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_bytes(b'complete')\n",
    )
    output = tmp_path / "finished.mov"

    def fail_link(source, destination):
        output.write_bytes(b"raced-existing")
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(FileExistsError):
        finish_video(_finish_manifest(tmp_path), output, fake)
    assert output.read_bytes() == b"raced-existing"
    assert not list(tmp_path.glob(".*.partial*"))
