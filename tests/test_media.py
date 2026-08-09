import os

import pytest

from minimax_h3_mlx.media import ffmpeg_status, resolve_ffmpeg


def _make_executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_resolve_ffmpeg_prefers_explicit_executable(tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg_custom")

    resolved = resolve_ffmpeg(executable, environ={"PATH": ""})

    assert resolved.path == executable.resolve()
    assert resolved.source == "node override"


def test_resolve_ffmpeg_uses_portable_environment_override(tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg_shared")

    resolved = resolve_ffmpeg(environ={"PATH": "", "WEETODD_FFMPEG": str(executable)})

    assert resolved.path == executable.resolve()
    assert resolved.source == "environment variable WEETODD_FFMPEG"


def test_environment_override_command_uses_supplied_process_path(tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg")

    resolved = resolve_ffmpeg(environ={"PATH": str(tmp_path), "WEETODD_FFMPEG": "ffmpeg"})

    assert resolved.path == executable.resolve()


def test_resolve_ffmpeg_rejects_invalid_explicit_override(tmp_path):
    with pytest.raises(RuntimeError, match="node override"):
        resolve_ffmpeg(tmp_path / "missing_ffmpeg", environ={"PATH": ""})


def test_ffmpeg_status_does_not_report_process_path(monkeypatch):
    monkeypatch.setenv("PATH", "/private/example/path")

    status = ffmpeg_status("/missing/ffmpeg")

    assert status["available"] is False
    assert "/private/example/path" not in status["error"]


def test_resolved_executable_is_marked_executable(tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg")

    resolved = resolve_ffmpeg(executable)

    assert os.access(resolved.path, os.X_OK)
