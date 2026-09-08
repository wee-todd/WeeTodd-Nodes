from __future__ import annotations

import struct
import wave

import numpy as np
import pytest

from ltx25_mlx.chaining import _save_waveform
from wee_todd_mlx.media_serialization import write_all_contiguous


class _ShortWriter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.calls = 0
        self.data = bytearray()
        self.received_bytes_object = False

    def write(self, value: memoryview) -> int:
        self.calls += 1
        self.received_bytes_object |= isinstance(value, bytes)
        count = min(self.limit, len(value))
        self.data.extend(value[:count])
        return count


def test_write_all_contiguous_handles_short_writes_without_bytes_copy() -> None:
    source = np.arange(17, dtype=np.uint8)
    writer = _ShortWriter(limit=3)

    write_all_contiguous(source, writer)

    assert writer.data == bytearray(range(17))
    assert writer.calls > 1
    assert writer.received_bytes_object is False


@pytest.mark.parametrize("limit", [0, -1])
def test_write_all_contiguous_rejects_stalled_stream(limit: int) -> None:
    with pytest.raises(OSError, match="cannot make progress"):
        write_all_contiguous(np.arange(4, dtype=np.uint8), _ShortWriter(limit=limit))


def test_write_all_contiguous_rejects_noncontiguous_input() -> None:
    source = np.arange(24, dtype=np.uint8).reshape(4, 6).T
    assert source.flags.c_contiguous is False

    with pytest.raises(TypeError, match="C-contiguous"):
        write_all_contiguous(source, _ShortWriter(limit=100))


def test_chained_waveform_writer_preserves_reference_pcm_bytes(tmp_path) -> None:
    waveform = np.array([[1.0, 0.5], [-1.0, -0.5]], dtype=np.float32)
    path = tmp_path / "stereo.wav"

    _save_waveform(path, waveform, 24000)

    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 2
        assert stream.getframerate() == 24000
        payload = stream.readframes(stream.getnframes())

    assert payload == struct.pack("<4h", 32767, -32767, 16383, -16383)
