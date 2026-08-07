import pytest

from minimax_h3_mlx.algorithm_search.modalities import t2va_modality_rows


def test_smoke_t2va_modality_rows_match_captured_sequence():
    rows = t2va_modality_rows(
        total_rows=9477,
        text_rows=183,
        duration_seconds=5.0,
        height=384,
        width=640,
    )
    assert rows.counts == {"text": 183, "audio": 414, "video": 8880}
    assert rows.text == slice(0, 183)
    assert rows.audio == slice(183, 597)
    assert rows.video == slice(597, 9477)


def test_modality_rows_reject_mismatched_capture():
    with pytest.raises(ValueError, match="packed row count mismatch"):
        t2va_modality_rows(
            total_rows=9476,
            text_rows=183,
            duration_seconds=5.0,
            height=384,
            width=640,
        )
