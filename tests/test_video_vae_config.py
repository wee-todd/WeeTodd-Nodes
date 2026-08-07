from __future__ import annotations

from minimax_h3_mlx.video_vae import DEFAULT_DECODE_BATCH, resolved_decode_batch


def test_default_decode_batch_uses_measured_memory_efficient_policy(monkeypatch) -> None:
    monkeypatch.delenv("H3_VAE_BATCH", raising=False)

    assert DEFAULT_DECODE_BATCH == 4
    assert resolved_decode_batch() == 4


def test_decode_batch_environment_override_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("H3_VAE_BATCH", "8")

    assert resolved_decode_batch() == 8
