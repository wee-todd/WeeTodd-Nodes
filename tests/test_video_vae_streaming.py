from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("token_drop", "latent_frames"),
    [(0, 5), (1, 7)],
)
def test_decode_chunks_matches_full_decode(token_drop, latent_frames):
    mx = pytest.importorskip("mlx.core")

    from minimax_h3_mlx.video_vae import VideoVAE

    class SyntheticDecoder:
        tokens_chunk_size = 3
        token_overlap = (-token_drop) % tokens_chunk_size
        frame_pre_padding = 1 if token_drop else 0
        frame_overlap = max(token_overlap * 2 - frame_pre_padding, 0)
        config = SimpleNamespace(
            token_drop=token_drop,
            temporal_compression_ratio=2,
            clip_length=5 if token_drop else 6,
        )

        _blend = staticmethod(VideoVAE._blend)

        def _decode_clip(self, chunk):
            if token_drop:
                base = chunk[:, :1]
                offsets = mx.arange(12, dtype=chunk.dtype).reshape(1, 12, 1, 1, 1)
                return mx.broadcast_to(base, (1, 12, 1, 1, 1)) + offsets
            return mx.repeat(chunk, 2, axis=1)

    decoder = SyntheticDecoder()
    latents = mx.arange(latent_frames, dtype=mx.float32).reshape(1, 1, latent_frames, 1, 1)

    complete = VideoVAE.decode(decoder, latents)
    streamed = mx.concatenate(list(VideoVAE.decode_chunks(decoder, latents)), axis=2)

    assert mx.array_equal(complete, streamed)
