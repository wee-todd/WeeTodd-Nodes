"""Parity of the MLX packing geometry and scheduler against the diffusers reference.

The rotary grid is the audio/video alignment, so it is checked bit-exactly in float64 before the
float32 cast the transformer applies; the scheduler is checked over a full denoising trajectory.

    ./.venv/bin/python tests/test_packing_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusers.modular_pipelines.minimax_h3 import packing as ref_packing
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler as RefScheduler

from minimax_h3_mlx import packing as mine
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_canvas_and_frames() -> None:
    for aw, ah in [(16, 9), (9, 16), (1, 1), (4, 1), (1, 4), (21, 9), (3, 2)]:
        got = mine.resolve_canvas_size(aw, ah)
        want = ref_packing.resolve_canvas_size(aw, ah)
        check(f"canvas {aw}:{ah}", got == want, f"{got} vs {want}")

    for n in [1, 5, 6, 22, 23, 100, 360, 361]:
        got, want = mine.align_num_frames(n), ref_packing.align_num_frames(n)
        check(f"align_num_frames({n})", got == want, f"{got} vs {want}")

    for n in [5, 22, 39, 90, 361]:
        aligned = ref_packing.align_num_frames(n)
        got = mine.video_latent_num_frames(aligned)
        want = ref_packing.video_latent_num_frames(aligned)
        check(f"video_latent_num_frames({aligned})", got == want, f"{got} vs {want}")
        got_a = mine.audio_latent_num_frames(aligned)
        want_a = ref_packing.audio_latent_num_frames(aligned)
        check(f"audio_latent_num_frames({aligned})", got_a == want_a, f"{got_a} vs {want_a}")


def test_packed_sequence() -> None:
    """The (t, h, w) grid must match the reference exactly in float64."""
    cases = [
        # (num_text, num_latent_frames, latent_h, latent_w, num_audio, anchors)
        (7, 12, 24, 42, 20, ()),
        (11, 17, 48, 24, 28, ("first",)),
        (5, 22, 24, 24, 36, ("first", "last")),
    ]
    patch = (1, 2, 2)
    for num_text, frames, lh, lw, n_audio, anchors in cases:
        tags = np.full(num_text, mine.TAG_TEXT, dtype=np.int64)
        tags[: max(0, num_text // 3)] = mine.TAG_VIDEO  # a keyframe vision block

        got = mine.build_packed_sequence(tags, frames, lh, lw, n_audio, patch, anchors)
        want = ref_packing.build_packed_sequence(
            torch.from_numpy(tags), frames, lh, lw, n_audio, patch, anchors
        )

        label = f"packed[{num_text}t,{frames}f,{lh}x{lw},{n_audio}a,{anchors}]"
        check(f"{label} seq_len", got.sequence_length == want.sequence_length)

        # float64 grid, pre-cast: the reference's own float32 view is the fair comparison.
        want_pos = want.position_ids.numpy().astype(np.float32)
        got_pos = np.array(got.position_ids)
        check(f"{label} position_ids", np.array_equal(got_pos, want_pos),
              f"max delta {np.abs(got_pos - want_pos).max():.3e}")

        for field in ("token_tags", "video_indices", "audio_indices", "text_indices"):
            g = np.array(getattr(got, field)).astype(np.int64)
            w = getattr(want, field).numpy().astype(np.int64)
            check(f"{label} {field}", np.array_equal(g, w))

        check(f"{label} n_cond_video", got.num_condition_video_rows == want.num_condition_video_rows)

        # Row timesteps over the same layout.
        g_ts, g_idx = mine.build_row_timesteps(got, 0.4, 0.6, mine.KEYFRAME_NOISE_AUG, 1.0)
        w_ts, w_idx = ref_packing.build_row_timesteps(want, 0.4, 0.6, mine.KEYFRAME_NOISE_AUG, 1.0)
        check(f"{label} distinct timesteps", np.array_equal(np.array(g_ts), w_ts.numpy()))
        check(f"{label} timestep_indices",
              np.array_equal(np.array(g_idx).astype(np.int64), w_idx.numpy().astype(np.int64)))


def test_patchify_roundtrip() -> None:
    patch = (1, 2, 2)
    c, f, h, w = 4, 6, 8, 10
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((1, c, f, h, w)).astype(np.float32)

    got = np.array(mine.patchify_video_latents(mx.array(latents), patch))
    want = ref_packing.patchify_video_latents(torch.from_numpy(latents), patch).numpy()
    check("patchify_video_latents", np.array_equal(got, want))

    back = np.array(
        mine.unpatchify_video_tokens(mx.array(got), f, h, w, c, patch)
    )
    check("unpatchify roundtrip", np.allclose(back, latents, atol=0, rtol=0))

    rows = rng.standard_normal((2 * 15, 32)).astype(np.float32)
    got_a = np.array(mine.unpack_audio_tokens(mx.array(rows), 15))
    want_a = ref_packing.unpack_audio_tokens(torch.from_numpy(rows), 15).numpy()
    check("unpack_audio_tokens", np.array_equal(got_a, want_a))


def test_scheduler() -> None:
    for shift in (12.0, 3.0, 1.0):
        for steps in (2, 8, 32, 50):
            mine_s = MiniMaxH3Scheduler(shift=shift)
            mine_s.set_timesteps(steps)
            ref_s = RefScheduler(shift=shift)
            ref_s.set_timesteps(steps)

            label = f"scheduler[shift={shift},steps={steps}]"
            g_sig, w_sig = np.array(mine_s.sigmas), ref_s.sigmas.numpy()
            check(f"{label} sigmas", np.array_equal(g_sig, w_sig),
                  f"max delta {np.abs(g_sig - w_sig).max():.3e}" if g_sig.shape == w_sig.shape
                  else f"shape {g_sig.shape} vs {w_sig.shape}")
            g_t, w_t = np.array(mine_s.timesteps), ref_s.timesteps.numpy()
            check(f"{label} timesteps", np.array_equal(g_t, w_t))

    # A full trajectory: same velocities in, same samples out at every step.
    rng = np.random.default_rng(0)
    mine_s, ref_s = MiniMaxH3Scheduler(shift=12.0), RefScheduler(shift=12.0)
    mine_s.set_timesteps(16)
    ref_s.set_timesteps(16)
    sample = rng.standard_normal((1, 8, 16)).astype(np.float32)
    m_sample, r_sample = mx.array(sample), torch.from_numpy(sample)

    worst = 0.0
    for t in ref_s.timesteps.tolist():
        v = rng.standard_normal((1, 8, 16)).astype(np.float32)
        m_sample = mine_s.step(mx.array(v), t, m_sample)
        r_sample = ref_s.step(torch.from_numpy(v), t, r_sample, return_dict=False)[0]
        worst = max(worst, float(np.abs(np.array(m_sample) - r_sample.numpy()).max()))
    check("scheduler trajectory (16 steps)", worst == 0.0, f"max delta {worst:.3e}")

    # scale_noise on a conditioning anchor.
    x0 = rng.standard_normal((1, 4, 8)).astype(np.float32)
    noise = rng.standard_normal((1, 4, 8)).astype(np.float32)
    g = np.array(mine_s.scale_noise(mx.array(x0), mine.KEYFRAME_NOISE_AUG, mx.array(noise)))
    w = ref_s.scale_noise(
        torch.from_numpy(x0), mine.KEYFRAME_NOISE_AUG, torch.from_numpy(noise)
    ).numpy()
    check("scale_noise", np.allclose(g, w, atol=1e-7), f"max delta {np.abs(g - w).max():.3e}")


def main() -> int:
    test_canvas_and_frames()
    test_patchify_roundtrip()
    test_packed_sequence()
    test_scheduler()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("all packing + scheduler parity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
