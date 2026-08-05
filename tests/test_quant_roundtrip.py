"""A quantized build must survive save -> load and produce identical outputs.

Quantized layers hold packed weights plus scales and biases under different names than `nn.Linear`,
so a published quant is only usable if the loader rebuilds the same structure before reading the
shards. This writes a real directory with `scripts/build_quant.py`'s writer, reads it back through
`load_dit`, and requires the two models to agree **exactly** — not approximately, since nothing
lossy happens between saving and loading.

    ./.venv/bin/python tests/test_quant_roundtrip.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_quant import save_sharded

from minimax_h3_mlx.config import TAG_AUDIO, TAG_TEXT, TAG_VIDEO, DiTConfig
from minimax_h3_mlx.dit import MiniMaxH3DiT
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.quantize import QuantConfig, quantize_dit, resident_footprint

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def tiny_config() -> DiTConfig:
    hidden = 256
    return DiTConfig(
        hidden_size=hidden,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_hidden_size=128,
        latents_dim=4,
        audio_latents_dim=8,
        text_dim=128,
        timestep_input_dim=16,
        time_embed_hidden_size=hidden,
        time_embed_dim=64,
        adaln_out_features=6 * 3 * hidden,
        final_adaln_out_features=2 * hidden,
        rope_inv_freq_len=4,
    )


def sample_inputs(cfg: DiTConfig):
    n_text, n_video, n_audio = 4, 8, 4
    seq = n_text + n_video + n_audio
    rng = np.random.default_rng(0)
    tags = np.concatenate([
        np.full(n_text, TAG_TEXT), np.full(n_video, TAG_VIDEO), np.full(n_audio, TAG_AUDIO)
    ]).astype(np.int32)
    ts_i = np.concatenate([np.zeros(n_text), np.ones(n_video), np.zeros(n_audio)]).astype(np.int32)
    pos = np.stack([np.arange(seq) % 3, np.arange(seq) % 5, np.arange(seq) % 7], -1).astype(np.float32)
    return (
        mx.array(rng.standard_normal((1, n_video, cfg.video_patch_dim)).astype(np.float32)),
        mx.array(rng.standard_normal((1, n_audio, cfg.audio_latents_dim)).astype(np.float32)),
        mx.array(rng.standard_normal((1, n_text, cfg.text_dim)).astype(np.float32)),
        mx.array(np.array([0.0, 0.6], np.float32)),
        mx.array(ts_i), mx.array(tags), mx.array(pos),
        mx.array(np.arange(n_text, n_text + n_video, dtype=np.int32)),
        mx.array(np.arange(n_text + n_video, seq, dtype=np.int32)),
        mx.array(np.arange(n_text, dtype=np.int32)),
    )


def main() -> int:
    cfg = tiny_config()
    mx.random.seed(0)
    model = MiniMaxH3DiT(cfg)
    mx.eval(model.parameters())

    recipe = QuantConfig(bits=4, group_size=64)
    summary = quantize_dit(model, recipe)
    check("quantization applied", summary["quantized_layers"].get(4, 0) > 0,
          f"{summary['quantized_layers']} layers, {summary['ratio']:.2f}x smaller")

    args = sample_inputs(cfg)
    want_v, want_a = model(*args)
    mx.eval(want_v, want_a)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        with open(out / "config.json", "w") as fh:
            json.dump({
                "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers,
                "token_refiner_num_layers": cfg.token_refiner_num_layers,
                "num_attention_heads": cfg.num_attention_heads,
                "attention_head_dim": cfg.attention_head_dim,
                "ffn_hidden_size": cfg.ffn_hidden_size, "latents_dim": cfg.latents_dim,
                "audio_latents_dim": cfg.audio_latents_dim, "patch_size": list(cfg.patch_size),
                "text_dim": cfg.text_dim, "timestep_input_dim": cfg.timestep_input_dim,
                "time_embed_hidden_size": cfg.time_embed_hidden_size,
                "time_embed_dim": cfg.time_embed_dim,
                "adaln_out_features": cfg.adaln_out_features,
                "final_adaln_out_features": cfg.final_adaln_out_features,
                "rope_inv_freq_len": cfg.rope_inv_freq_len,
            }, fh)
        quant_meta = {"bits": 4, "group_size": 64, "quantize_adaln": False, "adaln_bits": None}
        with open(out / "quant_config.json", "w") as fh:
            json.dump(quant_meta, fh)
        names = save_sharded(model, out, {"quantization": json.dumps(quant_meta)})
        check("wrote shards", len(names) >= 1, f"{len(names)} shard(s) + index")

        reloaded = load_dit(out)

    keys_before = sorted(k for k, _ in tree_flatten(model.parameters()))
    keys_after = sorted(k for k, _ in tree_flatten(reloaded.parameters()))
    check("key sets identical", keys_before == keys_after,
          f"{len(keys_before)} vs {len(keys_after)}")

    got_v, got_a = reloaded(*args)
    mx.eval(got_v, got_a)
    dv = float(mx.max(mx.abs(got_v - want_v)).item())
    da = float(mx.max(mx.abs(got_a - want_a)).item())
    check("video output exact after reload", dv == 0.0, f"max|delta| {dv:.3e}")
    check("audio output exact after reload", da == 0.0, f"max|delta| {da:.3e}")

    footprint = resident_footprint(reloaded)
    check("adaln accounted separately", footprint["adaln_gb"] > 0,
          f"resident {footprint['resident_gb'] * 1e3:.1f} MB of {footprint['total_gb'] * 1e3:.1f} MB")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("quant round-trip passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
