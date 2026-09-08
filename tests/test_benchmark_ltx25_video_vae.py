from pathlib import Path

import pytest

from scripts.benchmark_ltx25_video_vae import _decode_targets


def test_ltx25_vae_benchmark_can_isolate_each_decoder():
    conv = Path("conv.safetensors")
    diffusion = Path("diffusion.safetensors")

    assert _decode_targets("conv", conv, diffusion) == (("conv", conv),)
    assert _decode_targets("diffusion", conv, diffusion) == (("diffusion", diffusion),)
    assert _decode_targets("both", conv, diffusion) == (
        ("conv", conv),
        ("diffusion", diffusion),
    )


def test_ltx25_vae_benchmark_rejects_unknown_decode_mode():
    with pytest.raises(ValueError, match="Unknown decode mode"):
        _decode_targets("exact", Path("conv"), Path("diffusion"))
