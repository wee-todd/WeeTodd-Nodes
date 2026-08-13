import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import save_file

from ltx25_mlx.components import LTX25LatentNormalizer, remap_convolution_layout


class _Kernels(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 5, kernel_size=7)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3)
        self.conv3 = nn.Conv3d(3, 5, kernel_size=3)
        self.linear = nn.Linear(3, 5)


def test_official_pytorch_convolutions_are_shape_verified_for_mlx():
    model = _Kernels()
    weights = {
        "conv1.weight": mx.zeros((5, 3, 7)),
        "conv2.weight": mx.zeros((5, 3, 3, 3)),
        "conv3.weight": mx.zeros((5, 3, 3, 3, 3)),
        "linear.weight": mx.zeros((5, 3)),
    }
    mapped = remap_convolution_layout(model, weights)
    assert mapped["conv1.weight"].shape == model.conv1.weight.shape
    assert mapped["conv2.weight"].shape == model.conv2.weight.shape
    assert mapped["conv3.weight"].shape == model.conv3.weight.shape
    assert mapped["linear.weight"].shape == model.linear.weight.shape
    assert mapped["linear.weight"] is weights["linear.weight"]


def test_latent_normalizer_uses_statistics_without_loading_video_encoder(tmp_path):
    path = tmp_path / "video_vae.safetensors"
    mean = np.arange(4, dtype=np.float32)
    std = np.full((4,), 2.0, dtype=np.float32)
    save_file(
        {
            "per_channel_statistics.mean-of-means": mean,
            "per_channel_statistics.std-of-means": std,
        },
        path,
    )
    normalizer = LTX25LatentNormalizer(path)
    latent = mx.array(mean).reshape(1, 1, 1, 1, 4)
    normalized = normalizer.normalize_latent(latent)
    restored = normalizer.denormalize_latent(normalized)
    assert mx.array_equal(normalized, mx.zeros_like(normalized)).item()
    assert mx.array_equal(restored, latent).item()
