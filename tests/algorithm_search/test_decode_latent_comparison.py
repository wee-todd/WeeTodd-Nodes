import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path("scripts/algorithm_search/decode_latent_comparison.py")
    spec = importlib.util.spec_from_file_location("decode_latent_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_report_exact_and_changed_arrays():
    module = _module()
    baseline = np.ones((2, 3), dtype=np.float32)

    exact = module._metrics(baseline, baseline)
    changed = module._metrics(baseline, baseline * 0.5)

    assert exact["relative_l2_error"] == 0.0
    assert exact["cosine_similarity"] == pytest.approx(1.0)
    assert changed["relative_l2_error"] > 0.0
    assert changed["psnr_db"] > 0.0


def test_labeled_frames_preserve_shape_and_change_banner_pixels():
    module = _module()
    frames = np.zeros((2, 64, 96, 3), dtype=np.uint8)

    labeled = module._labeled(frames, "BF16")

    assert labeled.shape == frames.shape
    assert np.any(labeled != frames)
