import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx


def _module():
    path = Path("scripts/algorithm_search/profile_evaluation.py")
    spec = importlib.util.spec_from_file_location("profile_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_save_final_latents_round_trips(tmp_path: Path):
    module = _module()
    result = SimpleNamespace(
        video_latents=mx.ones((1, 2, 3)),
        audio_latents=mx.zeros((2, 3, 4)),
    )

    path = module._save_final_latents(tmp_path, result)
    loaded = mx.load(str(path))

    assert loaded["video_latents"].shape == (1, 2, 3)
    assert loaded["audio_latents"].shape == (2, 3, 4)
