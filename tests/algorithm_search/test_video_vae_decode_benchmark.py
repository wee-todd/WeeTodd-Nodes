from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/algorithm_search/video_vae_decode_benchmark.py")
    spec = importlib.util.spec_from_file_location("video_vae_decode_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_digest_is_stable_and_value_sensitive() -> None:
    mx = pytest.importorskip("mlx.core")
    module = _module()

    first = module._digest(mx.array([1.0, 2.0]))
    same = module._digest(mx.array([1.0, 2.0]))
    different = module._digest(mx.array([1.0, 3.0]))

    assert first == same
    assert first != different
