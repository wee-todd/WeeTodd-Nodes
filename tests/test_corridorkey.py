import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wee_todd_nodes.corridorkey_nodes import (
    CORRIDORKEY_RUNTIME,
    CorridorKeyModelSpec,
    WeeToddCorridorKeyAutoHint,
    WeeToddCorridorKeyComposite,
    WeeToddCorridorKeyKeyer,
    WeeToddCorridorKeyMaskRefine,
    WeeToddCorridorKeyModelLoader,
    _CorridorKeyRuntime,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def float(self):
        return self

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.value, axis))

    def repeat(self, *repeats):
        return FakeTensor(np.tile(self.value, repeats))

    def __add__(self, other):
        return FakeTensor(self.value + _array(other))

    def __radd__(self, other):
        return FakeTensor(_array(other) + self.value)

    def __sub__(self, other):
        return FakeTensor(self.value - _array(other))

    def __rsub__(self, other):
        return FakeTensor(_array(other) - self.value)

    def __mul__(self, other):
        return FakeTensor(self.value * _array(other))

    def __rmul__(self, other):
        return FakeTensor(_array(other) * self.value)


def _array(value):
    return value.value if isinstance(value, FakeTensor) else value


@pytest.fixture
def fake_torch(monkeypatch):
    module = SimpleNamespace(
        from_numpy=lambda value: FakeTensor(value),
        clamp=lambda value, low, high: FakeTensor(np.clip(_array(value), low, high)),
    )
    monkeypatch.setattr("wee_todd_nodes.corridorkey_nodes._torch_module", lambda: module)
    return module


def test_auto_chroma_hint_is_coarse_and_temporally_batched(fake_torch):
    images = np.zeros((2, 64, 96, 3), dtype=np.float32)
    images[:, :, :, 1] = 1.0
    images[:, 16:48, 28:68] = np.array([0.8, 0.2, 0.1])

    mask, raw_info = WeeToddCorridorKeyAutoHint().build(
        FakeTensor(images),
        "auto from first frame",
        0.06,
        0.12,
        2,
        3,
        0.15,
    )

    assert tuple(mask.shape) == (2, 64, 96)
    assert float(mask.value[:, 28:36, 40:56].mean()) > 0.9
    assert float(mask.value[:, :8, :8].mean()) < 0.05
    info = json.loads(raw_info)
    assert info["frames"] == 2
    assert info["method"] == "border-sampled chromaticity distance"


def test_auto_chroma_hint_rejects_blue_until_mlx_checkpoint_exists(fake_torch):
    images = np.zeros((1, 32, 32, 3), dtype=np.float32)
    images[:, :, :, 2] = 1.0

    with pytest.raises(ValueError, match="blue screen"):
        WeeToddCorridorKeyAutoHint().build(
            FakeTensor(images), "auto from first frame", 0.06, 0.12, 4, 6, 0.15
        )


def test_mask_refine_accepts_standard_comfy_mask_and_shrinks_default(fake_torch):
    mask = np.zeros((1, 32, 32), dtype=np.float32)
    mask[:, 4:28, 4:28] = 1.0

    (refined,) = WeeToddCorridorKeyMaskRefine().refine(FakeTensor(mask), -2, 0, 0, False)

    assert tuple(refined.shape) == (1, 32, 32)
    assert float(refined.value.sum()) < float(mask.sum())


def test_composite_supports_straight_color_and_single_background_broadcast(fake_torch):
    fg = FakeTensor(np.ones((2, 4, 4, 3), dtype=np.float32))
    bg = FakeTensor(np.zeros((1, 4, 4, 3), dtype=np.float32))
    matte = FakeTensor(np.full((2, 4, 4), 0.25, dtype=np.float32))

    (result,) = WeeToddCorridorKeyComposite().composite(fg, matte, bg, "straight color")

    assert tuple(result.shape) == (2, 4, 4, 3)
    assert np.allclose(result.value, 0.25)


def test_loader_profiles_are_lazy_and_reject_non_mlx_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "corridorkey_mlx.safetensors"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        "wee_todd_nodes.corridorkey_nodes._resolve_checkpoint", lambda _name: checkpoint
    )

    spec, raw_info = WeeToddCorridorKeyModelLoader().configure(
        checkpoint.name, "low memory — tiled 512", False
    )

    assert spec.tile_size == 512
    assert spec.compile_model is False
    assert spec.keep_loaded is False
    assert json.loads(raw_info)["external_license"].startswith("CorridorKey Licence")
    assert CORRIDORKEY_RUNTIME.loaded is False

    wrong = tmp_path / "legacy.pth"
    wrong.write_bytes(b"fixture")
    monkeypatch.setattr(
        "wee_todd_nodes.corridorkey_nodes._resolve_checkpoint", lambda _name: wrong
    )
    with pytest.raises(ValueError, match="safetensors"):
        WeeToddCorridorKeyModelLoader().configure(wrong.name, "balanced — compiled 1024", True)


def test_keyer_reuses_one_hint_explicitly_and_unloads_staged_runtime(
    monkeypatch, tmp_path, fake_torch
):
    class Engine:
        calls = 0

        def process_frame(self, image, mask, **_kwargs):
            self.calls += 1
            assert image.shape == (8, 12, 3)
            assert mask.shape == (8, 12)
            return {
                "fg": np.full((8, 12, 3), 128, dtype=np.uint8),
                "alpha": np.full((8, 12), 192, dtype=np.uint8),
            }

    runtime = SimpleNamespace(engine=Engine(), loaded=True, unloaded=False)

    def load(_spec):
        return runtime.engine, True

    def unload():
        runtime.unloaded = True
        runtime.loaded = False
        return True

    runtime.load = load
    runtime.unload = unload
    monkeypatch.setattr("wee_todd_nodes.corridorkey_nodes.CORRIDORKEY_RUNTIME", runtime)
    spec = CorridorKeyModelSpec(
        checkpoint_name="corridorkey_mlx.safetensors",
        checkpoint_path=Path(tmp_path / "corridorkey_mlx.safetensors"),
        profile="speed — compiled 512",
        inference_resolution=512,
        compile_model=True,
        tile_size=None,
        overlap=64,
        keep_loaded=False,
    )

    result = WeeToddCorridorKeyKeyer().key(
        spec,
        FakeTensor(np.zeros((2, 8, 12, 3), dtype=np.float32)),
        FakeTensor(np.ones((1, 8, 12), dtype=np.float32)),
        1.0,
    )

    assert runtime.engine.calls == 2
    assert runtime.unloaded is True
    assert tuple(result[0].shape) == (2, 8, 12, 3)
    assert tuple(result[1].shape) == (2, 8, 12)
    assert json.loads(result[4])["model_resident_after_run"] is False


def test_runtime_unloads_after_failed_external_import(monkeypatch, tmp_path):
    runtime = _CorridorKeyRuntime()
    spec = CorridorKeyModelSpec(
        checkpoint_name="corridorkey_mlx.safetensors",
        checkpoint_path=tmp_path / "corridorkey_mlx.safetensors",
        profile="speed — compiled 512",
        inference_resolution=512,
        compile_model=True,
        tile_size=None,
        overlap=64,
        keep_loaded=False,
    )
    monkeypatch.setitem(__import__("sys").modules, "corridorkey_mlx", None)

    with pytest.raises(RuntimeError, match="separately licensed"):
        runtime.load(spec)
    assert runtime.loaded is False
