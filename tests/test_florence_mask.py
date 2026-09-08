import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wee_todd_nodes.florence_mask_nodes import (
    Florence2ModelSpec,
    WeeToddFlorence2ModelLoader,
    WeeToddFlorence2TextMask,
    _guided_silhouette_mask,
    _interpolate_boxes,
    _parse_florence_boxes,
    _sample_indices,
    _validate_tokenizer_contract,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    @property
    def shape(self):
        return self.value.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def fake_torch(monkeypatch):
    module = SimpleNamespace(from_numpy=lambda value: FakeTensor(value))
    monkeypatch.setattr("wee_todd_nodes.florence_mask_nodes._torch_module", lambda: module)
    return module


def test_parse_florence_boxes_uses_official_bin_center_dequantization():
    boxes = _parse_florence_boxes("<s>person<loc_100><loc_200><loc_600><loc_800></s>", 1000, 500)

    assert boxes == [{"label": "person", "bbox": [100.5, 100.25, 600.5, 400.25]}]


def test_sparse_indices_include_final_frame_and_box_interpolation_is_linear():
    assert _sample_indices(10, 4) == [0, 4, 8, 9]
    boxes = _interpolate_boxes(
        {
            0: np.array([0, 0, 10, 10], dtype=np.float32),
            2: np.array([10, 10, 20, 20], dtype=np.float32),
        },
        3,
    )

    np.testing.assert_array_equal(boxes[1], np.array([5, 5, 15, 15]))


def test_guided_silhouette_removes_box_background():
    frame = np.ones((96, 96, 3), dtype=np.float32) * 0.9
    cv2 = pytest.importorskip("cv2")
    cv2.circle(frame, (48, 48), 24, (0.05, 0.1, 0.2), -1)

    mask, used_fallback = _guided_silhouette_mask(
        frame, np.array([16, 16, 80, 80], dtype=np.float32), 0.0, 0
    )

    assert used_fallback is False
    assert mask[48, 48] == pytest.approx(1.0)
    assert mask[18, 18] == pytest.approx(0.0)


def test_loader_is_lazy_and_records_mit_provenance(tmp_path, monkeypatch):
    bundle = tmp_path / "Florence-2-base-ft-4bit"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        json.dumps({"model_type": "florence2", "quantization": {"bits": 8}})
    )
    monkeypatch.setattr(
        "wee_todd_nodes.florence_mask_nodes._resolve_model_bundle", lambda _name: bundle
    )

    spec, raw_info = WeeToddFlorence2ModelLoader().select(bundle.name)

    assert spec == Florence2ModelSpec(bundle.name, bundle)
    info = json.loads(raw_info)
    assert info["backend"] == "mlx-vlm"
    assert info["license"] == "MIT"
    assert info["loaded"] is False


def test_loader_rejects_unreliable_four_bit_coordinate_decoder(tmp_path, monkeypatch):
    bundle = tmp_path / "Florence-2-base-ft-4bit"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        json.dumps({"model_type": "florence2", "quantization": {"bits": 4}})
    )
    monkeypatch.setattr(
        "wee_todd_nodes.florence_mask_nodes._resolve_model_bundle", lambda _name: bundle
    )

    with pytest.raises(ValueError, match="4-bit coordinate decoding"):
        WeeToddFlorence2ModelLoader().select(bundle.name)


def test_florence_bundle_requires_declared_bart_tokenizer(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BartTokenizer"})
    )
    assert _validate_tokenizer_contract(tmp_path) == "BartTokenizer"

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "RobertaTokenizerFast"})
    )
    with pytest.raises(ValueError, match="BART tokenizer contract"):
        _validate_tokenizer_contract(tmp_path)


def test_text_mask_runs_only_sparse_frames_and_unloads(monkeypatch, tmp_path, fake_torch):
    calls = []
    runtime = SimpleNamespace(unloaded=False)

    def load(_spec):
        return object(), object(), True

    def unload():
        runtime.unloaded = True
        return True

    runtime.load = load
    runtime.unload = unload
    monkeypatch.setattr("wee_todd_nodes.florence_mask_nodes.FLORENCE2_RUNTIME", runtime)

    def generate(_model, _processor, prompt, image, **_kwargs):
        calls.append((prompt, image.size))
        return SimpleNamespace(
            text="person<loc_250><loc_200><loc_750><loc_800>",
            generation_tokens=5,
            finish_reason="stop",
        )

    import mlx_vlm

    monkeypatch.setattr(mlx_vlm, "generate", generate)
    images = FakeTensor(np.zeros((5, 40, 80, 3), dtype=np.float32))
    spec = Florence2ModelSpec("fixture", Path(tmp_path))

    mask, preview, boxes_json, info_json = WeeToddFlorence2TextMask().detect(
        images,
        spec,
        "person",
        "bounding box (fast)",
        3,
        0.0,
        0,
        64,
        False,
    )

    assert len(calls) == 3
    assert all(call[0] == "<OPEN_VOCABULARY_DETECTION>person" for call in calls)
    assert tuple(mask.shape) == (5, 40, 80)
    assert tuple(preview.shape) == (5, 40, 80, 3)
    assert float(mask.value[:, 12:28, 28:52].mean()) == pytest.approx(1.0)
    assert sorted(json.loads(boxes_json)) == ["0", "3", "4"]
    assert json.loads(info_json)["evaluations"] == 3
    assert json.loads(info_json)["mask_method"] == "Florence text box"
    assert runtime.unloaded is True


def test_text_mask_rejects_empty_subject_before_loading(fake_torch):
    with pytest.raises(ValueError, match="non-empty"):
        WeeToddFlorence2TextMask().detect(
            FakeTensor(np.zeros((1, 8, 8, 3), dtype=np.float32)),
            Florence2ModelSpec("fixture", Path(".")),
            " ",
            "guided silhouette (recommended)",
            1,
            0.0,
            0,
            32,
            False,
        )
