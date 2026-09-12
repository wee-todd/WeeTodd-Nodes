import hashlib
import math

import pytest

from wee_todd_remote.conditioning import (
    canonical_inputs,
    canonical_loras,
    validate_canonical_inputs,
)


def test_first_frame_resolves_absolute_image_and_hashes_exact_bytes(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"exact-image-bytes")
    result = canonical_inputs(
        [{"assetID": "asset-1", "role": "first", "strength": 1}],
        [{"id": "asset-1", "kind": "image", "path": str(image)}],
    )
    assert result == [
        {
            "role": "first",
            "path": str(image.resolve()),
            "sha256": hashlib.sha256(b"exact-image-bytes").hexdigest(),
            "frameIndex": 0,
            "strength": 1,
        }
    ]


def test_first_frame_hash_changes_when_file_contents_change(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"before")
    attachments = [{"assetID": "asset-1", "role": "first"}]
    assets = [{"id": "asset-1", "kind": "image", "path": str(image)}]
    before = canonical_inputs(attachments, assets)
    image.write_bytes(b"after")
    assert canonical_inputs(attachments, assets)[0]["sha256"] != before[0]["sha256"]


def test_h3_endpoint_files_are_both_checked_before_submission(tmp_path):
    images = [tmp_path / "first.png", tmp_path / "last.png"]
    for image in images:
        image.write_bytes(image.name.encode())
    inputs = canonical_inputs(
        [{"role": role, "assetID": role} for role in ("first", "last")],
        [{"id": role, "kind": "image", "path": str(image)}
         for role, image in zip(("first", "last"), images, strict=True)],
        model_family="minimaxH3", num_frames=124,
    )
    request = {"operation": "video",
               "configuration": {"numFrames": 124}, "inputs": inputs}
    validate_canonical_inputs(request)
    with pytest.raises(ValueError, match="final frame"):
        validate_canonical_inputs({**request, "configuration": {"numFrames": 141}})
    images[1].write_bytes(b"changed after prepare")
    with pytest.raises(ValueError, match="hash"):
        validate_canonical_inputs(request)


@pytest.mark.parametrize(
    "role", ["reference", "last", "keyframe", "audioDriver", "control", "lora"]
)
def test_unverified_attachment_roles_are_rejected_actionably(tmp_path, role):
    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    with pytest.raises(ValueError, match=role):
        canonical_inputs(
            [{"assetID": "asset-1", "role": role}],
            [{"id": "asset-1", "kind": "image", "path": str(image)}],
        )


def test_duplicate_first_frames_are_rejected(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"image")
    with pytest.raises(ValueError, match="only one first-frame"):
        canonical_inputs(
            [{"assetID": "asset-1", "role": "first"}, {"assetID": "asset-1", "role": "first"}],
            [{"id": "asset-1", "kind": "image", "path": str(image)}],
        )


def test_missing_nonimage_and_nonunit_first_frame_are_rejected(tmp_path):
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    with pytest.raises(ValueError, match="image"):
        canonical_inputs(
            [{"assetID": "asset-1", "role": "first"}],
            [{"id": "asset-1", "kind": "video", "path": str(movie)}],
        )
    with pytest.raises(ValueError, match="strength 1"):
        canonical_inputs(
            [{"assetID": "asset-1", "role": "first", "strength": 0.5}],
            [{"id": "asset-1", "kind": "image", "path": str(movie)}],
        )


def test_loras_are_canonical_server_ids_with_bounded_finite_weights():
    assert canonical_loras(
        [{"modelID": "style-a", "weight": 0}, {"modelID": "style-b", "weight": 2}]
    ) == [{"modelID": "style-a", "weight": 0.0}, {"modelID": "style-b", "weight": 2.0}]
    for weight in (-0.1, 2.1, math.nan, math.inf, True):
        with pytest.raises(ValueError, match="weight"):
            canonical_loras([{"modelID": "style-a", "weight": weight}])


def test_loras_reject_duplicate_ids_local_paths_and_more_than_sixteen():
    with pytest.raises(ValueError, match="duplicate"):
        canonical_loras([{"modelID": "same", "weight": 1}, {"modelID": "same", "weight": 0.5}])
    with pytest.raises(ValueError, match="server-resident"):
        canonical_loras([{"path": "/tmp/style.safetensors", "weight": 1}])
    with pytest.raises(ValueError, match="16"):
        canonical_loras([{"modelID": f"style-{index}", "weight": 1} for index in range(17)])
