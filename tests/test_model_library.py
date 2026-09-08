import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from wee_todd_mlx.model_library import (
    content_hash,
    conversion_cache_key,
    inspect_safetensors_header,
    scan_model_library,
)


def tensor_file(path, value=b"1234", name="layer.weight"):
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + value)
    return path


def test_shared_paths_and_cycles_do_not_duplicate_weights(tmp_path):
    weights = tensor_file(tmp_path / "original.safetensors")
    (tmp_path / "alias.safetensors").symlink_to(weights)
    (tmp_path / "hardlink.safetensors").hardlink_to(weights)
    (tmp_path / "cycle").symlink_to(tmp_path, target_is_directory=True)
    before = weights.read_bytes()
    report = scan_model_library([tmp_path])
    assert report["distinct_physical_files"] == 1
    assert len(report["assets"][0]["paths"]) == 3
    assert len(report["directory_aliases"]) == 1
    assert report["assets"][0]["compatibility"] == "not_evaluated"
    assert report["verified_content_duplicates"] == []
    assert weights.read_bytes() == before


def test_same_size_is_not_duplicate_proof_and_renaming_is_irrelevant(tmp_path):
    first = tensor_file(tmp_path / "publisher_a.safetensors")
    second = tensor_file(tmp_path / "renamed_by_user.safetensors")
    tensor_file(tmp_path / "different_training.safetensors", b"5678")
    assert scan_model_library([tmp_path])["verified_content_duplicates"] == []
    report = scan_model_library([tmp_path], hash_duplicates=True)
    assert len(report["verified_content_duplicates"]) == 1
    assert set(report["verified_content_duplicates"][0]["files"]) == {str(first), str(second)}
    assert content_hash(first) == content_hash(second)


def test_unsupported_and_malformed_formats_never_execute(tmp_path):
    (tmp_path / "untrusted.pt").write_bytes(b"not loaded or unpickled")
    (tmp_path / "broken.safetensors").write_bytes(struct.pack("<Q", 2**63))
    report = scan_model_library([tmp_path])
    assert report["distinct_physical_files"] == 2
    broken = next(a for a in report["assets"] if a["format"] == "safetensors")
    assert broken["header_valid"] is False
    assert "header" not in next(a for a in report["assets"] if a["format"] == "pt")


def test_header_checks_payload_size(tmp_path):
    weights = tensor_file(tmp_path / "adapter.safetensors", name="target.lora_A.weight")
    assert inspect_safetensors_header(weights)["has_adapter_keys"]
    weights.write_bytes(weights.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload offsets"):
        inspect_safetensors_header(weights)


def test_missing_root_reported_and_no_implicit_scan(tmp_path):
    report = scan_model_library([tmp_path / "missing"])
    assert len(report["errors"]) == 1
    assert report["assets"] == []
    with pytest.raises(ValueError, match="at least one"):
        scan_model_library([])


def test_invalid_dtype_reported_without_crashing_inventory(tmp_path):
    header = json.dumps({"bad": {"dtype": [], "shape": [1], "data_offsets": [0, 4]}}).encode()
    (tmp_path / "invalid.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"1234"
    )
    asset = scan_model_library([tmp_path])["assets"][0]
    assert not asset["header_valid"]
    assert "dtype" in asset["header_error"]


def test_metadata_is_optional_including_mlx_null(tmp_path):
    header = json.dumps(
        {"__metadata__": None, "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    weights = tmp_path / "user_variant.safetensors"
    weights.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    assert inspect_safetensors_header(weights)["metadata"] == {}


def test_conversion_cache_identity_includes_settings_and_version():
    sha = "a" * 64
    key = conversion_cache_key(sha, "converter-v1", {"bits": 8, "group": 64})
    assert key == conversion_cache_key(sha, "converter-v1", {"group": 64, "bits": 8})
    assert key != conversion_cache_key(sha, "converter-v2", {"bits": 8, "group": 64})
    assert key != conversion_cache_key(sha, "converter-v1", {"bits": 4, "group": 64})
    assert key != conversion_cache_key("b" * 64, "converter-v1", {"bits": 8, "group": 64})
    with pytest.raises(ValueError, match="full source"):
        conversion_cache_key("filename", "converter-v1", {})


def test_library_import_is_independent_of_render_runtimes():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wee_todd_mlx.model_library; "
            "assert not any(n.split('.')[0] in {'mlx','torch','comfy','wee_todd_nodes'} "
            "for n in sys.modules)",
        ],
        cwd="/",
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
