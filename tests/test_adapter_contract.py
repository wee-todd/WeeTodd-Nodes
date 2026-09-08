import json
import math
import struct

import pytest

from wee_todd_mlx.adapter_contract import inspect_adapter, inspect_adapter_scaling


def adapter_file(path, entries):
    header, offset = {}, 0
    for key, shape in entries.items():
        stop = offset + math.prod(shape) * 4
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, stop]}
        offset = stop
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return path


@pytest.mark.parametrize(
    "a,b",
    [
        ("lora_A", "lora_B"),
        ("lora_A.default", "lora_B.default"),
        ("lora_A.turbo", "lora_B.turbo"),
        ("lora_down", "lora_up"),
        ("lora_a", "lora_b"),
    ],
)
def test_schema_normalization_is_rename_invariant(tmp_path, a, b):
    file = adapter_file(
        tmp_path / "any_source.safetensors",
        {f"block.{a}.weight": [2, 4], f"block.{b}.weight": [6, 2], "block.alpha": []},
    )
    before = inspect_adapter(file)
    renamed = tmp_path / "turbo_no_longer_changes_math.safetensors"
    file.rename(renamed)
    assert before == inspect_adapter(renamed)
    assert before["pairs"][0]["logical_shape"] == [6, 4]
    assert before["pairs"][0]["rank"] == 2
    assert before["pairs"][0]["alpha_tensor"] == "block.alpha"
    assert before["format"] == "weetodd-adapter-contract-v2"
    assert before["ranks"] == [2]
    assert len(before["target_fingerprint"]) == 64


def test_target_normalization_makes_schema_and_prefix_independent_fingerprint(tmp_path):
    first = adapter_file(
        tmp_path / "peft.safetensors",
        {
            "base_model.model.transformer.block.lora_A.weight": [2, 4],
            "base_model.model.transformer.block.lora_B.weight": [6, 2],
        },
    )
    second = adapter_file(
        tmp_path / "down-up.safetensors",
        {
            "block.lora_down.weight": [2, 4],
            "block.lora_up.weight": [6, 2],
        },
    )

    def normalize(target):
        return target.removeprefix("base_model.model.transformer.")

    first_report = inspect_adapter(first, target_normalizer=normalize)
    second_report = inspect_adapter(second, target_normalizer=normalize)

    assert first_report["target_fingerprint"] == second_report["target_fingerprint"]
    assert first_report["pair_schemas"] == ["ab"]
    assert second_report["pair_schemas"] == ["down_up"]


def test_only_explicit_auxiliary_tensors_are_accepted(tmp_path):
    file = adapter_file(
        tmp_path / "adapter.safetensors",
        {
            "block.lora_A.weight": [2, 4],
            "block.lora_B.weight": [6, 2],
            "slots.weight": [1],
        },
    )

    report = inspect_adapter(file, allowed_auxiliary_names={"slots.weight"})

    assert report["auxiliary_tensors"] == {"slots.weight": {"shape": [1], "dtype": "F32"}}


def test_exporter_scaling_metadata_normalizes_aliases_and_baked_markers():
    assert inspect_adapter_scaling(
        {
            "lora_rank": "4",
            "ss_network_dim": "4.0",
            "ss_network_alpha": "Dynamic",
            "network_alpha": "1",
        }
    ) == {"rank": 4.0, "alpha": 1.0}


@pytest.mark.parametrize(
    ("metadata", "message"),
    (
        ({"lora_rank": "0"}, "invalid rank"),
        ({"lora_alpha": "nan"}, "invalid alpha"),
        ({"lora_alpha": "1", "network_alpha": "2"}, "conflicting alpha"),
    ),
)
def test_exporter_scaling_metadata_fails_closed(metadata, message):
    with pytest.raises(ValueError, match=message):
        inspect_adapter_scaling(metadata)


@pytest.mark.parametrize("extra", ["block.lora_magnitude_vector", "unknown.weight"])
def test_unknown_fields_fail_closed(tmp_path, extra):
    file = adapter_file(
        tmp_path / "adapter.safetensors",
        {"block.lora_A.weight": [2, 4], "block.lora_B.weight": [6, 2], extra: [1]},
    )
    with pytest.raises(ValueError, match="refusing partial application"):
        inspect_adapter(file)


@pytest.mark.parametrize(
    "entries,message",
    [
        ({"x.lora_A.weight": [2, 4]}, "Incomplete"),
        ({"x.lora_A.weight": [2, 4], "x.lora_B.default.weight": [6, 2]}, "Ambiguous"),
        ({"x.lora_A.weight": [2, 4], "x.lora_B.weight": [6, 3]}, "rank or shape"),
        ({"x.lora_A.weight": [0, 4], "x.lora_B.weight": [6, 0]}, "rank or shape"),
        ({"x.lora_A.weight": [2, 4], "x.lora_B.weight": [6, 2], "z.alpha": []}, "without"),
        ({"x.lora_A.weight": [2, 4], "x.lora_B.weight": [6, 2], "x.alpha": [2]}, "scalar"),
    ],
)
def test_invalid_pair_contracts(tmp_path, entries, message):
    with pytest.raises(ValueError, match=message):
        inspect_adapter(adapter_file(tmp_path / "bad.safetensors", entries))
