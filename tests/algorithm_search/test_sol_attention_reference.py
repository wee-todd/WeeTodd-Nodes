from __future__ import annotations

import json
import subprocess
import sys

import mlx.core as mx


def test_reference_script_records_density_error_and_audio_contract(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    shape = (1, 1, 20, 8)
    mx.random.seed(31)
    qkv = {f"tensor_{index}": mx.random.normal(shape).astype(mx.bfloat16) for index in range(3)}
    mx.save_safetensors(str(capture / "qkv.safetensors"), qkv)
    (capture / "metadata.json").write_text(
        json.dumps(
            {
                "captures": [
                    {
                        "name": "attention_qkv",
                        "block": 2,
                        "evaluation_index": 1,
                        "path": "qkv.safetensors",
                        "metadata": {
                            "prefix_rows": 4,
                            "text_rows": 2,
                            "audio_rows": 2,
                            "condition_video_rows": 0,
                            "target_video_rows": 16,
                            "attention_heads": [0],
                        },
                    }
                ]
            }
        )
    )
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm_search/sol_attention_reference.py",
            str(capture),
            "--block",
            "2",
            "--evaluation",
            "1",
            "--block-size",
            "4",
            "--query-blocks",
            "2",
            "--beta",
            "0.5",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert payload["audiovisual_contract"]["audio_rows_exact"] == 2
    experiment = payload["experiments"][0]
    assert 0.0 <= experiment["variants"]["corrected"]["telemetry"]["exact_route_density"] <= 1.0
    assert experiment["correction_improves_relative_l2"]
