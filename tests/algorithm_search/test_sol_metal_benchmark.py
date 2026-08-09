from __future__ import annotations

import json
import subprocess
import sys

import mlx.core as mx


def test_metal_benchmark_emits_explicit_generation_gate(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    mx.random.seed(47)
    shape = (1, 1, 192, 128)
    mx.save_safetensors(
        str(capture / "qkv.safetensors"),
        {f"tensor_{index}": mx.random.normal(shape).astype(mx.bfloat16) for index in range(3)},
    )
    (capture / "metadata.json").write_text(
        json.dumps(
            {
                "captures": [
                    {
                        "name": "attention_qkv",
                        "block": 24,
                        "evaluation_index": 1,
                        "path": "qkv.safetensors",
                        "metadata": {
                            "prefix_rows": 64,
                            "audio_rows": 32,
                            "target_video_rows": 128,
                        },
                    }
                ]
            }
        )
    )
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/algorithm_search/sol_metal_benchmark.py",
            str(capture),
            "--block",
            "24",
            "--evaluation",
            "1",
            "--warmups",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text())
    assert "performance_pass" in payload["gate"]
    assert "quality_pass" in payload["gate"]
    assert payload["candidate"]["operator"] == "complete_attention_with_pooling"
