"""Verify a saved FastH3 Speed artifact without equating waveform similarity with quality.

Technical checks are automatic. Optional local AudioSet classification is supporting evidence,
not a replacement for listening to sound-effect fidelity and event synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def technical_checks(metadata, video, audio):
    sampling = metadata.get("sampling", {})
    thinning = sampling.get("fastvideo") or {}
    attention = sampling.get("sol_attention") or {}
    kept = thinning.get("active_layer_indices", [])
    skipped = thinning.get("skipped_layer_indices", [])
    return {
        "four_evaluations": sampling.get("transformer_evaluations") == 4,
        "forty_joint_layers": (
            thinning.get("executed_layers") == 40
            and thinning.get("skipped_layers") == 10
            and len(kept) == 40
            and len(skipped) == 10
            and sorted([*kept, *skipped]) == list(range(50))
            and {0, 1, 48, 49}.issubset(kept)
        ),
        "compact_metal_dispatch": (
            attention.get("executed_calls") == 160
            and attention.get("fallback_calls") == 0
            and attention.get("storage_layout") == "compact_preordered"
        ),
        "staged_transformer_release": sampling.get("transformer_resident") is False,
        "frames_preserved": video["frames"] == metadata.get("frames"),
        "video_not_collapsed": (
            video["near_frozen_frame_pair_fraction"] < 0.95
            and video["black_pixel_fraction"] < 0.99
            and video["white_pixel_fraction"] < 0.99
        ),
        "audio_finite": audio["finite"],
        "audio_present_stereo": audio["channels"] == 2 and audio["rms"] > 1e-5,
        "audio_not_materially_clipped": audio["clipped_sample_fraction"] < 0.001,
        "stream_synchronization": abs(metadata.get("av_drift_seconds", 1)) <= 0.025,
        "delivery_rates": metadata.get("fps") == 24 and metadata.get("sample_rate") == 32000,
    }


def stream_checks(metadata, probe):
    """Check the actual container, not just the generation sidecar's claims."""
    streams = {stream["codec_type"]: stream for stream in probe["streams"]}
    video = streams.get("video", {})
    audio = streams.get("audio", {})
    try:
        contract = (
            video["width"] == metadata["width"]
            and video["height"] == metadata["height"]
            and int(video["nb_read_frames"]) == metadata["frames"]
            and video["r_frame_rate"] == "24/1"
            and audio["channels"] == 2
            and int(audio["sample_rate"]) == 32000
        )
        start_delta = abs(float(video["start_time"]) - float(audio["start_time"]))
        duration_delta = abs(float(video["duration"]) - float(audio["duration"]))
        synchronized = start_delta <= 0.025 and duration_delta <= 0.025
    except (KeyError, TypeError, ValueError):
        contract, synchronized = False, False
    return {
        "decoded_stream_contract": contract,
        "container_av_synchronization": synchronized,
    }


def classify_audio(video_path, ffmpeg, model_path):
    import numpy as np
    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    torch.set_num_threads(4)
    decoded = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    waveform = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
    extractor = AutoFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(str(model_path), local_files_only=True).eval()
    duration = waveform.size / 16000
    intervals = [(0.0, duration)]
    intervals.extend(
        (float(start), min(float(start) + 2.0, duration))
        for start in range(max(1, int(duration) - 1))
    )
    results = []
    for start, end in intervals:
        inputs = extractor(
            waveform[int(start * 16000) : int(end * 16000)],
            sampling_rate=16000,
            return_tensors="pt",
        )
        with torch.inference_mode():
            probabilities = model(**inputs).logits.sigmoid()[0]
        values, indices = probabilities.topk(10)
        results.append(
            {
                "start_seconds": start,
                "end_seconds": end,
                "labels": [
                    {"label": model.config.id2label[int(index)], "score": float(value)}
                    for value, index in zip(values, indices, strict=True)
                ],
            }
        )
    return {
        "model": str(model_path),
        "intervals": results,
        "scope": "automated_support_only_not_semantic_listening",
    }


def transcribe_audio(video_path, ffmpeg, model_path, expected):
    import mlx_whisper
    import numpy as np
    from transcribe_fasth3_acceptance import _word_error_rate

    decoded = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    waveform = np.frombuffer(decoded.stdout, dtype=np.float32).copy()
    result = mlx_whisper.transcribe(
        waveform,
        path_or_hf_repo=str(model_path),
        language="en",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        verbose=False,
    )
    transcript = result["text"].strip()
    return {
        "model": str(model_path),
        "expected": expected,
        "transcript": transcript,
        "word_error_rate": _word_error_rate(expected, transcript),
        "scope": "speech_content_only_not_vocal_naturalness_or_lip_sync",
    }


def contact_sheet(video, reference, target):
    import cv2
    import numpy as np

    rows = [("40-layer Speed candidate", video)]
    if reference is not None:
        rows.insert(0, ("50-layer Balanced control", reference))
    cell_width = 384
    cell_height = round(video.shape[1] * cell_width / video.shape[2])
    sheet = np.full((len(rows) * (cell_height + 32), 5 * cell_width, 3), 245, dtype=np.uint8)
    for row, (label, frames) in enumerate(rows):
        top = row * (cell_height + 32)
        cv2.putText(
            sheet,
            label,
            (10, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for column, index in enumerate(np.linspace(0, len(frames) - 1, 5).round().astype(int)):
            left = column * cell_width
            resized = cv2.resize(
                frames[index], (cell_width, cell_height), interpolation=cv2.INTER_AREA
            )
            sheet[top + 32 : top + 32 + cell_height, left : left + cell_width] = resized
            cv2.putText(
                sheet,
                f"{index / 24:.2f}s",
                (left + 8, top + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(target), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write contact sheet: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--ast-model", type=Path)
    parser.add_argument("--whisper-model", type=Path)
    parser.add_argument("--expected-speech")
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    args = parser.parse_args()
    required = [args.video, args.metadata, args.ffmpeg, args.ffprobe]
    if args.reference:
        required.append(args.reference)
    if args.ast_model:
        required.append(args.ast_model / "config.json")
    if bool(args.whisper_model) != bool(args.expected_speech):
        parser.error("--whisper-model and --expected-speech must be supplied together")
    if args.whisper_model:
        required.extend(
            [args.whisper_model / "config.json", args.whisper_model / "weights.safetensors"]
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError("Choose a unique verification report path.")

    import numpy as np
    from analyze_fasth3_acceptance import _audio_health, _decode_audio, _decode_video, _video_health

    metadata = json.loads(args.metadata.read_text())
    probe = json.loads(
        subprocess.run(
            [
                str(args.ffprobe),
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(args.video),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    video = _decode_video(args.video)
    audio = _decode_audio(args.video, args.ffmpeg)
    video_health = _video_health(video)
    audio_health = {**_audio_health(audio), "finite": bool(np.isfinite(audio).all())}
    checks = technical_checks(metadata, video_health, audio_health)
    checks.update(stream_checks(metadata, probe))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet = args.output.with_name(args.output.stem + "-contact-sheet.jpg")
    reference = _decode_video(args.reference) if args.reference else None
    contact_sheet(video, reference, sheet)
    dialogue = (
        transcribe_audio(args.video, args.ffmpeg, args.whisper_model, args.expected_speech)
        if args.whisper_model
        else None
    )
    if dialogue is not None:
        checks["dialogue_transcription"] = dialogue["word_error_rate"] == 0.0
    report = {
        "artifact": str(args.video.resolve()),
        "sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "technical_gate": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "video_health": video_health,
        "audio_health": audio_health,
        "probe": probe,
        "contact_sheet": str(sheet.resolve()),
        "dialogue": dialogue,
        "automated_sound_events": classify_audio(args.video, args.ffmpeg, args.ast_model)
        if args.ast_model
        else None,
        "sound_effect_listening": "pending",
        "promotion": "candidate_until_sound_fidelity_and_event_sync_are_accepted",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "technical_gate": report["technical_gate"],
                "failed_checks": [key for key, passed in checks.items() if not passed],
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
