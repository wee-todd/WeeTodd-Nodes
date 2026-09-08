"""Lightweight ComfyUI adapters for the shared experimental H3 motion renderer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


class WeeToddH3MotionSettings:
    CATEGORY = "WeeTodd/H3/sampling"
    FUNCTION = "configure"
    RETURN_TYPES = ("WEETODD_H3_MOTION_SETTINGS",)
    RETURN_NAMES = ("motion_settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["adaptive", "uniform"],),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1, "step": 0.05}),
                "max_hold": ("INT", {"default": 2, "min": 2, "max": 4}),
                "sensitivity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
                "max_frames": ("INT", {"default": 345, "min": 73, "max": 345}),
            }
        }

    def configure(self, mode, strength, max_hold, sensitivity, seed, max_frames):
        from wee_todd_mlx.motion_fidelity import MotionSettings

        settings = MotionSettings(True, mode, strength, max_hold, sensitivity, seed, max_frames)
        settings.validate()
        return (asdict(settings),)


class WeeToddH3MotionRefine:
    CATEGORY = "WeeTodd/H3/output"
    FUNCTION = "refine"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "analysis_report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "components": ("WEETODD_H3_COMPONENTS",),
                "config": ("WEETODD_H3_CONFIG",),
                "motion_settings": ("WEETODD_H3_MOTION_SETTINGS",),
                "source_video": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "source_in": ("FLOAT", {"default": 0, "min": 0, "step": 1 / 24}),
                "duration": ("FLOAT", {"default": 2.5, "min": 2.5, "max": 14.375, "step": 1 / 24}),
                "analyze_only": ("BOOLEAN", {"default": True}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, source_video, **kwargs):
        source = Path(source_video).expanduser()
        return (
            (source.stat().st_size, source.stat().st_mtime_ns) if source.is_file() else float("nan")
        )

    def refine(
        self,
        components,
        config,
        motion_settings,
        source_video,
        prompt,
        source_in,
        duration,
        analyze_only=True,
    ):
        import signal
        import subprocess
        import sys
        import uuid

        import comfy.model_management
        import folder_paths

        from minimax_h3_mlx.media import resolve_ffmpeg
        from wee_todd_mlx.motion_fidelity import validate_recipe

        recipe = {
            "format": "weetodd-headless-v2",
            "engine": "h3",
            "components": asdict(components),
            "config": asdict(config),
            "prompt": prompt,
        }
        validate_recipe(recipe)
        source = Path(source_video).expanduser().resolve(strict=True)
        destination = (
            Path(folder_paths.get_output_directory()) / "weetodd-motion" / uuid.uuid4().hex
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        request_path = destination.with_suffix(".json")
        ffmpeg = resolve_ffmpeg().path
        request = {
            "recipe": recipe,
            "clip": {"sourcePath": str(source), "sourceIn": source_in, "duration": duration},
            "settings": motion_settings,
            "runtime": {"ffmpegPath": str(ffmpeg)},
        }
        request_path.write_text(json.dumps(request, indent=2))
        root = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(root / "scripts/motion_fidelity.py"),
            "--request",
            str(request_path),
            "--output-directory",
            str(destination),
        ]
        if analyze_only:
            command.append("--analyze-only")
        # A preceding graph may have kept H3 stages resident. Release this project's
        # parent caches before starting the isolated weighted worker.
        from .conditioning import TEXT_ENCODER_RUNTIME
        from .decoding import AUDIO_VAE_RUNTIME, VIDEO_VAE_RUNTIME
        from .runtime import RUNTIME
        from .sampling import TRANSFORMER_RUNTIME

        for cache in (
            TEXT_ENCODER_RUNTIME,
            VIDEO_VAE_RUNTIME,
            AUDIO_VAE_RUNTIME,
            TRANSFORMER_RUNTIME,
            RUNTIME,
        ):
            cache.unload()
        child = subprocess.Popen(command)
        try:
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                try:
                    code = child.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if code:
                raise RuntimeError(
                    f"Motion Fidelity exited with status {code}; see the console log."
                )
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        result = json.loads((destination / "result.json").read_text())
        return (result["video"], json.dumps(result["plan"], indent=2))


NODE_CLASS_MAPPINGS = {
    "WeeToddH3MotionSettings": WeeToddH3MotionSettings,
    "WeeToddH3MotionRefine": WeeToddH3MotionRefine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddH3MotionSettings": "WeeTodd H3 Motion Fidelity Settings (Experimental)",
    "WeeToddH3MotionRefine": "WeeTodd H3 Motion Fidelity Refine (Experimental)",
}
