"""Classic ComfyUI node contracts backed by the MLX MiniMax H3 pipeline."""
import json
from dataclasses import asdict
from pathlib import Path

from .runtime import RUNTIME, H3GenerationConfig, H3ModelSpec


def _output_directory() -> Path:
    try:
        import folder_paths
        return Path(folder_paths.get_output_directory())
    except ImportError:
        return Path.cwd() / "output"


class WeeToddH3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "checkpoint": ("STRING", {"default": "models/MiniMax-H3/FL2VA"}),
            "transformer": ("STRING", {"default": ""}),
            "load_vision": ("BOOLEAN", {"default": False}),
        }}
    RETURN_TYPES = ("WEETODD_H3_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "WeeTodd/H3"
    DESCRIPTION = "Describe an MLX MiniMax H3 checkpoint. Weights load lazily at generation time."

    def load(self, checkpoint, transformer, load_vision):
        return (H3ModelSpec(checkpoint, transformer or None, load_vision),)


class WeeToddH3GenerationConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.1}),
            "steps": ("INT", {"default": 16, "min": 2, "max": 100}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "width": ("INT", {"default": 640, "min": 32, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 384, "min": 32, "max": 2048, "step": 32}),
            "drop_adaln": ("BOOLEAN", {"default": True}),
        }}
    RETURN_TYPES = ("WEETODD_H3_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "configure"
    CATEGORY = "WeeTodd/H3"

    def configure(self, duration_seconds, steps, seed, width, height, drop_adaln):
        config = H3GenerationConfig(duration_seconds, steps, seed, width, height, drop_adaln)
        config.validate()
        return (config,)


class WeeToddH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("WEETODD_H3_MODEL",),
            "config": ("WEETODD_H3_CONFIG",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "filename_prefix": ("STRING", {"default": "WeeTodd/H3"}),
        }}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "generation_info")
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "WeeTodd/H3"
    DESCRIPTION = "Generate synchronized video and audio with MiniMax H3 through MLX."

    def generate(self, model, config, prompt, filename_prefix):
        config.validate()
        result = RUNTIME.get(model)(
            prompt, duration_seconds=config.duration_seconds,
            num_inference_steps=config.steps, seed=config.seed,
            height=config.height, width=config.width, drop_adaln=config.drop_adaln,
        )
        from minimax_h3_mlx.media import save_mp4

        prefix = Path(filename_prefix)
        output = _output_directory() / prefix.parent
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"{prefix.name}_{config.seed}.mp4"
        save_mp4(target, result.video, result.fps, result.audio, result.sample_rate)
        info = {"prompt": prompt, **asdict(config), "video_path": str(target),
                "seconds_per_step": result.seconds_per_step, "total_seconds": result.total_seconds}
        target.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        return (str(target), json.dumps(info, indent=2))


class WeeToddH3Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "release"
    CATEGORY = "WeeTodd/H3"

    def release(self, unload):
        if unload:
            RUNTIME.unload()
            return ("MiniMax H3 MLX runtime unloaded",)
        return ("MiniMax H3 MLX runtime kept warm",)


NODE_CLASS_MAPPINGS = {
    "WeeToddH3ModelLoader": WeeToddH3ModelLoader,
    "WeeToddH3GenerationConfig": WeeToddH3GenerationConfig,
    "WeeToddH3Generate": WeeToddH3Generate,
    "WeeToddH3Unload": WeeToddH3Unload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeToddH3ModelLoader": "WeeTodd H3 Model Loader (MLX)",
    "WeeToddH3GenerationConfig": "WeeTodd H3 Generation Config",
    "WeeToddH3Generate": "WeeTodd H3 Generate Video + Audio",
    "WeeToddH3Unload": "WeeTodd H3 Unload MLX Runtime",
}
