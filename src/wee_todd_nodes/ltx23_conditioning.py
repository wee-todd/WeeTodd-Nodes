"""Lightweight Comfy inputs; materialize media only for the duration of a render."""

import shutil
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from ltx23_mlx.ic_lora import CONTROL_TYPES, LTX23ICLoRASpec


class WeeToddLTX23ICLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WEETODD_LTX23_MODEL",),
                "lora": ("STRING", {"default": ""}),
                "family": (sorted(set(CONTROL_TYPES.values())),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}),
                "alpha": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 65536.0}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_MODEL", "STRING")
    RETURN_NAMES = ("model", "ic_lora_report")
    FUNCTION = "attach"
    CATEGORY = "WeeTodd/LTX 2.3/loaders"
    DESCRIPTION = (
        "Experimental IC-LoRA with task-aware topology. Union and Motion use resident "
        "distilled mode; Ingredients uses Dev two_stage with a validated distilled helper. "
        "Declare the trained family; filenames are not used to infer it."
    )

    def attach(self, model, lora, family, strength=1.0, alpha=-1.0):
        import json

        if model.loras or model.ic_loras:
            raise ValueError("IC-LoRA cannot be stacked with other adapters yet")
        path = Path(lora).expanduser()
        if not path.is_file() and not path.is_absolute():
            if ".." in path.parts:
                raise ValueError("Relative adapter paths cannot contain '..'")
            try:
                import folder_paths

                resolved = folder_paths.get_full_path("loras", lora)
                if resolved:
                    path = Path(resolved)
            except ImportError:
                pass
        spec = LTX23ICLoRASpec(
            str(path.resolve()), family, strength, None if alpha == -1 else alpha
        )
        report = spec.inspect()
        return replace(model, ic_loras=(spec,)), json.dumps(report, indent=2)


class WeeToddLTX23Keyframe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 1800,
                        "tooltip": "Zero-based frame index; -1 resolves to the final frame.",
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {"previous": ("WEETODD_LTX23_KEYFRAMES",)},
        }

    RETURN_TYPES = ("WEETODD_LTX23_KEYFRAMES",)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "append"
    CATEGORY = "WeeTodd/LTX 2.3/conditioning"
    DESCRIPTION = (
        "Chain timed keyframes for resident Dev two_stage generation. "
        "No post-decode frame insertion."
    )

    def append(self, image, frame_index=0, strength=1.0, previous=None):
        items = tuple(previous or ())
        if len(items) >= 12 or any(i["frame_index"] == frame_index for i in items):
            raise ValueError("Keyframes require unique indices and at most twelve images")
        if type(frame_index) is not int or frame_index < -1 or not 0 <= strength <= 1:
            raise ValueError("Invalid keyframe index or strength")
        return ((*items, {"image": image, "frame_index": frame_index, "strength": strength}),)


class WeeToddLTX23ControlVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "control_type": ([k for k in CONTROL_TYPES if k != "ingredients_reference_sheet"],),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_CONTROL",)
    RETURN_NAMES = ("control",)
    FUNCTION = "specify"
    CATEGORY = "WeeTodd/LTX 2.3/conditioning"
    DESCRIPTION = (
        "Preprocessed local control video, matching output fps and covering every output frame. "
        "Does not extract edges/depth/pose/tracks."
    )

    def specify(self, video_path, control_type, strength=1.0):
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if control_type not in CONTROL_TYPES or not 0 <= strength <= 1:
            raise ValueError("Invalid control type or strength")
        return (
            {
                "path": str(path),
                "kind": "video",
                "control_type": control_type,
                "strength": strength,
            },
        )


class WeeToddLTX23VideoExtension:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "direction": (["after", "before"], {"default": "after"}),
                "additional_frames": (
                    "INT",
                    {
                        "default": 24,
                        "min": 8,
                        "max": 720,
                        "step": 8,
                        "tooltip": "New output frames; LTX 2.3 extends in exact groups of eight.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_EXTENSION",)
    RETURN_NAMES = ("extension",)
    FUNCTION = "specify"
    CATEGORY = "WeeTodd/LTX 2.3/conditioning"
    DESCRIPTION = (
        "Extend one existing 8n+1-frame video before or after its retained source timeline. "
        "Use distilled mode for the qualified eight-evaluation speed path or Dev one_stage "
        "for the slower quality path; new frames are generated in groups of eight."
    )

    def specify(self, video_path, direction="after", additional_frames=24):
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if direction not in {"before", "after"}:
            raise ValueError("Extension direction must be before or after")
        if (
            type(additional_frames) is not int
            or not 8 <= additional_frames <= 720
            or additional_frames % 8
        ):
            raise ValueError("Extension frames must be a multiple of 8 in [8, 720]")
        return (
            {
                "path": str(path),
                "direction": direction,
                "additional_frames": additional_frames,
            },
        )


class WeeToddLTX23ControlFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_frames": ("IMAGE",),
                "control_type": (
                    [key for key in CONTROL_TYPES if key != "ingredients_reference_sheet"],
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_CONTROL",)
    RETURN_NAMES = ("control",)
    FUNCTION = "prepare"
    CATEGORY = "WeeTodd/LTX 2.3/conditioning"
    DESCRIPTION = (
        "Bridge an IMAGE batch from the MLX Canny, depth, DWPose, or motion-track "
        "preprocessors into an exact LTX 2.3 IC-LoRA guide."
    )

    def prepare(self, control_frames, control_type, strength=1.0):
        shape = getattr(control_frames, "shape", None)
        if shape is None or len(shape) != 4 or shape[0] < 1 or shape[-1] != 3:
            raise ValueError("Control frames require an RGB IMAGE batch")
        if control_type not in CONTROL_TYPES or not 0 <= strength <= 1:
            raise ValueError("Invalid control type or strength")
        return (
            {
                "images": control_frames,
                "kind": "frame_batch",
                "control_type": control_type,
                "strength": strength,
            },
        )


class WeeToddLTX23IngredientsReferenceSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_sheet": ("IMAGE",),
                "reference_description": ("STRING", {"multiline": True}),
                "generated_video_description": ("STRING", {"multiline": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("WEETODD_LTX23_CONTROL", "STRING")
    RETURN_NAMES = ("control", "trained_prompt")
    FUNCTION = "prepare"
    CATEGORY = "WeeTodd/LTX 2.3/conditioning"
    DESCRIPTION = (
        "Prepare one black-background Ingredients sheet and its trained two-part prompt. "
        "Use Dev two_stage mode, 768x448, 121+ frames, 24 fps, and adapter strength 1.4 "
        "in the loader."
    )

    def prepare(
        self, reference_sheet, reference_description, generated_video_description, strength=1.0
    ):
        if not reference_description.strip() or not generated_video_description.strip():
            raise ValueError("Ingredients requires both reference and generated-video descriptions")
        if not 0 <= strength <= 1:
            raise ValueError("Ingredients reference strength must be in [0, 1]")
        shape = getattr(reference_sheet, "shape", None)
        if shape is None or len(shape) != 4 or shape[0] != 1 or shape[-1] != 3:
            raise ValueError("Ingredients requires exactly one RGB reference-sheet IMAGE")
        prompt = (
            f"Reference sheet: {reference_description.strip()}\n\n"
            f"Generated video: {generated_video_description.strip()}"
        )
        return (
            {
                "image": reference_sheet,
                "kind": "image",
                "control_type": "ingredients_reference_sheet",
                "strength": strength,
            },
            prompt,
        )


@contextmanager
def prepared_conditioning(
    model, config, *, prompt=None, keyframes=None, audio=None, control=None, extension=None
):
    """Use the same task/media contract as headless; remove owned inputs on every exit."""
    import numpy as np
    from PIL import Image

    from ltx23_mlx.upscale import _host_audio
    from wee_todd_mlx.conditioning_media import (
        inspect_media,
        ltx_conditioning_kwargs,
        materialize_ingredients_video,
    )
    from wee_todd_mlx.task_conditioning import validate_conditioning

    with TemporaryDirectory(prefix="weetodd-ltx23-conditioning-") as directory:
        inputs = []
        for n, item in enumerate(keyframes or ()):
            value = item["image"]
            value = value.detach().cpu() if hasattr(value, "detach") else value
            array = np.asarray(value, dtype=np.float32)
            if (
                array.ndim != 4
                or array.shape[0] != 1
                or array.shape[-1] != 3
                or not np.isfinite(array).all()
            ):
                raise ValueError("Each keyframe requires exactly one finite RGB IMAGE")
            path = Path(directory) / f"keyframe-{n}.png"
            Image.fromarray((np.clip(array[0], 0, 1) * 255).astype(np.uint8)).save(path)
            inputs.append(
                dict(
                    id=f"keyframe-{n}",
                    kind="image",
                    role="keyframe",
                    path=str(path),
                    frame_index="last" if item["frame_index"] == -1 else item["frame_index"],
                    strength=item["strength"],
                )
            )
        if audio is not None:
            import wave

            waveform, sample_rate = _host_audio(audio)
            if not 0 < waveform.shape[1] / sample_rate <= 30 or sample_rate > 192000:
                raise ValueError("Audio driver must be nonempty, at most 30s and at most 192 kHz")
            path = Path(directory) / "driver.wav"
            pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2")
            with wave.open(str(path), "wb") as handle:
                handle.setparams((2, 2, sample_rate, pcm.shape[1], "NONE", "not compressed"))
                handle.writeframes(pcm.T.tobytes())
            inputs.append(dict(id="audio", kind="audio", role="audio_driver", path=str(path)))
        if control is not None:
            control = dict(control)
            if control["kind"] == "frame_batch":
                import subprocess

                value = control.pop("images")
                value = value.detach().cpu() if hasattr(value, "detach") else value
                array = np.asarray(value, dtype=np.float32)
                expected = (config.num_frames, config.height, config.width, 3)
                if array.shape != expected or not np.isfinite(array).all():
                    raise ValueError(
                        "Control IMAGE batch must exactly match output frames, height, and width"
                    )
                frames = Path(directory) / "control-frames"
                frames.mkdir()
                for index, frame in enumerate(array):
                    Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(
                        frames / f"{index:06d}.png"
                    )
                guide = Path(directory) / "control-guide.mp4"
                ffmpeg = shutil.which("ffmpeg")
                if not ffmpeg:
                    raise FileNotFoundError("LTX 2.3 control-frame transport requires ffmpeg")
                subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-nostdin",
                        "-y",
                        "-framerate",
                        str(config.frame_rate),
                        "-i",
                        str(frames / "%06d.png"),
                        "-frames:v",
                        str(config.num_frames),
                        "-pix_fmt",
                        "yuv420p",
                        str(guide),
                    ],
                    check=True,
                    timeout=120,
                )
                control.update(path=str(guide), kind="video")
            elif control["kind"] == "image":
                value = control.pop("image")
                value = value.detach().cpu() if hasattr(value, "detach") else value
                array = np.asarray(value, dtype=np.float32)
                path = Path(directory) / "ingredients-sheet.png"
                Image.fromarray((np.clip(array[0], 0, 1) * 255).astype(np.uint8)).save(path)
                control["path"] = str(path)
            inputs.append(dict(control, id="control", role="control"))
        if extension is not None:
            inputs.append(
                {
                    "id": "extension-source",
                    "kind": "video",
                    "role": "reference",
                    "path": extension["path"],
                    "strength": 1.0,
                }
            )
        if extension is not None:
            task = "extension"
        elif control is not None:
            task = (
                "ref2va"
                if control.get("control_type") == "ingredients_reference_sheet"
                else "control"
            )
        elif audio is not None:
            task = "a2v"
        elif keyframes:
            task = "fflf"
        else:
            task = "t2v"
        recipe = {
            "engine": "ltx23",
            "components": asdict(model),
            "config": asdict(config),
            "conditioning": {"version": 1, "task": task, "inputs": inputs},
        }
        if extension is not None:
            recipe["conditioning"].update(
                audio_policy="source_reencoded_and_generated_extension",
                extension={
                    "direction": extension["direction"],
                    "additional_frames": extension["additional_frames"],
                },
            )
        if prompt is not None:
            recipe["prompt"] = prompt
        # JSON recipes use lists while model specifications deliberately use immutable tuples.
        recipe["components"]["ic_loras"] = list(recipe["components"]["ic_loras"])
        contract = validate_conditioning(recipe)["contract"]
        reports = inspect_media(recipe, contract)
        kwargs = ltx_conditioning_kwargs(recipe, contract, reports)
        if "reference_sheet_input" in kwargs:
            item = kwargs.pop("reference_sheet_input")
            kwargs["control_inputs"] = [materialize_ingredients_video(item, config, directory)]
        yield kwargs, contract
