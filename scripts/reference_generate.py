"""Generate a MiniMax-H3 clip on Apple Silicon.

    ./.venv/bin/python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4

Read the performance section of the README first: a step at the released 768-pixel canvas is ~8.8
minutes for a 5 s clip and ~1 hour for 15 s on an M3 Ultra, because MiniMax has not released its
sparse-attention implementation. `--height/--width` shrink the canvas for wiring checks, but H3 was
trained for a 768 short edge and anything else is off-distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.media import save_frames, save_mp4, save_wav
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline

DEFAULT_CHECKPOINT = "models/MiniMax-H3/FL2VA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("-o", "--output", default="out.mp4")
    parser.add_argument("-c", "--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="the upstream release; supplies the VAEs and text encoder")
    parser.add_argument("-t", "--transformer", default=None,
                        help="a quantized transformer directory to use instead of the release's")
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="seconds, 5 to 15")
    parser.add_argument("-s", "--steps", type=int, default=16, help="sigma grid points; drives steps - 1 forwards")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect", type=int, nargs=2, default=(16, 9))
    parser.add_argument("--height", type=int, default=None, help="canvas override, multiple of 32")
    parser.add_argument("--width", type=int, default=None, help="canvas override, multiple of 32")
    parser.add_argument("--image", action="append", default=None, help="keyframe image (repeatable)")
    parser.add_argument("--anchor", action="append", default=None, choices=["first", "last"],
                        help="anchor for each --image, in order")
    parser.add_argument("--keep-adaln", action="store_true",
                        help="keep the 13B adaln_proj resident instead of caching and dropping it")
    args = parser.parse_args()

    images = None
    if args.image:
        from PIL import Image, ImageOps

        images = [ImageOps.exif_transpose(Image.open(p).convert("RGB")) for p in args.image]
    anchors = tuple(args.anchor or ())
    if images and len(anchors) != len(images):
        parser.error(f"--anchor must be given once per --image ({len(images)} images, {len(anchors)} anchors)")

    pipe = MiniMaxH3Pipeline.from_pretrained(
        args.checkpoint, transformer_dir=args.transformer, load_vision=bool(images)
    )
    result = pipe(
        args.prompt,
        duration_seconds=args.duration,
        aspect=tuple(args.aspect),
        num_inference_steps=args.steps,
        seed=args.seed,
        images=images,
        keyframe_anchors=anchors,
        height=args.height,
        width=args.width,
        drop_adaln=not args.keep_adaln,
    )

    output = Path(args.output)
    try:
        save_mp4(output, result.video, result.fps, result.audio, result.sample_rate)
        print(f"\nwrote {output} ({result.video.shape[0]} frames, "
              f"{result.audio.shape[-1] / result.sample_rate:.2f}s audio)")
    except RuntimeError as exc:
        print(f"\nffmpeg unavailable ({exc}); writing frames and wav instead")
        save_frames(output.with_suffix(""), result.video)
        save_wav(output.with_suffix(".wav"), result.audio, result.sample_rate)

    print(f"{result.seconds_per_step:.1f}s per step, {result.total_seconds / 60:.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
