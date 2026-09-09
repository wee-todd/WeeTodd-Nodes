# Headless recipe examples

Studio’s **Model setup** creates validated recipes from existing components without JSON editing.
The [LTX 2.5 distilled Q8 example](ltx25_distilled_q8_t2v.json) also documents the direct format.
Replace every `/REPLACE/WITH/YOUR/MODELS/` value and supply your prompt before use. These placeholders
are portable documentation, not a runtime-ready checkpoint layout.

## Shared recipe format

- `format`: `weetodd-headless-v2`.
- `engine`: `h3`, `ltx23`, or `ltx25`; selects the shared renderer adapter.
- `candidate`: descriptive identifier used in the run record.
- `components`: engine-specific paths or registered asset references. LTX 2.5 requires transformer,
  Gemma text encoder, video VAE, audio VAE and (for two-stage generation) spatial upscaler paths.
  Q8-paged transformer/text paths identify directories; the other example paths identify files.
- `config`: the selected engine’s generation configuration. The example uses distilled 8+3 sampling,
  staged unloading and Q8 block streaming, 768×512, five seconds, 24 fps. These are a starting point,
  not measured 36 GB qualification. Optional configuration uses the engine’s defaults.
- `prompt`: the generation prompt; Studio replaces it with the clip prompt.
- `conditioning`: `{ "version": 1, "task": "t2v", "inputs": [], "audio_policy": "generated" }`
  for this example. Image/reference/control tasks require their corresponding media inputs.

Run from the repository with its Python environment:

```bash
python scripts/render_headless.py --recipe /path/to/edited-recipe.json \
  --output-directory /path/to/new-preflight --preflight-only
python scripts/render_headless.py --recipe /path/to/edited-recipe.json \
  --output-directory /path/to/new-render
```

Output directories must be new. FFmpeg must be available to the renderer (or supplied with the
recipe’s `ffmpeg` field). In Studio, **Import model recipes…** accepts the edited file. Clip
geometry, duration, seed and prompt override the corresponding recipe values. Final media preflight
is authoritative; importing a recipe does not qualify every future clip’s settings.
