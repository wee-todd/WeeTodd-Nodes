# Headless recipe examples

Studio’s **Model setup** creates validated recipes from existing components without JSON editing.
The [LTX 2.5 distilled Q8 example](ltx25_distilled_q8_t2v.json) also documents the direct format.
Replace every `/REPLACE/WITH/YOUR/MODELS/` value and supply your prompt before use. These placeholders
are portable documentation, not a runtime-ready checkpoint layout.

## Download and create an LTX 2.5 recipe

Run these commands from WeeTodd-Nodes with its compatible Python environment. Accept access to
the [prepared model repository](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged) and sign in with
`hf auth login` before downloading. Studio can instead store a read token in macOS Keychain for
its own setup downloads; that saved token is not a general shell login.

```bash
python scripts/setup_models.py download ltx25-distilled-q8-preconverted \
  --destination /path/to/shared-models
python scripts/setup_models.py scan ltx25-text \
  /path/to/shared-models/ltx25-distilled-q8-preconverted
```

The destination is a parent library folder. Setup creates the named package directory within it.
Optionally add `--existing-root /path/to/ComfyUI/models` to the download command to reuse files with
matching checksums. Keep the whole package, including all pages, manifests and notices.

Create a recipe using its five component paths:

```bash
python scripts/setup_models.py create ltx25-text \
  --component transformer_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/transformer \
  --component text_encoder_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/gemma \
  --component video_vae_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --component audio_vae_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --component spatial_upscaler_path=/path/to/shared-models/ltx25-distilled-q8-preconverted/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --profiles-directory /path/to/model-recipes --memory-mode lower_memory
```

Replace the example paths with yours; quote each entire `KEY=PATH` argument if it contains spaces.
Creation runs component/configuration preflight and returns `recipePath` for a new, uniquely named
file. Import that file into Studio, select it for an LTX 2.5 text clip, and enter the clip prompt.
For direct headless use, edit the generated recipe's `prompt` and desired `config` values, then run
the preflight/render commands below with that file. The lower-memory policy enables supported
savings; it is not a measured 36 GB fit guarantee. Reference/control tasks need their additional
media and compatible adapters; the distilled base package does not include those adapters.

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
