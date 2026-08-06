# WeeTodd Nodes agent guide

This is a standalone ComfyUI custom-node project for MiniMax H3 on Apple Silicon through MLX.

## Scope boundary

- Use MiniMax H3, MLX, ComfyUI, and directly relevant media utilities.
- Do not import unrelated Phosphene UI, Pinokio, LTX, image-generation, launch, or account material.
- Treat the separately checked-out Phosphene `minimax-h3-mlx` repository as an upstream technical source only.
- Never commit model weights, outputs, caches, tokens, credentials, or machine-specific paths.

## Development rules

- Keep node imports lightweight; load MLX weights only when a graph executes.
- Preserve synchronized audio and video as a single H3 generation contract.
- Keep model state process-local and explicitly unloadable.
- Default weighted stages to staged unloading: Qwen3-VL, transformer, video VAE, then audio VAE.
  Keep a component warm only through an explicit node control and report its resident state.
- Release the active component after success, failure, or cancellation when staged unloading is
  selected. Do not load the next weighted stage before the prior stage is releasable.
- Validate dimensions, duration, checkpoint paths, and task support before expensive work.
- Add a focused test for every node contract or engine behavior changed.
- Record third-party design research in `docs/ATTRIBUTION.md`; do not copy GPL code into Apache files.
- Keep durable agent-facing knowledge in the OKF v0.2 bundle under `knowledge/`; validate it after changes.
- Follow the selective controlled-English profile in `docs/DOCUMENTATION.md`: apply it strictly to
  procedures and user-facing text, and flexibly to research and architecture prose.
- Run the documentation linter for Markdown changes.
- Use `.agents/skills/wee-todd-h3-mlx/SKILL.md` for H3 implementation work.
- Before any pip, Python, venv, MLX, or dependency change, use
  `.agents/skills/python-environment-preflight/SKILL.md`. Run its preflight before mutation.

## Validation

```bash
python .agents/skills/python-environment-preflight/scripts/preflight.py \
  --project . --python python --require-architecture arm64
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py
ruff check src/wee_todd_nodes tests
python scripts/validate_okf.py knowledge
python scripts/lint_docs.py
```

Full parity and checkpoint tests are optional and expensive. State clearly when they were not run.
