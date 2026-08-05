# WeeTodd Nodes agent guide

This is a standalone ComfyUI custom-node project for MiniMax H3 on Apple Silicon through MLX.

## Scope boundary

- Use MiniMax H3, MLX, ComfyUI, and directly relevant media utilities.
- Do not import Draw Things code, documentation, binaries, recipes, or terminology.
- Do not import unrelated Phosphene UI, Pinokio, LTX, image-generation, launch, or account material.
- Treat the separately checked-out Phosphene `minimax-h3-mlx` repository as an upstream technical source only.
- Never commit model weights, outputs, caches, tokens, credentials, or machine-specific paths.

## Development rules

- Keep node imports lightweight; load MLX weights only when a graph executes.
- Preserve synchronized audio and video as a single H3 generation contract.
- Keep model state process-local and explicitly unloadable.
- Validate dimensions, duration, checkpoint paths, and task support before expensive work.
- Add a focused test for every node contract or engine behavior changed.
- Record third-party design research in `docs/ATTRIBUTION.md`; do not copy GPL code into Apache files.
- Use `.agents/skills/wee-todd-h3-mlx/SKILL.md` for H3 implementation work.

## Validation

```bash
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py
ruff check src/wee_todd_nodes tests
```

Full parity and checkpoint tests are optional and expensive. State clearly when they were not run.
