# WeeTodd Nodes agent guide

This is a standalone ComfyUI custom-node project for MiniMax H3 and LTX 2.3 on Apple Silicon
through MLX.

## Scope boundary

- Use MiniMax H3, LTX 2.3, MLX, ComfyUI, and directly relevant media utilities.
- Do not import unrelated UI, launch, account, or image-generation functionality.
- Treat third-party implementations as research references only; independently implement and test project behavior.
- Never commit model weights, outputs, caches, tokens, credentials, or machine-specific paths.

## Development rules

- Keep node imports lightweight; load MLX weights only when a graph executes.
- Keep the H3 and LTX 2.3 engines isolated behind separate ComfyUI adapters.
- Preserve synchronized audio and video as a single H3 generation contract.
- Keep model state process-local and explicitly unloadable.
- Default weighted stages to staged unloading: Qwen3-VL, transformer, video VAE, then audio VAE.
  Keep a component warm only through an explicit node control and report its resident state.
- Release the active component after success, failure, or cancellation when staged unloading is
  selected. Do not load the next weighted stage before the prior stage is releasable.
- Validate dimensions, duration, checkpoint paths, and task support before expensive work.
- Add a focused test for every node contract or engine behavior changed.
- Keep local research, attribution, and knowledge-store material outside the tracked repository.
- Do not copy incompatible or unlicensed third-party code into Apache-2.0 files.
- Use `.agents/skills/wee-todd-h3-mlx/SKILL.md` for H3 implementation work.
- Before any pip, Python, venv, MLX, or dependency change, use
  `.agents/skills/python-environment-preflight/SKILL.md`. Run its preflight before mutation.
- Before risky edits, use `python3 scripts/create_source_backup.py --name <short-name>`.
  Do not pass the repository root to a generic recursive snapshot tool: ignored local state may be
  many gigabytes. The source-backup command uses Git's tracked/unignored file set, applies the
  protected prefixes in `.source-backupignore`, writes outside the checkout by default, and rejects
  unexpectedly large inputs before copying.
- Before every commit, use `.agents/skills/readme-workflow-commit-gate/SKILL.md`. Audit the complete
  `README.md` and every shipped UI/API workflow, then record the review for the exact staged
  snapshot. Do not commit when the gate or its local pre-commit hook fails.

## Validation

```bash
python .agents/skills/python-environment-preflight/scripts/preflight.py \
  --project . --python python --require-architecture arm64
python -m compileall -q src __init__.py
python -m pytest -q tests/test_nodes.py tests/test_runtime.py tests/test_readme.py \
  tests/test_workflows.py
ruff check src/wee_todd_nodes tests
./.venv/bin/python .agents/skills/readme-workflow-commit-gate/scripts/audit.py \
  --project . --record-review \
  --confirm-readme-reviewed --confirm-workflows-reviewed
```

Full parity and checkpoint tests are optional and expensive. State clearly when they were not run.
