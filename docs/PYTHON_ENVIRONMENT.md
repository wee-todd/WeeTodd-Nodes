# Python environment policy

Run environment preflight before any package installation, virtual-environment replacement, MLX
setup, or dependency upgrade. Do not assume that an existing `.venv` satisfies the project Python
requirement.

## Preflight

1. Run the check with the intended interpreter:

   ```bash
   python .agents/skills/python-environment-preflight/scripts/preflight.py \
     --project . --python python --require-architecture arm64
   ```

2. Confirm that `compatible` is `true`.
3. Confirm that the reported architecture is `arm64` before MLX installation.
4. Confirm that the executable belongs to the intended project or ComfyUI environment.
5. Stop before installation if any confirmation fails.

The project currently requires Python 3.11 or newer. A pip upgrade cannot correct an incompatible
Python interpreter.

## Existing incompatible venv

1. Find or install a compatible Python interpreter.
2. Preserve the incompatible venv outside the repository.
3. Create a replacement venv with the selected interpreter.
4. Run preflight against the replacement interpreter.
5. Install the project with `python -m pip install -e .`.
6. Run `python -m pip check`.
7. Run the project validation suite.

Do not delete the preserved environment until rollback is unnecessary. Do not retain venv backups,
package caches, wheels, or interpreter paths in the repository.

## ComfyUI host environments

Treat the ComfyUI Python environment as host-owned. Inspect its Python and installed packages before
changing it. Do not replace the ComfyUI environment or upgrade unrelated packages without explicit
authorization. Install WeeTodd Nodes into that same compatible environment for runtime testing.

## MLX verification

After installation, import MLX and print its version without loading weights:

```bash
python -c "import mlx.core as mx; print(mx.__version__)"
```

Run `python -m pip check` after the import check. Do not download models or start generation to prove
that the Python environment is valid.
