# Environment replacement policy

## Required evidence

Record these facts before installation:

- Project `requires-python` value.
- Candidate interpreter executable and version.
- Candidate machine architecture and operating system.
- Existing venv interpreter and `pyvenv.cfg` home when a venv exists.
- Pip version and editable-build support when an editable install is required.

## Incompatible environment

Do not mutate an incompatible environment in an attempt to make it compatible. A pip upgrade cannot
change the Python language version or wheel compatibility.

1. Find or install a compatible interpreter.
2. Preserve the incompatible venv outside the repository.
3. Create a new venv with the compatible interpreter.
4. Run preflight against the new venv.
5. Install the project and tools.
6. Run `python -m pip check` and project tests.

Delete the preserved environment only after the user confirms that rollback is unnecessary.

## MLX requirements

MLX requires Apple Silicon and compatible macOS wheels. Confirm `arm64`, the Python wheel version,
and the project's MLX constraint before installation. Import MLX and print its version after
installation. Do not download model weights as an environment validation step.
