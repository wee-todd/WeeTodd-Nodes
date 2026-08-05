# Assumption log

## Project foundation

Assumption: Phosphene's standalone `minimax-h3-mlx` subtree is the intended H3 foundation.

Unknown: Whether a different unpublished runtime was intended.

Decision: Check now.

Result: The subtree is an independent Apache-2.0 repository with complete MLX modules and parity tests; surrounding Phosphene code is primarily unrelated.

Lesson: Import the H3 engine only and retain attributed provenance.

## ComfyUI integration

Assumption: Direct MLX execution is preferable to converting full model state through PyTorch abstractions.

Unknown: Whether a future ComfyUI API will provide a native non-PyTorch lifecycle.

Decision: Build a reversible adapter boundary now.

Result: The first surface passes immutable specifications and writes synchronized media directly.

Lesson: Keep engine and host adapters separate.
