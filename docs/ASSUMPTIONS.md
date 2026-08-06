# Assumption log

## 2026-08-05 — Automatic H3 EasyCache

Assumption: Generation parameters alone can determine a safe H3 residual-reuse threshold.

Unknown: No validated quality curve maps duration, resolution, and step count to a safe threshold.

Decision: Check now. Use the step count only for calibration and skip caps. Derive the bounded
threshold from the live joint video/audio trajectory.

Result: Automatic mode protects two calibration evaluations and the final evaluation, prevents
consecutive skips, and reports the resolved threshold.

Lesson: Treat automatic cache selection as a measured runtime heuristic, not a static quality
formula.

## 2026-08-05 — Speed-first automatic H3 EasyCache

Assumption: Two calibration evaluations, final-step protection, no more than two consecutive
reuses, and a 50-percent skip ceiling provide a useful aggressive policy for an initial H3 speed
comparison. This policy can change audiovisual detail and must remain explicitly user-selected.

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
