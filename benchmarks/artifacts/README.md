# Local benchmark artifacts

This directory is the repository-local entry point for WeeTodd benchmark evidence.

Tracked charts remain under `charts/`. Generated MP4 files, JSON sidecars, extracted frames, and
contact sheets remain under `media/`, which Git ignores. This separation keeps benchmark evidence
available locally without committing generated media or machine-specific paths.

The current local bundle contains the 384P-class 640 by 384 EasyCache matrix and the native 768P
1344 by 768 EasyCache matrix. Each matrix covers no cache, conservative, balanced, and speed at 8,
12, 16, and 20 requested steps.

Canonical reports:

- [640 by 384 policy scaling](../../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING.md)
- [1344 by 768 policy scaling](../../docs/reference/H3_EASYCACHE_POLICY_STEP_SCALING_768P.md)

Raw measurements:

- [640 by 384 CSV](../h3_easycache_policy_step_scaling.csv)
- [1344 by 768 CSV](../h3_easycache_policy_step_scaling_768p.csv)
