# Documentation standard

WeeTodd separates narrative project documentation from its portable knowledge bundle.

## Locations

- `README.md` is the user-facing project entry point.
- `docs/` holds canonical architecture, audit, attribution, research, and roadmap documents.
- `docs/reference/` holds detailed evidence and experimental records.
- `knowledge/` is the OKF v0.2 bundle for durable, agent-facing concepts and navigation.
- `.agents/skills/` holds concise procedural instructions, not duplicate project documentation.

An OKF concept may summarize a canonical document and point to it with `resource`; it must not grow
into a second competing source of truth. Update the canonical document first when technical details
change, then refresh the concept and its `generated.at` value.

## Markdown rules

- Use UTF-8 with LF line endings and a final newline.
- Do not leave trailing whitespace or tab indentation in prose.
- Use ATX headings (`#`) with one space after the marker and do not skip heading levels.
- Put blank lines around headings, lists, tables, and fenced code blocks.
- Use descriptive link labels and relative repository links for local documents.
- Specify a language on fenced code blocks when one applies.
- Keep claims close to their citations and record third-party design sources in
  `docs/ATTRIBUTION.md`.
- Do not place credentials, private information, machine-specific paths, model weights, or generated
  media in documentation or knowledge.

## Controlled technical English profile

Use a selective controlled-English profile. This profile is inspired by Simplified Technical
English, but it is not a claim of ASD-STE100 compliance.

Apply these rules to all project writing:

- Use one preferred term for each concept.
- Define an abbreviation at its first meaningful use.
- Use direct sentences and identify the actor when the actor matters.
- Put a condition before the action that depends on it.
- Replace unclear pronouns such as “it” or “this” with the specific noun when ambiguity is possible.
- Keep exact MiniMax H3, MLX, Metal, Python, and ComfyUI terminology when a simpler word would change
  the technical meaning.
- Do not use promotional claims or unsupported adjectives.
- Use `MUST`, `SHOULD`, and `MAY` only for requirements, recommendations, and permitted options,
  respectively.

### Strict surfaces

Use the controlled profile strictly for installation instructions, procedures, validation steps,
workflow examples, node labels and descriptions, tooltips, warnings, and error messages.

- Start an instruction with an imperative verb.
- Put one main action in each numbered step.
- State prerequisites before the action.
- State the expected result when the result is not obvious.
- Make an error message identify the subject, the problem, and a corrective action when one exists.
- Prefer sentences of 25 words or fewer. Identifiers, commands, links, and necessary technical terms
  are exceptions.

### Flexible surfaces

Use the profile flexibly for architecture notes, audits, research, benchmarks, and design analysis.
Preserve causal detail, uncertainty, equations, and exact source terminology. Longer sentences are
acceptable when splitting them would reduce precision. Keep the claim, evidence, limitation, and
inference distinguishable.

### Controlled terminology

| Preferred term | Meaning |
| --- | --- |
| checkpoint | Model component files stored on disk. |
| model specification | Immutable information that describes how to find and load a checkpoint. |
| pipeline | Process-local loaded components used for inference. |
| generation | One request that produces synchronized video and audio. |
| frame | A decoded video image. Use `latent frame` for a temporal latent position. |
| sampling step | One scheduler interval. State transformer evaluation count separately when it differs. |

Review controlled-language rules manually. `lint_docs.py` enforces mechanical Markdown rules only;
automated vocabulary restrictions would produce false positives for ML and framework terminology.

## OKF profile

The knowledge store follows Google Open Knowledge Format v0.2. Every concept file has YAML
frontmatter and a non-empty `type`. Root and directory `index.md` files provide progressive
disclosure; `log.md` files use descending ISO `YYYY-MM-DD` headings.

Apply the controlled-English profile strictly to every OKF concept because concepts are retrieval
units. Keep the canonical research document flexible when technical nuance requires it.

- Give each concept one retrievable subject.
- Make the title and description direct and specific.
- Separate facts, claims, evidence, limitations, decisions, and required validation with headings
  when the distinction helps retrieval.
- Put one principal claim in each paragraph.
- Use explicit nouns instead of context-dependent pronouns.
- Preserve exact source titles, quotations, equations, identifiers, and technical terms.
- Link to the canonical document for detail instead of duplicating long explanations.

Recommended concept metadata:

```yaml
---
type: Reference
title: Short display name
description: One sentence describing the concept.
resource: ../docs/canonical-document.md
tags: [minimax-h3, mlx]
status: stable
generated:
  by: process:codex-review
  at: 2026-08-05T00:00:00-07:00
sources:
  - id: stable-source-id
    resource: https://example.com/authoritative-source
    title: Authoritative source title
---
```

Omit `verified` unless the named human or deterministic process actually checked the concept
against its sources. Unknown fields are allowed and must be preserved when round-tripping.

## Validation

Run:

```bash
python scripts/validate_okf.py knowledge
python scripts/lint_docs.py
```

`validate_okf.py` enforces the OKF conformance surface and validates optional standard metadata when
present. `lint_docs.py` checks repository Markdown hygiene without rewriting files.
