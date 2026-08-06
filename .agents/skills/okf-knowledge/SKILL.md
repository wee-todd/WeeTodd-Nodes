---
name: okf-knowledge
description: Author, migrate, review, index, or validate durable project knowledge in the Google Open Knowledge Format (OKF) v0.2 bundle under knowledge/. Use for knowledge-store changes, research summaries, decisions, references, provenance, trust, lifecycle, cross-links, indexes, and OKF conformance.
---

# OKF Knowledge

Keep canonical prose documentation in `docs/`. Put durable, agent-facing concepts and navigation in
the `knowledge/` bundle.

## Workflow

1. Read `docs/DOCUMENTATION.md` and the relevant canonical document before editing knowledge.
2. Update an existing concept when its subject already exists; avoid parallel summaries.
3. Give every non-reserved Markdown file YAML frontmatter with a non-empty `type`.
4. Prefer `title`, one-sentence `description`, `tags`, `status`, `sources`, and `generated` when
   they improve discovery or provenance. Do not fabricate `verified` events.
5. Link related concepts with ordinary Markdown links. Use `resource` for the canonical document or
   asset the concept describes.
6. Update the nearest `index.md` and `log.md` for material additions, moves, or deprecations.
7. Preserve unknown frontmatter keys when editing an existing concept.
8. Apply the controlled-English profile strictly to concept summaries. Separate claims, evidence,
   limitations, decisions, and required validation when the distinction helps retrieval. Preserve
   exact research terms when simplified wording would lose meaning.
9. Run `python scripts/validate_okf.py knowledge` and `python scripts/lint_docs.py`.

## Boundaries

- Target OKF v0.2 as declared by `knowledge/index.md`.
- Treat `index.md` and `log.md` as reserved filenames, not concepts.
- Keep concepts narrow enough to retrieve independently.
- Record sources as facts, not subjective credibility scores.
- Use `status: deprecated` instead of deleting a concept that may have inbound links.
- Never copy private inputs, credentials, model files, generated media, or machine-specific paths.
- Use an Attested Computation only when there is a real sanctioned computation, executor receipt,
  and deterministic attester; do not use it as decoration for ordinary validation commands.
