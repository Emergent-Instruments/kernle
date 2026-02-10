# Golden Corpus

A curated test corpus for validating the full memory pipeline:
corpus ingestion -> processing exhaustion -> cognitive quality assertions.

## Structure

- `docs/` — Documentation files (Markdown)
- `src/` — Source code files (Python)

## Purpose

This corpus is designed to produce deterministic results when processed
with the DeterministicMockModel, enabling golden snapshot testing of
the complete memory pipeline.
