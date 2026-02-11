"""Unified dedup helpers for raw content hashing.

Centralizes content-hash loading and corpus header stripping so that
corpus.py, import_cmd.py, and processing.py share a single code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kernle.processing import compute_content_hash

logger = logging.getLogger(__name__)

_DEFAULT_SCAN_LIMIT = 100_000


def strip_corpus_header(text: str) -> str:
    """Strip [corpus:...] metadata header from raw entry content.

    Corpus entries store blobs as ``[corpus:repo] [file:x.py] ...\\nactual content``.
    This returns the content after the header line, or the original text
    if no header is present.
    """
    if text.startswith("[corpus:"):
        nl = text.find("\n")
        if nl >= 0:
            return text[nl + 1 :]
    return text


@dataclass
class DedupResult:
    """Result of scanning raw entries for content hashes."""

    hashes: set
    rows_scanned: int
    rows_matched: int  # rows that passed source_filter


def load_raw_content_hashes(
    storage,
    limit: int = _DEFAULT_SCAN_LIMIT,
    source_filter: str | None = None,
) -> DedupResult:
    """Load content hashes from raw entries for dedup.

    Args:
        storage: Storage backend with ``list_raw(limit=...)``.
        limit: Max entries to scan.
        source_filter: If set, only include entries whose source matches
            (e.g. ``"corpus"`` to only scan corpus entries). ``None`` scans all.

    Returns:
        DedupResult with hashes, rows_scanned, and rows_matched so
        callers can report actionable warnings.

    Handles corpus entries by stripping ``[corpus:...]`` metadata headers.
    Uses canonical ``blob`` field with ``content`` fallback.
    """
    hashes: set = set()
    rows_scanned = 0
    rows_matched = 0

    all_raw = storage.list_raw(limit=limit)
    if len(all_raw) >= limit:
        logger.warning(
            "Raw entry scan hit %d limit — dedup may be incomplete",
            limit,
        )

    for entry in all_raw:
        rows_scanned += 1

        # Source filtering
        if source_filter:
            if source_filter == "corpus":
                # Corpus dedup: match on blob header only (preserves original
                # corpus.py behavior — source field is unreliable, see known issues)
                blob = getattr(entry, "blob", None) or ""
                if not blob.startswith("[corpus:"):
                    continue
            else:
                entry_source = getattr(entry, "source", None) or ""
                if entry_source != source_filter:
                    continue

        rows_matched += 1
        text = getattr(entry, "blob", None) or getattr(entry, "content", None) or ""
        # Strip corpus metadata header if present
        text = strip_corpus_header(text)
        h = compute_content_hash(text)
        if h:
            hashes.add(h)

    return DedupResult(hashes=hashes, rows_scanned=rows_scanned, rows_matched=rows_matched)
