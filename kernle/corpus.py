"""Corpus ingestion pipeline — seed agent memory from repositories and documentation."""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional, Set

from kernle.dedup import strip_corpus_header
from kernle.processing import compute_content_hash

logger = logging.getLogger(__name__)

# Default file extensions for code and docs
DEFAULT_CODE_EXTENSIONS = [
    "py",
    "js",
    "ts",
    "jsx",
    "tsx",
    "go",
    "rs",
    "java",
    "rb",
    "c",
    "cpp",
    "h",
    "hpp",
    "cs",
    "swift",
    "kt",
    "scala",
    "sh",
    "bash",
    "zsh",
]
DEFAULT_DOC_EXTENSIONS = ["md", "txt", "rst"]

# Directories always excluded from traversal
ALWAYS_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


@dataclass
class CorpusResult:
    """Result of a corpus ingestion run."""

    files_scanned: int = 0
    chunks_created: int = 0
    chunks_skipped: int = 0
    raw_entries_created: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking functions
# ---------------------------------------------------------------------------


def chunk_python(content: str, file_path: str, max_chunk_size: int = 2000) -> List[dict]:
    """Chunk Python source code using AST-based splitting.

    Extracts top-level function and class definitions. Falls back to
    chunk_generic() on parse errors.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return chunk_generic(content, file_path, max_chunk_size)

    chunks: List[dict] = []
    lines = content.splitlines(keepends=True)
    covered: Set[int] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            chunk_text = "".join(lines[start:end])
            if len(chunk_text) > max_chunk_size:
                # Split large functions at logical boundaries
                sub_chunks = _split_large_block(chunk_text, node.name, "function", max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    {
                        "content": chunk_text,
                        "chunk_type": "function",
                        "chunk_name": node.name,
                    }
                )
            covered.update(range(start, end))

        elif isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            chunk_text = "".join(lines[start:end])
            if len(chunk_text) > max_chunk_size:
                sub_chunks = _split_large_block(chunk_text, node.name, "class", max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    {
                        "content": chunk_text,
                        "chunk_type": "class",
                        "chunk_name": node.name,
                    }
                )
            covered.update(range(start, end))

    # Collect module-level code not inside any function/class
    module_lines = []
    for i, line in enumerate(lines):
        if i not in covered:
            module_lines.append(line)

    module_text = "".join(module_lines).strip()
    if module_text:
        if len(module_text) > max_chunk_size:
            sub_chunks = chunk_generic(module_text, file_path, max_chunk_size)
            for sc in sub_chunks:
                sc["chunk_type"] = "module"
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                {
                    "content": module_text,
                    "chunk_type": "module",
                    "chunk_name": "module_level",
                }
            )

    return chunks


def _split_large_block(text: str, name: str, chunk_type: str, max_chunk_size: int) -> List[dict]:
    """Split a large code block into smaller chunks at blank-line boundaries."""
    parts: List[dict] = []
    current: List[str] = []
    current_size = 0
    part_idx = 0

    for line in text.splitlines(keepends=True):
        if current_size + len(line) > max_chunk_size and current:
            parts.append(
                {
                    "content": "".join(current),
                    "chunk_type": chunk_type,
                    "chunk_name": f"{name}_part{part_idx}",
                }
            )
            part_idx += 1
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line)

    if current:
        parts.append(
            {
                "content": "".join(current),
                "chunk_type": chunk_type,
                "chunk_name": f"{name}_part{part_idx}" if part_idx > 0 else name,
            }
        )

    return parts


def chunk_markdown(content: str, file_path: str, max_chunk_size: int = 2000) -> List[dict]:
    """Chunk Markdown content by heading boundaries."""
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        return chunk_generic(content, file_path, max_chunk_size)

    chunks: List[dict] = []

    # Content before first heading
    if matches[0].start() > 0:
        preamble = content[: matches[0].start()].strip()
        if preamble:
            chunks.append(
                {
                    "content": preamble,
                    "chunk_type": "section",
                    "chunk_name": "preamble",
                }
            )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        heading_text = match.group(2).strip()

        if len(section_text) > max_chunk_size:
            sub_chunks = chunk_generic(section_text, file_path, max_chunk_size)
            for j, sc in enumerate(sub_chunks):
                sc["chunk_type"] = "section"
                sc["chunk_name"] = f"{heading_text}_part{j}" if j > 0 else heading_text
            chunks.extend(sub_chunks)
        else:
            chunks.append(
                {
                    "content": section_text,
                    "chunk_type": "section",
                    "chunk_name": heading_text,
                }
            )

    return chunks


def chunk_generic(content: str, file_path: str, max_chunk_size: int = 2000) -> List[dict]:
    """Chunk content by paragraph boundaries (double newlines)."""
    paragraphs = re.split(r"\n\s*\n", content)
    chunks: List[dict] = []
    current_parts: List[str] = []
    current_size = 0
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_size + len(para) + 2 > max_chunk_size and current_parts:
            chunks.append(
                {
                    "content": "\n\n".join(current_parts),
                    "chunk_type": "paragraph",
                    "chunk_name": f"chunk_{chunk_idx}",
                }
            )
            chunk_idx += 1
            current_parts = []
            current_size = 0

        # If a single paragraph exceeds max_chunk_size, include it as-is
        if len(para) > max_chunk_size and not current_parts:
            chunks.append(
                {
                    "content": para,
                    "chunk_type": "paragraph",
                    "chunk_name": f"chunk_{chunk_idx}",
                }
            )
            chunk_idx += 1
            continue

        current_parts.append(para)
        current_size += len(para) + 2  # +2 for the "\n\n" separator

    if current_parts:
        chunks.append(
            {
                "content": "\n\n".join(current_parts),
                "chunk_type": "paragraph",
                "chunk_name": f"chunk_{chunk_idx}",
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# CorpusIngestor
# ---------------------------------------------------------------------------


class CorpusIngestor:
    """Ingest source code and documentation into Kernle's raw memory layer."""

    def __init__(self, kernle_instance):
        """Initialize with a Kernle instance.

        Args:
            kernle_instance: A Kernle instance used for raw entry creation.
        """
        self._k = kernle_instance
        self._seen_hashes: Set[str] = set()

    def ingest_repo(
        self,
        path: str,
        *,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_chunk_size: int = 2000,
        dry_run: bool = False,
    ) -> CorpusResult:
        """Ingest source code files from a repository path.

        Args:
            path: Path to the repository root.
            extensions: File extensions to include (without dots). Defaults to common code extensions.
            exclude: Glob patterns to exclude.
            max_chunk_size: Maximum chunk size in characters.
            dry_run: If True, preview without creating entries.

        Returns:
            CorpusResult with ingestion statistics.
        """
        exts = extensions or DEFAULT_CODE_EXTENSIONS
        return self._ingest(path, exts, exclude, max_chunk_size, dry_run, corpus_type="repo")

    def ingest_docs(
        self,
        path: str,
        *,
        extensions: Optional[List[str]] = None,
        max_chunk_size: int = 2000,
        dry_run: bool = False,
    ) -> CorpusResult:
        """Ingest documentation files.

        Args:
            path: Path to the docs directory.
            extensions: File extensions to include (without dots). Defaults to md, txt, rst.
            max_chunk_size: Maximum chunk size in characters.
            dry_run: If True, preview without creating entries.

        Returns:
            CorpusResult with ingestion statistics.
        """
        exts = extensions or DEFAULT_DOC_EXTENSIONS
        return self._ingest(path, exts, None, max_chunk_size, dry_run, corpus_type="docs")

    def get_status(self) -> dict:
        """Return counts of corpus raw entries.

        Note: scans up to 100k raw entries. Stacks exceeding this limit
        may report incomplete counts (a warning is logged).

        Returns:
            Dict with corpus entry counts.
        """
        corpus_entries = _collect_all_corpus_entries(self._k._storage)
        return {
            "total_corpus_entries": len(corpus_entries),
            "repo_entries": sum(1 for e in corpus_entries if "[corpus:repo]" in (e.blob or "")),
            "docs_entries": sum(1 for e in corpus_entries if "[corpus:docs]" in (e.blob or "")),
        }

    def _ingest(
        self,
        path: str,
        extensions: List[str],
        exclude: Optional[List[str]],
        max_chunk_size: int,
        dry_run: bool,
        corpus_type: str,
    ) -> CorpusResult:
        """Core ingestion logic shared by ingest_repo and ingest_docs."""
        result = CorpusResult()
        root = Path(path).resolve()

        if not root.is_dir():
            result.errors.append(f"Path is not a directory: {path}")
            return result

        # Build existing hash set for dedup
        self._load_existing_hashes()

        ext_set = {"." + e.lstrip(".") for e in extensions}
        exclude_patterns = exclude or []

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden and excluded directories
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in ALWAYS_EXCLUDE_DIRS
                and not any(fnmatch(d, p) for p in ALWAYS_EXCLUDE_DIRS)
            ]

            for filename in filenames:
                filepath = Path(dirpath) / filename
                rel_path = str(filepath.relative_to(root))

                # Extension filter
                if filepath.suffix not in ext_set:
                    continue

                # Exclude patterns
                if any(fnmatch(rel_path, pat) for pat in exclude_patterns):
                    continue

                result.files_scanned += 1

                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    result.errors.append(f"Failed to read {rel_path}: {e}")
                    continue

                if not content.strip():
                    continue

                # Select chunker
                chunks = self._chunk_file(content, rel_path, filepath.suffix, max_chunk_size)

                for chunk_info in chunks:
                    content_hash = compute_content_hash(chunk_info["content"])

                    if content_hash in self._seen_hashes:
                        result.chunks_skipped += 1
                        continue

                    self._seen_hashes.add(content_hash)
                    result.chunks_created += 1

                    if not dry_run:
                        entry_id = self._create_raw_entry(
                            chunk_info["content"], rel_path, chunk_info, corpus_type
                        )
                        if entry_id:
                            result.raw_entries_created += 1
                        else:
                            result.errors.append(
                                f"Failed to create entry for {rel_path}:{chunk_info['chunk_name']}"
                            )

        return result

    def _chunk_file(
        self, content: str, rel_path: str, suffix: str, max_chunk_size: int
    ) -> List[dict]:
        """Select and run the appropriate chunker for a file."""
        if suffix == ".py":
            return chunk_python(content, rel_path, max_chunk_size)
        elif suffix in (".md", ".rst"):
            return chunk_markdown(content, rel_path, max_chunk_size)
        else:
            return chunk_generic(content, rel_path, max_chunk_size)

    def _create_raw_entry(
        self, content: str, file_path: str, chunk_info: dict, corpus_type: str
    ) -> Optional[str]:
        """Create a raw entry with corpus metadata header in blob.

        Format: [corpus:{type}] [file:{path}] [chunk:{chunk_type}:{chunk_name}]
        followed by the content.
        """
        chunk_type = chunk_info.get("chunk_type", "unknown")
        chunk_name = chunk_info.get("chunk_name", "unknown")
        header = f"[corpus:{corpus_type}] [file:{file_path}] [chunk:{chunk_type}:{chunk_name}]"
        blob = f"{header}\n{content}"

        try:
            return self._k.raw(blob, source="corpus")
        except Exception as e:
            logger.error("Failed to create corpus raw entry: %s", e, exc_info=True)
            return None

    def _load_existing_hashes(self) -> None:
        """Load content hashes from all existing corpus entries for dedup.

        Uses the same full-pagination path as ``_collect_all_corpus_entries``
        to avoid silent truncation on large stacks.
        """
        try:
            corpus_entries = _collect_all_corpus_entries(self._k._storage)
            for entry in corpus_entries:
                text = getattr(entry, "blob", None) or getattr(entry, "content", None) or ""
                text = strip_corpus_header(text)
                h = compute_content_hash(text)
                if h:
                    self._seen_hashes.add(h)
            logger.debug(
                "Loaded %d corpus hashes from %d corpus entries",
                len(self._seen_hashes),
                len(corpus_entries),
            )
        except Exception as e:
            logger.warning("Could not load existing corpus hashes: %s", e, exc_info=True)


_CORPUS_BATCH_SIZE = 10000


def _collect_all_corpus_entries(storage) -> list:
    """Paginate through all raw entries and filter to corpus entries.

    Uses OFFSET pagination with deterministic ordering (captured_at DESC, id DESC).
    Correctness depends on no concurrent writes during scan — acceptable for
    corpus ingestion which serializes its own writes.
    """
    corpus = []
    offset = 0
    total_scanned = 0
    while True:
        batch = storage.list_raw(limit=_CORPUS_BATCH_SIZE, offset=offset)
        total_scanned += len(batch)
        corpus.extend(e for e in batch if _is_corpus_entry(e))
        if len(batch) < _CORPUS_BATCH_SIZE:
            break
        offset += _CORPUS_BATCH_SIZE
    logger.debug(
        "Corpus scan: %d scanned, %d corpus entries, %d batches",
        total_scanned,
        len(corpus),
        (offset // _CORPUS_BATCH_SIZE) + 1,
    )
    return corpus


def _is_corpus_entry(entry) -> bool:
    """Check if a raw entry is a corpus entry by its blob header."""
    blob = getattr(entry, "blob", None) or ""
    return blob.startswith("[corpus:")


def _extract_corpus_content(blob: str) -> str:
    """Extract the content portion of a corpus blob (after the header line)."""
    stripped = strip_corpus_header(blob)
    # strip_corpus_header returns original text when no header found or
    # no newline in header-only blobs — treat both as empty content
    return "" if stripped is blob else stripped
