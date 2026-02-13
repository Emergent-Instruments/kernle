"""Handlers for corpus seeding tools: seed_repo, seed_docs, seed_status."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from kernle.core import Kernle
from kernle.mcp.sanitize import sanitize_string

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _sanitize_string_list(value: Any) -> Optional[list]:
    """Coerce value to a list of strings, or None if invalid/empty."""
    if value is None:
        return None
    if isinstance(value, str):
        # Common tool-call mistake: string instead of array — split on comma
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return None


_SEED_ROOT_ENV = "KERNLE_MCP_SEED_ROOT"
_SEED_ALLOW_UNSAFE_ENV = "KERNLE_MCP_ALLOW_UNSAFE_SEED_PATHS"


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_seed_root() -> Path:
    configured_root = os.environ.get(_SEED_ROOT_ENV)
    root = Path(configured_root).expanduser() if configured_root else Path.cwd()
    return root.resolve()


def _is_path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sanitize_seed_path(path_value: Any) -> str:
    path = sanitize_string(path_value, "path", 1000, required=True)
    resolved_path = Path(path).expanduser().resolve()

    if _is_truthy(os.environ.get(_SEED_ALLOW_UNSAFE_ENV)):
        return str(resolved_path)

    seed_root = _resolve_seed_root()
    if not _is_path_within_root(resolved_path, seed_root):
        raise ValueError(
            f"path '{resolved_path}' is outside allowed seed root '{seed_root}'. "
            f"Set {_SEED_ALLOW_UNSAFE_ENV}=1 to opt out."
        )

    return str(resolved_path)


def validate_memory_seed_repo(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["path"] = _sanitize_seed_path(arguments.get("path"))
    sanitized["extensions"] = _sanitize_string_list(arguments.get("extensions"))
    sanitized["exclude"] = _sanitize_string_list(arguments.get("exclude"))
    sanitized["max_chunk_size"] = arguments.get("max_chunk_size", 2000)
    if not isinstance(sanitized["max_chunk_size"], int):
        sanitized["max_chunk_size"] = 2000
    sanitized["dry_run"] = arguments.get("dry_run", False)
    if not isinstance(sanitized["dry_run"], bool):
        sanitized["dry_run"] = False
    return sanitized


def validate_memory_seed_docs(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["path"] = _sanitize_seed_path(arguments.get("path"))
    sanitized["extensions"] = _sanitize_string_list(arguments.get("extensions"))
    sanitized["max_chunk_size"] = arguments.get("max_chunk_size", 2000)
    if not isinstance(sanitized["max_chunk_size"], int):
        sanitized["max_chunk_size"] = 2000
    sanitized["dry_run"] = arguments.get("dry_run", False)
    if not isinstance(sanitized["dry_run"], bool):
        sanitized["dry_run"] = False
    return sanitized


def validate_memory_seed_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_seed_repo(args: Dict[str, Any], k: Kernle) -> str:
    from kernle.corpus import CorpusIngestor

    ingestor = CorpusIngestor(k)
    result = ingestor.ingest_repo(
        args["path"],
        extensions=args.get("extensions"),
        exclude=args.get("exclude"),
        max_chunk_size=args.get("max_chunk_size", 2000),
        dry_run=args.get("dry_run", False),
    )

    mode = " (dry run)" if args.get("dry_run") else ""
    lines = [f"Corpus ingestion complete{mode}:"]
    lines.append(f"  Files scanned: {result.files_scanned}")
    lines.append(f"  Chunks created: {result.chunks_created}")
    lines.append(f"  Chunks skipped: {result.chunks_skipped} (dedup)")
    lines.append(f"  Raw entries created: {result.raw_entries_created}")
    if result.errors:
        lines.append(f"  Errors: {len(result.errors)}")
        for err in result.errors[:5]:
            lines.append(f"    - {err}")
    return "\n".join(lines)


def handle_memory_seed_docs(args: Dict[str, Any], k: Kernle) -> str:
    from kernle.corpus import CorpusIngestor

    ingestor = CorpusIngestor(k)
    result = ingestor.ingest_docs(
        args["path"],
        extensions=args.get("extensions"),
        max_chunk_size=args.get("max_chunk_size", 2000),
        dry_run=args.get("dry_run", False),
    )

    mode = " (dry run)" if args.get("dry_run") else ""
    lines = [f"Docs ingestion complete{mode}:"]
    lines.append(f"  Files scanned: {result.files_scanned}")
    lines.append(f"  Chunks created: {result.chunks_created}")
    lines.append(f"  Chunks skipped: {result.chunks_skipped} (dedup)")
    lines.append(f"  Raw entries created: {result.raw_entries_created}")
    if result.errors:
        lines.append(f"  Errors: {len(result.errors)}")
        for err in result.errors[:5]:
            lines.append(f"    - {err}")
    return "\n".join(lines)


def handle_memory_seed_status(args: Dict[str, Any], k: Kernle) -> str:
    from kernle.corpus import CorpusIngestor

    ingestor = CorpusIngestor(k)
    status = ingestor.get_status()

    lines = ["Corpus Ingestion Status"]
    lines.append(f"  Total corpus entries: {status['total_corpus_entries']}")
    lines.append(f"  Repo entries: {status['repo_entries']}")
    lines.append(f"  Docs entries: {status['docs_entries']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_seed_repo": handle_memory_seed_repo,
    "memory_seed_docs": handle_memory_seed_docs,
    "memory_seed_status": handle_memory_seed_status,
}

VALIDATORS = {
    "memory_seed_repo": validate_memory_seed_repo,
    "memory_seed_docs": validate_memory_seed_docs,
    "memory_seed_status": validate_memory_seed_status,
}
