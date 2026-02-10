"""Handlers for corpus seeding tools: seed_repo, seed_docs, seed_status."""

from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import sanitize_string

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_seed_repo(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["path"] = sanitize_string(arguments.get("path"), "path", 1000, required=True)
    sanitized["extensions"] = arguments.get("extensions")
    sanitized["exclude"] = arguments.get("exclude")
    sanitized["max_chunk_size"] = arguments.get("max_chunk_size", 2000)
    if not isinstance(sanitized["max_chunk_size"], int):
        sanitized["max_chunk_size"] = 2000
    sanitized["dry_run"] = arguments.get("dry_run", False)
    if not isinstance(sanitized["dry_run"], bool):
        sanitized["dry_run"] = False
    return sanitized


def validate_memory_seed_docs(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["path"] = sanitize_string(arguments.get("path"), "path", 1000, required=True)
    sanitized["extensions"] = arguments.get("extensions")
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
