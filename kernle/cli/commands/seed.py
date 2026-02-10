"""Corpus seeding commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_seed(args, k: "Kernle"):
    """Handle seed subcommands (repo, docs, status)."""
    from kernle.corpus import CorpusIngestor

    ingestor = CorpusIngestor(k)

    if args.seed_action == "repo":
        extensions = None
        if getattr(args, "extensions", None):
            extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]

        exclude = None
        if getattr(args, "exclude", None):
            exclude = [p.strip() for p in args.exclude.split(",") if p.strip()]

        result = ingestor.ingest_repo(
            args.path,
            extensions=extensions,
            exclude=exclude,
            max_chunk_size=getattr(args, "max_chunk_size", 2000),
            dry_run=getattr(args, "dry_run", False),
        )

        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "files_scanned": result.files_scanned,
                        "chunks_created": result.chunks_created,
                        "chunks_skipped": result.chunks_skipped,
                        "raw_entries_created": result.raw_entries_created,
                        "errors": result.errors,
                        "dry_run": getattr(args, "dry_run", False),
                    },
                    indent=2,
                )
            )
        else:
            mode = " (dry run)" if getattr(args, "dry_run", False) else ""
            print(f"Corpus ingestion complete{mode}:")
            print(f"  Files scanned:      {result.files_scanned}")
            print(f"  Chunks created:     {result.chunks_created}")
            print(f"  Chunks skipped:     {result.chunks_skipped} (dedup)")
            print(f"  Raw entries created: {result.raw_entries_created}")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for err in result.errors[:5]:
                    print(f"    - {err}")

    elif args.seed_action == "docs":
        extensions = None
        if getattr(args, "extensions", None):
            extensions = [e.strip() for e in args.extensions.split(",") if e.strip()]

        result = ingestor.ingest_docs(
            args.path,
            extensions=extensions,
            max_chunk_size=getattr(args, "max_chunk_size", 2000),
            dry_run=getattr(args, "dry_run", False),
        )

        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "files_scanned": result.files_scanned,
                        "chunks_created": result.chunks_created,
                        "chunks_skipped": result.chunks_skipped,
                        "raw_entries_created": result.raw_entries_created,
                        "errors": result.errors,
                        "dry_run": getattr(args, "dry_run", False),
                    },
                    indent=2,
                )
            )
        else:
            mode = " (dry run)" if getattr(args, "dry_run", False) else ""
            print(f"Docs ingestion complete{mode}:")
            print(f"  Files scanned:      {result.files_scanned}")
            print(f"  Chunks created:     {result.chunks_created}")
            print(f"  Chunks skipped:     {result.chunks_skipped} (dedup)")
            print(f"  Raw entries created: {result.raw_entries_created}")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for err in result.errors[:5]:
                    print(f"    - {err}")

    elif args.seed_action == "status":
        status = ingestor.get_status()

        if getattr(args, "json", False):
            print(json.dumps(status, indent=2))
        else:
            print("Corpus Ingestion Status")
            print("=" * 40)
            print(f"  Total corpus entries: {status['total_corpus_entries']}")
            print(f"  Repo entries:        {status['repo_entries']}")
            print(f"  Docs entries:        {status['docs_entries']}")

    else:
        print("Usage: kernle seed {repo|docs|status}")
