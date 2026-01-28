"""Raw entry commands for Kernle CLI."""

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def resolve_raw_id(k: "Kernle", partial_id: str) -> str:
    """Resolve a partial raw entry ID to full ID.

    Tries exact match first, then prefix match.
    Returns full ID or raises ValueError if not found or ambiguous.
    """
    # First try exact match
    entry = k.get_raw(partial_id)
    if entry:
        return partial_id

    # Try prefix match by listing all entries
    entries = k.list_raw(limit=1000)  # Get enough to search
    matches = [e for e in entries if e["id"].startswith(partial_id)]

    if len(matches) == 0:
        raise ValueError(f"Raw entry '{partial_id}' not found")
    elif len(matches) == 1:
        return matches[0]["id"]
    else:
        # Multiple matches - show them
        match_ids = [m["id"][:12] for m in matches[:5]]
        suffix = "..." if len(matches) > 5 else ""
        raise ValueError(f"Ambiguous ID '{partial_id}' matches {len(matches)} entries: {', '.join(match_ids)}{suffix}")


def cmd_raw(args, k: "Kernle"):
    """Handle raw entry subcommands."""
    if args.raw_action == "capture" or args.raw_action is None:
        # Default action: capture a raw entry
        content = validate_input(args.content, "content", 5000)
        tags = [validate_input(t, "tag", 100) for t in (args.tags.split(",") if args.tags else [])]
        tags = [t.strip() for t in tags if t.strip()]
        source = getattr(args, 'source', None) or "cli"

        raw_id = k.raw(content, tags=tags if tags else None, source=source)
        print(f"✓ Raw entry captured: {raw_id[:8]}...")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
        if source and source != "cli":
            print(f"  Source: {source}")

    elif args.raw_action == "list":
        # Filter by processed state
        processed = None
        if args.unprocessed:
            processed = False
        elif args.processed:
            processed = True

        entries = k.list_raw(processed=processed, limit=args.limit)

        if not entries:
            print("No raw entries found.")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            unprocessed_count = sum(1 for e in entries if not e["processed"])
            print(f"Raw Entries ({len(entries)} total, {unprocessed_count} unprocessed)")
            print("=" * 50)
            for e in entries:
                status = "✓" if e["processed"] else "○"
                timestamp = e["timestamp"][:16] if e["timestamp"] else "unknown"
                content_preview = e["content"][:60].replace("\n", " ")
                if len(e["content"]) > 60:
                    content_preview += "..."
                print(f"\n{status} [{e['id'][:8]}] {timestamp}")
                print(f"  {content_preview}")
                if e["tags"]:
                    print(f"  Tags: {', '.join(e['tags'])}")
                if e["processed"] and e["processed_into"]:
                    print(f"  → {', '.join(e['processed_into'])}")

    elif args.raw_action == "show":
        try:
            full_id = resolve_raw_id(k, args.id)
        except ValueError as e:
            print(f"✗ {e}")
            return

        entry = k.get_raw(full_id)
        if not entry:
            print(f"Raw entry {args.id} not found.")
            return

        if args.json:
            print(json.dumps(entry, indent=2, default=str))
        else:
            status = "✓ Processed" if entry["processed"] else "○ Unprocessed"
            print(f"Raw Entry: {entry['id']}")
            print(f"Status: {status}")
            print(f"Timestamp: {entry['timestamp']}")
            print(f"Source: {entry['source']}")
            if entry["tags"]:
                print(f"Tags: {', '.join(entry['tags'])}")
            print()
            print("Content:")
            print("-" * 40)
            print(entry["content"])
            print("-" * 40)
            if entry["processed_into"]:
                print(f"\nProcessed into: {', '.join(entry['processed_into'])}")

    elif args.raw_action == "process":
        # Support batch processing with comma-separated IDs
        raw_ids = [id.strip() for id in args.id.split(",") if id.strip()]
        
        success_count = 0
        for raw_id in raw_ids:
            try:
                full_id = resolve_raw_id(k, raw_id)
                memory_id = k.process_raw(
                    raw_id=full_id,
                    as_type=args.type,
                    objective=args.objective,
                    outcome=args.outcome,
                )
                print(f"✓ Processed {full_id[:8]}... → {args.type}:{memory_id[:8]}...")
                success_count += 1
            except ValueError as e:
                print(f"✗ {raw_id}: {e}")
        
        if len(raw_ids) > 1:
            print(f"\nProcessed {success_count}/{len(raw_ids)} entries")

    elif args.raw_action == "review":
        # Guided review of unprocessed entries
        entries = k.list_raw(processed=False, limit=args.limit)

        if not entries:
            print("✓ No unprocessed raw entries - memory is up to date!")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
            return

        print("## Raw Entry Review")
        print(f"Found {len(entries)} unprocessed entries to review.\n")
        print("For each entry, consider:")
        print("  - **Episode**: Significant experience with a lesson learned")
        print("  - **Note**: Important observation, decision, or fact")
        print("  - **Belief**: Pattern or principle you've discovered")
        print("  - **Skip**: Keep as raw (not everything needs promotion)")
        print()
        print("=" * 60)

        for i, e in enumerate(entries, 1):
            timestamp = e["timestamp"][:16] if e["timestamp"] else "unknown"
            print(f"\n[{i}/{len(entries)}] {timestamp} - ID: {e['id'][:8]}")
            print("-" * 40)
            print(e["content"])
            print("-" * 40)
            if e["tags"]:
                print(f"Tags: {', '.join(e['tags'])}")

            # Provide promotion suggestions based on content
            content_lower = e["content"].lower()
            suggestions = []
            if any(word in content_lower for word in ["learned", "lesson", "realized", "discovered"]):
                suggestions.append("episode (contains learning)")
            if any(word in content_lower for word in ["decided", "decision", "chose", "will"]):
                suggestions.append("note (contains decision)")
            if any(word in content_lower for word in ["always", "never", "should", "principle", "pattern"]):
                suggestions.append("belief (contains principle)")

            if suggestions:
                print(f"💡 Suggestions: {', '.join(suggestions)}")

            print(f"\nTo promote: kernle -a {k.agent_id} raw process {e['id'][:8]} --type <episode|note|belief>")

        print("\n" + "=" * 60)
        print(f"\nReviewed {len(entries)} entries. Promote the meaningful ones, skip the rest.")

    elif args.raw_action == "clean":
        # Clean up old unprocessed raw entries
        age_days = getattr(args, 'age', 7) or 7
        junk_mode = getattr(args, 'junk', False)
        dry_run = not getattr(args, 'confirm', False)
        
        entries = k.list_raw(processed=False, limit=500)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=age_days)
        
        # Junk detection patterns
        junk_keywords = ["test", "testing", "list", "show me", "show", "help", "hi", "hello", 
                        "asdf", "aaa", "xxx", "foo", "bar", "baz", "123", "abc"]
        
        def is_junk(entry):
            """Detect likely junk entries."""
            content = entry.get("content", "").strip().lower()
            # Very short content
            if len(content) < 10:
                return True
            # Exact match to junk keywords
            if content in junk_keywords:
                return True
            # Starts with test-like patterns
            if content.startswith(("test ", "testing ")):
                return True
            return False
        
        stale_entries = []
        junk_entries = []
        
        for entry in entries:
            # Check for junk first if junk mode
            if junk_mode and is_junk(entry):
                junk_entries.append(entry)
                continue
                
            # Check age for stale entries
            if not junk_mode:
                try:
                    ts = entry.get("timestamp", "")
                    if ts:
                        entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if entry_time < cutoff:
                            stale_entries.append(entry)
                except (ValueError, TypeError):
                    continue
        
        target_entries = junk_entries if junk_mode else stale_entries
        label = "junk" if junk_mode else f"older than {age_days} days"
        
        if not target_entries:
            print(f"✓ No unprocessed raw entries detected as {label}.")
            return
        
        print(f"Found {len(target_entries)} entries ({label}):\n")
        
        for entry in target_entries[:15]:  # Show max 15
            timestamp = entry["timestamp"][:10] if entry["timestamp"] else "unknown"
            content_preview = entry["content"][:50].replace("\n", " ")
            if len(entry["content"]) > 50:
                content_preview += "..."
            print(f"  [{entry['id'][:8]}] {timestamp}: {content_preview}")
        
        if len(target_entries) > 15:
            print(f"  ... and {len(target_entries) - 15} more")
        
        if dry_run:
            print(f"\n⚠ DRY RUN: Would delete {len(target_entries)} entries.")
            if junk_mode:
                print(f"  To actually delete, run: kernle raw clean --junk --confirm")
            else:
                print(f"  To actually delete, run: kernle raw clean --age {age_days} --confirm")
        else:
            deleted = 0
            for entry in target_entries:
                try:
                    k._storage.delete_raw(entry["id"])
                    deleted += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {entry['id'][:8]}: {e}")
            print(f"\n✓ Deleted {deleted} {label} raw entries.")

    elif args.raw_action == "promote":
        # Alias for process - simpler UX
        args.raw_action = "process"
        # Fall through to process handler would require refactor, so duplicate minimal logic
        try:
            full_id = resolve_raw_id(k, args.id)
        except ValueError as e:
            print(f"✗ {e}")
            return

        entry = k.get_raw(full_id)
        if not entry:
            print(f"✗ Raw entry {args.id} not found.")
            return

        target_type = args.type
        content = entry["content"]

        if target_type == "episode":
            objective = args.objective or content[:100]
            outcome = args.outcome or "Promoted from raw capture"
            result_id = k.episode(objective=objective, outcome=outcome, tags=["promoted"])
            print(f"✓ Promoted to episode: {result_id[:8]}...")
        elif target_type == "note":
            result_id = k.note(content=content, type="note", tags=["promoted"])
            print(f"✓ Promoted to note: {result_id[:8]}...")
        elif target_type == "belief":
            result_id = k.belief(statement=content, confidence=0.7)
            print(f"✓ Promoted to belief: {result_id[:8]}...")

        # Mark as processed
        k._storage.mark_raw_processed(full_id, [f"{target_type}:{result_id}"])
        print(f"  Raw entry marked as processed.")

    elif args.raw_action == "triage":
        # Guided triage of unprocessed entries
        limit = getattr(args, 'limit', 10)
        entries = k.list_raw(processed=False, limit=limit)
        
        if not entries:
            print("✓ No unprocessed raw entries to triage.")
            return
        
        print(f"Raw Entry Triage ({len(entries)} entries)")
        print("=" * 50)
        print()
        print("Suggestions: [E]pisode | [N]ote | [B]elief | [D]elete | [S]kip")
        print()
        
        for entry in entries:
            content = entry["content"]
            timestamp = entry["timestamp"][:16] if entry["timestamp"] else "unknown"
            
            # Auto-suggest based on content analysis
            suggestion = "S"  # default skip
            content_lower = content.lower()
            
            # Junk detection
            if len(content.strip()) < 10 or content_lower in ["test", "list", "show", "help"]:
                suggestion = "D"
            # Session summaries / work logs → Episode
            elif any(x in content_lower for x in ["session", "completed", "shipped", "implemented", "built", "fixed"]):
                suggestion = "E"
            # Insights / decisions → Note
            elif any(x in content_lower for x in ["insight", "decision", "realized", "learned", "important"]):
                suggestion = "N"
            # Beliefs / observations about the world
            elif any(x in content_lower for x in ["believe", "think that", "seems like", "pattern"]):
                suggestion = "B"
            
            suggestion_labels = {"E": "Episode", "N": "Note", "B": "Belief", "D": "Delete", "S": "Skip"}
            
            print(f"[{entry['id'][:8]}] {timestamp}")
            print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")
            print(f"  → Suggested: {suggestion_labels[suggestion]}")
            print()
            print(f"  To act: kernle raw promote {entry['id'][:8]} --type <episode|note|belief>")
            print(f"          kernle raw clean --junk --confirm  (to delete junk)")
            print("-" * 50)

    elif args.raw_action == "files":
        # Show flat file locations
        raw_dir = k._storage.get_raw_dir()
        files = k._storage.get_raw_files()
        
        print(f"Raw Flat Files Directory: {raw_dir}")
        print("=" * 50)
        
        if not files:
            print("\nNo raw files yet. Capture something with: kernle raw \"thought\"")
        else:
            print(f"\nFiles ({len(files)} total):")
            total_size = 0
            for f in files[:10]:
                size = f.stat().st_size
                total_size += size
                print(f"  {f.name:20} {size:>6} bytes")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
            print(f"\nTotal: {total_size:,} bytes")
        
        print(f"\n💡 Tips:")
        print(f"  • Edit directly: vim {raw_dir}/<date>.md")
        print(f"  • Search: grep -r 'pattern' {raw_dir}/")
        print(f"  • Git track: cd {raw_dir.parent} && git init")
        
        if getattr(args, 'open', False):
            import subprocess
            subprocess.run(["open", str(raw_dir)], check=False)

    elif args.raw_action == "sync":
        # Sync from flat files to SQLite
        dry_run = getattr(args, 'dry_run', False)
        
        if dry_run:
            print("DRY RUN: Scanning flat files for unindexed entries...")
        else:
            print("Syncing flat files to SQLite index...")
        
        result = k._storage.sync_raw_from_files()
        
        print(f"\nFiles processed: {result['files_processed']}")
        print(f"Entries imported: {result['imported']}")
        print(f"Entries skipped (already indexed): {result['skipped']}")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result['errors'][:5]:
                print(f"  • {err}")
            if len(result['errors']) > 5:
                print(f"  ... and {len(result['errors']) - 5} more")
        
        if result['imported'] > 0:
            print(f"\n✓ Imported {result['imported']} entries from flat files")
        elif result['skipped'] > 0:
            print("\n✓ All entries already indexed")
        else:
            print("\n✓ No entries to import")
