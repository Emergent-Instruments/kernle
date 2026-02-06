"""Summary (fractal summarization) commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_summary(args, k: "Kernle"):
    """Handle summary subcommands."""
    if args.summary_action == "write":
        scope = args.scope
        content = validate_input(args.content, "content", 10000)
        period_start = validate_input(args.period_start, "period_start", 30)
        period_end = validate_input(args.period_end, "period_end", 30)
        key_themes = args.theme if hasattr(args, "theme") and args.theme else None
        epoch_id = getattr(args, "epoch_id", None)

        try:
            summary_id = k.summary_save(
                content=content,
                scope=scope,
                period_start=period_start,
                period_end=period_end,
                key_themes=key_themes,
                epoch_id=epoch_id,
            )
            if args.json:
                print(json.dumps({"summary_id": summary_id, "scope": scope}))
            else:
                print(f"Summary created ({scope})")
                print(f"  ID: {summary_id[:8]}...")
                print(f"  Period: {period_start} to {period_end}")
                if key_themes:
                    print(f"  Themes: {', '.join(key_themes)}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.summary_action == "list":
        scope = getattr(args, "scope", None)
        try:
            summaries = k.summary_list(scope=scope)
        except ValueError as e:
            print(f"Error: {e}")
            return

        if args.json:
            data = [
                {
                    "id": s.id,
                    "scope": s.scope,
                    "period_start": s.period_start,
                    "period_end": s.period_end,
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "key_themes": s.key_themes,
                    "supersedes": s.supersedes,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                }
                for s in summaries
            ]
            print(json.dumps(data, indent=2, default=str))
        else:
            if not summaries:
                print("No summaries found.")
                return

            print("Summaries")
            print("=" * 60)

            for s in summaries:
                preview = s.content[:80] + "..." if len(s.content) > 80 else s.content
                created = s.created_at.strftime("%Y-%m-%d") if s.created_at else "unknown"

                print(f"\n  [{s.scope}] {s.period_start} to {s.period_end}")
                print(f"      ID: {s.id[:8]}...")
                print(f"      {preview}")
                if s.key_themes:
                    print(f"      Themes: {', '.join(s.key_themes)}")
                if s.supersedes:
                    print(f"      Supersedes: {len(s.supersedes)} summaries")
                print(f"      Created: {created}")

    elif args.summary_action == "show":
        summary_id = validate_input(args.id, "summary_id", 100)
        summary = k.summary_get(summary_id)

        if not summary:
            print(f"Summary {summary_id[:8]}... not found.")
            return

        if args.json:
            data = {
                "id": summary.id,
                "scope": summary.scope,
                "period_start": summary.period_start,
                "period_end": summary.period_end,
                "content": summary.content,
                "key_themes": summary.key_themes,
                "supersedes": summary.supersedes,
                "epoch_id": summary.epoch_id,
                "is_protected": summary.is_protected,
                "created_at": summary.created_at.isoformat() if summary.created_at else None,
                "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
            }
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Summary [{summary.scope}]: {summary.period_start} to {summary.period_end}")
            print("=" * 60)
            print(f"  ID: {summary.id}")
            print(f"  Protected: {'yes' if summary.is_protected else 'no'}")
            if summary.epoch_id:
                print(f"  Epoch: {summary.epoch_id[:8]}...")
            if summary.key_themes:
                print(f"  Themes: {', '.join(summary.key_themes)}")
            if summary.supersedes:
                print(f"  Supersedes: {len(summary.supersedes)} summaries")
                for sid in summary.supersedes:
                    print(f"    - {sid[:8]}...")
            print()
            print(summary.content)

    else:
        print("Usage: kernle summary {write|list|show}")
        print("  write --scope SCOPE --content TEXT --period-start DATE --period-end DATE")
        print("  list [--scope SCOPE]")
        print("  show <id>")
