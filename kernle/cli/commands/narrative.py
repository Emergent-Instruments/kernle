"""Self-narrative commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_narrative(args, k: "Kernle"):
    """Handle narrative subcommands."""
    if args.narrative_action == "show":
        narrative_type = getattr(args, "type", "identity") or "identity"
        narrative = k.narrative_get_active(narrative_type=narrative_type)

        if not narrative:
            if args.json:
                print(json.dumps(None))
            else:
                print(f"No active {narrative_type} narrative.")
            return

        if args.json:
            data = {
                "id": narrative.id,
                "narrative_type": narrative.narrative_type,
                "content": narrative.content,
                "key_themes": narrative.key_themes,
                "unresolved_tensions": narrative.unresolved_tensions,
                "epoch_id": narrative.epoch_id,
                "is_active": narrative.is_active,
                "created_at": narrative.created_at.isoformat() if narrative.created_at else None,
                "updated_at": narrative.updated_at.isoformat() if narrative.updated_at else None,
            }
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"Self-Narrative [{narrative.narrative_type}]")
            print("=" * 60)
            print(f"  ID: {narrative.id}")
            if narrative.epoch_id:
                print(f"  Epoch: {narrative.epoch_id[:8]}...")
            if narrative.key_themes:
                print(f"  Themes: {', '.join(narrative.key_themes)}")
            if narrative.unresolved_tensions:
                print(f"  Tensions: {', '.join(narrative.unresolved_tensions)}")
            created = (
                narrative.created_at.strftime("%Y-%m-%d") if narrative.created_at else "unknown"
            )
            print(f"  Created: {created}")
            print()
            print(narrative.content)

    elif args.narrative_action == "update":
        narrative_type = getattr(args, "type", "identity") or "identity"
        content = validate_input(args.content, "content", 10000)
        key_themes = args.theme if hasattr(args, "theme") and args.theme else None
        tensions = args.tension if hasattr(args, "tension") and args.tension else None
        epoch_id = getattr(args, "epoch_id", None)

        try:
            narrative_id = k.narrative_save(
                content=content,
                narrative_type=narrative_type,
                key_themes=key_themes,
                unresolved_tensions=tensions,
                epoch_id=epoch_id,
            )
            if args.json:
                print(json.dumps({"narrative_id": narrative_id, "narrative_type": narrative_type}))
            else:
                print(f"Narrative updated ({narrative_type})")
                print(f"  ID: {narrative_id[:8]}...")
                if key_themes:
                    print(f"  Themes: {', '.join(key_themes)}")
                if tensions:
                    print(f"  Tensions: {', '.join(tensions)}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.narrative_action == "history":
        narrative_type = getattr(args, "type", None)
        try:
            narratives = k.narrative_list(narrative_type=narrative_type, active_only=False)
        except ValueError as e:
            print(f"Error: {e}")
            return

        if args.json:
            data = [
                {
                    "id": n.id,
                    "narrative_type": n.narrative_type,
                    "content": n.content[:200] + "..." if len(n.content) > 200 else n.content,
                    "key_themes": n.key_themes,
                    "is_active": n.is_active,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in narratives
            ]
            print(json.dumps(data, indent=2, default=str))
        else:
            if not narratives:
                print("No narratives found.")
                return

            print("Self-Narrative History")
            print("=" * 60)

            for n in narratives:
                status = "ACTIVE" if n.is_active else "inactive"
                preview = n.content[:80] + "..." if len(n.content) > 80 else n.content
                created = n.created_at.strftime("%Y-%m-%d") if n.created_at else "unknown"

                print(f"\n  [{n.narrative_type}] ({status})")
                print(f"      ID: {n.id[:8]}...")
                print(f"      {preview}")
                if n.key_themes:
                    print(f"      Themes: {', '.join(n.key_themes)}")
                print(f"      Created: {created}")

    else:
        print("Usage: kernle narrative {show|update|history}")
        print("  show [--type TYPE]")
        print("  update --content TEXT [--type TYPE] [--theme T]... [--tension T]...")
        print("  history [--type TYPE]")
