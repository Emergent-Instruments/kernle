"""Self-narrative CLI commands for Kernle (KEP v3)."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_narrative(args, k: "Kernle"):
    """Handle narrative subcommands."""
    action = getattr(args, "narrative_action", None)

    if action == "show":
        narrative_type = getattr(args, "type", "identity") or "identity"
        narrative = k.narrative_get_active(narrative_type)

        if args.json:
            if narrative:
                data = {
                    "id": narrative.id,
                    "narrative_type": narrative.narrative_type,
                    "content": narrative.content,
                    "key_themes": narrative.key_themes,
                    "unresolved_tensions": narrative.unresolved_tensions,
                    "epoch_id": narrative.epoch_id,
                    "created_at": (
                        narrative.created_at.isoformat() if narrative.created_at else None
                    ),
                    "updated_at": (
                        narrative.updated_at.isoformat() if narrative.updated_at else None
                    ),
                }
                print(json.dumps(data, indent=2, default=str))
            else:
                print(json.dumps(None))
        else:
            if narrative:
                print(f"Self-Narrative ({narrative.narrative_type})")
                print("=" * 60)
                print()
                print(narrative.content)
                print()
                if narrative.key_themes:
                    print(f"  Themes: {', '.join(narrative.key_themes)}")
                if narrative.unresolved_tensions:
                    print(f"  Tensions: {', '.join(narrative.unresolved_tensions)}")
                if narrative.epoch_id:
                    print(f"  Epoch: {narrative.epoch_id[:8]}...")
                updated = (
                    narrative.updated_at.strftime("%Y-%m-%d %H:%M")
                    if narrative.updated_at
                    else "unknown"
                )
                print(f"  Updated: {updated}")
                print(f"  ID: {narrative.id[:8]}...")
            else:
                print(f"No active {narrative_type} narrative.")

    elif action == "update":
        narrative_type = getattr(args, "type", "identity") or "identity"
        content = validate_input(args.content, "content", 10000)

        themes = getattr(args, "theme", None)
        tensions = getattr(args, "tension", None)
        epoch_id = getattr(args, "epoch", None)

        try:
            narrative_id = k.narrative_save(
                content=content,
                narrative_type=narrative_type,
                key_themes=themes,
                unresolved_tensions=tensions,
                epoch_id=epoch_id,
            )
            if args.json:
                print(json.dumps({"narrative_id": narrative_id, "type": narrative_type}))
            else:
                print(f"Narrative saved ({narrative_type})")
                print(f"  ID: {narrative_id[:8]}...")
                if themes:
                    print(f"  Themes: {', '.join(themes)}")
                if tensions:
                    print(f"  Tensions: {', '.join(tensions)}")
        except ValueError as e:
            print(f"Error: {e}")

    elif action == "history":
        narrative_type = getattr(args, "type", None)
        narratives = k.narrative_list(narrative_type=narrative_type, active_only=False)

        if args.json:
            data = [
                {
                    "id": n.id,
                    "narrative_type": n.narrative_type,
                    "content": n.content[:100] + "..." if len(n.content) > 100 else n.content,
                    "is_active": n.is_active,
                    "key_themes": n.key_themes,
                    "unresolved_tensions": n.unresolved_tensions,
                    "supersedes": n.supersedes,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                    "updated_at": n.updated_at.isoformat() if n.updated_at else None,
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
                updated = n.updated_at.strftime("%Y-%m-%d") if n.updated_at else "unknown"
                preview = n.content[:80] + "..." if len(n.content) > 80 else n.content

                print(f"\n  [{n.narrative_type}] ({status})")
                print(f"      ID: {n.id[:8]}...")
                print(f"      Updated: {updated}")
                print(f"      Content: {preview}")
                if n.key_themes:
                    print(f"      Themes: {', '.join(n.key_themes)}")
                if n.supersedes:
                    print(f"      Supersedes: {n.supersedes[:8]}...")

    else:
        print("Usage: kernle narrative {show|update|history}")
        print("  show                     Show active narrative")
        print("  update <content>         Create/update narrative")
        print("  history                  Show all narratives (including inactive)")
