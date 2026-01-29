"""Suggestion commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def resolve_suggestion_id(k: "Kernle", partial_id: str) -> str:
    """Resolve a partial suggestion ID to full ID.

    Tries exact match first, then prefix match.
    Returns full ID or raises ValueError if not found or ambiguous.
    """
    # First try exact match
    suggestion = k.get_suggestion(partial_id)
    if suggestion:
        return partial_id

    # Try prefix match by listing all suggestions
    suggestions = k.get_suggestions(limit=1000)
    matches = [s for s in suggestions if s["id"].startswith(partial_id)]

    if len(matches) == 0:
        raise ValueError(f"Suggestion '{partial_id}' not found")
    elif len(matches) == 1:
        return matches[0]["id"]
    else:
        match_ids = [m["id"][:12] for m in matches[:5]]
        suffix = "..." if len(matches) > 5 else ""
        raise ValueError(
            f"Ambiguous ID '{partial_id}' matches {len(matches)} suggestions: {', '.join(match_ids)}{suffix}"
        )


def cmd_suggestions(args, k: "Kernle"):
    """Handle suggestion subcommands."""
    if args.suggestions_action == "list":
        # Filter by status
        status = None
        if hasattr(args, "pending") and args.pending:
            status = "pending"
        elif hasattr(args, "approved") and args.approved:
            status = "promoted"
        elif hasattr(args, "rejected") and args.rejected:
            status = "rejected"

        # Filter by type
        memory_type = getattr(args, "type", None)

        suggestions = k.get_suggestions(
            status=status,
            memory_type=memory_type,
            limit=getattr(args, "limit", 50),
        )

        if not suggestions:
            if status:
                print(f"No {status} suggestions found.")
            else:
                print("No suggestions found.")
            return

        if getattr(args, "json", False):
            print(json.dumps(suggestions, indent=2, default=str))
        else:
            # Group by status for display
            pending = [s for s in suggestions if s["status"] == "pending"]
            promoted = [s for s in suggestions if s["status"] in ("promoted", "modified")]
            rejected = [s for s in suggestions if s["status"] == "rejected"]

            if status:
                # Show only requested status
                display_suggestions = suggestions
                print(f"Suggestions ({len(display_suggestions)} {status})")
            else:
                print(
                    f"Suggestions ({len(suggestions)} total: {len(pending)} pending, {len(promoted)} approved, {len(rejected)} rejected)"
                )
                display_suggestions = suggestions

            print("=" * 60)

            for s in display_suggestions:
                status_icons = {
                    "pending": "?",
                    "promoted": "+",
                    "modified": "*",
                    "rejected": "x",
                }
                icon = status_icons.get(s["status"], "?")
                type_label = s["memory_type"].upper()[:3]

                # Get preview of content
                content = s.get("content", {})
                if s["memory_type"] == "episode":
                    preview = content.get("objective", "")[:60]
                elif s["memory_type"] == "belief":
                    preview = content.get("statement", "")[:60]
                else:
                    preview = content.get("content", "")[:60]

                if len(preview) >= 60:
                    preview += "..."

                print(f"\n[{icon}] {s['id'][:8]} [{type_label}] {s['confidence']:.0%}")
                print(f"    {preview}")

                if s["status"] in ("promoted", "modified") and s.get("promoted_to"):
                    print(f"    -> {s['promoted_to']}")

            print("\n" + "=" * 60)
            if pending:
                print("\nReview pending: kernle suggestions show <id>")
                print("Approve: kernle suggestions approve <id>")
                print("Reject: kernle suggestions reject <id> --reason '...'")

    elif args.suggestions_action == "show":
        try:
            full_id = resolve_suggestion_id(k, args.id)
        except ValueError as e:
            print(f"Error: {e}")
            return

        suggestion = k.get_suggestion(full_id)
        if not suggestion:
            print(f"Suggestion {args.id} not found.")
            return

        if getattr(args, "json", False):
            print(json.dumps(suggestion, indent=2, default=str))
        else:
            status_labels = {
                "pending": "Pending Review",
                "promoted": "Approved",
                "modified": "Approved with modifications",
                "rejected": "Rejected",
            }
            print(f"Suggestion: {suggestion['id']}")
            print(f"Status: {status_labels.get(suggestion['status'], suggestion['status'])}")
            print(f"Type: {suggestion['memory_type']}")
            print(f"Confidence: {suggestion['confidence']:.0%}")
            print(f"Created: {suggestion['created_at']}")

            if suggestion.get("source_raw_ids"):
                print(
                    f"Source raw entries: {', '.join(s[:8] for s in suggestion['source_raw_ids'])}"
                )

            print()
            print("Suggested Content:")
            print("-" * 40)

            content = suggestion.get("content", {})
            if suggestion["memory_type"] == "episode":
                print(f"Objective: {content.get('objective', '')}")
                print(f"Outcome: {content.get('outcome', '')}")
                if content.get("outcome_type"):
                    print(f"Outcome Type: {content['outcome_type']}")
                if content.get("lessons"):
                    print("Lessons:")
                    for lesson in content["lessons"]:
                        print(f"  - {lesson}")
            elif suggestion["memory_type"] == "belief":
                print(f"Statement: {content.get('statement', '')}")
                print(f"Type: {content.get('belief_type', 'fact')}")
                if content.get("confidence"):
                    print(f"Confidence: {content['confidence']:.0%}")
            else:
                print(content.get("content", ""))
                if content.get("note_type") and content["note_type"] != "note":
                    print(f"Note Type: {content['note_type']}")
                if content.get("speaker"):
                    print(f"Speaker: {content['speaker']}")
                if content.get("reason"):
                    print(f"Reason: {content['reason']}")

            print("-" * 40)

            if suggestion["status"] == "pending":
                print()
                print("Actions:")
                print(f"  Approve: kernle suggestions approve {suggestion['id'][:8]}")
                print(f"  Reject:  kernle suggestions reject {suggestion['id'][:8]} --reason '...'")

            if suggestion.get("promoted_to"):
                print(f"\nPromoted to: {suggestion['promoted_to']}")

            if suggestion.get("resolution_reason"):
                print(f"Resolution reason: {suggestion['resolution_reason']}")

    elif args.suggestions_action == "approve":
        try:
            full_id = resolve_suggestion_id(k, args.id)
        except ValueError as e:
            print(f"Error: {e}")
            return

        suggestion = k.get_suggestion(full_id)
        if not suggestion:
            print(f"Suggestion {args.id} not found.")
            return

        if suggestion["status"] != "pending":
            print(f"Suggestion is already {suggestion['status']}.")
            return

        # Parse modifications if provided
        modifications = None
        if hasattr(args, "objective") and args.objective:
            modifications = modifications or {}
            modifications["objective"] = args.objective
        if hasattr(args, "outcome") and args.outcome:
            modifications = modifications or {}
            modifications["outcome"] = args.outcome
        if hasattr(args, "statement") and args.statement:
            modifications = modifications or {}
            modifications["statement"] = args.statement
        if hasattr(args, "content") and args.content:
            modifications = modifications or {}
            modifications["content"] = args.content

        memory_id = k.promote_suggestion(full_id, modifications)
        if memory_id:
            status = "modified" if modifications else "promoted"
            print(f"Suggestion approved ({status}).")
            print(f"Created {suggestion['memory_type']}: {memory_id[:8]}...")
        else:
            print("Failed to promote suggestion.")

    elif args.suggestions_action == "reject":
        try:
            full_id = resolve_suggestion_id(k, args.id)
        except ValueError as e:
            print(f"Error: {e}")
            return

        suggestion = k.get_suggestion(full_id)
        if not suggestion:
            print(f"Suggestion {args.id} not found.")
            return

        if suggestion["status"] != "pending":
            print(f"Suggestion is already {suggestion['status']}.")
            return

        reason = getattr(args, "reason", None)
        if k.reject_suggestion(full_id, reason):
            print("Suggestion rejected.")
            if reason:
                print(f"Reason: {reason}")
        else:
            print("Failed to reject suggestion.")

    elif args.suggestions_action == "extract":
        # Extract suggestions from unprocessed raw entries
        limit = getattr(args, "limit", 50)

        print(f"Extracting suggestions from up to {limit} unprocessed raw entries...")
        suggestions = k.extract_suggestions_from_unprocessed(limit=limit)

        if not suggestions:
            print("No suggestions extracted.")
            return

        # Group by type
        by_type = {}
        for s in suggestions:
            t = s["memory_type"]
            by_type[t] = by_type.get(t, 0) + 1

        print(f"\nExtracted {len(suggestions)} suggestion(s):")
        for t, count in by_type.items():
            print(f"  {t}: {count}")

        print("\nReview with: kernle suggestions list --pending")
