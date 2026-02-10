"""Handlers for sync/suggestion tools: sync, suggestions_list/promote/reject/extract."""

import json
from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import (
    sanitize_string,
    validate_enum,
    validate_number,
)
from kernle.types import VALID_SUGGESTION_STATUSES

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_sync(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def _validate_suggestion_list_args(
    arguments: Dict[str, Any], include_format: bool = True
) -> Dict[str, Any]:
    """Shared validation for suggestion list args."""
    sanitized: Dict[str, Any] = {}
    status = arguments.get("status", "pending")
    if status == "all":
        sanitized["status"] = None
    else:
        sanitized["status"] = validate_enum(
            status, "status", sorted(VALID_SUGGESTION_STATUSES), "pending"
        )
    memory_type = arguments.get("memory_type")
    if memory_type:
        sanitized["memory_type"] = validate_enum(
            memory_type, "memory_type", ["episode", "belief", "note"], None
        )
    else:
        sanitized["memory_type"] = None
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 20))
    if include_format:
        sanitized["format"] = validate_enum(
            arguments.get("format"), "format", ["text", "json"], "text"
        )
    # New filter params
    min_confidence = arguments.get("min_confidence")
    if min_confidence is not None:
        sanitized["min_confidence"] = float(
            validate_number(min_confidence, "min_confidence", 0.0, 1.0, None)
        )
    else:
        sanitized["min_confidence"] = None
    max_age_hours = arguments.get("max_age_hours")
    if max_age_hours is not None:
        sanitized["max_age_hours"] = float(
            validate_number(max_age_hours, "max_age_hours", 0.1, 100000, None)
        )
    else:
        sanitized["max_age_hours"] = None
    sanitized["source_raw_id"] = (
        sanitize_string(arguments.get("source_raw_id"), "source_raw_id", 100, required=False)
        or None
    )
    return sanitized


def validate_memory_suggestions_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_suggestion_list_args(arguments, include_format=True)


def validate_memory_suggestions_promote(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["suggestion_id"] = sanitize_string(
        arguments.get("suggestion_id"), "suggestion_id", 100, required=True
    )
    sanitized["objective"] = (
        sanitize_string(arguments.get("objective"), "objective", 1000, required=False) or None
    )
    sanitized["outcome"] = (
        sanitize_string(arguments.get("outcome"), "outcome", 1000, required=False) or None
    )
    sanitized["statement"] = (
        sanitize_string(arguments.get("statement"), "statement", 1000, required=False) or None
    )
    sanitized["content"] = (
        sanitize_string(arguments.get("content"), "content", 2000, required=False) or None
    )
    return sanitized


def validate_memory_suggestions_reject(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["suggestion_id"] = sanitize_string(
        arguments.get("suggestion_id"), "suggestion_id", 100, required=True
    )
    sanitized["reason"] = (
        sanitize_string(arguments.get("reason"), "reason", 500, required=False) or None
    )
    return sanitized


def validate_memory_suggestions_extract(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 200, 50))
    return sanitized


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_sync(args: Dict[str, Any], k: Kernle) -> str:
    sync_result = k.sync()
    lines = ["Sync complete:"]
    lines.append(f"  Pushed: {sync_result.get('pushed', 0)}")
    lines.append(f"  Pulled: {sync_result.get('pulled', 0)}")
    if sync_result.get("conflicts"):
        lines.append(f"  Conflicts: {sync_result['conflicts']}")
    if sync_result.get("errors"):
        lines.append(f"  Errors: {len(sync_result['errors'])}")
        for err in sync_result["errors"][:3]:
            lines.append(f"    - {err}")
    return "\n".join(lines)


def handle_memory_suggestions_list(args: Dict[str, Any], k: Kernle) -> str:
    status = args.get("status")
    memory_type = args.get("memory_type")
    limit = args.get("limit", 20)
    format_type = args.get("format", "text")

    suggestions = k.get_suggestions(
        status=status,
        memory_type=memory_type,
        limit=limit,
        min_confidence=args.get("min_confidence"),
        max_age_hours=args.get("max_age_hours"),
        source_raw_id=args.get("source_raw_id"),
    )

    if not suggestions:
        status_str = f" {status}" if status else ""
        return f"No{status_str} suggestions found."

    if format_type == "json":
        return json.dumps(suggestions, indent=2, default=str)

    # Group counts
    pending = sum(1 for s in suggestions if s["status"] == "pending")
    promoted = sum(1 for s in suggestions if s["status"] in ("promoted", "modified"))
    rejected = sum(1 for s in suggestions if s["status"] == "rejected")

    lines = [f"Memory Suggestions ({len(suggestions)} total)\n"]
    if not status:  # Show breakdown only for unfiltered list
        lines.append(f"Pending: {pending} | Approved: {promoted} | Rejected: {rejected}\n")

    for s in suggestions:
        status_icons = {
            "pending": "?",
            "promoted": "+",
            "modified": "*",
            "rejected": "x",
            "dismissed": "-",
            "expired": "~",
        }
        icon = status_icons.get(s["status"], "?")
        type_label = s["memory_type"][:3].upper()

        content = s.get("content", {})
        if s["memory_type"] == "episode":
            preview = content.get("objective", "")[:50]
        elif s["memory_type"] == "belief":
            preview = content.get("statement", "")[:50]
        else:
            preview = content.get("content", "")[:50]

        lines.append(f"[{icon}] {s['id'][:8]} [{type_label}] {s['confidence']:.0%}: {preview}...")

        if s.get("promoted_to"):
            lines.append(f"    -> {s['promoted_to']}")

    return "\n".join(lines)


def handle_memory_suggestions_promote(args: Dict[str, Any], k: Kernle) -> str:
    suggestion_id = args["suggestion_id"]

    modifications = {}
    if args.get("objective"):
        modifications["objective"] = args["objective"]
    if args.get("outcome"):
        modifications["outcome"] = args["outcome"]
    if args.get("statement"):
        modifications["statement"] = args["statement"]
    if args.get("content"):
        modifications["content"] = args["content"]

    memory_id = k.promote_suggestion(
        suggestion_id,
        modifications if modifications else None,
    )

    if memory_id:
        status = "modified" if modifications else "promoted"
        return f"Suggestion {suggestion_id[:8]}... {status} to memory {memory_id[:8]}..."
    return f"Could not promote suggestion {suggestion_id[:8]}... (not found or not pending)"


def handle_memory_suggestions_reject(args: Dict[str, Any], k: Kernle) -> str:
    suggestion_id = args["suggestion_id"]
    reason = args.get("reason")

    if k.reject_suggestion(suggestion_id, reason):
        result = f"Suggestion {suggestion_id[:8]}... rejected"
        if reason:
            result += f" (reason: {reason})"
        return result
    return f"Could not reject suggestion {suggestion_id[:8]}... (not found or not pending)"


def handle_memory_suggestions_extract(args: Dict[str, Any], k: Kernle) -> str:
    limit = args.get("limit", 50)
    suggestions = k.extract_suggestions_from_unprocessed(limit=limit)

    if not suggestions:
        return "No suggestions extracted from raw entries."

    by_type = {}
    for s in suggestions:
        t = s["memory_type"]
        by_type[t] = by_type.get(t, 0) + 1

    lines = [f"Extracted {len(suggestions)} suggestion(s):\n"]
    for t, count in by_type.items():
        lines.append(f"  {t}: {count}")
    lines.append("\nUse memory_suggestions_list to review pending suggestions.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# New suggestion resolution tools
# ---------------------------------------------------------------------------


def validate_suggestion_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return _validate_suggestion_list_args(arguments, include_format=False)


def validate_suggestion_accept(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["suggestion_id"] = sanitize_string(
        arguments.get("suggestion_id"), "suggestion_id", 100, required=True
    )
    sanitized["objective"] = (
        sanitize_string(arguments.get("objective"), "objective", 1000, required=False) or None
    )
    sanitized["outcome"] = (
        sanitize_string(arguments.get("outcome"), "outcome", 1000, required=False) or None
    )
    sanitized["statement"] = (
        sanitize_string(arguments.get("statement"), "statement", 1000, required=False) or None
    )
    sanitized["content"] = (
        sanitize_string(arguments.get("content"), "content", 2000, required=False) or None
    )
    return sanitized


def validate_suggestion_dismiss(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["suggestion_id"] = sanitize_string(
        arguments.get("suggestion_id"), "suggestion_id", 100, required=True
    )
    sanitized["reason"] = (
        sanitize_string(arguments.get("reason"), "reason", 500, required=False) or None
    )
    return sanitized


def handle_suggestion_list(args: Dict[str, Any], k: Kernle) -> str:
    status = args.get("status")
    memory_type = args.get("memory_type")
    limit = args.get("limit", 20)

    suggestions = k.get_suggestions(
        status=status,
        memory_type=memory_type,
        limit=limit,
        min_confidence=args.get("min_confidence"),
        max_age_hours=args.get("max_age_hours"),
        source_raw_id=args.get("source_raw_id"),
    )

    if not suggestions:
        status_str = f" {status}" if status else ""
        return f"No{status_str} suggestions found."

    return json.dumps(suggestions, indent=2, default=str)


def handle_suggestion_accept(args: Dict[str, Any], k: Kernle) -> str:
    suggestion_id = args["suggestion_id"]

    modifications = {}
    if args.get("objective"):
        modifications["objective"] = args["objective"]
    if args.get("outcome"):
        modifications["outcome"] = args["outcome"]
    if args.get("statement"):
        modifications["statement"] = args["statement"]
    if args.get("content"):
        modifications["content"] = args["content"]

    memory_id = k.accept_suggestion(
        suggestion_id,
        modifications if modifications else None,
    )

    if memory_id:
        status = "modified" if modifications else "accepted"
        return f"Suggestion {suggestion_id[:8]}... {status}, created memory {memory_id[:8]}..."
    return f"Could not accept suggestion {suggestion_id[:8]}... (not found or not pending)"


def handle_suggestion_dismiss(args: Dict[str, Any], k: Kernle) -> str:
    suggestion_id = args["suggestion_id"]
    reason = args.get("reason")

    if k.dismiss_suggestion(suggestion_id, reason):
        result = f"Suggestion {suggestion_id[:8]}... dismissed"
        if reason:
            result += f" (reason: {reason})"
        return result
    return f"Could not dismiss suggestion {suggestion_id[:8]}... (not found or not pending)"


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_sync": handle_memory_sync,
    "memory_suggestions_list": handle_memory_suggestions_list,
    "memory_suggestions_promote": handle_memory_suggestions_promote,
    "memory_suggestions_reject": handle_memory_suggestions_reject,
    "memory_suggestions_extract": handle_memory_suggestions_extract,
    "suggestion_list": handle_suggestion_list,
    "suggestion_accept": handle_suggestion_accept,
    "suggestion_dismiss": handle_suggestion_dismiss,
}

VALIDATORS = {
    "memory_sync": validate_memory_sync,
    "memory_suggestions_list": validate_memory_suggestions_list,
    "memory_suggestions_promote": validate_memory_suggestions_promote,
    "memory_suggestions_reject": validate_memory_suggestions_reject,
    "memory_suggestions_extract": validate_memory_suggestions_extract,
    "suggestion_list": validate_suggestion_list,
    "suggestion_accept": validate_suggestion_accept,
    "suggestion_dismiss": validate_suggestion_dismiss,
}
