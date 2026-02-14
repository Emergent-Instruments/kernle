"""Handlers for core memory tools: load, checkpoint, episode, note, search, status, raw."""

import json
from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import (
    sanitize_array,
    sanitize_source_metadata,
    sanitize_string,
    validate_enum,
    validate_number,
)
from kernle.mcp.tool_definitions import VALID_SOURCE_TYPES

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_load(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["format"] = validate_enum(arguments.get("format"), "format", ["text", "json"], "text")
    from kernle.core import MAX_TOKEN_BUDGET, MIN_TOKEN_BUDGET

    sanitized["budget"] = int(
        validate_number(arguments.get("budget"), "budget", MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET, 8000)
    )
    sanitized["truncate"] = arguments.get("truncate", True)
    if not isinstance(sanitized["truncate"], bool):
        sanitized["truncate"] = True
    return sanitized


def validate_memory_checkpoint_save(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["task"] = sanitize_string(arguments.get("task"), "task", 500, required=True)
    sanitized["pending"] = sanitize_array(arguments.get("pending"), "pending", 200, 20)
    sanitized["context"] = sanitize_string(
        arguments.get("context"), "context", 1000, required=False
    )
    return sanitized


def validate_memory_checkpoint_load(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def validate_memory_episode(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["objective"] = sanitize_string(
        arguments.get("objective"), "objective", 1000, required=True
    )
    sanitized["outcome"] = sanitize_string(arguments.get("outcome"), "outcome", 1000, required=True)
    sanitized["lessons"] = sanitize_array(arguments.get("lessons"), "lessons", 500, 20)
    sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
    sanitized.update(sanitize_source_metadata(arguments, VALID_SOURCE_TYPES))
    return sanitized


def validate_memory_note(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["content"] = sanitize_string(arguments.get("content"), "content", 2000, required=True)
    sanitized["type"] = validate_enum(
        arguments.get("type"), "type", ["note", "decision", "insight", "quote"], "note"
    )
    sanitized["speaker"] = sanitize_string(arguments.get("speaker"), "speaker", 200, required=False)
    sanitized["reason"] = sanitize_string(arguments.get("reason"), "reason", 1000, required=False)
    sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
    sanitized.update(sanitize_source_metadata(arguments, VALID_SOURCE_TYPES))
    return sanitized


def validate_memory_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["query"] = sanitize_string(arguments.get("query"), "query", 500, required=True)
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
    return sanitized


def validate_memory_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def validate_memory_raw(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["blob"] = sanitize_string(
        arguments.get("blob"), "blob", max_length=1_000_000, required=True
    )
    return sanitized


def validate_memory_raw_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["query"] = sanitize_string(
        arguments.get("query"), "query", max_length=1000, required=True
    )
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 500, 50))
    return sanitized


def validate_memory_note_search(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["query"] = sanitize_string(arguments.get("query"), "query", 500, required=True)
    sanitized["note_type"] = validate_enum(
        arguments.get("note_type"),
        "note_type",
        ["note", "decision", "insight", "quote", "all"],
        "all",
    )
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
    return sanitized


def validate_memory_episode_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["episode_id"] = sanitize_string(
        arguments.get("episode_id"), "episode_id", 100, required=True
    )
    sanitized["outcome"] = sanitize_string(
        arguments.get("outcome"), "outcome", 1000, required=False
    )
    sanitized["lessons"] = sanitize_array(arguments.get("lessons"), "lessons", 500, 20)
    sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
    return sanitized


def validate_memory_auto_capture(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["text"] = sanitize_string(arguments.get("text"), "text", 10000, required=True)
    sanitized.update(
        sanitize_source_metadata(
            arguments,
            VALID_SOURCE_TYPES,
            coalesce_empty_to_none=False,
        )
    )
    if not sanitized.get("source"):
        sanitized["source"] = "auto"
    sanitized["extract_suggestions"] = arguments.get("extract_suggestions", False)
    if not isinstance(sanitized["extract_suggestions"], bool):
        sanitized["extract_suggestions"] = False
    return sanitized


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_load(args: Dict[str, Any], k: Kernle) -> str:
    format_type = args.get("format", "text")
    budget = args.get("budget", 8000)
    truncate = args.get("truncate", True)
    memory = k.load(budget=budget, truncate=truncate)
    if format_type == "json":
        return json.dumps(memory, indent=2, default=str)
    return k.format_memory(memory)


def handle_memory_checkpoint_save(args: Dict[str, Any], k: Kernle) -> str:
    checkpoint = k.checkpoint(
        task=args["task"],
        pending=args.get("pending"),
        context=args.get("context"),
    )
    result = f"Checkpoint saved: {checkpoint['current_task']}"
    if checkpoint.get("pending"):
        result += f"\nPending: {len(checkpoint['pending'])} items"
    return result


def handle_memory_checkpoint_load(args: Dict[str, Any], k: Kernle) -> str:
    loaded_checkpoint = k.load_checkpoint()
    if loaded_checkpoint:
        return json.dumps(loaded_checkpoint, indent=2, default=str)
    return "No checkpoint found."


def handle_memory_episode(args: Dict[str, Any], k: Kernle) -> str:
    episode_id = k.episode(
        objective=args["objective"],
        outcome=args["outcome"],
        lessons=args.get("lessons"),
        tags=args.get("tags"),
        context=args.get("context"),
        context_tags=args.get("context_tags"),
        source=args.get("source"),
        derived_from=args.get("derived_from"),
        source_type=args.get("source_type"),
    )
    return f"Episode saved: {episode_id[:8]}..."


def handle_memory_note(args: Dict[str, Any], k: Kernle) -> str:
    k.note(
        content=args["content"],
        type=args.get("type", "note"),
        speaker=args.get("speaker"),
        reason=args.get("reason"),
        tags=args.get("tags"),
        context=args.get("context"),
        context_tags=args.get("context_tags"),
        source=args.get("source"),
        derived_from=args.get("derived_from"),
        source_type=args.get("source_type"),
    )
    return f"Note saved: {args['content'][:50]}..."


def handle_memory_search(args: Dict[str, Any], k: Kernle) -> str:
    results = k.search(
        query=args["query"],
        limit=args.get("limit", 10),
    )
    if not results:
        return f"No results for '{args['query']}'"
    lines = [f"Found {len(results)} result(s):\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['type']}] {r['title']}")
        if r.get("lessons"):
            for lesson in r["lessons"][:2]:
                lines.append(f"   → {lesson[:60]}...")
    return "\n".join(lines)


def handle_memory_status(args: Dict[str, Any], k: Kernle) -> str:
    status = k.status()
    return f"""Memory Status ({status["stack_id"]})
=====================================
Values:     {status["values"]}
Beliefs:    {status["beliefs"]}
Goals:      {status["goals"]} active
Episodes:   {status["episodes"]}
Checkpoint: {"Yes" if status["checkpoint"] else "No"}"""


def handle_memory_raw(args: Dict[str, Any], k: Kernle) -> str:
    blob = args["blob"]
    capture_id = k.raw(blob=blob, source="mcp")
    return json.dumps(
        {
            "captured": True,
            "id": capture_id[:8],
            "full_id": capture_id,
            "source": "mcp",
        },
        indent=2,
    )


def handle_memory_raw_search(args: Dict[str, Any], k: Kernle) -> str:
    query = args["query"]
    limit = args.get("limit", 50)
    entries = k.search_raw(query, limit=limit)

    if not entries:
        return json.dumps({"results": [], "query": query, "total": 0}, indent=2)

    results = []
    for e in entries:
        results.append(
            {
                "id": e["id"][:8],
                "blob_preview": (e["blob"][:200] + "..." if len(e["blob"]) > 200 else e["blob"]),
                "captured_at": e.get("captured_at"),
                "source": e.get("source"),
                "processed": e.get("processed"),
            }
        )
    return json.dumps(
        {
            "results": results,
            "query": query,
            "total": len(results),
        },
        indent=2,
    )


def handle_memory_note_search(args: Dict[str, Any], k: Kernle) -> str:
    query = args["query"]
    note_type = args.get("note_type", "all")
    limit = args.get("limit", 10)

    results = k.search(query=query, limit=limit * 2)

    note_results = [r for r in results if r.get("type") in ["note", "decision", "insight", "quote"]]

    if note_type != "all":
        note_results = [r for r in note_results if r.get("type") == note_type]

    note_results = note_results[:limit]

    if not note_results:
        return f"No notes found for '{query}'"

    lines = [f"Found {len(note_results)} note(s):\n"]
    for i, n in enumerate(note_results, 1):
        lines.append(f"{i}. [{n['type']}] {n['title']}")
        if n.get("date"):
            lines.append(f"   {n['date']}")
    return "\n".join(lines)


def handle_memory_episode_update(args: Dict[str, Any], k: Kernle) -> str:
    episode_id = args["episode_id"]
    updated = k.update_episode(
        episode_id=episode_id,
        outcome=args.get("outcome"),
        lessons=args.get("lessons"),
        tags=args.get("tags"),
    )
    if updated:
        return f"Episode {episode_id[:8]}... updated successfully."
    return f"Episode {episode_id[:8]}... not found."


def handle_memory_auto_capture(args: Dict[str, Any], k: Kernle) -> str:
    source = args.get("source", "auto")
    if source not in {"cli", "mcp", "sdk", "import", "unknown"}:
        if "auto" in source.lower():
            source = "mcp"
        else:
            source = "mcp"
    extract_suggestions = args.get("extract_suggestions", False)

    blob = args["text"]
    if args.get("context"):
        blob = f"{blob}\n\n[Context: {args['context']}]"

    capture_id = k.raw(
        blob=blob,
        source=source,
    )

    if extract_suggestions:
        text_lower = args["text"].lower()
        suggestions = []

        if any(
            word in text_lower
            for word in [
                "session",
                "completed",
                "shipped",
                "implemented",
                "built",
                "fixed",
                "deployed",
                "finished",
            ]
        ):
            suggestions.append("episode")
        if any(
            word in text_lower
            for word in ["insight", "decision", "realized", "learned", "important", "noted"]
        ):
            suggestions.append("note")
        if any(
            word in text_lower
            for word in [
                "believe",
                "think that",
                "seems like",
                "pattern",
                "always",
                "never",
                "should",
            ]
        ):
            suggestions.append("belief")

        result_data = {
            "captured": True,
            "id": capture_id[:8],
            "source": source,
            "suggestions": suggestions or ["review"],
            "promote_command": f"kernle raw promote {capture_id[:8]} --type <episode|note|belief>",
        }
        return json.dumps(result_data, indent=2)

    return f"Auto-captured: {capture_id[:8]}... (source: {source})"


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_load": handle_memory_load,
    "memory_checkpoint_save": handle_memory_checkpoint_save,
    "memory_checkpoint_load": handle_memory_checkpoint_load,
    "memory_episode": handle_memory_episode,
    "memory_note": handle_memory_note,
    "memory_search": handle_memory_search,
    "memory_status": handle_memory_status,
    "memory_raw": handle_memory_raw,
    "memory_raw_search": handle_memory_raw_search,
    "memory_note_search": handle_memory_note_search,
    "memory_episode_update": handle_memory_episode_update,
    "memory_auto_capture": handle_memory_auto_capture,
}

VALIDATORS = {
    "memory_load": validate_memory_load,
    "memory_checkpoint_save": validate_memory_checkpoint_save,
    "memory_checkpoint_load": validate_memory_checkpoint_load,
    "memory_episode": validate_memory_episode,
    "memory_note": validate_memory_note,
    "memory_search": validate_memory_search,
    "memory_status": validate_memory_status,
    "memory_raw": validate_memory_raw,
    "memory_raw_search": validate_memory_raw_search,
    "memory_note_search": validate_memory_note_search,
    "memory_episode_update": validate_memory_episode_update,
    "memory_auto_capture": validate_memory_auto_capture,
}
