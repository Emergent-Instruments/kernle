"""Handlers for memory processing tools: process, process_status."""

from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import (
    sanitize_string,
)

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_process(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from kernle.processing import VALID_TRANSITIONS

    sanitized: Dict[str, Any] = {}
    transition = arguments.get("transition")
    if transition is not None:
        transition = sanitize_string(transition, "transition", 50, required=True)
        if transition not in VALID_TRANSITIONS:
            raise ValueError(
                f"Invalid transition: {transition}. "
                f"Valid: {', '.join(sorted(VALID_TRANSITIONS))}"
            )
    sanitized["transition"] = transition
    sanitized["force"] = arguments.get("force", False)
    if not isinstance(sanitized["force"], bool):
        sanitized["force"] = False
    return sanitized


def validate_memory_process_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_process(args: Dict[str, Any], k: Kernle) -> str:
    transition = args.get("transition")
    force = args.get("force", False)
    try:
        results = k.process(transition=transition, force=force)
    except RuntimeError:
        return "Memory processing requires a bound model. Use entity.set_model() first."

    if not results:
        result = "No processing triggered. "
        if not force:
            result += "Thresholds not met — use force=true to process anyway."
        else:
            result += "No unprocessed memories found."
        return result

    lines = [f"Processing complete ({len(results)} transition(s)):\n"]
    for r in results:
        if r.skipped:
            lines.append(f"  {r.layer_transition}: skipped ({r.skip_reason})")
        else:
            created_summary = ", ".join(f"{c['type']}:{c['id'][:8]}" for c in r.created)
            lines.append(
                f"  {r.layer_transition}: " f"{r.source_count} sources -> {len(r.created)} created"
            )
            if created_summary:
                lines.append(f"    Created: {created_summary}")
            if r.errors:
                for err in r.errors:
                    lines.append(f"    Error: {err}")
    return "\n".join(lines)


def handle_memory_process_status(args: Dict[str, Any], k: Kernle) -> str:
    from kernle.processing import DEFAULT_LAYER_CONFIGS

    lines = ["Memory Processing Status\n"]

    try:
        storage = k._storage
        unprocessed_raw = storage.list_raw(processed=False, limit=1000)
        raw_count = len(unprocessed_raw)
        lines.append(f"Unprocessed raw entries: {raw_count}")

        episodes = storage.get_episodes(limit=1000)
        unprocessed_eps = [e for e in episodes if not e.processed]
        lines.append(f"Unprocessed episodes: {len(unprocessed_eps)}")

        beliefs = storage.get_beliefs(limit=1000)
        unprocessed_beliefs = [b for b in beliefs if not getattr(b, "processed", False)]
        lines.append(f"Unprocessed beliefs: {len(unprocessed_beliefs)}")

        lines.append("\nTrigger Status:")
        trigger_checks = {
            "raw_to_episode": raw_count,
            "raw_to_note": raw_count,
            "episode_to_belief": len(unprocessed_eps),
            "episode_to_goal": len(unprocessed_eps),
            "episode_to_relationship": len(unprocessed_eps),
            "episode_to_drive": len(unprocessed_eps),
            "belief_to_value": len(unprocessed_beliefs),
        }
        for transition_name, count in trigger_checks.items():
            cfg = DEFAULT_LAYER_CONFIGS.get(transition_name)
            if cfg:
                would_fire = count >= cfg.quantity_threshold
                status = "READY" if would_fire else "waiting"
                lines.append(f"  {transition_name}: {count}/{cfg.quantity_threshold} ({status})")
    except Exception as e:
        lines.append(f"\nError gathering status: {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_process": handle_memory_process,
    "memory_process_status": handle_memory_process_status,
}

VALIDATORS = {
    "memory_process": validate_memory_process,
    "memory_process_status": validate_memory_process_status,
}
