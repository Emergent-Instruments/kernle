"""Utility functions and constants for Kernle core."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default token budget for memory loading
DEFAULT_TOKEN_BUDGET = 8000

# Maximum token budget allowed (consistent across CLI, MCP, and core)
MAX_TOKEN_BUDGET = 50000

# Minimum token budget allowed
MIN_TOKEN_BUDGET = 100

# Maximum characters per memory item (for truncation)
DEFAULT_MAX_ITEM_CHARS = 500

# Token estimation safety margin (actual JSON output is larger than text estimation)
TOKEN_ESTIMATION_SAFETY_MARGIN = 1.3

# Priority scores for each memory type (higher = more important)
MEMORY_TYPE_PRIORITIES = {
    "checkpoint": 1.00,  # Always loaded first
    "value": 0.90,
    "self_narrative": 0.90,  # Autobiographical identity — loads alongside values
    "summary_decade": 0.95,
    "summary_epoch": 0.85,
    "summary_year": 0.80,
    "belief": 0.70,
    "goal": 0.65,
    "drive": 0.60,
    "summary_quarter": 0.50,
    "episode": 0.40,
    "summary_month": 0.35,
    "note": 0.35,
    "relationship": 0.30,
}


def estimate_tokens(text: str, include_safety_margin: bool = True) -> int:
    """Estimate token count from text.

    Uses the simple heuristic of ~4 characters per token, with a safety
    margin to account for JSON serialization overhead.

    Args:
        text: The text to estimate tokens for
        include_safety_margin: If True, multiply by safety margin (default: True)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    base_estimate = len(text) // 4
    if include_safety_margin:
        return int(base_estimate * TOKEN_ESTIMATION_SAFETY_MARGIN)
    return base_estimate


def truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary with ellipsis.

    Args:
        text: Text to truncate
        max_chars: Maximum characters (including ellipsis)

    Returns:
        Truncated text with "..." if truncated
    """
    if not text or len(text) <= max_chars:
        return text

    # Leave room for ellipsis
    target = max_chars - 3
    if target <= 0:
        return "..."

    # Find last space before target
    truncated = text[:target]
    last_space = truncated.rfind(" ")

    if last_space > target // 2:  # Only use word boundary if reasonable
        truncated = truncated[:last_space]

    return truncated + "..."


def _get_memory_hint_text(memory_type: str, record: Any) -> str:
    """Get the primary text content of a memory record for echo hints."""
    if memory_type == "value":
        return f"{record.name}: {record.statement}"
    elif memory_type == "belief":
        return record.statement
    elif memory_type == "goal":
        return f"{record.title} {record.description or ''}"
    elif memory_type == "drive":
        return f"{record.drive_type}: {' '.join(record.focus_areas or [])}"
    elif memory_type == "episode":
        return f"{record.objective} {record.outcome}"
    elif memory_type == "note":
        return record.content
    elif memory_type == "relationship":
        return f"{record.entity_name}: {record.notes or ''}"
    return str(record)


def _truncate_to_words(text: str, max_words: int = 8) -> str:
    """Truncate text to approximately max_words words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _get_record_tags(memory_type: str, record: Any) -> List[str]:
    """Extract tags from a memory record."""
    tags = getattr(record, "tags", None) or []
    context_tags = getattr(record, "context_tags", None) or []
    focus_areas = []
    if memory_type == "drive":
        focus_areas = getattr(record, "focus_areas", None) or []
    return tags + context_tags + focus_areas


def _get_record_created_at(record: Any) -> Optional[datetime]:
    """Extract created_at datetime from a record."""
    return getattr(record, "created_at", None)


def _build_memory_echoes(
    excluded: list,
    max_echoes: int = 20,
) -> Dict[str, Any]:
    """Build memory echoes (peripheral awareness) from excluded candidates.

    After budget selection, this generates compact hints about memories that
    didn't fit in the token budget, giving the agent peripheral awareness
    of what else exists in memory.

    Args:
        excluded: Excluded candidate list [(priority, type, record), ...],
                  sorted by priority descending
        max_echoes: Maximum number of echo entries (default: 20)

    Returns:
        Dict with keys: echoes, temporal_summary, topic_clusters
    """
    if not excluded:
        return {
            "echoes": [],
            "temporal_summary": None,
            "topic_clusters": [],
        }

    echoes = []
    for priority, memory_type, record in excluded[:max_echoes]:
        hint_text = _get_memory_hint_text(memory_type, record)
        hint = _truncate_to_words(hint_text, max_words=8)
        echoes.append(
            {
                "type": memory_type,
                "id": record.id,
                "hint": hint,
                "salience": round(priority, 3),
            }
        )

    all_dates = []
    for _, memory_type, record in excluded:
        created = _get_record_created_at(record)
        if created:
            all_dates.append(created)

    temporal_summary = None
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        span_days = (max_date - min_date).days
        span_years = round(span_days / 365.25, 1)
        temporal_summary = (
            f"Memory spans {min_date.strftime('%Y-%m-%d')} to "
            f"{max_date.strftime('%Y-%m-%d')} ({span_years} years). "
            f"{len(excluded)} excluded memories."
        )

    tag_counts: Dict[str, int] = {}
    for _, memory_type, record in excluded:
        for tag in _get_record_tags(memory_type, record):
            tag_lower = tag.lower()
            tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    topic_clusters = [tag for tag, _count in sorted_tags[:6]]

    return {
        "echoes": echoes,
        "temporal_summary": temporal_summary,
        "topic_clusters": topic_clusters,
    }


def _get_record_attr(record: Any, attr: str, default: Any = None) -> Any:
    """Get an attribute from a record, supporting both dataclass and dict."""
    if hasattr(record, attr):
        return getattr(record, attr, default)
    if isinstance(record, dict):
        return record.get(attr, default)
    return default


def compute_priority_score(
    memory_type: str,
    record: Any,
    kernle_instance: Optional[Any] = None,
) -> float:
    """Compute priority score for a memory record.

    The score combines three weighted factors:
    - 55% type weight (base priority for memory type * type-specific factor)
    - 35% record factors (confidence, recency, etc.)
    - 10% emotional salience (abs(valence) * arousal * time-decay)

    Emotional salience uses a 90-day half-life decay so high-impact episodes
    remain cognitively available longer than standard 30-day salience.

    Args:
        memory_type: Type of memory (value, belief, etc.)
        record: The memory record (dataclass or dict)
        kernle_instance: Optional Kernle instance to compute decayed confidence

    Returns:
        Priority score (0.0-1.0)
    """
    base_priority = MEMORY_TYPE_PRIORITIES.get(memory_type, 0.5)

    # Get record value based on type
    if memory_type == "value":
        # priority is 0-100, normalize to 0-1
        priority = _get_record_attr(record, "priority", 50)
        type_factor = priority / 100.0
    elif memory_type == "belief":
        confidence = _get_record_attr(record, "confidence", 0.8)
        if kernle_instance is not None:
            decay_fn = getattr(kernle_instance, "get_confidence_with_decay", None)
            if callable(decay_fn):
                try:
                    decayed = decay_fn(record, "belief")
                    if isinstance(decayed, (int, float)):
                        confidence = decayed
                except Exception:
                    logger.debug("Failed to compute decayed confidence for belief priority")
        type_factor = confidence
    elif memory_type == "drive":
        type_factor = _get_record_attr(record, "intensity", 0.5)
    elif memory_type in ("goal", "episode", "note"):
        # For time-based priority, we'd need to compute recency
        # For now, use a default factor (records are already sorted by recency)
        type_factor = 0.7
    elif memory_type == "relationship":
        # Use sentiment as a factor
        sentiment = _get_record_attr(record, "sentiment", 0.0)
        type_factor = (sentiment + 1) / 2  # Normalize -1..1 to 0..1
    elif memory_type == "self_narrative":
        # Active narratives are always high priority
        type_factor = 0.9
    elif memory_type.startswith("summary_"):
        # Summaries use scope-based priority directly
        type_factor = 0.8
    else:
        type_factor = 0.5

    type_weight = base_priority  # base priority for this memory type
    record_factors = type_factor  # type-specific factor (confidence, priority, etc.)

    # Emotional salience: abs(valence) * arousal * time-decay(90-day half-life)
    valence = _get_record_attr(record, "emotional_valence", 0.0)
    arousal = _get_record_attr(record, "emotional_arousal", 0.0)

    emotional_salience = 0.0
    if abs(valence) > 0 or arousal > 0:
        half_life = 90.0  # 3x standard 30-day salience
        days_since = 0.0
        created_at = _get_record_attr(record, "created_at", None)
        if created_at is not None:
            try:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                delta = now - created_at
                days_since = max(0.0, delta.total_seconds() / 86400.0)
            except (ValueError, TypeError):
                pass
        emotional_salience = abs(valence) * arousal * (half_life / (days_since + half_life))

    # Weighted combination: 55% type weight, 35% record factors, 10% emotional salience
    score = 0.55 * type_weight + 0.35 * record_factors + 0.10 * emotional_salience

    # Belief scope boost: self-beliefs get +0.05 priority (KEP v3)
    if memory_type == "belief":
        belief_scope = _get_record_attr(record, "belief_scope", "world")
        if belief_scope == "self":
            score = min(1.0, score + 0.05)

    return score
