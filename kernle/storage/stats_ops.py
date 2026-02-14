"""Stats and batch-loading operations extracted from SQLiteStorage.

Contains get_stats (record counts) and load_all (batch memory loading
optimization). These are read-only operations that aggregate or load
data across multiple memory tables.

All functions receive dependencies explicitly (connection, converters)
to avoid circular imports and enable independent testing.
"""

import logging
import sqlite3
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def get_stats(
    conn: sqlite3.Connection,
    stack_id: str,
) -> Dict[str, int]:
    """Get counts of each record type.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.

    Returns:
        Dict mapping record type names to their counts.
    """
    stats = {}
    for table, key in [
        ("episodes", "episodes"),
        ("beliefs", "beliefs"),
        ("agent_values", "values"),
        ("goals", "goals"),
        ("notes", "notes"),
        ("drives", "drives"),
        ("relationships", "relationships"),
        ("raw_entries", "raw"),
        ("memory_suggestions", "suggestions"),
    ]:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE stack_id = ? AND deleted = 0",
            (stack_id,),
        ).fetchone()[0]
        stats[key] = count

    # Add count of pending suggestions specifically
    pending_count = conn.execute(
        "SELECT COUNT(*) FROM memory_suggestions WHERE stack_id = ? AND status = 'pending' AND deleted = 0",
        (stack_id,),
    ).fetchone()[0]
    stats["pending_suggestions"] = pending_count

    return stats


def load_all(
    conn: sqlite3.Connection,
    stack_id: str,
    row_converters: Dict[str, Callable],
    values_limit: Optional[int] = 10,
    beliefs_limit: Optional[int] = 20,
    goals_limit: Optional[int] = 10,
    goals_status: str = "active",
    episodes_limit: Optional[int] = 20,
    notes_limit: Optional[int] = 5,
    drives_limit: Optional[int] = None,
    relationships_limit: Optional[int] = None,
    epoch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load all memory types in a single database connection.

    This optimizes the common pattern of loading working memory context
    by batching all queries into a single connection, avoiding N+1 query
    patterns where each memory type requires a separate connection.

    Args:
        conn: Open sqlite3 connection.
        stack_id: The stack identifier for record isolation.
        row_converters: Dict mapping type name to row-to-record converter callable.
            Expected keys: "value", "belief", "goal", "drive", "episode",
            "note", "relationship".
        values_limit: Max values to load (None = 1000 for budget loading)
        beliefs_limit: Max beliefs to load (None = 1000 for budget loading)
        goals_limit: Max goals to load (None = 1000 for budget loading)
        goals_status: Goal status filter ("active", "all", etc.)
        episodes_limit: Max episodes to load (None = 1000 for budget loading)
        notes_limit: Max notes to load (None = 1000 for budget loading)
        drives_limit: Max drives to load (None = all drives)
        relationships_limit: Max relationships to load (None = all relationships)
        epoch_id: If set, filter candidates to this epoch only

    Returns:
        Dict with keys: values, beliefs, goals, drives, episodes, notes, relationships
    """
    # Use high limit (1000) when None is passed - for budget-based loading
    high_limit = 1000
    _values_limit = values_limit if values_limit is not None else high_limit
    _beliefs_limit = beliefs_limit if beliefs_limit is not None else high_limit
    _goals_limit = goals_limit if goals_limit is not None else high_limit
    _episodes_limit = episodes_limit if episodes_limit is not None else high_limit
    _notes_limit = notes_limit if notes_limit is not None else high_limit

    result: Dict[str, Any] = {
        "values": [],
        "beliefs": [],
        "goals": [],
        "drives": [],
        "episodes": [],
        "notes": [],
        "relationships": [],
    }

    # Build epoch filter clause
    epoch_clause = ""
    epoch_params: tuple = ()
    if epoch_id:
        epoch_clause = " AND epoch_id = ?"
        epoch_params = (epoch_id,)

    # Values - ordered by priority, exclude forgotten
    rows = conn.execute(
        f"SELECT * FROM agent_values WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY priority DESC LIMIT ?",
        (stack_id, *epoch_params, _values_limit),
    ).fetchall()
    result["values"] = [row_converters["value"](row) for row in rows]

    # Beliefs - ordered by confidence, exclude forgotten
    rows = conn.execute(
        f"SELECT * FROM beliefs WHERE stack_id = ? AND deleted = 0 AND strength > 0.0 AND (is_active = 1 OR is_active IS NULL){epoch_clause} ORDER BY confidence DESC LIMIT ?",
        (stack_id, *epoch_params, _beliefs_limit),
    ).fetchall()
    result["beliefs"] = [row_converters["belief"](row) for row in rows]

    # Goals - filtered by status, exclude forgotten
    if goals_status and goals_status != "all":
        rows = conn.execute(
            f"SELECT * FROM goals WHERE stack_id = ? AND deleted = 0 AND strength > 0.0 AND status = ?{epoch_clause} ORDER BY created_at DESC LIMIT ?",
            (stack_id, goals_status, *epoch_params, _goals_limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM goals WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
            (stack_id, *epoch_params, _goals_limit),
        ).fetchall()
    result["goals"] = [row_converters["goal"](row) for row in rows]

    # Drives - all for agent (or limited), exclude forgotten
    if drives_limit is not None:
        rows = conn.execute(
            f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} LIMIT ?",
            (stack_id, *epoch_params, drives_limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause}",
            (stack_id, *epoch_params),
        ).fetchall()
    result["drives"] = [row_converters["drive"](row) for row in rows]

    # Episodes - most recent, exclude forgotten
    rows = conn.execute(
        f"SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
        (stack_id, *epoch_params, _episodes_limit),
    ).fetchall()
    result["episodes"] = [row_converters["episode"](row) for row in rows]

    # Notes - most recent, exclude forgotten
    rows = conn.execute(
        f"SELECT * FROM notes WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} ORDER BY created_at DESC LIMIT ?",
        (stack_id, *epoch_params, _notes_limit),
    ).fetchall()
    result["notes"] = [row_converters["note"](row) for row in rows]

    # Relationships - all for agent (or limited), exclude forgotten
    if relationships_limit is not None:
        rows = conn.execute(
            f"SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause} LIMIT ?",
            (stack_id, *epoch_params, relationships_limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM relationships WHERE stack_id = ? AND deleted = 0 AND strength > 0.0{epoch_clause}",
            (stack_id, *epoch_params),
        ).fetchall()
    result["relationships"] = [row_converters["relationship"](row) for row in rows]

    return result
