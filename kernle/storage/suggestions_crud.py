"""Suggestions CRUD operations extracted from SQLiteStorage.

Handles memory suggestion management including creation, retrieval,
status updates, and deletion. All functions receive dependencies
explicitly (connection factory, serializers, sync callbacks) to avoid
circular imports and enable independent testing.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

from .base import MemorySuggestion
from .memory_crud import _row_to_suggestion as _mc_row_to_suggestion
from .raw_entries import escape_like_pattern

logger = logging.getLogger(__name__)


def save_suggestion(
    connect_fn: Callable,
    stack_id: str,
    suggestion: MemorySuggestion,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    queue_sync: Callable,
) -> str:
    """Save a memory suggestion. Returns the suggestion ID."""
    import uuid

    suggestion_id = suggestion.id or str(uuid.uuid4())
    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_suggestions
            (id, stack_id, memory_type, content, confidence, source_raw_ids,
             status, created_at, resolved_at, resolution_reason, promoted_to,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                suggestion_id,
                stack_id,
                suggestion.memory_type,
                to_json(suggestion.content),
                suggestion.confidence,
                to_json(suggestion.source_raw_ids),
                suggestion.status,
                suggestion.created_at.isoformat() if suggestion.created_at else now,
                suggestion.resolved_at.isoformat() if suggestion.resolved_at else None,
                suggestion.resolution_reason,
                suggestion.promoted_to,
                now,
                None,
                suggestion.version,
                0,
            ),
        )
        queue_sync(conn, "memory_suggestions", suggestion_id, "upsert")
        conn.commit()

    return suggestion_id


def get_suggestion(
    connect_fn: Callable,
    stack_id: str,
    suggestion_id: str,
) -> Optional[MemorySuggestion]:
    """Get a specific suggestion by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM memory_suggestions WHERE id = ? AND stack_id = ? AND deleted = 0",
            (suggestion_id, stack_id),
        ).fetchone()

    return _mc_row_to_suggestion(row) if row else None


def get_suggestions(
    connect_fn: Callable,
    stack_id: str,
    status: Optional[str] = None,
    memory_type: Optional[str] = None,
    limit: int = 100,
    min_confidence: Optional[float] = None,
    max_age_hours: Optional[float] = None,
    source_raw_id: Optional[str] = None,
) -> List[MemorySuggestion]:
    """Get suggestions, optionally filtered.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        status: Filter by status (pending, promoted, modified, rejected, dismissed, expired)
        memory_type: Filter by type (episode, belief, note)
        limit: Maximum suggestions to return
        min_confidence: Minimum confidence threshold
        max_age_hours: Only return suggestions created within this many hours
        source_raw_id: Filter to suggestions derived from this raw entry ID
    """
    query = "SELECT * FROM memory_suggestions WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if status is not None:
        query += " AND status = ?"
        params.append(status)

    if memory_type is not None:
        query += " AND memory_type = ?"
        params.append(memory_type)

    if min_confidence is not None:
        query += " AND confidence >= ?"
        params.append(min_confidence)

    if max_age_hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        query += " AND created_at >= ?"
        params.append(cutoff)

    if source_raw_id is not None:
        # source_raw_ids is stored as JSON array; use LIKE for containment
        escaped_raw_id = escape_like_pattern(source_raw_id)
        query += " AND source_raw_ids LIKE ? ESCAPE '\\'"
        params.append(f'%"{escaped_raw_id}"%')

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_suggestion(row) for row in rows]


def update_suggestion_status(
    connect_fn: Callable,
    stack_id: str,
    suggestion_id: str,
    status: str,
    now_fn: Callable[[], str],
    queue_sync: Callable,
    resolution_reason: Optional[str] = None,
    promoted_to: Optional[str] = None,
) -> bool:
    """Update the status of a suggestion."""
    now = now_fn()

    with connect_fn() as conn:
        cursor = conn.execute(
            """
            UPDATE memory_suggestions SET
                status = ?,
                resolved_at = ?,
                resolution_reason = ?,
                promoted_to = ?,
                local_updated_at = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND deleted = 0
        """,
            (status, now, resolution_reason, promoted_to, now, suggestion_id, stack_id),
        )
        if cursor.rowcount > 0:
            queue_sync(conn, "memory_suggestions", suggestion_id, "upsert")
            conn.commit()
            return True
    return False


def delete_suggestion(
    connect_fn: Callable,
    stack_id: str,
    suggestion_id: str,
    now_fn: Callable[[], str],
    queue_sync: Callable,
) -> bool:
    """Delete a suggestion (soft delete)."""
    now = now_fn()

    with connect_fn() as conn:
        cursor = conn.execute(
            """
            UPDATE memory_suggestions SET
                deleted = 1,
                local_updated_at = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND deleted = 0
        """,
            (now, suggestion_id, stack_id),
        )
        if cursor.rowcount > 0:
            queue_sync(conn, "memory_suggestions", suggestion_id, "delete")
            conn.commit()
            return True
    return False


def row_to_suggestion(row: sqlite3.Row) -> MemorySuggestion:
    """Convert a database row to a MemorySuggestion dataclass."""
    return _mc_row_to_suggestion(row)
