"""Goals CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
goal-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import Goal, VersionConflictError
from .memory_crud import _row_to_goal as _mc_row_to_goal

logger = logging.getLogger(__name__)


def save_goal(
    connect_fn: Callable,
    stack_id: str,
    goal: Goal,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save a goal.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        goal: The Goal to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
        sync_to_file: Syncs goals to flat file (no args).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not goal.id:
        goal.id = str(uuid.uuid4())

    if goal.derived_from and lineage_checker:
        lineage_checker("goal", goal.id, goal.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO goals
            (id, stack_id, title, description, goal_type, priority, status, created_at,
             confidence, source_type, source_episodes, derived_from,
             last_verified, verification_count, confidence_history,
             strength,
             context, context_tags,
             subject_ids, access_grants, consent_grants,
             epoch_id,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                goal.id,
                stack_id,
                goal.title,
                goal.description,
                goal.goal_type,
                goal.priority,
                goal.status,
                goal.created_at.isoformat() if goal.created_at else now,
                goal.confidence,
                goal.source_type,
                to_json(goal.source_episodes),
                to_json(goal.derived_from),
                goal.last_verified.isoformat() if goal.last_verified else None,
                goal.verification_count,
                to_json(goal.confidence_history),
                goal.strength,
                goal.context,
                to_json(goal.context_tags),
                to_json(getattr(goal, "subject_ids", None)),
                to_json(getattr(goal, "access_grants", None)),
                to_json(getattr(goal, "consent_grants", None)),
                goal.epoch_id,
                now,
                goal.cloud_synced_at.isoformat() if goal.cloud_synced_at else None,
                goal.version,
                1 if goal.deleted else 0,
            ),
        )
        # Queue for sync with record data
        goal_data = to_json(record_to_dict(goal))
        queue_sync(conn, "goals", goal.id, "upsert", data=goal_data)

        # Save embedding for search
        content = f"{goal.title} {goal.description or ''}"
        save_embedding(conn, "goals", goal.id, content)

        conn.commit()

    # Sync to flat file
    sync_to_file()

    return goal.id


def update_goal_atomic(
    connect_fn: Callable,
    stack_id: str,
    goal: Goal,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
    expected_version: Optional[int] = None,
) -> bool:
    """Update a goal with optimistic concurrency control.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        goal: The goal with updated fields.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation.
        save_embedding: Saves embedding for search.
        sync_to_file: Syncs goals to flat file.
        expected_version: The version we expect the record to have.
                         If None, uses goal.version.

    Returns:
        True if update succeeded.

    Raises:
        VersionConflictError: If the record's version doesn't match expected.
    """
    if expected_version is None:
        expected_version = goal.version

    now = now_fn()

    with connect_fn() as conn:
        # Check current version
        current = conn.execute(
            "SELECT version FROM goals WHERE id = ? AND stack_id = ?",
            (goal.id, stack_id),
        ).fetchone()

        if not current:
            return False

        current_version = current["version"]
        if current_version != expected_version:
            raise VersionConflictError("goals", goal.id, expected_version, current_version)

        # Atomic update with version increment
        cursor = conn.execute(
            """
            UPDATE goals SET
                status = ?,
                priority = ?,
                description = ?,
                title = ?,
                goal_type = ?,
                confidence = ?,
                source_type = ?,
                source_episodes = ?,
                derived_from = ?,
                last_verified = ?,
                verification_count = ?,
                confidence_history = ?,
                strength = ?,
                context = ?,
                context_tags = ?,
                local_updated_at = ?,
                deleted = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND version = ?
            """,
            (
                goal.status,
                goal.priority,
                goal.description,
                goal.title,
                goal.goal_type,
                goal.confidence,
                goal.source_type,
                to_json(goal.source_episodes),
                to_json(goal.derived_from),
                goal.last_verified.isoformat() if goal.last_verified else None,
                goal.verification_count,
                to_json(goal.confidence_history),
                goal.strength,
                goal.context,
                to_json(goal.context_tags),
                now,
                1 if goal.deleted else 0,
                goal.id,
                stack_id,
                expected_version,
            ),
        )

        if cursor.rowcount == 0:
            conn.rollback()
            new_current = conn.execute(
                "SELECT version FROM goals WHERE id = ? AND stack_id = ?",
                (goal.id, stack_id),
            ).fetchone()
            actual = new_current["version"] if new_current else -1
            raise VersionConflictError("goals", goal.id, expected_version, actual)

        # Queue for sync
        goal.version = expected_version + 1
        goal_data = to_json(record_to_dict(goal))
        queue_sync(conn, "goals", goal.id, "upsert", data=goal_data)

        # Update embedding
        content = f"{goal.title} {goal.description or ''}"
        save_embedding(conn, "goals", goal.id, content)

        conn.commit()

    # Sync to flat file
    sync_to_file()

    return True


def get_goals(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    status: Optional[str] = "active",
    limit: int = 100,
    requesting_entity: Optional[str] = None,
) -> List[Goal]:
    """Get goals."""
    query = "SELECT * FROM goals WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if status:
        query += " AND status = ?"
        params.append(status)

    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter

    query += " ORDER BY created_at DESC LIMIT ?"
    params.extend(access_params)
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_goal(row) for row in rows]


def row_to_goal(row: sqlite3.Row) -> Goal:
    """Convert a database row to a Goal dataclass."""
    return _mc_row_to_goal(row)
