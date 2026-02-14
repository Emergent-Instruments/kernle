"""Epoch CRUD operations extracted from SQLiteStorage.

Handles epoch (temporal era) management including creation, retrieval,
and closing. All functions receive dependencies explicitly (connection
factory, serializers) to avoid circular imports and enable independent
testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import Epoch
from .memory_crud import _row_to_epoch as _mc_row_to_epoch

logger = logging.getLogger(__name__)


def save_epoch(
    connect_fn: Callable,
    stack_id: str,
    epoch: Epoch,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
) -> str:
    """Save an epoch. Returns the epoch ID."""

    if not epoch.id:
        epoch.id = str(uuid.uuid4())

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO epochs
            (id, stack_id, epoch_number, name, started_at, ended_at,
             trigger_type, trigger_description, summary,
             key_belief_ids, key_relationship_ids,
             key_goal_ids, dominant_drive_ids,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                epoch.id,
                stack_id,
                epoch.epoch_number,
                epoch.name,
                epoch.started_at.isoformat() if epoch.started_at else now,
                epoch.ended_at.isoformat() if epoch.ended_at else None,
                epoch.trigger_type,
                epoch.trigger_description,
                epoch.summary,
                to_json(epoch.key_belief_ids),
                to_json(epoch.key_relationship_ids),
                to_json(epoch.key_goal_ids),
                to_json(epoch.dominant_drive_ids),
                now,
                epoch.cloud_synced_at.isoformat() if epoch.cloud_synced_at else None,
                epoch.version,
                1 if epoch.deleted else 0,
            ),
        )
        conn.commit()

    return epoch.id


def get_epoch(
    connect_fn: Callable,
    stack_id: str,
    epoch_id: str,
) -> Optional[Epoch]:
    """Get a specific epoch by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM epochs WHERE id = ? AND stack_id = ? AND deleted = 0",
            (epoch_id, stack_id),
        ).fetchone()

    return _mc_row_to_epoch(row) if row else None


def get_epochs(
    connect_fn: Callable,
    stack_id: str,
    limit: int = 100,
) -> List[Epoch]:
    """Get all epochs, ordered by epoch_number DESC."""
    with connect_fn() as conn:
        rows = conn.execute(
            "SELECT * FROM epochs WHERE stack_id = ? AND deleted = 0 "
            "ORDER BY epoch_number DESC LIMIT ?",
            (stack_id, limit),
        ).fetchall()

    return [_mc_row_to_epoch(row) for row in rows]


def get_current_epoch(
    connect_fn: Callable,
    stack_id: str,
) -> Optional[Epoch]:
    """Get the currently active (open) epoch, if any."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM epochs WHERE stack_id = ? AND ended_at IS NULL AND deleted = 0 "
            "ORDER BY epoch_number DESC LIMIT 1",
            (stack_id,),
        ).fetchone()

    return _mc_row_to_epoch(row) if row else None


def close_epoch(
    connect_fn: Callable,
    stack_id: str,
    epoch_id: str,
    now_fn: Callable[[], str],
    summary: Optional[str] = None,
) -> bool:
    """Close an epoch by setting ended_at. Returns True if closed."""
    now = now_fn()
    with connect_fn() as conn:
        cursor = conn.execute(
            "UPDATE epochs SET ended_at = ?, summary = COALESCE(?, summary), "
            "local_updated_at = ?, version = version + 1 "
            "WHERE id = ? AND stack_id = ? AND ended_at IS NULL AND deleted = 0",
            (now, summary, now, epoch_id, stack_id),
        )
        conn.commit()
    return cursor.rowcount > 0


def row_to_epoch(row: sqlite3.Row) -> Epoch:
    """Convert a database row to an Epoch dataclass."""
    return _mc_row_to_epoch(row)
