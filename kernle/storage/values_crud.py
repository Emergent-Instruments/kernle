"""Values CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
value-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import Value
from .memory_crud import _row_to_value as _mc_row_to_value

logger = logging.getLogger(__name__)


def save_value(
    connect_fn: Callable,
    stack_id: str,
    value: Value,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    sync_to_file: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save a value.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        value: The Value to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
        sync_to_file: Syncs values to flat file (no args).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not value.id:
        value.id = str(uuid.uuid4())

    if value.derived_from and lineage_checker:
        lineage_checker("value", value.id, value.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO agent_values
            (id, stack_id, name, statement, priority, created_at,
             confidence, source_type, source_episodes, derived_from,
             last_verified, verification_count, confidence_history,
             strength,
             context, context_tags,
             subject_ids, access_grants, consent_grants,
             epoch_id,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                value.id,
                stack_id,
                value.name,
                value.statement,
                value.priority,
                value.created_at.isoformat() if value.created_at else now,
                value.confidence,
                value.source_type,
                to_json(value.source_episodes),
                to_json(value.derived_from),
                value.last_verified.isoformat() if value.last_verified else None,
                value.verification_count,
                to_json(value.confidence_history),
                value.strength,
                value.context,
                to_json(value.context_tags),
                to_json(getattr(value, "subject_ids", None)),
                to_json(getattr(value, "access_grants", None)),
                to_json(getattr(value, "consent_grants", None)),
                value.epoch_id,
                now,
                value.cloud_synced_at.isoformat() if value.cloud_synced_at else None,
                value.version,
                1 if value.deleted else 0,
            ),
        )
        # Queue for sync with record data
        value_data = to_json(record_to_dict(value))
        queue_sync(conn, "agent_values", value.id, "upsert", data=value_data)

        # Save embedding for search
        content = f"{value.name}: {value.statement}"
        save_embedding(conn, "agent_values", value.id, content)

        conn.commit()

    # Sync to flat file
    sync_to_file()

    return value.id


def get_values(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    limit: int = 100,
    requesting_entity: Optional[str] = None,
) -> List[Value]:
    """Get values ordered by priority."""
    access_filter, access_params = build_access_filter(requesting_entity)
    with connect_fn() as conn:
        rows = conn.execute(
            f"SELECT * FROM agent_values WHERE stack_id = ? AND deleted = 0{access_filter} ORDER BY priority DESC LIMIT ?",
            [stack_id] + access_params + [limit],
        ).fetchall()

    return [_mc_row_to_value(row) for row in rows]


def row_to_value(row: sqlite3.Row) -> Value:
    """Convert a database row to a Value dataclass."""
    return _mc_row_to_value(row)
