"""Notes CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
note-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Callable, List, Optional

from .base import Note
from .memory_crud import _row_to_note as _mc_row_to_note

logger = logging.getLogger(__name__)


def save_note(
    connect_fn: Callable,
    stack_id: str,
    note: Note,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save a note.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        note: The Note to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not note.id:
        note.id = str(uuid.uuid4())

    if note.derived_from and lineage_checker:
        lineage_checker("note", note.id, note.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO notes
            (id, stack_id, content, note_type, speaker, reason, tags, created_at,
             confidence, source_type, source_episodes, derived_from,
             last_verified, verification_count, confidence_history,
             strength,
             context, context_tags, source_entity,
             subject_ids, access_grants, consent_grants,
             epoch_id,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                note.id,
                stack_id,
                note.content,
                note.note_type,
                note.speaker,
                note.reason,
                to_json(note.tags),
                note.created_at.isoformat() if note.created_at else now,
                note.confidence,
                note.source_type,
                to_json(note.source_episodes),
                to_json(note.derived_from),
                note.last_verified.isoformat() if note.last_verified else None,
                note.verification_count,
                to_json(note.confidence_history),
                note.strength,
                note.context,
                to_json(note.context_tags),
                getattr(note, "source_entity", None),
                to_json(getattr(note, "subject_ids", None)),
                to_json(getattr(note, "access_grants", None)),
                to_json(getattr(note, "consent_grants", None)),
                note.epoch_id,
                now,
                note.cloud_synced_at.isoformat() if note.cloud_synced_at else None,
                note.version,
                1 if note.deleted else 0,
            ),
        )
        # Queue for sync with record data
        note_data = to_json(record_to_dict(note))
        queue_sync(conn, "notes", note.id, "upsert", data=note_data)

        # Save embedding for search
        save_embedding(conn, "notes", note.id, note.content)

        conn.commit()

    return note.id


def save_notes_batch(
    connect_fn: Callable,
    stack_id: str,
    notes: List[Note],
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
) -> List[str]:
    """Save multiple notes in a single transaction."""
    if not notes:
        return []
    now = now_fn()
    ids = []
    with connect_fn() as conn:
        for note in notes:
            if not note.id:
                note.id = str(uuid.uuid4())
            ids.append(note.id)
            conn.execute(
                """
                INSERT OR REPLACE INTO notes
                (id, stack_id, content, note_type, speaker, reason, tags, created_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 times_accessed, last_accessed, is_protected, strength,
                 processed, context, context_tags,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    note.id,
                    stack_id,
                    note.content,
                    note.note_type,
                    note.speaker,
                    note.reason,
                    to_json(note.tags),
                    note.created_at.isoformat() if note.created_at else now,
                    note.confidence,
                    note.source_type,
                    to_json(note.source_episodes),
                    to_json(note.derived_from),
                    note.last_verified.isoformat() if note.last_verified else None,
                    note.verification_count,
                    to_json(note.confidence_history),
                    note.times_accessed,
                    note.last_accessed.isoformat() if note.last_accessed else None,
                    1 if note.is_protected else 0,
                    note.strength,
                    1 if note.processed else 0,
                    note.context,
                    to_json(note.context_tags),
                    note.epoch_id,
                    now,
                    note.cloud_synced_at.isoformat() if note.cloud_synced_at else None,
                    note.version,
                    1 if note.deleted else 0,
                ),
            )
            note_data = to_json(record_to_dict(note))
            queue_sync(conn, "notes", note.id, "upsert", data=note_data)
            save_embedding(conn, "notes", note.id, note.content)
        conn.commit()
    return ids


def get_notes(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    limit: int = 100,
    since: Optional[datetime] = None,
    note_type: Optional[str] = None,
    requesting_entity: Optional[str] = None,
) -> List[Note]:
    """Get notes."""
    query = "SELECT * FROM notes WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if since:
        query += " AND created_at >= ?"
        params.append(since.isoformat())

    if note_type:
        query += " AND note_type = ?"
        params.append(note_type)

    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter

    query += " ORDER BY created_at DESC LIMIT ?"
    params.extend(access_params)
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_mc_row_to_note(row) for row in rows]


def mark_note_processed(
    connect_fn: Callable,
    stack_id: str,
    note_id: str,
    mark_processed_fn: Callable,
) -> bool:
    """Mark note as processed.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        note_id: The note to mark as processed.
        mark_processed_fn: The raw_entries.mark_processed function.

    Returns:
        True if the note was marked as processed.
    """
    with connect_fn() as conn:
        return mark_processed_fn(conn, stack_id, "notes", note_id)


def row_to_note(row: sqlite3.Row) -> Note:
    """Convert a database row to a Note dataclass."""
    return _mc_row_to_note(row)
