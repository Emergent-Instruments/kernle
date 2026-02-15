"""Drives CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
drive-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import Drive, VersionConflictError
from .memory_crud import _row_to_drive as _mc_row_to_drive

logger = logging.getLogger(__name__)


def save_drive(
    connect_fn: Callable,
    stack_id: str,
    drive: Drive,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save or update a drive.

    If a drive with the same drive_type already exists for this stack,
    it will be updated. Otherwise a new drive is inserted.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        drive: The Drive to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not drive.id:
        drive.id = str(uuid.uuid4())

    if drive.derived_from and lineage_checker:
        lineage_checker("drive", drive.id, drive.derived_from)

    now = now_fn()

    with connect_fn() as conn:
        # Check if exists
        existing = conn.execute(
            "SELECT id FROM drives WHERE stack_id = ? AND drive_type = ?",
            (stack_id, drive.drive_type),
        ).fetchone()

        if existing:
            drive.id = existing["id"]
            conn.execute(
                """
                UPDATE drives SET
                    intensity = ?, focus_areas = ?, updated_at = ?,
                    confidence = ?, source_type = ?, source_episodes = ?,
                    derived_from = ?, last_verified = ?, verification_count = ?,
                    confidence_history = ?, context = ?, context_tags = ?,
                    subject_ids = ?, access_grants = ?, consent_grants = ?,
                    local_updated_at = ?, version = version + 1
                WHERE id = ?
            """,
                (
                    drive.intensity,
                    to_json(drive.focus_areas),
                    now,
                    drive.confidence,
                    drive.source_type,
                    to_json(drive.source_episodes),
                    to_json(drive.derived_from),
                    drive.last_verified.isoformat() if drive.last_verified else None,
                    drive.verification_count,
                    to_json(drive.confidence_history),
                    drive.context,
                    to_json(drive.context_tags),
                    to_json(getattr(drive, "subject_ids", None)),
                    to_json(getattr(drive, "access_grants", None)),
                    to_json(getattr(drive, "consent_grants", None)),
                    now,
                    drive.id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO drives
                (id, stack_id, drive_type, intensity, focus_areas, created_at, updated_at,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 strength,
                 context, context_tags,
                 subject_ids, access_grants, consent_grants,
                 epoch_id,
                 local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    drive.id,
                    stack_id,
                    drive.drive_type,
                    drive.intensity,
                    to_json(drive.focus_areas),
                    now,
                    now,
                    drive.confidence,
                    drive.source_type,
                    to_json(drive.source_episodes),
                    to_json(drive.derived_from),
                    drive.last_verified.isoformat() if drive.last_verified else None,
                    drive.verification_count,
                    to_json(drive.confidence_history),
                    drive.strength,
                    drive.context,
                    to_json(drive.context_tags),
                    to_json(getattr(drive, "subject_ids", None)),
                    to_json(getattr(drive, "access_grants", None)),
                    to_json(getattr(drive, "consent_grants", None)),
                    drive.epoch_id,
                    now,
                    None,
                    1,
                    0,
                ),
            )

        # Queue for sync with record data
        drive_data = to_json(record_to_dict(drive))
        queue_sync(conn, "drives", drive.id, "upsert", data=drive_data)
        conn.commit()

    return drive.id


def update_drive_atomic(
    connect_fn: Callable,
    stack_id: str,
    drive: Drive,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    expected_version: Optional[int] = None,
) -> bool:
    """Update a drive with optimistic concurrency control.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        drive: The drive with updated fields.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation.
        expected_version: The version we expect the record to have.
                         If None, uses drive.version.

    Returns:
        True if update succeeded.

    Raises:
        VersionConflictError: If the record's version doesn't match expected.
    """
    if expected_version is None:
        expected_version = drive.version

    now = now_fn()

    with connect_fn() as conn:
        # Check current version
        current = conn.execute(
            "SELECT version FROM drives WHERE id = ? AND stack_id = ?",
            (drive.id, stack_id),
        ).fetchone()

        if not current:
            return False

        current_version = current["version"]
        if current_version != expected_version:
            raise VersionConflictError("drives", drive.id, expected_version, current_version)

        # Atomic update with version increment
        cursor = conn.execute(
            """
            UPDATE drives SET
                intensity = ?,
                focus_areas = ?,
                updated_at = ?,
                confidence = ?,
                source_type = ?,
                source_episodes = ?,
                derived_from = ?,
                last_verified = ?,
                verification_count = ?,
                confidence_history = ?,
                context = ?,
                context_tags = ?,
                local_updated_at = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND version = ?
            """,
            (
                drive.intensity,
                to_json(drive.focus_areas),
                now,
                drive.confidence,
                drive.source_type,
                to_json(drive.source_episodes),
                to_json(drive.derived_from),
                drive.last_verified.isoformat() if drive.last_verified else None,
                drive.verification_count,
                to_json(drive.confidence_history),
                drive.context,
                to_json(drive.context_tags),
                now,
                drive.id,
                stack_id,
                expected_version,
            ),
        )

        if cursor.rowcount == 0:
            conn.rollback()
            new_current = conn.execute(
                "SELECT version FROM drives WHERE id = ? AND stack_id = ?",
                (drive.id, stack_id),
            ).fetchone()
            actual = new_current["version"] if new_current else -1
            raise VersionConflictError("drives", drive.id, expected_version, actual)

        # Queue for sync
        drive.version = expected_version + 1
        drive_data = to_json(record_to_dict(drive))
        queue_sync(conn, "drives", drive.id, "upsert", data=drive_data)

        conn.commit()

    return True


def get_drives(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    requesting_entity: Optional[str] = None,
) -> List[Drive]:
    """Get all drives."""
    access_filter, access_params = build_access_filter(requesting_entity)
    with connect_fn() as conn:
        rows = conn.execute(
            f"SELECT * FROM drives WHERE stack_id = ? AND deleted = 0{access_filter}",
            [stack_id] + access_params,
        ).fetchall()

    return [_mc_row_to_drive(row) for row in rows]


def get_drive(
    connect_fn: Callable,
    stack_id: str,
    drive_type: str,
) -> Optional[Drive]:
    """Get a specific drive by type."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM drives WHERE stack_id = ? AND drive_type = ? AND deleted = 0",
            (stack_id, drive_type),
        ).fetchone()

    return _mc_row_to_drive(row) if row else None


def row_to_drive(row: sqlite3.Row) -> Drive:
    """Convert a database row to a Drive dataclass."""
    return _mc_row_to_drive(row)
