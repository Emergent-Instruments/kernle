"""Sync engine for Kernle storage.

SyncEngine class handles sync queue management, push/pull operations,
merge logic, and conflict resolution. Receives the host SQLiteStorage
instance to access DB connection and record operations.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .base import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Playbook,
    QueuedChange,
    Relationship,
    SyncConflict,
    SyncResult,
    Value,
)

logger = logging.getLogger(__name__)

# Maximum size for merged arrays during sync to prevent resource exhaustion
MAX_SYNC_ARRAY_SIZE = 500

# Tables that are intentionally kept local and are not pushed to cloud storage.
LOCAL_ONLY_SYNC_TABLES = frozenset({"memory_suggestions"})

# Array fields that should be merged (set union) during sync rather than overwritten
SYNC_ARRAY_FIELDS: Dict[str, List[str]] = {
    "episodes": [
        "lessons",
        "tags",
        "emotional_tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "beliefs": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "notes": [
        "tags",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "drives": [
        "focus_areas",
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "agent_values": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "goals": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "relationships": [
        "source_episodes",
        "derived_from",
        "context_tags",
    ],
    "playbooks": [
        "trigger_conditions",
        "failure_modes",
        "recovery_steps",
        "source_episodes",
        "tags",
    ],
}


class SyncEngine:
    """Sync engine handling queue management, push/pull, merge, and conflict resolution.

    Args:
        host: The SQLiteStorage instance providing DB access and record operations.
        validate_table_name_fn: Callable to validate table names against allowlist.
    """

    def __init__(self, host, validate_table_name_fn: Callable):
        self._host = host
        self._validate_table_name = validate_table_name_fn

    # === Queue Operations ===

    def queue_sync_operation(
        self, operation: str, table: str, record_id: str, data: Optional[Dict[str, Any]] = None
    ) -> int:
        """Queue a sync operation for later synchronization."""
        if table in LOCAL_ONLY_SYNC_TABLES:
            return 0

        now = self._host._now()
        data_json = self._host._to_json(data) if data else None

        with self._host._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO sync_queue
                   (table_name, record_id, operation, data, local_updated_at, synced, queued_at)
                   VALUES (?, ?, ?, ?, ?, 0, ?)
                   ON CONFLICT(table_name, record_id) WHERE synced = 0
                   DO UPDATE SET
                       operation = excluded.operation,
                       data = excluded.data,
                       local_updated_at = excluded.local_updated_at,
                       queued_at = excluded.queued_at""",
                (table, record_id, operation, data_json, now, now),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_pending_sync_operations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all unsynced operations from the queue."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, operation, table_name, record_id, data, local_updated_at
                   FROM sync_queue
                   WHERE synced = 0
                   ORDER BY id
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "operation": row["operation"],
                "table_name": row["table_name"],
                "record_id": row["record_id"],
                "data": self._host._from_json(row["data"]) if row["data"] else None,
                "local_updated_at": self._host._parse_datetime(row["local_updated_at"]),
            }
            for row in rows
        ]

    def mark_synced(self, ids: List[int]) -> int:
        """Mark sync queue entries as synced."""
        if not ids:
            return 0

        with self._host._connect() as conn:
            placeholders = ",".join("?" * len(ids))
            cursor = conn.execute(
                f"UPDATE sync_queue SET synced = 1 WHERE id IN ({placeholders})", ids
            )
            conn.commit()
            return cursor.rowcount

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync queue status with counts."""
        with self._host._connect() as conn:
            pending = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0").fetchone()[0]
            synced = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 1").fetchone()[0]

            table_rows = conn.execute("""SELECT table_name, COUNT(*) as count
                   FROM sync_queue WHERE synced = 0
                   GROUP BY table_name""").fetchall()
            by_table = {row["table_name"]: row["count"] for row in table_rows}

            op_rows = conn.execute("""SELECT operation, COUNT(*) as count
                   FROM sync_queue WHERE synced = 0
                   GROUP BY operation""").fetchall()
            by_operation = {row["operation"]: row["count"] for row in op_rows}

        return {
            "pending": pending,
            "synced": synced,
            "total": pending + synced,
            "by_table": by_table,
            "by_operation": by_operation,
        }

    def get_pending_sync_count(self) -> int:
        """Get count of records pending sync."""
        with self._host._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sync_queue WHERE synced = 0").fetchone()[0]
        return count

    def get_queued_changes(self, limit: int = 100, max_retries: int = 5) -> List[QueuedChange]:
        """Get queued changes for sync."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, operation,
                          COALESCE(payload, data) as payload, queued_at,
                          COALESCE(retry_count, 0) as retry_count,
                          last_error, last_attempt_at
                   FROM sync_queue
                   WHERE synced = 0 AND COALESCE(retry_count, 0) < ?
                   ORDER BY id
                   LIMIT ?""",
                (max_retries, limit),
            ).fetchall()

        return [
            QueuedChange(
                id=row["id"],
                table_name=row["table_name"],
                record_id=row["record_id"],
                operation=row["operation"],
                payload=row["payload"],
                queued_at=self._host._parse_datetime(row["queued_at"]),
                retry_count=row["retry_count"] or 0,
                last_error=row["last_error"],
                last_attempt_at=self._host._parse_datetime(row["last_attempt_at"]),
            )
            for row in rows
        ]

    def _clear_queued_change(self, conn: sqlite3.Connection, queue_id: int):
        """Mark a change as synced."""
        conn.execute("UPDATE sync_queue SET synced = 1 WHERE id = ?", (queue_id,))

    def _record_sync_failure(self, conn: sqlite3.Connection, queue_id: int, error: str) -> int:
        """Record a sync failure and increment retry count."""
        now = self._host._now()
        conn.execute(
            """UPDATE sync_queue
               SET retry_count = COALESCE(retry_count, 0) + 1,
                   last_error = ?,
                   last_attempt_at = ?
               WHERE id = ?""",
            (error[:500], now, queue_id),
        )
        row = conn.execute(
            "SELECT retry_count FROM sync_queue WHERE id = ?", (queue_id,)
        ).fetchone()
        return row["retry_count"] if row else 0

    def get_failed_sync_records(self, min_retries: int = 5) -> List[QueuedChange]:
        """Get sync records that have exceeded max retries."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, operation,
                          COALESCE(payload, data) as payload, queued_at,
                          COALESCE(retry_count, 0) as retry_count,
                          last_error, last_attempt_at
                   FROM sync_queue
                   WHERE synced = 0 AND COALESCE(retry_count, 0) >= ?
                   ORDER BY last_attempt_at DESC
                   LIMIT 100""",
                (min_retries,),
            ).fetchall()

        return [
            QueuedChange(
                id=row["id"],
                table_name=row["table_name"],
                record_id=row["record_id"],
                operation=row["operation"],
                payload=row["payload"],
                queued_at=self._host._parse_datetime(row["queued_at"]),
                retry_count=row["retry_count"] or 0,
                last_error=row["last_error"],
                last_attempt_at=self._host._parse_datetime(row["last_attempt_at"]),
            )
            for row in rows
        ]

    def clear_failed_sync_records(self, older_than_days: int = 7) -> int:
        """Clear failed sync records older than the specified days."""
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff_str = cutoff.isoformat()

        with self._host._connect() as conn:
            cursor = conn.execute(
                """UPDATE sync_queue
                   SET synced = 1
                   WHERE synced = 0
                     AND COALESCE(retry_count, 0) >= 5
                     AND last_attempt_at < ?""",
                (cutoff_str,),
            )
            count = cursor.rowcount
            conn.commit()

        return count

    # === Sync Metadata ===

    def _get_sync_meta(self, key: str) -> Optional[str]:
        """Get a sync metadata value."""
        with self._host._connect() as conn:
            row = conn.execute("SELECT value FROM sync_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_sync_meta(self, key: str, value: str):
        """Set a sync metadata value."""
        with self._host._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_meta (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, self._host._now()),
            )
            conn.commit()

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync."""
        value = self._get_sync_meta("last_sync_time")
        return self._host._parse_datetime(value) if value else None

    # === Conflict Management ===

    def get_sync_conflicts(self, limit: int = 100) -> List[SyncConflict]:
        """Get recent sync conflict history."""
        with self._host._connect() as conn:
            rows = conn.execute(
                """SELECT id, table_name, record_id, local_version, cloud_version,
                          resolution, resolved_at, local_summary, cloud_summary
                   FROM sync_conflicts
                   ORDER BY resolved_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()

        return [
            SyncConflict(
                id=row["id"],
                table=row["table_name"],
                record_id=row["record_id"],
                local_version=json.loads(row["local_version"]),
                cloud_version=json.loads(row["cloud_version"]),
                resolution=row["resolution"],
                resolved_at=self._host._parse_datetime(row["resolved_at"])
                or datetime.now(timezone.utc),
                local_summary=row["local_summary"],
                cloud_summary=row["cloud_summary"],
            )
            for row in rows
        ]

    def save_sync_conflict(self, conflict: SyncConflict) -> str:
        """Save a sync conflict record."""
        with self._host._connect() as conn:
            conn.execute(
                """INSERT INTO sync_conflicts
                   (id, table_name, record_id, local_version, cloud_version,
                    resolution, resolved_at, local_summary, cloud_summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    conflict.id,
                    conflict.table,
                    conflict.record_id,
                    json.dumps(conflict.local_version),
                    json.dumps(conflict.cloud_version),
                    conflict.resolution,
                    (
                        conflict.resolved_at.isoformat()
                        if isinstance(conflict.resolved_at, datetime)
                        else conflict.resolved_at
                    ),
                    conflict.local_summary,
                    conflict.cloud_summary,
                ),
            )
            conn.commit()
        return conflict.id

    def clear_sync_conflicts(self, before: Optional[datetime] = None) -> int:
        """Clear sync conflict history."""
        with self._host._connect() as conn:
            if before:
                cursor = conn.execute(
                    "DELETE FROM sync_conflicts WHERE resolved_at < ?", (before.isoformat(),)
                )
            else:
                cursor = conn.execute("DELETE FROM sync_conflicts")
            conn.commit()
            return cursor.rowcount

    # === Connectivity ===

    def is_online(self) -> bool:
        """Check if cloud storage is reachable."""
        if not self._host.cloud_storage:
            return False

        now = datetime.now(timezone.utc)
        if self._host._last_connectivity_check:
            elapsed = (now - self._host._last_connectivity_check).total_seconds()
            if elapsed < self._host._connectivity_cache_ttl:
                return self._host._is_online_cached

        try:
            import socket

            old_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(self._host.CONNECTIVITY_TIMEOUT)
                self._host.cloud_storage.get_stats()
                self._host._is_online_cached = True
            except Exception as e:
                logger.debug(f"Connectivity check failed: {e}")
                self._host._is_online_cached = False
            finally:
                socket.setdefaulttimeout(old_timeout)
        except Exception as e:
            logger.debug(f"Connectivity check error: {e}")
            self._host._is_online_cached = False

        self._host._last_connectivity_check = now
        return self._host._is_online_cached

    # === Core Push/Pull ===

    def _mark_synced(self, conn: sqlite3.Connection, table: str, record_id: str):
        """Mark a record as synced with the cloud."""
        self._validate_table_name(table)
        now = self._host._now()
        conn.execute(
            f"UPDATE {table} SET cloud_synced_at = ? WHERE id = ? AND stack_id = ?",
            (now, record_id, self._host.stack_id),
        )

    def _get_record_for_push(self, table: str, record_id: str) -> Optional[Any]:
        """Get a record by table and ID for pushing to cloud."""
        self._validate_table_name(table)
        with self._host._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND stack_id = ?",
                (record_id, self._host.stack_id),
            ).fetchone()

        if not row:
            return None

        converters = {
            "episodes": self._host._row_to_episode,
            "notes": self._host._row_to_note,
            "beliefs": self._host._row_to_belief,
            "agent_values": self._host._row_to_value,
            "goals": self._host._row_to_goal,
            "drives": self._host._row_to_drive,
            "relationships": self._host._row_to_relationship,
            "playbooks": self._host._row_to_playbook,
            "raw_entries": self._host._row_to_raw_entry,
        }

        converter = converters.get(table)
        return converter(row) if converter else None

    def _push_record(self, table: str, record: Any) -> bool:
        """Push a single record to cloud storage."""
        if not self._host.cloud_storage:
            return False

        try:
            if table == "episodes":
                self._host.cloud_storage.save_episode(record)
            elif table == "notes":
                self._host.cloud_storage.save_note(record)
            elif table == "beliefs":
                self._host.cloud_storage.save_belief(record)
            elif table == "agent_values":
                self._host.cloud_storage.save_value(record)
            elif table == "goals":
                self._host.cloud_storage.save_goal(record)
            elif table == "drives":
                self._host.cloud_storage.save_drive(record)
            elif table == "relationships":
                self._host.cloud_storage.save_relationship(record)
            elif table == "playbooks":
                self._host.cloud_storage.save_playbook(record)
            else:
                logger.warning(f"Unknown table for push: {table}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to push record {table}:{record.id}: {e}")
            return False

    def sync(self) -> SyncResult:
        """Sync with cloud storage."""
        result = SyncResult()

        if not self._host.cloud_storage:
            logger.debug("No cloud storage configured, skipping sync")
            return result

        if not self.is_online():
            logger.info("Offline - sync skipped, changes queued")
            result.errors.append("Offline - cannot reach cloud storage")
            return result

        # Phase 1: Push queued changes
        queued = self.get_queued_changes(limit=100, max_retries=5)
        failed_count = len(self.get_failed_sync_records(min_retries=5))
        if failed_count > 0:
            logger.info(f"Skipping {failed_count} records that exceeded max retries")

        logger.debug(f"Pushing {len(queued)} queued changes")

        with self._host._connect() as conn:
            for change in queued:
                try:
                    if change.table_name in LOCAL_ONLY_SYNC_TABLES:
                        retry_count = self._record_sync_failure(
                            conn,
                            change.id,
                            f"Table {change.table_name} is local-only and cannot be pushed to cloud",
                        )
                        result.errors.append(
                            f"Skipped push for local-only table {change.table_name}:{change.record_id} "
                            f"(retry {retry_count}/5)"
                        )
                        continue

                    record = self._get_record_for_push(change.table_name, change.record_id)

                    if record is None:
                        if change.operation == "delete":
                            self._clear_queued_change(conn, change.id)
                            result.pushed += 1
                        else:
                            self._clear_queued_change(conn, change.id)
                        continue

                    if self._push_record(change.table_name, record):
                        self._mark_synced(conn, change.table_name, change.record_id)
                        self._clear_queued_change(conn, change.id)
                        result.pushed += 1
                    else:
                        retry_count = self._record_sync_failure(
                            conn, change.id, "Push failed - cloud returned error"
                        )
                        if retry_count >= 5:
                            logger.warning(
                                f"Record {change.table_name}:{change.record_id} "
                                f"exceeded max retries, moving to dead letter queue"
                            )
                        result.errors.append(
                            f"Failed to push {change.table_name}:{change.record_id} "
                            f"(retry {retry_count}/5)"
                        )

                except Exception as e:
                    error_msg = str(e)[:500]
                    retry_count = self._record_sync_failure(conn, change.id, error_msg)
                    logger.error(
                        f"Error pushing {change.table_name}:{change.record_id}: {e} "
                        f"(retry {retry_count}/5)"
                    )
                    result.errors.append(
                        f"Error pushing {change.table_name}:{change.record_id}: {error_msg}"
                    )

            conn.commit()

        # Phase 2: Pull remote changes
        pull_result = self.pull_changes()
        result.pulled = pull_result.pulled
        result.conflicts = pull_result.conflicts
        result.errors.extend(pull_result.errors)

        if result.success or (result.pushed > 0 or result.pulled > 0):
            self._set_sync_meta("last_sync_time", self._host._now())

        logger.info(
            f"Sync complete: pushed={result.pushed}, pulled={result.pulled}, conflicts={result.conflict_count}"
        )
        return result

    def pull_changes(self, since: Optional[datetime] = None) -> SyncResult:
        """Pull changes from cloud since the given timestamp."""
        result = SyncResult()

        if not self._host.cloud_storage:
            return result

        if since is None:
            since = self.get_last_sync_time()

        tables_and_getters = [
            ("episodes", self._host.cloud_storage.get_episodes, self._merge_episode),
            ("notes", self._host.cloud_storage.get_notes, self._merge_note),
            ("beliefs", self._host.cloud_storage.get_beliefs, self._merge_belief),
            ("agent_values", self._host.cloud_storage.get_values, self._merge_value),
            ("goals", self._host.cloud_storage.get_goals, self._merge_goal),
            ("drives", self._host.cloud_storage.get_drives, self._merge_drive),
            ("relationships", self._host.cloud_storage.get_relationships, self._merge_relationship),
            ("playbooks", self._host.cloud_storage.list_playbooks, self._merge_playbook),
        ]

        for table, getter, merger in tables_and_getters:
            try:
                if table == "episodes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "notes":
                    cloud_records = getter(limit=1000, since=since)
                elif table == "goals":
                    cloud_records = getter(status=None, limit=1000)
                else:
                    cloud_records = getter(limit=1000) if callable(getter) else getter()

                for cloud_record in cloud_records:
                    pull_count, conflict = merger(cloud_record)
                    result.pulled += pull_count
                    if conflict:
                        result.conflicts.append(conflict)

            except Exception as e:
                logger.error(f"Failed to pull from {table}: {e}")
                result.errors.append(f"Failed to pull {table}: {str(e)}")

        return result

    # === Merge Methods ===

    def _merge_episode(self, cloud_record: Episode) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud episode with local."""
        return self._merge_record("episodes", cloud_record, self._host.get_episode)

    def _merge_note(self, cloud_record: Note) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud note with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_note(row)
        return self._merge_generic(
            "notes", cloud_record, local, lambda: self._host.save_note(cloud_record)
        )

    def _merge_belief(self, cloud_record: Belief) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud belief with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM beliefs WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_belief(row)
        return self._merge_generic(
            "beliefs", cloud_record, local, lambda: self._host.save_belief(cloud_record)
        )

    def _merge_value(self, cloud_record: Value) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud value with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agent_values WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_value(row)
        return self._merge_generic(
            "agent_values", cloud_record, local, lambda: self._host.save_value(cloud_record)
        )

    def _merge_goal(self, cloud_record: Goal) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud goal with local."""
        local = None
        with self._host._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ? AND stack_id = ?",
                (cloud_record.id, self._host.stack_id),
            ).fetchone()
            if row:
                local = self._host._row_to_goal(row)
        return self._merge_generic(
            "goals", cloud_record, local, lambda: self._host.save_goal(cloud_record)
        )

    def _merge_drive(self, cloud_record: Drive) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud drive with local."""
        local = self._host.get_drive(cloud_record.drive_type)
        return self._merge_generic(
            "drives", cloud_record, local, lambda: self._host.save_drive(cloud_record)
        )

    def _merge_relationship(self, cloud_record: Relationship) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud relationship with local."""
        local = self._host.get_relationship(cloud_record.entity_name)
        return self._merge_generic(
            "relationships", cloud_record, local, lambda: self._host.save_relationship(cloud_record)
        )

    def _merge_playbook(self, cloud_record: Playbook) -> tuple[int, Optional[SyncConflict]]:
        """Merge a cloud playbook with local."""
        local = self._host.get_playbook(cloud_record.id)
        return self._merge_generic(
            "playbooks", cloud_record, local, lambda: self._host.save_playbook(cloud_record)
        )

    def _merge_record(
        self, table: str, cloud_record: Any, get_local
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge for records with an ID-based getter."""
        local = get_local(cloud_record.id)
        return self._merge_generic(
            table, cloud_record, local, lambda: self._save_from_cloud(table, cloud_record)
        )

    def _merge_array_fields(self, table: str, winner: Any, loser: Any) -> Any:
        """Merge array fields from loser into winner using set union."""
        from dataclasses import replace

        array_fields = SYNC_ARRAY_FIELDS.get(table, [])
        if not array_fields:
            return winner

        updates = {}
        for field_name in array_fields:
            if not hasattr(winner, field_name) or not hasattr(loser, field_name):
                continue

            winner_val = getattr(winner, field_name)
            loser_val = getattr(loser, field_name)

            if not loser_val:
                continue
            if not winner_val:
                updates[field_name] = list(loser_val)
                continue

            try:
                if winner_val and isinstance(winner_val[0], dict):
                    seen = set()
                    merged = []
                    for item in winner_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    for item in loser_val:
                        key = json.dumps(item, sort_keys=True)
                        if key not in seen:
                            seen.add(key)
                            merged.append(item)
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
                else:
                    merged = list(set(winner_val) | set(loser_val))
                    if len(merged) > MAX_SYNC_ARRAY_SIZE:
                        logger.warning(
                            f"Array field {field_name} exceeded max size ({len(merged)} > {MAX_SYNC_ARRAY_SIZE}), truncating"
                        )
                        merged = merged[:MAX_SYNC_ARRAY_SIZE]
                    updates[field_name] = merged
            except (TypeError, KeyError):
                logger.warning(f"Failed to merge array field {field_name}, keeping winner's value")
                continue

        if updates:
            return replace(winner, **updates)
        return winner

    def _merge_generic(
        self, table: str, cloud_record: Any, local_record: Optional[Any], save_fn
    ) -> tuple[int, Optional[SyncConflict]]:
        """Generic merge logic with last-write-wins for scalar fields, set union for arrays."""
        if local_record is None:
            save_fn()
            with self._host._connect() as conn:
                self._mark_synced(conn, table, cloud_record.id)
                conn.execute(
                    "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                    (table, cloud_record.id),
                )
                conn.commit()
            return (1, None)

        cloud_time = cloud_record.cloud_synced_at or cloud_record.local_updated_at
        local_time = local_record.local_updated_at

        if cloud_time and local_time:
            if cloud_time > local_time:
                merged_record = self._merge_array_fields(table, cloud_record, local_record)
                conflict = self._create_conflict(
                    table, cloud_record.id, local_record, cloud_record, "cloud_wins_arrays_merged"
                )
                self._save_from_cloud(table, merged_record)
                with self._host._connect() as conn:
                    self._mark_synced(conn, table, cloud_record.id)
                    conn.execute(
                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                        (table, cloud_record.id),
                    )
                    conn.commit()
                self.save_sync_conflict(conflict)
                return (1, conflict)
            else:
                merged_record = self._merge_array_fields(table, local_record, cloud_record)
                if merged_record is not local_record:
                    self._save_from_cloud(table, merged_record)
                conflict = self._create_conflict(
                    table, cloud_record.id, local_record, cloud_record, "local_wins_arrays_merged"
                )
                with self._host._connect() as conn:
                    self._mark_synced(conn, table, cloud_record.id)
                    conn.execute(
                        "DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?",
                        (table, cloud_record.id),
                    )
                    conn.commit()
                self.save_sync_conflict(conflict)
                return (0, conflict)
        elif cloud_time:
            save_fn()
            with self._host._connect() as conn:
                self._mark_synced(conn, table, cloud_record.id)
                conn.commit()
            return (1, None)
        else:
            return (0, None)

    def _create_conflict(
        self, table: str, record_id: str, local_record: Any, cloud_record: Any, resolution: str
    ) -> SyncConflict:
        """Create a SyncConflict record with human-readable summaries."""
        local_summary = self._get_record_summary(table, local_record)
        cloud_summary = self._get_record_summary(table, cloud_record)
        local_dict = self._record_to_dict(local_record)
        cloud_dict = self._record_to_dict(cloud_record)
        return SyncConflict(
            id=str(uuid.uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_dict,
            cloud_version=cloud_dict,
            resolution=resolution,
            resolved_at=datetime.now(timezone.utc),
            local_summary=local_summary,
            cloud_summary=cloud_summary,
        )

    def _get_record_summary(self, table: str, record: Any) -> str:
        """Get a human-readable summary of a record for conflict display."""
        if table == "episodes":
            return record.objective[:50] + "..." if len(record.objective) > 50 else record.objective
        elif table == "notes":
            return record.content[:50] + "..." if len(record.content) > 50 else record.content
        elif table == "beliefs":
            return record.statement[:50] + "..." if len(record.statement) > 50 else record.statement
        elif table == "agent_values":
            stmt = record.statement[:40] + "..." if len(record.statement) > 40 else record.statement
            return f"{record.name}: {stmt}"
        elif table == "goals":
            return record.title[:50] + "..." if len(record.title) > 50 else record.title
        elif table == "drives":
            return f"{record.drive_type} (intensity: {record.intensity})"
        elif table == "relationships":
            return f"{record.entity_name} ({record.relationship_type})"
        elif table == "playbooks":
            if record.name:
                return f"{record.name} :: {record.description[:60] + '...' if len(record.description) > 60 else record.description}"
            return f"{table}:{record.id}"
        return f"{table}:{record.id}"

    def _record_to_dict(self, record: Any) -> Dict[str, Any]:
        """Convert a dataclass record to a dict for JSON storage."""
        from dataclasses import asdict

        try:
            d = asdict(record)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            return d
        except Exception as e:
            logger.debug(f"Failed to serialize record, using fallback: {e}")
            return {"id": getattr(record, "id", "unknown")}

    def _save_from_cloud(self, table: str, record: Any):
        """Save a record that came from cloud."""
        if table == "episodes":
            self._host.save_episode(record)
        elif table == "notes":
            self._host.save_note(record)
        elif table == "beliefs":
            self._host.save_belief(record)
        elif table == "agent_values":
            self._host.save_value(record)
        elif table == "goals":
            self._host.save_goal(record)
        elif table == "drives":
            self._host.save_drive(record)
        elif table == "relationships":
            self._host.save_relationship(record)
        elif table == "playbooks":
            self._host.save_playbook(record)
