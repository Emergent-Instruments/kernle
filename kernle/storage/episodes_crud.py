"""Episodes CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
episode-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

from .base import Episode, VersionConflictError
from .memory_crud import _row_to_episode as _mc_row_to_episode

logger = logging.getLogger(__name__)


def save_episode(
    connect_fn: Callable,
    stack_id: str,
    episode: Episode,
    now_fn: Callable[[], str],
    parse_datetime_fn: Callable,
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    get_searchable_content: Callable,
    *,
    lineage_checker: Optional[Callable] = None,
) -> str:
    """Save an episode.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        episode: The Episode to save.
        now_fn: Returns current UTC timestamp as ISO string.
        parse_datetime_fn: Parses ISO string to datetime.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
        get_searchable_content: Gets searchable text from a record (type, record).
        lineage_checker: Optional callable for derived-from cycle detection.
            Receives (storage, memory_type, id, derived_from).
    """
    if not episode.id:
        episode.id = str(uuid.uuid4())

    if episode.derived_from and lineage_checker:
        lineage_checker("episode", episode.id, episode.derived_from)

    now = now_fn()
    episode.local_updated_at = parse_datetime_fn(now)

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO episodes
            (id, stack_id, objective, outcome, outcome_type, lessons, tags,
             emotional_valence, emotional_arousal, emotional_tags,
             confidence, source_type, source_episodes, derived_from,
             last_verified, verification_count, confidence_history,
             times_accessed, last_accessed, is_protected, strength,
             processed, context, context_tags,
             source_entity, subject_ids, access_grants, consent_grants,
             epoch_id, repeat, avoid,
             created_at, local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                episode.id,
                stack_id,
                episode.objective,
                episode.outcome,
                episode.outcome_type,
                to_json(episode.lessons),
                to_json(episode.tags),
                episode.emotional_valence,
                episode.emotional_arousal,
                to_json(episode.emotional_tags),
                episode.confidence,
                episode.source_type,
                to_json(episode.source_episodes),
                to_json(episode.derived_from),
                episode.last_verified.isoformat() if episode.last_verified else None,
                episode.verification_count,
                to_json(episode.confidence_history),
                episode.times_accessed,
                episode.last_accessed.isoformat() if episode.last_accessed else None,
                1 if episode.is_protected else 0,
                episode.strength,
                1 if episode.processed else 0,
                episode.context,
                to_json(episode.context_tags),
                getattr(episode, "source_entity", None),
                to_json(getattr(episode, "subject_ids", None)),
                to_json(getattr(episode, "access_grants", None)),
                to_json(getattr(episode, "consent_grants", None)),
                episode.epoch_id,
                to_json(episode.repeat),
                to_json(episode.avoid),
                episode.created_at.isoformat() if episode.created_at else now,
                now,
                episode.cloud_synced_at.isoformat() if episode.cloud_synced_at else None,
                episode.version,
                1 if episode.deleted else 0,
            ),
        )
        # Queue for sync with record data
        episode_data = to_json(record_to_dict(episode))
        queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)

        # Save embedding for search
        content = get_searchable_content("episode", episode)
        save_embedding(conn, "episodes", episode.id, content)

        conn.commit()

    return episode.id


def update_episode_atomic(
    connect_fn: Callable,
    stack_id: str,
    episode: Episode,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    from_json: Callable[[Optional[str]], Any],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    get_searchable_content: Callable,
    expected_version: Optional[int] = None,
) -> bool:
    """Update an episode with optimistic concurrency control.

    This method performs an atomic update that:
    1. Checks if the current version matches expected_version
    2. Increments the version atomically
    3. Updates all other fields

    Security: Provenance fields have special handling:
    - source_type: Write-once (preserved from original)
    - derived_from: Append-only (merged with original)
    - confidence_history: Append-only (merged with original)

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        episode: The episode with updated fields.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        from_json: Deserializes JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation.
        save_embedding: Saves embedding for search.
        get_searchable_content: Gets searchable text from a record.
        expected_version: The version we expect the record to have.
                         If None, uses episode.version.

    Returns:
        True if update succeeded.

    Raises:
        VersionConflictError: If the record's version doesn't match expected.
    """
    if expected_version is None:
        expected_version = episode.version

    now = now_fn()

    with connect_fn() as conn:
        # First check current version and get original provenance fields
        current = conn.execute(
            """SELECT version, source_type, derived_from, confidence_history
               FROM episodes WHERE id = ? AND stack_id = ?""",
            (episode.id, stack_id),
        ).fetchone()

        if not current:
            return False  # Record doesn't exist

        current_version = current["version"]
        if current_version != expected_version:
            raise VersionConflictError("episodes", episode.id, expected_version, current_version)

        # Security: Preserve provenance fields (write-once / append-only)
        # source_type is write-once - always use original
        original_source_type = current["source_type"] or episode.source_type

        # derived_from is append-only - merge lists
        original_derived = from_json(current["derived_from"]) or []
        new_derived = episode.derived_from or []
        merged_derived = list(set(original_derived) | set(new_derived))

        # confidence_history is append-only - merge lists
        original_history = from_json(current["confidence_history"]) or []
        new_history = episode.confidence_history or []
        # For history, append new entries that aren't already present
        merged_history = original_history + [h for h in new_history if h not in original_history]

        # Atomic update with version increment
        cursor = conn.execute(
            """
            UPDATE episodes SET
                objective = ?,
                outcome = ?,
                outcome_type = ?,
                lessons = ?,
                tags = ?,
                emotional_valence = ?,
                emotional_arousal = ?,
                emotional_tags = ?,
                confidence = ?,
                source_type = ?,
                source_episodes = ?,
                derived_from = ?,
                last_verified = ?,
                verification_count = ?,
                confidence_history = ?,
                times_accessed = ?,
                last_accessed = ?,
                is_protected = ?,
                strength = ?,
                context = ?,
                context_tags = ?,
                local_updated_at = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ? AND version = ?
            """,
            (
                episode.objective,
                episode.outcome,
                episode.outcome_type,
                to_json(episode.lessons),
                to_json(episode.tags),
                episode.emotional_valence,
                episode.emotional_arousal,
                to_json(episode.emotional_tags),
                episode.confidence,
                original_source_type,  # Write-once: preserve original
                to_json(episode.source_episodes),
                to_json(merged_derived),  # Append-only: merged
                episode.last_verified.isoformat() if episode.last_verified else None,
                episode.verification_count,
                to_json(merged_history),  # Append-only: merged
                episode.times_accessed,
                episode.last_accessed.isoformat() if episode.last_accessed else None,
                1 if episode.is_protected else 0,
                episode.strength,
                episode.context,
                to_json(episode.context_tags),
                now,
                episode.id,
                stack_id,
                expected_version,
            ),
        )

        if cursor.rowcount == 0:
            # Version changed between check and update (rare but possible)
            conn.rollback()
            new_current = conn.execute(
                "SELECT version FROM episodes WHERE id = ? AND stack_id = ?",
                (episode.id, stack_id),
            ).fetchone()
            actual = new_current["version"] if new_current else -1
            raise VersionConflictError("episodes", episode.id, expected_version, actual)

        # Queue for sync
        episode.version = expected_version + 1
        episode_data = to_json(record_to_dict(episode))
        queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)

        # Update embedding
        content = get_searchable_content("episode", episode)
        save_embedding(conn, "episodes", episode.id, content)

        conn.commit()

    return True


def save_episodes_batch(
    connect_fn: Callable,
    stack_id: str,
    episodes: List[Episode],
    now_fn: Callable[[], str],
    parse_datetime_fn: Callable,
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
    get_searchable_content: Callable,
) -> List[str]:
    """Save multiple episodes in a single transaction."""
    if not episodes:
        return []
    now = now_fn()
    ids = []
    with connect_fn() as conn:
        for episode in episodes:
            if not episode.id:
                episode.id = str(uuid.uuid4())
            ids.append(episode.id)
            episode.local_updated_at = parse_datetime_fn(now)
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes
                (id, stack_id, objective, outcome, outcome_type, lessons, tags,
                 emotional_valence, emotional_arousal, emotional_tags,
                 confidence, source_type, source_episodes, derived_from,
                 last_verified, verification_count, confidence_history,
                 times_accessed, last_accessed, is_protected, strength,
                 processed, context, context_tags,
                 epoch_id, repeat, avoid,
                 created_at, local_updated_at, cloud_synced_at, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.id,
                    stack_id,
                    episode.objective,
                    episode.outcome,
                    episode.outcome_type,
                    to_json(episode.lessons),
                    to_json(episode.tags),
                    episode.emotional_valence,
                    episode.emotional_arousal,
                    to_json(episode.emotional_tags),
                    episode.confidence,
                    episode.source_type,
                    to_json(episode.source_episodes),
                    to_json(episode.derived_from),
                    episode.last_verified.isoformat() if episode.last_verified else None,
                    episode.verification_count,
                    to_json(episode.confidence_history),
                    episode.times_accessed,
                    episode.last_accessed.isoformat() if episode.last_accessed else None,
                    1 if episode.is_protected else 0,
                    episode.strength,
                    1 if episode.processed else 0,
                    episode.context,
                    to_json(episode.context_tags),
                    episode.epoch_id,
                    to_json(episode.repeat),
                    to_json(episode.avoid),
                    episode.created_at.isoformat() if episode.created_at else now,
                    now,
                    episode.cloud_synced_at.isoformat() if episode.cloud_synced_at else None,
                    episode.version,
                    1 if episode.deleted else 0,
                ),
            )
            episode_data = to_json(record_to_dict(episode))
            queue_sync(conn, "episodes", episode.id, "upsert", data=episode_data)
            content = get_searchable_content("episode", episode)
            save_embedding(conn, "episodes", episode.id, content)
        conn.commit()
    return ids


def get_episodes(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    limit: int = 100,
    since: Optional[datetime] = None,
    tags: Optional[List[str]] = None,
    requesting_entity: Optional[str] = None,
    processed: Optional[bool] = None,
) -> List[Episode]:
    """Get episodes with optional filters."""
    query = "SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    # Apply privacy filter
    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter
    params.extend(access_params)

    if since:
        query += " AND created_at >= ?"
        params.append(since.isoformat())

    if processed is not None:
        query += " AND processed = ?"
        params.append(1 if processed else 0)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    episodes = [_mc_row_to_episode(row) for row in rows]

    # Filter by tags in Python (SQLite JSON support is limited)
    if tags:
        episodes = [e for e in episodes if e.tags and any(t in e.tags for t in tags)]

    return episodes


def get_episode(
    connect_fn: Callable,
    stack_id: str,
    episode_id: str,
    build_access_filter: Callable,
    requesting_entity: Optional[str] = None,
) -> Optional[Episode]:
    """Get a specific episode."""
    query = "SELECT * FROM episodes WHERE id = ? AND stack_id = ?"
    params: List[Any] = [episode_id, stack_id]

    # Apply privacy filter
    access_filter, access_params = build_access_filter(requesting_entity)
    query += access_filter
    params.extend(access_params)

    with connect_fn() as conn:
        row = conn.execute(query, params).fetchone()

    return _mc_row_to_episode(row) if row else None


def get_episodes_by_source_entity(
    connect_fn: Callable,
    stack_id: str,
    source_entity: str,
    limit: int = 500,
) -> List[Episode]:
    """Get episodes associated with a source entity for trust computation."""
    query = """
        SELECT * FROM episodes
        WHERE stack_id = ? AND source_entity = ? AND deleted = 0 AND strength > 0.0
        ORDER BY created_at DESC LIMIT ?
    """
    with connect_fn() as conn:
        rows = conn.execute(query, (stack_id, source_entity, limit)).fetchall()
    return [_mc_row_to_episode(row) for row in rows]


def update_episode_emotion(
    connect_fn: Callable,
    stack_id: str,
    episode_id: str,
    valence: float,
    arousal: float,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    queue_sync: Callable,
    tags: Optional[List[str]] = None,
) -> bool:
    """Update emotional associations for an episode.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        episode_id: The episode to update.
        valence: Emotional valence (-1.0 to 1.0).
        arousal: Emotional arousal (0.0 to 1.0).
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        queue_sync: Queues a sync operation.
        tags: Emotional tags (e.g., ["joy", "excitement"]).

    Returns:
        True if updated, False if episode not found.
    """
    # Clamp values to valid ranges
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))

    now = now_fn()

    with connect_fn() as conn:
        cursor = conn.execute(
            """UPDATE episodes SET
               emotional_valence = ?,
               emotional_arousal = ?,
               emotional_tags = ?,
               local_updated_at = ?,
               version = version + 1
               WHERE id = ? AND stack_id = ? AND deleted = 0""",
            (valence, arousal, to_json(tags), now, episode_id, stack_id),
        )
        if cursor.rowcount > 0:
            queue_sync(conn, "episodes", episode_id, "upsert")
            conn.commit()
            return True
    return False


def search_by_emotion(
    connect_fn: Callable,
    stack_id: str,
    valence_range: Optional[tuple] = None,
    arousal_range: Optional[tuple] = None,
    tags: Optional[List[str]] = None,
    limit: int = 10,
) -> List[Episode]:
    """Find episodes matching emotional criteria.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        valence_range: (min, max) valence filter, e.g. (0.5, 1.0) for positive.
        arousal_range: (min, max) arousal filter, e.g. (0.7, 1.0) for high arousal.
        tags: Emotional tags to match (any match).
        limit: Maximum results.

    Returns:
        List of matching episodes.
    """
    query = "SELECT * FROM episodes WHERE stack_id = ? AND deleted = 0"
    params: List[Any] = [stack_id]

    if valence_range:
        query += " AND emotional_valence >= ? AND emotional_valence <= ?"
        params.extend([valence_range[0], valence_range[1]])

    if arousal_range:
        query += " AND emotional_arousal >= ? AND emotional_arousal <= ?"
        params.extend([arousal_range[0], arousal_range[1]])

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit * 2 if tags else limit)  # Get more if we need to filter by tags

    with connect_fn() as conn:
        rows = conn.execute(query, params).fetchall()

    episodes = [_mc_row_to_episode(row) for row in rows]

    # Filter by emotional tags in Python
    if tags:
        episodes = [
            e for e in episodes if e.emotional_tags and any(t in e.emotional_tags for t in tags)
        ][:limit]

    return episodes


def get_emotional_episodes(
    connect_fn: Callable,
    stack_id: str,
    days: int = 7,
    limit: int = 100,
) -> List[Episode]:
    """Get episodes with emotional data for summary calculations.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        days: Number of days to look back.
        limit: Maximum episodes to retrieve.

    Returns:
        Episodes with non-zero emotional data.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    query = """SELECT * FROM episodes
               WHERE stack_id = ? AND deleted = 0
               AND created_at >= ?
               AND (emotional_valence != 0.0 OR emotional_arousal != 0.0 OR emotional_tags IS NOT NULL)
               ORDER BY created_at DESC
               LIMIT ?"""

    with connect_fn() as conn:
        rows = conn.execute(query, (stack_id, cutoff, limit)).fetchall()

    return [_mc_row_to_episode(row) for row in rows]


def mark_episode_processed(
    connect_fn: Callable,
    stack_id: str,
    episode_id: str,
    mark_processed_fn: Callable,
) -> bool:
    """Mark episode as processed.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        episode_id: The episode to mark as processed.
        mark_processed_fn: The raw_entries.mark_processed function.

    Returns:
        True if the episode was marked as processed.
    """
    with connect_fn() as conn:
        return mark_processed_fn(conn, stack_id, "episodes", episode_id)


def row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a database row to an Episode dataclass."""
    return _mc_row_to_episode(row)
