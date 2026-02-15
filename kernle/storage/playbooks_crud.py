"""Playbooks CRUD operations extracted from SQLiteStorage.

Part of the v0.13.10 architectural decomposition — focused module for
playbook-related database operations.

All functions receive dependencies explicitly (connection factory,
serializers, sync callbacks) to avoid circular imports and enable
independent testing.
"""

import logging
import sqlite3
from typing import Any, Callable, List, Optional

from .base import Playbook
from .memory_crud import _row_to_playbook as _mc_row_to_playbook

logger = logging.getLogger(__name__)


def save_playbook(
    connect_fn: Callable,
    stack_id: str,
    playbook: Playbook,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
    record_to_dict: Callable,
    queue_sync: Callable,
    save_embedding: Callable,
) -> str:
    """Save a playbook. Returns the playbook ID.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        playbook: The Playbook to save.
        now_fn: Returns current UTC timestamp as ISO string.
        to_json: Serializes objects to JSON strings.
        record_to_dict: Converts a dataclass to a dict for sync.
        queue_sync: Queues a sync operation (conn, table, id, op, data).
        save_embedding: Saves embedding for search (conn, table, id, content).
    """
    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO playbooks
            (id, stack_id, name, description, trigger_conditions, steps, failure_modes,
             recovery_steps, mastery_level, times_used, success_rate, source_episodes, tags,
             confidence, last_used, created_at,
             subject_ids, access_grants, consent_grants,
             local_updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                playbook.id,
                stack_id,
                playbook.name,
                playbook.description,
                to_json(playbook.trigger_conditions),
                to_json(playbook.steps),
                to_json(playbook.failure_modes),
                to_json(playbook.recovery_steps),
                playbook.mastery_level,
                playbook.times_used,
                playbook.success_rate,
                to_json(playbook.source_episodes),
                to_json(playbook.tags),
                playbook.confidence,
                playbook.last_used.isoformat() if playbook.last_used else None,
                playbook.created_at.isoformat() if playbook.created_at else now,
                to_json(getattr(playbook, "subject_ids", None)),
                to_json(getattr(playbook, "access_grants", None)),
                to_json(getattr(playbook, "consent_grants", None)),
                now,
                None,  # cloud_synced_at
                playbook.version,
                0,  # deleted
            ),
        )

        # Queue for sync with record data
        playbook_data = to_json(record_to_dict(playbook))
        queue_sync(conn, "playbooks", playbook.id, "upsert", data=playbook_data)

        # Add embedding for search
        content = f"{playbook.name} {playbook.description} {' '.join(playbook.trigger_conditions)}"
        save_embedding(conn, "playbooks", playbook.id, content)

        conn.commit()

    return playbook.id


def get_playbook(
    connect_fn: Callable,
    stack_id: str,
    playbook_id: str,
) -> Optional[Playbook]:
    """Get a specific playbook by ID."""
    with connect_fn() as conn:
        cur = conn.execute(
            "SELECT * FROM playbooks WHERE id = ? AND stack_id = ? AND deleted = 0",
            (playbook_id, stack_id),
        )
        row = cur.fetchone()

    return _mc_row_to_playbook(row) if row else None


def list_playbooks(
    connect_fn: Callable,
    stack_id: str,
    build_access_filter: Callable,
    tags: Optional[List[str]] = None,
    limit: int = 100,
    requesting_entity: Optional[str] = None,
) -> List[Playbook]:
    """Get playbooks, optionally filtered by tags."""
    access_filter, access_params = build_access_filter(requesting_entity)
    with connect_fn() as conn:
        query = f"""
            SELECT * FROM playbooks
            WHERE stack_id = ? AND deleted = 0{access_filter}
            ORDER BY times_used DESC, created_at DESC
            LIMIT ?
        """
        cur = conn.execute(query, [stack_id] + access_params + [limit])
        rows = cur.fetchall()

    playbooks = [_mc_row_to_playbook(row) for row in rows]

    # Filter by tags if provided
    if tags:
        tags_set = set(tags)
        playbooks = [p for p in playbooks if p.tags and tags_set.intersection(p.tags)]

    return playbooks


def search_playbooks(
    connect_fn: Callable,
    stack_id: str,
    query: str,
    limit: int = 10,
    *,
    has_vec: bool = False,
    embed_text: Optional[Callable] = None,
    pack_embedding_fn: Optional[Callable] = None,
    get_playbook_fn: Optional[Callable] = None,
    tokenize_query: Optional[Callable] = None,
    build_token_filter: Optional[Callable] = None,
    token_match_score: Optional[Callable] = None,
    escape_like_pattern: Optional[Callable] = None,
) -> List[Playbook]:
    """Search playbooks by name, description, or triggers using semantic search.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        query: The search query string.
        limit: Maximum results.
        has_vec: Whether sqlite-vec is available for vector search.
        embed_text: Callable to generate embeddings from text.
        pack_embedding_fn: Callable to pack embeddings for sqlite-vec.
        get_playbook_fn: Callable to get a playbook by ID (for vector search results).
        tokenize_query: Callable to tokenize a search query into tokens.
        build_token_filter: Callable to build SQL token filter.
        token_match_score: Callable to score token matches.
        escape_like_pattern: Callable to escape LIKE pattern special chars.
    """
    if has_vec:
        # Use vector search
        embedding = embed_text(query, context="playbook-search")
        if not embedding:
            return []
        packed = pack_embedding_fn(embedding)

        # Support both new format (stack_id:playbooks:id) and legacy (playbooks:id)
        new_prefix = f"{stack_id}:playbooks:"
        legacy_prefix = "playbooks:"

        with connect_fn() as conn:
            cur = conn.execute(
                """
                SELECT e.id, e.embedding, distance
                FROM vec_embeddings e
                WHERE (e.id LIKE ? OR e.id LIKE ?)
                ORDER BY distance
                LIMIT ?
            """.replace("distance", f"vec_distance_L2(e.embedding, X'{packed.hex()}')"),
                (f"{new_prefix}%", f"{legacy_prefix}%", limit * 2),
            )

            vec_results = cur.fetchall()

        # Extract playbook IDs from both formats
        playbook_ids = []
        for r in vec_results:
            vec_id = r[0]
            if vec_id.startswith(new_prefix):
                playbook_ids.append(vec_id[len(new_prefix) :])
            elif vec_id.startswith(legacy_prefix):
                playbook_ids.append(vec_id[len(legacy_prefix) :])

        playbooks = []
        for pid in playbook_ids:
            playbook = get_playbook_fn(pid)
            if playbook:
                playbooks.append(playbook)
            if len(playbooks) >= limit:
                break

        return playbooks
    else:
        # Fall back to tokenized text search
        tokens = tokenize_query(query)
        columns = ["name", "description", "trigger_conditions"]
        with connect_fn() as conn:
            if tokens:
                filt, filt_params = build_token_filter(tokens, columns)
            else:
                # All words too short, use full-phrase match
                escaped_query = escape_like_pattern(query)
                search_pattern = f"%{escaped_query}%"
                filt = "(name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' OR trigger_conditions LIKE ? ESCAPE '\\')"
                filt_params = [search_pattern, search_pattern, search_pattern]
            cur = conn.execute(
                f"""
                SELECT * FROM playbooks
                WHERE stack_id = ? AND deleted = 0
                AND {filt}
                ORDER BY times_used DESC
                LIMIT ?
            """,
                [stack_id] + filt_params + [limit],
            )
            rows = cur.fetchall()

        playbooks = [_mc_row_to_playbook(row) for row in rows]
        if tokens:
            # Sort by token match score
            def _score(pb: Playbook) -> float:
                triggers = " ".join(pb.trigger_conditions) if pb.trigger_conditions else ""
                combined = f"{pb.name or ''} {pb.description or ''} {triggers}"
                return token_match_score(combined, tokens)

            playbooks.sort(key=_score, reverse=True)
        return playbooks


def update_playbook_usage(
    connect_fn: Callable,
    stack_id: str,
    playbook_id: str,
    success: bool,
    now_fn: Callable[[], str],
    queue_sync: Callable,
    get_playbook_fn: Callable,
) -> bool:
    """Update playbook usage statistics.

    Args:
        connect_fn: Context manager returning a DB connection.
        stack_id: The stack ID for record isolation.
        playbook_id: The playbook to update.
        success: Whether this usage was successful.
        now_fn: Returns current UTC timestamp as ISO string.
        queue_sync: Queues a sync operation.
        get_playbook_fn: Callable to get a playbook by ID.

    Returns:
        True if the playbook was updated.
    """
    playbook = get_playbook_fn(playbook_id)
    if not playbook:
        return False

    now = now_fn()

    # Calculate new success rate
    new_times_used = playbook.times_used + 1
    if playbook.times_used == 0:
        new_success_rate = 1.0 if success else 0.0
    else:
        # Running average
        total_successes = playbook.success_rate * playbook.times_used
        total_successes += 1.0 if success else 0.0
        new_success_rate = total_successes / new_times_used

    # Update mastery level based on usage and success rate
    new_mastery = playbook.mastery_level
    if new_times_used >= 20 and new_success_rate >= 0.9:
        new_mastery = "expert"
    elif new_times_used >= 10 and new_success_rate >= 0.8:
        new_mastery = "proficient"
    elif new_times_used >= 5 and new_success_rate >= 0.7:
        new_mastery = "competent"

    with connect_fn() as conn:
        conn.execute(
            """
            UPDATE playbooks SET
                times_used = ?,
                success_rate = ?,
                mastery_level = ?,
                last_used = ?,
                local_updated_at = ?,
                version = version + 1
            WHERE id = ? AND stack_id = ?
        """,
            (
                new_times_used,
                new_success_rate,
                new_mastery,
                now,
                now,
                playbook_id,
                stack_id,
            ),
        )

        queue_sync(conn, "playbooks", playbook_id, "upsert")
        conn.commit()

    return True


def row_to_playbook(row: sqlite3.Row) -> Playbook:
    """Convert a database row to a Playbook dataclass."""
    return _mc_row_to_playbook(row)
