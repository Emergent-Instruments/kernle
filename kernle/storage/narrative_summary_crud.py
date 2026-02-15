"""Narrative and Summary CRUD operations extracted from SQLiteStorage.

Handles self-narrative management (KEP v3) and fractal summarization.
Combined into one module since both are relatively small. All functions
receive dependencies explicitly (connection factory, serializers) to
avoid circular imports and enable independent testing.
"""

import logging
import sqlite3
import uuid
from typing import Any, Callable, List, Optional

from .base import SelfNarrative, Summary
from .memory_crud import _row_to_self_narrative as _mc_row_to_self_narrative
from .memory_crud import _row_to_summary as _mc_row_to_summary

logger = logging.getLogger(__name__)


# === Self-Narratives (KEP v3) ===


def save_self_narrative(
    connect_fn: Callable,
    stack_id: str,
    narrative: SelfNarrative,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
) -> str:
    """Save a self-narrative. Returns the narrative ID."""
    if not narrative.id:
        narrative.id = str(uuid.uuid4())

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO self_narratives
            (id, stack_id, epoch_id, narrative_type, content,
             key_themes, unresolved_tensions, is_active, supersedes,
             created_at, updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                narrative.id,
                stack_id,
                narrative.epoch_id,
                narrative.narrative_type,
                narrative.content,
                to_json(narrative.key_themes),
                to_json(narrative.unresolved_tensions),
                1 if narrative.is_active else 0,
                narrative.supersedes,
                narrative.created_at.isoformat() if narrative.created_at else now,
                now,
                narrative.cloud_synced_at.isoformat() if narrative.cloud_synced_at else None,
                narrative.version,
                1 if narrative.deleted else 0,
            ),
        )
        conn.commit()

    return narrative.id


def get_self_narrative(
    connect_fn: Callable,
    stack_id: str,
    narrative_id: str,
) -> Optional[SelfNarrative]:
    """Get a specific self-narrative by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM self_narratives WHERE id = ? AND stack_id = ? AND deleted = 0",
            (narrative_id, stack_id),
        ).fetchone()

    return _mc_row_to_self_narrative(row) if row else None


def list_self_narratives(
    connect_fn: Callable,
    stack_id: str,
    narrative_type: Optional[str] = None,
    active_only: bool = True,
) -> List[SelfNarrative]:
    """Get self-narratives, optionally filtered."""
    with connect_fn() as conn:
        conditions = ["stack_id = ?", "deleted = 0"]
        params: list = [stack_id]

        if narrative_type:
            conditions.append("narrative_type = ?")
            params.append(narrative_type)

        if active_only:
            conditions.append("is_active = 1")

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM self_narratives WHERE {where} ORDER BY updated_at DESC",
            params,
        ).fetchall()

    return [_mc_row_to_self_narrative(row) for row in rows]


def deactivate_self_narratives(
    connect_fn: Callable,
    stack_id: str,
    narrative_type: str,
    now_fn: Callable[[], str],
) -> int:
    """Deactivate all active narratives of a given type."""
    now = now_fn()
    with connect_fn() as conn:
        cursor = conn.execute(
            "UPDATE self_narratives SET is_active = 0, updated_at = ? "
            "WHERE stack_id = ? AND narrative_type = ? AND is_active = 1 AND deleted = 0",
            (now, stack_id, narrative_type),
        )
        conn.commit()
        return cursor.rowcount


# === Summaries (Fractal Summarization) ===


def save_summary(
    connect_fn: Callable,
    stack_id: str,
    summary: Summary,
    now_fn: Callable[[], str],
    to_json: Callable[[Any], Optional[str]],
) -> str:
    """Save a summary. Returns the summary ID."""
    if not summary.id:
        summary.id = str(uuid.uuid4())

    now = now_fn()

    with connect_fn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO summaries
            (id, stack_id, scope, period_start, period_end, epoch_id,
             content, key_themes, supersedes, is_protected,
             created_at, updated_at, cloud_synced_at, version, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                summary.id,
                stack_id,
                summary.scope,
                summary.period_start,
                summary.period_end,
                summary.epoch_id,
                summary.content,
                to_json(summary.key_themes),
                to_json(summary.supersedes),
                1 if summary.is_protected else 0,
                summary.created_at.isoformat() if summary.created_at else now,
                now,
                summary.cloud_synced_at.isoformat() if summary.cloud_synced_at else None,
                summary.version,
                1 if summary.deleted else 0,
            ),
        )
        conn.commit()

    return summary.id


def get_summary(
    connect_fn: Callable,
    stack_id: str,
    summary_id: str,
) -> Optional[Summary]:
    """Get a specific summary by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM summaries WHERE id = ? AND stack_id = ? AND deleted = 0",
            (summary_id, stack_id),
        ).fetchone()

    return _mc_row_to_summary(row) if row else None


def list_summaries(
    connect_fn: Callable,
    stack_id: str,
    scope: Optional[str] = None,
) -> List[Summary]:
    """Get summaries, optionally filtered by scope."""
    with connect_fn() as conn:
        if scope:
            rows = conn.execute(
                "SELECT * FROM summaries WHERE stack_id = ? AND scope = ? AND deleted = 0 "
                "ORDER BY period_start DESC",
                (stack_id, scope),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM summaries WHERE stack_id = ? AND deleted = 0 "
                "ORDER BY period_start DESC",
                (stack_id,),
            ).fetchall()

    return [_mc_row_to_summary(row) for row in rows]


def row_to_self_narrative(row: sqlite3.Row) -> SelfNarrative:
    """Convert a database row to a SelfNarrative dataclass."""
    return _mc_row_to_self_narrative(row)


def row_to_summary(row: sqlite3.Row) -> Summary:
    """Convert a database row to a Summary dataclass."""
    return _mc_row_to_summary(row)
