"""Trust assessment CRUD operations extracted from SQLiteStorage.

Handles trust assessment management for entity trust tracking.
All functions receive dependencies explicitly (connection factory,
serializers) to avoid circular imports and enable independent testing.
"""

import json
import logging
import sqlite3
from typing import Callable, List, Optional

from .base import TrustAssessment, parse_datetime

logger = logging.getLogger(__name__)


def save_trust_assessment(
    connect_fn: Callable,
    stack_id: str,
    assessment: TrustAssessment,
    now_fn: Callable[[], str],
) -> str:
    """Save or update a trust assessment. Returns the assessment ID."""
    now = now_fn()
    with connect_fn() as conn:
        existing = conn.execute(
            "SELECT id FROM trust_assessments " "WHERE stack_id = ? AND entity = ? AND deleted = 0",
            (stack_id, assessment.entity),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE trust_assessments SET dimensions = ?, authority = ?, "
                "evidence_episode_ids = ?, last_updated = ?, local_updated_at = ?, "
                "version = version + 1 WHERE id = ?",
                (
                    json.dumps(assessment.dimensions),
                    json.dumps(assessment.authority or []),
                    json.dumps(assessment.evidence_episode_ids or []),
                    now,
                    now,
                    existing["id"],
                ),
            )
            return existing["id"]
        else:
            conn.execute(
                "INSERT INTO trust_assessments "
                "(id, stack_id, entity, dimensions, authority, evidence_episode_ids, "
                "last_updated, created_at, local_updated_at, cloud_synced_at, "
                "version, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    assessment.id,
                    stack_id,
                    assessment.entity,
                    json.dumps(assessment.dimensions),
                    json.dumps(assessment.authority or []),
                    json.dumps(assessment.evidence_episode_ids or []),
                    now,
                    now,
                    now,
                    None,
                    1,
                    0,
                ),
            )
            return assessment.id


def _row_to_trust_assessment(row: sqlite3.Row) -> TrustAssessment:
    """Convert a database row to a TrustAssessment."""
    return TrustAssessment(
        id=row["id"],
        stack_id=row["stack_id"],
        entity=row["entity"],
        dimensions=json.loads(row["dimensions"]),
        authority=(json.loads(row["authority"]) if row["authority"] else []),
        evidence_episode_ids=(
            json.loads(row["evidence_episode_ids"]) if row["evidence_episode_ids"] else []
        ),
        last_updated=(parse_datetime(row["last_updated"]) if row["last_updated"] else None),
        created_at=(parse_datetime(row["created_at"]) if row["created_at"] else None),
        local_updated_at=(
            parse_datetime(row["local_updated_at"]) if row["local_updated_at"] else None
        ),
        cloud_synced_at=(
            parse_datetime(row["cloud_synced_at"]) if row["cloud_synced_at"] else None
        ),
        version=row["version"],
        deleted=bool(row["deleted"]),
    )


def get_trust_assessment(
    connect_fn: Callable,
    stack_id: str,
    entity: str,
) -> Optional[TrustAssessment]:
    """Get a trust assessment for a specific entity."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM trust_assessments " "WHERE stack_id = ? AND entity = ? AND deleted = 0",
            (stack_id, entity),
        ).fetchone()
        if not row:
            return None
        return _row_to_trust_assessment(row)


def get_trust_assessments(
    connect_fn: Callable,
    stack_id: str,
) -> List[TrustAssessment]:
    """Get all trust assessments for the agent."""
    with connect_fn() as conn:
        rows = conn.execute(
            "SELECT * FROM trust_assessments " "WHERE stack_id = ? AND deleted = 0 ORDER BY entity",
            (stack_id,),
        ).fetchall()
        return [_row_to_trust_assessment(r) for r in rows]


def delete_trust_assessment(
    connect_fn: Callable,
    stack_id: str,
    entity: str,
    now_fn: Callable[[], str],
) -> bool:
    """Delete a trust assessment (soft delete)."""
    now = now_fn()
    with connect_fn() as conn:
        result = conn.execute(
            "UPDATE trust_assessments SET deleted = 1, "
            "local_updated_at = ? "
            "WHERE stack_id = ? AND entity = ? AND deleted = 0",
            (now, stack_id, entity),
        )
        return result.rowcount > 0
