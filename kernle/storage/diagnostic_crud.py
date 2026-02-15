"""Diagnostic CRUD operations extracted from SQLiteStorage.

Handles diagnostic sessions, diagnostic reports, and episode queries
for trust computation. All functions receive dependencies explicitly
(connection factory, serializers) to avoid circular imports and enable
independent testing.
"""

import json
import logging
import sqlite3
from typing import Callable, List, Optional

from .base import DiagnosticReport, DiagnosticSession, Episode
from .memory_crud import _row_to_diagnostic_report as _mc_row_to_diagnostic_report
from .memory_crud import _row_to_diagnostic_session as _mc_row_to_diagnostic_session
from .memory_crud import _row_to_episode as _mc_row_to_episode

logger = logging.getLogger(__name__)


def save_diagnostic_session(
    connect_fn: Callable,
    stack_id: str,
    session: DiagnosticSession,
    now_fn: Callable[[], str],
) -> str:
    """Save a diagnostic session. Returns the session ID."""
    now = now_fn()
    with connect_fn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO diagnostic_sessions "
            "(id, stack_id, session_type, access_level, status, consent_given, "
            "started_at, completed_at, local_updated_at, cloud_synced_at, "
            "version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session.id,
                stack_id,
                session.session_type,
                session.access_level,
                session.status,
                1 if session.consent_given else 0,
                (session.started_at.isoformat() if session.started_at else now),
                (session.completed_at.isoformat() if session.completed_at else None),
                now,
                None,
                session.version,
                1 if session.deleted else 0,
            ),
        )
    return session.id


def get_diagnostic_session(
    connect_fn: Callable,
    stack_id: str,
    session_id: str,
) -> Optional[DiagnosticSession]:
    """Get a specific diagnostic session by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM diagnostic_sessions " "WHERE id = ? AND stack_id = ? AND deleted = 0",
            (session_id, stack_id),
        ).fetchone()
        if not row:
            return None
        return _mc_row_to_diagnostic_session(row)


def get_diagnostic_sessions(
    connect_fn: Callable,
    stack_id: str,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[DiagnosticSession]:
    """Get diagnostic sessions, optionally filtered by status."""
    with connect_fn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM diagnostic_sessions "
                "WHERE stack_id = ? AND status = ? AND deleted = 0 "
                "ORDER BY started_at DESC LIMIT ?",
                (stack_id, status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM diagnostic_sessions "
                "WHERE stack_id = ? AND deleted = 0 "
                "ORDER BY started_at DESC LIMIT ?",
                (stack_id, limit),
            ).fetchall()
        return [_mc_row_to_diagnostic_session(r) for r in rows]


def complete_diagnostic_session(
    connect_fn: Callable,
    stack_id: str,
    session_id: str,
    now_fn: Callable[[], str],
) -> bool:
    """Mark a diagnostic session as completed. Returns True if updated."""
    now = now_fn()
    with connect_fn() as conn:
        result = conn.execute(
            "UPDATE diagnostic_sessions SET status = 'completed', "
            "completed_at = ?, local_updated_at = ?, version = version + 1 "
            "WHERE id = ? AND stack_id = ? AND deleted = 0 AND status = 'active'",
            (now, now, session_id, stack_id),
        )
        return result.rowcount > 0


def save_diagnostic_report(
    connect_fn: Callable,
    stack_id: str,
    report: DiagnosticReport,
    now_fn: Callable[[], str],
) -> str:
    """Save a diagnostic report. Returns the report ID."""
    now = now_fn()
    with connect_fn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO diagnostic_reports "
            "(id, stack_id, session_id, findings, summary, "
            "created_at, local_updated_at, cloud_synced_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report.id,
                stack_id,
                report.session_id,
                json.dumps(report.findings) if report.findings is not None else None,
                report.summary,
                (report.created_at.isoformat() if report.created_at else now),
                now,
                None,
                report.version,
                1 if report.deleted else 0,
            ),
        )
    return report.id


def get_diagnostic_report(
    connect_fn: Callable,
    stack_id: str,
    report_id: str,
) -> Optional[DiagnosticReport]:
    """Get a specific diagnostic report by ID."""
    with connect_fn() as conn:
        row = conn.execute(
            "SELECT * FROM diagnostic_reports " "WHERE id = ? AND stack_id = ? AND deleted = 0",
            (report_id, stack_id),
        ).fetchone()
        if not row:
            return None
        return _mc_row_to_diagnostic_report(row)


def get_diagnostic_reports(
    connect_fn: Callable,
    stack_id: str,
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[DiagnosticReport]:
    """Get diagnostic reports, optionally filtered by session."""
    with connect_fn() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM diagnostic_reports "
                "WHERE stack_id = ? AND session_id = ? AND deleted = 0 "
                "ORDER BY created_at DESC LIMIT ?",
                (stack_id, session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM diagnostic_reports "
                "WHERE stack_id = ? AND deleted = 0 "
                "ORDER BY created_at DESC LIMIT ?",
                (stack_id, limit),
            ).fetchall()
        return [_mc_row_to_diagnostic_report(r) for r in rows]


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


def row_to_diagnostic_session(row: sqlite3.Row) -> DiagnosticSession:
    """Convert a database row to a DiagnosticSession dataclass."""
    return _mc_row_to_diagnostic_session(row)


def row_to_diagnostic_report(row: sqlite3.Row) -> DiagnosticReport:
    """Convert a database row to a DiagnosticReport dataclass."""
    return _mc_row_to_diagnostic_report(row)
