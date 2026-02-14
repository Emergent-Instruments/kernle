"""Health check compliance tracking for Kernle storage.

Free functions for logging and querying health check events.
Operates on the isolated health_check_events table.
"""

import logging
import uuid
from typing import Any, Callable, Dict, Optional

from .base import parse_datetime

logger = logging.getLogger(__name__)


def log_health_check(
    connect_fn: Callable,
    stack_id: str,
    now: str,
    anxiety_score: Optional[int] = None,
    source: str = "cli",
    triggered_by: str = "manual",
) -> str:
    """Log a health check event for compliance tracking.

    Args:
        connect_fn: Callable that returns a DB connection context manager.
        stack_id: The stack identifier.
        now: Current timestamp string.
        anxiety_score: The anxiety score at time of check (0-100)
        source: Where the check originated from (cli, mcp)
        triggered_by: What triggered the check (boot, heartbeat, manual)

    Returns:
        The ID of the logged event
    """
    event_id = str(uuid.uuid4())

    with connect_fn() as conn:
        conn.execute(
            """INSERT INTO health_check_events
               (id, stack_id, checked_at, anxiety_score, source, triggered_by)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (event_id, stack_id, now, anxiety_score, source, triggered_by),
        )
        conn.commit()

    logger.debug(
        f"Logged health check: score={anxiety_score}, source={source}, triggered_by={triggered_by}"
    )
    return event_id


def get_health_check_stats(connect_fn: Callable, stack_id: str) -> Dict[str, Any]:
    """Get health check compliance statistics.

    Args:
        connect_fn: Callable that returns a DB connection context manager.
        stack_id: The stack identifier.

    Returns:
        Dict with:
        - total_checks: Total number of health checks
        - avg_per_day: Average checks per day
        - last_check_at: Timestamp of last check
        - last_anxiety_score: Anxiety score at last check
        - checks_by_source: Breakdown by source (cli/mcp)
        - checks_by_trigger: Breakdown by trigger (boot/heartbeat/manual)
    """
    with connect_fn() as conn:
        # Total checks
        cur = conn.execute(
            "SELECT COUNT(*) FROM health_check_events WHERE stack_id = ?", (stack_id,)
        )
        total_checks = cur.fetchone()[0]

        if total_checks == 0:
            return {
                "total_checks": 0,
                "avg_per_day": 0.0,
                "last_check_at": None,
                "last_anxiety_score": None,
                "checks_by_source": {},
                "checks_by_trigger": {},
            }

        # Get first and last check for calculating avg per day
        cur = conn.execute(
            """SELECT MIN(checked_at), MAX(checked_at)
               FROM health_check_events WHERE stack_id = ?""",
            (stack_id,),
        )
        first_check, last_check = cur.fetchone()

        # Calculate days spanned
        if first_check and last_check:
            first_dt = parse_datetime(first_check)
            last_dt = parse_datetime(last_check)
            if first_dt and last_dt:
                days_spanned = max(1, (last_dt - first_dt).days + 1)
                avg_per_day = total_checks / days_spanned
            else:
                avg_per_day = float(total_checks)
        else:
            avg_per_day = float(total_checks)

        # Last check details
        cur = conn.execute(
            """SELECT checked_at, anxiety_score FROM health_check_events
               WHERE stack_id = ? ORDER BY checked_at DESC LIMIT 1""",
            (stack_id,),
        )
        row = cur.fetchone()
        last_check_at = row[0] if row else None
        last_anxiety_score = row[1] if row else None

        # Checks by source
        cur = conn.execute(
            """SELECT source, COUNT(*) FROM health_check_events
               WHERE stack_id = ? GROUP BY source""",
            (stack_id,),
        )
        checks_by_source = {row[0]: row[1] for row in cur.fetchall()}

        # Checks by trigger
        cur = conn.execute(
            """SELECT triggered_by, COUNT(*) FROM health_check_events
               WHERE stack_id = ? GROUP BY triggered_by""",
            (stack_id,),
        )
        checks_by_trigger = {row[0]: row[1] for row in cur.fetchall()}

        return {
            "total_checks": total_checks,
            "avg_per_day": round(avg_per_day, 2),
            "last_check_at": last_check_at,
            "last_anxiety_score": last_anxiety_score,
            "checks_by_source": checks_by_source,
            "checks_by_trigger": checks_by_trigger,
        }


def check_relation_table_health(connect_fn: Callable, stack_id: str) -> Dict[str, Any]:
    """Check cross-table consistency for relationship data.

    Detects orphaned relationship_history entries whose relationship_id
    no longer exists in the relationships table.

    Args:
        connect_fn: Callable that returns a DB connection context manager.
        stack_id: The stack identifier.

    Returns:
        Dict with:
        - total_relationships: Count of relationships
        - total_history: Count of history entries
        - orphaned_history_count: History entries with no matching relationship
        - healthy: True if no orphans found
    """
    with connect_fn() as conn:
        # Check whether the required tables exist
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        has_relationships = "relationships" in tables
        has_history = "relationship_history" in tables

        if not has_relationships:
            return {
                "total_relationships": 0,
                "total_history": 0,
                "orphaned_history_count": 0,
                "healthy": True,
                "note": "relationships table does not exist",
            }

        total_rels = conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE stack_id = ?", (stack_id,)
        ).fetchone()[0]

        if not has_history:
            return {
                "total_relationships": total_rels,
                "total_history": 0,
                "orphaned_history_count": 0,
                "healthy": True,
                "note": "relationship_history table does not exist",
            }

        total_hist = conn.execute(
            "SELECT COUNT(*) FROM relationship_history WHERE stack_id = ?",
            (stack_id,),
        ).fetchone()[0]

        orphaned = conn.execute(
            """SELECT COUNT(*) FROM relationship_history h
               WHERE h.stack_id = ?
               AND NOT EXISTS (
                   SELECT 1 FROM relationships r
                   WHERE r.id = h.relationship_id AND r.stack_id = h.stack_id
               )""",
            (stack_id,),
        ).fetchone()[0]

        return {
            "total_relationships": total_rels,
            "total_history": total_hist,
            "orphaned_history_count": orphaned,
            "healthy": orphaned == 0,
        }
