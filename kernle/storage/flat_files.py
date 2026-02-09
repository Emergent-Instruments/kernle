"""Flat file sync for Kernle storage.

Free functions that write memory data to human-readable markdown files.
These are called by SQLiteStorage after save operations to keep flat
files in sync with the database.
"""

import logging
from pathlib import Path
from typing import List

from .base import Belief, Goal, Relationship, Value

logger = logging.getLogger(__name__)


def init_flat_files(
    beliefs_file: Path,
    values_file: Path,
    relationships_file: Path,
    goals_file: Path,
    sync_beliefs_fn,
    sync_values_fn,
    sync_goals_fn,
    sync_relationships_fn,
) -> None:
    """Initialize flat files from existing database data.

    Called on startup to ensure flat files exist and are in sync.
    """
    try:
        if not beliefs_file.exists() or beliefs_file.stat().st_size == 0:
            sync_beliefs_fn()
        if not values_file.exists() or values_file.stat().st_size == 0:
            sync_values_fn()
        if not relationships_file.exists() or relationships_file.stat().st_size == 0:
            sync_relationships_fn()
        if not goals_file.exists() or goals_file.stat().st_size == 0:
            sync_goals_fn()
    except Exception as e:
        logger.warning(f"Failed to initialize flat files: {e}")


def sync_beliefs_to_file(beliefs_file: Path, beliefs: List[Belief], now: str) -> None:
    """Write all active beliefs to flat file."""
    try:
        lines = ["# Beliefs", f"_Last updated: {now}_", ""]

        for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
            conf_bar = "█" * int(b.confidence * 5) + "░" * (5 - int(b.confidence * 5))
            lines.append(f"## [{conf_bar}] {int(b.confidence * 100)}% - {b.id[:8]}")
            lines.append(b.statement)
            if b.source_episodes:
                lines.append(f"Sources: {', '.join(b.source_episodes[:3])}")
            lines.append("")

        with open(beliefs_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        # Secure permissions
        import os

        os.chmod(beliefs_file, 0o600)
    except Exception as e:
        logger.warning(f"Failed to sync beliefs to file: {e}")


def sync_values_to_file(values_file: Path, values: List[Value], now: str) -> None:
    """Write all values to flat file."""
    try:
        lines = ["# Values", f"_Last updated: {now}_", ""]

        for v in sorted(values, key=lambda x: x.priority, reverse=True):
            lines.append(f"## {v.name} (priority: {v.priority}) - {v.id[:8]}")
            lines.append(v.statement)
            lines.append("")

        with open(values_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to sync values to file: {e}")


def sync_goals_to_file(goals_file: Path, goals: List[Goal], now: str) -> None:
    """Write all active goals to flat file."""
    try:
        lines = ["# Goals", f"_Last updated: {now}_", ""]

        # Group by status
        for status in ["active", "completed", "paused"]:
            status_goals = [g for g in goals if g.status == status]
            if status_goals:
                lines.append(f"## {status.title()}")
                for g in sorted(status_goals, key=lambda x: x.priority or "", reverse=True):
                    priority = f" [{g.priority}]" if g.priority else ""
                    goal_type_label = (
                        f" ({g.goal_type})" if g.goal_type and g.goal_type != "task" else ""
                    )
                    status_icon = (
                        "○" if status == "active" else "✓" if status == "completed" else "⏸"
                    )
                    lines.append(
                        f"- {status_icon} {g.title}{priority}{goal_type_label} ({g.id[:8]})"
                    )
                    if g.description:
                        lines.append(f"  {g.description[:100]}")
                lines.append("")

        with open(goals_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to sync goals to file: {e}")


def sync_relationships_to_file(
    relationships_file: Path, relationships: List[Relationship], now: str
) -> None:
    """Write all relationships to flat file."""
    try:
        lines = ["# Relationships", f"_Last updated: {now}_", ""]

        for r in sorted(relationships, key=lambda x: x.sentiment, reverse=True):
            trust_pct = int(((r.sentiment + 1) / 2) * 100)
            trust_bar = "█" * (trust_pct // 10) + "░" * (10 - trust_pct // 10)
            lines.append(f"## {r.entity_name} ({r.entity_type}) - {r.id[:8]}")
            lines.append(f"Trust: [{trust_bar}] {trust_pct}%")
            lines.append(f"Interactions: {r.interaction_count}")
            if r.last_interaction:
                lines.append(f"Last: {r.last_interaction.strftime('%Y-%m-%d')}")
            if r.notes:
                lines.append(f"Notes: {r.notes}")
            lines.append("")

        with open(relationships_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        logger.warning(f"Failed to sync relationships to file: {e}")
