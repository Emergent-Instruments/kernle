"""Playbook (procedural memory) mixin for Kernle.

Provides playbook creation, retrieval, search, and usage tracking.
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from kernle.core import Kernle


class PlaybookMixin:
    """Mixin providing playbook (procedural memory) capabilities."""

    MASTERY_LEVELS = ["novice", "competent", "proficient", "expert"]

    def playbook(
        self: "Kernle",
        name: str,
        description: str,
        steps: Union[List[Dict[str, Any]], List[str]],
        triggers: Optional[List[str]] = None,
        failure_modes: Optional[List[str]] = None,
        recovery_steps: Optional[List[str]] = None,
        source_episodes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        confidence: float = 0.8,
    ) -> str:
        """Create a new playbook (procedural memory).

        Args:
            name: Short name for the playbook (e.g., "Deploy to production")
            description: What this playbook does
            steps: List of steps - can be dicts with {action, details, adaptations}
                   or simple strings
            triggers: When to use this playbook (situation descriptions)
            failure_modes: What can go wrong
            recovery_steps: How to recover from failures
            source_episodes: Episode IDs this was learned from
            tags: Tags for categorization
            confidence: Initial confidence (0.0-1.0)

        Returns:
            Playbook ID
        """
        from kernle.storage import Playbook

        # Validate inputs
        name = self._validate_string_input(name, "name", 200)
        description = self._validate_string_input(description, "description", 2000)

        # Normalize steps to dict format
        normalized_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, str):
                normalized_steps.append(
                    {
                        "action": step,
                        "details": None,
                        "adaptations": None,
                    }
                )
            elif isinstance(step, dict):
                normalized_steps.append(
                    {
                        "action": step.get("action", f"Step {i + 1}"),
                        "details": step.get("details"),
                        "adaptations": step.get("adaptations"),
                    }
                )
            else:
                raise ValueError(f"Invalid step format at index {i}")

        # Validate optional lists
        if triggers:
            triggers = [self._validate_string_input(t, "trigger", 500) for t in triggers]
        if failure_modes:
            failure_modes = [
                self._validate_string_input(f, "failure_mode", 500) for f in failure_modes
            ]
        if recovery_steps:
            recovery_steps = [
                self._validate_string_input(r, "recovery_step", 500) for r in recovery_steps
            ]
        if tags:
            tags = [self._validate_string_input(t, "tag", 100) for t in tags]

        playbook_id = str(uuid.uuid4())

        playbook = Playbook(
            id=playbook_id,
            stack_id=self.stack_id,
            name=name,
            description=description,
            trigger_conditions=triggers or [],
            steps=normalized_steps,
            failure_modes=failure_modes or [],
            recovery_steps=recovery_steps,
            mastery_level="novice",
            times_used=0,
            success_rate=0.0,
            source_episodes=source_episodes,
            tags=tags,
            confidence=max(0.0, min(1.0, confidence)),
            last_used=None,
            created_at=datetime.now(timezone.utc),
        )

        self._storage.save_playbook(playbook)
        return playbook_id

    def load_playbooks(
        self: "Kernle", limit: int = 10, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load playbooks (procedural memories).

        Args:
            limit: Maximum number of playbooks to return
            tags: Filter by tags

        Returns:
            List of playbook dicts
        """
        playbooks = self._storage.list_playbooks(tags=tags, limit=limit)

        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "triggers": p.trigger_conditions,
                "steps": p.steps,
                "failure_modes": p.failure_modes,
                "recovery_steps": p.recovery_steps,
                "mastery_level": p.mastery_level,
                "times_used": p.times_used,
                "success_rate": p.success_rate,
                "confidence": p.confidence,
                "tags": p.tags,
                "last_used": p.last_used.isoformat() if p.last_used else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in playbooks
        ]

    def find_playbook(self: "Kernle", situation: str) -> Optional[Dict[str, Any]]:
        """Find the most relevant playbook for a given situation.

        Uses semantic search to match the situation against playbook
        triggers and descriptions.

        Args:
            situation: Description of the current situation/task

        Returns:
            Best matching playbook dict, or None if no good match
        """
        # Search for relevant playbooks
        playbooks = self._storage.search_playbooks(situation, limit=5)

        if not playbooks:
            return None

        # Return the best match (first result from search)
        p = playbooks[0]
        return {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "triggers": p.trigger_conditions,
            "steps": p.steps,
            "failure_modes": p.failure_modes,
            "recovery_steps": p.recovery_steps,
            "mastery_level": p.mastery_level,
            "times_used": p.times_used,
            "success_rate": p.success_rate,
            "confidence": p.confidence,
            "tags": p.tags,
        }

    def record_playbook_use(self: "Kernle", playbook_id: str, success: bool) -> bool:
        """Record a playbook usage and update statistics.

        Call this after executing a playbook to track its effectiveness.

        Args:
            playbook_id: ID of the playbook that was used
            success: Whether the execution was successful

        Returns:
            True if updated, False if playbook not found
        """
        return self._storage.update_playbook_usage(playbook_id, success)

    def get_playbook(self: "Kernle", playbook_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific playbook by ID.

        Args:
            playbook_id: ID of the playbook

        Returns:
            Playbook dict or None if not found
        """
        p = self._storage.get_playbook(playbook_id)
        if not p:
            return None

        return {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "triggers": p.trigger_conditions,
            "steps": p.steps,
            "failure_modes": p.failure_modes,
            "recovery_steps": p.recovery_steps,
            "mastery_level": p.mastery_level,
            "times_used": p.times_used,
            "success_rate": p.success_rate,
            "source_episodes": p.source_episodes,
            "confidence": p.confidence,
            "tags": p.tags,
            "last_used": p.last_used.isoformat() if p.last_used else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }

    def search_playbooks(self: "Kernle", query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search playbooks by query.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching playbook dicts
        """
        playbooks = self._storage.search_playbooks(query, limit=limit)

        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "triggers": p.trigger_conditions,
                "mastery_level": p.mastery_level,
                "times_used": p.times_used,
                "success_rate": p.success_rate,
                "tags": p.tags,
            }
            for p in playbooks
        ]
