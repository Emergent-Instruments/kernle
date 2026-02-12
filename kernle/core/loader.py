"""Memory loading operations for Kernle.

Delegates to stack.load() when available, falls back to individual
queries for non-SQLite backends.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from kernle.core.utils import (
    DEFAULT_MAX_ITEM_CHARS,
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    _build_memory_echoes,
    estimate_tokens,
)
from kernle.logging_config import log_load

logger = logging.getLogger(__name__)


class LoaderMixin:
    """Memory loading operations for Kernle."""

    def load(
        self,
        budget: int = DEFAULT_TOKEN_BUDGET,
        truncate: bool = True,
        max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
        sync: Optional[bool] = None,
        track_access: bool = True,
        epoch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load working memory context with budget-aware selection.

        Delegates core memory assembly to stack.load() which handles
        budget selection, strength filtering, and component hooks.
        Kernle adds checkpoint, boot_config, trust summary, and
        transforms the output shape.

        Args:
            budget: Token budget for memory (default: 8000, range: 100-50000)
            truncate: If True, truncate long items to fit more in budget
            max_item_chars: Max characters per item when truncating (default: 500)
            sync: Override auto_sync setting. If None, uses self.auto_sync.
            track_access: If True (default), record access for salience tracking.
            epoch_id: If set, filter candidates to this specific epoch.

        Returns:
            Dict containing all memory layers
        """
        # Validate budget parameter
        if not isinstance(budget, int) or budget < MIN_TOKEN_BUDGET:
            budget = MIN_TOKEN_BUDGET
        elif budget > MAX_TOKEN_BUDGET:
            budget = MAX_TOKEN_BUDGET

        # Validate max_item_chars parameter
        if not isinstance(max_item_chars, int) or max_item_chars < 10:
            max_item_chars = 10
        elif max_item_chars > 10000:
            max_item_chars = 10000

        # Sync before load if enabled
        should_sync = sync if sync is not None else self._auto_sync
        if should_sync:
            self._sync_before_load()

        # Load checkpoint first — Kernle-only concept
        checkpoint = self.load_checkpoint()

        # Reserve budget for checkpoint
        stack_budget = budget
        if checkpoint:
            checkpoint_text = json.dumps(checkpoint, default=str)
            stack_budget -= estimate_tokens(checkpoint_text)
            stack_budget = max(MIN_TOKEN_BUDGET, stack_budget)

        # Delegate to stack if available, otherwise fall back to individual queries
        if self.stack is not None:
            # Stack handles: budget selection, strength filtering, component on_load hooks,
            # access tracking, and memory echoes
            effective_max_chars = max_item_chars if truncate else 10000
            stack_result = self.stack.load(
                token_budget=stack_budget,
                epoch_id=epoch_id,
                max_item_chars=effective_max_chars,
                track_access=track_access,
            )

            # Transform stack output → Kernle output contract
            result = self._transform_stack_output(stack_result, truncate, max_item_chars)

            # Update meta with full budget info + memory echoes
            stack_meta = stack_result.get("_meta", {})
            cp_tokens = budget - stack_budget if checkpoint else 0
            excluded_candidates = stack_meta.get("_excluded_candidates", [])
            echoes_data = _build_memory_echoes(excluded_candidates)
            result["_meta"] = {
                "budget_used": stack_meta.get("budget_used", 0) + cp_tokens,
                "budget_total": budget,
                "excluded_count": stack_meta.get("excluded_count", 0),
                **echoes_data,
            }
        else:
            # Fallback for non-SQLite backends (no stack available)
            result = {
                "values": self.load_values(),
                "beliefs": self.load_beliefs(),
                "goals": self.load_goals(),
                "drives": self.load_drives(),
                "lessons": self.load_lessons(),
                "recent_work": self.load_recent_work(),
                "recent_notes": self.load_recent_notes(),
                "relationships": self.load_relationships(),
                "_meta": {
                    "budget_used": budget,
                    "budget_total": budget,
                    "excluded_count": 0,
                },
            }

        # Add checkpoint
        result["checkpoint"] = checkpoint

        # Kernle-specific augmentations
        boot = self.boot_list()
        if boot:
            result["boot_config"] = boot

        trust_summary = self._build_trust_summary()
        if trust_summary:
            result["trust"] = trust_summary

        # Log the load operation
        log_load(
            self.stack_id,
            values=len(result.get("values", [])),
            beliefs=len(result.get("beliefs", [])),
            episodes=len(result.get("recent_work", [])),
            checkpoint=checkpoint is not None,
        )

        return result

    def _transform_stack_output(
        self,
        stack_result: Dict[str, Any],
        truncate: bool = True,
        max_item_chars: int = DEFAULT_MAX_ITEM_CHARS,
    ) -> Dict[str, Any]:
        """Transform stack.load() output to Kernle.load() contract.

        Stack returns: values, beliefs, goals, drives, episodes, notes,
                       relationships, summaries, self_narratives
        Kernle returns: values (with value_type), beliefs (sorted by confidence),
                        goals, drives (with last_satisfied_at), lessons,
                        recent_work, recent_notes (with metadata), relationships
                        (with trust_level), summaries, self_narratives
        """
        result: Dict[str, Any] = {}

        # Values — add value_type for backward compat
        result["values"] = [
            {**v, "value_type": v.get("value_type", "core_value")}
            for v in stack_result.get("values", [])
        ]

        # Beliefs — sort by confidence descending
        beliefs = stack_result.get("beliefs", [])
        result["beliefs"] = sorted(beliefs, key=lambda b: b.get("confidence", 0), reverse=True)

        # Goals, drives — pass through
        result["goals"] = stack_result.get("goals", [])

        # Drives — add last_satisfied_at if missing
        result["drives"] = stack_result.get("drives", [])

        # Episodes → lessons + recent_work
        episodes = stack_result.get("episodes", [])
        lessons = []
        for ep in episodes:
            ep_lessons = ep.get("lessons") or []
            lessons.extend(ep_lessons[:2])
        result["lessons"] = lessons

        # recent_work — non-checkpoint episodes
        result["recent_work"] = [
            {
                "objective": ep.get("objective"),
                "outcome_type": ep.get("outcome_type"),
                "tags": ep.get("tags"),
                "created_at": ep.get("created_at"),
            }
            for ep in episodes
            if not ep.get("tags") or "checkpoint" not in (ep.get("tags") or [])
        ][:5]

        # Notes → recent_notes (add metadata wrapper)
        notes = stack_result.get("notes", [])
        result["recent_notes"] = [
            {
                "content": n.get("content"),
                "metadata": {
                    "note_type": n.get("note_type"),
                    "tags": n.get("tags"),
                    "speaker": n.get("speaker"),
                    "reason": n.get("reason"),
                },
                "created_at": n.get("created_at"),
            }
            for n in notes
        ]

        # Relationships — add trust_level, interaction_count, other_stack_id
        rels = stack_result.get("relationships", [])
        result["relationships"] = [
            {
                "other_stack_id": r.get("entity_name"),
                "entity_name": r.get("entity_name"),
                "trust_level": (r.get("sentiment", 0) + 1) / 2,
                "sentiment": r.get("sentiment"),
                "interaction_count": r.get("interaction_count", 0),
                "last_interaction": r.get("last_interaction"),
                "notes": r.get("notes"),
                "entity_type": r.get("entity_type"),
            }
            for r in rels
        ]

        # Summaries + self_narratives — pass through
        if stack_result.get("summaries"):
            result["summaries"] = stack_result["summaries"]
        if stack_result.get("self_narratives"):
            result["self_narratives"] = stack_result["self_narratives"]

        # Pass through component contributions (e.g., anxiety)
        for key in stack_result:
            if key not in (
                "values",
                "beliefs",
                "goals",
                "drives",
                "episodes",
                "notes",
                "relationships",
                "summaries",
                "self_narratives",
                "_meta",
            ):
                result[key] = stack_result[key]

        return result

    def load_values(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Load normative values (highest authority)."""
        values = self._storage.get_values(limit=limit)
        return [
            {
                "id": v.id,
                "name": v.name,
                "statement": v.statement,
                "priority": v.priority,
                "value_type": "core_value",  # Default for backwards compatibility
            }
            for v in values
        ]

    def load_beliefs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Load semantic beliefs."""
        beliefs = self._storage.get_beliefs(limit=limit)
        # Sort by confidence descending
        beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)
        return [
            {
                "id": b.id,
                "statement": b.statement,
                "belief_type": b.belief_type,
                "confidence": b.confidence,
            }
            for b in beliefs[:limit]
        ]

    def load_goals(self, limit: int = 10, status: str = "active") -> List[Dict[str, Any]]:
        """Load goals filtered by status.

        Args:
            limit: Maximum number of goals to return
            status: Filter by status - "active", "completed", "paused", or "all"
        """
        goals = self._storage.get_goals(status=None if status == "all" else status, limit=limit)
        return [
            {
                "id": g.id,
                "title": g.title,
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
            }
            for g in goals
        ]

    def load_lessons(self, limit: int = 20) -> List[str]:
        """Load lessons from reflected episodes."""
        episodes = self._storage.get_episodes(limit=limit)

        lessons = []
        for ep in episodes:
            if ep.lessons:
                lessons.extend(ep.lessons[:2])
        return lessons

    def load_recent_work(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Load recent episodes."""
        episodes = self._storage.get_episodes(limit=limit * 2)

        # Filter out checkpoints
        non_checkpoint = [e for e in episodes if not e.tags or "checkpoint" not in e.tags]

        return [
            {
                "objective": e.objective,
                "outcome_type": e.outcome_type,
                "tags": e.tags,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in non_checkpoint[:limit]
        ]

    def load_recent_notes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Load recent curated notes."""
        notes = self._storage.get_notes(limit=limit)
        return [
            {
                "content": n.content,
                "metadata": {
                    "note_type": n.note_type,
                    "tags": n.tags,
                    "speaker": n.speaker,
                    "reason": n.reason,
                },
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in notes
        ]
