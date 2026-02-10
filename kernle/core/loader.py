"""Memory loading operations for Kernle."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.core.utils import (
    DEFAULT_MAX_ITEM_CHARS,
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKEN_BUDGET,
    MIN_TOKEN_BUDGET,
    _build_memory_echoes,
    compute_priority_score,
    estimate_tokens,
    truncate_at_word_boundary,
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

        Memories are loaded by priority across all types until the budget
        is exhausted, preventing context overflow. Higher priority items
        are loaded first.

        Priority order (highest first):
        - Checkpoint: Always loaded (task continuity)
        - Values: 0.90 base, sorted by priority DESC
        - Beliefs: 0.70 base, sorted by confidence DESC
        - Goals: 0.65 base, sorted by recency
        - Drives: 0.60 base, sorted by intensity DESC
        - Episodes: 0.40 base, sorted by recency
        - Notes: 0.35 base, sorted by recency
        - Relationships: 0.30 base, sorted by last_interaction

        Args:
            budget: Token budget for memory (default: 8000, range: 100-50000)
            truncate: If True, truncate long items to fit more in budget
            max_item_chars: Max characters per item when truncating (default: 500)
            sync: Override auto_sync setting. If None, uses self.auto_sync.
            track_access: If True (default), record access for salience tracking.
                Set to False for internal operations (like sync) that should not
                affect salience decay.
            epoch_id: If set, filter candidates to this specific epoch before
                budget selection. NULL epoch_id memories are excluded.

        Returns:
            Dict containing all memory layers
        """
        # Validate budget parameter (defense in depth - also validated at MCP layer)
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

        # Load checkpoint first - always included
        checkpoint = self.load_checkpoint()
        remaining_budget = budget

        # Estimate checkpoint tokens
        if checkpoint:
            checkpoint_text = json.dumps(checkpoint, default=str)
            remaining_budget -= estimate_tokens(checkpoint_text)

        # Fetch candidates from all types with high limits for budget selection
        batched = self._storage.load_all(
            values_limit=None,  # Use high limit (1000)
            beliefs_limit=None,
            goals_limit=None,
            goals_status="active",
            episodes_limit=None,
            notes_limit=None,
            drives_limit=None,
            relationships_limit=None,
            epoch_id=epoch_id,
        )

        if batched is not None:
            # Build candidate list with priority scores
            candidates = []

            # Values - sorted by priority DESC
            for v in batched.get("values", []):
                candidates.append((compute_priority_score("value", v), "value", v))

            # Beliefs - sorted by confidence DESC
            for b in batched.get("beliefs", []):
                candidates.append((compute_priority_score("belief", b), "belief", b))

            # Goals - recency already handled by storage
            for g in batched.get("goals", []):
                candidates.append((compute_priority_score("goal", g), "goal", g))

            # Drives - sorted by intensity DESC
            for d in batched.get("drives", []):
                candidates.append((compute_priority_score("drive", d), "drive", d))

            # Episodes - recency already handled by storage
            for e in batched.get("episodes", []):
                candidates.append((compute_priority_score("episode", e), "episode", e))

            # Notes - recency already handled by storage
            for n in batched.get("notes", []):
                candidates.append((compute_priority_score("note", n), "note", n))

            # Relationships - sorted by last_interaction
            for r in batched.get("relationships", []):
                candidates.append((compute_priority_score("relationship", r), "relationship", r))

            # Summaries - with supersession logic
            all_summaries = self._storage.list_summaries(self.stack_id)
            # Collect IDs superseded by higher-scope summaries
            superseded_ids = set()
            for s in all_summaries:
                if s.supersedes:
                    superseded_ids.update(s.supersedes)
            # Only include non-superseded summaries
            for s in all_summaries:
                if s.id not in superseded_ids:
                    scope_key = f"summary_{s.scope}"
                    candidates.append((compute_priority_score(scope_key, s), "summary", s))

            # Self-narratives - only active ones
            active_narratives = self._storage.list_self_narratives(self.stack_id, active_only=True)
            for n in active_narratives:
                candidates.append(
                    (compute_priority_score("self_narrative", n), "self_narrative", n)
                )

            # Sort by priority descending
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Track total candidates for metadata
            total_candidates = len(candidates)
            selected_count = 0

            # Fill budget with highest priority items
            selected = {
                "values": [],
                "beliefs": [],
                "goals": [],
                "drives": [],
                "episodes": [],
                "notes": [],
                "relationships": [],
                "summaries": [],
                "self_narratives": [],
            }

            selected_indices = set()
            for idx, (priority, memory_type, record) in enumerate(candidates):
                # Format the record for token estimation
                if memory_type == "value":
                    text = f"{record.name}: {record.statement}"
                elif memory_type == "belief":
                    text = record.statement
                elif memory_type == "goal":
                    text = f"{record.title} {record.description or ''}"
                elif memory_type == "drive":
                    text = f"{record.drive_type}: {record.focus_areas or ''}"
                elif memory_type == "episode":
                    text = f"{record.objective} {record.outcome}"
                elif memory_type == "note":
                    text = record.content
                elif memory_type == "relationship":
                    text = f"{record.entity_name}: {record.notes or ''}"
                elif memory_type == "summary":
                    text = f"[{record.scope}] {record.content}"
                elif memory_type == "self_narrative":
                    text = f"[{record.narrative_type}] {record.content}"
                else:
                    text = str(record)

                # Truncate if enabled and text exceeds limit
                if truncate and len(text) > max_item_chars:
                    text = truncate_at_word_boundary(text, max_item_chars)

                # Estimate tokens for this item
                tokens = estimate_tokens(text)

                # Check if it fits in budget
                if tokens <= remaining_budget:
                    if memory_type == "value":
                        selected["values"].append(record)
                    elif memory_type == "belief":
                        selected["beliefs"].append(record)
                    elif memory_type == "goal":
                        selected["goals"].append(record)
                    elif memory_type == "drive":
                        selected["drives"].append(record)
                    elif memory_type == "episode":
                        selected["episodes"].append(record)
                    elif memory_type == "note":
                        selected["notes"].append(record)
                    elif memory_type == "relationship":
                        selected["relationships"].append(record)
                    elif memory_type == "summary":
                        selected["summaries"].append(record)
                    elif memory_type == "self_narrative":
                        selected["self_narratives"].append(record)

                    remaining_budget -= tokens
                    selected_count += 1
                    selected_indices.add(idx)

                # Stop if budget exhausted
                if remaining_budget <= 0:
                    break

            # Build excluded list (preserves priority order) for memory echoes
            excluded_candidates = [c for i, c in enumerate(candidates) if i not in selected_indices]

            # Extract lessons from selected episodes
            lessons = []
            for ep in selected["episodes"]:
                if ep.lessons:
                    lessons.extend(ep.lessons[:2])

            # Filter recent work (non-checkpoint episodes)
            recent_work = [
                {
                    "objective": e.objective,
                    "outcome_type": e.outcome_type,
                    "tags": e.tags,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in selected["episodes"]
                if not e.tags or "checkpoint" not in e.tags
            ][:5]

            # Format selected items for API compatibility
            batched_result = {
                "checkpoint": checkpoint,
                "values": [
                    {
                        "id": v.id,
                        "name": v.name,
                        "statement": (
                            truncate_at_word_boundary(v.statement, max_item_chars)
                            if truncate
                            else v.statement
                        ),
                        "priority": v.priority,
                        "value_type": "core_value",
                    }
                    for v in selected["values"]
                ],
                "beliefs": [
                    {
                        "id": b.id,
                        "statement": (
                            truncate_at_word_boundary(b.statement, max_item_chars)
                            if truncate
                            else b.statement
                        ),
                        "belief_type": b.belief_type,
                        "confidence": b.confidence,
                    }
                    for b in sorted(selected["beliefs"], key=lambda x: x.confidence, reverse=True)
                ],
                "goals": [
                    {
                        "id": g.id,
                        "title": g.title,
                        "description": (
                            truncate_at_word_boundary(g.description, max_item_chars)
                            if truncate and g.description
                            else g.description
                        ),
                        "priority": g.priority,
                        "status": g.status,
                    }
                    for g in selected["goals"]
                ],
                "drives": [
                    {
                        "id": d.id,
                        "drive_type": d.drive_type,
                        "intensity": d.intensity,
                        "last_satisfied_at": d.updated_at.isoformat() if d.updated_at else None,
                        "focus_areas": d.focus_areas,
                    }
                    for d in selected["drives"]
                ],
                "lessons": lessons,
                "recent_work": recent_work,
                "recent_notes": [
                    {
                        "content": (
                            truncate_at_word_boundary(n.content, max_item_chars)
                            if truncate
                            else n.content
                        ),
                        "metadata": {
                            "note_type": n.note_type,
                            "tags": n.tags,
                            "speaker": n.speaker,
                            "reason": n.reason,
                        },
                        "created_at": n.created_at.isoformat() if n.created_at else None,
                    }
                    for n in selected["notes"]
                ],
                "relationships": [
                    {
                        "other_stack_id": r.entity_name,
                        "entity_name": r.entity_name,
                        "trust_level": (r.sentiment + 1) / 2,
                        "sentiment": r.sentiment,
                        "interaction_count": r.interaction_count,
                        "last_interaction": (
                            r.last_interaction.isoformat() if r.last_interaction else None
                        ),
                        "notes": (
                            truncate_at_word_boundary(r.notes, max_item_chars)
                            if truncate and r.notes
                            else r.notes
                        ),
                    }
                    for r in sorted(
                        selected["relationships"],
                        key=lambda x: x.last_interaction
                        or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True,
                    )
                ],
                "summaries": [
                    {
                        "id": s.id,
                        "scope": s.scope,
                        "period_start": s.period_start,
                        "period_end": s.period_end,
                        "content": (
                            truncate_at_word_boundary(s.content, max_item_chars)
                            if truncate
                            else s.content
                        ),
                        "key_themes": s.key_themes,
                    }
                    for s in selected["summaries"]
                ],
                "self_narratives": [
                    {
                        "id": sn.id,
                        "narrative_type": sn.narrative_type,
                        "content": (
                            truncate_at_word_boundary(sn.content, max_item_chars)
                            if truncate
                            else sn.content
                        ),
                        "key_themes": sn.key_themes,
                        "unresolved_tensions": sn.unresolved_tensions,
                    }
                    for sn in selected["self_narratives"]
                ],
                "_meta": {
                    "budget_used": budget - remaining_budget,
                    "budget_total": budget,
                    "excluded_count": total_candidates - selected_count,
                    **_build_memory_echoes(excluded_candidates),
                },
            }

            # Track access for all loaded memories (for salience-based forgetting)
            if track_access:
                accesses = []
                for v in selected["values"]:
                    accesses.append(("value", v.id))
                for b in selected["beliefs"]:
                    accesses.append(("belief", b.id))
                for g in selected["goals"]:
                    accesses.append(("goal", g.id))
                for d in selected["drives"]:
                    accesses.append(("drive", d.id))
                for e in selected["episodes"]:
                    accesses.append(("episode", e.id))
                for n in selected["notes"]:
                    accesses.append(("note", n.id))
                for r in selected["relationships"]:
                    accesses.append(("relationship", r.id))

                if accesses:
                    self._storage.record_access_batch(accesses)

            # Log the load operation (batched path)
            log_load(
                self.stack_id,
                values=len(selected["values"]),
                beliefs=len(selected["beliefs"]),
                episodes=len(selected["episodes"]),
                checkpoint=checkpoint is not None,
            )

            # Include boot config (zero-cost, always available)
            boot = self.boot_list()
            if boot:
                batched_result["boot_config"] = boot

            # Include trust summary
            trust_summary = self._build_trust_summary()
            if trust_summary:
                batched_result["trust"] = trust_summary

            return batched_result

        # Fallback to individual queries (for backends without load_all)
        # Note: This path doesn't do budget-aware selection, so we report
        # the budget as fully used and no exclusions (legacy behavior)
        result = {
            "checkpoint": self.load_checkpoint(),
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

        # Include boot config
        boot = self.boot_list()
        if boot:
            result["boot_config"] = boot

        # Log the load operation
        log_load(
            self.stack_id,
            values=len(result.get("values", [])),
            beliefs=len(result.get("beliefs", [])),
            episodes=len(result.get("recent_work", [])),
            checkpoint=result.get("checkpoint") is not None,
        )

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
