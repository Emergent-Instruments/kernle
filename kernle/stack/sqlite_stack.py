"""SQLiteStack - StackProtocol implementation wrapping SQLiteStorage.

This is the default memory stack implementation. It wraps the existing
SQLiteStorage backend and adds:
- StackProtocol interface conformance
- Feature mixins (anxiety, consolidation, emotions, forgetting, knowledge,
  metamemory, suggestions) applied the same way as on Kernle
- Composition hooks (on_attach, on_detach, on_model_changed)
- Component registry infrastructure (for v0.5.0 stack components)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kernle.features import (
    AnxietyMixin,
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
)
from kernle.protocols import (
    InferenceService,
    StackComponentProtocol,
)
from kernle.protocols import (
    SearchResult as ProtocolSearchResult,
)
from kernle.protocols import (
    SyncResult as ProtocolSyncResult,
)
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
)

logger = logging.getLogger(__name__)

# Default token budget for memory loading
DEFAULT_TOKEN_BUDGET = 8000
MAX_TOKEN_BUDGET = 50000
MIN_TOKEN_BUDGET = 100
DEFAULT_MAX_ITEM_CHARS = 500
TOKEN_ESTIMATION_SAFETY_MARGIN = 1.3

# Priority scores for each memory type (higher = more important)
MEMORY_TYPE_PRIORITIES = {
    "value": 0.90,
    "self_narrative": 0.90,
    "summary_decade": 0.95,
    "summary_epoch": 0.85,
    "summary_year": 0.80,
    "belief": 0.70,
    "goal": 0.65,
    "drive": 0.60,
    "summary_quarter": 0.50,
    "episode": 0.40,
    "summary_month": 0.35,
    "note": 0.35,
    "relationship": 0.30,
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars/token with safety margin)."""
    if not text:
        return 0
    return int(len(text) // 4 * TOKEN_ESTIMATION_SAFETY_MARGIN)


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary with ellipsis."""
    if not text or len(text) <= max_chars:
        return text
    target = max_chars - 3
    if target <= 0:
        return "..."
    truncated = text[:target]
    last_space = truncated.rfind(" ")
    if last_space > target // 2:
        truncated = truncated[:last_space]
    return truncated + "..."


def _compute_priority_score(memory_type: str, record: Any) -> float:
    """Compute priority score for budget-based selection."""
    base = MEMORY_TYPE_PRIORITIES.get(memory_type, 0.20)
    bonus = 0.0
    if memory_type == "value":
        bonus = getattr(record, "priority", 50) / 1000.0
    elif memory_type == "belief":
        bonus = getattr(record, "confidence", 0.8) / 10.0
    elif memory_type == "drive":
        bonus = getattr(record, "intensity", 0.5) / 10.0
    return base + bonus


class SQLiteStack(
    AnxietyMixin,
    ConsolidationMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,
):
    """SQLite-backed memory stack conforming to StackProtocol.

    Wraps SQLiteStorage and exposes the full StackProtocol interface.
    Works in detached mode (no core attached) for read/write/search.
    Feature mixins are applied the same way as on Kernle.
    """

    def __init__(
        self,
        stack_id: str,
        db_path: Optional[Path] = None,
        cloud_storage: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ):
        self._backend = SQLiteStorage(
            stack_id=stack_id,
            db_path=db_path,
            cloud_storage=cloud_storage,
            embedder=embedder,
        )
        # Alias for mixin compatibility (mixins access self._storage)
        self._storage = self._backend

        # Component registry (v0.5.0 infrastructure)
        self._components: Dict[str, StackComponentProtocol] = {}

        # Composition state
        self._attached_core_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None

    # ---- Properties ----

    @property
    def stack_id(self) -> str:
        return self._backend.stack_id

    @property
    def schema_version(self) -> int:
        from kernle.storage.sqlite import SCHEMA_VERSION

        return SCHEMA_VERSION

    # ---- Component Management ----

    @property
    def components(self) -> Dict[str, StackComponentProtocol]:
        return dict(self._components)

    def add_component(self, component: StackComponentProtocol) -> None:
        """Add a component to this stack."""
        name = component.name
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered")
        component.attach(self.stack_id, self._inference)
        self._components[name] = component

    def remove_component(self, name: str) -> None:
        """Remove a component by name."""
        if name not in self._components:
            raise ValueError(f"Component '{name}' not found")
        component = self._components[name]
        if component.required:
            raise ValueError(f"Cannot remove required component '{name}'")
        component.detach()
        del self._components[name]

    def get_component(self, name: str) -> Optional[StackComponentProtocol]:
        return self._components.get(name)

    def maintenance(self) -> Dict[str, Any]:
        """Run maintenance on all components."""
        results: Dict[str, Any] = {}
        for name, component in self._components.items():
            try:
                stats = component.on_maintenance()
                if stats:
                    results[name] = stats
            except Exception as e:
                logger.warning(f"Component '{name}' maintenance failed: {e}")
                results[name] = {"error": str(e)}
        return results

    # ---- Write Operations ----

    def save_episode(self, episode: Episode) -> str:
        return self._backend.save_episode(episode)

    def save_belief(self, belief: Belief) -> str:
        return self._backend.save_belief(belief)

    def save_value(self, value: Value) -> str:
        return self._backend.save_value(value)

    def save_goal(self, goal: Goal) -> str:
        return self._backend.save_goal(goal)

    def save_note(self, note: Note) -> str:
        return self._backend.save_note(note)

    def save_drive(self, drive: Drive) -> str:
        return self._backend.save_drive(drive)

    def save_relationship(self, relationship: Relationship) -> str:
        return self._backend.save_relationship(relationship)

    def save_raw(self, raw: RawEntry) -> str:
        return self._backend.save_raw(
            blob=raw.blob or raw.content or "",
            source=raw.source,
        )

    def save_playbook(self, playbook: Playbook) -> str:
        return self._backend.save_playbook(playbook)

    def save_epoch(self, epoch: Epoch) -> str:
        return self._backend.save_epoch(epoch)

    def save_summary(self, summary: Summary) -> str:
        return self._backend.save_summary(summary)

    def save_self_narrative(self, narrative: SelfNarrative) -> str:
        return self._backend.save_self_narrative(narrative)

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        return self._backend.save_suggestion(suggestion)

    # ---- Batch Write ----

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        return self._backend.save_episodes_batch(episodes)

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        return self._backend.save_beliefs_batch(beliefs)

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        return self._backend.save_notes_batch(notes)

    # ---- Read Operations ----

    def get_episodes(
        self,
        *,
        limit: int = 50,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> List[Episode]:
        episodes = self._backend.get_episodes(limit=limit, tags=tags)
        if not include_forgotten:
            episodes = [e for e in episodes if not e.is_forgotten]
        return episodes

    def get_beliefs(
        self,
        *,
        limit: int = 50,
        belief_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> List[Belief]:
        beliefs = self._backend.get_beliefs(limit=limit)
        if belief_type:
            beliefs = [b for b in beliefs if b.belief_type == belief_type]
        if min_confidence is not None:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        if not include_forgotten:
            beliefs = [b for b in beliefs if not b.is_forgotten]
        return beliefs

    def get_values(
        self,
        *,
        limit: int = 50,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> List[Value]:
        values = self._backend.get_values(limit=limit)
        if not include_forgotten:
            values = [v for v in values if not v.is_forgotten]
        return values

    def get_goals(
        self,
        *,
        limit: int = 50,
        status: Optional[str] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> List[Goal]:
        goals = self._backend.get_goals(status=status, limit=limit)
        if not include_forgotten:
            goals = [g for g in goals if not g.is_forgotten]
        return goals

    def get_notes(
        self,
        *,
        limit: int = 50,
        note_type: Optional[str] = None,
        context: Optional[str] = None,
        include_forgotten: bool = False,
    ) -> List[Note]:
        notes = self._backend.get_notes(limit=limit, note_type=note_type)
        if not include_forgotten:
            notes = [n for n in notes if not n.is_forgotten]
        return notes

    def get_drives(self, *, include_expired: bool = False) -> List[Drive]:
        return self._backend.get_drives()

    def get_relationships(
        self,
        *,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_trust: Optional[float] = None,
    ) -> List[Relationship]:
        rels = self._backend.get_relationships(entity_type=entity_type)
        if entity_id:
            rels = [r for r in rels if r.entity_name == entity_id]
        if min_trust is not None:
            rels = [r for r in rels if (r.sentiment + 1) / 2 >= min_trust]
        return rels

    def get_raw(
        self,
        *,
        limit: int = 50,
        tags: Optional[List[str]] = None,
    ) -> List[RawEntry]:
        entries = self._backend.list_raw(limit=limit)
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if e.tags and tag_set.intersection(e.tags)]
        return entries

    def get_memory(self, memory_type: str, memory_id: str) -> Optional[Any]:
        return self._backend.get_memory(memory_type, memory_id)

    # ---- Search ----

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        record_types: Optional[List[str]] = None,
        context: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ProtocolSearchResult]:
        storage_results = self._backend.search(
            query=query,
            limit=limit,
            record_types=record_types,
        )
        results = []
        for sr in storage_results:
            record = sr.record
            content = ""
            if sr.record_type == "episode":
                content = f"{record.objective}: {record.outcome}"
            elif sr.record_type == "belief":
                content = record.statement
            elif sr.record_type == "value":
                content = f"{record.name}: {record.statement}"
            elif sr.record_type == "goal":
                content = f"{record.title}: {record.description or ''}"
            elif sr.record_type == "note":
                content = record.content
            elif sr.record_type == "relationship":
                content = f"{record.entity_name}: {record.notes or ''}"
            else:
                content = str(record)[:200]

            if min_confidence is not None:
                record_conf = getattr(record, "confidence", 1.0)
                if record_conf < min_confidence:
                    continue

            results.append(
                ProtocolSearchResult(
                    memory_type=sr.record_type,
                    memory_id=record.id,
                    content=content,
                    score=sr.score,
                    metadata={
                        "confidence": getattr(record, "confidence", None),
                    },
                )
            )
        return results

    # ---- Working Memory ----

    def load(
        self,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble working memory within a token budget."""
        budget = max(MIN_TOKEN_BUDGET, min(MAX_TOKEN_BUDGET, token_budget))
        remaining = budget

        # Fetch candidates from all types
        batched = self._backend.load_all(
            values_limit=None,
            beliefs_limit=None,
            goals_limit=None,
            goals_status="active",
            episodes_limit=None,
            notes_limit=None,
            drives_limit=None,
            relationships_limit=None,
        )

        if batched is None:
            # Fallback to individual queries
            return {
                "values": [
                    {"id": v.id, "name": v.name, "statement": v.statement, "priority": v.priority}
                    for v in self._backend.get_values(limit=50)
                ],
                "beliefs": [
                    {"id": b.id, "statement": b.statement, "confidence": b.confidence}
                    for b in self._backend.get_beliefs(limit=50)
                ],
                "goals": [
                    {"id": g.id, "title": g.title, "status": g.status}
                    for g in self._backend.get_goals(limit=50)
                ],
                "episodes": [
                    {"id": e.id, "objective": e.objective, "outcome": e.outcome}
                    for e in self._backend.get_episodes(limit=20)
                ],
                "_meta": {"budget_used": budget, "budget_total": budget},
            }

        # Build candidate list with priorities
        candidates = []
        for v in batched.get("values", []):
            candidates.append((_compute_priority_score("value", v), "value", v))
        for b in batched.get("beliefs", []):
            candidates.append((_compute_priority_score("belief", b), "belief", b))
        for g in batched.get("goals", []):
            candidates.append((_compute_priority_score("goal", g), "goal", g))
        for d in batched.get("drives", []):
            candidates.append((_compute_priority_score("drive", d), "drive", d))
        for e in batched.get("episodes", []):
            candidates.append((_compute_priority_score("episode", e), "episode", e))
        for n in batched.get("notes", []):
            candidates.append((_compute_priority_score("note", n), "note", n))
        for r in batched.get("relationships", []):
            candidates.append((_compute_priority_score("relationship", r), "relationship", r))

        # Summaries
        all_summaries = self._backend.list_summaries(self.stack_id)
        superseded_ids = set()
        for s in all_summaries:
            if s.supersedes:
                superseded_ids.update(s.supersedes)
        for s in all_summaries:
            if s.id not in superseded_ids:
                scope_key = f"summary_{s.scope}"
                candidates.append((_compute_priority_score(scope_key, s), "summary", s))

        # Self-narratives
        active_narratives = self._backend.list_self_narratives(self.stack_id, active_only=True)
        for n in active_narratives:
            candidates.append((_compute_priority_score("self_narrative", n), "self_narrative", n))

        candidates.sort(key=lambda x: x[0], reverse=True)

        selected: Dict[str, list] = {
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

        for priority, memory_type, record in candidates:
            text = self._record_to_text(memory_type, record)
            text = _truncate_at_word_boundary(text, DEFAULT_MAX_ITEM_CHARS)
            tokens = _estimate_tokens(text)
            if tokens <= remaining:
                key = (
                    memory_type + "s"
                    if memory_type not in ("summary", "self_narrative")
                    else ("summaries" if memory_type == "summary" else "self_narratives")
                )
                if key in selected:
                    selected[key].append(record)
                remaining -= tokens
            if remaining <= 0:
                break

        # Format output
        result: Dict[str, Any] = {
            "values": [
                {"id": v.id, "name": v.name, "statement": v.statement, "priority": v.priority}
                for v in selected["values"]
            ],
            "beliefs": [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "belief_type": b.belief_type,
                    "confidence": b.confidence,
                }
                for b in selected["beliefs"]
            ],
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "description": g.description,
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
                    "focus_areas": d.focus_areas,
                }
                for d in selected["drives"]
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "tags": e.tags,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in selected["episodes"]
            ],
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "note_type": n.note_type,
                    "tags": n.tags,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in selected["notes"]
            ],
            "relationships": [
                {
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "sentiment": r.sentiment,
                    "notes": r.notes,
                }
                for r in selected["relationships"]
            ],
            "_meta": {
                "budget_used": budget - remaining,
                "budget_total": budget,
            },
        }

        if selected["summaries"]:
            result["summaries"] = [
                {
                    "id": s.id,
                    "scope": s.scope,
                    "content": _truncate_at_word_boundary(s.content, DEFAULT_MAX_ITEM_CHARS),
                }
                for s in selected["summaries"]
            ]

        if selected["self_narratives"]:
            result["self_narratives"] = [
                {
                    "id": sn.id,
                    "narrative_type": sn.narrative_type,
                    "content": _truncate_at_word_boundary(sn.content, DEFAULT_MAX_ITEM_CHARS),
                }
                for sn in selected["self_narratives"]
            ]

        # Track access for salience
        accesses = []
        for key in ("values", "beliefs", "goals", "drives", "episodes", "notes", "relationships"):
            for rec in selected[key]:
                type_name = key.rstrip("s") if key != "values" else "value"
                accesses.append((type_name, rec.id))
        if accesses:
            self._backend.record_access_batch(accesses)

        return result

    # ---- Meta-Memory ----

    def record_access(self, memory_type: str, memory_id: str) -> bool:
        return self._backend.record_access(memory_type, memory_id)

    def update_memory_meta(
        self,
        memory_type: str,
        memory_id: str,
        *,
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        return self._backend.update_memory_meta(memory_type, memory_id, confidence=confidence)

    def forget_memory(
        self,
        memory_type: str,
        memory_id: str,
        reason: str,
    ) -> bool:
        return self._backend.forget_memory(memory_type, memory_id, reason)

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        return self._backend.recover_memory(memory_type, memory_id)

    def protect_memory(
        self,
        memory_type: str,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        return self._backend.protect_memory(memory_type, memory_id, protected)

    # ---- Trust Layer ----

    def save_trust_assessment(self, assessment: TrustAssessment) -> str:
        return self._backend.save_trust_assessment(assessment)

    def get_trust_assessments(
        self,
        *,
        entity_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[TrustAssessment]:
        assessments = self._backend.get_trust_assessments()
        if entity_id:
            assessments = [a for a in assessments if a.entity == entity_id]
        return assessments

    def compute_trust(
        self,
        entity_id: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate trust for an entity."""
        assessment = self._backend.get_trust_assessment(entity_id)
        if not assessment:
            return {
                "entity": entity_id,
                "domain": domain or "general",
                "score": 0.5,
                "source": "default",
            }
        dimensions = assessment.dimensions or {}
        d = domain or "general"
        dim_data = dimensions.get(d, {})
        score = dim_data.get("score", 0.5) if isinstance(dim_data, dict) else 0.5
        return {
            "entity": entity_id,
            "domain": d,
            "score": score,
            "source": "assessment",
        }

    # ---- Features ----

    def consolidate(
        self,
        *,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run memory consolidation."""
        episodes = self._backend.get_episodes(limit=50)
        if len(episodes) < 3:
            return {
                "consolidated": 0,
                "lessons_found": 0,
                "message": "Need at least 3 episodes to consolidate",
            }
        all_lessons = []
        for ep in episodes:
            if ep.lessons:
                all_lessons.extend(ep.lessons)
        lesson_counts = Counter(all_lessons)
        common = [lesson for lesson, cnt in lesson_counts.items() if cnt >= 2]
        return {
            "consolidated": len(episodes),
            "lessons_found": len(common),
            "common_lessons": common[:5],
        }

    def apply_forgetting(
        self,
        *,
        protect_identity: bool = True,
    ) -> Dict[str, Any]:
        """Apply salience-based forgetting."""
        report = self.run_forgetting_cycle(
            threshold=0.3,
            limit=10,
            dry_run=False,
        )
        return {
            "forgotten": report.get("forgotten", 0),
            "candidates": report.get("candidate_count", 0),
            "protected": report.get("protected", 0),
        }

    # ---- Sync ----

    def sync(self) -> ProtocolSyncResult:
        storage_result = self._backend.sync()
        return ProtocolSyncResult(
            pushed=storage_result.pushed,
            pulled=storage_result.pulled,
            conflicts=storage_result.conflict_count,
            errors=storage_result.errors,
        )

    def pull_changes(self, *, since: Optional[datetime] = None) -> ProtocolSyncResult:
        storage_result = self._backend.pull_changes(since=since)
        return ProtocolSyncResult(
            pushed=storage_result.pushed,
            pulled=storage_result.pulled,
            conflicts=storage_result.conflict_count,
            errors=storage_result.errors,
        )

    def get_pending_sync_count(self) -> int:
        return self._backend.get_pending_sync_count()

    def is_online(self) -> bool:
        return self._backend.is_online()

    # ---- Stats & Export ----

    def get_stats(self) -> Dict[str, int]:
        return self._backend.get_stats()

    def dump(
        self,
        *,
        format: str = "markdown",
        include_raw: bool = True,
        include_forgotten: bool = False,
    ) -> str:
        """Export all memories as a formatted string."""
        if format == "json":
            return self._dump_json(include_raw, include_forgotten)
        return self._dump_markdown(include_raw, include_forgotten)

    def export(self, path: str, *, format: str = "markdown") -> None:
        """Export all memories to a file."""
        content = self.dump(format=format)
        if format == "markdown" and path.endswith(".json"):
            content = self.dump(format="json")
        elif format == "json" and (path.endswith(".md") or path.endswith(".markdown")):
            content = self.dump(format="markdown")
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(content, encoding="utf-8")

    # ---- Composition Hooks ----

    def on_attach(
        self,
        core_id: str,
        inference: Optional[InferenceService] = None,
    ) -> None:
        self._attached_core_id = core_id
        self._inference = inference
        for component in self._components.values():
            component.set_inference(inference)

    def on_detach(self, core_id: str) -> None:
        self._attached_core_id = None
        self._inference = None
        for component in self._components.values():
            component.set_inference(None)

    def on_model_changed(
        self,
        inference: Optional[InferenceService],
    ) -> None:
        self._inference = inference
        for component in self._components.values():
            component.set_inference(inference)

    # ---- Private Helpers ----

    @staticmethod
    def _record_to_text(memory_type: str, record: Any) -> str:
        """Get text representation of a record for token estimation."""
        if memory_type == "value":
            return f"{record.name}: {record.statement}"
        elif memory_type == "belief":
            return record.statement
        elif memory_type == "goal":
            return f"{record.title} {record.description or ''}"
        elif memory_type == "drive":
            return f"{record.drive_type}: {record.focus_areas or ''}"
        elif memory_type == "episode":
            return f"{record.objective} {record.outcome}"
        elif memory_type == "note":
            return record.content
        elif memory_type == "relationship":
            return f"{record.entity_name}: {record.notes or ''}"
        elif memory_type == "summary":
            return f"[{record.scope}] {record.content}"
        elif memory_type == "self_narrative":
            return f"[{record.narrative_type}] {record.content}"
        return str(record)

    def _dump_markdown(self, include_raw: bool, include_forgotten: bool) -> str:
        """Export memory as markdown."""
        lines = [
            f"# Memory Dump for {self.stack_id}",
            f"_Exported at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
        ]
        values = self._backend.get_values(limit=100)
        if values:
            lines.append("## Values")
            for v in sorted(values, key=lambda x: x.priority, reverse=True):
                lines.append(f"- **{v.name}** (priority {v.priority}): {v.statement}")
            lines.append("")

        beliefs = self._backend.get_beliefs(limit=100)
        if beliefs:
            lines.append("## Beliefs")
            for b in sorted(beliefs, key=lambda x: x.confidence, reverse=True):
                lines.append(f"- [{b.confidence:.0%}] {b.statement}")
            lines.append("")

        goals = self._backend.get_goals(status=None, limit=100)
        if goals:
            lines.append("## Goals")
            for g in goals:
                icon = "+" if g.status == "completed" else "o" if g.status == "active" else "-"
                lines.append(f"- {icon} [{g.priority}] {g.title}")
            lines.append("")

        episodes = self._backend.get_episodes(limit=100)
        if episodes:
            lines.append("## Episodes")
            for e in episodes:
                date_str = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
                lines.append(f"### {e.objective}")
                lines.append(f"*{date_str}* | {e.outcome}")
                if e.lessons:
                    for lesson in e.lessons:
                        lines.append(f"  - {lesson}")
                lines.append("")

        notes = self._backend.get_notes(limit=100)
        if notes:
            lines.append("## Notes")
            for n in notes:
                lines.append(f"- [{n.note_type}] {n.content}")
            lines.append("")

        if include_raw:
            raw_entries = self._backend.list_raw(limit=100)
            if raw_entries:
                lines.append("## Raw Entries")
                for r in raw_entries:
                    status = "done" if r.processed else "pending"
                    lines.append(f"- [{status}] {r.content or r.blob or ''}")
                lines.append("")

        return "\n".join(lines)

    def _dump_json(self, include_raw: bool, include_forgotten: bool) -> str:
        """Export memory as JSON."""

        def _dt(dt: Optional[datetime]) -> Optional[str]:
            return dt.isoformat() if dt else None

        data: Dict[str, Any] = {
            "stack_id": self.stack_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "values": [
                {"id": v.id, "name": v.name, "statement": v.statement, "priority": v.priority}
                for v in self._backend.get_values(limit=100)
            ],
            "beliefs": [
                {"id": b.id, "statement": b.statement, "confidence": b.confidence}
                for b in self._backend.get_beliefs(limit=100)
            ],
            "goals": [
                {"id": g.id, "title": g.title, "status": g.status, "priority": g.priority}
                for g in self._backend.get_goals(status=None, limit=100)
            ],
            "episodes": [
                {
                    "id": e.id,
                    "objective": e.objective,
                    "outcome": e.outcome,
                    "created_at": _dt(e.created_at),
                }
                for e in self._backend.get_episodes(limit=100)
            ],
            "notes": [
                {"id": n.id, "content": n.content, "note_type": n.note_type}
                for n in self._backend.get_notes(limit=100)
            ],
        }
        if include_raw:
            data["raw_entries"] = [
                {"id": r.id, "content": r.content, "processed": r.processed}
                for r in self._backend.list_raw(limit=100)
            ]
        return json.dumps(data, indent=2, default=str)

    # ---- Mixin compatibility helpers ----
    # These are needed by the mixins that reference methods on Kernle
    # that don't exist on the stack. We provide minimal stubs.

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint (stub for mixin compatibility)."""
        # Checkpoints are a Kernle concept, not a stack concept.
        # Return None to signal no checkpoint.
        return None

    def list_raw(self, processed: Optional[bool] = None, limit: int = 100) -> List[Any]:
        """List raw entries (mixin compatibility)."""
        entries = self._backend.list_raw(processed=processed, limit=limit)
        # Return as dicts for AnxietyMixin compatibility
        result = []
        for e in entries:
            result.append(
                {
                    "id": e.id,
                    "captured_at": e.captured_at.isoformat() if e.captured_at else None,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "content": e.content,
                    "processed": e.processed,
                }
            )
        return result

    def get_identity_confidence(self) -> float:
        """Get identity confidence (mixin compatibility).

        Measures coherence from values and beliefs.
        """
        values = self._backend.get_values(limit=10)
        beliefs = self._backend.get_beliefs(limit=20)
        if not values and not beliefs:
            return 0.0
        total_conf = 0.0
        count = 0
        for v in values:
            total_conf += v.confidence
            count += 1
        for b in beliefs:
            total_conf += b.confidence
            count += 1
        return total_conf / count if count > 0 else 0.0

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status (mixin compatibility)."""
        return {
            "online": self._backend.is_online(),
            "pending": self._backend.get_pending_sync_count(),
        }

    def synthesize_identity(self) -> Dict[str, Any]:
        """Synthesize identity (stub for mixin compatibility)."""
        return {"confidence": self.get_identity_confidence()}

    def checkpoint(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Checkpoint (stub for mixin compatibility)."""
        return None

    def episode(self, **kwargs: Any) -> Optional[str]:
        """Episode creation (stub for mixin compatibility)."""
        return None

    def boot_list(self) -> Dict[str, str]:
        """Boot config (stub for mixin compatibility)."""
        return {}
