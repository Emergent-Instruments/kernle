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
    MaintenanceModeError,
    ProvenanceError,
    StackComponentProtocol,
    StackState,
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


# Provenance hierarchy: which source types are allowed for each memory type
PROVENANCE_RULES: Dict[str, List[str]] = {
    "episode": ["raw"],
    "note": ["raw"],
    "belief": ["episode", "note"],
    "goal": ["episode", "belief"],
    "relationship": ["episode"],
    "value": ["belief"],
    "drive": ["episode", "belief"],
}

# Annotation ref types: valid in derived_from but not traversable provenance.
# These are metadata markers (e.g., "context:cli", "kernle:system") that
# indicate how/where a memory was created, not what it was derived from.
# Lineage.py also skips these during cycle detection.
ANNOTATION_REF_TYPES = {"context", "kernle"}


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
        components: Optional[List[StackComponentProtocol]] = None,
        enforce_provenance: bool = False,
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
        self._enforce_provenance = enforce_provenance
        self._registered_plugins: set = set()

        # Load persisted state or default to INITIALIZING
        persisted_state = self._backend.get_stack_setting("stack_state")
        if persisted_state and persisted_state in StackState.__members__:
            self._state: StackState = StackState[persisted_state]
        else:
            self._state = StackState.INITIALIZING

        # Load enforce_provenance from settings if not explicitly set
        if not enforce_provenance:
            persisted_provenance = self._backend.get_stack_setting("enforce_provenance")
            if persisted_provenance == "true":
                self._enforce_provenance = True

        # Auto-load components: None = defaults, [] = bare, list = explicit
        if components is None:
            from kernle.stack.components import get_default_components

            components = get_default_components()
        for component in components:
            self.add_component(component)

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
        if hasattr(component, "set_storage"):
            component.set_storage(self._storage)
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

    # ---- Plugin Registration ----

    def register_plugin(self, plugin_name: str) -> None:
        """Register a plugin name as trusted for provenance bypass."""
        self._registered_plugins.add(plugin_name)

    def unregister_plugin(self, plugin_name: str) -> None:
        """Remove a plugin from the trusted set."""
        self._registered_plugins.discard(plugin_name)

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

    # ---- Component Dispatch Helpers ----

    def _dispatch_on_save(self, memory_type: str, memory_id: str, memory: Any) -> None:
        """Notify components after a memory is saved."""
        for name, component in self._components.items():
            try:
                component.on_save(memory_type, memory_id, memory)
            except Exception as e:
                logger.warning("Component '%s' on_save failed: %s", name, e)

    def _dispatch_on_search(
        self, query: str, results: List[ProtocolSearchResult]
    ) -> List[ProtocolSearchResult]:
        """Let components modify search results."""
        for name, component in self._components.items():
            try:
                modified = component.on_search(query, results)
                if modified is not None:
                    results = modified
            except Exception as e:
                logger.warning("Component '%s' on_search failed: %s", name, e)
        return results

    def _dispatch_on_load(self, context: Dict[str, Any]) -> None:
        """Notify components when working memory is loaded."""
        for name, component in self._components.items():
            try:
                component.on_load(context)
            except Exception as e:
                logger.warning("Component '%s' on_load failed: %s", name, e)

    # ---- State Management ----

    @property
    def state(self) -> StackState:
        """Current lifecycle state of the stack."""
        return self._state

    def enter_maintenance(self) -> None:
        """Enter maintenance mode. Only controlled admin operations allowed."""
        if self._state == StackState.MAINTENANCE:
            return
        self._state = StackState.MAINTENANCE
        self._backend.set_stack_setting("stack_state", StackState.MAINTENANCE.name)

    def exit_maintenance(self) -> None:
        """Exit maintenance mode, returning to ACTIVE state."""
        if self._state != StackState.MAINTENANCE:
            return
        self._state = StackState.ACTIVE
        self._backend.set_stack_setting("stack_state", StackState.ACTIVE.name)

    # ---- Provenance Validation ----

    def _validate_provenance(
        self, memory_type: str, derived_from: Optional[list], source_entity: Optional[str] = None
    ) -> None:
        """Validate provenance for a memory write.

        Only enforced when stack is ACTIVE. INITIALIZING allows any write
        (for seed data). MAINTENANCE always rejects writes (independent of
        provenance flag). Plugin-sourced writes have relaxed provenance
        requirements but are still blocked in maintenance mode.

        Raises:
            ProvenanceError: If provenance is missing or invalid
            MaintenanceModeError: If stack is in maintenance mode
        """
        if self._state == StackState.INITIALIZING:
            return  # Seed writes don't need provenance

        # Maintenance mode always blocks writes, regardless of provenance flag
        if self._state == StackState.MAINTENANCE:
            raise MaintenanceModeError(
                f"Cannot save {memory_type} in maintenance mode. " "Use exit_maintenance() first."
            )

        if not self._enforce_provenance:
            return  # Provenance enforcement disabled

        # Plugin-sourced writes have relaxed provenance requirements,
        # but only for plugins actually registered with this stack
        if source_entity and source_entity.startswith("plugin:"):
            plugin_name = source_entity[len("plugin:") :]
            if plugin_name in self._registered_plugins:
                return

        # Raw entries don't need provenance
        if memory_type not in PROVENANCE_RULES:
            return

        allowed_types = PROVENANCE_RULES[memory_type]

        if not derived_from:
            raise ProvenanceError(
                f"Cannot save {memory_type} without provenance. "
                f"derived_from must cite at least one: {', '.join(allowed_types)}"
            )

        has_real_ref = False
        for ref in derived_from:
            if ":" not in ref:
                raise ProvenanceError(
                    f"Invalid provenance reference '{ref}'. "
                    "Expected format 'type:id' (e.g., 'episode:abc123')"
                )
            ref_type, ref_id = ref.split(":", 1)
            # Annotation refs (context:, kernle:) are metadata markers,
            # not provenance sources. Skip hierarchy/existence checks.
            if ref_type in ANNOTATION_REF_TYPES:
                continue
            has_real_ref = True
            if ref_type not in allowed_types:
                raise ProvenanceError(
                    f"Invalid provenance for {memory_type}: '{ref_type}' is not an allowed source. "
                    f"Allowed sources: {', '.join(allowed_types)}"
                )
            # Verify the referenced memory exists
            if not self._backend.memory_exists(ref_type, ref_id):
                raise ProvenanceError(f"Referenced {ref_type}:{ref_id} does not exist in the stack")

        # Must have at least one real provenance ref (not just annotations)
        if not has_real_ref:
            raise ProvenanceError(
                f"Cannot save {memory_type} with only annotation refs. "
                f"derived_from must cite at least one: {', '.join(allowed_types)}"
            )

    # ---- Write Operations ----

    def save_episode(self, episode: Episode) -> str:
        self._validate_provenance(
            "episode", episode.derived_from, getattr(episode, "source_entity", None)
        )
        result_id = self._backend.save_episode(episode)
        self._dispatch_on_save("episode", result_id, episode)
        return result_id

    def save_belief(self, belief: Belief) -> str:
        self._validate_provenance(
            "belief", belief.derived_from, getattr(belief, "source_entity", None)
        )
        result_id = self._backend.save_belief(belief)
        self._dispatch_on_save("belief", result_id, belief)
        return result_id

    def save_value(self, value: Value) -> str:
        self._validate_provenance(
            "value", value.derived_from, getattr(value, "source_entity", None)
        )
        result_id = self._backend.save_value(value)
        self._dispatch_on_save("value", result_id, value)
        return result_id

    def save_goal(self, goal: Goal) -> str:
        self._validate_provenance("goal", goal.derived_from, getattr(goal, "source_entity", None))
        result_id = self._backend.save_goal(goal)
        self._dispatch_on_save("goal", result_id, goal)
        return result_id

    def save_note(self, note: Note) -> str:
        self._validate_provenance("note", note.derived_from, getattr(note, "source_entity", None))
        result_id = self._backend.save_note(note)
        self._dispatch_on_save("note", result_id, note)
        return result_id

    def save_drive(self, drive: Drive) -> str:
        self._validate_provenance(
            "drive", drive.derived_from, getattr(drive, "source_entity", None)
        )
        result_id = self._backend.save_drive(drive)
        self._dispatch_on_save("drive", result_id, drive)
        return result_id

    def save_relationship(self, relationship: Relationship) -> str:
        self._validate_provenance(
            "relationship", relationship.derived_from, getattr(relationship, "source_entity", None)
        )
        result_id = self._backend.save_relationship(relationship)
        self._dispatch_on_save("relationship", result_id, relationship)
        return result_id

    def save_raw(self, raw: RawEntry) -> str:
        self._validate_provenance("raw", None)  # Raw entries need no provenance
        result_id = self._backend.save_raw(
            blob=raw.blob or raw.content or "",
            source=raw.source,
        )
        self._dispatch_on_save("raw", result_id, raw)
        return result_id

    def save_playbook(self, playbook: Playbook) -> str:
        result_id = self._backend.save_playbook(playbook)
        self._dispatch_on_save("playbook", result_id, playbook)
        return result_id

    def save_epoch(self, epoch: Epoch) -> str:
        result_id = self._backend.save_epoch(epoch)
        self._dispatch_on_save("epoch", result_id, epoch)
        return result_id

    def save_summary(self, summary: Summary) -> str:
        result_id = self._backend.save_summary(summary)
        self._dispatch_on_save("summary", result_id, summary)
        return result_id

    def save_self_narrative(self, narrative: SelfNarrative) -> str:
        result_id = self._backend.save_self_narrative(narrative)
        self._dispatch_on_save("self_narrative", result_id, narrative)
        return result_id

    def save_suggestion(self, suggestion: MemorySuggestion) -> str:
        result_id = self._backend.save_suggestion(suggestion)
        self._dispatch_on_save("suggestion", result_id, suggestion)
        return result_id

    # ---- Batch Write ----

    def save_episodes_batch(self, episodes: List[Episode]) -> List[str]:
        for ep in episodes:
            self._validate_provenance(
                "episode", ep.derived_from, getattr(ep, "source_entity", None)
            )
        ids = self._backend.save_episodes_batch(episodes)
        for ep, eid in zip(episodes, ids):
            self._dispatch_on_save("episode", eid, ep)
        return ids

    def save_beliefs_batch(self, beliefs: List[Belief]) -> List[str]:
        for belief in beliefs:
            self._validate_provenance(
                "belief", belief.derived_from, getattr(belief, "source_entity", None)
            )
        ids = self._backend.save_beliefs_batch(beliefs)
        for belief, bid in zip(beliefs, ids):
            self._dispatch_on_save("belief", bid, belief)
        return ids

    def save_notes_batch(self, notes: List[Note]) -> List[str]:
        for note in notes:
            self._validate_provenance(
                "note", note.derived_from, getattr(note, "source_entity", None)
            )
        ids = self._backend.save_notes_batch(notes)
        for note, nid in zip(notes, ids):
            self._dispatch_on_save("note", nid, note)
        return ids

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
            episodes = [e for e in episodes if e.strength > 0.0]
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
            beliefs = [b for b in beliefs if b.strength > 0.0]
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
            values = [v for v in values if v.strength > 0.0]
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
            goals = [g for g in goals if g.strength > 0.0]
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
            notes = [n for n in notes if n.strength > 0.0]
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

            # Exclude forgotten/dormant memories from search
            record_strength = getattr(record, "strength", 1.0)
            if record_strength <= 0.0:
                continue

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
        results = self._dispatch_on_search(query, results)
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
            result = {
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
            self._dispatch_on_load(result)
            return result

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

        self._dispatch_on_load(result)
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
        success = self._backend.forget_memory(memory_type, memory_id, reason)
        if success:
            # Cascade: flag direct children via audit entries
            children = self._backend.get_memories_derived_from(memory_type, memory_id)
            for child_type, child_id in children:
                self._backend.log_audit(
                    child_type,
                    child_id,
                    "cascade_flag",
                    "system",
                    {"cascade_source": f"{memory_type}:{memory_id}", "reason": "source_forgotten"},
                )
        return success

    def recover_memory(self, memory_type: str, memory_id: str) -> bool:
        return self._backend.recover_memory(memory_type, memory_id)

    def protect_memory(
        self,
        memory_type: str,
        memory_id: str,
        protected: bool = True,
    ) -> bool:
        return self._backend.protect_memory(memory_type, memory_id, protected)

    def weaken_memory(
        self,
        memory_type: str,
        memory_id: str,
        amount: float,
    ) -> bool:
        # Check current strength before weakening to determine if cascade needed
        memory = self._backend.get_memory(memory_type, memory_id)
        old_strength = getattr(memory, "strength", 1.0) if memory else 1.0

        success = self._backend.weaken_memory(memory_type, memory_id, amount)
        if success:
            new_strength = old_strength - abs(amount)
            if new_strength < 0.0:
                new_strength = 0.0
            # Cascade only if strength drops below 0.2 (dormant threshold)
            if new_strength < 0.2 and old_strength >= 0.2:
                children = self._backend.get_memories_derived_from(memory_type, memory_id)
                for child_type, child_id in children:
                    self._backend.log_audit(
                        child_type,
                        child_id,
                        "cascade_flag",
                        "system",
                        {
                            "cascade_source": f"{memory_type}:{memory_id}",
                            "reason": "source_dormant",
                        },
                    )
        return success

    def verify_memory(
        self,
        memory_type: str,
        memory_id: str,
    ) -> bool:
        success = self._backend.verify_memory(memory_type, memory_id)
        if success:
            # Boost source memories referenced in derived_from
            memory = self._backend.get_memory(memory_type, memory_id)
            if memory:
                derived_from = getattr(memory, "derived_from", None) or []
                for ref in derived_from:
                    if not ref or ":" not in ref:
                        continue
                    ref_type, ref_id = ref.split(":", 1)
                    # Skip annotation refs
                    if ref_type in ANNOTATION_REF_TYPES:
                        continue
                    self._backend.boost_memory_strength(ref_type, ref_id, 0.02)
        return success

    # ---- Cascade Queries ----

    def get_memories_derived_from(self, memory_type: str, memory_id: str) -> List[tuple]:
        """Find all memories that cite 'type:id' in their derived_from."""
        return self._backend.get_memories_derived_from(memory_type, memory_id)

    def get_ungrounded_memories(self) -> List[tuple]:
        """Find memories where ALL source refs have strength 0.0 or don't exist."""
        return self._backend.get_ungrounded_memories(self.stack_id)

    def log_audit(
        self,
        memory_type: str,
        memory_id: str,
        operation: str,
        *,
        actor: str = "system",
        details: Optional[Any] = None,
    ) -> str:
        return self._backend.log_audit(memory_type, memory_id, operation, actor, details)

    def get_audit_log(
        self,
        *,
        memory_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 50,
    ) -> List[Any]:
        return self._backend.get_audit_log(
            memory_type=memory_type,
            memory_id=memory_id,
            operation=operation,
            limit=limit,
        )

    # ---- Processing ----

    def get_processing_config(self) -> List[Dict[str, Any]]:
        """Get all processing configuration entries."""
        return self._backend.get_processing_config()

    def set_processing_config(
        self,
        layer_transition: str,
        **kwargs: Any,
    ) -> bool:
        """Update processing configuration for a layer transition."""
        return self._backend.set_processing_config(layer_transition, **kwargs)

    def mark_episode_processed(self, episode_id: str) -> bool:
        """Mark an episode as processed."""
        return self._backend.mark_episode_processed(episode_id)

    def mark_note_processed(self, note_id: str) -> bool:
        """Mark a note as processed."""
        return self._backend.mark_note_processed(note_id)

    def mark_belief_processed(self, belief_id: str) -> bool:
        """Mark a belief as processed."""
        return self._backend.mark_belief_processed(belief_id)

    # ---- Stack Settings ----

    def get_stack_setting(self, key: str) -> Optional[str]:
        """Get a stack setting value by key."""
        return self._backend.get_stack_setting(key)

    def set_stack_setting(self, key: str, value: str) -> None:
        """Set a stack setting (upsert). Updates in-memory state for known keys."""
        self._backend.set_stack_setting(key, value)
        # Sync in-memory flags so changes take effect immediately
        if key == "enforce_provenance":
            self._enforce_provenance = value == "true"
        elif key == "stack_state" and value in StackState.__members__:
            self._state = StackState[value]

    def get_all_stack_settings(self) -> Dict[str, str]:
        """Get all stack settings as a dict."""
        return self._backend.get_all_stack_settings()

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
        # Transition to ACTIVE on first attach (provenance enforcement begins)
        if self._state == StackState.INITIALIZING:
            self._state = StackState.ACTIVE
            self._backend.set_stack_setting("stack_state", StackState.ACTIVE.name)

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
