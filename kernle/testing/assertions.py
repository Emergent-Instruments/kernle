"""Cognitive quality assertions for validating memory stack health.

Provides structural, coherence, quality, and pipeline health checks
that can be run against any StackProtocol implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from kernle.processing import compute_content_hash

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """Result of a single cognitive assertion."""

    category: str  # "structural", "coherence", "quality", "pipeline"
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveReport:
    """Aggregate report from running cognitive assertions."""

    assertions: List[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.assertions if a.passed)

    @property
    def failed(self) -> int:
        return sum(1 for a in self.assertions if not a.passed)

    @property
    def total(self) -> int:
        return len(self.assertions)

    @property
    def all_passed(self) -> bool:
        return all(a.passed for a in self.assertions)

    def summary(self) -> str:
        lines = [f"Cognitive Report: {self.passed}/{self.total} passed"]
        for a in self.assertions:
            status = "PASS" if a.passed else "FAIL"
            lines.append(f"  [{status}] {a.category}.{a.name}: {a.message}")
        return "\n".join(lines)


# Memory types that support derived_from provenance
_PROVENANCE_TYPES = ("episode", "belief", "value", "goal", "note", "drive", "relationship")

# Layer ordering for hierarchy checks (lower index = lower layer)
_LAYER_ORDER = {
    "raw": 0,
    "episode": 1,
    "note": 1,
    "belief": 2,
    "goal": 2,
    "relationship": 2,
    "drive": 2,
    "value": 3,
}


class CognitiveAssertions:
    """Cognitive quality assertions for a memory stack.

    Uses existing stack query methods to inspect memory health.
    Does NOT modify any memories -- read-only assertions.
    """

    def __init__(self, stack: Any) -> None:
        """Initialize with a stack (StackProtocol or Kernle instance).

        Accepts either a StackProtocol, a Kernle instance (extracts ._storage),
        or a SQLiteStorage directly.
        """
        if hasattr(stack, "_storage"):
            self._storage = stack._storage
        else:
            self._storage = stack

    def _get_all_memories(self) -> Dict[str, list]:
        """Fetch all memories grouped by type."""
        s = self._storage
        return {
            "episode": s.get_episodes(limit=10000),
            "belief": s.get_beliefs(limit=10000),
            "value": s.get_values(limit=10000),
            "goal": s.get_goals(limit=10000),
            "note": s.get_notes(limit=10000),
            "drive": s.get_drives(),
            "relationship": s.get_relationships(),
        }

    def _get_memory_by_ref(self, ref: str) -> Any:
        """Resolve a provenance ref like 'raw:abc123' to a memory object."""
        if ":" not in ref:
            return None
        mem_type, mem_id = ref.split(":", 1)
        try:
            if mem_type == "raw":
                return self._storage.get_raw(mem_id)
            if hasattr(self._storage, "get_memory"):
                return self._storage.get_memory(mem_type, mem_id)
        except Exception as exc:
            logger.debug("Swallowed %s in _get_memory_by_ref(%s): %s", type(exc).__name__, ref, exc)
            return None
        return None

    # ---- Structural Assertions ----

    def provenance_chain_intact(self) -> AssertionResult:
        """Check that all derived_from references point to existing memories."""
        all_memories = self._get_all_memories()
        broken_refs: List[Dict[str, str]] = []

        for mem_type, memories in all_memories.items():
            for mem in memories:
                derived = getattr(mem, "derived_from", None) or []
                for ref in derived:
                    target = self._get_memory_by_ref(ref)
                    if target is None:
                        broken_refs.append(
                            {
                                "memory_type": mem_type,
                                "memory_id": mem.id,
                                "broken_ref": ref,
                            }
                        )

        if not broken_refs:
            return AssertionResult(
                category="structural",
                name="provenance_chain_intact",
                passed=True,
                message="All provenance references resolve to existing memories",
            )
        return AssertionResult(
            category="structural",
            name="provenance_chain_intact",
            passed=False,
            message=f"{len(broken_refs)} broken provenance reference(s) found",
            details={"broken_refs": broken_refs},
        )

    def no_orphan_memories(self) -> AssertionResult:
        """Check that identity-layer memories (beliefs, values) have provenance."""
        all_memories = self._get_all_memories()
        orphans: List[Dict[str, str]] = []

        for mem_type in ("belief", "value"):
            for mem in all_memories.get(mem_type, []):
                derived = getattr(mem, "derived_from", None) or []
                if not derived:
                    orphans.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                        }
                    )

        if not orphans:
            return AssertionResult(
                category="structural",
                name="no_orphan_memories",
                passed=True,
                message="All identity-layer memories have provenance",
            )
        return AssertionResult(
            category="structural",
            name="no_orphan_memories",
            passed=False,
            message=f"{len(orphans)} identity-layer memory(ies) without provenance",
            details={"orphans": orphans},
        )

    def valid_source_types(self) -> AssertionResult:
        """Check that all source_type values are in VALID_SOURCE_TYPE_VALUES."""
        from kernle.types import VALID_SOURCE_TYPE_VALUES

        all_memories = self._get_all_memories()
        invalid: List[Dict[str, str]] = []

        for mem_type, memories in all_memories.items():
            for mem in memories:
                st = getattr(mem, "source_type", None)
                if st is not None and st not in VALID_SOURCE_TYPE_VALUES:
                    invalid.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                            "source_type": st,
                        }
                    )

        if not invalid:
            return AssertionResult(
                category="structural",
                name="valid_source_types",
                passed=True,
                message="All source_type values are valid",
            )
        return AssertionResult(
            category="structural",
            name="valid_source_types",
            passed=False,
            message=f"{len(invalid)} invalid source_type value(s) found",
            details={"invalid": invalid},
        )

    def strength_in_range(self) -> AssertionResult:
        """Check that all strength values are in [0.0, 1.0]."""
        all_memories = self._get_all_memories()
        out_of_range: List[Dict[str, Any]] = []

        for mem_type, memories in all_memories.items():
            for mem in memories:
                strength = getattr(mem, "strength", None)
                if strength is not None and (strength < 0.0 or strength > 1.0):
                    out_of_range.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                            "strength": strength,
                        }
                    )

        if not out_of_range:
            return AssertionResult(
                category="structural",
                name="strength_in_range",
                passed=True,
                message="All strength values are in [0.0, 1.0]",
            )
        return AssertionResult(
            category="structural",
            name="strength_in_range",
            passed=False,
            message=f"{len(out_of_range)} strength value(s) out of range",
            details={"out_of_range": out_of_range},
        )

    def no_duplicate_content(self) -> AssertionResult:
        """Check for duplicate content across same memory type."""
        all_memories = self._get_all_memories()
        duplicates: List[Dict[str, Any]] = []

        for mem_type, memories in all_memories.items():
            seen: Dict[str, str] = {}  # content_hash -> first memory ID
            for mem in memories:
                content = self._extract_content(mem_type, mem)
                if not content:
                    continue
                h = compute_content_hash(content)
                if h in seen:
                    duplicates.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                            "duplicate_of": seen[h],
                        }
                    )
                else:
                    seen[h] = mem.id

        if not duplicates:
            return AssertionResult(
                category="structural",
                name="no_duplicate_content",
                passed=True,
                message="No duplicate content found",
            )
        return AssertionResult(
            category="structural",
            name="no_duplicate_content",
            passed=False,
            message=f"{len(duplicates)} duplicate(s) found",
            details={"duplicates": duplicates},
        )

    # ---- Coherence Assertions ----

    def beliefs_have_evidence(self, min_evidence: int = 1) -> AssertionResult:
        """Check that beliefs have at least min_evidence derived_from refs."""
        beliefs = self._storage.get_beliefs(limit=10000)
        lacking: List[Dict[str, Any]] = []

        for b in beliefs:
            derived = getattr(b, "derived_from", None) or []
            if len(derived) < min_evidence:
                lacking.append(
                    {
                        "belief_id": b.id,
                        "evidence_count": len(derived),
                    }
                )

        if not lacking:
            return AssertionResult(
                category="coherence",
                name="beliefs_have_evidence",
                passed=True,
                message=f"All beliefs have >= {min_evidence} evidence ref(s)",
            )
        return AssertionResult(
            category="coherence",
            name="beliefs_have_evidence",
            passed=False,
            message=f"{len(lacking)} belief(s) lack sufficient evidence",
            details={"lacking": lacking},
        )

    def values_from_beliefs(self) -> AssertionResult:
        """Check that values are derived from beliefs (not directly created)."""
        values = self._storage.get_values(limit=10000)
        non_belief_derived: List[Dict[str, Any]] = []

        for v in values:
            derived = getattr(v, "derived_from", None) or []
            if not derived:
                non_belief_derived.append({"value_id": v.id, "reason": "no derived_from"})
                continue
            has_belief_source = any(ref.startswith("belief:") for ref in derived)
            if not has_belief_source:
                non_belief_derived.append(
                    {
                        "value_id": v.id,
                        "reason": "no belief in derived_from",
                        "derived_from": derived,
                    }
                )

        if not non_belief_derived:
            return AssertionResult(
                category="coherence",
                name="values_from_beliefs",
                passed=True,
                message="All values are derived from beliefs",
            )
        return AssertionResult(
            category="coherence",
            name="values_from_beliefs",
            passed=False,
            message=f"{len(non_belief_derived)} value(s) not derived from beliefs",
            details={"non_belief_derived": non_belief_derived},
        )

    def no_circular_provenance(self) -> AssertionResult:
        """Check for circular references in provenance chains."""
        all_memories = self._get_all_memories()
        # Build adjacency: memory_ref -> list of derived_from refs
        graph: Dict[str, List[str]] = {}
        for mem_type, memories in all_memories.items():
            for mem in memories:
                ref = f"{mem_type}:{mem.id}"
                derived = getattr(mem, "derived_from", None) or []
                graph[ref] = list(derived)

        # Also add raw entries to the graph (they have no derived_from)
        raw_entries = self._storage.list_raw(limit=10000)
        for r in raw_entries:
            graph[f"raw:{r.id}"] = []

        # DFS cycle detection
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        on_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in on_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            on_stack.add(node)
            path.append(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor, path)
            path.pop()
            on_stack.discard(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        if not cycles:
            return AssertionResult(
                category="coherence",
                name="no_circular_provenance",
                passed=True,
                message="No circular provenance references found",
            )
        return AssertionResult(
            category="coherence",
            name="no_circular_provenance",
            passed=False,
            message=f"{len(cycles)} circular provenance chain(s) found",
            details={"cycles": cycles},
        )

    def hierarchy_respected(self) -> AssertionResult:
        """Check that provenance flows upward (raw->episode->belief->value)."""
        all_memories = self._get_all_memories()
        violations: List[Dict[str, Any]] = []

        for mem_type, memories in all_memories.items():
            mem_layer = _LAYER_ORDER.get(mem_type)
            if mem_layer is None:
                continue
            for mem in memories:
                derived = getattr(mem, "derived_from", None) or []
                for ref in derived:
                    if ":" not in ref:
                        continue
                    source_type = ref.split(":", 1)[0]
                    source_layer = _LAYER_ORDER.get(source_type)
                    if source_layer is not None and source_layer >= mem_layer:
                        violations.append(
                            {
                                "memory_type": mem_type,
                                "memory_id": mem.id,
                                "ref": ref,
                                "memory_layer": mem_layer,
                                "source_layer": source_layer,
                            }
                        )

        if not violations:
            return AssertionResult(
                category="coherence",
                name="hierarchy_respected",
                passed=True,
                message="All provenance flows from lower to higher layers",
            )
        return AssertionResult(
            category="coherence",
            name="hierarchy_respected",
            passed=False,
            message=f"{len(violations)} hierarchy violation(s) found",
            details={"violations": violations},
        )

    # ---- Quality Assertions ----

    def no_empty_memories(self) -> AssertionResult:
        """Check that no memories have empty content fields."""
        all_memories = self._get_all_memories()
        empty: List[Dict[str, str]] = []

        for mem_type, memories in all_memories.items():
            for mem in memories:
                content = self._extract_content(mem_type, mem)
                if not content or not content.strip():
                    empty.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                        }
                    )

        if not empty:
            return AssertionResult(
                category="quality",
                name="no_empty_memories",
                passed=True,
                message="All memories have non-empty content",
            )
        return AssertionResult(
            category="quality",
            name="no_empty_memories",
            passed=False,
            message=f"{len(empty)} memory(ies) with empty content",
            details={"empty": empty},
        )

    def episodes_have_outcomes(self) -> AssertionResult:
        """Check that all episodes have non-empty outcomes."""
        episodes = self._storage.get_episodes(limit=10000)
        missing: List[str] = []

        for ep in episodes:
            if not ep.outcome or not ep.outcome.strip():
                missing.append(ep.id)

        if not missing:
            return AssertionResult(
                category="quality",
                name="episodes_have_outcomes",
                passed=True,
                message="All episodes have outcomes",
            )
        return AssertionResult(
            category="quality",
            name="episodes_have_outcomes",
            passed=False,
            message=f"{len(missing)} episode(s) missing outcomes",
            details={"episode_ids": missing},
        )

    def beliefs_have_statements(self) -> AssertionResult:
        """Check that all beliefs have non-empty statements."""
        beliefs = self._storage.get_beliefs(limit=10000)
        missing: List[str] = []

        for b in beliefs:
            if not b.statement or not b.statement.strip():
                missing.append(b.id)

        if not missing:
            return AssertionResult(
                category="quality",
                name="beliefs_have_statements",
                passed=True,
                message="All beliefs have statements",
            )
        return AssertionResult(
            category="quality",
            name="beliefs_have_statements",
            passed=False,
            message=f"{len(missing)} belief(s) missing statements",
            details={"belief_ids": missing},
        )

    def confidence_in_range(self) -> AssertionResult:
        """Check that belief confidence values are in [0.0, 1.0]."""
        beliefs = self._storage.get_beliefs(limit=10000)
        out_of_range: List[Dict[str, Any]] = []

        for b in beliefs:
            if b.confidence < 0.0 or b.confidence > 1.0:
                out_of_range.append(
                    {
                        "belief_id": b.id,
                        "confidence": b.confidence,
                    }
                )

        if not out_of_range:
            return AssertionResult(
                category="quality",
                name="confidence_in_range",
                passed=True,
                message="All belief confidence values are in [0.0, 1.0]",
            )
        return AssertionResult(
            category="quality",
            name="confidence_in_range",
            passed=False,
            message=f"{len(out_of_range)} belief(s) with out-of-range confidence",
            details={"out_of_range": out_of_range},
        )

    # ---- Pipeline Health Assertions ----

    def no_unprocessed_raw(self) -> AssertionResult:
        """Check that no raw entries remain unprocessed."""
        unprocessed = self._storage.list_raw(processed=False, limit=10000)

        if not unprocessed:
            return AssertionResult(
                category="pipeline",
                name="no_unprocessed_raw",
                passed=True,
                message="All raw entries have been processed",
            )
        return AssertionResult(
            category="pipeline",
            name="no_unprocessed_raw",
            passed=False,
            message=f"{len(unprocessed)} unprocessed raw entry(ies) remain",
            details={"count": len(unprocessed)},
        )

    def episodes_exist(self, min_count: int = 1) -> AssertionResult:
        """Check that at least min_count episodes exist."""
        episodes = self._storage.get_episodes(limit=min_count)

        if len(episodes) >= min_count:
            return AssertionResult(
                category="pipeline",
                name="episodes_exist",
                passed=True,
                message=f"At least {min_count} episode(s) exist",
            )
        return AssertionResult(
            category="pipeline",
            name="episodes_exist",
            passed=False,
            message=f"Only {len(episodes)} episode(s) exist (need >= {min_count})",
            details={"count": len(episodes), "required": min_count},
        )

    def processing_source_type(self) -> AssertionResult:
        """Check that processing-created memories have source_type='processing' and vice versa.

        Validates both directions:
        - source_type='processing' must have derived_from refs
        - derived_from refs present must have source_type='processing'
        """
        all_memories = self._get_all_memories()
        mismatched: List[Dict[str, str]] = []

        for mem_type, memories in all_memories.items():
            for mem in memories:
                derived = getattr(mem, "derived_from", None) or []
                st = getattr(mem, "source_type", None)
                # source_type='processing' but no provenance refs
                if st == "processing" and not derived:
                    mismatched.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                            "issue": "source_type='processing' but no derived_from",
                        }
                    )
                # Has provenance refs but source_type is not 'processing'
                elif derived and st and st != "processing":
                    mismatched.append(
                        {
                            "memory_type": mem_type,
                            "memory_id": mem.id,
                            "issue": f"has derived_from but source_type='{st}' (expected 'processing')",
                        }
                    )

        if not mismatched:
            return AssertionResult(
                category="pipeline",
                name="processing_source_type",
                passed=True,
                message="Processing-created memories have proper provenance",
            )
        return AssertionResult(
            category="pipeline",
            name="processing_source_type",
            passed=False,
            message=f"{len(mismatched)} processing memory(ies) with mismatched provenance",
            details={"mismatched": mismatched},
        )

    def raw_entries_marked_processed(self) -> AssertionResult:
        """Check that raw entries with promoted memories are marked processed."""
        all_memories = self._get_all_memories()
        raw_entries = self._storage.list_raw(limit=10000)

        # Collect all raw IDs referenced in derived_from
        referenced_raw_ids: Set[str] = set()
        for mem_type, memories in all_memories.items():
            for mem in memories:
                derived = getattr(mem, "derived_from", None) or []
                for ref in derived:
                    if ref.startswith("raw:"):
                        referenced_raw_ids.add(ref.split(":", 1)[1])

        # Check if referenced raw entries are marked processed
        unmarked: List[str] = []
        for raw in raw_entries:
            if raw.id in referenced_raw_ids and not raw.processed:
                unmarked.append(raw.id)

        if not unmarked:
            return AssertionResult(
                category="pipeline",
                name="raw_entries_marked_processed",
                passed=True,
                message="All referenced raw entries are marked as processed",
            )
        return AssertionResult(
            category="pipeline",
            name="raw_entries_marked_processed",
            passed=False,
            message=f"{len(unmarked)} raw entry(ies) referenced but not marked processed",
            details={"unmarked_ids": unmarked},
        )

    # ---- Batch Runners ----

    def run_structural(self) -> CognitiveReport:
        report = CognitiveReport()
        report.assertions.extend(
            [
                self.provenance_chain_intact(),
                self.no_orphan_memories(),
                self.valid_source_types(),
                self.strength_in_range(),
                self.no_duplicate_content(),
            ]
        )
        return report

    def run_coherence(self) -> CognitiveReport:
        report = CognitiveReport()
        report.assertions.extend(
            [
                self.beliefs_have_evidence(),
                self.values_from_beliefs(),
                self.no_circular_provenance(),
                self.hierarchy_respected(),
            ]
        )
        return report

    def run_quality(self) -> CognitiveReport:
        report = CognitiveReport()
        report.assertions.extend(
            [
                self.no_empty_memories(),
                self.episodes_have_outcomes(),
                self.beliefs_have_statements(),
                self.confidence_in_range(),
            ]
        )
        return report

    def run_pipeline(self) -> CognitiveReport:
        report = CognitiveReport()
        report.assertions.extend(
            [
                self.no_unprocessed_raw(),
                self.episodes_exist(),
                self.processing_source_type(),
                self.raw_entries_marked_processed(),
            ]
        )
        return report

    def run_all(self) -> CognitiveReport:
        report = CognitiveReport()
        report.assertions.extend(self.run_structural().assertions)
        report.assertions.extend(self.run_coherence().assertions)
        report.assertions.extend(self.run_quality().assertions)
        report.assertions.extend(self.run_pipeline().assertions)
        return report

    # ---- Helpers ----

    @staticmethod
    def _extract_content(mem_type: str, mem: Any) -> str:
        """Extract primary content text from a memory for hashing/checking."""
        if mem_type == "episode":
            return f"{mem.objective} {mem.outcome}"
        elif mem_type == "belief":
            return mem.statement
        elif mem_type == "value":
            return f"{mem.name} {mem.statement}"
        elif mem_type == "goal":
            return f"{mem.title} {getattr(mem, 'description', '') or ''}"
        elif mem_type == "note":
            return mem.content
        elif mem_type == "drive":
            return mem.drive_type
        elif mem_type == "relationship":
            return mem.entity_name
        return ""
