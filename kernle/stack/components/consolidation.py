"""Consolidation stack component.

Provides advanced memory consolidation: cross-domain pattern detection,
belief-to-value promotion, and entity model-to-belief promotion.
When inference is available, can synthesize episodes into beliefs.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Thresholds for cross-domain pattern detection
CROSS_DOMAIN_MIN_EPISODES = 2
CROSS_DOMAIN_MIN_DOMAINS = 2


class ConsolidationComponent:
    """Consolidation component for memory pattern synthesis.

    Detects cross-domain patterns, identifies belief-to-value promotion
    candidates, and runs consolidation sweeps during maintenance.
    Inference-dependent operations degrade gracefully without a model.
    """

    name = "consolidation"
    version = "1.0.0"
    required = False
    needs_inference = True

    def __init__(self) -> None:
        self._stack_id: Optional[str] = None
        self._inference: Optional[InferenceService] = None
        self._storage: Optional[Any] = None

    def attach(self, stack_id: str, inference: Optional[InferenceService] = None) -> None:
        self._stack_id = stack_id
        self._inference = inference

    def detach(self) -> None:
        self._stack_id = None
        self._inference = None
        self._storage = None

    def set_inference(self, inference: Optional[InferenceService]) -> None:
        self._inference = inference

    def set_storage(self, storage: Any) -> None:
        """Called by SQLiteStack after attach to provide storage access."""
        self._storage = storage

    # ---- Lifecycle Hooks ----

    def on_save(self, memory_type: str, memory_id: str, memory: Any) -> Any:
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        pass

    def on_maintenance(self) -> Dict[str, Any]:
        """Run consolidation: find common lessons across episodes."""
        if self._storage is None:
            logger.debug("ConsolidationComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        # Pattern detection works without inference
        episodes = self._storage.get_episodes(limit=50)
        if len(episodes) < 3:
            return {
                "consolidated": 0,
                "lessons_found": 0,
                "message": "Need at least 3 episodes to consolidate",
            }

        all_lessons: List[str] = []
        for ep in episodes:
            if ep.lessons:
                all_lessons.extend(ep.lessons)

        lesson_counts = Counter(all_lessons)
        common = [lesson for lesson, cnt in lesson_counts.items() if cnt >= 2]

        result: Dict[str, Any] = {
            "consolidated": len(episodes),
            "lessons_found": len(common),
            "common_lessons": common[:5],
        }

        # Cross-domain pattern detection (no inference needed)
        patterns = self._detect_cross_domain_patterns(episodes)
        if patterns:
            result["cross_domain_patterns"] = len(patterns)

        if self._inference is None:
            logger.debug("ConsolidationComponent: no inference, skipping synthesis")
            result["inference_available"] = False
        else:
            result["inference_available"] = True

        return result

    # ---- Core Logic ----

    def _detect_cross_domain_patterns(self, episodes: list) -> List[Dict[str, Any]]:
        """Detect structural similarities in outcomes across domains."""
        lesson_domain_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for ep in episodes:
            if ep.strength == 0.0:
                continue
            tags = set(ep.tags or [])
            ctx_tags = getattr(ep, "context_tags", None) or []
            all_tags = tags | set(ctx_tags) or {"untagged"}
            otype = ep.outcome_type or "unknown"

            if ep.lessons:
                for lesson in ep.lessons:
                    normalized = lesson.strip().lower()
                    for tag in all_tags:
                        lesson_domain_map[normalized].append(
                            {"domain": tag, "outcome": otype, "episode_id": ep.id}
                        )

        patterns = []
        for lesson, occurrences in lesson_domain_map.items():
            domains_by_outcome: Dict[str, set] = defaultdict(set)
            for occ in occurrences:
                domains_by_outcome[occ["outcome"]].add(occ["domain"])

            for outcome, domains in domains_by_outcome.items():
                if len(domains) >= CROSS_DOMAIN_MIN_DOMAINS:
                    patterns.append(
                        {
                            "lesson": lesson,
                            "outcome": outcome,
                            "domains": sorted(domains),
                        }
                    )

        return patterns
