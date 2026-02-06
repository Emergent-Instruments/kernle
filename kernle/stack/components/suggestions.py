"""Suggestion extraction stack component.

Provides pattern-based extraction of memory suggestions from raw entries.
Detects potential episodes, beliefs, and notes using regex patterns.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult
from kernle.types import MemorySuggestion

logger = logging.getLogger(__name__)

# Episode detection patterns
EPISODE_PATTERNS = [
    (r"\b(completed|finished|shipped|deployed|released|launched)\b", 0.7),
    (r"\b(did|made|built|created|implemented|fixed|resolved)\b", 0.6),
    (r"\b(worked on|working on|tackled|handled)\b", 0.5),
    (r"\b(succeeded|success|failed|failure|partial|blocked)\b", 0.7),
    (r"\b(achieved|accomplished|delivered)\b", 0.7),
    (r"\b(learned|discovered|realized|figured out|understood)\b", 0.6),
    (r"\b(lesson|takeaway|insight from)\b", 0.7),
]

# Belief detection patterns
BELIEF_PATTERNS = [
    (r"\b(i think|i believe|i feel that|in my opinion)\b", 0.8),
    (r"\b(seems like|appears that|looks like)\b", 0.6),
    (r"\b(always|never|usually|typically|generally)\b", 0.6),
    (r"\b(should|must|need to|have to)\b", 0.5),
    (r"\b(is better than|is worse than|prefer|favorite)\b", 0.7),
    (r"\b(the best way|the right way|the wrong way)\b", 0.8),
    (r"\b(pattern|principle|rule|guideline)\b", 0.7),
]

# Note detection patterns
NOTE_PATTERNS = [
    (r'["\'].*["\']', 0.6),
    (r"\b(said|told me|mentioned|asked)\b", 0.5),
    (r"\b(decided|decision|chose|choose|will)\b", 0.7),
    (r"\b(going to|plan to|planning)\b", 0.6),
    (r"\b(noticed|observed|saw that|found that)\b", 0.6),
    (r"\b(interesting|important|noteworthy|key)\b", 0.5),
    (r"\b(remember that|note that|don\'t forget)\b", 0.7),
]


class SuggestionComponent:
    """Suggestion extraction component.

    Extracts memory suggestions from raw entries during maintenance
    using pattern-based detection. When inference is available, can
    use the model for richer extraction.
    """

    name = "suggestions"
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
        """Extract suggestions from unprocessed raw entries."""
        if self._storage is None:
            logger.debug("SuggestionComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        raw_entries = self._storage.list_raw(processed=False, limit=50)
        total_suggestions = 0

        for entry in raw_entries:
            suggestions = self._extract_suggestions(entry)
            for suggestion in suggestions:
                self._storage.save_suggestion(suggestion)
                total_suggestions += 1

        result: Dict[str, Any] = {
            "raw_entries_processed": len(raw_entries),
            "suggestions_extracted": total_suggestions,
        }

        if self._inference is None and total_suggestions == 0:
            logger.debug("SuggestionComponent: no inference, pattern-only extraction")
            result["inference_available"] = False
        else:
            result["inference_available"] = self._inference is not None

        return result

    # ---- Core Logic ----

    def _extract_suggestions(self, raw_entry: Any) -> List[MemorySuggestion]:
        """Extract memory suggestions from a raw entry."""
        content = (
            getattr(raw_entry, "blob", None) or getattr(raw_entry, "content", None) or ""
        ).lower()
        suggestions = []
        threshold = 0.4

        episode_score = self._score_patterns(content, EPISODE_PATTERNS)
        belief_score = self._score_patterns(content, BELIEF_PATTERNS)
        note_score = self._score_patterns(content, NOTE_PATTERNS)

        if episode_score >= threshold:
            suggestion = self._make_suggestion(raw_entry, "episode", episode_score)
            if suggestion:
                suggestions.append(suggestion)

        if belief_score >= threshold:
            suggestion = self._make_suggestion(raw_entry, "belief", belief_score)
            if suggestion:
                suggestions.append(suggestion)

        if note_score >= threshold and episode_score < threshold and belief_score < threshold:
            suggestion = self._make_suggestion(raw_entry, "note", note_score)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    def _score_patterns(self, content: str, patterns: List[tuple]) -> float:
        """Score content against a set of patterns."""
        total_weight = 0.0
        matched_weight = 0.0

        for pattern, weight in patterns:
            total_weight += weight
            if re.search(pattern, content, re.IGNORECASE):
                matched_weight += weight

        if total_weight == 0:
            return 0.0
        return min(1.0, matched_weight / (total_weight * 0.5))

    def _make_suggestion(
        self, raw_entry: Any, memory_type: str, confidence: float
    ) -> Optional[MemorySuggestion]:
        """Create a suggestion for the given memory type."""
        content_text = getattr(raw_entry, "blob", None) or getattr(raw_entry, "content", None) or ""
        if len(content_text.strip()) < 10:
            return None

        if memory_type == "episode":
            content_dict = {
                "objective": content_text[:200].strip(),
                "outcome": "Extracted from raw capture",
                "outcome_type": "unknown",
            }
        elif memory_type == "belief":
            content_dict = {
                "statement": content_text[:500].strip(),
                "belief_type": "fact",
                "confidence": min(0.8, confidence),
            }
        else:
            content_dict = {
                "content": content_text.strip(),
                "note_type": "note",
            }

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            stack_id=self._stack_id or "",
            memory_type=memory_type,
            content=content_dict,
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
