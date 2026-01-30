"""Memory suggestion mixin for Kernle.

This module provides auto-extraction of memory suggestions from raw entries,
enabling the system to suggest structured memories while keeping the agent
in control of what gets promoted.

Pattern-based extraction (no LLM calls) detects:
- Episodes: Action words, outcome indicators, completion phrases
- Beliefs: Opinion patterns, generalizations, assertions
- Notes: Quotes, decisions, insights, observations
"""

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.storage.base import MemorySuggestion, RawEntry

if TYPE_CHECKING:
    from kernle.core import Kernle


# Episode detection patterns
EPISODE_PATTERNS = [
    # Action completion
    (r"\b(completed|finished|shipped|deployed|released|launched)\b", 0.7),
    (r"\b(did|made|built|created|implemented|fixed|resolved)\b", 0.6),
    (r"\b(worked on|working on|tackled|handled)\b", 0.5),
    # Outcome words
    (r"\b(succeeded|success|failed|failure|partial|blocked)\b", 0.7),
    (r"\b(achieved|accomplished|delivered)\b", 0.7),
    # Learning indicators
    (r"\b(learned|discovered|realized|figured out|understood)\b", 0.6),
    (r"\b(lesson|takeaway|insight from)\b", 0.7),
]

# Belief detection patterns
BELIEF_PATTERNS = [
    # Opinion phrases
    (r"\b(i think|i believe|i feel that|in my opinion)\b", 0.8),
    (r"\b(seems like|appears that|looks like)\b", 0.6),
    # Generalizations
    (r"\b(always|never|usually|typically|generally)\b", 0.6),
    (r"\b(should|must|need to|have to)\b", 0.5),
    # Assertions
    (r"\b(is better than|is worse than|prefer|favorite)\b", 0.7),
    (r"\b(the best way|the right way|the wrong way)\b", 0.8),
    # Pattern recognition
    (r"\b(pattern|principle|rule|guideline)\b", 0.7),
]

# Note detection patterns
NOTE_PATTERNS = [
    # Quotes
    (r'["\'].*["\']', 0.6),  # Quoted text
    (r"\b(said|told me|mentioned|asked)\b", 0.5),
    # Decisions
    (r"\b(decided|decision|chose|choose|will)\b", 0.7),
    (r"\b(going to|plan to|planning)\b", 0.6),
    # Insights/observations
    (r"\b(noticed|observed|saw that|found that)\b", 0.6),
    (r"\b(interesting|important|noteworthy|key)\b", 0.5),
    # Information
    (r"\b(remember that|note that|don\'t forget)\b", 0.7),
]


class SuggestionsMixin:
    """Mixin providing memory suggestion capabilities.

    Enables:
    - Pattern-based extraction of suggestions from raw entries
    - Review workflow (approve, modify, reject)
    - Promotion of suggestions to structured memories
    """

    def extract_suggestions(
        self: "Kernle",
        raw_entry: RawEntry,
        auto_save: bool = True,
    ) -> List[MemorySuggestion]:
        """Extract memory suggestions from a raw entry.

        Uses pattern-based extraction to identify potential memories.
        Multiple suggestions may be extracted from a single raw entry.

        Args:
            raw_entry: The raw entry to analyze
            auto_save: If True, save extracted suggestions to storage

        Returns:
            List of extracted suggestions
        """
        content = (raw_entry.blob or raw_entry.content or "").lower()
        suggestions = []

        # Score each memory type
        episode_score = self._score_patterns(content, EPISODE_PATTERNS)
        belief_score = self._score_patterns(content, BELIEF_PATTERNS)
        note_score = self._score_patterns(content, NOTE_PATTERNS)

        # Only create suggestions above threshold
        threshold = 0.4

        # Episode suggestion
        if episode_score >= threshold:
            suggestion = self._create_episode_suggestion(raw_entry, episode_score)
            if suggestion:
                suggestions.append(suggestion)

        # Belief suggestion
        if belief_score >= threshold:
            suggestion = self._create_belief_suggestion(raw_entry, belief_score)
            if suggestion:
                suggestions.append(suggestion)

        # Note suggestion (if not already captured as episode/belief)
        if note_score >= threshold and episode_score < threshold and belief_score < threshold:
            suggestion = self._create_note_suggestion(raw_entry, note_score)
            if suggestion:
                suggestions.append(suggestion)

        # Save suggestions if requested
        if auto_save:
            for suggestion in suggestions:
                self._storage.save_suggestion(suggestion)

        return suggestions

    def _score_patterns(
        self: "Kernle",
        content: str,
        patterns: List[tuple],
    ) -> float:
        """Score content against a set of patterns.

        Args:
            content: Text to analyze (lowercase)
            patterns: List of (regex, weight) tuples

        Returns:
            Combined score (0.0 to 1.0)
        """
        total_weight = 0.0
        matched_weight = 0.0

        for pattern, weight in patterns:
            total_weight += weight
            if re.search(pattern, content, re.IGNORECASE):
                matched_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalize and cap at 1.0
        return min(1.0, matched_weight / (total_weight * 0.5))

    def _create_episode_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create an episode suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = raw_entry.blob or raw_entry.content or ""

        # Extract objective (first sentence or line)
        objective = self._extract_first_sentence(content)
        if not objective or len(objective) < 10:
            return None

        # Attempt to extract outcome
        outcome = self._extract_outcome(content)

        # Extract lessons if present
        lessons = self._extract_lessons(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            memory_type="episode",
            content={
                "objective": objective,
                "outcome": outcome or "Extracted from raw capture",
                "outcome_type": self._infer_outcome_type(content),
                "lessons": lessons,
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    def _create_belief_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create a belief suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = raw_entry.blob or raw_entry.content or ""

        # Extract the belief statement
        statement = self._extract_belief_statement(content)
        if not statement or len(statement) < 10:
            return None

        # Infer belief type
        belief_type = self._infer_belief_type(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            memory_type="belief",
            content={
                "statement": statement,
                "belief_type": belief_type,
                "confidence": min(0.8, confidence),  # Start with modest confidence
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    def _create_note_suggestion(
        self: "Kernle",
        raw_entry: RawEntry,
        confidence: float,
    ) -> Optional[MemorySuggestion]:
        """Create a note suggestion from a raw entry.

        Args:
            raw_entry: Source raw entry
            confidence: Extraction confidence

        Returns:
            MemorySuggestion or None
        """
        content = (raw_entry.blob or raw_entry.content or "").strip()
        if len(content) < 10:
            return None

        # Infer note type
        note_type = self._infer_note_type(content)

        # Extract speaker if quote
        speaker = None
        if note_type == "quote":
            speaker = self._extract_speaker(content)

        # Extract reason if decision
        reason = None
        if note_type == "decision":
            reason = self._extract_reason(content)

        return MemorySuggestion(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            memory_type="note",
            content={
                "content": content,
                "note_type": note_type,
                "speaker": speaker,
                "reason": reason,
            },
            confidence=confidence,
            source_raw_ids=[raw_entry.id],
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    # === Helper Methods for Extraction ===

    def _extract_first_sentence(self: "Kernle", content: str) -> str:
        """Extract the first meaningful sentence from content."""
        # Split by sentence boundaries
        sentences = re.split(r"[.!?\n]", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:
                return sentence
        return content[:200] if content else ""

    def _extract_outcome(self: "Kernle", content: str) -> Optional[str]:
        """Try to extract an outcome statement from content."""
        content_lower = content.lower()

        # Look for outcome indicators
        outcome_patterns = [
            r"(result(?:ed)?(?:\s+in)?|outcome|conclusion)[:\s]+(.+?)(?:\.|$)",
            r"(succeeded|failed|achieved|completed)[:\s]*(.+?)(?:\.|$)",
            r"(in the end|finally|ultimately)[,\s]+(.+?)(?:\.|$)",
        ]

        for pattern in outcome_patterns:
            match = re.search(pattern, content_lower)
            if match:
                return match.group(2).strip()[:200]

        return None

    def _extract_lessons(self: "Kernle", content: str) -> List[str]:
        """Extract lesson statements from content."""
        lessons = []
        content_lower = content.lower()

        # Look for lesson indicators
        lesson_patterns = [
            r"(?:learned|lesson|takeaway|insight)[:\s]+(.+?)(?:\.|$)",
            r"(?:realized|discovered|figured out)[:\s]+(.+?)(?:\.|$)",
            r"(?:key point|important)[:\s]+(.+?)(?:\.|$)",
        ]

        for pattern in lesson_patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                lesson = match.strip()
                if len(lesson) >= 10 and lesson not in lessons:
                    lessons.append(lesson[:200])

        return lessons[:5]  # Limit to 5 lessons

    def _infer_outcome_type(self: "Kernle", content: str) -> str:
        """Infer outcome type from content."""
        content_lower = content.lower()

        # Check partial first since it may co-occur with success words
        if any(
            word in content_lower for word in ["partial", "partially", "mostly", "some progress"]
        ):
            return "partial"
        elif any(word in content_lower for word in ["failed", "failure", "blocked", "stuck"]):
            return "failure"
        elif any(
            word in content_lower
            for word in ["success", "succeeded", "achieved", "completed", "shipped"]
        ):
            return "success"
        else:
            return "unknown"

    def _extract_belief_statement(self: "Kernle", content: str) -> str:
        """Extract the core belief statement from content."""
        content = content.strip()

        # Try to find opinion phrases and extract the belief
        patterns = [
            r"i (?:think|believe|feel that)\s+(.+?)(?:\.|$)",
            r"(?:seems like|appears that)\s+(.+?)(?:\.|$)",
            r"(.+?)\s+(?:is better than|is worse than)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:500]

        # Fall back to first sentence
        return self._extract_first_sentence(content)

    def _infer_belief_type(self: "Kernle", content: str) -> str:
        """Infer belief type from content."""
        content_lower = content.lower()

        if any(word in content_lower for word in ["rule", "must", "should always", "never"]):
            return "rule"
        elif any(word in content_lower for word in ["prefer", "like", "favorite", "better"]):
            return "preference"
        elif any(word in content_lower for word in ["constraint", "limit", "cannot", "must not"]):
            return "constraint"
        elif any(word in content_lower for word in ["learned", "discovered", "realized"]):
            return "learned"
        else:
            return "fact"

    def _infer_note_type(self: "Kernle", content: str) -> str:
        """Infer note type from content."""
        content_lower = content.lower()

        # Check for quotes (has quoted text and attribution)
        if re.search(r'["\'].+["\']', content) and any(
            word in content_lower for word in ["said", "told", "mentioned"]
        ):
            return "quote"
        # Check for decisions
        elif any(
            word in content_lower for word in ["decided", "decision", "chose", "going to", "will"]
        ):
            return "decision"
        # Check for insights
        elif any(
            word in content_lower for word in ["insight", "realized", "noticed", "interesting"]
        ):
            return "insight"
        else:
            return "note"

    def _extract_speaker(self: "Kernle", content: str) -> Optional[str]:
        """Extract speaker name from a quote."""
        patterns = [
            r"(\w+)\s+said",
            r"(\w+)\s+told",
            r"(\w+)\s+mentioned",
            r"according to\s+(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_reason(self: "Kernle", content: str) -> Optional[str]:
        """Extract reason from a decision."""
        patterns = [
            r"because\s+(.+?)(?:\.|$)",
            r"reason[:\s]+(.+?)(?:\.|$)",
            r"since\s+(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:200]

        return None

    # === Suggestion Management ===

    def get_suggestions(
        self: "Kernle",
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get memory suggestions.

        Args:
            status: Filter by status (pending, promoted, modified, rejected)
            memory_type: Filter by type (episode, belief, note)
            limit: Maximum suggestions to return

        Returns:
            List of suggestion dicts
        """
        suggestions = self._storage.get_suggestions(
            status=status,
            memory_type=memory_type,
            limit=limit,
        )

        return [self._suggestion_to_dict(s) for s in suggestions]

    def get_suggestion(self: "Kernle", suggestion_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific suggestion by ID.

        Args:
            suggestion_id: ID of the suggestion

        Returns:
            Suggestion dict or None
        """
        suggestion = self._storage.get_suggestion(suggestion_id)
        if suggestion:
            return self._suggestion_to_dict(suggestion)
        return None

    def promote_suggestion(
        self: "Kernle",
        suggestion_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Promote a suggestion to a structured memory.

        Args:
            suggestion_id: ID of the suggestion to promote
            modifications: Optional modifications to apply before promotion

        Returns:
            ID of the created memory, or None if failed
        """
        suggestion = self._storage.get_suggestion(suggestion_id)
        if not suggestion or suggestion.status != "pending":
            return None

        # Apply modifications if provided
        content = suggestion.content.copy()
        if modifications:
            content.update(modifications)

        # Create the appropriate memory type
        memory_id = None
        memory_type = suggestion.memory_type

        if memory_type == "episode":
            memory_id = self.episode(
                objective=content.get("objective", ""),
                outcome=content.get("outcome", ""),
                lessons=content.get("lessons"),
                tags=["auto-suggested"],
            )
        elif memory_type == "belief":
            memory_id = self.belief(
                statement=content.get("statement", ""),
                type=content.get("belief_type", "fact"),
                confidence=content.get("confidence", 0.7),
            )
        elif memory_type == "note":
            memory_id = self.note(
                content=content.get("content", ""),
                type=content.get("note_type", "note"),
                speaker=content.get("speaker"),
                reason=content.get("reason"),
                tags=["auto-suggested"],
            )

        if memory_id:
            # Update suggestion status
            status = "modified" if modifications else "promoted"
            self._storage.update_suggestion_status(
                suggestion_id=suggestion_id,
                status=status,
                promoted_to=f"{memory_type}:{memory_id}",
            )

            # Mark source raw entries as processed
            for raw_id in suggestion.source_raw_ids:
                self._storage.mark_raw_processed(
                    raw_id=raw_id,
                    processed_into=[f"{memory_type}:{memory_id}"],
                )

        return memory_id

    def reject_suggestion(
        self: "Kernle",
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Reject a suggestion.

        Args:
            suggestion_id: ID of the suggestion to reject
            reason: Optional reason for rejection

        Returns:
            True if rejected, False if failed
        """
        return self._storage.update_suggestion_status(
            suggestion_id=suggestion_id,
            status="rejected",
            resolution_reason=reason,
        )

    def _suggestion_to_dict(
        self: "Kernle",
        suggestion: MemorySuggestion,
    ) -> Dict[str, Any]:
        """Convert a suggestion to a dict representation."""
        return {
            "id": suggestion.id,
            "memory_type": suggestion.memory_type,
            "content": suggestion.content,
            "confidence": suggestion.confidence,
            "source_raw_ids": suggestion.source_raw_ids,
            "status": suggestion.status,
            "created_at": suggestion.created_at.isoformat() if suggestion.created_at else None,
            "resolved_at": suggestion.resolved_at.isoformat() if suggestion.resolved_at else None,
            "resolution_reason": suggestion.resolution_reason,
            "promoted_to": suggestion.promoted_to,
        }

    def extract_suggestions_from_unprocessed(
        self: "Kernle",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Extract suggestions from all unprocessed raw entries.

        Useful for batch processing raw entries that haven't been
        analyzed yet.

        Args:
            limit: Maximum raw entries to process

        Returns:
            List of all extracted suggestions
        """
        raw_entries = self._storage.list_raw(processed=False, limit=limit)
        all_suggestions = []

        for entry in raw_entries:
            suggestions = self.extract_suggestions(entry, auto_save=True)
            all_suggestions.extend([self._suggestion_to_dict(s) for s in suggestions])

        return all_suggestions
