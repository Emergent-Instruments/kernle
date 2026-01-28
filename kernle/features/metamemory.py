"""Meta-memory mixin for Kernle.

This module provides memory provenance and confidence tracking,
enabling memory verification and lineage analysis.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)


class MetaMemoryMixin:
    """Mixin providing meta-memory capabilities.

    Enables:
    - Confidence tracking for memories
    - Memory verification with evidence
    - Provenance/lineage tracking
    - Uncertainty identification
    """

    def get_memory_confidence(self: "Kernle", memory_type: str, memory_id: str) -> float:
        """Get confidence score for a memory.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note)
            memory_id: ID of the memory

        Returns:
            Confidence score (0.0-1.0), or -1.0 if not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if record:
            return getattr(record, 'confidence', 0.8)
        return -1.0

    def verify_memory(
        self: "Kernle",
        memory_type: str,
        memory_id: str,
        evidence: Optional[str] = None,
    ) -> bool:
        """Verify a memory, increasing its confidence.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            evidence: Optional supporting evidence

        Returns:
            True if verified, False if memory not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return False

        old_confidence = getattr(record, 'confidence', 0.8)
        new_confidence = min(1.0, old_confidence + 0.1)

        # Track confidence change
        confidence_history = getattr(record, 'confidence_history', None) or []
        confidence_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old": old_confidence,
            "new": new_confidence,
            "reason": evidence or "verification",
        })

        return self._storage.update_memory_meta(
            memory_type=memory_type,
            memory_id=memory_id,
            confidence=new_confidence,
            verification_count=(getattr(record, 'verification_count', 0) or 0) + 1,
            last_verified=datetime.now(timezone.utc),
            confidence_history=confidence_history,
        )

    def get_memory_lineage(self: "Kernle", memory_type: str, memory_id: str) -> Dict[str, Any]:
        """Get provenance chain for a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            Lineage information including source and derivations
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}

        return {
            "id": memory_id,
            "type": memory_type,
            "source_type": getattr(record, 'source_type', 'unknown'),
            "source_episodes": getattr(record, 'source_episodes', None),
            "derived_from": getattr(record, 'derived_from', None),
            "current_confidence": getattr(record, 'confidence', None),
            "verification_count": getattr(record, 'verification_count', 0),
            "last_verified": (
                getattr(record, 'last_verified').isoformat()
                if getattr(record, 'last_verified', None) else None
            ),
            "confidence_history": getattr(record, 'confidence_history', None),
        }

    def get_uncertain_memories(
        self: "Kernle",
        threshold: float = 0.5,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get memories with confidence below threshold.

        Args:
            threshold: Confidence threshold
            limit: Maximum results

        Returns:
            List of low-confidence memories
        """
        results = self._storage.get_memories_by_confidence(
            threshold=threshold,
            below=True,
            limit=limit,
        )

        formatted = []
        for r in results:
            record = r.record
            formatted.append({
                "id": record.id,
                "type": r.record_type,
                "confidence": getattr(record, 'confidence', None),
                "summary": self._get_memory_summary(r.record_type, record),
                "created_at": (
                    record.created_at.strftime("%Y-%m-%d")
                    if getattr(record, 'created_at', None) else "unknown"
                ),
            })

        return formatted

    def _get_memory_summary(self: "Kernle", memory_type: str, record: Any) -> str:
        """Get a brief summary of a memory record."""
        if memory_type == "episode":
            return record.objective[:60] if record.objective else ""
        elif memory_type == "belief":
            return record.statement[:60] if record.statement else ""
        elif memory_type == "value":
            return f"{record.name}: {record.statement[:40]}" if record.name else ""
        elif memory_type == "goal":
            return record.title[:60] if record.title else ""
        elif memory_type == "note":
            return record.content[:60] if record.content else ""
        return str(record)[:60]

    def propagate_confidence(
        self: "Kernle",
        memory_type: str,
        memory_id: str,
    ) -> Dict[str, Any]:
        """Propagate confidence changes to derived memories.

        When a source memory's confidence changes, this updates
        derived memories accordingly.

        Args:
            memory_type: Type of source memory
            memory_id: ID of source memory

        Returns:
            Result dict with number of updated memories
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}

        source_confidence = getattr(record, 'confidence', 0.8)
        source_ref = f"{memory_type}:{memory_id}"

        # Find memories derived from this one
        # This is a simplified implementation - would need to query all tables
        updated = 0

        # For now, return the source confidence info
        return {
            "source_confidence": source_confidence,
            "source_ref": source_ref,
            "updated": updated,
        }

    def set_memory_source(
        self: "Kernle",
        memory_type: str,
        memory_id: str,
        source_type: str,
        source_episodes: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
    ) -> bool:
        """Set provenance information for a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of memory
            source_type: Source type (direct_experience, inference, told_by_agent, consolidation)
            source_episodes: List of supporting episode IDs
            derived_from: List of memory refs this was derived from (format: type:id)

        Returns:
            True if updated, False if memory not found
        """
        return self._storage.update_memory_meta(
            memory_type=memory_type,
            memory_id=memory_id,
            source_type=source_type,
            source_episodes=source_episodes,
            derived_from=derived_from,
        )
