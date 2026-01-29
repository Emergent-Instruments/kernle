"""Controlled forgetting mixin for Kernle.

This module provides salience-based forgetting, enabling memories to
gracefully fade while preserving core identity memories.
"""

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from kernle.core import Kernle


class ForgettingMixin:
    """Mixin providing controlled forgetting capabilities.

    Enables:
    - Salience calculation for memories
    - Tombstoning (soft delete) of low-salience memories
    - Memory protection for core identity
    - Recovery of forgotten memories
    """

    # Default half-life for salience decay (in days)
    DEFAULT_HALF_LIFE = 30.0

    def calculate_salience(self: "Kernle", memory_type: str, memory_id: str) -> float:
        """Calculate current salience score for a memory.

        Salience formula:
        salience = (confidence × reinforcement_weight) / (age_factor + 1)
        where:
            reinforcement_weight = log(times_accessed + 1)
            age_factor = days_since_last_access / half_life

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note, drive, relationship)
            memory_id: ID of the memory

        Returns:
            Salience score (0.0-1.0 typical range, can exceed 1.0 for very active memories)
            Returns -1.0 if memory not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return -1.0

        confidence = getattr(record, "confidence", 0.8)
        times_accessed = getattr(record, "times_accessed", 0) or 0
        last_accessed = getattr(record, "last_accessed", None)
        created_at = getattr(record, "created_at", None)

        # Use last_accessed if available, otherwise created_at
        reference_time = last_accessed or created_at

        now = datetime.now(timezone.utc)
        if reference_time:
            days_since = (now - reference_time).total_seconds() / 86400
        else:
            days_since = 365  # Very old if unknown

        # Guard against zero half-life (would cause division by zero)
        half_life = max(0.001, self.DEFAULT_HALF_LIFE)
        age_factor = days_since / half_life
        reinforcement_weight = math.log(times_accessed + 1)

        # Salience calculation with minimum base value
        salience = (confidence * (reinforcement_weight + 0.1)) / (age_factor + 1)

        return salience

    def get_forgetting_candidates(
        self: "Kernle",
        threshold: float = 0.3,
        limit: int = 20,
        memory_types: Optional[List[str]] = None,
    ) -> List[dict]:
        """Find low-salience memories eligible for forgetting.

        Returns memories that are:
        - Not protected (is_protected = False)
        - Not already forgotten (is_forgotten = False)
        - Have salience below the threshold

        Args:
            threshold: Salience threshold (memories below this are candidates)
            limit: Maximum candidates to return
            memory_types: Filter by memory type (default: episode, belief, goal, note, relationship)

        Returns:
            List of dicts with memory info and salience score, sorted by salience (lowest first)
        """
        results = self._storage.get_forgetting_candidates(
            memory_types=memory_types,
            limit=limit * 2,  # Get more to filter by threshold
        )

        candidates = []
        for r in results:
            if r.score < threshold:
                record = r.record
                candidates.append(
                    {
                        "type": r.record_type,
                        "id": record.id,
                        "salience": round(r.score, 4),
                        "summary": self._get_memory_summary(r.record_type, record),
                        "confidence": getattr(record, "confidence", 0.8),
                        "times_accessed": getattr(record, "times_accessed", 0),
                        "last_accessed": (
                            getattr(record, "last_accessed").isoformat()
                            if getattr(record, "last_accessed", None)
                            else None
                        ),
                        "created_at": (
                            getattr(record, "created_at").strftime("%Y-%m-%d")
                            if getattr(record, "created_at", None)
                            else "unknown"
                        ),
                    }
                )

        return candidates[:limit]

    def forget(
        self: "Kernle",
        memory_type: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Tombstone a memory (mark forgotten, don't delete).

        Forgotten memories are not deleted - they can be recovered later.
        Protected memories cannot be forgotten.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            reason: Optional reason for forgetting (for audit trail)

        Returns:
            True if forgotten, False if not found, already forgotten, or protected
        """
        return self._storage.forget_memory(memory_type, memory_id, reason)

    def recover(self: "Kernle", memory_type: str, memory_id: str) -> bool:
        """Recover a forgotten memory.

        Restores a tombstoned memory back to active status.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if recovered, False if not found or not forgotten
        """
        return self._storage.recover_memory(memory_type, memory_id)

    def protect(self: "Kernle", memory_type: str, memory_id: str, protected: bool = True) -> bool:
        """Mark memory as protected from forgetting.

        Protected memories never decay and cannot be forgotten.
        Use this for core identity memories.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory
            protected: True to protect, False to unprotect

        Returns:
            True if updated, False if memory not found
        """
        return self._storage.protect_memory(memory_type, memory_id, protected)

    def run_forgetting_cycle(
        self: "Kernle",
        threshold: float = 0.3,
        limit: int = 10,
        dry_run: bool = True,
    ) -> dict:
        """Review and optionally forget low-salience memories.

        This is the main forgetting maintenance function. It:
        1. Finds memories below the salience threshold
        2. Optionally forgets them (if dry_run=False)
        3. Returns a report of what was/would be forgotten

        Args:
            threshold: Salience threshold (memories below this are candidates)
            limit: Maximum memories to forget in one cycle
            dry_run: If True, only report what would be forgotten (don't actually forget)

        Returns:
            Report dict with:
            - candidates: List of forgetting candidates
            - forgotten: Number actually forgotten (0 if dry_run)
            - protected: Number skipped because protected
            - dry_run: Whether this was a dry run
        """
        candidates = self.get_forgetting_candidates(threshold=threshold, limit=limit)

        report = {
            "threshold": threshold,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "forgotten": 0,
            "protected": 0,
            "dry_run": dry_run,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not dry_run:
            for candidate in candidates:
                success = self.forget(
                    memory_type=candidate["type"],
                    memory_id=candidate["id"],
                    reason=f"Low salience ({candidate['salience']:.4f}) in forgetting cycle",
                )
                if success:
                    report["forgotten"] += 1
                else:
                    # Likely protected or already forgotten
                    report["protected"] += 1

        return report

    def get_forgotten_memories(
        self: "Kernle",
        memory_types: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Get all forgotten (tombstoned) memories.

        These can be recovered using the recover() method.

        Args:
            memory_types: Filter by memory type
            limit: Maximum results

        Returns:
            List of forgotten memory info dicts
        """
        results = self._storage.get_forgotten_memories(
            memory_types=memory_types,
            limit=limit,
        )

        forgotten = []
        for r in results:
            record = r.record
            forgotten.append(
                {
                    "type": r.record_type,
                    "id": record.id,
                    "summary": self._get_memory_summary(r.record_type, record),
                    "forgotten_at": (
                        getattr(record, "forgotten_at").isoformat()
                        if getattr(record, "forgotten_at", None)
                        else None
                    ),
                    "forgotten_reason": getattr(record, "forgotten_reason", None),
                    "created_at": (
                        getattr(record, "created_at").strftime("%Y-%m-%d")
                        if getattr(record, "created_at", None)
                        else "unknown"
                    ),
                }
            )

        return forgotten

    def record_access(self: "Kernle", memory_type: str, memory_id: str) -> bool:
        """Record that a memory was accessed (for salience tracking).

        Call this when retrieving a memory to update its access statistics.
        This helps the salience calculation favor frequently-accessed memories.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            True if updated, False if memory not found
        """
        return self._storage.record_access(memory_type, memory_id)
