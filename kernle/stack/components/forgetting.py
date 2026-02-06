"""Forgetting stack component.

Provides salience-based forgetting: decay calculation, candidate identification,
and forgetting sweeps during maintenance. Preserves protected memories.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Default half-life for salience decay (in days)
DEFAULT_HALF_LIFE = 30.0

# Half-life overrides by goal_type
GOAL_TYPE_HALF_LIVES = {
    "aspiration": 180.0,
    "commitment": 365.0,
    "task": 30.0,
    "exploration": 30.0,
}


class ForgettingComponent:
    """Salience-based forgetting component.

    Calculates memory salience and tombstones low-salience memories
    during maintenance sweeps. Protected memories are never forgotten.
    """

    name = "forgetting"
    version = "1.0.0"
    required = False
    needs_inference = False

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
        return None  # Forgetting doesn't act on save

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results  # No search modification

    def on_load(self, context: Dict[str, Any]) -> None:
        pass  # No working memory contribution

    def on_maintenance(self) -> Dict[str, Any]:
        """Run forgetting sweep: decay salience, forget low-salience memories."""
        if self._storage is None:
            logger.debug("ForgettingComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}

        candidates = self._get_forgetting_candidates(threshold=0.3, limit=10)
        forgotten = 0
        protected = 0

        for candidate in candidates:
            success = self._storage.forget_memory(
                candidate["type"],
                candidate["id"],
                f"Low salience ({candidate['salience']:.4f}) in forgetting cycle",
            )
            if success:
                forgotten += 1
            else:
                protected += 1

        return {
            "candidates_found": len(candidates),
            "forgotten": forgotten,
            "protected": protected,
        }

    # ---- Core Logic ----

    def calculate_salience(self, memory_type: str, memory_id: str) -> float:
        """Calculate current salience score for a memory.

        Returns -1.0 if memory not found or no storage available.
        """
        if self._storage is None:
            return -1.0

        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return -1.0

        confidence = getattr(record, "confidence", 0.8)
        times_accessed = getattr(record, "times_accessed", 0) or 0
        last_accessed = getattr(record, "last_accessed", None)
        created_at = getattr(record, "created_at", None)

        reference_time = last_accessed or created_at
        now = datetime.now(timezone.utc)
        if reference_time:
            days_since = (now - reference_time).total_seconds() / 86400
        else:
            days_since = 365

        half_life = self._get_half_life(memory_type, record)
        age_factor = days_since / half_life
        reinforcement_weight = math.log(times_accessed + 1)

        salience = (confidence * (reinforcement_weight + 0.1)) / (age_factor + 1)
        return salience

    def _get_half_life(self, memory_type: str, record: Any) -> float:
        """Get the appropriate half-life for a record."""
        if memory_type == "goal":
            goal_type = getattr(record, "goal_type", "task")
            return GOAL_TYPE_HALF_LIVES.get(goal_type, DEFAULT_HALF_LIFE)
        return DEFAULT_HALF_LIFE

    def _get_forgetting_candidates(
        self, threshold: float = 0.3, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find low-salience memories eligible for forgetting."""
        if self._storage is None:
            return []

        results = self._storage.get_forgetting_candidates(
            memory_types=None,
            limit=limit * 2,
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
                    }
                )

        return candidates[:limit]
