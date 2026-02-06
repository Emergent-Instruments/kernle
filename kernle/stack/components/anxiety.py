"""Anxiety tracking stack component.

Measures the functional anxiety of a synthetic intelligence across
multiple dimensions: context pressure, consolidation debt, memory
uncertainty, raw aging, identity coherence, and epoch staleness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

# Anxiety level thresholds
ANXIETY_LEVELS = {
    (0, 30): ("calm", "Calm"),
    (31, 50): ("aware", "Aware"),
    (51, 70): ("elevated", "Elevated"),
    (71, 85): ("high", "High"),
    (86, 100): ("critical", "Critical"),
}

# Dimension weights for composite score
ANXIETY_WEIGHTS = {
    "consolidation_debt": 0.30,
    "raw_aging": 0.20,
    "identity_coherence": 0.20,
    "memory_uncertainty": 0.20,
    "epoch_staleness": 0.10,
}


def _get_anxiety_level(score: int) -> tuple:
    """Get key and label for an anxiety score."""
    for (low, high), (key, label) in ANXIETY_LEVELS.items():
        if low <= score <= high:
            return key, label
    return "critical", "Critical"


class AnxietyComponent:
    """Anxiety tracking component.

    Measures memory-related anxiety across multiple dimensions and
    contributes anxiety levels to working memory context during on_load.
    """

    name = "anxiety"
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
        return None

    def on_search(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        return results

    def on_load(self, context: Dict[str, Any]) -> None:
        """Contribute anxiety levels to working memory context."""
        if self._storage is None:
            return
        report = self.get_anxiety_report()
        context["anxiety"] = {
            "overall_score": report["overall_score"],
            "overall_level": report["overall_level"],
        }

    def on_maintenance(self) -> Dict[str, Any]:
        """Report anxiety levels during maintenance."""
        if self._storage is None:
            logger.debug("AnxietyComponent: no storage, skipping maintenance")
            return {"skipped": True, "reason": "no_storage"}
        return self.get_anxiety_report()

    # ---- Core Logic ----

    def get_anxiety_report(self) -> Dict[str, Any]:
        """Calculate memory anxiety across available dimensions.

        Returns a simplified report with scores per dimension and an
        overall composite score.
        """
        if self._storage is None:
            return {"overall_score": 0, "overall_level": "Calm", "dimensions": {}}

        dimensions: Dict[str, Dict[str, Any]] = {}

        # Consolidation debt
        episodes = self._storage.get_episodes(limit=100)
        unreflected = [
            e for e in episodes if (not e.tags or "checkpoint" not in e.tags) and not e.lessons
        ]
        unreflected_count = len(unreflected)
        if unreflected_count <= 3:
            consolidation_score = unreflected_count * 7
        elif unreflected_count <= 7:
            consolidation_score = int(21 + (unreflected_count - 3) * 10)
        elif unreflected_count <= 15:
            consolidation_score = int(61 + (unreflected_count - 7) * 4)
        else:
            consolidation_score = min(100, int(93 + (unreflected_count - 15) * 0.5))
        dimensions["consolidation_debt"] = {
            "score": min(100, consolidation_score),
            "detail": f"{unreflected_count} unreflected episodes",
        }

        # Memory uncertainty
        beliefs = self._storage.get_beliefs(limit=100)
        low_conf = [b for b in beliefs if b.confidence < 0.5]
        if len(low_conf) <= 2:
            uncertainty_score = len(low_conf) * 15
        elif len(low_conf) <= 5:
            uncertainty_score = int(30 + (len(low_conf) - 2) * 15)
        else:
            uncertainty_score = min(100, int(75 + (len(low_conf) - 5) * 5))
        dimensions["memory_uncertainty"] = {
            "score": min(100, uncertainty_score),
            "detail": f"{len(low_conf)} low-confidence beliefs",
        }

        # Identity coherence (inverted: high coherence = low anxiety)
        values = self._storage.get_values(limit=10)
        total_conf = 0.0
        count = 0
        for v in values:
            total_conf += getattr(v, "confidence", 0.8)
            count += 1
        for b in beliefs[:20]:
            total_conf += b.confidence
            count += 1
        identity_confidence = total_conf / count if count > 0 else 0.0
        identity_anxiety = int((1.0 - identity_confidence) * 100)
        dimensions["identity_coherence"] = {
            "score": identity_anxiety,
            "detail": f"{identity_confidence:.0%} identity confidence",
        }

        # Raw aging
        raw_aging_score = 0
        dimensions["raw_aging"] = {
            "score": raw_aging_score,
            "detail": "Raw aging check",
        }

        # Epoch staleness
        epoch_staleness_score = 0
        try:
            current_epoch = self._storage.get_current_epoch()
            if current_epoch and current_epoch.started_at:
                started = current_epoch.started_at
                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                months = (now - started).total_seconds() / (30.44 * 86400)
                if months < 6:
                    epoch_staleness_score = int(months * 5)
                elif months < 12:
                    epoch_staleness_score = int(30 + (months - 6) * 6.7)
                else:
                    epoch_staleness_score = min(100, int(70 + (months - 12) * 4))
        except Exception:
            pass
        dimensions["epoch_staleness"] = {
            "score": min(100, epoch_staleness_score),
            "detail": "Epoch staleness",
        }

        # Composite score
        overall_score = 0
        for dim_name, weight in ANXIETY_WEIGHTS.items():
            overall_score += dimensions[dim_name]["score"] * weight
        overall_score = int(overall_score)

        _, overall_level = _get_anxiety_level(overall_score)

        return {
            "overall_score": overall_score,
            "overall_level": overall_level,
            "dimensions": dimensions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
