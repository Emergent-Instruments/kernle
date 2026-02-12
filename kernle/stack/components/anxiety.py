"""Anxiety tracking stack component.

Measures the functional anxiety of a synthetic intelligence across
multiple dimensions: context pressure, consolidation debt, memory
uncertainty, raw aging, identity coherence, and epoch staleness.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kernle.anxiety_core import (
    FIVE_DIM_WEIGHTS,
    compute_consolidation_score,
    compute_epoch_staleness_score,
    compute_identity_coherence_score,
    compute_memory_uncertainty_score,
)
from kernle.anxiety_core import get_anxiety_level as _get_anxiety_level
from kernle.protocols import InferenceService, SearchResult

logger = logging.getLogger(__name__)

ANXIETY_WEIGHTS = FIVE_DIM_WEIGHTS


class AnxietyComponent:
    """Anxiety tracking component.

    Measures memory-related anxiety across multiple dimensions and
    contributes anxiety levels to working memory context during on_load.
    """

    name = "anxiety"
    version = "1.0.0"
    required = False
    needs_inference = False
    inference_scope = "none"
    priority = 250

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
        consolidation_score = compute_consolidation_score(unreflected_count)
        dimensions["consolidation_debt"] = {
            "score": min(100, consolidation_score),
            "detail": f"{unreflected_count} unreflected episodes",
        }

        # Memory uncertainty
        beliefs = self._storage.get_beliefs(limit=100)
        low_conf = [b for b in beliefs if b.confidence < 0.5]
        uncertainty_score = compute_memory_uncertainty_score(len(low_conf))
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
        identity_anxiety = compute_identity_coherence_score(identity_confidence)
        dimensions["identity_coherence"] = {
            "score": identity_anxiety,
            "detail": f"{identity_confidence:.0%} identity confidence",
        }

        # Raw aging
        raw_aging_score = 0
        raw_aging_detail = "No unprocessed raw entries"
        try:
            if hasattr(self._storage, "list_raw"):
                from kernle.anxiety_core import compute_raw_aging_score

                raw_entries = self._storage.list_raw(processed=False, limit=100)
                now = datetime.now(timezone.utc)
                total_unprocessed = len(raw_entries)
                aging_count = 0
                oldest_hours = 0.0
                for entry in raw_entries:
                    try:
                        entry_time = getattr(entry, "captured_at", None) or getattr(
                            entry, "timestamp", None
                        )
                        if entry_time:
                            if isinstance(entry_time, str):
                                entry_time = datetime.fromisoformat(
                                    entry_time.replace("Z", "+00:00")
                                )
                            age = now - entry_time
                            entry_hours = age.total_seconds() / 3600
                            if entry_hours > 24:
                                aging_count += 1
                            if entry_hours > oldest_hours:
                                oldest_hours = entry_hours
                    except (ValueError, TypeError, AttributeError):
                        continue
                raw_aging_score = compute_raw_aging_score(
                    total_unprocessed, aging_count, oldest_hours
                )
                if total_unprocessed == 0:
                    raw_aging_detail = "No unprocessed raw entries"
                elif aging_count == 0:
                    raw_aging_detail = f"{total_unprocessed} unprocessed (all fresh)"
                else:
                    raw_aging_detail = f"{aging_count}/{total_unprocessed} entries >24h old"
        except Exception:
            pass
        dimensions["raw_aging"] = {
            "score": raw_aging_score,
            "detail": raw_aging_detail,
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
                epoch_staleness_score = compute_epoch_staleness_score(months)
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
