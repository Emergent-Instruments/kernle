"""Shared anxiety scoring functions.

Pure functions for computing anxiety dimension scores, used by both
AnxietyMixin (7-dim, Kernle layer) and AnxietyComponent (5-dim, stack layer).
"""

from typing import Optional, Tuple

# ---- Anxiety Level Thresholds ----

ANXIETY_LEVELS = {
    (0, 30): ("calm", "Calm"),
    (31, 50): ("aware", "Aware"),
    (51, 70): ("elevated", "Elevated"),
    (71, 85): ("high", "High"),
    (86, 100): ("critical", "Critical"),
}


def get_anxiety_level(score: int) -> Tuple[str, str]:
    """Get (key, label) for an anxiety score."""
    for (low, high), (key, label) in ANXIETY_LEVELS.items():
        if low <= score <= high:
            return key, label
    return "critical", "Critical"


# ---- Dimension Weights ----

# 7-dimension weights (Kernle layer — includes context_pressure + unsaved_work)
SEVEN_DIM_WEIGHTS = {
    "context_pressure": 0.25,
    "unsaved_work": 0.20,
    "consolidation_debt": 0.15,
    "raw_aging": 0.10,
    "identity_coherence": 0.10,
    "memory_uncertainty": 0.10,
    "epoch_staleness": 0.10,
}

# 5-dimension weights (stack layer — renormalized from 7-dim shared subset)
_shared_keys = (
    "consolidation_debt",
    "raw_aging",
    "identity_coherence",
    "memory_uncertainty",
    "epoch_staleness",
)
_shared_total = sum(SEVEN_DIM_WEIGHTS[k] for k in _shared_keys)
FIVE_DIM_WEIGHTS = {k: SEVEN_DIM_WEIGHTS[k] / _shared_total for k in _shared_keys}


# ---- Scoring Functions ----


def compute_consolidation_score(unreflected_count: int) -> int:
    """Score for unreflected episodes (0-100)."""
    if unreflected_count <= 3:
        return unreflected_count * 7
    elif unreflected_count <= 7:
        return int(21 + (unreflected_count - 3) * 10)
    elif unreflected_count <= 15:
        return int(61 + (unreflected_count - 7) * 4)
    else:
        return min(100, int(93 + (unreflected_count - 15) * 0.5))


def compute_identity_coherence_score(identity_confidence: float) -> int:
    """Score for identity coherence (inverted: high confidence = low anxiety)."""
    return int((1.0 - identity_confidence) * 100)


def compute_memory_uncertainty_score(low_conf_count: int) -> int:
    """Score for low-confidence beliefs (0-100)."""
    if low_conf_count <= 2:
        return low_conf_count * 15
    elif low_conf_count <= 5:
        return int(30 + (low_conf_count - 2) * 15)
    else:
        return min(100, int(75 + (low_conf_count - 5) * 5))


def compute_raw_aging_score(total_unprocessed: int, aging_count: int, oldest_hours: float) -> int:
    """Score for aging raw entries (0-100).

    Args:
        total_unprocessed: Total number of unprocessed raw entries.
        aging_count: Number of entries older than the age threshold.
        oldest_hours: Age of the oldest entry in hours.
    """
    if total_unprocessed == 0:
        return 0
    elif aging_count == 0:
        return min(30, total_unprocessed * 3)
    elif aging_count <= 3:
        return int(30 + aging_count * 15)
    elif aging_count <= 7:
        return int(60 + (aging_count - 3) * 8)
    else:
        return min(100, int(92 + (aging_count - 7) * 1))


def compute_epoch_staleness_score(months: Optional[float]) -> int:
    """Score for epoch staleness (0-100)."""
    if months is None:
        return 0
    elif months < 6:
        return int(months * 5)
    elif months < 12:
        return int(30 + (months - 6) * 6.7)
    elif months < 18:
        return int(70 + (months - 12) * 4)
    else:
        return min(100, int(90 + (months - 18) * 1.67))


def compute_context_pressure_score(context_pressure_pct: int) -> int:
    """Score for context pressure (0-100). Kernle-only dimension."""
    if context_pressure_pct < 50:
        return int(context_pressure_pct * 0.6)
    elif context_pressure_pct < 70:
        return int(30 + (context_pressure_pct - 50) * 1.5)
    elif context_pressure_pct < 85:
        return int(60 + (context_pressure_pct - 70) * 2)
    else:
        return int(90 + (context_pressure_pct - 85) * 0.67)


def compute_unsaved_work_score(checkpoint_age_minutes: Optional[int]) -> int:
    """Score for unsaved work (0-100). Kernle-only dimension."""
    if checkpoint_age_minutes is None:
        return 50
    elif checkpoint_age_minutes < 15:
        return int(checkpoint_age_minutes * 2)
    elif checkpoint_age_minutes < 60:
        return int(30 + (checkpoint_age_minutes - 15) * 1.1)
    else:
        return min(100, int(80 + (checkpoint_age_minutes - 60) * 0.33))
