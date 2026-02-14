"""Structural health checks for kernle memory stacks.

Provides importable API for programmatic health inspection:
- StructuralFinding dataclass
- Individual check functions (orphaned refs, low confidence, stale rels, contradictions, stale goals)
- run_structural_checks() coordinator
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


@dataclass
class StructuralFinding:
    """A single structural finding from the health check."""

    check: str
    severity: str  # "error", "warning", "info"
    memory_type: str
    memory_id: str
    message: str

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "memory_type": self.memory_type,
            "memory_id": self.memory_id,
            "message": self.message,
        }


def check_orphaned_references(k: "Kernle") -> List[StructuralFinding]:
    """Check for orphaned derived_from and source_episodes references."""
    findings: List[StructuralFinding] = []
    result = k.find_orphaned_references(memory_types=["episode", "belief", "note"])
    for orphan in result.get("orphans", []):
        mem_ref = orphan["memory"]
        parts = mem_ref.split(":", 1)
        mem_type = parts[0] if len(parts) == 2 else "unknown"
        mem_id = parts[1] if len(parts) == 2 else mem_ref
        findings.append(
            StructuralFinding(
                check="orphaned_reference",
                severity="error",
                memory_type=mem_type,
                memory_id=mem_id,
                message=(
                    f"{mem_type.capitalize()} #{mem_id[:12]} has broken "
                    f"{orphan['field']} ref -> {orphan['broken_ref']}"
                ),
            )
        )
    return findings


def check_low_confidence_beliefs(k: "Kernle", threshold: float = 0.3) -> List[StructuralFinding]:
    """Check for beliefs with low confidence that haven't been verified recently."""
    findings: List[StructuralFinding] = []
    now = datetime.now(timezone.utc)
    try:
        beliefs = k._storage.get_beliefs(limit=500, include_inactive=False)
    except Exception as e:
        logger.debug(f"Failed to get beliefs for structural check: {e}", exc_info=True)
        return findings
    for belief in beliefs:
        if belief.confidence < threshold:
            last_verified = belief.last_verified
            verified_str = "never verified"
            if last_verified:
                days_since = (now - last_verified).days
                verified_str = f"last verified {days_since}d ago"
            findings.append(
                StructuralFinding(
                    check="low_confidence_belief",
                    severity="warning",
                    memory_type="belief",
                    memory_id=belief.id,
                    message=(
                        f"Belief #{belief.id[:12]} (confidence {belief.confidence:.2f}) "
                        f"-- low confidence, {verified_str}"
                    ),
                )
            )
    return findings


def check_stale_relationships(k: "Kernle", stale_days: int = 90) -> List[StructuralFinding]:
    """Check for relationships with no interactions or stale last_interaction."""
    findings: List[StructuralFinding] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=stale_days)
    try:
        relationships = k._storage.get_relationships()
    except Exception as e:
        logger.debug(f"Failed to get relationships for structural check: {e}", exc_info=True)
        return findings
    for rel in relationships:
        if rel.interaction_count == 0:
            findings.append(
                StructuralFinding(
                    check="stale_relationship",
                    severity="warning",
                    memory_type="relationship",
                    memory_id=rel.id,
                    message=(
                        f"Relationship #{rel.id[:12]} ({rel.entity_name}) " f"-- zero interactions"
                    ),
                )
            )
        elif rel.last_interaction and rel.last_interaction < cutoff:
            days_ago = (now - rel.last_interaction).days
            findings.append(
                StructuralFinding(
                    check="stale_relationship",
                    severity="info",
                    memory_type="relationship",
                    memory_id=rel.id,
                    message=(
                        f"Relationship #{rel.id[:12]} ({rel.entity_name}) "
                        f"-- last interaction {days_ago}d ago"
                    ),
                )
            )
    return findings


_STOP_WORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "it",
    "its",
    "i",
    "me",
    "my",
    "that",
    "this",
    "these",
    "those",
    "and",
    "or",
    "but",
    "if",
    "as",
    "so",
    "than",
    "when",
    "while",
    "about",
    "into",
    "through",
    "after",
}


def check_belief_contradictions(k: "Kernle") -> List[StructuralFinding]:
    """Detect active beliefs that may contradict each other."""
    findings: List[StructuralFinding] = []
    try:
        beliefs = k._storage.get_beliefs(limit=200, include_inactive=False)
    except Exception as e:
        logger.debug(f"Failed to get beliefs for contradiction check: {e}", exc_info=True)
        return findings

    negation_words = {
        "never",
        "not",
        "no",
        "don't",
        "doesn't",
        "shouldn't",
        "cannot",
        "won't",
        "avoid",
        "reject",
        "false",
        "wrong",
        "bad",
        "dislike",
        "hate",
        "impossible",
        "refuse",
    }
    affirmation_words = {
        "always",
        "should",
        "must",
        "can",
        "will",
        "prefer",
        "accept",
        "true",
        "right",
        "good",
        "like",
        "love",
        "possible",
        "embrace",
    }

    seen_pairs: set = set()
    for i, b1 in enumerate(beliefs):
        words1 = set(b1.statement.lower().split())
        has_neg1 = bool(words1 & negation_words)
        has_aff1 = bool(words1 & affirmation_words)
        content_words1 = words1 - negation_words - affirmation_words - _STOP_WORDS

        for b2 in beliefs[i + 1 :]:
            pair_key = tuple(sorted([b1.id, b2.id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            words2 = set(b2.statement.lower().split())
            has_neg2 = bool(words2 & negation_words)
            has_aff2 = bool(words2 & affirmation_words)
            content_words2 = words2 - negation_words - affirmation_words - _STOP_WORDS

            if not content_words1 or not content_words2:
                continue
            overlap = content_words1 & content_words2
            overlap_ratio = len(overlap) / min(len(content_words1), len(content_words2))
            if overlap_ratio < 0.3:
                continue

            polarity_mismatch = (has_neg1 and has_aff2) or (has_aff1 and has_neg2)
            if not polarity_mismatch:
                continue

            findings.append(
                StructuralFinding(
                    check="belief_contradiction",
                    severity="warning",
                    memory_type="belief",
                    memory_id=b1.id,
                    message=(
                        f"Potential contradiction: Belief #{b1.id[:12]} "
                        f"vs #{b2.id[:12]} (overlap: {overlap_ratio:.0%})"
                    ),
                )
            )
    return findings


def check_stale_goals(k: "Kernle", stale_days: int = 60) -> List[StructuralFinding]:
    """Check for active goals that are old with no recent progress."""
    findings: List[StructuralFinding] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=stale_days)
    try:
        goals = k._storage.get_goals(status="active", limit=200)
    except Exception as e:
        logger.debug(f"Failed to get goals for structural check: {e}", exc_info=True)
        return findings
    for goal in goals:
        if not goal.created_at:
            continue
        if goal.created_at >= cutoff:
            continue
        days_old = (now - goal.created_at).days
        findings.append(
            StructuralFinding(
                check="stale_goal",
                severity="info",
                memory_type="goal",
                memory_id=goal.id,
                message=(
                    f"Goal #{goal.id[:12]} is active but {days_old}d old "
                    f"-- consider reviewing status"
                ),
            )
        )
    return findings


def run_structural_checks(k: "Kernle") -> List[StructuralFinding]:
    """Run all structural health checks and return combined findings."""
    findings: List[StructuralFinding] = []
    findings.extend(check_orphaned_references(k))
    findings.extend(check_low_confidence_beliefs(k))
    findings.extend(check_stale_relationships(k))
    findings.extend(check_belief_contradictions(k))
    findings.extend(check_stale_goals(k))
    return findings
