"""Trust layer mixin for Kernle.

Provides trust assessment management, dynamic trust computation,
trust decay, and transitive trust chains (KEP v3).
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from kernle.storage.base import (
    DEFAULT_TRUST,
    SEED_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_DEPTH_DECAY,
    TRUST_THRESHOLDS,
    TrustAssessment,
)

if TYPE_CHECKING:
    from kernle.core import Kernle


class TrustMixin:
    """Mixin providing trust assessment and dynamic trust computation."""

    # === Trust Layer (KEP v3) ===

    def seed_trust(self: "Kernle") -> int:
        """Apply seed trust templates. Returns number of assessments created."""
        created = 0
        for seed in SEED_TRUST:
            existing = self._storage.get_trust_assessment(seed["entity"])
            if existing is None:
                assessment = TrustAssessment(
                    id=str(uuid.uuid4()),
                    stack_id=self.stack_id,
                    entity=seed["entity"],
                    dimensions=seed["dimensions"],
                    authority=seed.get("authority", []),
                )
                self._storage.save_trust_assessment(assessment)
                created += 1
        return created

    def trust_list(self: "Kernle") -> List[Dict[str, Any]]:
        """List all trust assessments."""
        return [
            {
                "entity": a.entity,
                "dimensions": a.dimensions,
                "authority": a.authority or [],
                "evidence_count": len(a.evidence_episode_ids or []),
                "last_updated": a.last_updated.isoformat() if a.last_updated else None,
            }
            for a in self._storage.get_trust_assessments()
        ]

    def trust_show(self: "Kernle", entity: str) -> Optional[Dict[str, Any]]:
        """Show detailed trust assessment for an entity."""
        a = self._storage.get_trust_assessment(entity)
        if a is None:
            return None
        return {
            "id": a.id,
            "entity": a.entity,
            "dimensions": a.dimensions,
            "authority": a.authority or [],
            "evidence_episode_ids": a.evidence_episode_ids or [],
            "last_updated": a.last_updated.isoformat() if a.last_updated else None,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }

    def trust_set(
        self: "Kernle",
        entity: str,
        domain: str = "general",
        score: float = 0.5,
        authority: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Set or update trust for an entity in a specific domain."""
        score = max(0.0, min(1.0, score))
        existing = self._storage.get_trust_assessment(entity)
        if existing:
            existing.dimensions[domain] = {"score": score}
            if authority is not None:
                existing.authority = authority
            return self._storage.save_trust_assessment(existing)
        else:
            return self._storage.save_trust_assessment(
                TrustAssessment(
                    id=str(uuid.uuid4()),
                    stack_id=self.stack_id,
                    entity=entity,
                    dimensions={domain: {"score": score}},
                    authority=authority or [],
                )
            )

    def gate_memory_input(
        self: "Kernle",
        source_entity: str,
        action: str,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check whether a source entity has sufficient trust for an action.

        This is advisory -- the agent retains sovereignty over final decisions.
        """
        threshold = TRUST_THRESHOLDS.get(action)
        if threshold is None:
            return {
                "allowed": False,
                "trust_level": 0.0,
                "domain": "unknown",
                "reason": f"Unknown action type: {action}",
            }

        assessment = self._storage.get_trust_assessment(source_entity)
        if assessment is None:
            return {
                "allowed": False,
                "trust_level": 0.0,
                "domain": target or "general",
                "reason": f"No trust assessment for entity: {source_entity}",
            }

        authority = assessment.authority or []
        has_all_authority = any(a.get("scope") == "all" for a in authority)

        domain = target or "general"
        dims = assessment.dimensions or {}
        domain_trust = dims.get(domain, dims.get("general", {}))
        trust_score = domain_trust.get("score", 0.0) if isinstance(domain_trust, dict) else 0.0

        allowed = trust_score >= threshold or has_all_authority
        if allowed:
            reason = f"Trust {trust_score:.2f} >= threshold {threshold:.2f} for {action}"
        else:
            reason = f"Trust {trust_score:.2f} < threshold {threshold:.2f} for {action}"

        return {
            "allowed": allowed,
            "trust_level": trust_score,
            "domain": domain,
            "reason": reason,
        }

    def _build_trust_summary(self: "Kernle") -> Optional[Dict[str, Any]]:
        """Build trust summary for load() output."""
        assessments = self._storage.get_trust_assessments()
        if not assessments:
            return None
        summary = {}
        for a in assessments:
            general = a.dimensions.get("general", {})
            score = general.get("score", 0.0) if isinstance(general, dict) else 0.0
            summary[a.entity] = {
                "trust": round(score, 2),
                "authority": [auth.get("scope", "unknown") for auth in (a.authority or [])],
            }
        return summary

    # === Dynamic Trust (KEP v3 section 8.6-8.7) ===

    def compute_direct_trust(
        self: "Kernle", entity: str, domain: str = "general"
    ) -> Dict[str, Any]:
        """Compute trust score from episode history with recency weighting.

        Looks at episodes with the given source_entity, classifies outcomes
        as positive/negative, and computes a recency-weighted score.
        """
        episodes = self._storage.get_episodes_by_source_entity(entity)
        if not episodes:
            return {
                "entity": entity,
                "domain": domain,
                "score": DEFAULT_TRUST,
                "positive": 0,
                "negative": 0,
                "total": 0,
                "source": "default",
            }

        now = datetime.now(timezone.utc)
        positive_weight = 0.0
        negative_weight = 0.0

        for ep in episodes:
            # Recency weight: exponential decay, halving every 30 days
            if ep.created_at:
                days_ago = max(0, (now - ep.created_at).total_seconds() / 86400)
            else:
                days_ago = 30.0
            recency = 0.5 ** (days_ago / 30.0)

            # Classify outcome: positive if outcome_type is "success" or
            # emotional_valence > 0, negative otherwise
            is_positive = False
            if ep.outcome_type in ("success", "positive"):
                is_positive = True
            elif ep.outcome_type in ("failure", "negative"):
                is_positive = False
            elif ep.emotional_valence > 0:
                is_positive = True

            if is_positive:
                positive_weight += recency
            else:
                negative_weight += recency

        total_weight = positive_weight + negative_weight
        if total_weight == 0:
            score = DEFAULT_TRUST
        else:
            score = positive_weight / total_weight

        return {
            "entity": entity,
            "domain": domain,
            "score": round(score, 4),
            "positive": round(positive_weight, 4),
            "negative": round(negative_weight, 4),
            "total": len(episodes),
            "source": "computed",
        }

    def apply_trust_decay(
        self: "Kernle", entity: str, days_since_interaction: float
    ) -> Dict[str, Any]:
        """Apply trust decay toward neutral (0.5) without reinforcement.

        Formula: current + (0.5 - current) * min(decay_factor, 1.0)
        where decay_factor = TRUST_DECAY_RATE * days_since_interaction
        """
        assessment = self._storage.get_trust_assessment(entity)
        if assessment is None:
            return {
                "entity": entity,
                "error": f"No trust assessment for entity: {entity}",
            }

        decay_factor = min(TRUST_DECAY_RATE * days_since_interaction, 1.0)
        updated_dims = {}

        for domain, dim_data in assessment.dimensions.items():
            if not isinstance(dim_data, dict):
                updated_dims[domain] = dim_data
                continue
            current = dim_data.get("score", DEFAULT_TRUST)
            # Self-trust has a floor
            if entity == "self":
                floor = SELF_TRUST_FLOOR
                decayed = current + (floor - current) * decay_factor
                decayed = max(floor, decayed)
            else:
                decayed = current + (DEFAULT_TRUST - current) * decay_factor
            updated_dims[domain] = {"score": round(decayed, 4)}

        assessment.dimensions = updated_dims
        self._storage.save_trust_assessment(assessment)

        return {
            "entity": entity,
            "days": days_since_interaction,
            "decay_factor": round(decay_factor, 4),
            "dimensions": updated_dims,
        }

    def compute_transitive_trust(
        self: "Kernle", target: str, chain: List[str], domain: str = "general"
    ) -> Dict[str, Any]:
        """Compute transitive trust through a chain of entities.

        Trust flows through entity chains with 15% decay per hop:
        trust = product of (direct_trust * depth_decay^i) for each entity in chain.
        """
        if not chain:
            return {
                "target": target,
                "chain": [],
                "domain": domain,
                "score": 0.0,
                "hops": [],
                "error": "Empty chain",
            }

        trust = 1.0
        hops = []

        for i, entity in enumerate(chain):
            assessment = self._storage.get_trust_assessment(entity)
            if assessment is None:
                direct = DEFAULT_TRUST
            else:
                dims = assessment.dimensions or {}
                domain_data = dims.get(domain, dims.get("general", {}))
                direct = (
                    domain_data.get("score", DEFAULT_TRUST)
                    if isinstance(domain_data, dict)
                    else DEFAULT_TRUST
                )

            hop_factor = direct * (TRUST_DEPTH_DECAY**i)
            trust *= hop_factor
            hops.append(
                {
                    "entity": entity,
                    "direct_trust": round(direct, 4),
                    "depth_decay": round(TRUST_DEPTH_DECAY**i, 4),
                    "cumulative": round(trust, 4),
                }
            )

        return {
            "target": target,
            "chain": chain,
            "domain": domain,
            "score": round(trust, 4),
            "hops": hops,
        }

    def compute_self_trust_floor(self: "Kernle") -> Dict[str, Any]:
        """Compute self-trust floor from historical accuracy.

        self_trust_floor = max(0.5, historical_accuracy_rate)
        where accuracy is based on episodes where source_entity is 'self'.
        """
        episodes = self._storage.get_episodes_by_source_entity("self")
        if not episodes:
            return {
                "floor": SELF_TRUST_FLOOR,
                "accuracy": None,
                "total_episodes": 0,
                "source": "default",
            }

        positive = sum(
            1
            for e in episodes
            if e.outcome_type in ("success", "positive") or e.emotional_valence > 0
        )
        total = len(episodes)
        accuracy = positive / total if total > 0 else 0.5
        floor = max(SELF_TRUST_FLOOR, accuracy)

        return {
            "floor": round(floor, 4),
            "accuracy": round(accuracy, 4),
            "total_episodes": total,
            "positive_episodes": positive,
            "source": "computed",
        }

    def trust_compute(self: "Kernle", entity: str, domain: str = "general") -> Dict[str, Any]:
        """Compute and optionally update trust for an entity from episode history.

        This is the main entry point for dynamic trust computation. It computes
        direct trust from episodes and returns the result without automatically
        updating the stored assessment (the caller can use trust_set to persist).
        """
        result = self.compute_direct_trust(entity, domain)

        # If entity is "self", also compute the floor
        if entity == "self":
            floor_result = self.compute_self_trust_floor()
            result["self_trust_floor"] = floor_result["floor"]
            result["score"] = max(result["score"], floor_result["floor"])

        return result

    def trust_chain(
        self: "Kernle", target: str, chain: List[str], domain: str = "general"
    ) -> Dict[str, Any]:
        """Compute transitive trust through a chain (CLI entry point)."""
        return self.compute_transitive_trust(target, chain, domain)
