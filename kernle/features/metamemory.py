"""Meta-memory mixin for Kernle.

This module provides memory provenance and confidence tracking,
enabling memory verification and lineage analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from kernle.core import Kernle

logger = logging.getLogger(__name__)


# Default decay constants
DEFAULT_DECAY_RATE = 0.01  # 1% per period
DEFAULT_DECAY_PERIOD_DAYS = 30
DEFAULT_DECAY_FLOOR = 0.5


@dataclass
class DecayConfig:
    """Configuration for time-based confidence decay.

    Attributes:
        decay_rate: Amount of confidence lost per decay period (default: 0.01 = 1%)
        decay_period_days: Number of days per decay period (default: 30)
        decay_floor: Minimum confidence after decay (default: 0.5)
        enabled: Whether decay is enabled (default: True)
    """

    decay_rate: float = DEFAULT_DECAY_RATE
    decay_period_days: int = DEFAULT_DECAY_PERIOD_DAYS
    decay_floor: float = DEFAULT_DECAY_FLOOR
    enabled: bool = True

    def __post_init__(self):
        """Validate decay config values."""
        if self.decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        if self.decay_rate > 1.0:
            raise ValueError("decay_rate must be <= 1.0")
        if self.decay_period_days <= 0:
            raise ValueError("decay_period_days must be positive")
        if self.decay_floor < 0 or self.decay_floor > 1.0:
            raise ValueError("decay_floor must be between 0.0 and 1.0")


# Default decay configs per memory type
DEFAULT_DECAY_CONFIGS: Dict[str, DecayConfig] = {
    # Episodes use standard decay
    "episode": DecayConfig(),
    # Beliefs use standard decay
    "belief": DecayConfig(),
    # Values decay slower (more stable beliefs)
    "value": DecayConfig(decay_rate=0.005, decay_period_days=60, decay_floor=0.7),
    # Goals use standard decay
    "goal": DecayConfig(),
    # Notes can decay faster
    "note": DecayConfig(decay_rate=0.015, decay_period_days=30, decay_floor=0.4),
    # Drives are relatively stable
    "drive": DecayConfig(decay_rate=0.005, decay_period_days=60, decay_floor=0.6),
    # Relationships use standard decay
    "relationship": DecayConfig(),
    # Default fallback
    "default": DecayConfig(),
}


class MetaMemoryMixin:
    """Mixin providing meta-memory capabilities.

    Enables:
    - Confidence tracking for memories
    - Memory verification with evidence
    - Provenance/lineage tracking
    - Uncertainty identification
    """

    def get_memory_confidence(
        self: "Kernle",
        memory_type: str,
        memory_id: str,
        apply_decay: bool = True,
    ) -> float:
        """Get confidence score for a memory.

        Args:
            memory_type: Type of memory (episode, belief, value, goal, note)
            memory_id: ID of the memory
            apply_decay: Whether to apply time-based decay (default: True)

        Returns:
            Confidence score (0.0-1.0), or -1.0 if not found
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if record:
            if apply_decay:
                return self.get_confidence_with_decay(record, memory_type)
            return getattr(record, "confidence", 0.8)
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

        old_confidence = getattr(record, "confidence", 0.8)
        new_confidence = min(1.0, old_confidence + 0.1)

        # Track confidence change
        confidence_history = getattr(record, "confidence_history", None) or []
        confidence_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old": old_confidence,
                "new": new_confidence,
                "reason": evidence or "verification",
            }
        )

        return self._storage.update_memory_meta(
            memory_type=memory_type,
            memory_id=memory_id,
            confidence=new_confidence,
            verification_count=(getattr(record, "verification_count", 0) or 0) + 1,
            last_verified=datetime.now(timezone.utc),
            confidence_history=confidence_history,
        )

    def get_memory_lineage(self: "Kernle", memory_type: str, memory_id: str) -> Dict[str, Any]:
        """Get provenance chain for a memory.

        Args:
            memory_type: Type of memory
            memory_id: ID of the memory

        Returns:
            Lineage information including source, derivations, and decay info
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}

        stored_confidence = getattr(record, "confidence", None)
        effective_confidence = self.get_confidence_with_decay(record, memory_type)
        decay_config = self.get_decay_config(memory_type)

        return {
            "id": memory_id,
            "type": memory_type,
            "source_type": getattr(record, "source_type", "unknown"),
            "source_episodes": getattr(record, "source_episodes", None),
            "derived_from": getattr(record, "derived_from", None),
            "stored_confidence": stored_confidence,
            "effective_confidence": effective_confidence,
            "confidence_decayed": (
                effective_confidence < stored_confidence if stored_confidence is not None else False
            ),
            "current_confidence": stored_confidence,  # Legacy field
            "verification_count": getattr(record, "verification_count", 0),
            "last_verified": (
                getattr(record, "last_verified").isoformat()
                if getattr(record, "last_verified", None)
                else None
            ),
            "confidence_history": getattr(record, "confidence_history", None),
            "decay_config": {
                "decay_rate": decay_config.decay_rate,
                "decay_period_days": decay_config.decay_period_days,
                "decay_floor": decay_config.decay_floor,
                "enabled": decay_config.enabled,
            },
        }

    def get_uncertain_memories(
        self: "Kernle",
        threshold: float = 0.5,
        limit: int = 20,
        apply_decay: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get memories with confidence below threshold.

        Args:
            threshold: Confidence threshold
            limit: Maximum results
            apply_decay: Whether to use decayed confidence for filtering

        Returns:
            List of low-confidence memories
        """
        # Get more results than needed to filter by decayed confidence
        fetch_limit = limit * 3 if apply_decay else limit
        results = self._storage.get_memories_by_confidence(
            threshold=1.0 if apply_decay else threshold,  # Get all if applying decay
            below=True,
            limit=fetch_limit,
        )

        formatted = []
        for r in results:
            record = r.record
            stored_confidence = getattr(record, "confidence", 0.8)

            if apply_decay:
                effective_confidence = self.get_confidence_with_decay(record, r.record_type)
                # Filter by decayed confidence
                if effective_confidence >= threshold:
                    continue
            else:
                effective_confidence = stored_confidence
                if stored_confidence >= threshold:
                    continue

            formatted.append(
                {
                    "id": record.id,
                    "type": r.record_type,
                    "confidence": effective_confidence,
                    "stored_confidence": stored_confidence,
                    "summary": self._get_memory_summary(r.record_type, record),
                    "created_at": (
                        record.created_at.strftime("%Y-%m-%d")
                        if getattr(record, "created_at", None)
                        else "unknown"
                    ),
                }
            )

            if len(formatted) >= limit:
                break

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

        **STATUS: STUB - Not yet implemented.**

        This method is planned but not yet functional. It currently returns
        the source memory's confidence but does not actually propagate changes
        to derived memories. The `derived_from` field tracking is in place;
        the propagation logic will be implemented in a future release.

        When implemented, this will:
        1. Find all memories where `derived_from` contains the source reference
        2. Update their confidence based on the source's change
        3. Cascade updates to further derivations

        Args:
            memory_type: Type of source memory
            memory_id: ID of source memory

        Returns:
            Result dict with source confidence info. Note: `updated` is always 0
            until this feature is implemented.
        """
        record = self._storage.get_memory(memory_type, memory_id)
        if not record:
            return {"error": f"Memory {memory_type}:{memory_id} not found"}

        source_confidence = getattr(record, "confidence", 0.8)
        source_ref = f"{memory_type}:{memory_id}"

        # TODO: Implement actual propagation
        # - Query all memory tables for records where derived_from contains source_ref
        # - Update confidence based on propagation rules (min? weighted average?)
        # - Handle cascading updates
        updated = 0

        return {
            "source_confidence": source_confidence,
            "source_ref": source_ref,
            "updated": updated,
            "note": "Propagation not yet implemented - see roadmap",
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

    def get_decay_config(self: "Kernle", memory_type: str) -> DecayConfig:
        """Get decay configuration for a memory type.

        Args:
            memory_type: Type of memory (episode, belief, value, etc.)

        Returns:
            DecayConfig for the memory type
        """
        if not hasattr(self, "_decay_configs") or self._decay_configs is None:
            self._decay_configs = {}
        if memory_type in self._decay_configs:
            return self._decay_configs[memory_type]
        return DEFAULT_DECAY_CONFIGS.get(memory_type, DEFAULT_DECAY_CONFIGS["default"])

    def set_decay_config(
        self: "Kernle",
        memory_type: str,
        config: Optional["DecayConfig"] = None,
        decay_rate: Optional[float] = None,
        decay_period_days: Optional[int] = None,
        decay_floor: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Set decay configuration for a memory type.

        Args:
            memory_type: Type of memory to configure
            config: Full DecayConfig to use (overrides individual params)
            decay_rate: Amount of confidence lost per decay period
            decay_period_days: Number of days per decay period
            decay_floor: Minimum confidence after decay
            enabled: Whether decay is enabled
        """
        if not hasattr(self, "_decay_configs") or self._decay_configs is None:
            self._decay_configs = {}

        if config is not None:
            self._decay_configs[memory_type] = config
            return

        # Start with existing or default config
        base = self.get_decay_config(memory_type)

        self._decay_configs[memory_type] = DecayConfig(
            decay_rate=decay_rate if decay_rate is not None else base.decay_rate,
            decay_period_days=(
                decay_period_days if decay_period_days is not None else base.decay_period_days
            ),
            decay_floor=decay_floor if decay_floor is not None else base.decay_floor,
            enabled=enabled if enabled is not None else base.enabled,
        )

    def get_confidence_with_decay(
        self: "Kernle",
        memory: Any,
        memory_type: str,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """Calculate confidence with time-based decay.

        Confidence decays based on time since last verification.
        Protected memories do not decay.

        Args:
            memory: The memory record
            memory_type: Type of memory (episode, belief, etc.)
            reference_time: Time to calculate decay from (default: now)

        Returns:
            Effective confidence after decay
        """
        config = self.get_decay_config(memory_type)

        # If decay disabled, return stored confidence
        if not config.enabled:
            return getattr(memory, "confidence", 0.8)

        # Protected memories don't decay
        if getattr(memory, "is_protected", False):
            return getattr(memory, "confidence", 0.8)

        base_confidence = getattr(memory, "confidence", 0.8)

        # Get last verified date, fall back to created_at
        last_verified = getattr(memory, "last_verified", None)
        if last_verified is None:
            last_verified = getattr(memory, "created_at", None)

        if last_verified is None:
            return base_confidence

        # Ensure timezone-aware comparison
        now = reference_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if last_verified.tzinfo is None:
            last_verified = last_verified.replace(tzinfo=timezone.utc)

        days_since = (now - last_verified).days

        # No decay for recently verified memories (within 24 hours)
        if days_since <= 0:
            return base_confidence

        # Calculate decay
        decay_periods = days_since / config.decay_period_days
        decay_amount = config.decay_rate * decay_periods

        effective_confidence = max(config.decay_floor, base_confidence - decay_amount)

        return effective_confidence
