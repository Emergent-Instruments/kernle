"""Process-until-exhaustion runner -- drives memory processing to convergence.

Wraps Entity/Kernle process() with multi-cycle convergence detection,
intensity scaling, and safety caps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kernle.processing import ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class ExhaustCycleResult:
    """Result of a single exhaustion cycle."""

    cycle_number: int
    intensity: str  # "light", "medium", "heavy"
    transitions_run: List[str]
    promotions: int
    errors: List[str] = field(default_factory=list)
    results: List[ProcessingResult] = field(default_factory=list)


@dataclass
class ExhaustResult:
    """Result of a complete exhaustion run."""

    cycles_completed: int
    total_promotions: int
    converged: bool
    convergence_reason: str
    cycle_results: List[ExhaustCycleResult] = field(default_factory=list)
    snapshot: Optional[Dict[str, Any]] = None


# Intensity levels: which transitions run at each stage
LIGHT_TRANSITIONS = ["raw_to_episode", "raw_to_note"]
MEDIUM_TRANSITIONS = LIGHT_TRANSITIONS + [
    "episode_to_belief",
    "episode_to_goal",
    "episode_to_relationship",
    "episode_to_drive",
]
HEAVY_TRANSITIONS = MEDIUM_TRANSITIONS + ["belief_to_value"]


class ExhaustionRunner:
    """Drive memory processing to convergence.

    Runs process() in cycles with escalating intensity until convergence
    (2 consecutive cycles with 0 promotions) or max_cycles reached.
    """

    def __init__(self, kernle_instance, *, max_cycles: int = 20, auto_promote: bool = True):
        self._k = kernle_instance
        self._max_cycles = max_cycles
        self._auto_promote = auto_promote

    def run(self, *, dry_run: bool = False) -> ExhaustResult:
        """Run processing cycles until convergence or max_cycles.

        Convergence: 2 consecutive cycles with 0 promotions.

        Intensity scaling:
        - Cycles 1-3 (light): raw_to_episode, raw_to_note only
        - Cycles 4-6 (medium): + episode_to_belief/goal/relationship/drive
        - Cycles 7+ (heavy): + belief_to_value

        Safety: Creates a checkpoint before first cycle.
        """
        result = ExhaustResult(
            cycles_completed=0,
            total_promotions=0,
            converged=False,
            convergence_reason="",
        )

        # Create safety checkpoint before first cycle
        if not dry_run:
            try:
                snapshot = self._k.checkpoint("pre-exhaust")
                result.snapshot = snapshot
            except Exception as e:
                logger.warning("Could not create pre-exhaust checkpoint: %s", e)

        consecutive_zero = 0

        for cycle_num in range(1, self._max_cycles + 1):
            # Determine intensity
            if cycle_num <= 3:
                intensity = "light"
                transitions = LIGHT_TRANSITIONS
            elif cycle_num <= 6:
                intensity = "medium"
                transitions = MEDIUM_TRANSITIONS
            else:
                intensity = "heavy"
                transitions = HEAVY_TRANSITIONS

            cycle_result = ExhaustCycleResult(
                cycle_number=cycle_num,
                intensity=intensity,
                transitions_run=transitions,
                promotions=0,
            )

            if dry_run:
                # In dry run, just record what would run
                result.cycle_results.append(cycle_result)
                result.cycles_completed = cycle_num
                # Can't determine convergence in dry run, just run one cycle
                result.convergence_reason = "dry_run"
                break

            # Run each transition in this cycle
            for transition in transitions:
                try:
                    processing_results = self._k.process(
                        transition=transition,
                        force=True,
                        auto_promote=self._auto_promote,
                    )
                    for pr in processing_results:
                        cycle_result.results.append(pr)
                        if self._auto_promote:
                            cycle_result.promotions += len(pr.created)
                        else:
                            cycle_result.promotions += len(pr.suggestions)
                        cycle_result.errors.extend(pr.errors)
                except Exception as e:
                    cycle_result.errors.append(f"{transition}: {e}")

            result.cycle_results.append(cycle_result)
            result.cycles_completed = cycle_num
            result.total_promotions += cycle_result.promotions

            # Check convergence — error-only cycles don't count as zero-promotion
            if cycle_result.errors and cycle_result.promotions == 0:
                consecutive_zero = 0  # Reset: errors mean we can't trust "no promotions"
            elif cycle_result.promotions == 0:
                consecutive_zero += 1
                if consecutive_zero >= 2:
                    result.converged = True
                    result.convergence_reason = "two_consecutive_zero_promotion_cycles"
                    break
            else:
                consecutive_zero = 0

        if not result.converged:
            if result.cycles_completed >= self._max_cycles:
                result.convergence_reason = "max_cycles_reached"
            elif dry_run:
                result.convergence_reason = "dry_run"

        return result
