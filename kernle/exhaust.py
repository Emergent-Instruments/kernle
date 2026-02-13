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

INTENSITY_LEVELS = [
    ("light", LIGHT_TRANSITIONS),
    ("medium", MEDIUM_TRANSITIONS),
    ("heavy", HEAVY_TRANSITIONS),
]


class ExhaustionRunner:
    """Drive memory processing to convergence.

    Runs process() in cycles with convergence-driven intensity escalation.
    Each intensity level runs until 2 consecutive cycles with 0 promotions,
    then escalates to the next level. Stops when the heaviest level converges
    or max_cycles is reached.
    """

    def __init__(
        self,
        kernle_instance,
        *,
        max_cycles: int = 20,
        auto_promote: bool = True,
        batch_size: Optional[int] = None,
    ):
        self._k = kernle_instance
        self._max_cycles = max_cycles
        self._auto_promote = auto_promote
        self._batch_size = batch_size

    def run(self, *, dry_run: bool = False) -> ExhaustResult:
        """Run processing cycles until convergence or max_cycles.

        Each intensity level runs until 2 consecutive zero-promotion cycles,
        then escalates. Convergence means the final (heavy) level produced
        2 consecutive zero-promotion cycles.

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

        if dry_run:
            # In dry run, record one cycle per intensity level
            for _idx, (intensity, transitions) in enumerate(INTENSITY_LEVELS):
                cycle_result = ExhaustCycleResult(
                    cycle_number=result.cycles_completed + 1,
                    intensity=intensity,
                    transitions_run=transitions,
                    promotions=0,
                )
                result.cycle_results.append(cycle_result)
                result.cycles_completed += 1
            result.convergence_reason = "dry_run"
            return result

        if self._max_cycles <= 0:
            result.convergence_reason = "max_cycles_reached"
            return result

        total_cycle = 0
        any_inference_blocked = False

        for intensity, transitions in INTENSITY_LEVELS:
            logger.info("Starting %s intensity", intensity)
            consecutive_zero = 0
            level_cycles = 0
            level_converged = False

            while total_cycle < self._max_cycles:
                total_cycle += 1
                level_cycles += 1

                cycle_result = ExhaustCycleResult(
                    cycle_number=total_cycle,
                    intensity=intensity,
                    transitions_run=transitions,
                    promotions=0,
                )

                # Run each transition in this cycle
                for transition in transitions:
                    try:
                        processing_results = self._k.process(
                            transition=transition,
                            force=True,
                            auto_promote=self._auto_promote,
                            batch_size=self._batch_size,
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
                result.cycles_completed = total_cycle
                result.total_promotions += cycle_result.promotions

                # Build breakdown string for logging
                breakdown_parts = []
                for pr in cycle_result.results:
                    count = len(pr.created) if self._auto_promote else len(pr.suggestions)
                    if count > 0:
                        breakdown_parts.append(f"{pr.layer_transition}={count}")
                breakdown_str = ", ".join(breakdown_parts) if breakdown_parts else "none"

                logger.info(
                    "Cycle %d [%s]: %d promotions (%s)",
                    total_cycle,
                    intensity,
                    cycle_result.promotions,
                    breakdown_str,
                )

                # Check if all transitions were inference-blocked
                all_inference_blocked = cycle_result.results and all(
                    getattr(pr, "inference_blocked", False) for pr in cycle_result.results
                )
                if all_inference_blocked:
                    logger.info(
                        "%s intensity blocked: no inference model bound",
                        intensity,
                    )
                    any_inference_blocked = True
                    level_converged = True
                    break

                # Check convergence.
                # A cycle with 0 promotions but source_count > 0 means
                # the system is draining unprocessed items (they were
                # evaluated, just nothing promoted).  Only count as a
                # "true zero" when no sources remained to process.
                total_sources = sum(pr.source_count for pr in cycle_result.results)

                if cycle_result.errors and cycle_result.promotions == 0:
                    # Error-only cycles don't count toward convergence
                    consecutive_zero = 0
                elif cycle_result.promotions == 0 and total_sources == 0:
                    # Truly nothing left to process at this intensity
                    consecutive_zero += 1
                    if consecutive_zero >= 2:
                        level_converged = True
                        logger.info(
                            "%s intensity converged after %d cycles",
                            intensity,
                            level_cycles,
                        )
                        break
                elif cycle_result.promotions == 0 and total_sources > 0:
                    # Sources were processed but nothing promoted — keep going
                    consecutive_zero = 0
                    logger.debug(
                        "Cycle %d: 0 promotions but %d sources processed, continuing",
                        total_cycle,
                        total_sources,
                    )
                else:
                    consecutive_zero = 0

            if not level_converged:
                # Hit max_cycles before this level converged
                break

        if any_inference_blocked and result.total_promotions == 0:
            # All levels were inference-blocked with no promotions
            result.converged = True
            result.convergence_reason = "inference_unavailable"
        elif total_cycle >= self._max_cycles and not result.cycle_results[-1].promotions == 0:
            result.convergence_reason = "max_cycles_reached"
        elif total_cycle >= self._max_cycles:
            # Check if the last intensity actually converged
            last_intensity = result.cycle_results[-1].intensity if result.cycle_results else None
            if last_intensity == INTENSITY_LEVELS[-1][0]:
                # Check if last 2 cycles were zero-promotion
                last_two = result.cycle_results[-2:]
                if len(last_two) == 2 and all(
                    cr.promotions == 0 and not cr.errors for cr in last_two
                ):
                    result.converged = True
                    result.convergence_reason = "converged"
                else:
                    result.convergence_reason = "max_cycles_reached"
            else:
                result.convergence_reason = "max_cycles_reached"
        else:
            # Finished all intensity levels
            result.converged = True
            result.convergence_reason = "converged"

        logger.info(
            "Exhaust complete: %d cycles, %d promotions, converged=%s",
            result.cycles_completed,
            result.total_promotions,
            result.converged,
        )

        return result
