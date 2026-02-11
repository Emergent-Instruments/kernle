"""Tests for process-until-exhaustion runner."""

from __future__ import annotations

from unittest.mock import MagicMock

from kernle.exhaust import (
    HEAVY_TRANSITIONS,
    INTENSITY_LEVELS,
    LIGHT_TRANSITIONS,
    MEDIUM_TRANSITIONS,
    ExhaustCycleResult,
    ExhaustionRunner,
    ExhaustResult,
)
from kernle.processing import ProcessingResult

# =============================================================================
# Helpers
# =============================================================================


def _make_result(transition, created=None, suggestions=None, errors=None, auto_promote=True):
    """Build a ProcessingResult with given created/suggestions counts."""
    return ProcessingResult(
        layer_transition=transition,
        source_count=1,
        created=created or [],
        suggestions=suggestions or [],
        errors=errors or [],
        auto_promote=auto_promote,
    )


def _mock_kernle(process_side_effect=None):
    """Create a mock Kernle-like object with process() and checkpoint()."""
    k = MagicMock()
    k.checkpoint.return_value = {"timestamp": "2026-01-01T00:00:00Z", "task": "pre-exhaust"}
    if process_side_effect is not None:
        k.process.side_effect = process_side_effect
    else:
        # Default: always return empty list (no promotions)
        k.process.return_value = []
    return k


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestResultDataclasses:
    def test_exhaust_cycle_result_fields(self):
        cr = ExhaustCycleResult(
            cycle_number=1,
            intensity="light",
            transitions_run=LIGHT_TRANSITIONS,
            promotions=5,
        )
        assert cr.cycle_number == 1
        assert cr.intensity == "light"
        assert cr.transitions_run == LIGHT_TRANSITIONS
        assert cr.promotions == 5
        assert cr.errors == []
        assert cr.results == []

    def test_exhaust_result_fields(self):
        r = ExhaustResult(
            cycles_completed=3,
            total_promotions=10,
            converged=True,
            convergence_reason="converged",
        )
        assert r.cycles_completed == 3
        assert r.total_promotions == 10
        assert r.converged is True
        assert r.convergence_reason == "converged"
        assert r.cycle_results == []
        assert r.snapshot is None


# =============================================================================
# Convergence-Driven Intensity Tests
# =============================================================================


class TestConvergenceDrivenIntensity:
    def test_light_converges_before_medium_begins(self):
        """Light intensity converges (2 consecutive zeros), then medium starts."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        # With no promotions, each intensity converges after 2 cycles
        # light: cycles 1-2, medium: cycles 3-4, heavy: cycles 5-6
        assert result.converged is True
        assert result.convergence_reason == "converged"
        assert result.cycles_completed == 6  # 2 per intensity * 3 intensities

        # Verify intensity ordering
        intensities = [cr.intensity for cr in result.cycle_results]
        assert intensities == ["light", "light", "medium", "medium", "heavy", "heavy"]

    def test_light_with_promotions_delays_medium(self):
        """If light produces promotions, it keeps running until 2 consecutive zeros."""
        call_count = [0]

        def promote_first_light(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Promote on cycle 1 only (first call to raw_to_episode)
            if transition == "raw_to_episode" and call_count[0] == 1:
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(promote_first_light)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        # Cycle 1 (light): promotion -> reset. Cycle 2: zero. Cycle 3: zero -> converge light
        # Then medium 2 cycles, heavy 2 cycles = 3 + 2 + 2 = 7
        light_cycles = [cr for cr in result.cycle_results if cr.intensity == "light"]
        assert len(light_cycles) == 3
        assert light_cycles[0].promotions == 1
        assert light_cycles[1].promotions == 0
        assert light_cycles[2].promotions == 0

    def test_medium_converges_before_heavy(self):
        """Medium must converge before heavy starts."""
        call_count = [0]

        def promote_in_medium(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Light converges immediately (no promotions).
            # In medium, promote on the first episode_to_belief call
            if transition == "episode_to_belief" and call_count[0] <= 10:
                return [_make_result(transition, created=[{"type": "belief", "id": "b1"}])]
            return []

        k = _mock_kernle(promote_in_medium)
        runner = ExhaustionRunner(k, max_cycles=30)
        result = runner.run()

        assert result.converged is True
        intensities = [cr.intensity for cr in result.cycle_results]
        # Light converges first, then medium, then heavy
        assert intensities[0] == "light"
        assert "medium" in intensities
        assert "heavy" in intensities
        # Medium appears after light
        first_medium = intensities.index("medium")
        first_heavy = intensities.index("heavy")
        assert first_medium < first_heavy

    def test_max_cycles_stops_within_intensity(self):
        """max_cycles cap applies across all intensity levels."""

        def always_promote(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_episode":
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(always_promote)
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        assert result.converged is False
        assert result.convergence_reason == "max_cycles_reached"
        assert result.cycles_completed == 5
        # All 5 cycles should be light (never converged to escalate)
        assert all(cr.intensity == "light" for cr in result.cycle_results)

    def test_max_cycles_across_levels(self):
        """max_cycles is shared across intensity levels."""
        call_count = [0]

        def promote_in_medium_always(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Medium always promotes to prevent convergence
            if transition == "episode_to_belief":
                return [_make_result(transition, created=[{"type": "belief", "id": "b1"}])]
            return []

        k = _mock_kernle(promote_in_medium_always)
        runner = ExhaustionRunner(k, max_cycles=8)
        result = runner.run()

        assert result.converged is False
        assert result.convergence_reason == "max_cycles_reached"
        # 2 light cycles + 6 medium cycles = 8 total
        assert result.cycles_completed == 8

    def test_empty_stack_converges_all_three_levels(self):
        """Empty stack converges all three levels in 6 cycles."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        assert result.convergence_reason == "converged"
        assert result.cycles_completed == 6
        assert result.total_promotions == 0


# =============================================================================
# Convergence Logic Tests
# =============================================================================


class TestConvergenceLogic:
    def test_two_consecutive_zero_required(self):
        """A single zero-promotion cycle is not enough to converge a level."""
        call_count = [0]

        def alternate(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Pattern: promote, empty, promote, empty, empty (converge on 5th light cycle)
            if transition == "raw_to_episode" and call_count[0] in (1, 5):
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(alternate)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        light_cycles = [cr for cr in result.cycle_results if cr.intensity == "light"]
        # Cycle 1: promote, Cycle 2: zero, Cycle 3: promote, Cycle 4: zero, Cycle 5: zero -> converge
        assert len(light_cycles) == 5

    def test_error_only_cycles_reset_convergence(self):
        """Cycles with errors but 0 promotions don't count toward convergence."""

        def always_error(**kwargs):
            transition = kwargs.get("transition", "")
            return [_make_result(transition, errors=["model unavailable"])]

        k = _mock_kernle(always_error)
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        assert result.converged is False
        assert result.convergence_reason == "max_cycles_reached"
        # All cycles stuck in light (errors reset convergence counter)
        assert all(cr.intensity == "light" for cr in result.cycle_results)

    def test_error_then_clean_zeros_converges(self):
        """Error cycle followed by 2 clean zeros converges the level."""
        call_count = [0]

        def error_then_empty(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # First 2 calls (cycle 1): errors
            if call_count[0] <= 2:
                return [_make_result(transition, errors=["temporary failure"])]
            return []

        k = _mock_kernle(error_then_empty)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        light_cycles = [cr for cr in result.cycle_results if cr.intensity == "light"]
        # Cycle 1: errors (reset), Cycle 2: zero, Cycle 3: zero -> converge
        assert len(light_cycles) == 3


# =============================================================================
# Batch Size Tests
# =============================================================================


class TestBatchSize:
    def test_batch_size_passed_through(self):
        """batch_size is forwarded to k.process() calls."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20, batch_size=5)
        runner.run()

        # Every process() call should include batch_size=5
        for call in k.process.call_args_list:
            assert call.kwargs.get("batch_size") == 5

    def test_batch_size_none_by_default(self):
        """When no batch_size specified, None is passed through."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        runner.run()

        for call in k.process.call_args_list:
            assert call.kwargs.get("batch_size") is None

    def test_batch_size_stored_on_runner(self):
        """batch_size parameter is stored on the runner instance."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, batch_size=42)
        assert runner._batch_size == 42


# =============================================================================
# Dry Run Tests
# =============================================================================


class TestDryRun:
    def test_dry_run_no_writes(self):
        """Dry run does not call process() or checkpoint()."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run(dry_run=True)

        k.process.assert_not_called()
        k.checkpoint.assert_not_called()
        assert result.convergence_reason == "dry_run"

    def test_dry_run_records_all_intensities(self):
        """Dry run records one cycle per intensity level."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run(dry_run=True)

        assert len(result.cycle_results) == 3
        assert result.cycle_results[0].intensity == "light"
        assert result.cycle_results[1].intensity == "medium"
        assert result.cycle_results[2].intensity == "heavy"
        assert result.cycles_completed == 3


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshot:
    def test_snapshot_saved(self):
        """Checkpoint is created before first cycle."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        k.checkpoint.assert_called_once_with("pre-exhaust")
        assert result.snapshot is not None
        assert result.snapshot["task"] == "pre-exhaust"

    def test_snapshot_failure_logged(self):
        """Checkpoint failure is logged but doesn't abort the run."""
        k = _mock_kernle()
        k.checkpoint.side_effect = RuntimeError("checkpoint failed")
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.snapshot is None
        assert result.cycles_completed > 0


# =============================================================================
# Promotion Counting Tests
# =============================================================================


class TestPromotionCounting:
    def test_promotions_counted_auto_promote(self):
        """Total promotions are correctly summed in auto_promote mode."""

        def promote_first_cycle(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_episode":
                return [
                    _make_result(
                        transition,
                        created=[
                            {"type": "episode", "id": "e1"},
                            {"type": "episode", "id": "e2"},
                        ],
                    )
                ]
            return []

        k = _mock_kernle(promote_first_cycle)
        runner = ExhaustionRunner(k, max_cycles=20, auto_promote=True)
        result = runner.run()

        assert result.cycle_results[0].promotions == 2
        assert result.total_promotions >= 2

    def test_auto_promote_false_counts_suggestions(self):
        """When auto_promote=False, counts suggestions instead of created."""

        def suggest_first_cycle(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_episode":
                return [
                    _make_result(
                        transition,
                        suggestions=[
                            {"type": "episode", "id": "s1"},
                            {"type": "episode", "id": "s2"},
                        ],
                        auto_promote=False,
                    )
                ]
            return []

        k = _mock_kernle(suggest_first_cycle)
        runner = ExhaustionRunner(k, max_cycles=20, auto_promote=False)
        result = runner.run()

        assert result.cycle_results[0].promotions == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    def test_errors_collected(self):
        """Errors from processing are captured in cycle results."""

        def error_on_notes(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_note":
                return [_make_result(transition, errors=["parse error"])]
            return []

        k = _mock_kernle(error_on_notes)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert "parse error" in result.cycle_results[0].errors

    def test_exception_in_process_captured(self):
        """Exceptions from process() are captured, not raised."""

        def raise_on_notes(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_note":
                raise RuntimeError("model unavailable")
            return []

        k = _mock_kernle(raise_on_notes)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.cycles_completed > 0
        errors = result.cycle_results[0].errors
        assert any("raw_to_note" in e and "model unavailable" in e for e in errors)


# =============================================================================
# MCP Handler/Validator Tests
# =============================================================================


class TestMCPValidation:
    def test_validate_defaults(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust({})
        assert result["max_cycles"] == 20
        assert result["auto_promote"] is True
        assert result["dry_run"] is False
        assert result["batch_size"] is None

    def test_validate_custom_values(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust(
            {"max_cycles": 10, "auto_promote": False, "dry_run": True, "batch_size": 5}
        )
        assert result["max_cycles"] == 10
        assert result["auto_promote"] is False
        assert result["dry_run"] is True
        assert result["batch_size"] == 5

    def test_validate_max_cycles_bounds(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust({"max_cycles": 200})
        assert result["max_cycles"] == 20

        result = validate_memory_process_exhaust({"max_cycles": 0})
        assert result["max_cycles"] == 20

        result = validate_memory_process_exhaust({"max_cycles": "abc"})
        assert result["max_cycles"] == 20

    def test_validate_bad_types_use_defaults(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust({"auto_promote": "yes", "dry_run": 1})
        assert result["auto_promote"] is True
        assert result["dry_run"] is False

    def test_validate_batch_size_invalid(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust({"batch_size": -1})
        assert result["batch_size"] is None

        result = validate_memory_process_exhaust({"batch_size": "abc"})
        assert result["batch_size"] is None

        result = validate_memory_process_exhaust({"batch_size": 0})
        assert result["batch_size"] is None


# =============================================================================
# Transition Constants Tests
# =============================================================================


class TestTransitionConstants:
    def test_light_transitions(self):
        assert LIGHT_TRANSITIONS == ["raw_to_episode", "raw_to_note"]

    def test_medium_includes_light(self):
        for t in LIGHT_TRANSITIONS:
            assert t in MEDIUM_TRANSITIONS

    def test_heavy_includes_medium(self):
        for t in MEDIUM_TRANSITIONS:
            assert t in HEAVY_TRANSITIONS

    def test_heavy_includes_belief_to_value(self):
        assert "belief_to_value" in HEAVY_TRANSITIONS
        assert "belief_to_value" not in MEDIUM_TRANSITIONS

    def test_intensity_levels_ordered(self):
        assert len(INTENSITY_LEVELS) == 3
        assert INTENSITY_LEVELS[0][0] == "light"
        assert INTENSITY_LEVELS[1][0] == "medium"
        assert INTENSITY_LEVELS[2][0] == "heavy"


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    def test_intensity_start_logged(self, caplog):
        """Each intensity level logs a start message."""
        import logging

        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        with caplog.at_level(logging.INFO, logger="kernle.exhaust"):
            runner.run()

        messages = [r.message for r in caplog.records]
        assert any("Starting light intensity" in m for m in messages)
        assert any("Starting medium intensity" in m for m in messages)
        assert any("Starting heavy intensity" in m for m in messages)

    def test_cycle_logged(self, caplog):
        """Each cycle logs its number, intensity, and promotions."""
        import logging

        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        with caplog.at_level(logging.INFO, logger="kernle.exhaust"):
            runner.run()

        messages = [r.message for r in caplog.records]
        assert any("Cycle 1 [light]" in m for m in messages)

    def test_convergence_logged(self, caplog):
        """Intensity convergence is logged."""
        import logging

        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        with caplog.at_level(logging.INFO, logger="kernle.exhaust"):
            runner.run()

        messages = [r.message for r in caplog.records]
        assert any("light intensity converged" in m for m in messages)

    def test_completion_logged(self, caplog):
        """Final summary is logged."""
        import logging

        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        with caplog.at_level(logging.INFO, logger="kernle.exhaust"):
            runner.run()

        messages = [r.message for r in caplog.records]
        assert any("Exhaust complete" in m for m in messages)
