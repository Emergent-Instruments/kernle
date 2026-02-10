"""Tests for process-until-exhaustion runner."""

from __future__ import annotations

from unittest.mock import MagicMock

from kernle.exhaust import (
    HEAVY_TRANSITIONS,
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
            convergence_reason="two_consecutive_zero_promotion_cycles",
        )
        assert r.cycles_completed == 3
        assert r.total_promotions == 10
        assert r.converged is True
        assert r.convergence_reason == "two_consecutive_zero_promotion_cycles"
        assert r.cycle_results == []
        assert r.snapshot is None


# =============================================================================
# Convergence Tests
# =============================================================================


class TestConvergence:
    def test_convergence_with_empty_stack(self):
        """Empty stack produces 0 promotions per cycle, converges after 2."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        assert result.convergence_reason == "two_consecutive_zero_promotion_cycles"
        assert result.cycles_completed == 2
        assert result.total_promotions == 0

    def test_two_consecutive_zero_cycles(self):
        """Convergence requires TWO consecutive zero-promotion cycles."""
        # Cycle 1: 1 promotion, Cycle 2: 0, Cycle 3: 1, Cycle 4: 0, Cycle 5: 0 -> converge
        call_count = [0]

        def process_side_effect(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Return a promotion on first and third call to raw_to_episode
            if transition == "raw_to_episode" and call_count[0] in (1, 5):
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(process_side_effect)
        runner = ExhaustionRunner(k, max_cycles=20)
        result = runner.run()

        assert result.converged is True
        assert result.convergence_reason == "two_consecutive_zero_promotion_cycles"

    def test_max_cycles_respected(self):
        """Runner stops at max_cycles even without convergence."""

        def always_promote(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_episode":
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(always_promote)
        runner = ExhaustionRunner(k, max_cycles=3)
        result = runner.run()

        assert result.converged is False
        assert result.convergence_reason == "max_cycles_reached"
        assert result.cycles_completed == 3


# =============================================================================
# Intensity Scaling Tests
# =============================================================================


class TestIntensityScaling:
    def test_light_cycles_1_to_3(self):
        """Cycles 1-3 only run light transitions."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=3)
        # Won't converge within 3 cycles since convergence needs 2 zeros
        # but with empty process it will converge at cycle 2
        result = runner.run()

        # Should converge at cycle 2 with empty stack
        for cr in result.cycle_results:
            assert cr.intensity == "light"
            assert cr.transitions_run == LIGHT_TRANSITIONS

    def test_intensity_medium_cycles_4_to_6(self):
        """Cycles 4-6 use medium transitions."""
        # Need to prevent convergence until cycle 4+
        call_count = [0]

        def promote_sometimes(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Promote in cycles 1-4 (transitions 1-8 for light, 1-12 for medium)
            # Light has 2 transitions per cycle, so calls 1-2=cycle1, 3-4=cycle2, 5-6=cycle3
            # At cycle 4 (medium, 6 transitions): calls 7-12
            if transition == "raw_to_episode" and call_count[0] <= 7:
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(promote_sometimes)
        runner = ExhaustionRunner(k, max_cycles=10)
        result = runner.run()

        # Verify cycle 4+ uses medium intensity
        cycle_4_and_later = [cr for cr in result.cycle_results if cr.cycle_number >= 4]
        for cr in cycle_4_and_later:
            if cr.cycle_number <= 6:
                assert cr.intensity == "medium"
                assert cr.transitions_run == MEDIUM_TRANSITIONS

    def test_intensity_heavy_cycles_7_plus(self):
        """Cycles 7+ use heavy transitions."""
        call_count = [0]

        def promote_until_7(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Keep promoting on raw_to_episode until we get well into cycle 7
            # Light (2 per cycle) * 3 = 6 calls
            # Medium (6 per cycle) * 3 = 18 calls
            # So calls 1-24 cover cycles 1-6
            if transition == "raw_to_episode" and call_count[0] <= 25:
                return [_make_result(transition, created=[{"type": "episode", "id": "e1"}])]
            return []

        k = _mock_kernle(promote_until_7)
        runner = ExhaustionRunner(k, max_cycles=10)
        result = runner.run()

        cycle_7_plus = [cr for cr in result.cycle_results if cr.cycle_number >= 7]
        for cr in cycle_7_plus:
            assert cr.intensity == "heavy"
            assert cr.transitions_run == HEAVY_TRANSITIONS


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
        assert result.cycles_completed == 1
        assert len(result.cycle_results) == 1

    def test_dry_run_records_transitions(self):
        """Dry run still records what transitions would run."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run(dry_run=True)

        cr = result.cycle_results[0]
        assert cr.cycle_number == 1
        assert cr.intensity == "light"
        assert cr.transitions_run == LIGHT_TRANSITIONS
        assert cr.promotions == 0


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshot:
    def test_snapshot_saved(self):
        """Checkpoint is created before first cycle."""
        k = _mock_kernle()
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        k.checkpoint.assert_called_once_with("pre-exhaust")
        assert result.snapshot is not None
        assert result.snapshot["task"] == "pre-exhaust"

    def test_snapshot_failure_logged(self):
        """Checkpoint failure is logged but doesn't abort the run."""
        k = _mock_kernle()
        k.checkpoint.side_effect = RuntimeError("checkpoint failed")
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        assert result.snapshot is None
        # Run should still complete
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
        runner = ExhaustionRunner(k, max_cycles=5, auto_promote=True)
        result = runner.run()

        # Cycle 1: 2 promotions (from raw_to_episode)
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
        runner = ExhaustionRunner(k, max_cycles=5, auto_promote=False)
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
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        # First cycle should have errors
        assert "parse error" in result.cycle_results[0].errors

    def test_exception_in_process_captured(self):
        """Exceptions from process() are captured, not raised."""

        def raise_on_notes(**kwargs):
            transition = kwargs.get("transition", "")
            if transition == "raw_to_note":
                raise RuntimeError("model unavailable")
            return []

        k = _mock_kernle(raise_on_notes)
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        # Should still complete (not raise)
        assert result.cycles_completed > 0
        # Error should be captured
        errors = result.cycle_results[0].errors
        assert any("raw_to_note" in e and "model unavailable" in e for e in errors)

    def test_error_only_cycles_do_not_converge(self):
        """Cycles with errors but 0 promotions should NOT count as convergence."""

        def always_error(**kwargs):
            transition = kwargs.get("transition", "")
            return [_make_result(transition, errors=["model unavailable"])]

        k = _mock_kernle(always_error)
        runner = ExhaustionRunner(k, max_cycles=5)
        result = runner.run()

        # Should NOT converge — every cycle has errors
        assert result.converged is False
        assert result.convergence_reason == "max_cycles_reached"
        assert result.cycles_completed == 5

    def test_error_then_clean_zero_converges(self):
        """Error cycle followed by 2 clean zero-promotion cycles converges."""
        call_count = [0]

        def error_then_empty(**kwargs):
            call_count[0] += 1
            transition = kwargs.get("transition", "")
            # Cycle 1 (calls 1-2): errors on everything
            if call_count[0] <= 2:
                return [_make_result(transition, errors=["temporary failure"])]
            # Cycles 2+ (calls 3+): clean, no promotions
            return []

        k = _mock_kernle(error_then_empty)
        runner = ExhaustionRunner(k, max_cycles=10)
        result = runner.run()

        assert result.converged is True
        assert result.convergence_reason == "two_consecutive_zero_promotion_cycles"
        # Cycle 1: errors (reset streak), Cycle 2: clean zero, Cycle 3: clean zero -> converge
        assert result.cycles_completed == 3


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

    def test_validate_custom_values(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust(
            {"max_cycles": 10, "auto_promote": False, "dry_run": True}
        )
        assert result["max_cycles"] == 10
        assert result["auto_promote"] is False
        assert result["dry_run"] is True

    def test_validate_max_cycles_bounds(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        # Over 100 should clamp to default
        result = validate_memory_process_exhaust({"max_cycles": 200})
        assert result["max_cycles"] == 20

        # 0 should clamp to default
        result = validate_memory_process_exhaust({"max_cycles": 0})
        assert result["max_cycles"] == 20

        # Non-int should clamp to default
        result = validate_memory_process_exhaust({"max_cycles": "abc"})
        assert result["max_cycles"] == 20

    def test_validate_bad_types_use_defaults(self):
        from kernle.mcp.handlers.processing import validate_memory_process_exhaust

        result = validate_memory_process_exhaust({"auto_promote": "yes", "dry_run": 1})
        assert result["auto_promote"] is True
        assert result["dry_run"] is False


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
