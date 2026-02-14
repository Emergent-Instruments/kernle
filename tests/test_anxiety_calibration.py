"""Tests for anxiety calibration: length-weighted scoring, confidence, hysteresis.

Issue #713: Calibrate anxiety heuristics on short/ambiguous prompts.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kernle import Kernle
from kernle.anxiety_core import (
    LEVEL_HYSTERESIS,
    apply_hysteresis,
    compute_raw_aging_score,
    compute_raw_aging_score_weighted,
    score_with_confidence,
)
from kernle.storage import SQLiteStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k(temp_checkpoint_dir, temp_db_path):
    """Kernle instance for calibration tests."""
    storage = SQLiteStorage(
        stack_id="test_calibration",
        db_path=temp_db_path,
    )
    return Kernle(
        stack_id="test_calibration",
        storage=storage,
        checkpoint_dir=temp_checkpoint_dir,
        strict=False,
    )


# ---------------------------------------------------------------------------
# Length-weighted raw entry scoring
# ---------------------------------------------------------------------------


class TestLengthWeightedScoring:
    """compute_raw_aging_score_weighted weights entries by content length."""

    def test_short_entries_weighted_less(self):
        """5 entries < 20 chars produce lower score than 5 entries > 200 chars."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        short_entries = [
            SimpleNamespace(content="hi", captured_at=old, timestamp=None) for _ in range(5)
        ]
        long_entries = [
            SimpleNamespace(content="x" * 250, captured_at=old, timestamp=None) for _ in range(5)
        ]

        short_score = compute_raw_aging_score_weighted(short_entries)
        long_score = compute_raw_aging_score_weighted(long_entries)

        assert short_score < long_score
        # Short entries should have meaningfully lower score
        assert long_score - short_score >= 10

    def test_empty_entries_minimal_weight(self):
        """Empty/very short entries contribute minimally."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        empty_entries = [
            SimpleNamespace(content="", captured_at=old, timestamp=None) for _ in range(5)
        ]

        score = compute_raw_aging_score_weighted(empty_entries)
        # 5 entries * weight 0.3 = 1.5 -> rounds to 2 effective entries
        # All aging -> compute_raw_aging_score(2, 2, ...) = 60
        assert score >= 0
        # Compare against full-weight: 5 full entries all aging would score
        # compute_raw_aging_score(5, 5, ...) = 76. Empty entries must score less.
        full_weight_score = compute_raw_aging_score(5, 5, 48)
        assert score < full_weight_score

    def test_mixed_length_entries_weighted(self):
        """Mix of short and long entries produces intermediate score."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        short_entries = [
            SimpleNamespace(content="hi", captured_at=old, timestamp=None) for _ in range(5)
        ]
        long_entries = [
            SimpleNamespace(content="x" * 250, captured_at=old, timestamp=None) for _ in range(5)
        ]
        mixed_entries = [
            SimpleNamespace(content="hi", captured_at=old, timestamp=None),
            SimpleNamespace(content="hi", captured_at=old, timestamp=None),
            SimpleNamespace(content="x" * 250, captured_at=old, timestamp=None),
            SimpleNamespace(content="x" * 250, captured_at=old, timestamp=None),
            SimpleNamespace(content="x" * 50, captured_at=old, timestamp=None),
        ]

        short_score = compute_raw_aging_score_weighted(short_entries)
        long_score = compute_raw_aging_score_weighted(long_entries)
        mixed_score = compute_raw_aging_score_weighted(mixed_entries)

        assert short_score < mixed_score < long_score

    def test_no_entries_returns_zero(self):
        """Empty list produces zero score."""
        assert compute_raw_aging_score_weighted([]) == 0

    def test_fresh_entries_low_score(self):
        """Entries created recently should produce a low score regardless of length."""
        now = datetime.now(timezone.utc)
        recent = now - timedelta(hours=1)

        entries = [
            SimpleNamespace(content="x" * 300, captured_at=recent, timestamp=None) for _ in range(5)
        ]

        score = compute_raw_aging_score_weighted(entries)
        # No aging entries, so score should be modest
        assert score <= 30

    def test_weight_tiers(self):
        """Verify the three weight tiers: <20 chars=0.3, 20-200=0.7, >200=1.0."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        # Single entry in each tier, all aging
        tiny = [SimpleNamespace(content="a" * 5, captured_at=old, timestamp=None)]
        medium = [SimpleNamespace(content="a" * 100, captured_at=old, timestamp=None)]
        large = [SimpleNamespace(content="a" * 300, captured_at=old, timestamp=None)]

        tiny_score = compute_raw_aging_score_weighted(tiny)
        medium_score = compute_raw_aging_score_weighted(medium)
        large_score = compute_raw_aging_score_weighted(large)

        # Higher weight tiers should produce higher scores
        assert tiny_score <= medium_score <= large_score

    def test_entries_without_content_attr_treated_as_empty(self):
        """Entries missing the content attribute should be treated as weight 0.3."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        entries = [
            SimpleNamespace(captured_at=old, timestamp=None) for _ in range(3)  # no content attr
        ]

        score = compute_raw_aging_score_weighted(entries)
        assert score >= 0  # Should not raise


# ---------------------------------------------------------------------------
# Score with confidence
# ---------------------------------------------------------------------------


class TestScoreWithConfidence:
    """score_with_confidence returns score + confidence based on sample size."""

    def test_score_with_confidence_zero_samples(self):
        """0 samples -> confidence 0.0."""
        result = score_with_confidence(50, 0)
        assert result["score"] == 50
        assert result["confidence"] == 0.0

    def test_score_with_confidence_few_samples(self):
        """2 samples -> confidence 0.5."""
        result = score_with_confidence(75, 2)
        assert result["score"] == 75
        assert result["confidence"] == 0.5

    def test_score_with_confidence_one_sample(self):
        """1 sample -> confidence 0.5."""
        result = score_with_confidence(30, 1)
        assert result["score"] == 30
        assert result["confidence"] == 0.5

    def test_score_with_confidence_moderate_samples(self):
        """3-5 samples -> confidence 0.7."""
        for n in (3, 4, 5):
            result = score_with_confidence(60, n)
            assert result["confidence"] == 0.7, f"Failed for n={n}"

    def test_score_with_confidence_many_samples(self):
        """10 samples -> confidence 0.9."""
        result = score_with_confidence(80, 10)
        assert result["score"] == 80
        assert result["confidence"] == 0.9

    def test_score_with_confidence_six_samples(self):
        """6 samples -> confidence 0.9 (above the <=5 boundary)."""
        result = score_with_confidence(40, 6)
        assert result["confidence"] == 0.9


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    """apply_hysteresis prevents oscillation near thresholds."""

    # -- No previous level (first call) --------------------------------

    def test_no_previous_level_uses_standard(self):
        """No previous level should use standard threshold logic."""
        assert apply_hysteresis(15, None) == "calm"
        assert apply_hysteresis(40, None) == "aware"
        assert apply_hysteresis(60, None) == "elevated"
        assert apply_hysteresis(80, None) == "high"
        assert apply_hysteresis(95, None) == "critical"

    # -- calm -> aware boundary (enter=35, exit=25) --------------------

    def test_calm_to_aware_enters_at_threshold(self):
        """Score of 35 transitions calm -> aware."""
        assert apply_hysteresis(35, "calm") == "aware"

    def test_calm_to_aware_below_enter_stays_calm(self):
        """Score of 34 with previous calm stays calm (below enter=35)."""
        assert apply_hysteresis(34, "calm") == "calm"

    def test_aware_to_calm_exits_below_threshold(self):
        """Score of 24 drops aware -> calm (below exit=25)."""
        assert apply_hysteresis(24, "aware") == "calm"

    def test_aware_stays_in_hysteresis_band(self):
        """Score of 30 with previous aware stays aware (above exit=25)."""
        assert apply_hysteresis(30, "aware") == "aware"

    # -- aware -> elevated boundary (enter=55, exit=45) ----------------

    def test_aware_to_elevated_enters_at_threshold(self):
        """Score of 55 transitions aware -> elevated."""
        assert apply_hysteresis(55, "aware") == "elevated"

    def test_aware_to_elevated_below_enter_stays_aware(self):
        """Score of 54 with previous aware stays aware (below enter=55)."""
        assert apply_hysteresis(54, "aware") == "aware"

    def test_elevated_to_aware_exits_below_threshold(self):
        """Score of 44 drops elevated -> aware (below exit=45)."""
        assert apply_hysteresis(44, "elevated") == "aware"

    def test_elevated_stays_in_hysteresis_band(self):
        """Score of 50 with previous elevated stays elevated (above exit=45)."""
        assert apply_hysteresis(50, "elevated") == "elevated"

    # -- elevated -> high boundary (enter=75, exit=65) -----------------

    def test_elevated_to_high_enters_at_threshold(self):
        """Score of 75 transitions elevated -> high."""
        assert apply_hysteresis(75, "elevated") == "high"

    def test_elevated_to_high_below_enter_stays_elevated(self):
        """Score of 74 with previous elevated stays elevated (below enter=75)."""
        assert apply_hysteresis(74, "elevated") == "elevated"

    def test_high_to_elevated_exits_below_threshold(self):
        """Score of 64 drops high -> elevated (below exit=65)."""
        assert apply_hysteresis(64, "high") == "elevated"

    def test_high_stays_in_hysteresis_band(self):
        """Score of 70 with previous high stays high (above exit=65)."""
        assert apply_hysteresis(70, "high") == "high"

    # -- high -> critical boundary (enter=90, exit=80) -----------------

    def test_high_to_critical_enters_at_threshold(self):
        """Score of 90 transitions high -> critical."""
        assert apply_hysteresis(90, "high") == "critical"

    def test_high_to_critical_below_enter_stays_high(self):
        """Score of 89 with previous high stays high (below enter=90)."""
        assert apply_hysteresis(89, "high") == "high"

    def test_critical_to_high_exits_below_threshold(self):
        """Score of 79 drops critical -> high (below exit=80)."""
        assert apply_hysteresis(79, "critical") == "high"

    def test_critical_stays_in_hysteresis_band(self):
        """Score of 85 with previous critical stays critical (above exit=80)."""
        assert apply_hysteresis(85, "critical") == "critical"

    # -- Multi-level transitions ---------------------------------------

    def test_multi_level_jump_up(self):
        """Score of 90 from calm should jump through all levels to critical."""
        assert apply_hysteresis(90, "calm") == "critical"

    def test_multi_level_jump_down(self):
        """Score of 10 from critical should drop through all levels to calm."""
        assert apply_hysteresis(10, "critical") == "calm"

    def test_multi_level_partial_up(self):
        """Score of 60 from calm: passes aware enter (35) and elevated enter (55),
        but not high enter (75), so lands at elevated."""
        assert apply_hysteresis(60, "calm") == "elevated"

    def test_multi_level_partial_down(self):
        """Score of 50 from critical: below critical exit (80) and high exit (65),
        but not below elevated exit (45), so lands at elevated."""
        assert apply_hysteresis(50, "critical") == "elevated"

    # -- Boundary and edge cases ---------------------------------------

    def test_same_level_stays(self):
        """When standard level matches previous level, no change."""
        assert apply_hysteresis(15, "calm") == "calm"
        assert apply_hysteresis(40, "aware") == "aware"
        assert apply_hysteresis(60, "elevated") == "elevated"
        assert apply_hysteresis(75, "high") == "high"
        assert apply_hysteresis(95, "critical") == "critical"

    def test_unknown_previous_level_uses_standard(self):
        """Unknown previous level falls back to standard."""
        assert apply_hysteresis(60, "unknown") == "elevated"

    def test_level_hysteresis_constant_has_all_non_calm_levels(self):
        """LEVEL_HYSTERESIS should have entries for all levels except calm."""
        assert "aware" in LEVEL_HYSTERESIS
        assert "elevated" in LEVEL_HYSTERESIS
        assert "high" in LEVEL_HYSTERESIS
        assert "critical" in LEVEL_HYSTERESIS
        assert "calm" not in LEVEL_HYSTERESIS

    def test_enter_thresholds_are_above_exit_thresholds(self):
        """Sanity check: enter > exit for all levels."""
        for level, (enter, exit_) in LEVEL_HYSTERESIS.items():
            assert enter > exit_, f"{level}: enter ({enter}) must be > exit ({exit_})"

    # -- Downward step-by-step transitions -----------------------------

    def test_critical_to_high_boundary(self):
        """Score in high range but above critical exit stays critical."""
        assert apply_hysteresis(82, "critical") == "critical"

    def test_high_to_elevated_boundary(self):
        """Score in elevated range but above high exit stays high."""
        assert apply_hysteresis(67, "high") == "high"

    def test_elevated_to_aware_boundary(self):
        """Score in aware range but above elevated exit stays elevated."""
        assert apply_hysteresis(47, "elevated") == "elevated"

    def test_aware_to_calm_boundary(self):
        """Score in calm range but above aware exit stays aware."""
        assert apply_hysteresis(27, "aware") == "aware"


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    """Existing compute_raw_aging_score still works with counts."""

    def test_backwards_compatible_scoring(self):
        """Existing compute_raw_aging_score still works with counts."""
        # These should all produce the same results as before
        assert compute_raw_aging_score(0, 0, 0) == 0
        assert compute_raw_aging_score(5, 0, 0) == 15  # 5 * 3
        assert compute_raw_aging_score(5, 3, 72) == 75  # 30 + 3*15
        assert compute_raw_aging_score(10, 8, 168) <= 100

    def test_weighted_and_unweighted_are_independent(self):
        """Weighted function does not change unweighted function behavior."""
        # Call weighted with some entries
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)
        entries = [
            SimpleNamespace(content="x" * 250, captured_at=old, timestamp=None) for _ in range(5)
        ]
        compute_raw_aging_score_weighted(entries)

        # Original function still works identically
        assert compute_raw_aging_score(0, 0, 0) == 0
        assert compute_raw_aging_score(5, 3, 72) == 75


# ---------------------------------------------------------------------------
# Anxiety report includes confidence
# ---------------------------------------------------------------------------


class TestAnxietyReportConfidence:
    """Anxiety reports include confidence field per dimension."""

    def test_anxiety_report_includes_confidence(self, k):
        """Report dict has confidence field per dimension."""
        report = k.get_anxiety_report()

        for dim_name, dim_data in report["dimensions"].items():
            assert "confidence" in dim_data, f"Dimension '{dim_name}' missing 'confidence' field"
            assert (
                0.0 <= dim_data["confidence"] <= 1.0
            ), f"Dimension '{dim_name}' confidence out of range: {dim_data['confidence']}"

    def test_anxiety_report_includes_overall_confidence(self, k):
        """Report includes overall_confidence field."""
        report = k.get_anxiety_report()
        assert "overall_confidence" in report
        assert 0.0 <= report["overall_confidence"] <= 1.0

    def test_report_still_has_original_fields(self, k):
        """Report still contains all original fields (backwards compat)."""
        report = k.get_anxiety_report()

        assert "overall_score" in report
        assert "overall_level" in report
        assert "overall_emoji" in report
        assert "dimensions" in report
        assert "timestamp" in report
        assert "stack_id" in report

        for dim_name in report["dimensions"]:
            dim = report["dimensions"][dim_name]
            assert "score" in dim
            assert "detail" in dim
            assert "emoji" in dim


# ---------------------------------------------------------------------------
# Stack-layer anxiety report confidence
# ---------------------------------------------------------------------------


class TestAnxietyComponentConfidence:
    """AnxietyComponent reports also include confidence."""

    def test_component_report_includes_confidence(self, k):
        """Stack-layer report should also include confidence per dimension."""
        # Access the component through the stack
        if hasattr(k, "_stack") and hasattr(k._stack, "_components"):
            for comp in k._stack._components:
                if hasattr(comp, "get_anxiety_report") and comp.name == "anxiety":
                    report = comp.get_anxiety_report()
                    for dim_name, dim_data in report.get("dimensions", {}).items():
                        assert (
                            "confidence" in dim_data
                        ), f"Component dimension '{dim_name}' missing 'confidence'"
                    return

        # If we can't reach the component, test via the stack directly
        # This is acceptable -- the mixin path is the primary test target
        pytest.skip("AnxietyComponent not accessible via stack")
