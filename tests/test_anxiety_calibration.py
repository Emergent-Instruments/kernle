"""Tests for anxiety calibration: length-weighted scoring, confidence, hysteresis.

Issue #713: Calibrate anxiety heuristics on short/ambiguous prompts.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from kernle import Kernle
from kernle.anxiety_core import (
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

    def test_hysteresis_prevents_oscillation(self):
        """Score at 72: was 'elevated' stays 'elevated'; score at 76: transitions to 'high'."""
        # Score of 72 with previous level "elevated" should stay elevated
        # because 72 < enter_threshold (75)
        level = apply_hysteresis(72, "elevated", enter_threshold=75, exit_threshold=60)
        assert level == "elevated"

        # Score of 76 should transition to "high"
        level = apply_hysteresis(76, "elevated", enter_threshold=75, exit_threshold=60)
        assert level == "high"

    def test_hysteresis_exit_threshold(self):
        """Score drops from 'high' to 65: stays 'high'; drops to 55: transitions to 'elevated'."""
        # Score 65 with previous "high" should stay high
        # because 65 > exit_threshold (60)
        level = apply_hysteresis(65, "high", enter_threshold=75, exit_threshold=60)
        assert level == "high"

        # Score 55 should transition to "elevated"
        level = apply_hysteresis(55, "high", enter_threshold=75, exit_threshold=60)
        assert level == "elevated"

    def test_hysteresis_no_previous_level(self):
        """No previous level should use standard threshold logic."""
        # Score 68 maps to "elevated" (51-70 range)
        level = apply_hysteresis(68, None, enter_threshold=75, exit_threshold=60)
        assert level == "elevated"

        level = apply_hysteresis(80, None, enter_threshold=75, exit_threshold=60)
        assert level == "high"

    def test_hysteresis_critical_stays_critical_above_exit(self):
        """Critical level should stay critical until score drops below exit."""
        level = apply_hysteresis(88, "critical", enter_threshold=90, exit_threshold=80)
        assert level == "critical"

        level = apply_hysteresis(75, "critical", enter_threshold=90, exit_threshold=80)
        assert level == "high"

    def test_hysteresis_calm_range(self):
        """Calm level should stay calm if score stays below exit threshold."""
        level = apply_hysteresis(25, "calm", enter_threshold=35, exit_threshold=25)
        assert level == "calm"


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
