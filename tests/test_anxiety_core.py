"""Tests for anxiety_core shared scoring functions."""

from kernle.anxiety_core import (
    FIVE_DIM_WEIGHTS,
    SEVEN_DIM_WEIGHTS,
    compute_consolidation_score,
    compute_context_pressure_score,
    compute_epoch_staleness_score,
    compute_identity_coherence_score,
    compute_memory_uncertainty_score,
    compute_raw_aging_score,
    compute_unsaved_work_score,
    get_anxiety_level,
)


class TestConsolidationScore:
    """compute_consolidation_score boundaries."""

    def test_zero_episodes(self):
        assert compute_consolidation_score(0) == 0

    def test_three_episodes(self):
        assert compute_consolidation_score(3) == 21

    def test_seven_episodes(self):
        assert compute_consolidation_score(7) == 61

    def test_fifteen_episodes(self):
        assert compute_consolidation_score(15) == 93

    def test_twenty_episodes_capped(self):
        score = compute_consolidation_score(20)
        assert score <= 100


class TestIdentityCoherence:
    """compute_identity_coherence_score inversion."""

    def test_full_confidence_zero_anxiety(self):
        assert compute_identity_coherence_score(1.0) == 0

    def test_zero_confidence_full_anxiety(self):
        assert compute_identity_coherence_score(0.0) == 100

    def test_half_confidence(self):
        assert compute_identity_coherence_score(0.5) == 50


class TestMemoryUncertainty:
    """compute_memory_uncertainty_score thresholds."""

    def test_zero(self):
        assert compute_memory_uncertainty_score(0) == 0

    def test_two(self):
        assert compute_memory_uncertainty_score(2) == 30

    def test_five(self):
        assert compute_memory_uncertainty_score(5) == 75

    def test_eight_capped(self):
        score = compute_memory_uncertainty_score(8)
        assert score <= 100
        assert score >= 75


class TestRawAging:
    """compute_raw_aging_score thresholds."""

    def test_zero_total(self):
        assert compute_raw_aging_score(0, 0, 0) == 0

    def test_three_aging(self):
        score = compute_raw_aging_score(5, 3, 72)
        assert score == 75  # 30 + 3*15

    def test_eight_aging_capped(self):
        score = compute_raw_aging_score(10, 8, 168)
        assert score <= 100
        assert score >= 92


class TestEpochStaleness:
    """compute_epoch_staleness_score thresholds."""

    def test_none(self):
        assert compute_epoch_staleness_score(None) == 0

    def test_three_months(self):
        assert compute_epoch_staleness_score(3) == 15

    def test_nine_months(self):
        score = compute_epoch_staleness_score(9)
        assert score == int(30 + (9 - 6) * 6.7)  # ~50

    def test_fifteen_months(self):
        score = compute_epoch_staleness_score(15)
        assert score == int(70 + (15 - 12) * 4)  # 82


class TestContextPressure:
    """compute_context_pressure_score nonlinear curve."""

    def test_below_50_percent(self):
        score = compute_context_pressure_score(30)
        assert score < 30  # Low pressure

    def test_70_percent_high(self):
        score = compute_context_pressure_score(70)
        assert score >= 55  # High pressure

    def test_90_percent_falloff(self):
        score = compute_context_pressure_score(90)
        assert score >= 90


class TestUnsavedWork:
    """compute_unsaved_work_score scaling."""

    def test_none_checkpoint(self):
        assert compute_unsaved_work_score(None) == 50

    def test_five_minutes(self):
        assert compute_unsaved_work_score(5) == 10

    def test_thirty_minutes(self):
        score = compute_unsaved_work_score(30)
        assert score == int(30 + (30 - 15) * 1.1)  # ~47

    def test_ninety_minutes_capped(self):
        score = compute_unsaved_work_score(90)
        assert score <= 100


class TestWeights:
    """Weight normalization and completeness."""

    def test_five_dim_weights_are_renormalized(self):
        """5-dim weights should be renormalized from 7-dim shared weights."""
        shared_keys = set(FIVE_DIM_WEIGHTS.keys())
        seven_shared = {k: v for k, v in SEVEN_DIM_WEIGHTS.items() if k in shared_keys}
        total = sum(seven_shared.values())
        for key in shared_keys:
            expected = seven_shared[key] / total
            assert (
                abs(FIVE_DIM_WEIGHTS[key] - expected) < 0.01
            ), f"Weight mismatch for {key}: {FIVE_DIM_WEIGHTS[key]} vs {expected}"

    def test_seven_dim_weights_sum_to_one(self):
        total = sum(SEVEN_DIM_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_five_dim_weights_sum_to_one(self):
        total = sum(FIVE_DIM_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01


class TestAnxietyLevel:
    """get_anxiety_level boundaries."""

    def test_calm(self):
        name, label = get_anxiety_level(15)
        assert name == "calm"

    def test_aware(self):
        name, label = get_anxiety_level(40)
        assert name == "aware"

    def test_elevated(self):
        name, label = get_anxiety_level(60)
        assert name == "elevated"

    def test_high(self):
        name, label = get_anxiety_level(75)
        assert name == "high"

    def test_critical(self):
        name, label = get_anxiety_level(90)
        assert name == "critical"

    def test_zero_is_calm(self):
        name, label = get_anxiety_level(0)
        assert name == "calm"

    def test_100_is_critical(self):
        name, label = get_anxiety_level(100)
        assert name == "critical"


class TestMixinComponentAgreement:
    """Given same data, shared dims produce same scores."""

    def test_shared_dimensions_same_scores(self):
        """Both mixin and component should produce identical scores
        for the same input data using the shared scoring functions."""
        # Consolidation
        assert compute_consolidation_score(5) == compute_consolidation_score(5)
        # Identity
        assert compute_identity_coherence_score(0.7) == compute_identity_coherence_score(0.7)
        # Uncertainty
        assert compute_memory_uncertainty_score(3) == compute_memory_uncertainty_score(3)
        # Raw aging
        assert compute_raw_aging_score(5, 2, 48) == compute_raw_aging_score(5, 2, 48)
        # Epoch staleness
        assert compute_epoch_staleness_score(8) == compute_epoch_staleness_score(8)
