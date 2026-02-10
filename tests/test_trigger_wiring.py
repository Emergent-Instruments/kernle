"""Tests for time/valence trigger wiring in check_triggers() — Issue #404.

Tests verify that check_triggers() computes and passes time-based and
valence-based trigger signals from actual memory data, not just quantity.

Covers:
- Raw aging: stale raw entries trigger processing even below count threshold
- Emotional arousal: high-arousal episodes trigger consolidation even below count
- Time-based triggers for episode and belief transitions
- Existing quantity triggers still work unchanged
- Helper functions: _hours_since_oldest_raw, _hours_since_oldest_episode,
  _hours_since_oldest_belief, _cumulative_arousal
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from kernle.processing import (
    LayerConfig,
    MemoryProcessor,
    _cumulative_arousal,
    _hours_since_oldest_belief,
    _hours_since_oldest_episode,
    _hours_since_oldest_raw,
    evaluate_triggers,
)

STACK_ID = "test-triggers"


# =============================================================================
# Fixtures
# =============================================================================


class MockInference:
    def infer(self, prompt, *, system=None):
        return "[]"

    def embed(self, text):
        return [0.0] * 64


def _make_mock_stack():
    mock_stack = MagicMock()
    mock_stack.stack_id = STACK_ID
    mock_stack._backend = MagicMock()
    return mock_stack


def _make_processor(mock_stack, configs=None):
    return MemoryProcessor(
        stack=mock_stack,
        inference=MockInference(),
        core_id="test",
        configs=configs,
    )


# =============================================================================
# Helper function unit tests
# =============================================================================


class TestHourssinceOldestRaw:
    def test_empty_list_returns_none(self):
        assert _hours_since_oldest_raw([], datetime.now(timezone.utc)) is None

    def test_no_timestamps_returns_none(self):
        entry = MagicMock(spec=["id", "blob"])
        del entry.captured_at
        assert _hours_since_oldest_raw([entry], datetime.now(timezone.utc)) is None

    def test_single_entry_computes_hours(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        entry = MagicMock()
        entry.captured_at = now - timedelta(hours=36)
        result = _hours_since_oldest_raw([entry], now)
        assert abs(result - 36.0) < 0.01

    def test_multiple_entries_uses_oldest(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        e1 = MagicMock()
        e1.captured_at = now - timedelta(hours=10)
        e2 = MagicMock()
        e2.captured_at = now - timedelta(hours=48)
        e3 = MagicMock()
        e3.captured_at = now - timedelta(hours=5)
        result = _hours_since_oldest_raw([e1, e2, e3], now)
        assert abs(result - 48.0) < 0.01

    def test_naive_datetime_treated_as_utc(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        entry = MagicMock()
        entry.captured_at = datetime(2025, 6, 14, 12, 0, 0)  # naive
        result = _hours_since_oldest_raw([entry], now)
        assert abs(result - 24.0) < 0.01


class TestHoursSinceOldestEpisode:
    def test_empty_list_returns_none(self):
        assert _hours_since_oldest_episode([], datetime.now(timezone.utc)) is None

    def test_single_episode_computes_hours(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        ep = MagicMock()
        ep.created_at = now - timedelta(hours=72)
        result = _hours_since_oldest_episode([ep], now)
        assert abs(result - 72.0) < 0.01

    def test_no_created_at_returns_none(self):
        ep = MagicMock(spec=["id", "objective"])
        del ep.created_at
        assert _hours_since_oldest_episode([ep], datetime.now(timezone.utc)) is None


class TestHoursSinceOldestBelief:
    def test_empty_list_returns_none(self):
        assert _hours_since_oldest_belief([], datetime.now(timezone.utc)) is None

    def test_single_belief_computes_hours(self):
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        b = MagicMock()
        b.created_at = now - timedelta(hours=50)
        result = _hours_since_oldest_belief([b], now)
        assert abs(result - 50.0) < 0.01


class TestCumulativeArousal:
    def test_empty_list_returns_zero(self):
        assert _cumulative_arousal([]) == 0.0

    def test_single_episode(self):
        ep = MagicMock()
        ep.emotional_arousal = 0.8
        assert abs(_cumulative_arousal([ep]) - 0.8) < 0.001

    def test_multiple_episodes_sum(self):
        e1 = MagicMock()
        e1.emotional_arousal = 0.5
        e2 = MagicMock()
        e2.emotional_arousal = 0.9
        e3 = MagicMock()
        e3.emotional_arousal = 0.3
        assert abs(_cumulative_arousal([e1, e2, e3]) - 1.7) < 0.001

    def test_missing_arousal_defaults_to_zero(self):
        ep = MagicMock(spec=["id", "objective"])
        del ep.emotional_arousal
        assert _cumulative_arousal([ep]) == 0.0


# =============================================================================
# evaluate_triggers — valence now applies to all transitions
# =============================================================================


class TestEvaluateTriggersValenceAllTransitions:
    """Verify that valence threshold applies to all transitions, not just raw."""

    def test_episode_to_belief_valence_triggers(self):
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert evaluate_triggers("episode_to_belief", config, 1, cumulative_valence=2.5)

    def test_episode_to_goal_valence_triggers(self):
        config = LayerConfig(
            layer_transition="episode_to_goal",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert evaluate_triggers("episode_to_goal", config, 1, cumulative_valence=3.0)

    def test_episode_to_drive_valence_triggers(self):
        config = LayerConfig(
            layer_transition="episode_to_drive",
            quantity_threshold=100,
            valence_threshold=1.5,
        )
        assert evaluate_triggers("episode_to_drive", config, 1, cumulative_valence=1.5)

    def test_below_valence_does_not_trigger(self):
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=5.0,
        )
        assert not evaluate_triggers("episode_to_belief", config, 1, cumulative_valence=4.9)

    def test_belief_to_value_valence_triggers(self):
        """belief_to_value should also respond to valence if passed."""
        config = LayerConfig(
            layer_transition="belief_to_value",
            quantity_threshold=100,
            valence_threshold=2.0,
        )
        assert evaluate_triggers("belief_to_value", config, 1, cumulative_valence=2.5)


# =============================================================================
# check_triggers — time-based wiring
# =============================================================================


class TestCheckTriggersTimeBased:
    """check_triggers() passes time signals computed from actual data."""

    def test_raw_aging_triggers_below_count(self):
        """Stale raw entries trigger processing even below quantity threshold."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,  # High threshold — won't trigger on count
            time_threshold_hours=24,
        )
        # Only 2 raw entries, but one is 48 hours old
        now = datetime.now(timezone.utc)
        raw1 = MagicMock()
        raw1.captured_at = now - timedelta(hours=48)
        raw2 = MagicMock()
        raw2.captured_at = now - timedelta(hours=1)
        mock_stack._backend.list_raw.return_value = [raw1, raw2]

        processor = _make_processor(mock_stack, configs={"raw_to_episode": config})
        assert processor.check_triggers("raw_to_episode")

    def test_raw_no_aging_below_count_does_not_trigger(self):
        """Fresh raw entries below count threshold do not trigger."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="raw_to_episode",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        now = datetime.now(timezone.utc)
        raw1 = MagicMock()
        raw1.captured_at = now - timedelta(hours=2)
        mock_stack._backend.list_raw.return_value = [raw1]

        processor = _make_processor(mock_stack, configs={"raw_to_episode": config})
        assert not processor.check_triggers("raw_to_episode")

    def test_raw_to_note_aging_triggers(self):
        """raw_to_note also uses time-based triggers."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="raw_to_note",
            quantity_threshold=100,
            time_threshold_hours=12,
        )
        now = datetime.now(timezone.utc)
        raw1 = MagicMock()
        raw1.captured_at = now - timedelta(hours=15)
        mock_stack._backend.list_raw.return_value = [raw1]

        processor = _make_processor(mock_stack, configs={"raw_to_note": config})
        assert processor.check_triggers("raw_to_note")

    def test_episode_to_belief_time_triggers(self):
        """Old unprocessed episodes trigger episode_to_belief."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            time_threshold_hours=48,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=72)
        ep.emotional_arousal = 0.1
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        assert processor.check_triggers("episode_to_belief")

    def test_episode_to_goal_time_triggers(self):
        """Old unprocessed episodes trigger episode_to_goal."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_goal",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=30)
        ep.emotional_arousal = 0.0
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_goal": config})
        assert processor.check_triggers("episode_to_goal")

    def test_episode_to_relationship_time_triggers(self):
        """Old unprocessed episodes trigger episode_to_relationship."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_relationship",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=36)
        ep.emotional_arousal = 0.0
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_relationship": config})
        assert processor.check_triggers("episode_to_relationship")

    def test_episode_to_drive_time_triggers(self):
        """Old unprocessed episodes trigger episode_to_drive."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_drive",
            quantity_threshold=100,
            time_threshold_hours=24,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=25)
        ep.emotional_arousal = 0.0
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_drive": config})
        assert processor.check_triggers("episode_to_drive")

    def test_belief_to_value_time_triggers(self):
        """Old unprocessed beliefs trigger belief_to_value."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="belief_to_value",
            quantity_threshold=100,
            time_threshold_hours=48,
        )
        now = datetime.now(timezone.utc)
        belief = MagicMock()
        belief.processed = False
        belief.created_at = now - timedelta(hours=72)
        mock_stack.get_beliefs.return_value = [belief]

        processor = _make_processor(mock_stack, configs={"belief_to_value": config})
        assert processor.check_triggers("belief_to_value")


# =============================================================================
# check_triggers — valence/arousal wiring
# =============================================================================


class TestCheckTriggersValenceBased:
    """check_triggers() computes emotional arousal and passes it through."""

    def test_high_arousal_episode_triggers_consolidation(self):
        """A single high-arousal episode triggers episode_to_belief even at count=1."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=0.7,  # Low threshold
            time_threshold_hours=0,  # Disable time trigger
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now
        ep.emotional_arousal = 0.9  # High arousal
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        assert processor.check_triggers("episode_to_belief")

    def test_cumulative_arousal_across_episodes(self):
        """Multiple moderate-arousal episodes trigger when summed."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_goal",
            quantity_threshold=100,
            valence_threshold=1.5,
            time_threshold_hours=0,
        )
        now = datetime.now(timezone.utc)
        episodes = []
        for arousal in [0.4, 0.5, 0.3, 0.5]:
            ep = MagicMock()
            ep.processed = False
            ep.created_at = now
            ep.emotional_arousal = arousal
            episodes.append(ep)
        mock_stack.get_episodes.return_value = episodes

        processor = _make_processor(mock_stack, configs={"episode_to_goal": config})
        assert processor.check_triggers("episode_to_goal")  # 0.4+0.5+0.3+0.5 = 1.7 >= 1.5

    def test_low_arousal_below_threshold_does_not_trigger(self):
        """Low-arousal episodes below threshold do not trigger."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=3.0,
            time_threshold_hours=0,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now
        ep.emotional_arousal = 0.2
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        assert not processor.check_triggers("episode_to_belief")

    def test_arousal_wired_for_episode_to_drive(self):
        """Episode-to-drive also uses arousal-based triggers."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_drive",
            quantity_threshold=100,
            valence_threshold=0.8,
            time_threshold_hours=0,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now
        ep.emotional_arousal = 0.95
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_drive": config})
        assert processor.check_triggers("episode_to_drive")


# =============================================================================
# check_triggers — quantity still works (regression)
# =============================================================================


class TestCheckTriggersQuantityRegression:
    """Verify existing quantity-based triggers still work unchanged."""

    def test_raw_quantity_still_triggers(self):
        mock_stack = _make_mock_stack()
        raws = [MagicMock() for _ in range(11)]
        # Give them recent timestamps so time doesn't trigger
        now = datetime.now(timezone.utc)
        for r in raws:
            r.captured_at = now
        mock_stack._backend.list_raw.return_value = raws
        processor = _make_processor(mock_stack)
        assert processor.check_triggers("raw_to_episode")

    def test_episode_quantity_still_triggers(self):
        mock_stack = _make_mock_stack()
        now = datetime.now(timezone.utc)
        episodes = []
        for _ in range(6):
            ep = MagicMock()
            ep.processed = False
            ep.created_at = now
            ep.emotional_arousal = 0.0
            episodes.append(ep)
        mock_stack.get_episodes.return_value = episodes
        processor = _make_processor(mock_stack)
        assert processor.check_triggers("episode_to_belief")

    def test_belief_quantity_still_triggers(self):
        mock_stack = _make_mock_stack()
        now = datetime.now(timezone.utc)
        beliefs = []
        for _ in range(6):
            b = MagicMock()
            b.processed = False
            b.created_at = now
            beliefs.append(b)
        mock_stack.get_beliefs.return_value = beliefs
        processor = _make_processor(mock_stack)
        assert processor.check_triggers("belief_to_value")

    def test_below_quantity_no_trigger(self):
        """Below quantity, without time or valence, should not trigger."""
        mock_stack = _make_mock_stack()
        now = datetime.now(timezone.utc)
        raw = MagicMock()
        raw.captured_at = now
        mock_stack._backend.list_raw.return_value = [raw]  # 1 < 10
        processor = _make_processor(mock_stack)
        assert not processor.check_triggers("raw_to_episode")


# =============================================================================
# check_triggers — combined triggers
# =============================================================================


class TestCheckTriggersCombined:
    """Test interaction of multiple trigger signals."""

    def test_time_and_valence_either_sufficient(self):
        """Either time OR valence alone is sufficient to trigger."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=5.0,  # Won't be met
            time_threshold_hours=24,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=30)  # Will trigger on time
        ep.emotional_arousal = 0.1  # Low arousal
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        assert processor.check_triggers("episode_to_belief")

    def test_neither_time_nor_valence_nor_count(self):
        """No trigger fires when all signals are below threshold."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=5.0,
            time_threshold_hours=48,
        )
        now = datetime.now(timezone.utc)
        ep = MagicMock()
        ep.processed = False
        ep.created_at = now - timedelta(hours=10)  # Not old enough
        ep.emotional_arousal = 0.1  # Not aroused enough
        mock_stack.get_episodes.return_value = [ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        assert not processor.check_triggers("episode_to_belief")

    def test_processed_episodes_excluded(self):
        """Only unprocessed episodes contribute to trigger signals."""
        mock_stack = _make_mock_stack()
        config = LayerConfig(
            layer_transition="episode_to_belief",
            quantity_threshold=100,
            valence_threshold=0.5,
            time_threshold_hours=0,
        )
        now = datetime.now(timezone.utc)
        processed_ep = MagicMock()
        processed_ep.processed = True
        processed_ep.created_at = now - timedelta(hours=100)
        processed_ep.emotional_arousal = 1.0

        unprocessed_ep = MagicMock()
        unprocessed_ep.processed = False
        unprocessed_ep.created_at = now
        unprocessed_ep.emotional_arousal = 0.1

        mock_stack.get_episodes.return_value = [processed_ep, unprocessed_ep]

        processor = _make_processor(mock_stack, configs={"episode_to_belief": config})
        # Only the unprocessed ep's arousal (0.1) counts, below 0.5
        assert not processor.check_triggers("episode_to_belief")
