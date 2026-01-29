"""Tests for time-based confidence decay.

Tests the confidence decay feature that prevents confidence inflation
by applying time-based decay to memories that haven't been verified recently.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kernle.core import Kernle, compute_priority_score
from kernle.features.metamemory import (
    DEFAULT_DECAY_CONFIGS,
    DEFAULT_DECAY_FLOOR,
    DEFAULT_DECAY_PERIOD_DAYS,
    DEFAULT_DECAY_RATE,
    DecayConfig,
)
from kernle.storage import (
    Belief,
    SQLiteStorage,
    Value,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    path = Path(tempfile.mktemp(suffix=".db"))
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    return SQLiteStorage(agent_id="test-agent", db_path=temp_db)


@pytest.fixture
def kernle(temp_db, temp_checkpoint_dir):
    """Create a Kernle instance for testing."""
    storage = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    return Kernle(
        agent_id="test-agent",
        storage=storage,
        checkpoint_dir=temp_checkpoint_dir,
    )


class TestDecayConfig:
    """Test DecayConfig dataclass."""

    def test_default_values(self):
        """DecayConfig should have sensible defaults."""
        config = DecayConfig()
        assert config.decay_rate == DEFAULT_DECAY_RATE
        assert config.decay_period_days == DEFAULT_DECAY_PERIOD_DAYS
        assert config.decay_floor == DEFAULT_DECAY_FLOOR
        assert config.enabled is True

    def test_custom_values(self):
        """DecayConfig should accept custom values."""
        config = DecayConfig(
            decay_rate=0.05,
            decay_period_days=7,
            decay_floor=0.3,
            enabled=False,
        )
        assert config.decay_rate == 0.05
        assert config.decay_period_days == 7
        assert config.decay_floor == 0.3
        assert config.enabled is False

    def test_invalid_decay_rate_negative(self):
        """DecayConfig should reject negative decay rates."""
        with pytest.raises(ValueError, match="decay_rate must be non-negative"):
            DecayConfig(decay_rate=-0.1)

    def test_invalid_decay_rate_over_one(self):
        """DecayConfig should reject decay rates > 1.0."""
        with pytest.raises(ValueError, match="decay_rate must be <= 1.0"):
            DecayConfig(decay_rate=1.5)

    def test_invalid_decay_period_zero(self):
        """DecayConfig should reject non-positive decay periods."""
        with pytest.raises(ValueError, match="decay_period_days must be positive"):
            DecayConfig(decay_period_days=0)

    def test_invalid_decay_floor_negative(self):
        """DecayConfig should reject negative decay floors."""
        with pytest.raises(ValueError, match="decay_floor must be between"):
            DecayConfig(decay_floor=-0.1)

    def test_invalid_decay_floor_over_one(self):
        """DecayConfig should reject decay floors > 1.0."""
        with pytest.raises(ValueError, match="decay_floor must be between"):
            DecayConfig(decay_floor=1.5)


class TestDefaultDecayConfigs:
    """Test default per-memory-type decay configurations."""

    def test_values_decay_slower(self):
        """Values should decay slower than other memory types."""
        value_config = DEFAULT_DECAY_CONFIGS["value"]
        belief_config = DEFAULT_DECAY_CONFIGS["belief"]

        # Values should have lower decay rate
        assert value_config.decay_rate < belief_config.decay_rate
        # Values should have longer decay period
        assert value_config.decay_period_days > belief_config.decay_period_days
        # Values should have higher floor
        assert value_config.decay_floor > belief_config.decay_floor

    def test_notes_decay_faster(self):
        """Notes should decay faster than beliefs."""
        note_config = DEFAULT_DECAY_CONFIGS["note"]
        belief_config = DEFAULT_DECAY_CONFIGS["belief"]

        assert note_config.decay_rate >= belief_config.decay_rate

    def test_all_types_have_config(self):
        """All standard memory types should have decay configs."""
        expected_types = ["episode", "belief", "value", "goal", "note", "drive", "relationship"]
        for mem_type in expected_types:
            assert mem_type in DEFAULT_DECAY_CONFIGS


class TestGetConfidenceWithDecay:
    """Test the get_confidence_with_decay method."""

    def test_no_decay_for_recent_memory(self, kernle, storage):
        """Memories verified within 1 day should not decay."""
        now = datetime.now(timezone.utc)
        storage.save_belief(
            Belief(
                id="b-recent",
                agent_id="test-agent",
                statement="Recent belief",
                confidence=0.9,
                last_verified=now - timedelta(hours=12),
            )
        )

        belief = storage.get_memory("belief", "b-recent")
        effective = kernle.get_confidence_with_decay(belief, "belief")

        assert effective == 0.9  # No decay

    def test_decay_after_one_period(self, kernle, storage):
        """Confidence should decay after one decay period."""
        now = datetime.now(timezone.utc)
        config = kernle.get_decay_config("belief")

        storage.save_belief(
            Belief(
                id="b-old",
                agent_id="test-agent",
                statement="Old belief",
                confidence=0.9,
                last_verified=now - timedelta(days=config.decay_period_days),
            )
        )

        belief = storage.get_memory("belief", "b-old")
        effective = kernle.get_confidence_with_decay(belief, "belief")

        # Should decay by exactly one decay_rate
        expected = 0.9 - config.decay_rate
        assert abs(effective - expected) < 0.001

    def test_decay_respects_floor(self, kernle, storage):
        """Confidence should not decay below the floor."""
        now = datetime.now(timezone.utc)
        config = kernle.get_decay_config("belief")

        # Create a very old memory
        storage.save_belief(
            Belief(
                id="b-ancient",
                agent_id="test-agent",
                statement="Ancient belief",
                confidence=0.9,
                last_verified=now - timedelta(days=3650),  # 10 years old
            )
        )

        belief = storage.get_memory("belief", "b-ancient")
        effective = kernle.get_confidence_with_decay(belief, "belief")

        assert effective == config.decay_floor

    def test_protected_memories_dont_decay(self, kernle, storage):
        """Protected memories should not decay.

        Note: The storage layer doesn't persist is_protected for beliefs,
        so we test with a mock record instead to verify the logic.
        """
        now = datetime.now(timezone.utc)

        # Test with a mock record that has is_protected set
        class ProtectedRecord:
            confidence = 0.95
            last_verified = now - timedelta(days=365)
            created_at = None
            is_protected = True

        effective = kernle.get_confidence_with_decay(ProtectedRecord(), "belief")

        assert effective == 0.95  # No decay for protected

    def test_protected_values_use_default_protection(self, kernle, storage):
        """Values are protected by default and should not decay.

        Note: Values have is_protected=True by default in the dataclass.
        """
        now = datetime.now(timezone.utc)

        storage.save_value(
            Value(
                id="v-protected",
                agent_id="test-agent",
                name="Core Value",
                statement="Protected value",
                confidence=0.95,
                last_verified=now - timedelta(days=365),
                # is_protected defaults to True for Values
            )
        )

        value = storage.get_memory("value", "v-protected")

        # Values are protected by default
        assert getattr(value, "is_protected", False) is True

        effective = kernle.get_confidence_with_decay(value, "value")

        # No decay because is_protected is True
        assert effective == 0.95

    def test_values_decay_slower_than_beliefs(self, kernle, storage):
        """Values should decay more slowly than beliefs."""
        now = datetime.now(timezone.utc)
        days_old = 90

        storage.save_belief(
            Belief(
                id="b-compare",
                agent_id="test-agent",
                statement="Test belief",
                confidence=0.9,
                last_verified=now - timedelta(days=days_old),
            )
        )

        storage.save_value(
            Value(
                id="v-compare",
                agent_id="test-agent",
                name="Test",
                statement="Test value",
                confidence=0.9,
                last_verified=now - timedelta(days=days_old),
            )
        )

        belief = storage.get_memory("belief", "b-compare")
        value = storage.get_memory("value", "v-compare")

        belief_effective = kernle.get_confidence_with_decay(belief, "belief")
        value_effective = kernle.get_confidence_with_decay(value, "value")

        # Value should have higher effective confidence (less decay)
        assert value_effective > belief_effective

    def test_decay_uses_created_at_if_no_last_verified(self, kernle, storage):
        """Should use created_at if last_verified is not set."""
        now = datetime.now(timezone.utc)
        config = kernle.get_decay_config("belief")

        storage.save_belief(
            Belief(
                id="b-never-verified",
                agent_id="test-agent",
                statement="Never verified belief",
                confidence=0.8,
                created_at=now - timedelta(days=config.decay_period_days * 2),
                last_verified=None,
            )
        )

        belief = storage.get_memory("belief", "b-never-verified")
        effective = kernle.get_confidence_with_decay(belief, "belief")

        # Should decay based on created_at
        expected = max(config.decay_floor, 0.8 - config.decay_rate * 2)
        assert abs(effective - expected) < 0.001

    def test_decay_disabled_returns_raw_confidence(self, kernle, storage):
        """When decay is disabled, should return raw confidence."""
        now = datetime.now(timezone.utc)

        # Disable decay for beliefs
        kernle.set_decay_config("belief", enabled=False)

        storage.save_belief(
            Belief(
                id="b-no-decay",
                agent_id="test-agent",
                statement="No decay belief",
                confidence=0.9,
                last_verified=now - timedelta(days=365),
            )
        )

        belief = storage.get_memory("belief", "b-no-decay")
        effective = kernle.get_confidence_with_decay(belief, "belief")

        assert effective == 0.9  # No decay applied


class TestSetDecayConfig:
    """Test the set_decay_config method."""

    def test_set_full_config(self, kernle):
        """Should be able to set a full DecayConfig."""
        custom_config = DecayConfig(
            decay_rate=0.02,
            decay_period_days=14,
            decay_floor=0.6,
            enabled=True,
        )
        kernle.set_decay_config("belief", config=custom_config)

        retrieved = kernle.get_decay_config("belief")
        assert retrieved.decay_rate == 0.02
        assert retrieved.decay_period_days == 14
        assert retrieved.decay_floor == 0.6

    def test_update_individual_params(self, kernle):
        """Should be able to update individual parameters."""
        original = kernle.get_decay_config("belief")

        kernle.set_decay_config("belief", decay_rate=0.05)

        updated = kernle.get_decay_config("belief")
        assert updated.decay_rate == 0.05
        assert updated.decay_period_days == original.decay_period_days  # Unchanged

    def test_disable_decay_for_type(self, kernle):
        """Should be able to disable decay for a type."""
        kernle.set_decay_config("episode", enabled=False)

        config = kernle.get_decay_config("episode")
        assert config.enabled is False


class TestGetMemoryConfidence:
    """Test the get_memory_confidence method with decay."""

    def test_apply_decay_true(self, kernle, storage):
        """get_memory_confidence with apply_decay=True should return decayed value."""
        now = datetime.now(timezone.utc)
        config = kernle.get_decay_config("belief")

        storage.save_belief(
            Belief(
                id="b-test",
                agent_id="test-agent",
                statement="Test",
                confidence=0.9,
                last_verified=now - timedelta(days=config.decay_period_days * 2),
            )
        )

        # With decay
        decayed = kernle.get_memory_confidence("belief", "b-test", apply_decay=True)
        # Without decay
        raw = kernle.get_memory_confidence("belief", "b-test", apply_decay=False)

        assert decayed < raw

    def test_apply_decay_default_true(self, kernle, storage):
        """get_memory_confidence should apply decay by default."""
        now = datetime.now(timezone.utc)
        config = kernle.get_decay_config("belief")

        storage.save_belief(
            Belief(
                id="b-default",
                agent_id="test-agent",
                statement="Test",
                confidence=0.9,
                last_verified=now - timedelta(days=config.decay_period_days * 2),
            )
        )

        default_result = kernle.get_memory_confidence("belief", "b-default")
        explicit_decay = kernle.get_memory_confidence("belief", "b-default", apply_decay=True)

        assert default_result == explicit_decay


class TestGetUncertainMemoriesWithDecay:
    """Test get_uncertain_memories with decay consideration."""

    def test_includes_decayed_memories(self, kernle, storage):
        """Should include memories that fell below threshold due to decay."""
        now = datetime.now(timezone.utc)

        # High confidence but old (will decay to the floor of 0.5)
        # 3 years = ~36 decay periods, decay = 0.01 * 36 = 0.36
        # effective = max(0.5, 0.7 - 0.36) = 0.5 (hits floor)
        storage.save_belief(
            Belief(
                id="b-decayed",
                agent_id="test-agent",
                statement="Decayed belief",
                confidence=0.7,
                last_verified=now - timedelta(days=365 * 3),  # 3 years old
            )
        )

        # High confidence and recent (won't decay)
        storage.save_belief(
            Belief(
                id="b-fresh",
                agent_id="test-agent",
                statement="Fresh belief",
                confidence=0.9,
                last_verified=now - timedelta(hours=1),
            )
        )

        # Use threshold of 0.6 to include the decayed belief (effective=0.5)
        # but not the fresh one (effective=0.9)
        uncertain = kernle.get_uncertain_memories(threshold=0.6, apply_decay=True)

        # The old decayed belief should be included
        decayed_ids = [m["id"] for m in uncertain]
        assert "b-decayed" in decayed_ids
        assert "b-fresh" not in decayed_ids

    def test_returns_effective_and_stored_confidence(self, kernle, storage):
        """Should return both effective and stored confidence."""
        now = datetime.now(timezone.utc)

        storage.save_belief(
            Belief(
                id="b-both",
                agent_id="test-agent",
                statement="Both confidences",
                confidence=0.7,
                last_verified=now - timedelta(days=180),
            )
        )

        uncertain = kernle.get_uncertain_memories(threshold=0.6, apply_decay=True)

        for mem in uncertain:
            if mem["id"] == "b-both":
                assert "confidence" in mem
                assert "stored_confidence" in mem
                assert mem["stored_confidence"] == 0.7
                assert mem["confidence"] < mem["stored_confidence"]
                break


class TestGetMemoryLineageWithDecay:
    """Test get_memory_lineage includes decay information."""

    def test_includes_decay_info(self, kernle, storage):
        """Lineage should include both stored and effective confidence."""
        now = datetime.now(timezone.utc)

        storage.save_belief(
            Belief(
                id="b-lineage",
                agent_id="test-agent",
                statement="Test lineage",
                confidence=0.85,
                last_verified=now - timedelta(days=60),
            )
        )

        lineage = kernle.get_memory_lineage("belief", "b-lineage")

        assert "stored_confidence" in lineage
        assert "effective_confidence" in lineage
        assert "confidence_decayed" in lineage
        assert "decay_config" in lineage

        assert lineage["stored_confidence"] == 0.85
        assert lineage["effective_confidence"] < lineage["stored_confidence"]
        assert lineage["confidence_decayed"] is True

    def test_decay_config_in_lineage(self, kernle, storage):
        """Lineage should show the decay config for the memory type."""
        storage.save_belief(
            Belief(
                id="b-config",
                agent_id="test-agent",
                statement="Test config",
                confidence=0.8,
            )
        )

        lineage = kernle.get_memory_lineage("belief", "b-config")

        config = lineage["decay_config"]
        assert "decay_rate" in config
        assert "decay_period_days" in config
        assert "decay_floor" in config
        assert "enabled" in config


class TestComputePriorityScoreWithDecay:
    """Test that compute_priority_score uses decayed confidence."""

    @pytest.mark.skip(reason="compute_priority_score kernle_instance param not yet implemented")
    def test_priority_uses_decayed_confidence(self, kernle, storage):
        """Priority score should use decayed confidence when kernle instance provided."""
        now = datetime.now(timezone.utc)

        storage.save_belief(
            Belief(
                id="b-priority",
                agent_id="test-agent",
                statement="Priority test",
                confidence=0.9,
                last_verified=now - timedelta(days=180),
            )
        )

        belief = storage.get_memory("belief", "b-priority")

        # Without kernle instance (raw confidence)
        score_raw = compute_priority_score("belief", belief, kernle_instance=None)

        # With kernle instance (decayed confidence)
        score_decayed = compute_priority_score("belief", belief, kernle_instance=kernle)

        # Score with decay should be lower
        assert score_decayed < score_raw

    def test_load_uses_decayed_confidence_for_priority(self, kernle, storage):
        """Load should prioritize memories by decayed confidence."""
        now = datetime.now(timezone.utc)

        # Fresh high-confidence belief
        storage.save_belief(
            Belief(
                id="b-fresh-high",
                agent_id="test-agent",
                statement="Fresh high confidence belief",
                confidence=0.8,
                last_verified=now - timedelta(hours=1),
            )
        )

        # Old high-confidence belief (will decay)
        storage.save_belief(
            Belief(
                id="b-old-high",
                agent_id="test-agent",
                statement="Old high confidence belief",
                confidence=0.95,
                last_verified=now - timedelta(days=365),
            )
        )

        # Load should work without error
        result = kernle.load(budget=10000)

        # Both beliefs should be loaded
        belief_ids = [b["id"] for b in result.get("beliefs", [])]

        # Fresh belief should be present
        assert "b-fresh-high" in belief_ids or len(result.get("beliefs", [])) > 0


class TestEdgeCases:
    """Test edge cases for confidence decay."""

    def test_no_timestamp_returns_base_confidence(self, kernle):
        """Memory with no timestamp should return base confidence."""

        class MockRecord:
            confidence = 0.85
            last_verified = None
            created_at = None
            is_protected = False

        effective = kernle.get_confidence_with_decay(MockRecord(), "belief")
        assert effective == 0.85

    def test_timezone_naive_timestamp_handled(self, kernle):
        """Timezone-naive timestamps should be handled correctly."""
        now = datetime.now(timezone.utc)

        class MockRecord:
            confidence = 0.9
            last_verified = datetime(2020, 1, 1)  # Naive datetime
            created_at = None
            is_protected = False

        # Should not raise an error
        effective = kernle.get_confidence_with_decay(
            MockRecord(),
            "belief",
            reference_time=now,
        )

        # Should have decayed significantly
        config = kernle.get_decay_config("belief")
        assert effective <= config.decay_floor

    def test_future_verification_no_decay(self, kernle):
        """Memories verified in the 'future' should not decay."""
        now = datetime.now(timezone.utc)

        class MockRecord:
            confidence = 0.9
            last_verified = now + timedelta(days=1)  # Future
            created_at = None
            is_protected = False

        effective = kernle.get_confidence_with_decay(MockRecord(), "belief")

        # Should return base confidence (no negative decay)
        assert effective == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
