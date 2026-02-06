"""Tests for dynamic trust features (KEP v3 section 8.6-8.7)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from kernle.storage.base import (
    DEFAULT_TRUST,
    SELF_TRUST_FLOOR,
    TRUST_DECAY_RATE,
    TRUST_DEPTH_DECAY,
    Episode,
    TrustAssessment,
)


@pytest.fixture
def setup(tmp_path):
    """Create a Kernle instance for dynamic trust testing."""
    db_path = tmp_path / "test_dynamic_trust.db"
    storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    k = Kernle(agent_id="test_agent", storage=storage, checkpoint_dir=checkpoint_dir)
    k.seed_trust()
    yield k, storage
    storage.close()


def _make_episode(
    source_entity,
    outcome_type="success",
    emotional_valence=0.5,
    days_ago=0,
):
    """Helper to create an episode with the given source entity."""
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return Episode(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        objective="test objective",
        outcome="test outcome",
        outcome_type=outcome_type,
        emotional_valence=emotional_valence,
        created_at=created,
        source_entity=source_entity,
    )


class TestDynamicTrustConstants:
    """Tests for dynamic trust constants."""

    def test_default_trust(self):
        assert DEFAULT_TRUST == 0.5

    def test_decay_rate(self):
        assert TRUST_DECAY_RATE == 0.01

    def test_depth_decay(self):
        assert TRUST_DEPTH_DECAY == 0.85

    def test_self_trust_floor(self):
        assert SELF_TRUST_FLOOR == 0.5


class TestComputeDirectTrust:
    """Tests for computing trust from episode history."""

    def test_no_episodes_returns_default(self, setup):
        k, storage = setup
        result = k.compute_direct_trust("unknown-entity")
        assert result["score"] == DEFAULT_TRUST
        assert result["source"] == "default"
        assert result["total"] == 0

    def test_all_positive_episodes(self, setup):
        k, storage = setup
        for i in range(5):
            ep = _make_episode("test-source", outcome_type="success", days_ago=i)
            storage.save_episode(ep)

        result = k.compute_direct_trust("test-source")
        assert result["score"] > 0.9
        assert result["total"] == 5
        assert result["source"] == "computed"

    def test_all_negative_episodes(self, setup):
        k, storage = setup
        for i in range(5):
            ep = _make_episode(
                "bad-source", outcome_type="failure", emotional_valence=-0.5, days_ago=i
            )
            storage.save_episode(ep)

        result = k.compute_direct_trust("bad-source")
        assert result["score"] < 0.1
        assert result["total"] == 5

    def test_mixed_episodes(self, setup):
        k, storage = setup
        # 3 positive, 2 negative, all recent
        for i in range(3):
            ep = _make_episode("mixed-source", outcome_type="success", days_ago=i)
            storage.save_episode(ep)
        for i in range(2):
            ep = _make_episode(
                "mixed-source", outcome_type="failure", emotional_valence=-0.5, days_ago=i
            )
            storage.save_episode(ep)

        result = k.compute_direct_trust("mixed-source")
        assert 0.4 < result["score"] < 0.8
        assert result["total"] == 5

    def test_recency_weighting(self, setup):
        k, storage = setup
        # Old positive episode (60 days ago) and recent negative (today)
        old_pos = _make_episode("recency-test", outcome_type="success", days_ago=60)
        recent_neg = _make_episode(
            "recency-test", outcome_type="failure", emotional_valence=-0.5, days_ago=0
        )
        storage.save_episode(old_pos)
        storage.save_episode(recent_neg)

        result = k.compute_direct_trust("recency-test")
        # Recent negative should dominate over old positive
        assert result["score"] < 0.5

    def test_emotional_valence_fallback(self, setup):
        k, storage = setup
        # Episode with no outcome_type but positive valence
        ep = _make_episode("valence-test", outcome_type=None, emotional_valence=0.8, days_ago=0)
        storage.save_episode(ep)

        result = k.compute_direct_trust("valence-test")
        assert result["score"] > 0.9


class TestTrustDecay:
    """Tests for trust decay toward neutral."""

    def test_high_trust_decays_toward_neutral(self, setup):
        k, storage = setup
        # Set high trust
        k.trust_set("test-entity", domain="general", score=0.9)

        result = k.apply_trust_decay("test-entity", days_since_interaction=50)
        new_score = result["dimensions"]["general"]["score"]
        assert new_score < 0.9
        assert new_score > 0.5  # Should not go below neutral

    def test_low_trust_decays_toward_neutral(self, setup):
        k, storage = setup
        # Set low trust
        k.trust_set("low-entity", domain="general", score=0.1)

        result = k.apply_trust_decay("low-entity", days_since_interaction=50)
        new_score = result["dimensions"]["general"]["score"]
        assert new_score > 0.1
        assert new_score < 0.5  # Should not go above neutral

    def test_neutral_trust_stays_neutral(self, setup):
        k, storage = setup
        k.trust_set("neutral-entity", domain="general", score=0.5)

        result = k.apply_trust_decay("neutral-entity", days_since_interaction=100)
        new_score = result["dimensions"]["general"]["score"]
        assert abs(new_score - 0.5) < 0.01

    def test_decay_factor_capped_at_one(self, setup):
        k, storage = setup
        k.trust_set("cap-entity", domain="general", score=0.9)

        result = k.apply_trust_decay("cap-entity", days_since_interaction=200)
        assert result["decay_factor"] <= 1.0

    def test_self_trust_has_floor(self, setup):
        k, storage = setup
        # Self trust starts at 0.8 from seed
        result = k.apply_trust_decay("self", days_since_interaction=1000)
        new_score = result["dimensions"]["general"]["score"]
        # Self-trust should not go below SELF_TRUST_FLOOR
        assert new_score >= SELF_TRUST_FLOOR

    def test_unknown_entity_returns_error(self, setup):
        k, storage = setup
        result = k.apply_trust_decay("nonexistent", days_since_interaction=10)
        assert "error" in result

    def test_multi_domain_decay(self, setup):
        k, storage = setup
        # Create entity with multiple domains
        assessment = TrustAssessment(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            entity="multi-domain",
            dimensions={
                "general": {"score": 0.9},
                "coding": {"score": 0.3},
            },
        )
        storage.save_trust_assessment(assessment)

        result = k.apply_trust_decay("multi-domain", days_since_interaction=50)
        general = result["dimensions"]["general"]["score"]
        coding = result["dimensions"]["coding"]["score"]
        # Both should move toward 0.5
        assert general < 0.9
        assert coding > 0.3


class TestTransitiveTrust:
    """Tests for transitive trust chains."""

    def test_single_hop(self, setup):
        k, storage = setup
        # stack-owner has 0.95 trust
        result = k.compute_transitive_trust("target", ["stack-owner"], domain="general")
        # 0.95 * 0.85^0 = 0.95
        assert abs(result["score"] - 0.95) < 0.01
        assert len(result["hops"]) == 1

    def test_two_hop_chain(self, setup):
        k, storage = setup
        # stack-owner (0.95) -> self (0.8)
        result = k.compute_transitive_trust("target", ["stack-owner", "self"], domain="general")
        # 0.95 * 1.0 * 0.8 * 0.85 = 0.646
        expected = 0.95 * (0.8 * TRUST_DEPTH_DECAY)
        assert abs(result["score"] - expected) < 0.01
        assert len(result["hops"]) == 2

    def test_depth_decay_per_hop(self, setup):
        k, storage = setup
        # Three entities with 0.9 trust each
        for name in ["a", "b", "c"]:
            k.trust_set(name, score=0.9)

        result = k.compute_transitive_trust("target", ["a", "b", "c"])
        # 0.9 * 0.85^0 * 0.9 * 0.85^1 * 0.9 * 0.85^2
        expected = (0.9 * 1.0) * (0.9 * 0.85) * (0.9 * 0.85**2)
        assert abs(result["score"] - expected) < 0.01

    def test_unknown_entity_in_chain(self, setup):
        k, storage = setup
        result = k.compute_transitive_trust("target", ["unknown-entity"])
        # Unknown entity gets DEFAULT_TRUST (0.5)
        assert abs(result["score"] - DEFAULT_TRUST) < 0.01

    def test_empty_chain(self, setup):
        k, storage = setup
        result = k.compute_transitive_trust("target", [])
        assert result["score"] == 0.0
        assert "error" in result

    def test_chain_with_zero_trust(self, setup):
        k, storage = setup
        # context-injection has 0.0 trust
        result = k.compute_transitive_trust("target", ["context-injection", "stack-owner"])
        assert result["score"] == 0.0

    def test_domain_specific_chain(self, setup):
        k, storage = setup
        # web-search has medical: 0.3
        result = k.compute_transitive_trust("target", ["web-search"], domain="medical")
        assert abs(result["score"] - 0.3) < 0.01


class TestSelfTrustFloor:
    """Tests for self-trust floor computation."""

    def test_no_self_episodes(self, setup):
        k, storage = setup
        result = k.compute_self_trust_floor()
        assert result["floor"] == SELF_TRUST_FLOOR
        assert result["source"] == "default"

    def test_high_accuracy_raises_floor(self, setup):
        k, storage = setup
        # 9 positive, 1 negative -> 90% accuracy
        for i in range(9):
            ep = _make_episode("self", outcome_type="success", days_ago=i)
            storage.save_episode(ep)
        ep = _make_episode("self", outcome_type="failure", emotional_valence=-0.5, days_ago=10)
        storage.save_episode(ep)

        result = k.compute_self_trust_floor()
        assert result["floor"] == 0.9
        assert result["accuracy"] == 0.9
        assert result["source"] == "computed"

    def test_low_accuracy_uses_minimum(self, setup):
        k, storage = setup
        # 2 positive, 8 negative -> 20% accuracy, floor stays at 0.5
        for i in range(2):
            ep = _make_episode("self", outcome_type="success", days_ago=i)
            storage.save_episode(ep)
        for i in range(8):
            ep = _make_episode(
                "self", outcome_type="failure", emotional_valence=-0.5, days_ago=i + 2
            )
            storage.save_episode(ep)

        result = k.compute_self_trust_floor()
        assert result["floor"] == SELF_TRUST_FLOOR
        assert result["accuracy"] == 0.2


class TestTrustCompute:
    """Tests for the trust_compute entry point."""

    def test_compute_unknown_entity(self, setup):
        k, storage = setup
        result = k.trust_compute("unknown")
        assert result["score"] == DEFAULT_TRUST

    def test_compute_self_includes_floor(self, setup):
        k, storage = setup
        result = k.trust_compute("self")
        assert "self_trust_floor" in result

    def test_compute_with_episodes(self, setup):
        k, storage = setup
        for i in range(3):
            ep = _make_episode("compute-test", outcome_type="success", days_ago=i)
            storage.save_episode(ep)

        result = k.trust_compute("compute-test")
        assert result["score"] > 0.9
        assert result["total"] == 3


class TestTrustChain:
    """Tests for trust_chain entry point."""

    def test_chain_entry_point(self, setup):
        k, storage = setup
        result = k.trust_chain("target", ["stack-owner"])
        assert "score" in result
        assert "hops" in result
        assert result["score"] > 0.9


class TestGetEpisodesBySourceEntity:
    """Tests for storage-level episode query by source entity."""

    def test_returns_matching_episodes(self, setup):
        k, storage = setup
        ep1 = _make_episode("entity-a", outcome_type="success")
        ep2 = _make_episode("entity-b", outcome_type="failure", emotional_valence=-0.5)
        ep3 = _make_episode("entity-a", outcome_type="success")
        storage.save_episode(ep1)
        storage.save_episode(ep2)
        storage.save_episode(ep3)

        results = storage.get_episodes_by_source_entity("entity-a")
        assert len(results) == 2
        for ep in results:
            assert ep.source_entity == "entity-a"

    def test_returns_empty_for_unknown(self, setup):
        k, storage = setup
        results = storage.get_episodes_by_source_entity("nonexistent")
        assert results == []

    def test_respects_limit(self, setup):
        k, storage = setup
        for i in range(10):
            ep = _make_episode("limited-entity", outcome_type="success", days_ago=i)
            storage.save_episode(ep)

        results = storage.get_episodes_by_source_entity("limited-entity", limit=3)
        assert len(results) == 3

    def test_excludes_deleted(self, setup):
        k, storage = setup
        ep = _make_episode("del-entity", outcome_type="success")
        storage.save_episode(ep)
        # Soft delete
        storage.forget_memory("episode", ep.id)

        results = storage.get_episodes_by_source_entity("del-entity")
        assert len(results) == 0
