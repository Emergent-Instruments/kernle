"""Tests for belief scope and domain metadata (KEP v3, Issue #166)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.core import compute_priority_score
from kernle.storage.base import Belief
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """SQLite storage for belief scope tests."""
    db_path = tmp_path / "test_belief_scope.db"
    s = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    yield s
    s.close()


def _make_belief(**kwargs) -> Belief:
    """Helper to create a Belief with defaults."""
    defaults = {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "statement": "Test belief",
        "belief_type": "fact",
        "confidence": 0.8,
    }
    defaults.update(kwargs)
    return Belief(**defaults)


# === Dataclass defaults ===


class TestBeliefDataclassDefaults:
    def test_default_belief_scope(self):
        b = _make_belief()
        assert b.belief_scope == "world"

    def test_default_source_domain(self):
        b = _make_belief()
        assert b.source_domain is None

    def test_default_cross_domain_applications(self):
        b = _make_belief()
        assert b.cross_domain_applications is None

    def test_default_abstraction_level(self):
        b = _make_belief()
        assert b.abstraction_level == "specific"

    def test_custom_belief_scope(self):
        b = _make_belief(belief_scope="self")
        assert b.belief_scope == "self"

    def test_custom_source_domain(self):
        b = _make_belief(source_domain="coding")
        assert b.source_domain == "coding"

    def test_custom_cross_domain_applications(self):
        b = _make_belief(cross_domain_applications=["communication", "management"])
        assert b.cross_domain_applications == ["communication", "management"]

    def test_custom_abstraction_level(self):
        b = _make_belief(abstraction_level="universal")
        assert b.abstraction_level == "universal"


# === SQLite round-trip ===


class TestSQLiteBeliefScope:
    def test_save_and_get_default_scope(self, storage):
        belief = _make_belief()
        storage.save_belief(belief)
        beliefs = storage.get_beliefs(limit=10)
        assert len(beliefs) >= 1
        found = next(b for b in beliefs if b.id == belief.id)
        assert found.belief_scope == "world"
        assert found.source_domain is None
        assert found.cross_domain_applications is None
        assert found.abstraction_level == "specific"

    def test_save_and_get_self_scope(self, storage):
        belief = _make_belief(
            statement="I learn best through examples",
            belief_scope="self",
            source_domain="learning",
            cross_domain_applications=["coding", "music"],
            abstraction_level="domain",
        )
        storage.save_belief(belief)
        found = storage.find_belief("I learn best through examples")
        assert found is not None
        assert found.belief_scope == "self"
        assert found.source_domain == "learning"
        assert found.cross_domain_applications == ["coding", "music"]
        assert found.abstraction_level == "domain"

    def test_save_and_get_relational_scope(self, storage):
        belief = _make_belief(
            statement="Sean prefers concise responses",
            belief_scope="relational",
            source_domain="communication",
            abstraction_level="specific",
        )
        storage.save_belief(belief)
        found = storage.find_belief("Sean prefers concise responses")
        assert found is not None
        assert found.belief_scope == "relational"
        assert found.source_domain == "communication"

    def test_save_and_get_universal(self, storage):
        belief = _make_belief(
            statement="Breaking changes require careful migration",
            belief_scope="world",
            source_domain="coding",
            abstraction_level="universal",
        )
        storage.save_belief(belief)
        found = storage.find_belief("Breaking changes require careful migration")
        assert found is not None
        assert found.abstraction_level == "universal"

    def test_get_belief_by_id_has_scope(self, storage):
        belief = _make_belief(
            belief_scope="self",
            source_domain="coding",
        )
        storage.save_belief(belief)
        found = storage.get_belief(belief.id)
        assert found is not None
        assert found.belief_scope == "self"
        assert found.source_domain == "coding"

    def test_batch_save_with_scope(self, storage):
        beliefs = [
            _make_belief(
                statement=f"Batch belief {i}",
                belief_scope="self" if i % 2 == 0 else "world",
                source_domain="testing",
            )
            for i in range(3)
        ]
        ids = storage.save_beliefs_batch(beliefs)
        assert len(ids) == 3

        all_beliefs = storage.get_beliefs(limit=10)
        for i, b_id in enumerate(ids):
            found = next(b for b in all_beliefs if b.id == b_id)
            expected_scope = "self" if i % 2 == 0 else "world"
            assert found.belief_scope == expected_scope
            assert found.source_domain == "testing"


# === Priority scoring ===


class TestBeliefScopePriority:
    def test_self_scope_priority_boost(self):
        """Self-scoped beliefs get a +0.05 priority boost."""
        world_belief = _make_belief(belief_scope="world", confidence=0.8)
        self_belief = _make_belief(belief_scope="self", confidence=0.8)

        world_score = compute_priority_score("belief", world_belief)
        self_score = compute_priority_score("belief", self_belief)

        assert self_score == pytest.approx(world_score + 0.05, abs=1e-10)

    def test_relational_scope_no_boost(self):
        """Relational beliefs get no priority boost."""
        world_belief = _make_belief(belief_scope="world", confidence=0.8)
        rel_belief = _make_belief(belief_scope="relational", confidence=0.8)

        world_score = compute_priority_score("belief", world_belief)
        rel_score = compute_priority_score("belief", rel_belief)

        assert rel_score == pytest.approx(world_score, abs=1e-10)

    def test_self_scope_priority_capped_at_1(self):
        """Self-scope boost should not exceed 1.0."""
        high_conf_belief = _make_belief(belief_scope="self", confidence=1.0)
        score = compute_priority_score("belief", high_conf_belief)
        assert score <= 1.0

    def test_non_belief_unaffected(self):
        """Non-belief memory types are not affected by belief_scope logic."""
        from kernle.storage.base import Episode

        episode = Episode(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            objective="test",
            outcome="test",
        )
        score = compute_priority_score("episode", episode)
        # Should not raise even though Episode has no belief_scope
        assert 0.0 <= score <= 1.0


# === Decay behavior ===


class TestBeliefScopeDecay:
    def test_self_scope_uses_value_decay(self, storage):
        """Self-scoped beliefs should decay slower, like values."""
        from kernle import Kernle

        k = Kernle(agent_id="test_agent", storage=storage)

        old_date = datetime.now(timezone.utc) - timedelta(days=90)

        self_belief = _make_belief(
            belief_scope="self",
            confidence=0.9,
            created_at=old_date,
        )
        world_belief = _make_belief(
            belief_scope="world",
            confidence=0.9,
            created_at=old_date,
        )

        self_conf = k.get_confidence_with_decay(self_belief, "belief")
        world_conf = k.get_confidence_with_decay(world_belief, "belief")

        # Self-scoped belief should have higher effective confidence (slower decay)
        assert self_conf > world_conf

    def test_world_scope_uses_standard_decay(self, storage):
        """World-scoped beliefs should use standard belief decay."""
        from kernle import Kernle

        k = Kernle(agent_id="test_agent", storage=storage)

        old_date = datetime.now(timezone.utc) - timedelta(days=30)

        world_belief = _make_belief(
            belief_scope="world",
            confidence=0.9,
            created_at=old_date,
        )

        conf = k.get_confidence_with_decay(world_belief, "belief")
        # Should have some decay after 30 days with standard config
        assert conf <= 0.9


# === Migration ===


class TestBeliefScopeMigration:
    def test_new_columns_exist(self, storage):
        """New belief scope columns should exist after migration."""
        import sqlite3

        conn = sqlite3.connect(str(storage.db_path))
        cols = conn.execute("PRAGMA table_info(beliefs)").fetchall()
        col_names = {c[1] for c in cols}
        conn.close()

        assert "belief_scope" in col_names
        assert "source_domain" in col_names
        assert "cross_domain_applications" in col_names
        assert "abstraction_level" in col_names
