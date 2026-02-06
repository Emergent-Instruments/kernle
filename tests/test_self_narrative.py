"""Tests for self-narrative layer (KEP v3 section 9).

Covers:
- Storage: save, get, list, active constraint, deactivation
- Core: create, deactivate old, active retrieval
- Priority scoring for self_narrative type
- Load integration
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.core import Kernle, compute_priority_score
from kernle.storage import SQLiteStorage
from kernle.storage.base import SelfNarrative


@pytest.fixture
def storage(tmp_path):
    """SQLite storage instance for self-narrative tests."""
    db_path = tmp_path / "test_narrative.db"
    s = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def kernle(tmp_path):
    """Kernle instance for self-narrative tests."""
    db_path = tmp_path / "test_narrative.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    s = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    k = Kernle(agent_id="test_agent", storage=s, checkpoint_dir=checkpoint_dir)
    yield k
    s.close()


# === Storage Tests ===


class TestSelfNarrativeStorage:
    """Test storage-level self-narrative operations."""

    def test_save_and_get(self, storage):
        narrative = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="I am an agent focused on learning and growth.",
            narrative_type="identity",
            key_themes=["learning", "growth"],
            unresolved_tensions=["speed vs quality"],
            created_at=datetime.now(timezone.utc),
        )
        result_id = storage.save_self_narrative(narrative)
        assert result_id == narrative.id

        retrieved = storage.get_self_narrative(narrative.id)
        assert retrieved is not None
        assert retrieved.content == "I am an agent focused on learning and growth."
        assert retrieved.narrative_type == "identity"
        assert retrieved.key_themes == ["learning", "growth"]
        assert retrieved.unresolved_tensions == ["speed vs quality"]
        assert retrieved.is_active is True

    def test_get_nonexistent(self, storage):
        result = storage.get_self_narrative("nonexistent-id")
        assert result is None

    def test_list_active_only(self, storage):
        # Save two active narratives of different types
        n1 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Identity narrative",
            narrative_type="identity",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        n2 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Developmental narrative",
            narrative_type="developmental",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        n3 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Old identity narrative",
            narrative_type="identity",
            is_active=False,
            created_at=datetime.now(timezone.utc),
        )
        storage.save_self_narrative(n1)
        storage.save_self_narrative(n2)
        storage.save_self_narrative(n3)

        # List active only (default)
        active = storage.list_self_narratives("test_agent", active_only=True)
        assert len(active) == 2

        # List all
        all_narratives = storage.list_self_narratives("test_agent", active_only=False)
        assert len(all_narratives) == 3

    def test_list_by_type(self, storage):
        n1 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Identity",
            narrative_type="identity",
            created_at=datetime.now(timezone.utc),
        )
        n2 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Aspirational",
            narrative_type="aspirational",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_self_narrative(n1)
        storage.save_self_narrative(n2)

        identity = storage.list_self_narratives("test_agent", narrative_type="identity")
        assert len(identity) == 1
        assert identity[0].narrative_type == "identity"

        aspirational = storage.list_self_narratives("test_agent", narrative_type="aspirational")
        assert len(aspirational) == 1
        assert aspirational[0].narrative_type == "aspirational"

    def test_deactivate_by_type(self, storage):
        n1 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Old identity 1",
            narrative_type="identity",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        n2 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Old identity 2",
            narrative_type="identity",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        n3 = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Developmental",
            narrative_type="developmental",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        storage.save_self_narrative(n1)
        storage.save_self_narrative(n2)
        storage.save_self_narrative(n3)

        # Deactivate identity type
        count = storage.deactivate_self_narratives("test_agent", "identity")
        assert count == 2

        # Verify identity narratives are inactive
        active_identity = storage.list_self_narratives(
            "test_agent", narrative_type="identity", active_only=True
        )
        assert len(active_identity) == 0

        # Verify developmental narrative is still active
        active_dev = storage.list_self_narratives(
            "test_agent", narrative_type="developmental", active_only=True
        )
        assert len(active_dev) == 1

    def test_supersedes_field(self, storage):
        old_id = str(uuid.uuid4())
        n_old = SelfNarrative(
            id=old_id,
            agent_id="test_agent",
            content="Old narrative",
            narrative_type="identity",
            created_at=datetime.now(timezone.utc),
        )
        n_new = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="New narrative",
            narrative_type="identity",
            supersedes=old_id,
            created_at=datetime.now(timezone.utc),
        )
        storage.save_self_narrative(n_old)
        storage.save_self_narrative(n_new)

        retrieved = storage.get_self_narrative(n_new.id)
        assert retrieved.supersedes == old_id

    def test_deleted_not_returned(self, storage):
        n = SelfNarrative(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Deleted narrative",
            narrative_type="identity",
            deleted=True,
            created_at=datetime.now(timezone.utc),
        )
        storage.save_self_narrative(n)

        assert storage.get_self_narrative(n.id) is None
        assert len(storage.list_self_narratives("test_agent")) == 0


# === Core API Tests ===


class TestSelfNarrativeCore:
    """Test Kernle core API for self-narratives."""

    def test_narrative_save(self, kernle):
        narrative_id = kernle.narrative_save(
            content="I am an agent that values careful reasoning.",
            narrative_type="identity",
            key_themes=["reasoning", "care"],
        )
        assert narrative_id is not None

        active = kernle.narrative_get_active("identity")
        assert active is not None
        assert active.content == "I am an agent that values careful reasoning."
        assert active.key_themes == ["reasoning", "care"]

    def test_narrative_save_deactivates_old(self, kernle):
        # Save first narrative
        first_id = kernle.narrative_save(
            content="First identity",
            narrative_type="identity",
        )

        # Save second narrative of same type
        second_id = kernle.narrative_save(
            content="Updated identity",
            narrative_type="identity",
        )

        assert first_id != second_id

        # Only the new one should be active
        active = kernle.narrative_get_active("identity")
        assert active is not None
        assert active.id == second_id
        assert active.content == "Updated identity"

        # The new one should supersede the old one
        assert active.supersedes == first_id

        # Old one should be inactive
        all_narratives = kernle.narrative_list(narrative_type="identity", active_only=False)
        assert len(all_narratives) == 2
        inactive = [n for n in all_narratives if not n.is_active]
        assert len(inactive) == 1
        assert inactive[0].id == first_id

    def test_narrative_save_different_types_independent(self, kernle):
        kernle.narrative_save(content="Identity", narrative_type="identity")
        kernle.narrative_save(content="Developmental", narrative_type="developmental")

        identity = kernle.narrative_get_active("identity")
        developmental = kernle.narrative_get_active("developmental")

        assert identity is not None
        assert developmental is not None
        assert identity.content == "Identity"
        assert developmental.content == "Developmental"

    def test_narrative_get_active_none(self, kernle):
        result = kernle.narrative_get_active("identity")
        assert result is None

    def test_narrative_list_all(self, kernle):
        kernle.narrative_save(content="First", narrative_type="identity")
        kernle.narrative_save(content="Second", narrative_type="identity")
        kernle.narrative_save(content="Aspirational", narrative_type="aspirational")

        all_narratives = kernle.narrative_list(active_only=False)
        assert len(all_narratives) == 3

        active_only = kernle.narrative_list(active_only=True)
        assert len(active_only) == 2  # One identity + one aspirational

    def test_narrative_save_invalid_type(self, kernle):
        with pytest.raises(ValueError, match="narrative_type"):
            kernle.narrative_save(content="Test", narrative_type="invalid")

    def test_narrative_with_epoch(self, kernle):
        epoch_id = kernle.epoch_create(name="test-epoch")
        kernle.narrative_save(
            content="Narrative during epoch",
            narrative_type="identity",
            epoch_id=epoch_id,
        )

        narrative = kernle.narrative_get_active("identity")
        assert narrative.epoch_id == epoch_id

    def test_narrative_with_tensions(self, kernle):
        kernle.narrative_save(
            content="I balance multiple priorities",
            narrative_type="identity",
            unresolved_tensions=["autonomy vs connection", "speed vs quality"],
        )

        narrative = kernle.narrative_get_active("identity")
        assert narrative.unresolved_tensions == ["autonomy vs connection", "speed vs quality"]


# === Priority Scoring Tests ===


class TestSelfNarrativePriority:
    """Test priority scoring for self-narrative type."""

    def test_self_narrative_base_priority(self):
        from kernle.core import MEMORY_TYPE_PRIORITIES

        assert "self_narrative" in MEMORY_TYPE_PRIORITIES
        assert MEMORY_TYPE_PRIORITIES["self_narrative"] == 0.90

    def test_compute_priority_score(self):
        narrative = SelfNarrative(
            id="test",
            agent_id="test_agent",
            content="Test narrative",
            created_at=datetime.now(timezone.utc),
        )
        score = compute_priority_score("self_narrative", narrative)
        # base_priority * 0.6 + type_factor * 0.4
        # 0.90 * 0.6 + 0.9 * 0.4 = 0.54 + 0.36 = 0.90
        assert score == pytest.approx(0.90, abs=0.01)

    def test_self_narrative_higher_than_beliefs(self):
        narrative = SelfNarrative(
            id="test",
            agent_id="test_agent",
            content="Test narrative",
            created_at=datetime.now(timezone.utc),
        )
        from kernle.storage.base import Belief

        belief = Belief(
            id="test",
            agent_id="test_agent",
            statement="Test belief",
            confidence=0.8,
            created_at=datetime.now(timezone.utc),
        )
        narrative_score = compute_priority_score("self_narrative", narrative)
        belief_score = compute_priority_score("belief", belief)
        assert narrative_score > belief_score


# === Load Integration Tests ===


class TestSelfNarrativeLoad:
    """Test self-narrative integration with load()."""

    def test_load_includes_active_narratives(self, kernle):
        kernle.narrative_save(
            content="I am a helpful assistant",
            narrative_type="identity",
            key_themes=["helpfulness"],
        )

        result = kernle.load(track_access=False)
        assert "self_narrative" in result
        assert len(result["self_narrative"]) == 1
        assert result["self_narrative"][0]["narrative_type"] == "identity"
        assert result["self_narrative"][0]["content"] == "I am a helpful assistant"
        assert result["self_narrative"][0]["key_themes"] == ["helpfulness"]

    def test_load_excludes_inactive_narratives(self, kernle):
        kernle.narrative_save(content="First", narrative_type="identity")
        kernle.narrative_save(content="Second", narrative_type="identity")

        result = kernle.load(track_access=False)
        assert "self_narrative" in result
        assert len(result["self_narrative"]) == 1
        assert result["self_narrative"][0]["content"] == "Second"

    def test_load_multiple_types(self, kernle):
        kernle.narrative_save(content="Identity", narrative_type="identity")
        kernle.narrative_save(content="Aspirational", narrative_type="aspirational")

        result = kernle.load(track_access=False)
        assert "self_narrative" in result
        assert len(result["self_narrative"]) == 2

    def test_load_no_narratives(self, kernle):
        result = kernle.load(track_access=False)
        assert "self_narrative" not in result


# === Schema Migration Tests ===


class TestSelfNarrativeMigration:
    """Test that schema migration creates the table correctly."""

    def test_table_exists_after_init(self, storage):
        """The agent_self_narrative table should exist after storage initialization."""
        import sqlite3

        conn = sqlite3.connect(str(storage.db_path))
        conn.row_factory = sqlite3.Row
        tables = [
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        conn.close()
        assert "agent_self_narrative" in tables

    def test_allowed_tables_includes_narrative(self):
        """agent_self_narrative should be in ALLOWED_TABLES."""
        from kernle.storage.sqlite import ALLOWED_TABLES

        assert "agent_self_narrative" in ALLOWED_TABLES
