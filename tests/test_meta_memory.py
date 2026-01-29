"""Tests for meta-memory operations.

Tests memory about memory functionality:
- Confidence tracking
- Source attribution
- Memory verification
- Lineage tracking
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Relationship,
    SourceType,
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
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    return SQLiteStorage(agent_id="test-agent", db_path=temp_db)


class TestMetaMemoryFields:
    """Test that meta-memory fields are properly saved and loaded."""

    def test_episode_meta_fields(self, storage):
        """Episode should have meta-memory fields."""
        episode = Episode(
            id="ep-meta-1",
            agent_id="test-agent",
            objective="Test meta-memory",
            outcome="Success",
            confidence=0.9,
            source_type="direct_experience",
            source_episodes=["ep-prev-1", "ep-prev-2"],
            derived_from=["belief:b1", "note:n1"],
            verification_count=2,
        )

        storage.save_episode(episode)
        retrieved = storage.get_episode("ep-meta-1")

        assert retrieved is not None
        assert retrieved.confidence == 0.9
        assert retrieved.source_type == "direct_experience"
        assert retrieved.source_episodes == ["ep-prev-1", "ep-prev-2"]
        assert retrieved.derived_from == ["belief:b1", "note:n1"]
        assert retrieved.verification_count == 2

    def test_belief_meta_fields(self, storage):
        """Belief should have meta-memory fields."""
        belief = Belief(
            id="b-meta-1",
            agent_id="test-agent",
            statement="Testing is good",
            confidence=0.7,
            source_type="inference",
            source_episodes=["ep-1"],
        )

        storage.save_belief(belief)
        beliefs = storage.get_beliefs()

        assert len(beliefs) == 1
        assert beliefs[0].confidence == 0.7
        assert beliefs[0].source_type == "inference"
        assert beliefs[0].source_episodes == ["ep-1"]

    def test_value_meta_fields(self, storage):
        """Value should have meta-memory fields."""
        value = Value(
            id="v-meta-1",
            agent_id="test-agent",
            name="Quality",
            statement="Quality matters",
            confidence=0.95,
            source_type="told_by_agent",
        )

        storage.save_value(value)
        values = storage.get_values()

        assert len(values) == 1
        assert values[0].confidence == 0.95
        assert values[0].source_type == "told_by_agent"

    def test_goal_meta_fields(self, storage):
        """Goal should have meta-memory fields."""
        goal = Goal(
            id="g-meta-1",
            agent_id="test-agent",
            title="Complete meta-memory",
            confidence=0.8,
            source_type="direct_experience",
        )

        storage.save_goal(goal)
        goals = storage.get_goals()

        assert len(goals) == 1
        assert goals[0].confidence == 0.8
        assert goals[0].source_type == "direct_experience"

    def test_note_meta_fields(self, storage):
        """Note should have meta-memory fields."""
        note = Note(
            id="n-meta-1",
            agent_id="test-agent",
            content="Meta-memory note",
            confidence=0.6,
            source_type="consolidation",
            derived_from=["episode:ep-1", "belief:b-1"],
        )

        storage.save_note(note)
        notes = storage.get_notes()

        assert len(notes) == 1
        assert notes[0].confidence == 0.6
        assert notes[0].source_type == "consolidation"
        assert notes[0].derived_from == ["episode:ep-1", "belief:b-1"]

    def test_drive_meta_fields(self, storage):
        """Drive should have meta-memory fields."""
        drive = Drive(
            id="d-meta-1",
            agent_id="test-agent",
            drive_type="curiosity",
            intensity=0.8,
            confidence=0.75,
            source_type="inference",
        )

        storage.save_drive(drive)
        drives = storage.get_drives()

        assert len(drives) == 1
        assert drives[0].confidence == 0.75
        assert drives[0].source_type == "inference"

    def test_relationship_meta_fields(self, storage):
        """Relationship should have meta-memory fields."""
        rel = Relationship(
            id="r-meta-1",
            agent_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            confidence=0.85,
            source_type="direct_experience",
            source_episodes=["ep-meeting-1"],
        )

        storage.save_relationship(rel)
        rels = storage.get_relationships()

        assert len(rels) == 1
        assert rels[0].confidence == 0.85
        assert rels[0].source_type == "direct_experience"
        assert rels[0].source_episodes == ["ep-meeting-1"]


class TestGetMemory:
    """Test the get_memory method."""

    def test_get_memory_episode(self, storage):
        """Should retrieve episode by type and ID."""
        storage.save_episode(
            Episode(id="ep-get-1", agent_id="test-agent", objective="Test", outcome="OK")
        )

        memory = storage.get_memory("episode", "ep-get-1")
        assert memory is not None
        assert memory.objective == "Test"

    def test_get_memory_belief(self, storage):
        """Should retrieve belief by type and ID."""
        storage.save_belief(Belief(id="b-get-1", agent_id="test-agent", statement="Test belief"))

        memory = storage.get_memory("belief", "b-get-1")
        assert memory is not None
        assert memory.statement == "Test belief"

    def test_get_memory_not_found(self, storage):
        """Should return None for non-existent memory."""
        memory = storage.get_memory("episode", "nonexistent")
        assert memory is None

    def test_get_memory_invalid_type(self, storage):
        """Should return None for invalid memory type."""
        memory = storage.get_memory("invalid_type", "id")
        assert memory is None


class TestUpdateMemoryMeta:
    """Test updating meta-memory fields."""

    def test_update_confidence(self, storage):
        """Should update confidence field."""
        storage.save_episode(
            Episode(
                id="ep-upd-1", agent_id="test-agent", objective="Test", outcome="OK", confidence=0.5
            )
        )

        result = storage.update_memory_meta("episode", "ep-upd-1", confidence=0.9)

        assert result is True

        episode = storage.get_episode("ep-upd-1")
        assert episode.confidence == 0.9

    def test_update_source_type(self, storage):
        """Should update source_type field."""
        storage.save_belief(
            Belief(
                id="b-upd-1",
                agent_id="test-agent",
                statement="Test",
                source_type="direct_experience",
            )
        )

        result = storage.update_memory_meta("belief", "b-upd-1", source_type="inference")

        assert result is True

        belief = storage.get_memory("belief", "b-upd-1")
        assert belief.source_type == "inference"

    def test_update_verification_count(self, storage):
        """Should update verification_count."""
        storage.save_note(
            Note(id="n-upd-1", agent_id="test-agent", content="Test note", verification_count=0)
        )

        storage.update_memory_meta(
            "note", "n-upd-1", verification_count=3, last_verified=datetime.now(timezone.utc)
        )

        note = storage.get_memory("note", "n-upd-1")
        assert note.verification_count == 3
        assert note.last_verified is not None

    def test_update_derived_from(self, storage):
        """Should update derived_from list."""
        storage.save_belief(Belief(id="b-upd-2", agent_id="test-agent", statement="Derived belief"))

        storage.update_memory_meta("belief", "b-upd-2", derived_from=["episode:ep-1", "note:n-1"])

        belief = storage.get_memory("belief", "b-upd-2")
        assert belief.derived_from == ["episode:ep-1", "note:n-1"]

    def test_update_nonexistent_memory(self, storage):
        """Should return False for non-existent memory."""
        result = storage.update_memory_meta("episode", "nonexistent", confidence=0.5)
        assert result is False


class TestGetMemoriesByConfidence:
    """Test filtering memories by confidence threshold."""

    def test_get_low_confidence_memories(self, storage):
        """Should get memories below threshold."""
        storage.save_belief(
            Belief(id="b-lo-1", agent_id="test-agent", statement="Low confidence", confidence=0.3)
        )
        storage.save_belief(
            Belief(id="b-hi-1", agent_id="test-agent", statement="High confidence", confidence=0.9)
        )
        storage.save_episode(
            Episode(
                id="ep-lo-1",
                agent_id="test-agent",
                objective="Low conf ep",
                outcome="OK",
                confidence=0.4,
            )
        )

        results = storage.get_memories_by_confidence(0.5, below=True)

        # Should get the low confidence ones
        assert len(results) == 2
        assert all(r.score < 0.5 for r in results)

    def test_get_high_confidence_memories(self, storage):
        """Should get memories above threshold."""
        storage.save_belief(
            Belief(id="b-lo-2", agent_id="test-agent", statement="Low confidence", confidence=0.3)
        )
        storage.save_belief(
            Belief(id="b-hi-2", agent_id="test-agent", statement="High confidence", confidence=0.9)
        )

        results = storage.get_memories_by_confidence(0.5, below=False)

        # Should get only high confidence
        assert len(results) >= 1
        assert all(r.score >= 0.5 for r in results)

    def test_filter_by_memory_type(self, storage):
        """Should filter by memory type."""
        storage.save_belief(
            Belief(id="b-filter-1", agent_id="test-agent", statement="Test", confidence=0.3)
        )
        storage.save_episode(
            Episode(
                id="ep-filter-1",
                agent_id="test-agent",
                objective="Test",
                outcome="OK",
                confidence=0.3,
            )
        )

        results = storage.get_memories_by_confidence(0.5, below=True, memory_types=["belief"])

        assert all(r.record_type == "belief" for r in results)


class TestGetMemoriesBySource:
    """Test filtering memories by source type."""

    def test_get_inferred_memories(self, storage):
        """Should get memories with inference source."""
        storage.save_belief(
            Belief(
                id="b-inf-1",
                agent_id="test-agent",
                statement="Inferred belief",
                source_type="inference",
            )
        )
        storage.save_belief(
            Belief(
                id="b-dir-1",
                agent_id="test-agent",
                statement="Direct belief",
                source_type="direct_experience",
            )
        )

        results = storage.get_memories_by_source("inference")

        assert len(results) >= 1
        assert any(
            r.record.statement == "Inferred belief" for r in results if r.record_type == "belief"
        )

    def test_get_consolidated_memories(self, storage):
        """Should get memories from consolidation."""
        storage.save_note(
            Note(
                id="n-cons-1",
                agent_id="test-agent",
                content="Consolidated note",
                source_type="consolidation",
            )
        )

        results = storage.get_memories_by_source("consolidation", memory_types=["note"])

        assert len(results) >= 1


class TestSourceTypes:
    """Test SourceType enum."""

    def test_source_type_values(self):
        """SourceType should have expected values."""
        assert SourceType.DIRECT_EXPERIENCE.value == "direct_experience"
        assert SourceType.INFERENCE.value == "inference"
        assert SourceType.TOLD_BY_AGENT.value == "told_by_agent"
        assert SourceType.CONSOLIDATION.value == "consolidation"
        assert SourceType.UNKNOWN.value == "unknown"


class TestConfidenceHistory:
    """Test confidence history tracking."""

    def test_save_confidence_history(self, storage):
        """Should save and retrieve confidence history."""
        history = [
            {"timestamp": "2024-01-01T00:00:00Z", "old": 0.5, "new": 0.6, "reason": "verified"},
            {"timestamp": "2024-01-02T00:00:00Z", "old": 0.6, "new": 0.8, "reason": "confirmed"},
        ]

        storage.save_belief(
            Belief(
                id="b-hist-1",
                agent_id="test-agent",
                statement="Belief with history",
                confidence_history=history,
            )
        )

        belief = storage.get_memory("belief", "b-hist-1")
        assert belief.confidence_history is not None
        assert len(belief.confidence_history) == 2
        assert belief.confidence_history[0]["reason"] == "verified"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
