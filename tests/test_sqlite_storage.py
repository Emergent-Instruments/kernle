"""Tests for SQLite storage backend.

Tests local-first SQLite storage with:
- Basic CRUD operations
- Vector search (with sqlite-vec)
- Sync metadata
- Embedding management
"""

import tempfile
from pathlib import Path

import pytest

from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    HashEmbedder,
    Note,
    Relationship,
    SQLiteStorage,
    Value,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    path = Path(tempfile.mktemp(suffix='.db'))
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    storage = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


class TestSQLiteStorageBasics:
    """Basic CRUD operation tests."""

    def test_storage_creation(self, storage, temp_db):
        """Storage creates database and directory."""
        assert temp_db.exists()
        assert storage.agent_id == "test-agent"

    def test_sqlite_vec_available(self, storage):
        """sqlite-vec should be available."""
        assert storage._has_vec is True

    def test_embedder_default(self, storage):
        """Default embedder should be HashEmbedder."""
        assert isinstance(storage._embedder, HashEmbedder)


class TestEpisodes:
    """Episode storage tests."""

    def test_save_and_get_episode(self, storage):
        episode = Episode(
            id="ep-1",
            agent_id="test-agent",
            objective="Test objective",
            outcome="Test outcome",
            outcome_type="success",
            lessons=["lesson 1", "lesson 2"],
            tags=["test", "unit"]
        )

        saved_id = storage.save_episode(episode)
        assert saved_id == "ep-1"

        retrieved = storage.get_episode("ep-1")
        assert retrieved is not None
        assert retrieved.objective == "Test objective"
        assert retrieved.lessons == ["lesson 1", "lesson 2"]
        assert retrieved.tags == ["test", "unit"]

    def test_get_episodes(self, storage):
        # Save multiple episodes
        for i in range(5):
            storage.save_episode(Episode(
                id=f"ep-{i}",
                agent_id="test-agent",
                objective=f"Objective {i}",
                outcome=f"Outcome {i}",
            ))

        episodes = storage.get_episodes(limit=3)
        assert len(episodes) == 3

    def test_episode_auto_id(self, storage):
        episode = Episode(
            id="",  # Empty ID
            agent_id="test-agent",
            objective="Auto ID test",
            outcome="Should generate UUID"
        )
        saved_id = storage.save_episode(episode)
        assert saved_id  # Should have a UUID


class TestNotes:
    """Note storage tests."""

    def test_save_and_get_note(self, storage):
        note = Note(
            id="note-1",
            agent_id="test-agent",
            content="This is a test note",
            note_type="observation",
            speaker="user",
            tags=["important"]
        )

        storage.save_note(note)
        notes = storage.get_notes(limit=1)

        assert len(notes) == 1
        assert notes[0].content == "This is a test note"
        assert notes[0].note_type == "observation"


class TestBeliefs:
    """Belief storage tests."""

    def test_save_and_get_belief(self, storage):
        belief = Belief(
            id="belief-1",
            agent_id="test-agent",
            statement="Testing is important",
            belief_type="principle",
            confidence=0.95
        )

        storage.save_belief(belief)
        beliefs = storage.get_beliefs()

        assert len(beliefs) == 1
        assert beliefs[0].statement == "Testing is important"
        assert beliefs[0].confidence == 0.95

    def test_find_belief(self, storage):
        storage.save_belief(Belief(
            id="b1",
            agent_id="test-agent",
            statement="Unique statement here",
        ))

        found = storage.find_belief("Unique statement here")
        assert found is not None
        assert found.id == "b1"

        not_found = storage.find_belief("Nonexistent")
        assert not_found is None


class TestValues:
    """Value storage tests."""

    def test_save_and_get_value(self, storage):
        value = Value(
            id="val-1",
            agent_id="test-agent",
            name="Reliability",
            statement="I prioritize dependability in my work",
            priority=90
        )

        storage.save_value(value)
        values = storage.get_values()

        assert len(values) == 1
        assert values[0].name == "Reliability"
        assert values[0].priority == 90


class TestGoals:
    """Goal storage tests."""

    def test_save_and_get_goal(self, storage):
        goal = Goal(
            id="goal-1",
            agent_id="test-agent",
            title="Complete project",
            description="Finish all tasks",
            priority="high",
            status="active"
        )

        storage.save_goal(goal)
        goals = storage.get_goals(status="active")

        assert len(goals) == 1
        assert goals[0].title == "Complete project"


class TestDrives:
    """Drive storage tests."""

    def test_save_and_get_drive(self, storage):
        drive = Drive(
            id="drive-1",
            agent_id="test-agent",
            drive_type="curiosity",
            intensity=0.8,
            focus_areas=["learning", "exploration"]
        )

        storage.save_drive(drive)
        drives = storage.get_drives()

        assert len(drives) == 1
        assert drives[0].drive_type == "curiosity"
        assert drives[0].intensity == 0.8

    def test_update_drive(self, storage):
        storage.save_drive(Drive(
            id="d1",
            agent_id="test-agent",
            drive_type="growth",
            intensity=0.5
        ))

        # Update same drive type
        storage.save_drive(Drive(
            id="d2",  # Different ID but same type
            agent_id="test-agent",
            drive_type="growth",
            intensity=0.9
        ))

        drive = storage.get_drive("growth")
        assert drive.intensity == 0.9


class TestRelationships:
    """Relationship storage tests."""

    def test_save_and_get_relationship(self, storage):
        rel = Relationship(
            id="rel-1",
            agent_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            sentiment=0.7,
            interaction_count=5
        )

        storage.save_relationship(rel)
        rels = storage.get_relationships()

        assert len(rels) == 1
        assert rels[0].entity_name == "Alice"
        assert rels[0].sentiment == 0.7

    def test_get_relationship_by_name(self, storage):
        storage.save_relationship(Relationship(
            id="r1",
            agent_id="test-agent",
            entity_name="Bob",
            entity_type="human",
            relationship_type="friend"
        ))

        rel = storage.get_relationship("Bob")
        assert rel is not None
        assert rel.entity_name == "Bob"


class TestVectorSearch:
    """Vector search tests."""

    def test_search_episodes(self, storage):
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Learn Python programming",
            outcome="Mastered basic syntax"
        ))
        storage.save_episode(Episode(
            id="ep2",
            agent_id="test-agent",
            objective="Write documentation",
            outcome="Created user guide"
        ))

        results = storage.search("programming language", limit=5)
        assert len(results) > 0
        # Python-related episode should rank higher
        assert results[0].record_type == "episode"

    def test_search_mixed_types(self, storage):
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Database design",
            outcome="Created schema"
        ))
        storage.save_note(Note(
            id="note1",
            agent_id="test-agent",
            content="SQLite is a great embedded database"
        ))
        storage.save_belief(Belief(
            id="b1",
            agent_id="test-agent",
            statement="Databases should be ACID compliant"
        ))

        results = storage.search("database", limit=10)
        assert len(results) == 3

        # Check we got all types
        types = {r.record_type for r in results}
        assert "episode" in types
        assert "note" in types
        assert "belief" in types

    def test_search_with_type_filter(self, storage):
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Search test",
            outcome="Testing filters"
        ))
        storage.save_note(Note(
            id="note1",
            agent_id="test-agent",
            content="Search test content"
        ))

        results = storage.search("search test", record_types=["note"])
        assert all(r.record_type == "note" for r in results)

    def test_search_scores_ranked(self, storage):
        """Search results should be ranked by similarity."""
        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Machine learning and neural networks"
        ))
        storage.save_note(Note(
            id="n2",
            agent_id="test-agent",
            content="Cooking recipes and kitchen tips"
        ))

        results = storage.search("deep learning AI", limit=5)
        # ML note should rank higher
        assert results[0].record.content == "Machine learning and neural networks"


class TestSyncMetadata:
    """Sync metadata tests."""

    def test_sync_metadata_on_save(self, storage):
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Sync test",
            outcome="Testing"
        ))

        episode = storage.get_episode("ep1")
        assert episode.local_updated_at is not None
        assert episode.cloud_synced_at is None  # Not synced yet
        assert episode.version >= 1
        assert episode.deleted is False

    def test_pending_sync_count(self, storage):
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Test",
            outcome="Test"
        ))
        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test"
        ))

        pending = storage.get_pending_sync_count()
        assert pending >= 2  # At least our 2 records


class TestStats:
    """Stats tests."""

    def test_get_stats(self, storage):
        storage.save_episode(Episode(
            id="ep1", agent_id="test-agent",
            objective="Test", outcome="Test"
        ))
        storage.save_note(Note(
            id="n1", agent_id="test-agent",
            content="Test"
        ))
        storage.save_belief(Belief(
            id="b1", agent_id="test-agent",
            statement="Test"
        ))

        stats = storage.get_stats()
        assert stats["episodes"] == 1
        assert stats["notes"] == 1
        assert stats["beliefs"] == 1
        assert stats["values"] == 0
        assert stats["goals"] == 0


class TestHashEmbedder:
    """Hash embedder tests."""

    def test_embedding_dimension(self):
        embedder = HashEmbedder(dim=384)
        embedding = embedder.embed("test text")
        assert len(embedding) == 384

    def test_embedding_normalized(self):
        embedder = HashEmbedder()
        embedding = embedder.embed("test text")
        norm = sum(x*x for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01  # Should be unit length

    def test_embedding_deterministic(self):
        embedder = HashEmbedder()
        e1 = embedder.embed("same text")
        e2 = embedder.embed("same text")
        assert e1 == e2

    def test_similar_texts_closer(self):
        """Similar texts should have closer embeddings."""
        embedder = HashEmbedder()

        e1 = embedder.embed("machine learning algorithms")
        e2 = embedder.embed("machine learning models")
        e3 = embedder.embed("cooking recipes food")

        # Cosine similarity
        def cos_sim(a, b):
            return sum(x*y for x, y in zip(a, b))

        sim_12 = cos_sim(e1, e2)
        sim_13 = cos_sim(e1, e3)

        assert sim_12 > sim_13  # ML texts more similar


class TestOfflineOperation:
    """Test that storage works completely offline."""

    def test_no_network_required(self, temp_db):
        """Storage should work without any network access."""
        import os

        # Unset any cloud credentials
        old_env = {}
        for key in ["OPENAI_API_KEY", "SUPABASE_URL", "KERNLE_SUPABASE_URL"]:
            old_env[key] = os.environ.pop(key, None)

        try:
            storage = SQLiteStorage(agent_id="offline-agent", db_path=temp_db)

            # All operations should work
            storage.save_note(Note(
                id="n1",
                agent_id="offline-agent",
                content="Fully offline operation"
            ))

            results = storage.search("offline")
            assert len(results) == 1

            stats = storage.get_stats()
            assert stats["notes"] == 1
        finally:
            # Restore env
            for key, val in old_env.items():
                if val is not None:
                    os.environ[key] = val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
