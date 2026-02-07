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
    path = Path(tempfile.mktemp(suffix=".db"))
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def storage(temp_db, monkeypatch):
    """Create a SQLiteStorage instance for testing.

    Cloud credentials are disabled to ensure tests use local storage only.
    """
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    # Disable cloud search to ensure test isolation
    monkeypatch.setattr(storage, "has_cloud_credentials", lambda: False)
    yield storage
    storage.close()


class TestSQLiteStorageBasics:
    """Basic CRUD operation tests."""

    def test_storage_creation(self, storage, temp_db):
        """Storage creates database and directory."""
        assert temp_db.exists()
        assert storage.stack_id == "test-agent"

    def test_sqlite_vec_available(self, storage):
        """sqlite-vec availability depends on environment."""
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available in this environment")
        assert storage._has_vec is True

    def test_embedder_default(self, storage):
        """Default embedder should be HashEmbedder."""
        assert isinstance(storage._embedder, HashEmbedder)


class TestEpisodes:
    """Episode storage tests."""

    def test_save_and_get_episode(self, storage):
        episode = Episode(
            id="ep-1",
            stack_id="test-agent",
            objective="Test objective",
            outcome="Test outcome",
            outcome_type="success",
            lessons=["lesson 1", "lesson 2"],
            tags=["test", "unit"],
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
            storage.save_episode(
                Episode(
                    id=f"ep-{i}",
                    stack_id="test-agent",
                    objective=f"Objective {i}",
                    outcome=f"Outcome {i}",
                )
            )

        episodes = storage.get_episodes(limit=3)
        assert len(episodes) == 3

    def test_episode_auto_id(self, storage):
        episode = Episode(
            id="",  # Empty ID
            stack_id="test-agent",
            objective="Auto ID test",
            outcome="Should generate UUID",
        )
        saved_id = storage.save_episode(episode)
        assert saved_id  # Should have a UUID


class TestNotes:
    """Note storage tests."""

    def test_save_and_get_note(self, storage):
        note = Note(
            id="note-1",
            stack_id="test-agent",
            content="This is a test note",
            note_type="observation",
            speaker="user",
            tags=["important"],
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
            stack_id="test-agent",
            statement="Testing is important",
            belief_type="principle",
            confidence=0.95,
        )

        storage.save_belief(belief)
        beliefs = storage.get_beliefs()

        assert len(beliefs) == 1
        assert beliefs[0].statement == "Testing is important"
        assert beliefs[0].confidence == 0.95

    def test_find_belief(self, storage):
        storage.save_belief(
            Belief(
                id="b1",
                stack_id="test-agent",
                statement="Unique statement here",
            )
        )

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
            stack_id="test-agent",
            name="Reliability",
            statement="I prioritize dependability in my work",
            priority=90,
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
            stack_id="test-agent",
            title="Complete project",
            description="Finish all tasks",
            priority="high",
            status="active",
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
            stack_id="test-agent",
            drive_type="curiosity",
            intensity=0.8,
            focus_areas=["learning", "exploration"],
        )

        storage.save_drive(drive)
        drives = storage.get_drives()

        assert len(drives) == 1
        assert drives[0].drive_type == "curiosity"
        assert drives[0].intensity == 0.8

    def test_update_drive(self, storage):
        storage.save_drive(
            Drive(id="d1", stack_id="test-agent", drive_type="growth", intensity=0.5)
        )

        # Update same drive type
        storage.save_drive(
            Drive(
                id="d2",  # Different ID but same type
                stack_id="test-agent",
                drive_type="growth",
                intensity=0.9,
            )
        )

        drive = storage.get_drive("growth")
        assert drive.intensity == 0.9


class TestRelationships:
    """Relationship storage tests."""

    def test_save_and_get_relationship(self, storage):
        rel = Relationship(
            id="rel-1",
            stack_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            sentiment=0.7,
            interaction_count=5,
        )

        storage.save_relationship(rel)
        rels = storage.get_relationships()

        assert len(rels) == 1
        assert rels[0].entity_name == "Alice"
        assert rels[0].sentiment == 0.7

    def test_get_relationship_by_name(self, storage):
        storage.save_relationship(
            Relationship(
                id="r1",
                stack_id="test-agent",
                entity_name="Bob",
                entity_type="human",
                relationship_type="friend",
            )
        )

        rel = storage.get_relationship("Bob")
        assert rel is not None
        assert rel.entity_name == "Bob"


class TestVectorSearch:
    """Vector search tests."""

    def test_search_episodes(self, storage):
        storage.save_episode(
            Episode(
                id="ep1",
                stack_id="test-agent",
                objective="Learn Python programming",
                outcome="Mastered basic syntax",
            )
        )
        storage.save_episode(
            Episode(
                id="ep2",
                stack_id="test-agent",
                objective="Write documentation",
                outcome="Created user guide",
            )
        )

        results = storage.search("programming language", limit=5)
        assert len(results) > 0
        # Python-related episode should rank higher
        assert results[0].record_type == "episode"

    def test_search_mixed_types(self, storage):
        storage.save_episode(
            Episode(
                id="ep1",
                stack_id="test-agent",
                objective="Database design",
                outcome="Created schema",
            )
        )
        storage.save_note(
            Note(id="note1", stack_id="test-agent", content="SQLite is a great embedded database")
        )
        storage.save_belief(
            Belief(id="b1", stack_id="test-agent", statement="Databases should be ACID compliant")
        )

        results = storage.search("database", limit=10)
        assert len(results) == 3

        # Check we got all types
        types = {r.record_type for r in results}
        assert "episode" in types
        assert "note" in types
        assert "belief" in types

    def test_search_with_type_filter(self, storage):
        storage.save_episode(
            Episode(
                id="ep1", stack_id="test-agent", objective="Search test", outcome="Testing filters"
            )
        )
        storage.save_note(Note(id="note1", stack_id="test-agent", content="Search test content"))

        results = storage.search("search test", record_types=["note"])
        assert all(r.record_type == "note" for r in results)

    def test_search_scores_ranked(self, storage):
        """Search results should be ranked by similarity."""
        storage.save_note(
            Note(id="n1", stack_id="test-agent", content="Machine learning and neural networks")
        )
        storage.save_note(
            Note(id="n2", stack_id="test-agent", content="Cooking recipes and kitchen tips")
        )

        results = storage.search("deep learning AI", limit=5)
        # ML note should rank higher
        assert results[0].record.content == "Machine learning and neural networks"


class TestSyncMetadata:
    """Sync metadata tests."""

    def test_sync_metadata_on_save(self, storage):
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Sync test", outcome="Testing")
        )

        episode = storage.get_episode("ep1")
        assert episode.local_updated_at is not None
        assert episode.cloud_synced_at is None  # Not synced yet
        assert episode.version >= 1
        assert episode.deleted is False

    def test_pending_sync_count(self, storage):
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        pending = storage.get_pending_sync_count()
        assert pending >= 2  # At least our 2 records


class TestStats:
    """Stats tests."""

    def test_get_stats(self, storage):
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))
        storage.save_belief(Belief(id="b1", stack_id="test-agent", statement="Test"))

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
        norm = sum(x * x for x in embedding) ** 0.5
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
            return sum(x * y for x, y in zip(a, b))

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
            storage = SQLiteStorage(stack_id="offline-agent", db_path=temp_db)

            # All operations should work
            storage.save_note(
                Note(id="n1", stack_id="offline-agent", content="Fully offline operation")
            )

            results = storage.search("offline")
            assert len(results) == 1

            stats = storage.get_stats()
            assert stats["notes"] == 1
        finally:
            # Restore env
            for key, val in old_env.items():
                if val is not None:
                    os.environ[key] = val


class TestSyncQueue:
    """Tests for the enhanced sync queue functionality."""

    def test_queue_sync_operation(self, storage):
        """Test manually queueing a sync operation."""
        queue_id = storage.queue_sync_operation(
            operation="insert",
            table="episodes",
            record_id="test-ep-1",
            data={"objective": "Test", "outcome": "Testing queue"},
        )
        assert queue_id is not None
        assert isinstance(queue_id, int)

    def test_get_pending_sync_operations(self, storage):
        """Test getting pending sync operations."""
        # Save some records (which automatically queue sync)
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test note"))

        pending = storage.get_pending_sync_operations()
        assert len(pending) >= 2

        # Check structure of pending operations
        for op in pending:
            assert "id" in op
            assert "operation" in op
            assert "table_name" in op
            assert "record_id" in op
            assert "data" in op
            assert "local_updated_at" in op

    def test_pending_operations_have_data(self, storage):
        """Test that pending operations include record data."""
        storage.save_episode(
            Episode(
                id="ep-with-data",
                stack_id="test-agent",
                objective="Data test",
                outcome="Should have data in queue",
            )
        )

        pending = storage.get_pending_sync_operations()
        episode_ops = [op for op in pending if op["record_id"] == "ep-with-data"]
        assert len(episode_ops) == 1

        op = episode_ops[0]
        assert op["data"] is not None
        assert op["data"]["objective"] == "Data test"
        assert op["data"]["outcome"] == "Should have data in queue"

    def test_mark_synced(self, storage):
        """Test marking operations as synced."""
        # Create some pending operations
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        # Get pending
        pending = storage.get_pending_sync_operations()
        initial_count = len(pending)
        assert initial_count >= 2

        # Mark first one as synced
        ids_to_mark = [pending[0]["id"]]
        marked = storage.mark_synced(ids_to_mark)
        assert marked == 1

        # Verify count decreased
        pending_after = storage.get_pending_sync_operations()
        assert len(pending_after) == initial_count - 1

    def test_mark_synced_multiple(self, storage):
        """Test marking multiple operations as synced."""
        # Create operations
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test1", outcome="Test1")
        )
        storage.save_episode(
            Episode(id="ep2", stack_id="test-agent", objective="Test2", outcome="Test2")
        )
        storage.save_episode(
            Episode(id="ep3", stack_id="test-agent", objective="Test3", outcome="Test3")
        )

        pending = storage.get_pending_sync_operations()
        ids_to_mark = [op["id"] for op in pending[:2]]

        marked = storage.mark_synced(ids_to_mark)
        assert marked == 2

    def test_get_sync_status(self, storage):
        """Test getting sync status summary."""
        # Create some operations
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", stack_id="test-agent", content="Test"))

        status = storage.get_sync_status()

        # Check structure
        assert "pending" in status
        assert "synced" in status
        assert "total" in status
        assert "by_table" in status
        assert "by_operation" in status

        # Check values
        assert status["pending"] >= 2
        assert status["total"] == status["pending"] + status["synced"]

        # Check breakdown
        assert "episodes" in status["by_table"]
        assert "notes" in status["by_table"]

    def test_sync_status_by_operation(self, storage):
        """Test sync status breakdown by operation type."""
        storage.save_episode(
            Episode(id="ep1", stack_id="test-agent", objective="Test", outcome="Test")
        )

        status = storage.get_sync_status()

        # All saves should queue as "upsert"
        assert "upsert" in status["by_operation"]
        assert status["by_operation"]["upsert"] >= 1

    def test_queue_deduplication(self, storage):
        """Test that multiple saves to same record don't create duplicate queue entries."""
        # Save same episode multiple times
        for i in range(3):
            storage.save_episode(
                Episode(
                    id="ep-same", stack_id="test-agent", objective=f"Update {i}", outcome="Test"
                )
            )

        # Should only have one pending operation for this record
        pending = storage.get_pending_sync_operations()
        ep_ops = [op for op in pending if op["record_id"] == "ep-same"]
        assert len(ep_ops) == 1

        # Should have the latest data
        assert ep_ops[0]["data"]["objective"] == "Update 2"

    def test_mark_synced_empty_list(self, storage):
        """Test marking empty list returns 0."""
        marked = storage.mark_synced([])
        assert marked == 0


class TestBatchInsertion:
    """Tests for batch insertion methods."""

    def test_save_episodes_batch_empty(self, storage):
        """Empty batch should return empty list."""
        ids = storage.save_episodes_batch([])
        assert ids == []

    def test_save_episodes_batch_single(self, storage):
        """Single episode batch should work."""
        episodes = [
            Episode(
                id="ep-batch-1",
                stack_id="test-agent",
                objective="Batch test 1",
                outcome="Success",
            )
        ]
        ids = storage.save_episodes_batch(episodes)

        assert len(ids) == 1
        assert ids[0] == "ep-batch-1"

        # Verify saved
        saved = storage.get_episode("ep-batch-1")
        assert saved is not None
        assert saved.objective == "Batch test 1"

    def test_save_episodes_batch_multiple(self, storage):
        """Multiple episodes should be saved in single transaction."""
        episodes = [
            Episode(
                id=f"ep-batch-{i}",
                stack_id="test-agent",
                objective=f"Batch objective {i}",
                outcome=f"Batch outcome {i}",
                tags=["batch", f"test-{i}"],
            )
            for i in range(10)
        ]

        ids = storage.save_episodes_batch(episodes)

        assert len(ids) == 10
        for i, ep_id in enumerate(ids):
            assert ep_id == f"ep-batch-{i}"

        # Verify all saved
        all_episodes = storage.get_episodes(limit=20)
        assert len(all_episodes) == 10

    def test_save_episodes_batch_auto_generates_ids(self, storage):
        """Episodes without IDs should get auto-generated UUIDs."""
        episodes = [
            Episode(
                id="",  # Empty ID
                stack_id="test-agent",
                objective="Auto ID test",
                outcome="Success",
            ),
            Episode(
                id="",  # Empty ID
                stack_id="test-agent",
                objective="Auto ID test 2",
                outcome="Success 2",
            ),
        ]

        ids = storage.save_episodes_batch(episodes)

        assert len(ids) == 2
        assert ids[0] != ""  # Should have generated UUID
        assert ids[1] != ""
        assert ids[0] != ids[1]  # Different IDs

    def test_save_beliefs_batch_empty(self, storage):
        """Empty batch should return empty list."""
        ids = storage.save_beliefs_batch([])
        assert ids == []

    def test_save_beliefs_batch_multiple(self, storage):
        """Multiple beliefs should be saved in single transaction."""
        beliefs = [
            Belief(
                id=f"belief-batch-{i}",
                stack_id="test-agent",
                statement=f"Belief statement {i}",
                confidence=0.8 + i * 0.02,
            )
            for i in range(5)
        ]

        ids = storage.save_beliefs_batch(beliefs)

        assert len(ids) == 5
        for i, belief_id in enumerate(ids):
            assert belief_id == f"belief-batch-{i}"

        # Verify all saved
        all_beliefs = storage.get_beliefs(limit=20)
        assert len(all_beliefs) == 5

    def test_save_notes_batch_empty(self, storage):
        """Empty batch should return empty list."""
        ids = storage.save_notes_batch([])
        assert ids == []

    def test_save_notes_batch_multiple(self, storage):
        """Multiple notes should be saved in single transaction."""
        notes = [
            Note(
                id=f"note-batch-{i}",
                stack_id="test-agent",
                content=f"Note content {i}",
                note_type="note",
                tags=["batch"],
            )
            for i in range(5)
        ]

        ids = storage.save_notes_batch(notes)

        assert len(ids) == 5
        for i, note_id in enumerate(ids):
            assert note_id == f"note-batch-{i}"

        # Verify all saved
        all_notes = storage.get_notes(limit=20)
        assert len(all_notes) == 5

    def test_batch_preserves_all_fields(self, storage):
        """Batch insert should preserve all episode fields."""
        episodes = [
            Episode(
                id="ep-full",
                stack_id="test-agent",
                objective="Full test",
                outcome="Complete success",
                outcome_type="success",
                lessons=["Lesson 1", "Lesson 2"],
                tags=["full", "test"],
                emotional_valence=0.8,
                emotional_arousal=0.5,
                emotional_tags=["joy"],
                confidence=0.95,
                source_type="direct_experience",
            )
        ]

        storage.save_episodes_batch(episodes)
        saved = storage.get_episode("ep-full")

        assert saved is not None
        assert saved.objective == "Full test"
        assert saved.outcome == "Complete success"
        assert saved.outcome_type == "success"
        assert saved.lessons == ["Lesson 1", "Lesson 2"]
        assert saved.tags == ["full", "test"]
        assert saved.emotional_valence == 0.8
        assert saved.emotional_arousal == 0.5
        assert saved.emotional_tags == ["joy"]
        assert saved.confidence == 0.95

    def test_batch_queues_sync(self, storage):
        """Batch inserts should queue sync operations."""
        episodes = [
            Episode(
                id=f"ep-sync-{i}",
                stack_id="test-agent",
                objective=f"Sync test {i}",
                outcome="Success",
            )
            for i in range(3)
        ]

        storage.save_episodes_batch(episodes)

        # Check that sync operations were queued
        pending = storage.get_pending_sync_operations()
        ep_ops = [op for op in pending if op["record_id"].startswith("ep-sync-")]
        assert len(ep_ops) == 3

    def test_batch_saves_embeddings(self, storage):
        """Batch inserts should save embeddings for search."""
        episodes = [
            Episode(
                id="ep-embed-1",
                stack_id="test-agent",
                objective="Python programming tutorial",
                outcome="Learned basics",
            ),
            Episode(
                id="ep-embed-2",
                stack_id="test-agent",
                objective="Database optimization",
                outcome="Improved performance",
            ),
        ]

        storage.save_episodes_batch(episodes)

        # Search should find the episodes
        results = storage.search("Python programming")
        assert len(results) >= 1
        assert any(r.record.id == "ep-embed-1" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestProvenanceProtection:
    """Test that provenance fields are write-once/append-only."""

    def test_source_type_is_write_once(self, storage):
        """source_type should not change on update."""
        episode = Episode(
            id="prov_test_1",
            stack_id="test_agent",
            objective="Test objective",
            outcome="Test outcome",
            source_type="direct_experience",
        )
        storage.save_episode(episode)

        # Try to change source_type
        episode.source_type = "inference"
        storage.update_episode_atomic(episode, expected_version=1)

        # Verify source_type is preserved
        retrieved = storage.get_episode("prov_test_1")
        assert retrieved.source_type == "direct_experience"

    def test_derived_from_is_append_only(self, storage):
        """derived_from should only allow adding, not removing."""
        episode = Episode(
            id="prov_test_2",
            stack_id="test_agent",
            objective="Test objective",
            outcome="Test outcome",
            derived_from=["episode:orig_1", "episode:orig_2"],
        )
        storage.save_episode(episode)

        # Try to replace derived_from (removing orig_1)
        episode.derived_from = ["episode:orig_3"]
        storage.update_episode_atomic(episode, expected_version=1)

        # Verify all originals are preserved and new one added
        retrieved = storage.get_episode("prov_test_2")
        assert "episode:orig_1" in retrieved.derived_from
        assert "episode:orig_2" in retrieved.derived_from
        assert "episode:orig_3" in retrieved.derived_from

    def test_confidence_history_is_append_only(self, storage):
        """confidence_history should only allow appending."""
        episode = Episode(
            id="prov_test_3",
            stack_id="test_agent",
            objective="Test objective",
            outcome="Test outcome",
            confidence_history=["0.8@2024-01-01", "0.9@2024-01-02"],
        )
        storage.save_episode(episode)

        # Try to replace history
        episode.confidence_history = ["0.7@2024-01-03"]
        storage.update_episode_atomic(episode, expected_version=1)

        # Verify original history preserved and new entry added
        retrieved = storage.get_episode("prov_test_3")
        assert "0.8@2024-01-01" in retrieved.confidence_history
        assert "0.9@2024-01-02" in retrieved.confidence_history
        assert "0.7@2024-01-03" in retrieved.confidence_history

    def test_update_episode_atomic_no_belief_columns(self, storage):
        """Regression: update_episode_atomic must not reference belief-only columns."""
        episode = Episode(
            id="prov_test_regression",
            stack_id="test_agent",
            objective="Test objective",
            outcome="Test outcome",
            source_type="direct_experience",
            derived_from=["episode:a"],
            confidence_history=["0.8@2024-01-01"],
        )
        storage.save_episode(episode)

        # This should succeed without 'table episodes has no column named belief_scope'
        episode.objective = "Updated objective"
        storage.update_episode_atomic(episode, expected_version=1)

        retrieved = storage.get_episode("prov_test_regression")
        assert retrieved.objective == "Updated objective"
        assert retrieved.version == 2


class TestStackIdValidation:
    """Test that SQLiteStorage rejects path traversal in stack_id."""

    @pytest.mark.parametrize(
        "bad_id,match",
        [
            ("..", "relative path component"),
            (".", "relative path component"),
            ("../x", "path separators"),
            ("nested/evil", "path separators"),
            ("a\\b", "path separators"),
            ("foo/../bar", "path separators"),
        ],
    )
    def test_reject_path_traversal(self, temp_db, bad_id, match):
        """SQLiteStorage must reject stack IDs with path traversal."""
        with pytest.raises(ValueError, match=match):
            SQLiteStorage(stack_id=bad_id, db_path=temp_db)


class TestReadOnlyEnvironmentFallback:
    """Tests for SQLiteStorage initialization when home directory is not writable."""

    def test_falls_back_to_temp_dir_when_home_not_writable(self, tmp_path):
        """SQLiteStorage should fall back to temp dir when ~/.kernle is not writable."""
        from unittest.mock import patch

        # Point home to a read-only directory
        readonly_home = tmp_path / "readonly_home"
        readonly_home.mkdir()

        with (
            patch("kernle.storage.sqlite.get_kernle_home", return_value=readonly_home / ".kernle"),
            patch.object(Path, "mkdir", side_effect=PermissionError("Read-only filesystem")),
        ):
            # Attempting without explicit db_path should fall back to tempdir
            # We can't easily make mkdir fail for some paths but not others,
            # so test the _resolve_db_path method directly
            pass

        # More targeted test: use _resolve_db_path directly
        storage = SQLiteStorage(stack_id="test-agent", db_path=tmp_path / "test.db")
        try:
            with patch(
                "kernle.storage.sqlite.get_kernle_home",
                return_value=readonly_home / ".kernle",
            ):
                # Simulate mkdir failing for the default path
                original_mkdir = Path.mkdir

                def mock_mkdir(self_path, *args, **kwargs):
                    if str(self_path).startswith(str(readonly_home)):
                        raise PermissionError("Read-only filesystem")
                    return original_mkdir(self_path, *args, **kwargs)

                with patch.object(Path, "mkdir", mock_mkdir):
                    resolved = storage._resolve_db_path(None)
                    # Should NOT be under the readonly home
                    assert not str(resolved).startswith(str(readonly_home))
                    # Should be under temp dir
                    assert ".kernle" in str(resolved)
                    assert "memories.db" in str(resolved)
        finally:
            storage.close()

    def test_explicit_db_path_does_not_fallback(self, tmp_path):
        """When db_path is explicitly provided, no fallback should occur."""
        explicit_path = tmp_path / "explicit.db"
        storage = SQLiteStorage(stack_id="test-agent", db_path=explicit_path)
        try:
            assert storage.db_path == explicit_path.resolve()
        finally:
            storage.close()
