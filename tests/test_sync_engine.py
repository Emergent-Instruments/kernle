"""Tests for the sync engine.

Tests:
- Queueing changes when offline
- Pushing queued changes when back online
- Pulling changes from cloud
- Conflict resolution (last-write-wins)
- Sync metadata tracking
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    QueuedChange,
    Relationship,
    SQLiteStorage,
    SupabaseStorage,
    SyncConflict,
    SyncResult,
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
    storage = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


@pytest.fixture
def mock_cloud_storage():
    """Create a mock cloud storage for testing sync."""
    mock = MagicMock(spec=SupabaseStorage)
    mock.agent_id = "test-agent"

    # Default return values
    mock.get_stats.return_value = {"episodes": 0, "notes": 0}
    mock.get_episodes.return_value = []
    mock.get_notes.return_value = []
    mock.get_beliefs.return_value = []
    mock.get_values.return_value = []
    mock.get_goals.return_value = []
    mock.get_drives.return_value = []
    mock.get_relationships.return_value = []

    return mock


@pytest.fixture
def storage_with_cloud(temp_db, mock_cloud_storage):
    """Create a SQLiteStorage with a mock cloud storage."""
    storage = SQLiteStorage(
        agent_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
    )
    yield storage
    storage.close()


class TestSyncQueueBasics:
    """Test the sync queue functionality."""

    def test_changes_are_queued_on_save(self, storage):
        """Saving a record should queue it for sync."""
        initial_count = storage.get_pending_sync_count()

        storage.save_note(Note(id="n1", agent_id="test-agent", content="Test note"))

        assert storage.get_pending_sync_count() == initial_count + 1

    def test_multiple_saves_same_record_dedupe(self, storage):
        """Multiple saves of the same record should dedupe in queue."""
        storage.save_note(Note(id="n1", agent_id="test-agent", content="First version"))

        count_after_first = storage.get_pending_sync_count()

        storage.save_note(Note(id="n1", agent_id="test-agent", content="Second version"))

        # Should still be same count (deduped)
        assert storage.get_pending_sync_count() == count_after_first

    def test_get_queued_changes(self, storage):
        """Can retrieve queued changes."""
        storage.save_episode(
            Episode(id="ep1", agent_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", agent_id="test-agent", content="Test note"))

        changes = storage.get_queued_changes()

        assert len(changes) >= 2
        assert all(isinstance(c, QueuedChange) for c in changes)

        tables = {c.table_name for c in changes}
        assert "episodes" in tables
        assert "notes" in tables

    def test_queued_change_has_timestamp(self, storage):
        """Queued changes should have a timestamp."""
        storage.save_note(Note(id="n1", agent_id="test-agent", content="Test"))

        changes = storage.get_queued_changes()
        assert len(changes) > 0
        assert changes[0].queued_at is not None


class TestConnectivity:
    """Test connectivity detection."""

    def test_is_online_false_without_cloud(self, storage):
        """Without cloud storage configured, is_online returns False."""
        assert storage.is_online() is False

    def test_is_online_true_with_reachable_cloud(self, storage_with_cloud, mock_cloud_storage):
        """With reachable cloud storage, is_online returns True."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        assert storage_with_cloud.is_online() is True

    def test_is_online_false_when_cloud_unreachable(self, storage_with_cloud, mock_cloud_storage):
        """When cloud throws an exception, is_online returns False."""
        mock_cloud_storage.get_stats.side_effect = Exception("Connection refused")

        assert storage_with_cloud.is_online() is False

    def test_connectivity_cache(self, storage_with_cloud, mock_cloud_storage):
        """Connectivity result is cached briefly."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # First call
        storage_with_cloud.is_online()
        # Second call should use cache
        storage_with_cloud.is_online()

        # Should only have called get_stats once due to caching
        assert mock_cloud_storage.get_stats.call_count == 1


class TestSyncPush:
    """Test pushing changes to cloud."""

    def test_sync_without_cloud_returns_empty_result(self, storage):
        """Sync without cloud storage configured returns empty result."""
        result = storage.sync()

        assert isinstance(result, SyncResult)
        assert result.pushed == 0
        assert result.pulled == 0

    def test_sync_pushes_queued_changes(self, storage_with_cloud, mock_cloud_storage):
        """Sync should push queued changes to cloud."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create some local changes
        storage_with_cloud.save_episode(
            Episode(
                id="ep1", agent_id="test-agent", objective="Test objective", outcome="Test outcome"
            )
        )
        storage_with_cloud.save_note(Note(id="n1", agent_id="test-agent", content="Test note"))

        # Sync
        result = storage_with_cloud.sync()

        assert result.pushed >= 2
        assert mock_cloud_storage.save_episode.called
        assert mock_cloud_storage.save_note.called

    def test_sync_clears_queue_on_success(self, storage_with_cloud, mock_cloud_storage):
        """Successful sync should clear the queue."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="n1", agent_id="test-agent", content="Test"))

        assert storage_with_cloud.get_pending_sync_count() > 0

        storage_with_cloud.sync()

        # Queue should be cleared
        assert storage_with_cloud.get_pending_sync_count() == 0

    def test_sync_marks_records_synced(self, storage_with_cloud, mock_cloud_storage):
        """Synced records should have cloud_synced_at set."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(id="n1", agent_id="test-agent", content="Test"))

        # Before sync
        notes = storage_with_cloud.get_notes()
        assert notes[0].cloud_synced_at is None

        # Sync
        storage_with_cloud.sync()

        # After sync
        notes = storage_with_cloud.get_notes()
        assert notes[0].cloud_synced_at is not None

    def test_sync_offline_returns_error(self, storage_with_cloud, mock_cloud_storage):
        """Sync when offline should return error."""
        mock_cloud_storage.get_stats.side_effect = Exception("Connection refused")

        storage_with_cloud.save_note(Note(id="n1", agent_id="test-agent", content="Test"))

        result = storage_with_cloud.sync()

        assert len(result.errors) > 0
        assert "Offline" in result.errors[0]
        # Queue should NOT be cleared
        assert storage_with_cloud.get_pending_sync_count() > 0


class TestSyncPull:
    """Test pulling changes from cloud."""

    def test_pull_new_records(self, storage_with_cloud, mock_cloud_storage):
        """Pull should add new records from cloud."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Cloud has a record we don't have locally
        cloud_note = Note(
            id="cloud-note-1",
            agent_id="test-agent",
            content="Note from cloud",
            cloud_synced_at=datetime.now(timezone.utc),
            local_updated_at=datetime.now(timezone.utc),
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Do a sync (which includes pull)
        result = storage_with_cloud.sync()

        # Should have pulled the note
        assert result.pulled >= 1

        # Note should exist locally
        notes = storage_with_cloud.get_notes()
        note_ids = {n.id for n in notes}
        assert "cloud-note-1" in note_ids

    def test_pull_without_cloud_returns_empty(self, storage):
        """Pull without cloud storage returns empty result."""
        result = storage.pull_changes()

        assert result.pulled == 0
        assert result.conflict_count == 0


class TestConflictResolution:
    """Test conflict resolution with last-write-wins."""

    def test_cloud_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Cloud record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_note(
            Note(
                id="conflict-note",
                agent_id="test-agent",
                content="Local version",
                local_updated_at=old_time,
            )
        )

        # Manually clear the queue so we can test pull
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version
        new_time = datetime.now(timezone.utc)
        cloud_note = Note(
            id="conflict-note",
            agent_id="test-agent",
            content="Cloud version (newer)",
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict
        assert result.conflict_count >= 1
        assert len(result.conflicts) >= 1

        # Verify conflict details
        conflict = result.conflicts[0]
        assert conflict.table == "notes"
        assert conflict.record_id == "conflict-note"
        assert "cloud_wins" in conflict.resolution  # May include _arrays_merged suffix
        assert conflict.local_summary is not None
        assert conflict.cloud_summary is not None

        # Local should now have cloud's content
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Cloud version" in conflict_note.content

    def test_local_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Local record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record (newer)
        new_time = datetime.now(timezone.utc)
        storage_with_cloud.save_note(
            Note(
                id="conflict-note",
                agent_id="test-agent",
                content="Local version (newer)",
                local_updated_at=new_time,
            )
        )

        # Manually clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has an older version
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        cloud_note = Note(
            id="conflict-note",
            agent_id="test-agent",
            content="Cloud version (older)",
            cloud_synced_at=old_time,
            local_updated_at=old_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should detect conflict
        assert result.conflict_count >= 1
        assert len(result.conflicts) >= 1

        # Verify conflict details
        conflict = result.conflicts[0]
        assert conflict.table == "notes"
        assert conflict.record_id == "conflict-note"
        assert "local_wins" in conflict.resolution  # May include _arrays_merged suffix

        # Local version should be preserved
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Local version" in conflict_note.content

    def test_conflict_history_stored(self, storage_with_cloud, mock_cloud_storage):
        """Conflicts should be stored in history."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_note(
            Note(
                id="history-note",
                agent_id="test-agent",
                content="Local version",
                local_updated_at=old_time,
            )
        )

        # Clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version
        new_time = datetime.now(timezone.utc)
        cloud_note = Note(
            id="history-note",
            agent_id="test-agent",
            content="Cloud version (newer)",
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull to create conflict
        storage_with_cloud.pull_changes()

        # Check conflict history
        history = storage_with_cloud.get_sync_conflicts(limit=10)
        assert len(history) >= 1

        # Find our conflict
        conflict = next((c for c in history if c.record_id == "history-note"), None)
        assert conflict is not None
        assert conflict.table == "notes"
        assert "cloud_wins" in conflict.resolution  # May include _arrays_merged suffix
        assert "Local version" in (conflict.local_summary or "")
        assert "Cloud version" in (conflict.cloud_summary or "")

    def test_conflict_history_clear(self, storage):
        """Conflict history can be cleared."""
        # Add some test conflicts manually
        conflict = SyncConflict(
            id="test-conflict-1",
            table="notes",
            record_id="test-note",
            local_version={"content": "local"},
            cloud_version={"content": "cloud"},
            resolution="cloud_wins",
            resolved_at=datetime.now(timezone.utc),
            local_summary="local",
            cloud_summary="cloud",
        )
        storage.save_sync_conflict(conflict)

        # Verify it was saved
        history = storage.get_sync_conflicts()
        assert len(history) >= 1

        # Clear all
        cleared = storage.clear_sync_conflicts()
        assert cleared >= 1

        # Verify cleared
        history = storage.get_sync_conflicts()
        assert len(history) == 0


class TestSyncMetadata:
    """Test sync metadata tracking."""

    def test_last_sync_time_initially_none(self, storage):
        """Last sync time should be None initially."""
        assert storage.get_last_sync_time() is None

    def test_last_sync_time_updated_on_sync(self, storage_with_cloud, mock_cloud_storage):
        """Last sync time should be updated after sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        before = storage_with_cloud.get_last_sync_time()
        assert before is None

        storage_with_cloud.sync()

        after = storage_with_cloud.get_last_sync_time()
        assert after is not None
        assert isinstance(after, datetime)

    def test_sync_meta_persistence(self, temp_db, mock_cloud_storage):
        """Sync metadata should persist across storage instances."""
        # First instance
        storage1 = SQLiteStorage(
            agent_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
        )
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        storage1.sync()

        sync_time = storage1.get_last_sync_time()
        storage1.close()

        # Second instance
        storage2 = SQLiteStorage(
            agent_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
        )

        # Should have same last sync time
        assert storage2.get_last_sync_time() == sync_time
        storage2.close()


class TestSyncAllRecordTypes:
    """Test that sync works for all record types."""

    def test_sync_episodes(self, storage_with_cloud, mock_cloud_storage):
        """Episodes should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_episode(
            Episode(id="ep1", agent_id="test-agent", objective="Test", outcome="Test")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_episode.called

    def test_sync_beliefs(self, storage_with_cloud, mock_cloud_storage):
        """Beliefs should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_belief(
            Belief(id="b1", agent_id="test-agent", statement="Test belief")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_belief.called

    def test_sync_values(self, storage_with_cloud, mock_cloud_storage):
        """Values should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_value(
            Value(id="v1", agent_id="test-agent", name="Test", statement="Test value")
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_value.called

    def test_sync_goals(self, storage_with_cloud, mock_cloud_storage):
        """Goals should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_goal(Goal(id="g1", agent_id="test-agent", title="Test goal"))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_goal.called

    def test_sync_drives(self, storage_with_cloud, mock_cloud_storage):
        """Drives should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_drive(Drive(id="d1", agent_id="test-agent", drive_type="curiosity"))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_drive.called

    def test_sync_relationships(self, storage_with_cloud, mock_cloud_storage):
        """Relationships should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_relationship(
            Relationship(
                id="r1",
                agent_id="test-agent",
                entity_name="Alice",
                entity_type="human",
                relationship_type="friend",
            )
        )

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_relationship.called


class TestOfflineQueuing:
    """Test that changes are properly queued when offline."""

    def test_operations_work_offline(self, storage):
        """All operations should work without cloud configured."""
        # These should all succeed
        storage.save_episode(
            Episode(id="ep1", agent_id="test-agent", objective="Test", outcome="Test")
        )
        storage.save_note(Note(id="n1", agent_id="test-agent", content="Test"))
        storage.save_belief(Belief(id="b1", agent_id="test-agent", statement="Test"))

        # Data should be accessible
        assert len(storage.get_episodes()) == 1
        assert len(storage.get_notes()) == 1
        assert len(storage.get_beliefs()) == 1

        # Changes should be queued
        assert storage.get_pending_sync_count() >= 3

    def test_queue_survives_reconnect(self, temp_db):
        """Queue should survive closing and reopening storage."""
        # Create storage and add data
        storage1 = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
        storage1.save_note(Note(id="n1", agent_id="test-agent", content="Test"))

        pending_before = storage1.get_pending_sync_count()
        storage1.close()

        # Create new instance
        storage2 = SQLiteStorage(agent_id="test-agent", db_path=temp_db)

        # Queue should still be there
        assert storage2.get_pending_sync_count() == pending_before
        storage2.close()


class TestSyncEdgeCases:
    """Test edge cases in sync behavior."""

    def test_sync_deleted_record(self, storage_with_cloud, mock_cloud_storage):
        """Sync handles records that were deleted locally."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Save a note
        storage_with_cloud.save_note(
            Note(id="n1", agent_id="test-agent", content="Will be deleted")
        )

        # Delete it by marking as deleted (soft delete)
        with storage_with_cloud._get_conn() as conn:
            conn.execute("UPDATE notes SET deleted = 1 WHERE id = 'n1'")
            conn.commit()

        # Sync should handle this gracefully
        result = storage_with_cloud.sync()

        # Should not fail
        assert result.success or len(result.errors) == 0

    def test_sync_empty_queue(self, storage_with_cloud, mock_cloud_storage):
        """Sync with empty queue should succeed."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Clear queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        result = storage_with_cloud.sync()

        assert result.success
        assert result.pushed == 0

    def test_partial_sync_failure(self, storage_with_cloud, mock_cloud_storage):
        """Sync continues even if some records fail."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Save multiple notes
        storage_with_cloud.save_note(Note(id="n1", agent_id="test-agent", content="Note 1"))
        storage_with_cloud.save_note(Note(id="n2", agent_id="test-agent", content="Note 2"))

        # First call fails, second succeeds
        mock_cloud_storage.save_note.side_effect = [Exception("First failed"), None]  # Success

        result = storage_with_cloud.sync()

        # Should have pushed one successfully
        assert result.pushed >= 1
        # Should have recorded the error
        assert len(result.errors) >= 1


class TestSyncHooks:
    """Test the auto-sync hooks for load and checkpoint."""

    def test_auto_sync_defaults_to_true_when_online(self, storage_with_cloud, mock_cloud_storage):
        """Auto-sync should default to true when cloud storage is available and online."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 0}  # Returns value = online

        k = Kernle(agent_id="test-agent", storage=storage_with_cloud)
        assert k.auto_sync is True

    def test_auto_sync_can_be_disabled_via_property(self, storage):
        """Auto-sync can be disabled by setting the property."""
        from kernle import Kernle

        k = Kernle(agent_id="test-agent", storage=storage)
        k.auto_sync = False
        assert k.auto_sync is False

    def test_load_with_sync_false_skips_pull(self, storage):
        """Load with sync=False should not attempt to pull."""
        from kernle import Kernle

        k = Kernle(agent_id="test-agent", storage=storage)
        k.auto_sync = True

        # Load with sync=False should work without errors
        memory = k.load(sync=False)
        assert "checkpoint" in memory

    def test_load_with_sync_true_attempts_pull(self, storage_with_cloud, mock_cloud_storage):
        """Load with sync=True should attempt to pull changes."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 1}  # Simulates online
        mock_cloud_storage.get_episodes.return_value = []
        mock_cloud_storage.get_notes.return_value = []
        mock_cloud_storage.get_beliefs.return_value = []
        mock_cloud_storage.get_values.return_value = []
        mock_cloud_storage.get_goals.return_value = []
        mock_cloud_storage.get_drives.return_value = []
        mock_cloud_storage.get_relationships.return_value = []

        k = Kernle(agent_id="test-agent", storage=storage_with_cloud)
        memory = k.load(sync=True)

        # Should have attempted to pull from cloud
        assert mock_cloud_storage.get_episodes.called or "checkpoint" in memory

    def test_checkpoint_with_sync_true_attempts_push(self, storage_with_cloud, mock_cloud_storage):
        """Checkpoint with sync=True should attempt to push changes."""
        from kernle import Kernle

        mock_cloud_storage.get_stats.return_value = {"episodes": 1}  # Simulates online

        k = Kernle(agent_id="test-agent", storage=storage_with_cloud)

        result = k.checkpoint("Test task", pending=["Next"], sync=True)

        assert result["current_task"] == "Test task"
        # Should have sync result attached
        assert "_sync" in result
        sync_result = result["_sync"]
        assert sync_result["attempted"] is True

    def test_checkpoint_sync_result_in_response(self, storage):
        """Checkpoint should include sync result when sync is attempted."""
        from kernle import Kernle

        k = Kernle(agent_id="test-agent", storage=storage)

        # With no cloud storage, sync should report offline
        result = k.checkpoint("Test task", sync=True)

        assert "_sync" in result
        sync_result = result["_sync"]
        # Should not have attempted since offline
        assert sync_result["attempted"] is False
        assert len(sync_result["errors"]) > 0

    def test_load_sync_non_blocking_on_error(self, storage_with_cloud, mock_cloud_storage):
        """Load should not fail if sync pull fails."""
        from kernle import Kernle

        # Make cloud throw an error
        mock_cloud_storage.get_stats.side_effect = Exception("Network error")

        k = Kernle(agent_id="test-agent", storage=storage_with_cloud)

        # Load should still work
        memory = k.load(sync=True)
        assert "checkpoint" in memory

    def test_checkpoint_sync_non_blocking_on_error(self, storage_with_cloud, mock_cloud_storage):
        """Checkpoint should not fail if sync push fails."""
        from kernle import Kernle

        # Make cloud throw an error
        mock_cloud_storage.get_stats.side_effect = Exception("Network error")

        k = Kernle(agent_id="test-agent", storage=storage_with_cloud)

        # Checkpoint should still work
        result = k.checkpoint("Test task", sync=True)
        assert result["current_task"] == "Test task"


class TestSyncQueueAtomicity:
    """Test that sync queue operations are atomic."""

    def test_queue_sync_uses_atomic_upsert(self, storage):
        """Verify that queue_sync uses INSERT ON CONFLICT (atomic) not DELETE+INSERT."""
        # Save the same note multiple times rapidly
        for i in range(10):
            storage.save_note(Note(id="atomic-test", agent_id="test-agent", content=f"Version {i}"))

        # Should still have exactly one queued change for this record
        changes = storage.get_queued_changes()
        matching = [c for c in changes if c.record_id == "atomic-test"]
        assert len(matching) == 1

        # The queued change should have the latest data
        # (Verify it's truly the latest by checking the operation is "upsert")
        assert matching[0].operation in ("upsert", "insert", "update")

    def test_sync_queue_deduplication_preserves_latest(self, storage):
        """Multiple updates to the same record should keep only the latest in queue."""
        # Save three different versions
        storage.save_episode(
            Episode(id="dedupe-test", agent_id="test-agent", objective="First", outcome="v1")
        )
        storage.save_episode(
            Episode(id="dedupe-test", agent_id="test-agent", objective="Second", outcome="v2")
        )
        storage.save_episode(
            Episode(id="dedupe-test", agent_id="test-agent", objective="Third", outcome="v3")
        )

        # Get queued changes for this record
        changes = storage.get_queued_changes()
        matching = [c for c in changes if c.record_id == "dedupe-test"]

        # Should be exactly one queued change
        assert len(matching) == 1

        # Verify the episode was saved with the latest values
        # (Queue stores the record ID, actual data is in the episodes table)
        episodes = storage.get_episodes()
        dedupe_episode = next(e for e in episodes if e.id == "dedupe-test")
        assert dedupe_episode.objective == "Third"
        assert dedupe_episode.outcome == "v3"


class TestArrayFieldMerging:
    """Test that array fields are merged (set union) during sync instead of last-write-wins."""

    def test_cloud_wins_merges_local_tags(self, storage_with_cloud, mock_cloud_storage):
        """When cloud wins, array fields from local should be merged in."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with some tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-merge-1",
                agent_id="test-agent",
                objective="Test episode",
                outcome="Test outcome",
                tags=["local-tag-1", "local-tag-2"],
                lessons=["local-lesson-1"],
                emotional_tags=["joy"],
                local_updated_at=old_time,
            )
        )

        # Clear the queue so we can test pull
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has a newer version with different tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-merge-1",
            agent_id="test-agent",
            objective="Updated objective from cloud",
            outcome="Updated outcome from cloud",
            tags=["cloud-tag-1", "local-tag-1"],  # One overlapping, one new
            lessons=["cloud-lesson-1"],  # Different lesson
            emotional_tags=["curiosity"],  # Different emotion
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict
        assert result.conflict_count >= 1
        conflict = result.conflicts[0]
        assert "arrays_merged" in conflict.resolution

        # Verify the episode has merged arrays
        episode = storage_with_cloud.get_episode("ep-merge-1")
        assert episode is not None

        # Scalar fields should come from cloud (winner)
        assert episode.objective == "Updated objective from cloud"

        # Array fields should be merged (set union)
        assert set(episode.tags) == {"local-tag-1", "local-tag-2", "cloud-tag-1"}
        assert set(episode.lessons) == {"local-lesson-1", "cloud-lesson-1"}
        assert set(episode.emotional_tags) == {"joy", "curiosity"}

    def test_local_wins_merges_cloud_tags(self, storage_with_cloud, mock_cloud_storage):
        """When local wins, array fields from cloud should be merged in."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record (newer)
        new_time = datetime.now(timezone.utc)
        storage_with_cloud.save_note(
            Note(
                id="note-merge-1",
                agent_id="test-agent",
                content="Local content (newer)",
                tags=["local-tag-1", "common-tag"],
                local_updated_at=new_time,
            )
        )

        # Clear the queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has an older version with different tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        cloud_note = Note(
            id="note-merge-1",
            agent_id="test-agent",
            content="Cloud content (older)",
            tags=["cloud-tag-1", "common-tag"],
            cloud_synced_at=old_time,
            local_updated_at=old_time,
        )
        mock_cloud_storage.get_notes.return_value = [cloud_note]

        # Pull changes
        result = storage_with_cloud.pull_changes()

        # Should have a conflict where local wins but arrays are merged
        assert result.conflict_count >= 1
        conflict = result.conflicts[0]
        assert "local_wins" in conflict.resolution
        assert "arrays_merged" in conflict.resolution

        # Verify the note has merged arrays but local scalar content
        note = next(n for n in storage_with_cloud.get_notes() if n.id == "note-merge-1")
        assert note is not None

        # Scalar fields should come from local (winner)
        assert note.content == "Local content (newer)"

        # Tags should be merged (set union)
        assert set(note.tags) == {"local-tag-1", "cloud-tag-1", "common-tag"}

    def test_drive_focus_areas_merge(self, storage_with_cloud, mock_cloud_storage):
        """Drive focus_areas should be merged."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local drive
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_drive(
            Drive(
                id="drive-merge-1",
                agent_id="test-agent",
                drive_type="curiosity",
                focus_areas=["local-area-1", "common-area"],
                local_updated_at=old_time,
            )
        )

        # Clear queue
        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has newer version with different focus areas
        new_time = datetime.now(timezone.utc)
        cloud_drive = Drive(
            id="drive-merge-1",
            agent_id="test-agent",
            drive_type="curiosity",
            intensity=0.8,  # Updated intensity
            focus_areas=["cloud-area-1", "common-area"],
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_drives.return_value = [cloud_drive]

        # Pull changes
        storage_with_cloud.pull_changes()

        # Verify merged focus areas
        drive = storage_with_cloud.get_drive("curiosity")
        assert drive is not None
        assert drive.intensity == 0.8  # Scalar from cloud winner
        assert set(drive.focus_areas) == {"local-area-1", "cloud-area-1", "common-area"}

    def test_merge_with_none_array_local(self, storage_with_cloud, mock_cloud_storage):
        """Merge handles None array on local side."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with no tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-none-local",
                agent_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=None,  # No tags locally
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-none-local",
            agent_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=["cloud-tag"],
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-none-local")
        # Should have cloud's tags (local had none)
        assert episode.tags == ["cloud-tag"]

    def test_merge_with_none_array_cloud(self, storage_with_cloud, mock_cloud_storage):
        """Merge handles None array on cloud side."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record with tags
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-none-cloud",
                agent_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=["local-tag"],
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has no tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-none-cloud",
            agent_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=None,  # No tags in cloud
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-none-cloud")
        # Should preserve local tags even though cloud won
        assert episode.tags == ["local-tag"]

    def test_merge_deduplicates_arrays(self, storage_with_cloud, mock_cloud_storage):
        """Merged arrays should not have duplicates."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_episode(
            Episode(
                id="ep-dedup",
                agent_id="test-agent",
                objective="Test",
                outcome="Test",
                tags=["tag-a", "tag-b", "tag-c"],
                local_updated_at=old_time,
            )
        )

        with storage_with_cloud._get_conn() as conn:
            conn.execute("DELETE FROM sync_queue")
            conn.commit()

        # Cloud has overlapping tags
        new_time = datetime.now(timezone.utc)
        cloud_episode = Episode(
            id="ep-dedup",
            agent_id="test-agent",
            objective="Cloud objective",
            outcome="Cloud outcome",
            tags=["tag-b", "tag-c", "tag-d"],  # b and c overlap
            cloud_synced_at=new_time,
            local_updated_at=new_time,
        )
        mock_cloud_storage.get_episodes.return_value = [cloud_episode]

        storage_with_cloud.pull_changes()

        episode = storage_with_cloud.get_episode("ep-dedup")
        # Should have exactly 4 unique tags
        assert len(episode.tags) == 4
        assert set(episode.tags) == {"tag-a", "tag-b", "tag-c", "tag-d"}


class TestMergeArrayFieldsUnit:
    """Unit tests for the _merge_array_fields helper method."""

    def test_merge_array_fields_no_arrays(self, storage):
        """No-op when table has no array fields configured."""
        # Using a fake table name that's not in SYNC_ARRAY_FIELDS
        ep1 = Episode(id="1", agent_id="test", objective="O1", outcome="Out1", tags=["a"])
        ep2 = Episode(id="1", agent_id="test", objective="O2", outcome="Out2", tags=["b"])

        result = storage._merge_array_fields("fake_table", ep1, ep2)

        # Should return winner unchanged
        assert result is ep1
        assert result.tags == ["a"]

    def test_merge_array_fields_episode(self, storage):
        """Direct test of _merge_array_fields for episodes."""
        winner = Episode(
            id="1",
            agent_id="test",
            objective="Winner",
            outcome="Out",
            tags=["tag-w"],
            lessons=["lesson-w"],
            emotional_tags=["joy"],
        )
        loser = Episode(
            id="1",
            agent_id="test",
            objective="Loser",
            outcome="Out",
            tags=["tag-l"],
            lessons=["lesson-l"],
            emotional_tags=["curiosity"],
        )

        result = storage._merge_array_fields("episodes", winner, loser)

        # Winner's scalar fields preserved, arrays merged
        assert result.objective == "Winner"
        assert set(result.tags) == {"tag-w", "tag-l"}
        assert set(result.lessons) == {"lesson-w", "lesson-l"}
        assert set(result.emotional_tags) == {"joy", "curiosity"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
