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
    SyncResult,
    Value,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    path = Path(tempfile.mktemp(suffix='.db'))
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
        agent_id="test-agent",
        db_path=temp_db,
        cloud_storage=mock_cloud_storage
    )
    yield storage
    storage.close()


class TestSyncQueueBasics:
    """Test the sync queue functionality."""

    def test_changes_are_queued_on_save(self, storage):
        """Saving a record should queue it for sync."""
        initial_count = storage.get_pending_sync_count()

        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test note"
        ))

        assert storage.get_pending_sync_count() == initial_count + 1

    def test_multiple_saves_same_record_dedupe(self, storage):
        """Multiple saves of the same record should dedupe in queue."""
        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="First version"
        ))

        count_after_first = storage.get_pending_sync_count()

        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Second version"
        ))

        # Should still be same count (deduped)
        assert storage.get_pending_sync_count() == count_after_first

    def test_get_queued_changes(self, storage):
        """Can retrieve queued changes."""
        storage.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Test",
            outcome="Test"
        ))
        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test note"
        ))

        changes = storage.get_queued_changes()

        assert len(changes) >= 2
        assert all(isinstance(c, QueuedChange) for c in changes)

        tables = {c.table_name for c in changes}
        assert "episodes" in tables
        assert "notes" in tables

    def test_queued_change_has_timestamp(self, storage):
        """Queued changes should have a timestamp."""
        storage.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test"
        ))

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
        storage_with_cloud.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Test objective",
            outcome="Test outcome"
        ))
        storage_with_cloud.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test note"
        ))

        # Sync
        result = storage_with_cloud.sync()

        assert result.pushed >= 2
        assert mock_cloud_storage.save_episode.called
        assert mock_cloud_storage.save_note.called

    def test_sync_clears_queue_on_success(self, storage_with_cloud, mock_cloud_storage):
        """Successful sync should clear the queue."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test"
        ))

        assert storage_with_cloud.get_pending_sync_count() > 0

        storage_with_cloud.sync()

        # Queue should be cleared
        assert storage_with_cloud.get_pending_sync_count() == 0

    def test_sync_marks_records_synced(self, storage_with_cloud, mock_cloud_storage):
        """Synced records should have cloud_synced_at set."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test"
        ))

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

        storage_with_cloud.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Test"
        ))

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
        assert result.conflicts == 0


class TestConflictResolution:
    """Test conflict resolution with last-write-wins."""

    def test_cloud_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Cloud record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage_with_cloud.save_note(Note(
            id="conflict-note",
            agent_id="test-agent",
            content="Local version",
            local_updated_at=old_time,
        ))

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
        assert result.conflicts >= 1

        # Local should now have cloud's content
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Cloud version" in conflict_note.content

    def test_local_wins_when_newer(self, storage_with_cloud, mock_cloud_storage):
        """Local record wins when it's newer."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        # Create local record (newer)
        new_time = datetime.now(timezone.utc)
        storage_with_cloud.save_note(Note(
            id="conflict-note",
            agent_id="test-agent",
            content="Local version (newer)",
            local_updated_at=new_time,
        ))

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
        assert result.conflicts >= 1

        # Local version should be preserved
        notes = storage_with_cloud.get_notes()
        conflict_note = next(n for n in notes if n.id == "conflict-note")
        assert "Local version" in conflict_note.content


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
            agent_id="test-agent",
            db_path=temp_db,
            cloud_storage=mock_cloud_storage
        )
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}
        storage1.sync()

        sync_time = storage1.get_last_sync_time()
        storage1.close()

        # Second instance
        storage2 = SQLiteStorage(
            agent_id="test-agent",
            db_path=temp_db,
            cloud_storage=mock_cloud_storage
        )

        # Should have same last sync time
        assert storage2.get_last_sync_time() == sync_time
        storage2.close()


class TestSyncAllRecordTypes:
    """Test that sync works for all record types."""

    def test_sync_episodes(self, storage_with_cloud, mock_cloud_storage):
        """Episodes should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_episode(Episode(
            id="ep1",
            agent_id="test-agent",
            objective="Test",
            outcome="Test"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_episode.called

    def test_sync_beliefs(self, storage_with_cloud, mock_cloud_storage):
        """Beliefs should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_belief(Belief(
            id="b1",
            agent_id="test-agent",
            statement="Test belief"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_belief.called

    def test_sync_values(self, storage_with_cloud, mock_cloud_storage):
        """Values should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_value(Value(
            id="v1",
            agent_id="test-agent",
            name="Test",
            statement="Test value"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_value.called

    def test_sync_goals(self, storage_with_cloud, mock_cloud_storage):
        """Goals should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_goal(Goal(
            id="g1",
            agent_id="test-agent",
            title="Test goal"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_goal.called

    def test_sync_drives(self, storage_with_cloud, mock_cloud_storage):
        """Drives should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_drive(Drive(
            id="d1",
            agent_id="test-agent",
            drive_type="curiosity"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_drive.called

    def test_sync_relationships(self, storage_with_cloud, mock_cloud_storage):
        """Relationships should sync."""
        mock_cloud_storage.get_stats.return_value = {"episodes": 0}

        storage_with_cloud.save_relationship(Relationship(
            id="r1",
            agent_id="test-agent",
            entity_name="Alice",
            entity_type="human",
            relationship_type="friend"
        ))

        storage_with_cloud.sync()

        assert mock_cloud_storage.save_relationship.called


class TestOfflineQueuing:
    """Test that changes are properly queued when offline."""

    def test_operations_work_offline(self, storage):
        """All operations should work without cloud configured."""
        # These should all succeed
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
        storage1.save_note(Note(
            id="n1", agent_id="test-agent",
            content="Test"
        ))

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
        storage_with_cloud.save_note(Note(
            id="n1",
            agent_id="test-agent",
            content="Will be deleted"
        ))

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
        storage_with_cloud.save_note(Note(
            id="n1", agent_id="test-agent", content="Note 1"
        ))
        storage_with_cloud.save_note(Note(
            id="n2", agent_id="test-agent", content="Note 2"
        ))

        # First call fails, second succeeds
        mock_cloud_storage.save_note.side_effect = [
            Exception("First failed"),
            None  # Success
        ]

        result = storage_with_cloud.sync()

        # Should have pushed one successfully
        assert result.pushed >= 1
        # Should have recorded the error
        assert len(result.errors) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
