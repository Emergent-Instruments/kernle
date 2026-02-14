"""Tests for sync engine race window between save and queue cleanup (issue #731).

These tests verify that:
- Calling _merge_generic twice with the same record doesn't create duplicates
- Queue cleanup is atomic with the synced-at status update
- Recovery after simulated crash (record saved but queue not cleaned)
- The _record_already_applied guard detects previously-synced records
- _mark_synced_and_cleanup_queue consolidates operations into one transaction
- Conflict branches (cloud_time > local_time) skip save on crash recovery
- Both-timestamps-None fallback saves the cloud record instead of dropping it
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle.storage import (
    Belief,
    Episode,
    Note,
    SQLiteStorage,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    storage = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


@pytest.fixture
def mock_cloud_storage():
    """Create a mock cloud storage for testing sync."""
    mock = MagicMock(spec=SQLiteStorage)
    mock.stack_id = "test-agent"
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
        stack_id="test-agent", db_path=temp_db, cloud_storage=mock_cloud_storage
    )
    yield storage
    storage.close()


class TestMergeGenericIdempotency:
    """Test that _merge_generic is idempotent -- calling it twice with the
    same record must not create duplicates or leave orphaned queue entries."""

    def test_duplicate_merge_no_local_record_does_not_double_save(self, storage):
        """Calling _merge_generic twice for a new record skips save on the second call."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="idempotent-1",
            stack_id="test-agent",
            content="Cloud note",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        save_call_count = []

        def tracked_save():
            save_call_count.append(1)
            storage.save_note(cloud)

        # First merge: local_record is None, save_fn should be called
        count1, conflict1 = engine._merge_generic("notes", cloud, None, tracked_save)
        assert count1 == 1
        assert conflict1 is None
        assert len(save_call_count) == 1

        # Verify the note was saved
        with storage._connect() as conn:
            row = conn.execute(
                "SELECT id, version, cloud_synced_at FROM notes WHERE id = ?",
                ("idempotent-1",),
            ).fetchone()
        assert row is not None

        # Second merge: same cloud record, local_record is None
        # The save_fn should NOT be called because _record_already_applied detects it
        count2, conflict2 = engine._merge_generic("notes", cloud, None, tracked_save)
        assert count2 == 1
        assert conflict2 is None
        assert len(save_call_count) == 1  # Still 1 -- second call was skipped

    def test_duplicate_merge_does_not_leave_queue_entry(self, storage):
        """After a duplicate merge, no sync queue entry should remain."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="idempotent-2",
            stack_id="test-agent",
            content="Cloud note for queue test",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        # First merge
        engine._merge_generic("notes", cloud, None, lambda: storage.save_note(cloud))

        # Verify queue is clean after first merge
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE table_name = 'notes' AND record_id = ?",
                ("idempotent-2",),
            ).fetchone()[0]
        assert queue_count == 0

        # Second merge (simulating recovery)
        engine._merge_generic("notes", cloud, None, lambda: storage.save_note(cloud))

        # Queue should still be clean
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE table_name = 'notes' AND record_id = ?",
                ("idempotent-2",),
            ).fetchone()[0]
        assert queue_count == 0

    def test_duplicate_merge_with_cloud_time_no_local_time(self, storage):
        """The cloud_time-only branch also skips save on duplicate."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="idempotent-3",
            stack_id="test-agent",
            content="Cloud only time",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )
        # local_record exists but has no local_updated_at
        local = Note(
            id="idempotent-3",
            stack_id="test-agent",
            content="Local",
            local_updated_at=None,
            cloud_synced_at=None,
        )

        save_count = []

        def tracked_save():
            save_count.append(1)
            storage.save_note(cloud)

        # First call: cloud_time set, local_time None -> save_fn called
        count1, _ = engine._merge_generic("notes", cloud, local, tracked_save)
        assert count1 == 1
        assert len(save_count) == 1

        # Second call with same params (recovery scenario)
        count2, _ = engine._merge_generic("notes", cloud, local, tracked_save)
        assert count2 == 1
        assert len(save_count) == 1  # Skipped on second call


class TestQueueCleanupAtomicity:
    """Test that queue cleanup and synced-at marking happen in a single transaction."""

    def test_mark_synced_and_cleanup_single_transaction(self, storage):
        """_mark_synced_and_cleanup_queue runs in one transaction."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        # Save a note (creates a queue entry)
        storage.save_note(
            Note(
                id="atomic-cleanup-1",
                stack_id="test-agent",
                content="Test",
                local_updated_at=now,
            )
        )

        # Verify queue entry exists
        with storage._connect() as conn:
            queue_before = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE table_name = 'notes' "
                "AND record_id = 'atomic-cleanup-1'",
            ).fetchone()[0]
        assert queue_before >= 1

        # Run cleanup
        engine._mark_synced_and_cleanup_queue("notes", "atomic-cleanup-1")

        # Both synced-at and queue should be updated
        with storage._connect() as conn:
            note_row = conn.execute(
                "SELECT cloud_synced_at FROM notes WHERE id = 'atomic-cleanup-1'"
            ).fetchone()
            queue_after = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE table_name = 'notes' "
                "AND record_id = 'atomic-cleanup-1'",
            ).fetchone()[0]

        assert note_row["cloud_synced_at"] is not None
        assert queue_after == 0

    def test_cleanup_removes_all_queue_entries_for_record(self, storage):
        """Cleanup removes ALL queue entries for a table+record pair."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        # Save a note to create a queue entry
        storage.save_note(
            Note(
                id="multi-queue-1",
                stack_id="test-agent",
                content="First",
                local_updated_at=now,
            )
        )

        # Manually insert a second queue entry (simulating a race condition)
        with storage._connect() as conn:
            conn.execute(
                """INSERT INTO sync_queue
                   (table_name, record_id, operation, synced, queued_at, local_updated_at)
                   VALUES ('notes', 'multi-queue-1', 'upsert', 1, ?, ?)""",
                (now.isoformat(), now.isoformat()),
            )
            conn.commit()

        # Run cleanup -- should delete ALL entries for this record
        engine._mark_synced_and_cleanup_queue("notes", "multi-queue-1")

        with storage._connect() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE table_name = 'notes' "
                "AND record_id = 'multi-queue-1'",
            ).fetchone()[0]
        assert remaining == 0


class TestCrashRecoveryScenario:
    """Test recovery after simulated crash where record was saved but
    queue was not cleaned up."""

    def test_recovery_after_save_without_cleanup(self, storage):
        """Simulate crash: save_fn completes but queue cleanup never runs.

        On recovery (next sync), the record should be detected as already
        applied, save_fn skipped, and the queue cleaned up.
        """
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="crash-recovery-1",
            stack_id="test-agent",
            content="Cloud content",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        # Step 1: Simulate the first half of _merge_generic -- save the record
        # but do NOT clean up the queue (simulating a crash after save_fn)
        storage.save_note(cloud)

        # Manually set cloud_synced_at to simulate what _mark_synced would have done
        # if it had run partially before crash
        with storage._connect() as conn:
            conn.execute(
                "UPDATE notes SET cloud_synced_at = ? WHERE id = ?",
                (now.isoformat(), "crash-recovery-1"),
            )
            conn.commit()

        # Verify queue entry still exists (simulating post-crash state)
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'crash-recovery-1' AND synced = 0",
            ).fetchone()[0]
        assert queue_count >= 1, "Queue entry should exist (crash left it behind)"

        # Step 2: Recovery -- run _merge_generic again
        save_called = []

        def recovery_save():
            save_called.append(1)
            storage.save_note(cloud)

        count, conflict = engine._merge_generic("notes", cloud, None, recovery_save)

        # Should succeed but NOT call save_fn (duplicate detected)
        assert count == 1
        assert conflict is None
        assert len(save_called) == 0, "save_fn should be skipped during recovery"

        # Queue should now be cleaned up
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'crash-recovery-1'",
            ).fetchone()[0]
        assert queue_count == 0, "Queue should be clean after recovery"

    def test_recovery_logs_duplicate_detection(self, storage, caplog):
        """When a duplicate sync is detected, it should be logged."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="log-dup-1",
            stack_id="test-agent",
            content="Duplicate test",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        # First merge (normal)
        engine._merge_generic("notes", cloud, None, lambda: storage.save_note(cloud))

        # Second merge (recovery) -- should log
        with caplog.at_level(logging.INFO, logger="kernle.storage.sync_engine"):
            engine._merge_generic("notes", cloud, None, lambda: storage.save_note(cloud))

        assert any(
            "Duplicate sync detected" in msg for msg in caplog.messages
        ), f"Expected 'Duplicate sync detected' in log, got: {caplog.messages}"

    def test_recovery_works_for_episodes(self, storage):
        """Recovery scenario works for episodes, not just notes."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Episode(
            id="ep-crash-1",
            stack_id="test-agent",
            objective="Cloud episode",
            outcome="Test outcome",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        save_count = []

        def tracked_save():
            save_count.append(1)
            storage.save_episode(cloud)

        # First merge
        engine._merge_generic("episodes", cloud, None, tracked_save)
        assert len(save_count) == 1

        # Second merge (recovery)
        engine._merge_generic("episodes", cloud, None, tracked_save)
        assert len(save_count) == 1  # Skipped

        # Queue clean
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'ep-crash-1'",
            ).fetchone()[0]
        assert queue_count == 0


class TestRecordAlreadyApplied:
    """Test the _record_already_applied guard method."""

    def test_returns_false_for_nonexistent_record(self, storage):
        """Returns False when the record doesn't exist locally."""
        engine = storage._sync_engine
        cloud = Note(
            id="nonexistent-1",
            stack_id="test-agent",
            content="Ghost",
            version=1,
        )
        assert engine._record_already_applied("notes", cloud) is False

    def test_returns_false_for_record_without_cloud_synced_at(self, storage):
        """Returns False when the record exists but hasn't been marked as synced."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        # Save locally without cloud_synced_at
        storage.save_note(
            Note(
                id="no-sync-1",
                stack_id="test-agent",
                content="Local only",
                local_updated_at=now,
                version=1,
            )
        )

        # Clear cloud_synced_at to simulate pre-sync state
        with storage._connect() as conn:
            conn.execute("UPDATE notes SET cloud_synced_at = NULL WHERE id = 'no-sync-1'")
            conn.commit()

        cloud = Note(
            id="no-sync-1",
            stack_id="test-agent",
            content="Cloud version",
            version=1,
        )
        assert engine._record_already_applied("notes", cloud) is False

    def test_returns_true_for_matching_version_and_synced(self, storage):
        """Returns True when local record matches version and has cloud_synced_at set."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        # Save and mark as synced
        storage.save_note(
            Note(
                id="synced-1",
                stack_id="test-agent",
                content="Synced content",
                local_updated_at=now,
                cloud_synced_at=now,
                version=3,
            )
        )

        cloud = Note(
            id="synced-1",
            stack_id="test-agent",
            content="Same cloud content",
            version=3,
        )
        assert engine._record_already_applied("notes", cloud) is True

    def test_returns_false_for_version_mismatch(self, storage):
        """Returns False when local version differs from cloud version."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        # Save version 2 locally
        storage.save_note(
            Note(
                id="version-mismatch-1",
                stack_id="test-agent",
                content="V2 content",
                local_updated_at=now,
                cloud_synced_at=now,
                version=2,
            )
        )

        # Cloud has version 3
        cloud = Note(
            id="version-mismatch-1",
            stack_id="test-agent",
            content="V3 content",
            version=3,
        )
        assert engine._record_already_applied("notes", cloud) is False

    def test_works_for_beliefs(self, storage):
        """_record_already_applied works across different table types."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        storage.save_belief(
            Belief(
                id="belief-applied-1",
                stack_id="test-agent",
                statement="Test belief",
                local_updated_at=now,
                cloud_synced_at=now,
                version=1,
            )
        )

        cloud = Belief(
            id="belief-applied-1",
            stack_id="test-agent",
            statement="Cloud belief",
            version=1,
        )
        assert engine._record_already_applied("beliefs", cloud) is True


class TestEndToEndSyncRecovery:
    """Integration tests that verify the full sync flow handles
    the race window correctly."""

    def test_full_pull_idempotent_for_same_record(self, storage_with_cloud, mock_cloud_storage):
        """Pulling the same record twice via full sync does not create duplicates."""
        now = datetime.now(timezone.utc)
        cloud_note = Note(
            id="pull-idem-1",
            stack_id="test-agent",
            content="Cloud note",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        # Configure cloud to return the same note on both pulls
        mock_cloud_storage.get_notes.return_value = [cloud_note]
        mock_cloud_storage.list_playbooks.return_value = []

        # First pull
        result1 = storage_with_cloud.pull_changes()
        assert result1.pulled >= 1

        # Second pull (simulating repeated sync)
        storage_with_cloud.pull_changes()

        # The note should exist exactly once
        with storage_with_cloud._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM notes WHERE id = 'pull-idem-1'").fetchone()[
                0
            ]
        assert count == 1

        # No queue entries should remain
        with storage_with_cloud._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'pull-idem-1' AND synced = 0"
            ).fetchone()[0]
        assert queue_count == 0

    def test_merge_generic_save_fn_requeue_is_cleaned_up(self, storage):
        """When save_fn internally re-queues the record, the cleanup
        step should remove that queue entry."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)

        cloud = Note(
            id="requeue-cleanup-1",
            stack_id="test-agent",
            content="Will be re-queued by save",
            local_updated_at=now,
            cloud_synced_at=now,
            version=1,
        )

        # save_note internally calls _queue_sync, creating a new queue entry
        engine._merge_generic("notes", cloud, None, lambda: storage.save_note(cloud))

        # Despite save_fn creating a queue entry, cleanup should have removed it
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'requeue-cleanup-1' AND synced = 0"
            ).fetchone()[0]
        assert queue_count == 0, (
            "Queue entry created by save_fn should be cleaned up by "
            "_mark_synced_and_cleanup_queue"
        )


class TestConflictBranchCrashRecovery:
    """Test that conflict resolution branches (cloud_time > local_time, tie-break)
    skip _save_from_cloud when the record was already applied (crash recovery)."""

    def test_cloud_wins_branch_skips_save_on_recovery(self, storage):
        """cloud_time > local_time: if the record was already saved (crash recovery),
        _save_from_cloud should be skipped."""
        engine = storage._sync_engine
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(minutes=5)

        local = Note(
            id="conflict-recovery-1",
            stack_id="test-agent",
            content="Local version",
            local_updated_at=earlier,
        )
        cloud = Note(
            id="conflict-recovery-1",
            stack_id="test-agent",
            content="Cloud version (newer)",
            local_updated_at=now,
            cloud_synced_at=now,
            version=2,
        )

        # Pre-save: put a local record in place so the conflict branch is hit
        storage.save_note(local)

        # First merge: cloud_time > local_time, should save
        save_fn_calls = []
        count1, conflict1 = engine._merge_generic(
            "notes", cloud, local, lambda: save_fn_calls.append(1)
        )
        assert count1 == 1
        assert conflict1 is not None
        assert conflict1.policy_decision == "newer_cloud_timestamp"

        # Simulate crash recovery: the record exists with matching version + cloud_synced_at.
        # Manually set version and cloud_synced_at to match cloud record (as _save_from_cloud would).
        with storage._connect() as conn:
            conn.execute(
                "UPDATE notes SET version = ?, cloud_synced_at = ? WHERE id = ?",
                (2, now.isoformat(), "conflict-recovery-1"),
            )
            conn.commit()

        # Second merge (recovery): _save_from_cloud should be skipped
        with patch.object(engine, "_save_from_cloud", wraps=engine._save_from_cloud) as mock_save:
            count2, conflict2 = engine._merge_generic(
                "notes", cloud, local, lambda: save_fn_calls.append(1)
            )
            mock_save.assert_not_called()

        assert count2 == 1
        assert conflict2 is not None

    def test_tie_break_branch_skips_save_on_recovery(self, storage):
        """Equal timestamps with different content: tie-break branch skips save
        when record was already applied."""
        engine = storage._sync_engine
        shared_time = datetime.now(timezone.utc)

        local = Note(
            id="tie-recovery-1",
            stack_id="test-agent",
            content="local-content",
            tags=["local"],
            local_updated_at=shared_time,
        )
        cloud = Note(
            id="tie-recovery-1",
            stack_id="test-agent",
            content="cloud-content",
            tags=["cloud"],
            local_updated_at=shared_time,
            cloud_synced_at=shared_time,
            version=3,
        )

        # Pre-save the local record
        storage.save_note(local)

        # First merge to establish the tie-break
        count1, conflict1 = engine._merge_generic("notes", cloud, local, lambda: None)
        assert conflict1 is not None
        assert conflict1.policy_decision in {"local_wins_tie_hash", "cloud_wins_tie_hash"}

        # Set matching version and cloud_synced_at to simulate recovery state
        with storage._connect() as conn:
            conn.execute(
                "UPDATE notes SET version = ?, cloud_synced_at = ? WHERE id = ?",
                (3, shared_time.isoformat(), "tie-recovery-1"),
            )
            conn.commit()

        # Second merge (recovery): _save_from_cloud should be skipped
        with patch.object(engine, "_save_from_cloud", wraps=engine._save_from_cloud) as mock_save:
            count2, conflict2 = engine._merge_generic("notes", cloud, local, lambda: None)
            mock_save.assert_not_called()


class TestBothTimestampsNoneFallback:
    """Test that when both cloud_time and local_time are None, the cloud record
    is saved as a fallback instead of being silently dropped."""

    def test_both_timestamps_none_saves_cloud_record(self, storage):
        """When both timestamps are None, save_fn should be called and count = 1."""
        engine = storage._sync_engine

        cloud = Note(
            id="no-timestamp-1",
            stack_id="test-agent",
            content="Cloud with no timestamps",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        local = Note(
            id="no-timestamp-1",
            stack_id="test-agent",
            content="Local with no timestamps",
            cloud_synced_at=None,
            local_updated_at=None,
        )

        save_called = []
        count, conflict = engine._merge_generic(
            "notes", cloud, local, lambda: save_called.append(True)
        )

        assert count == 1, "Both-timestamps-None should save (count=1), not drop (count=0)"
        assert conflict is None
        assert len(save_called) == 1, "save_fn must be called for the fallback"

    def test_both_timestamps_none_logs_warning(self, storage, caplog):
        """The fallback path should log a warning about missing timestamps."""
        engine = storage._sync_engine

        cloud = Note(
            id="no-timestamp-warn",
            stack_id="test-agent",
            content="Cloud",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        local = Note(
            id="no-timestamp-warn",
            stack_id="test-agent",
            content="Local",
            cloud_synced_at=None,
            local_updated_at=None,
        )

        with caplog.at_level(logging.WARNING, logger="kernle.storage.sync_engine"):
            engine._merge_generic("notes", cloud, local, lambda: storage.save_note(cloud))

        assert any(
            "no timestamps" in msg for msg in caplog.messages
        ), f"Expected 'no timestamps' warning in log, got: {caplog.messages}"

    def test_both_timestamps_none_cleans_up_queue(self, storage):
        """The fallback path should clean up the sync queue after saving."""
        engine = storage._sync_engine

        cloud = Note(
            id="no-timestamp-queue",
            stack_id="test-agent",
            content="Cloud",
            cloud_synced_at=None,
            local_updated_at=None,
        )
        local = Note(
            id="no-timestamp-queue",
            stack_id="test-agent",
            content="Local",
            cloud_synced_at=None,
            local_updated_at=None,
        )

        engine._merge_generic("notes", cloud, local, lambda: storage.save_note(cloud))

        # Queue should be cleaned up
        with storage._connect() as conn:
            queue_count = conn.execute(
                "SELECT COUNT(*) FROM sync_queue WHERE record_id = 'no-timestamp-queue' AND synced = 0"
            ).fetchone()[0]
        assert queue_count == 0, "Queue should be cleaned up after fallback save"
