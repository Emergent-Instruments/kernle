"""Tests for sync conflict durable artifacts.

Tests:
- diff_hash fingerprinting on CLI pull and push conflict save paths
- save_sync_conflict deduplication by diff_hash
- save_sync_conflict allows null diff_hash (no dedup check)
- diff_hash determinism (same payloads produce same hash)
"""

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from kernle.storage import SQLiteStorage, SyncConflict


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    yield s
    s.close()


def _make_conflict(
    *,
    table="episodes",
    record_id=None,
    local_version=None,
    cloud_version=None,
    resolution="pull_apply_failed",
    diff_hash=None,
):
    """Helper to build a SyncConflict with sensible defaults."""
    record_id = record_id or str(uuid4())
    local_version = local_version or {"payload": {"table": table, "record_id": record_id}}
    cloud_version = cloud_version or {"error": "test error"}
    return SyncConflict(
        id=str(uuid4()),
        table=table,
        record_id=record_id,
        local_version=local_version,
        cloud_version=cloud_version,
        resolution=resolution,
        resolved_at=datetime.now(timezone.utc),
        local_summary="local",
        cloud_summary="cloud",
        diff_hash=diff_hash,
    )


def _compute_diff_hash(local_version, cloud_version):
    """Compute diff_hash the same way the CLI conflict paths do."""
    diff_payload = json.dumps(
        {"local": local_version, "cloud": cloud_version}, sort_keys=True, default=str
    )
    return hashlib.sha256(diff_payload.encode("utf-8")).hexdigest()


class TestSaveSyncConflictDeduplication:
    """save_sync_conflict should deduplicate by diff_hash when present."""

    def test_deduplicates_by_diff_hash(self, storage):
        """Saving the same conflict twice (same diff_hash) should produce only 1 row."""
        local_v = {"payload": {"table": "notes", "record_id": "r1"}}
        cloud_v = {"error": "version mismatch"}
        diff_hash = _compute_diff_hash(local_v, cloud_v)

        c1 = _make_conflict(
            table="notes",
            record_id="r1",
            local_version=local_v,
            cloud_version=cloud_v,
            diff_hash=diff_hash,
        )
        c2 = _make_conflict(
            table="notes",
            record_id="r1",
            local_version=local_v,
            cloud_version=cloud_v,
            diff_hash=diff_hash,
        )

        id1 = storage.save_sync_conflict(c1)
        id2 = storage.save_sync_conflict(c2)

        # Second save should return the existing row's ID
        assert id1 == c1.id
        assert id2 == c1.id  # deduped to original

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].diff_hash == diff_hash

    def test_different_diff_hash_creates_separate_rows(self, storage):
        """Conflicts with different diff_hash values should each be stored."""
        c1 = _make_conflict(
            table="notes",
            record_id="r1",
            local_version={"a": 1},
            cloud_version={"b": 2},
            diff_hash=_compute_diff_hash({"a": 1}, {"b": 2}),
        )
        c2 = _make_conflict(
            table="notes",
            record_id="r1",
            local_version={"a": 99},
            cloud_version={"b": 100},
            diff_hash=_compute_diff_hash({"a": 99}, {"b": 100}),
        )

        storage.save_sync_conflict(c1)
        storage.save_sync_conflict(c2)

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 2

    def test_allows_null_diff_hash(self, storage):
        """Saving conflicts with diff_hash=None should always insert (no dedup)."""
        c1 = _make_conflict(diff_hash=None)
        c2 = _make_conflict(diff_hash=None)

        storage.save_sync_conflict(c1)
        storage.save_sync_conflict(c2)

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 2
        assert conflicts[0].diff_hash is None
        assert conflicts[1].diff_hash is None

    def test_null_diff_hash_does_not_collide_with_existing(self, storage):
        """A None-hash conflict should not match an existing hashed conflict."""
        local_v = {"x": 1}
        cloud_v = {"y": 2}
        diff_hash = _compute_diff_hash(local_v, cloud_v)

        c_hashed = _make_conflict(
            local_version=local_v,
            cloud_version=cloud_v,
            diff_hash=diff_hash,
        )
        c_null = _make_conflict(
            local_version=local_v,
            cloud_version=cloud_v,
            diff_hash=None,
        )

        storage.save_sync_conflict(c_hashed)
        storage.save_sync_conflict(c_null)

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 2


class TestDiffHashDeterminism:
    """diff_hash computation must be deterministic and stable."""

    def test_same_payloads_produce_same_hash(self):
        """Identical local/cloud payloads must yield identical diff_hash."""
        local_v = {"payload": {"table": "episodes", "record_id": "ep-1"}}
        cloud_v = {"operation": {"table": "episodes"}, "error": "conflict"}

        hash1 = _compute_diff_hash(local_v, cloud_v)
        hash2 = _compute_diff_hash(local_v, cloud_v)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_different_payloads_produce_different_hash(self):
        """Different payloads must yield different diff_hash values."""
        local_v = {"payload": {"table": "episodes", "record_id": "ep-1"}}
        cloud_v_a = {"error": "conflict A"}
        cloud_v_b = {"error": "conflict B"}

        hash_a = _compute_diff_hash(local_v, cloud_v_a)
        hash_b = _compute_diff_hash(local_v, cloud_v_b)

        assert hash_a != hash_b

    def test_key_order_does_not_affect_hash(self):
        """JSON sort_keys=True means key order in source dicts does not matter."""
        local_a = {"z_key": 1, "a_key": 2}
        local_b = {"a_key": 2, "z_key": 1}
        cloud = {"error": "test"}

        assert _compute_diff_hash(local_a, cloud) == _compute_diff_hash(local_b, cloud)

    def test_hash_matches_sync_engine_build_conflict_hash(self, storage):
        """CLI diff_hash computation should match sync engine's _build_conflict_hash."""
        local_v = {"payload": {"table": "notes", "record_id": "n-1"}}
        cloud_v = {"operation": "update", "error": "version mismatch"}

        cli_hash = _compute_diff_hash(local_v, cloud_v)
        engine_hash = storage._sync_engine._build_conflict_hash(local_v, cloud_v)

        assert cli_hash == engine_hash


class TestPullApplyConflictDiffHash:
    """Verify that _save_pull_apply_conflict sets diff_hash on the conflict.

    Since _save_pull_apply_conflict is a nested function inside cmd_sync, we
    test the behavior indirectly by saving a conflict the same way the function
    does, and verifying the diff_hash is present and correct.
    """

    def test_pull_conflict_diff_hash_is_set(self, storage):
        """A pull-apply conflict should have a non-None diff_hash after save."""
        op = {"table": "episodes", "record_id": "ep-1", "operation": "upsert"}
        error = "unhandled pull operation for table=episodes"

        local_version = {"payload": op}
        cloud_version = {"operation": op, "conflict_envelope": {"error": error}}
        diff_hash = _compute_diff_hash(local_version, cloud_version)

        conflict = SyncConflict(
            id=str(uuid4()),
            table="episodes",
            record_id="ep-1",
            local_version=local_version,
            cloud_version=cloud_version,
            resolution="pull_apply_failed",
            resolved_at=datetime.now(timezone.utc),
            local_summary="apply failed",
            cloud_summary=f"pull apply failed: {error}"[:200],
            diff_hash=diff_hash,
        )
        storage.save_sync_conflict(conflict)

        saved = storage.get_sync_conflicts()
        assert len(saved) == 1
        assert saved[0].diff_hash is not None
        assert saved[0].diff_hash == diff_hash

    def test_pull_conflict_deduplicates_on_retry(self, storage):
        """The same pull-apply failure saved twice should not create duplicates."""
        op = {"table": "notes", "record_id": "n-1", "operation": "upsert"}
        local_version = {"payload": op}
        cloud_version = {"operation": op, "conflict_envelope": {"error": "fail"}}
        diff_hash = _compute_diff_hash(local_version, cloud_version)

        for _ in range(3):
            conflict = SyncConflict(
                id=str(uuid4()),
                table="notes",
                record_id="n-1",
                local_version=local_version,
                cloud_version=cloud_version,
                resolution="pull_apply_failed",
                resolved_at=datetime.now(timezone.utc),
                local_summary="apply failed",
                cloud_summary="pull apply failed: fail",
                diff_hash=diff_hash,
            )
            storage.save_sync_conflict(conflict)

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 1


class TestPushApplyConflictDiffHash:
    """Verify that _save_push_apply_conflict sets diff_hash on the conflict.

    Same indirect-testing approach as TestPullApplyConflictDiffHash.
    """

    def test_push_conflict_diff_hash_is_set(self, storage):
        """A push-apply conflict should have a non-None diff_hash after save."""
        table = "notes"
        record_id = "n-1"
        operation = "update"
        error = "version conflict"

        local_version = {
            "operation": {
                "table": table,
                "record_id": record_id,
                "operation": operation,
            },
            "payload_hash": "abc123",
            "payload_snapshot": '{"content":"hello"}',
            "operation_identity": ("notes", "n-1"),
            "timestamp": "2025-01-01T00:00:00Z",
        }
        cloud_version = {
            "backend_payload": {"raw": {}},
            "error": error,
        }
        diff_hash = _compute_diff_hash(local_version, cloud_version)

        conflict = SyncConflict(
            id=str(uuid4()),
            table=table,
            record_id=record_id,
            local_version=local_version,
            cloud_version=cloud_version,
            resolution="backend_rejected",
            resolved_at=datetime.now(timezone.utc),
            local_summary=f"{table}:{record_id} {operation}",
            cloud_summary=error,
            diff_hash=diff_hash,
        )
        storage.save_sync_conflict(conflict)

        saved = storage.get_sync_conflicts()
        assert len(saved) == 1
        assert saved[0].diff_hash is not None
        assert saved[0].diff_hash == diff_hash

    def test_push_conflict_deduplicates_on_retry(self, storage):
        """The same push-apply failure saved twice should not create duplicates."""
        local_version = {
            "operation": {"table": "notes", "record_id": "n-2", "operation": "update"},
            "payload_hash": "def456",
        }
        cloud_version = {"error": "rejected"}
        diff_hash = _compute_diff_hash(local_version, cloud_version)

        for _ in range(3):
            conflict = SyncConflict(
                id=str(uuid4()),
                table="notes",
                record_id="n-2",
                local_version=local_version,
                cloud_version=cloud_version,
                resolution="backend_rejected",
                resolved_at=datetime.now(timezone.utc),
                local_summary="notes:n-2 update",
                cloud_summary="rejected",
                diff_hash=diff_hash,
            )
            storage.save_sync_conflict(conflict)

        conflicts = storage.get_sync_conflicts()
        assert len(conflicts) == 1
