"""Tests for kernle.storage.beliefs_crud — extracted beliefs CRUD module.

Verifies the extracted functions work correctly with injected dependencies,
independent of SQLiteStorage.
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.storage.beliefs_crud import (
    find_belief,
    get_belief,
    get_belief_by_id,
    get_beliefs,
    save_belief,
    save_beliefs_batch,
    update_belief_atomic,
)
from kernle.types import Belief

STACK_ID = "test-beliefs-crud"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "beliefs_crud_test.db"


@pytest.fixture
def db(db_path):
    """Create a minimal DB with the beliefs table schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY,
            stack_id TEXT,
            statement TEXT,
            belief_type TEXT DEFAULT 'fact',
            confidence REAL DEFAULT 0.5,
            created_at TEXT,
            source_type TEXT,
            source_episodes TEXT,
            derived_from TEXT,
            last_verified TEXT,
            verification_count INTEGER DEFAULT 0,
            confidence_history TEXT,
            supersedes TEXT,
            superseded_by TEXT,
            times_reinforced INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            times_accessed INTEGER DEFAULT 0,
            last_accessed TEXT,
            is_protected INTEGER DEFAULT 0,
            strength REAL DEFAULT 1.0,
            context TEXT,
            context_tags TEXT,
            source_entity TEXT,
            subject_ids TEXT,
            access_grants TEXT,
            consent_grants TEXT,
            processed INTEGER DEFAULT 0,
            belief_scope TEXT DEFAULT 'world',
            source_domain TEXT,
            cross_domain_applications TEXT,
            abstraction_level TEXT DEFAULT 'specific',
            epoch_id TEXT,
            local_updated_at TEXT,
            cloud_synced_at TEXT,
            version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def connect_fn(db):
    """Connection factory matching SQLiteStorage._connect() pattern."""

    @contextmanager
    def _connect():
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _connect


@pytest.fixture
def now_fn():
    return lambda: "2025-06-01T00:00:00+00:00"


@pytest.fixture
def to_json():
    import json

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, datetime):
                return o.isoformat()
            return super().default(o)

    def _to_json(obj):
        if obj is None:
            return None
        return json.dumps(obj, cls=_Encoder)

    return _to_json


@pytest.fixture
def record_to_dict():
    def _record_to_dict(record):
        from dataclasses import asdict

        return asdict(record)

    return _record_to_dict


@pytest.fixture
def queue_sync():
    return MagicMock()


@pytest.fixture
def save_embedding():
    return MagicMock()


@pytest.fixture
def sync_to_file():
    return MagicMock()


@pytest.fixture
def build_access_filter():
    def _build(requesting_entity):
        if requesting_entity is None:
            return ("", [])
        return (" AND access_grants LIKE ?", [f'%"{requesting_entity}"%'])

    return _build


def _make_belief(**overrides) -> Belief:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": STACK_ID,
        "statement": "Test belief statement",
        "belief_type": "fact",
        "confidence": 0.8,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Belief(**defaults)


class TestSaveBelief:
    def test_saves_and_returns_id(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief()
        result = save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        assert result == belief.id

    def test_generates_id_if_missing(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief(id="")
        result = save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        assert result  # non-empty UUID
        assert belief.id == result

    def test_queues_sync(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief()
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        queue_sync.assert_called_once()
        args = queue_sync.call_args
        assert args[0][1] == "beliefs"
        assert args[0][2] == belief.id
        assert args[0][3] == "upsert"

    def test_saves_embedding(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief(statement="Sky is blue")
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        save_embedding.assert_called_once()
        args = save_embedding.call_args[0]
        assert args[1] == "beliefs"
        assert args[3] == "Sky is blue"

    def test_syncs_to_file(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief()
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        sync_to_file.assert_called_once()

    def test_calls_lineage_checker(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        checker = MagicMock()
        belief = _make_belief(derived_from=["episode:abc"])
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
            lineage_checker=checker,
        )
        checker.assert_called_once_with("belief", belief.id, ["episode:abc"])


class TestGetBeliefs:
    def test_returns_empty_for_no_beliefs(self, connect_fn, build_access_filter):
        result = get_beliefs(connect_fn, STACK_ID, build_access_filter)
        assert result == []

    def test_returns_saved_beliefs(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        b1 = _make_belief(statement="Belief one")
        b2 = _make_belief(statement="Belief two")
        for b in [b1, b2]:
            save_belief(
                connect_fn,
                STACK_ID,
                b,
                now_fn,
                to_json,
                record_to_dict,
                queue_sync,
                save_embedding,
                sync_to_file,
            )

        result = get_beliefs(connect_fn, STACK_ID, build_access_filter)
        assert len(result) == 2

    def test_respects_limit(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        for i in range(5):
            save_belief(
                connect_fn,
                STACK_ID,
                _make_belief(statement=f"Belief {i}"),
                now_fn,
                to_json,
                record_to_dict,
                queue_sync,
                save_embedding,
                sync_to_file,
            )

        result = get_beliefs(connect_fn, STACK_ID, build_access_filter, limit=3)
        assert len(result) == 3

    def test_excludes_inactive_by_default(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        active = _make_belief(statement="Active", is_active=True)
        inactive = _make_belief(statement="Inactive", is_active=False)
        for b in [active, inactive]:
            save_belief(
                connect_fn,
                STACK_ID,
                b,
                now_fn,
                to_json,
                record_to_dict,
                queue_sync,
                save_embedding,
                sync_to_file,
            )

        result = get_beliefs(connect_fn, STACK_ID, build_access_filter)
        assert len(result) == 1
        assert result[0].statement == "Active"

    def test_includes_inactive_when_requested(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        active = _make_belief(statement="Active", is_active=True)
        inactive = _make_belief(statement="Inactive", is_active=False)
        for b in [active, inactive]:
            save_belief(
                connect_fn,
                STACK_ID,
                b,
                now_fn,
                to_json,
                record_to_dict,
                queue_sync,
                save_embedding,
                sync_to_file,
            )

        result = get_beliefs(connect_fn, STACK_ID, build_access_filter, include_inactive=True)
        assert len(result) == 2


class TestFindBelief:
    def test_finds_by_statement(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief(statement="Unique statement")
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )

        result = find_belief(connect_fn, STACK_ID, "Unique statement")
        assert result is not None
        assert result.id == belief.id

    def test_returns_none_for_missing(self, connect_fn):
        result = find_belief(connect_fn, STACK_ID, "Nonexistent")
        assert result is None


class TestGetBelief:
    def test_gets_by_id(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        belief = _make_belief()
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )

        result = get_belief(connect_fn, STACK_ID, belief.id, build_access_filter)
        assert result is not None
        assert result.id == belief.id

    def test_returns_none_for_missing_id(self, connect_fn, build_access_filter):
        result = get_belief(connect_fn, STACK_ID, "no-such-id", build_access_filter)
        assert result is None


class TestGetBeliefById:
    def test_gets_internal_by_id(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief()
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )

        result = get_belief_by_id(connect_fn, STACK_ID, belief.id)
        assert result is not None
        assert result.statement == belief.statement


class TestUpdateBeliefAtomic:
    def test_updates_successfully(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief(version=1)
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )

        belief.statement = "Updated statement"
        result = update_belief_atomic(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
            expected_version=1,
        )
        assert result is True
        assert belief.version == 2

    def test_version_conflict_raises(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        from kernle.storage import VersionConflictError

        belief = _make_belief(version=1)
        save_belief(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )

        belief.statement = "Conflict test"
        with pytest.raises(VersionConflictError):
            update_belief_atomic(
                connect_fn,
                STACK_ID,
                belief,
                now_fn,
                to_json,
                record_to_dict,
                queue_sync,
                save_embedding,
                sync_to_file,
                expected_version=99,
            )

    def test_returns_false_for_missing_belief(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        belief = _make_belief()
        result = update_belief_atomic(
            connect_fn,
            STACK_ID,
            belief,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
            expected_version=1,
        )
        assert result is False


class TestSaveBeliefsBatch:
    def test_saves_multiple(
        self,
        connect_fn,
        now_fn,
        to_json,
        record_to_dict,
        queue_sync,
        save_embedding,
        sync_to_file,
        build_access_filter,
    ):
        beliefs = [_make_belief(statement=f"Batch {i}") for i in range(3)]
        ids = save_beliefs_batch(
            connect_fn,
            STACK_ID,
            beliefs,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        assert len(ids) == 3
        result = get_beliefs(connect_fn, STACK_ID, build_access_filter)
        assert len(result) == 3

    def test_empty_batch_returns_empty(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        ids = save_beliefs_batch(
            connect_fn,
            STACK_ID,
            [],
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        assert ids == []

    def test_syncs_file_once(
        self, connect_fn, now_fn, to_json, record_to_dict, queue_sync, save_embedding, sync_to_file
    ):
        beliefs = [_make_belief() for _ in range(3)]
        save_beliefs_batch(
            connect_fn,
            STACK_ID,
            beliefs,
            now_fn,
            to_json,
            record_to_dict,
            queue_sync,
            save_embedding,
            sync_to_file,
        )
        sync_to_file.assert_called_once()
