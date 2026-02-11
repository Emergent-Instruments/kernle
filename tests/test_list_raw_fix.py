"""Tests for list_raw returning full RawEntry objects and AnxietyMixin compat."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from kernle.stack import SQLiteStack
from kernle.types import RawEntry


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_list_raw.db"


@pytest.fixture
def stack(tmp_db):
    return SQLiteStack(
        stack_id="test-stack", db_path=tmp_db, components=[], enforce_provenance=False
    )


def _save_raw(stack, blob="test blob"):
    entry = RawEntry(
        id=str(uuid.uuid4()),
        stack_id="test-stack",
        blob=blob,
        source="cli",
    )
    rid = stack.save_raw(entry)
    return rid


class TestListRawReturnsRawEntry:
    """list_raw must return RawEntry objects, not stripped dicts."""

    def test_returns_raw_entry_objects(self, stack):
        _save_raw(stack)
        entries = stack.list_raw()
        assert len(entries) == 1
        assert isinstance(entries[0], RawEntry)

    def test_has_processed_into_field(self, stack):
        _save_raw(stack)
        entries = stack.list_raw()
        entry = entries[0]
        assert hasattr(entry, "processed_into")

    def test_has_all_fields(self, stack):
        _save_raw(stack, blob="full check")
        entries = stack.list_raw()
        entry = entries[0]
        for field in (
            "id",
            "stack_id",
            "blob",
            "captured_at",
            "source",
            "processed",
            "processed_into",
            "version",
            "deleted",
        ):
            assert hasattr(entry, field), f"Missing field: {field}"

    def test_processed_true_filter(self, stack):
        rid1 = _save_raw(stack, blob="will be processed")
        _save_raw(stack, blob="stays unprocessed")
        # Mark one as processed via the backend API
        stack._backend.mark_raw_processed(rid1, ["episode:test123"])
        entries = stack.list_raw(processed=True)
        assert len(entries) == 1
        assert all(e.processed for e in entries)

    def test_processed_false_filter(self, stack):
        rid1 = _save_raw(stack, blob="will be processed")
        _save_raw(stack, blob="stays unprocessed")
        stack._backend.mark_raw_processed(rid1, ["episode:test123"])
        entries = stack.list_raw(processed=False)
        assert len(entries) == 1
        assert all(not e.processed for e in entries)

    def test_limit(self, stack):
        for _ in range(5):
            _save_raw(stack)
        entries = stack.list_raw(limit=3)
        assert len(entries) == 3


class TestAnxietyMixinCompat:
    """AnxietyMixin._get_aging_raw_entries works with RawEntry objects."""

    def test_aging_raw_entries_with_raw_entry_objects(self, stack):
        rid_old = _save_raw(stack, blob="old entry")
        _save_raw(stack, blob="recent entry")

        # Backdate the old entry's captured_at in the DB
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        with stack._backend._connect() as conn:
            conn.execute(
                "UPDATE raw_entries SET captured_at = ?, timestamp = ? WHERE id = ?",
                (old_time.isoformat(), old_time.isoformat(), rid_old),
            )

        total, aging, oldest = stack._get_aging_raw_entries(age_hours=24)
        assert total == 2
        assert aging == 1
        assert oldest >= 47  # at least ~47 hours old

    def test_no_entries(self, stack):
        total, aging, oldest = stack._get_aging_raw_entries(age_hours=24)
        assert total == 0
        assert aging == 0
        assert oldest == 0

    def test_all_recent(self, stack):
        _save_raw(stack)
        total, aging, oldest = stack._get_aging_raw_entries(age_hours=24)
        assert total == 1
        assert aging == 0
