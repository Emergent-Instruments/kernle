"""Tests for database-level raw entry prefix search (#708).

Verifies that find_by_id_prefix uses SQL LIKE for scalable prefix
matching instead of loading all entries into Python.
"""

import pytest

from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """SQLiteStorage with some raw entries for prefix testing."""
    s = SQLiteStorage(stack_id="test-prefix", db_path=tmp_path / "prefix.db")
    # Insert raw entries with predictable IDs
    with s._connect() as conn:
        for i, prefix in enumerate(["abc", "abc", "abc", "def", "def", "xyz"]):
            rid = f"{prefix}{i:010d}"
            conn.execute(
                "INSERT INTO raw_entries (id, stack_id, blob, processed, timestamp, deleted, "
                "captured_at, local_updated_at, version) "
                "VALUES (?, ?, ?, 0, datetime('now'), 0, datetime('now'), datetime('now'), 1)",
                (rid, "test-prefix", f"content-{i}"),
            )
        conn.commit()
    yield s
    s.close()


class TestFindByIdPrefix:
    """Database-level prefix search via LIKE query."""

    def test_prefix_matches_correct_entries(self, storage):
        """Prefix 'abc' matches only abc-prefixed entries."""
        matches = storage.find_raw_by_prefix("abc")
        assert len(matches) == 3
        assert all(m.id.startswith("abc") for m in matches)

    def test_prefix_respects_limit(self, storage):
        """Limit parameter is respected."""
        matches = storage.find_raw_by_prefix("abc", limit=2)
        assert len(matches) == 2

    def test_prefix_no_matches(self, storage):
        """Non-matching prefix returns empty list."""
        matches = storage.find_raw_by_prefix("zzz")
        assert len(matches) == 0

    def test_exact_id_as_prefix(self, storage):
        """Exact ID used as prefix returns that single entry."""
        matches = storage.find_raw_by_prefix("abc0000000000")
        assert len(matches) == 1
        assert matches[0].id == "abc0000000000"

    def test_empty_prefix_matches_all(self, storage):
        """Empty prefix matches all entries (up to limit)."""
        matches = storage.find_raw_by_prefix("", limit=100)
        assert len(matches) == 6

    def test_like_special_chars_escaped(self, storage):
        """LIKE special characters (%, _) in prefix are escaped."""
        # Insert entry with special chars in ID
        with storage._connect() as conn:
            conn.execute(
                "INSERT INTO raw_entries (id, stack_id, blob, processed, timestamp, deleted, "
                "captured_at, local_updated_at, version) "
                "VALUES (?, ?, ?, 0, datetime('now'), 0, datetime('now'), datetime('now'), 1)",
                ("test%under_score", "test-prefix", "special"),
            )
            conn.commit()

        # Searching for "test%" should not match everything
        matches = storage.find_raw_by_prefix("test%")
        assert len(matches) == 1
        assert matches[0].id == "test%under_score"

    def test_prefix_only_matches_non_deleted(self, storage):
        """Deleted entries are excluded from prefix search."""
        with storage._connect() as conn:
            conn.execute("UPDATE raw_entries SET deleted = 1 WHERE id = 'abc0000000000'")
            conn.commit()

        matches = storage.find_raw_by_prefix("abc")
        assert len(matches) == 2  # Was 3, now 2 after deleting one
