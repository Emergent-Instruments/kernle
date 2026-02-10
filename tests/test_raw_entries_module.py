"""Tests for kernle.storage.raw_entries module.

Tests the extracted raw entry CRUD functions.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kernle.storage.raw_entries import (
    _from_json,
    _safe_get,
    _to_json,
    append_raw_to_file,
    delete_raw,
    escape_like_pattern,
    get_all_stack_settings,
    get_processing_config,
    get_raw,
    get_raw_dir,
    get_raw_files,
    get_stack_setting,
    import_raw_entry,
    list_raw,
    mark_processed,
    mark_raw_processed,
    row_to_raw_entry,
    save_raw,
    search_raw_fts,
    set_processing_config,
    set_stack_setting,
    should_sync_raw,
    sync_raw_from_files,
    update_raw_fts,
)
from kernle.storage.schema import SCHEMA


def _make_conn():
    """Create an in-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _make_legacy_conn():
    """Create conn with legacy schema (content/timestamp, no NOT NULL on blob).

    Used for import/sync tests that insert with legacy column names.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE raw_entries (
            id TEXT PRIMARY KEY,
            stack_id TEXT NOT NULL,
            blob TEXT,
            captured_at TEXT,
            content TEXT,
            timestamp TEXT,
            source TEXT,
            processed INTEGER DEFAULT 0,
            processed_into TEXT,
            tags TEXT,
            confidence REAL DEFAULT 1.0,
            source_type TEXT DEFAULT 'direct_experience',
            local_updated_at TEXT,
            cloud_synced_at TEXT,
            version INTEGER DEFAULT 1,
            deleted INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


class TestHelpers:
    """Tests for helper functions."""

    def test_to_json_none(self):
        assert _to_json(None) is None

    def test_to_json_dict(self):
        assert _to_json({"a": 1}) == '{"a": 1}'

    def test_to_json_list(self):
        assert _to_json([1, 2]) == "[1, 2]"

    def test_from_json_none(self):
        assert _from_json(None) is None

    def test_from_json_empty(self):
        assert _from_json("") is None

    def test_from_json_valid(self):
        assert _from_json('{"a": 1}') == {"a": 1}

    def test_from_json_invalid(self):
        assert _from_json("not json") is None

    def test_safe_get_present(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (a TEXT, b TEXT)")
        conn.execute("INSERT INTO t VALUES ('x', NULL)")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "a") == "x"
        conn.close()

    def test_safe_get_null_returns_default(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES (NULL)")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "a", "fallback") == "fallback"
        conn.close()

    def test_safe_get_missing_key_returns_default(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
        row = conn.execute("SELECT * FROM t").fetchone()
        assert _safe_get(row, "missing_col", "default") == "default"
        conn.close()

    def test_escape_like_pattern(self):
        assert escape_like_pattern("normal") == "normal"
        assert escape_like_pattern("100%") == "100\\%"
        assert escape_like_pattern("a_b") == "a\\_b"
        assert escape_like_pattern("a\\b") == "a\\\\b"


class TestSaveRaw:
    """Tests for save_raw()."""

    def test_save_raw_basic(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        queue_fn = MagicMock()
        embed_fn = MagicMock()

        raw_id = save_raw(
            conn=conn,
            stack_id="test-stack",
            blob="Hello world",
            source="cli",
            raw_dir=raw_dir,
            queue_sync_fn=queue_fn,
            save_embedding_fn=embed_fn,
            should_sync_raw_fn=lambda: False,
        )

        assert raw_id is not None
        row = conn.execute("SELECT * FROM raw_entries WHERE id = ?", (raw_id,)).fetchone()
        assert row is not None
        assert row["blob"] == "Hello world"
        assert row["source"] == "cli"
        assert row["processed"] == 0
        embed_fn.assert_called_once()
        conn.close()

    def test_save_raw_normalizes_manual_source(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        raw_id = save_raw(
            conn=conn,
            stack_id="s",
            blob="test",
            source="manual",
            raw_dir=raw_dir,
            queue_sync_fn=MagicMock(),
            save_embedding_fn=MagicMock(),
            should_sync_raw_fn=lambda: False,
        )

        row = conn.execute("SELECT source FROM raw_entries WHERE id = ?", (raw_id,)).fetchone()
        assert row["source"] == "cli"
        conn.close()

    def test_save_raw_normalizes_auto_source(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        raw_id = save_raw(
            conn=conn,
            stack_id="s",
            blob="test",
            source="auto-capture",
            raw_dir=raw_dir,
            queue_sync_fn=MagicMock(),
            save_embedding_fn=MagicMock(),
            should_sync_raw_fn=lambda: False,
        )

        row = conn.execute("SELECT source FROM raw_entries WHERE id = ?", (raw_id,)).fetchone()
        assert row["source"] == "sdk"
        conn.close()

    def test_save_raw_normalizes_unknown_source(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        raw_id = save_raw(
            conn=conn,
            stack_id="s",
            blob="test",
            source="some_weird_source",
            raw_dir=raw_dir,
            queue_sync_fn=MagicMock(),
            save_embedding_fn=MagicMock(),
            should_sync_raw_fn=lambda: False,
        )

        row = conn.execute("SELECT source FROM raw_entries WHERE id = ?", (raw_id,)).fetchone()
        assert row["source"] == "unknown"
        conn.close()

    def test_save_raw_rejects_huge_blob(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        huge_blob = "x" * (51 * 1024 * 1024)  # > 50MB
        with pytest.raises(ValueError, match="too large"):
            save_raw(
                conn=conn,
                stack_id="s",
                blob=huge_blob,
                source="cli",
                raw_dir=raw_dir,
                queue_sync_fn=MagicMock(),
                save_embedding_fn=MagicMock(),
                should_sync_raw_fn=lambda: False,
            )
        conn.close()

    def test_save_raw_warns_large_blob(self, tmp_path, caplog):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        import logging

        with caplog.at_level(logging.WARNING, logger="kernle.storage.raw_entries"):
            large_blob = "x" * (11 * 1024 * 1024)  # > 10MB
            save_raw(
                conn=conn,
                stack_id="s",
                blob=large_blob,
                source="cli",
                raw_dir=raw_dir,
                queue_sync_fn=MagicMock(),
                save_embedding_fn=MagicMock(),
                should_sync_raw_fn=lambda: False,
            )
        assert "Extremely large" in caplog.text
        conn.close()

    def test_save_raw_warns_medium_blob(self, tmp_path, caplog):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        import logging

        with caplog.at_level(logging.WARNING, logger="kernle.storage.raw_entries"):
            blob = "x" * (2 * 1024 * 1024)  # > 1MB
            save_raw(
                conn=conn,
                stack_id="s",
                blob=blob,
                source="cli",
                raw_dir=raw_dir,
                queue_sync_fn=MagicMock(),
                save_embedding_fn=MagicMock(),
                should_sync_raw_fn=lambda: False,
            )
        assert "Very large" in caplog.text
        conn.close()

    def test_save_raw_info_100kb_blob(self, tmp_path, caplog):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        import logging

        with caplog.at_level(logging.INFO, logger="kernle.storage.raw_entries"):
            blob = "x" * (200 * 1024)  # > 100KB
            save_raw(
                conn=conn,
                stack_id="s",
                blob=blob,
                source="cli",
                raw_dir=raw_dir,
                queue_sync_fn=MagicMock(),
                save_embedding_fn=MagicMock(),
                should_sync_raw_fn=lambda: False,
            )
        assert "Large raw entry" in caplog.text
        conn.close()

    def test_save_raw_queues_sync_when_enabled(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        queue_fn = MagicMock()

        save_raw(
            conn=conn,
            stack_id="s",
            blob="test",
            source="cli",
            raw_dir=raw_dir,
            queue_sync_fn=queue_fn,
            save_embedding_fn=MagicMock(),
            should_sync_raw_fn=lambda: True,
        )

        queue_fn.assert_called_once()
        call_args = queue_fn.call_args
        assert call_args[0][1] == "raw_entries"
        assert call_args[0][3] == "upsert"
        conn.close()


class TestUpdateRawFts:
    """Tests for update_raw_fts()."""

    def test_update_fts_no_table(self):
        """Gracefully handles missing FTS table."""
        conn = _make_conn()
        # Don't create FTS table - just raw_entries via SCHEMA
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "local_updated_at, version, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, "2026-01-01", 1, 0),
        )
        # Should not raise
        update_raw_fts(conn, "r1", "hello")
        conn.close()

    def test_update_fts_no_matching_row(self):
        conn = _make_conn()
        # No matching row - should not error
        update_raw_fts(conn, "nonexistent", "hello")
        conn.close()


class TestAppendRawToFile:
    """Tests for append_raw_to_file()."""

    def test_creates_daily_file(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        append_raw_to_file(raw_dir, "abc12345", "test content", "2026-01-28T10:30:00", "cli", None)

        files = list(raw_dir.glob("*.md"))
        assert len(files) == 1
        assert "2026-01-28" in files[0].name

    def test_appends_content_with_header(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        append_raw_to_file(raw_dir, "abc12345", "my content", "2026-01-28T10:30:00", "cli", None)

        content = (raw_dir / "2026-01-28.md").read_text()
        assert "abc12345" in content
        assert "my content" in content

    def test_appends_tags_when_present(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        append_raw_to_file(
            raw_dir, "abc12345", "content", "2026-01-28T10:30:00", "cli", ["tag1", "tag2"]
        )

        content = (raw_dir / "2026-01-28.md").read_text()
        assert "tag1" in content
        assert "tag2" in content

    def test_handles_write_error_gracefully(self, tmp_path):
        # Pass a non-directory path as raw_dir
        bad_dir = tmp_path / "not_a_dir"
        # Should not raise (logs warning instead)
        append_raw_to_file(bad_dir, "abc12345", "content", "2026-01-28T10:30:00", "cli", None)


class TestGetRawDir:
    """Tests for get_raw_dir()."""

    def test_returns_same_path(self, tmp_path):
        assert get_raw_dir(tmp_path) == tmp_path


class TestGetRawFiles:
    """Tests for get_raw_files()."""

    def test_returns_empty_for_nonexistent_dir(self, tmp_path):
        result = get_raw_files(tmp_path / "nonexistent")
        assert result == []

    def test_returns_sorted_md_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "2026-01-01.md").write_text("a")
        (raw_dir / "2026-01-02.md").write_text("b")
        (raw_dir / "2026-01-03.md").write_text("c")

        result = get_raw_files(raw_dir)
        assert len(result) == 3
        # Sorted descending
        assert "2026-01-03" in result[0].name


class TestSyncRawFromFiles:
    """Tests for sync_raw_from_files()."""

    def test_sync_from_empty_dir(self, tmp_path):
        conn = _make_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        result = sync_raw_from_files(conn, "s", raw_dir, MagicMock())
        assert result["imported"] == 0
        assert result["skipped"] == 0
        assert result["files_processed"] == 0
        conn.close()

    def test_sync_imports_entries(self, tmp_path):
        conn = _make_legacy_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Create a valid flat file
        content = (
            "# Raw Captures - 2026-01-28\n\n"
            "## 10:30:00 [abc12345] cli\n"
            "Hello world capture\n\n"
            "## 11:00:00 [def67890] mcp\n"
            "Second capture\n"
        )
        (raw_dir / "2026-01-28.md").write_text(content)

        embed_fn = MagicMock()
        result = sync_raw_from_files(conn, "s", raw_dir, embed_fn)

        assert result["imported"] == 2
        assert result["files_processed"] == 1
        assert embed_fn.call_count == 2
        conn.close()

    def test_sync_skips_existing_entries(self, tmp_path):
        conn = _make_legacy_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Insert an existing entry with matching prefix
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, content, timestamp, source, processed, "
            "local_updated_at, version, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("abc12345-xxxx-xxxx", "s", "old", "2026-01-28", "cli", 0, "2026-01-28", 1, 0),
        )
        conn.commit()

        content = (
            "# Raw Captures - 2026-01-28\n\n" "## 10:30:00 [abc12345] cli\n" "Hello world capture\n"
        )
        (raw_dir / "2026-01-28.md").write_text(content)

        result = sync_raw_from_files(conn, "s", raw_dir, MagicMock())
        assert result["skipped"] == 1
        assert result["imported"] == 0
        conn.close()

    def test_sync_with_tags(self, tmp_path):
        conn = _make_legacy_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        content = (
            "# Raw Captures - 2026-01-28\n\n"
            "## 10:30:00 [abc12345] cli\n"
            "Tagged capture\n"
            "Tags: tag1, tag2\n"
        )
        (raw_dir / "2026-01-28.md").write_text(content)

        result = sync_raw_from_files(conn, "s", raw_dir, MagicMock())
        assert result["imported"] == 1
        conn.close()

    def test_sync_skips_empty_entries(self, tmp_path):
        conn = _make_legacy_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        content = (
            "# Raw Captures - 2026-01-28\n\n"
            "## 10:30:00 [abc12345] cli\n"
            "\n"
            "## 11:00:00 [def67890] mcp\n"
            "Some content\n"
        )
        (raw_dir / "2026-01-28.md").write_text(content)

        result = sync_raw_from_files(conn, "s", raw_dir, MagicMock())
        # First entry has no content lines, so it's silently dropped by the parser
        # (content_lines is [] which is falsy, so it never reaches import_raw_entry)
        # Only the second entry with actual content is imported
        assert result["imported"] == 1
        conn.close()

    def test_sync_handles_file_errors(self, tmp_path):
        conn = _make_legacy_conn()
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Create an unreadable file (write a file then make it a directory - will cause error)
        bad_file = raw_dir / "2026-01-28.md"
        bad_file.mkdir()  # Making it a directory instead of a file

        result = sync_raw_from_files(conn, "s", raw_dir, MagicMock())
        assert len(result["errors"]) == 1
        conn.close()


class TestImportRawEntry:
    """Tests for import_raw_entry()."""

    def test_import_new_entry(self):
        conn = _make_legacy_conn()
        entry = {
            "id_prefix": "abc",
            "content_lines": ["Hello world"],
            "timestamp": "2026-01-28T10:30:00",
            "source": "cli",
            "tags": None,
        }
        result = {"imported": 0, "skipped": 0, "errors": []}
        existing_ids = set()

        import_raw_entry(conn, "s", entry, existing_ids, result, MagicMock())
        assert result["imported"] == 1
        assert "abc" in list(existing_ids)[0]  # New id has prefix
        conn.close()

    def test_import_skips_matching_prefix(self):
        conn = _make_legacy_conn()
        entry = {
            "id_prefix": "abc",
            "content_lines": ["Hello world"],
            "timestamp": "2026-01-28T10:30:00",
            "source": "cli",
        }
        result = {"imported": 0, "skipped": 0, "errors": []}
        existing_ids = {"abc12345-xxxx"}

        import_raw_entry(conn, "s", entry, existing_ids, result, MagicMock())
        assert result["skipped"] == 1
        assert result["imported"] == 0
        conn.close()

    def test_import_skips_empty_content(self):
        conn = _make_legacy_conn()
        entry = {
            "id_prefix": "abc",
            "content_lines": ["   "],  # whitespace only
            "timestamp": "2026-01-28T10:30:00",
            "source": "cli",
        }
        result = {"imported": 0, "skipped": 0, "errors": []}
        existing_ids = set()

        import_raw_entry(conn, "s", entry, existing_ids, result, MagicMock())
        assert result["skipped"] == 1
        conn.close()

    def test_import_handles_db_error(self):
        conn = _make_legacy_conn()
        conn.execute("DROP TABLE raw_entries")
        entry = {
            "id_prefix": "abc",
            "content_lines": ["Hello"],
            "timestamp": "2026-01-28T10:30:00",
            "source": "cli",
            "tags": None,
        }
        result = {"imported": 0, "skipped": 0, "errors": []}
        existing_ids = set()

        import_raw_entry(conn, "s", entry, existing_ids, result, MagicMock())
        assert len(result["errors"]) == 1
        conn.close()


class TestGetRaw:
    """Tests for get_raw()."""

    def test_get_existing_raw(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
        )
        conn.commit()

        result = get_raw(conn, "s", "r1")
        assert result is not None
        assert result.blob == "hello"
        conn.close()

    def test_get_nonexistent_raw(self):
        conn = _make_conn()
        result = get_raw(conn, "s", "nonexistent")
        assert result is None
        conn.close()

    def test_get_deleted_raw_returns_none(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 1),
        )
        conn.commit()
        result = get_raw(conn, "s", "r1")
        assert result is None
        conn.close()


class TestListRaw:
    """Tests for list_raw()."""

    def test_list_raw_basic(self):
        conn = _make_conn()
        for i in range(3):
            conn.execute(
                "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
                "processed_into, local_updated_at, version, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"r{i}", "s", f"blob{i}", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
            )
        conn.commit()

        result = list_raw(conn, "s")
        assert len(result) == 3
        conn.close()

    def test_list_raw_respects_limit(self):
        conn = _make_conn()
        for i in range(5):
            conn.execute(
                "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
                "processed_into, local_updated_at, version, deleted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"r{i}", "s", f"blob{i}", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
            )
        conn.commit()

        result = list_raw(conn, "s", limit=2)
        assert len(result) == 2
        conn.close()


class TestSearchRawFts:
    """Tests for search_raw_fts()."""

    def test_search_falls_back_to_like(self):
        """Without FTS5 table, falls back to LIKE search."""
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello world", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
        )
        conn.commit()

        result = search_raw_fts(conn, "s", "hello")
        assert len(result) == 1
        assert result[0].blob == "hello world"
        conn.close()

    def test_search_no_results(self):
        conn = _make_conn()
        result = search_raw_fts(conn, "s", "nonexistent")
        assert len(result) == 0
        conn.close()


class TestMarkRawProcessed:
    """Tests for mark_raw_processed()."""

    def test_mark_processed_success(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
        )
        conn.commit()
        queue_fn = MagicMock()

        result = mark_raw_processed(conn, "s", "r1", ["e1", "b1"], queue_fn)
        assert result is True
        queue_fn.assert_called_once()

        row = conn.execute(
            "SELECT processed, processed_into FROM raw_entries WHERE id = 'r1'"
        ).fetchone()
        assert row["processed"] == 1
        conn.close()

    def test_mark_processed_nonexistent(self):
        conn = _make_conn()
        result = mark_raw_processed(conn, "s", "nonexistent", ["e1"], MagicMock())
        assert result is False
        conn.close()


class TestMarkProcessed:
    """Tests for mark_processed() (unified episodes/notes/beliefs)."""

    def test_mark_episode_processed(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO episodes (id, stack_id, objective, outcome, created_at, "
            "local_updated_at, version, deleted, processed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e1", "s", "test", "done", "2026-01-01", "2026-01-01", 1, 0, 0),
        )
        conn.commit()
        queue_fn = MagicMock()

        result = mark_processed(conn, "s", "episodes", "e1", queue_fn)
        assert result is True
        queue_fn.assert_called_once()
        conn.close()

    def test_mark_processed_nonexistent_returns_false(self):
        conn = _make_conn()
        result = mark_processed(conn, "s", "episodes", "nonexistent", MagicMock())
        assert result is False
        conn.close()


class TestProcessingConfig:
    """Tests for get/set_processing_config()."""

    def test_get_empty_config(self):
        conn = _make_conn()
        result = get_processing_config(conn)
        assert result == []
        conn.close()

    def test_set_and_get_config(self):
        conn = _make_conn()
        set_processing_config(conn, "raw_to_episode", enabled=True, batch_size=5)

        result = get_processing_config(conn)
        assert len(result) == 1
        assert result[0]["layer_transition"] == "raw_to_episode"
        assert result[0]["enabled"] is True
        assert result[0]["batch_size"] == 5
        conn.close()

    def test_update_existing_config(self):
        conn = _make_conn()
        set_processing_config(conn, "raw_to_episode", enabled=True, batch_size=5)
        set_processing_config(conn, "raw_to_episode", batch_size=20)

        result = get_processing_config(conn)
        assert len(result) == 1
        assert result[0]["batch_size"] == 20
        conn.close()

    def test_set_config_with_all_fields(self):
        conn = _make_conn()
        set_processing_config(
            conn,
            "episode_to_belief",
            enabled=False,
            model_id="claude-3",
            quantity_threshold=10,
            valence_threshold=0.5,
            time_threshold_hours=24,
            batch_size=15,
            max_sessions_per_day=5,
        )

        result = get_processing_config(conn)
        assert len(result) == 1
        cfg = result[0]
        assert cfg["enabled"] is False
        assert cfg["model_id"] == "claude-3"
        assert cfg["quantity_threshold"] == 10
        assert cfg["valence_threshold"] == 0.5
        assert cfg["time_threshold_hours"] == 24
        assert cfg["batch_size"] == 15
        assert cfg["max_sessions_per_day"] == 5
        conn.close()

    def test_set_config_insert_disabled(self):
        conn = _make_conn()
        set_processing_config(conn, "raw_to_episode", enabled=False)
        result = get_processing_config(conn)
        assert result[0]["enabled"] is False
        conn.close()


class TestStackSettings:
    """Tests for get/set/get_all stack_settings."""

    def test_get_nonexistent_setting(self):
        conn = _make_conn()
        result = get_stack_setting(conn, "s", "nonexistent")
        assert result is None
        conn.close()

    def test_set_and_get_setting(self):
        conn = _make_conn()
        set_stack_setting(conn, "s", "theme", "dark")
        result = get_stack_setting(conn, "s", "theme")
        assert result == "dark"
        conn.close()

    def test_upsert_setting(self):
        conn = _make_conn()
        set_stack_setting(conn, "s", "theme", "dark")
        set_stack_setting(conn, "s", "theme", "light")
        result = get_stack_setting(conn, "s", "theme")
        assert result == "light"
        conn.close()

    def test_get_all_settings(self):
        conn = _make_conn()
        set_stack_setting(conn, "s", "a", "1")
        set_stack_setting(conn, "s", "b", "2")

        result = get_all_stack_settings(conn, "s")
        assert result == {"a": "1", "b": "2"}
        conn.close()

    def test_settings_scoped_to_stack(self):
        conn = _make_conn()
        set_stack_setting(conn, "s1", "key", "val1")
        set_stack_setting(conn, "s2", "key", "val2")

        assert get_stack_setting(conn, "s1", "key") == "val1"
        assert get_stack_setting(conn, "s2", "key") == "val2"
        conn.close()


class TestDeleteRaw:
    """Tests for delete_raw()."""

    def test_delete_existing(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 0),
        )
        conn.commit()
        queue_fn = MagicMock()

        result = delete_raw(conn, "s", "r1", queue_fn)
        assert result is True
        queue_fn.assert_called_once()

        row = conn.execute("SELECT deleted FROM raw_entries WHERE id = 'r1'").fetchone()
        assert row["deleted"] == 1
        conn.close()

    def test_delete_nonexistent(self):
        conn = _make_conn()
        result = delete_raw(conn, "s", "nonexistent", MagicMock())
        assert result is False
        conn.close()

    def test_delete_already_deleted(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-01", "cli", 0, None, "2026-01-01", 1, 1),
        )
        conn.commit()

        result = delete_raw(conn, "s", "r1", MagicMock())
        assert result is False
        conn.close()


class TestRowToRawEntry:
    """Tests for row_to_raw_entry()."""

    def test_converts_new_schema_row(self):
        conn = _make_conn()
        conn.execute(
            "INSERT INTO raw_entries (id, stack_id, blob, captured_at, source, processed, "
            "processed_into, local_updated_at, version, deleted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("r1", "s", "hello", "2026-01-28T10:00:00", "cli", 0, None, "2026-01-28", 1, 0),
        )
        row = conn.execute("SELECT * FROM raw_entries WHERE id = 'r1'").fetchone()

        entry = row_to_raw_entry(row)
        assert entry.id == "r1"
        assert entry.blob == "hello"
        assert entry.source == "cli"
        assert entry.processed is False
        conn.close()


class TestShouldSyncRaw:
    """Tests for should_sync_raw()."""

    def test_default_is_false(self, monkeypatch):
        monkeypatch.delenv("KERNLE_RAW_SYNC", raising=False)
        monkeypatch.setattr("kernle.utils.get_kernle_home", lambda: Path("/nonexistent"))
        assert should_sync_raw() is False

    def test_env_true(self, monkeypatch):
        monkeypatch.setenv("KERNLE_RAW_SYNC", "true")
        assert should_sync_raw() is True

    def test_env_false(self, monkeypatch):
        monkeypatch.setenv("KERNLE_RAW_SYNC", "false")
        assert should_sync_raw() is False

    def test_env_yes(self, monkeypatch):
        monkeypatch.setenv("KERNLE_RAW_SYNC", "yes")
        assert should_sync_raw() is True

    def test_config_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KERNLE_RAW_SYNC", raising=False)
        monkeypatch.setattr("kernle.utils.get_kernle_home", lambda: tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text('{"sync": {"raw": true}}')
        assert should_sync_raw() is True

    def test_config_file_disabled(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KERNLE_RAW_SYNC", raising=False)
        monkeypatch.setattr("kernle.utils.get_kernle_home", lambda: tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text('{"sync": {"raw": false}}')
        assert should_sync_raw() is False

    def test_config_file_invalid_json(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KERNLE_RAW_SYNC", raising=False)
        monkeypatch.setattr("kernle.utils.get_kernle_home", lambda: tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text("not json")
        assert should_sync_raw() is False
