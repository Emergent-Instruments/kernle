"""Tests for exception handling hardening in SQLiteStorage.

Covers:
- _save_embedding: exception chaining when both save and cleanup fail
- _save_embedding: graceful warning when cleanup succeeds after save failure
- _load_vec: error-level logging on extension load failure
"""

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from kernle.storage import SQLiteStorage


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db, monkeypatch):
    """Create a SQLiteStorage instance with vec disabled for controlled testing."""
    s = SQLiteStorage(stack_id="test-stack", db_path=temp_db)
    monkeypatch.setattr(s, "has_cloud_credentials", lambda: False)
    yield s
    s.close()


class TestSaveEmbeddingExceptionChaining:
    """Tests for _save_embedding exception propagation and cleanup behavior."""

    def test_raises_original_error_when_both_save_and_cleanup_fail(self, storage, caplog):
        """When the save INSERT fails and the cleanup DELETE also fails,
        the original save exception must be raised with the cleanup error
        chained via __cause__."""
        storage._has_vec = True
        storage._embed_text = MagicMock(return_value=[0.1, 0.2, 0.3])

        conn = MagicMock(spec=sqlite3.Connection)
        # First call: SELECT for existing hash check returns no existing row
        fetch_mock = MagicMock()
        fetch_mock.fetchone.return_value = None
        # Track execute calls to fail on the right ones
        original_error = sqlite3.OperationalError("disk I/O error on INSERT")
        cleanup_error = sqlite3.OperationalError("disk I/O error on DELETE")

        call_count = 0

        def execute_side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # SELECT content_hash - return no existing row
                return fetch_mock
            elif call_count == 2:
                # INSERT into vec_embeddings - fail
                raise original_error
            else:
                # DELETE cleanup attempts - also fail
                raise cleanup_error

        conn.execute.side_effect = execute_side_effect

        with caplog.at_level(logging.ERROR):
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                storage._save_embedding(conn, "episodes", "ep-1", "test content")

        # The raised exception should be the original save error
        assert "disk I/O error on INSERT" in str(exc_info.value)
        # The __cause__ should be the cleanup error (exception chaining)
        assert exc_info.value.__cause__ is cleanup_error
        # ERROR log should mention the cleanup failure
        assert any("Failed to clean stale embedding" in r.message for r in caplog.records)

    def test_raises_original_error_when_cleanup_fails_with_unexpected_error(self, storage, caplog):
        """When cleanup fails with a non-OperationalError, the original
        exception is still raised with proper chaining."""
        storage._has_vec = True
        storage._embed_text = MagicMock(return_value=[0.1, 0.2, 0.3])

        conn = MagicMock(spec=sqlite3.Connection)
        fetch_mock = MagicMock()
        fetch_mock.fetchone.return_value = None

        original_error = RuntimeError("unexpected save failure")
        cleanup_error = TypeError("unexpected cleanup failure")

        call_count = 0

        def execute_side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fetch_mock
            elif call_count == 2:
                raise original_error
            else:
                raise cleanup_error

        conn.execute.side_effect = execute_side_effect

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError) as exc_info:
                storage._save_embedding(conn, "episodes", "ep-1", "test content")

        assert exc_info.value is original_error
        assert exc_info.value.__cause__ is cleanup_error
        assert any("Unexpected error cleaning stale embedding" in r.message for r in caplog.records)

    def test_logs_warning_but_does_not_raise_when_cleanup_succeeds(self, storage, caplog):
        """When the save fails but cleanup DELETEs succeed, the method
        should log a warning and NOT raise -- the embedding is gracefully
        skipped."""
        storage._has_vec = True
        storage._embed_text = MagicMock(return_value=[0.1, 0.2, 0.3])

        conn = MagicMock(spec=sqlite3.Connection)
        fetch_mock = MagicMock()
        fetch_mock.fetchone.return_value = None

        save_error = sqlite3.OperationalError("disk full")

        call_count = 0

        def execute_side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return fetch_mock
            elif call_count == 2:
                raise save_error
            else:
                # Cleanup DELETEs succeed
                return MagicMock()

        conn.execute.side_effect = execute_side_effect

        with caplog.at_level(logging.WARNING):
            # Should NOT raise
            storage._save_embedding(conn, "episodes", "ep-1", "test content")

        # Should have logged warnings about the save failure and successful cleanup
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Failed to save embedding" in msg for msg in warning_messages)
        assert any("Cleaned stale embedding cache entry" in msg for msg in warning_messages)

    def test_skips_when_vec_disabled(self, storage):
        """_save_embedding returns immediately when _has_vec is False."""
        storage._has_vec = False
        conn = MagicMock(spec=sqlite3.Connection)

        # Should return without calling anything on conn
        storage._save_embedding(conn, "episodes", "ep-1", "test content")
        conn.execute.assert_not_called()

    def test_skips_when_embedding_returns_none(self, storage):
        """_save_embedding returns when _embed_text returns None."""
        storage._has_vec = True
        storage._embed_text = MagicMock(return_value=None)
        conn = MagicMock(spec=sqlite3.Connection)
        fetch_mock = MagicMock()
        fetch_mock.fetchone.return_value = None
        conn.execute.return_value = fetch_mock

        storage._save_embedding(conn, "episodes", "ep-1", "test content")

        # Only the SELECT for content_hash should have been called
        assert conn.execute.call_count == 1


class TestLoadVecErrorLogging:
    """Tests for _load_vec logging at error level on failure."""

    def test_logs_error_on_import_failure(self, storage, caplog):
        """When sqlite_vec import fails, _load_vec logs at ERROR level."""
        conn = MagicMock(spec=sqlite3.Connection)

        with patch.dict("sys.modules", {"sqlite_vec": None}):
            # Force ImportError by removing the module
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("sqlite_vec"),
            ):
                with caplog.at_level(logging.ERROR):
                    storage._load_vec(conn)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert any("Could not load sqlite-vec" in r.message for r in error_records)

    def test_logs_error_on_load_extension_failure(self, storage, caplog):
        """When enable_load_extension fails, _load_vec logs at ERROR level."""
        conn = MagicMock()
        conn.enable_load_extension.side_effect = sqlite3.OperationalError("not authorized")

        mock_vec = MagicMock()
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            with caplog.at_level(logging.ERROR):
                storage._load_vec(conn)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert any("Could not load sqlite-vec" in r.message for r in error_records)

    def test_logs_error_on_vec_load_failure(self, storage, caplog):
        """When sqlite_vec.load() raises, _load_vec logs at ERROR level."""
        conn = MagicMock(spec=sqlite3.Connection)

        mock_vec = MagicMock()
        mock_vec.load.side_effect = OSError("extension file not found")
        with patch.dict("sys.modules", {"sqlite_vec": mock_vec}):
            with caplog.at_level(logging.ERROR):
                storage._load_vec(conn)

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1
        assert any("Could not load sqlite-vec" in r.message for r in error_records)


def _make_import_blocker(blocked_module):
    """Create an import side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return real_import(name, *args, **kwargs)

    return blocker
