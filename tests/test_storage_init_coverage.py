"""Tests for SQLiteStorage init, embedder fallback, and embed_text branches.

Targets coverage for:
- __init__ validation (empty stack_id, path traversal)
- _make_fallback_embedder logic
- _embed_text fallback cascade (preferred -> fallback -> None)
- _maybe_restore_preferred_embedder retry logic
- _normalize_source_type edge cases
"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle.storage.embeddings import EmbeddingProvider, HashEmbedder
from kernle.storage.sqlite import SQLiteStorage, _normalize_source_type

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp database path for storage tests."""
    return tmp_path / "test_init.db"


@pytest.fixture
def storage(tmp_db):
    """Create a basic SQLiteStorage for testing."""
    s = SQLiteStorage(stack_id="test-init", db_path=tmp_db)
    yield s
    s.close()


# =============================================================================
# __init__ Stack ID Validation
# =============================================================================


class TestStackIdValidation:
    """Tests for stack_id validation in SQLiteStorage.__init__."""

    def test_empty_stack_id_raises(self, tmp_db):
        """An empty string stack_id should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SQLiteStorage(stack_id="", db_path=tmp_db)

    def test_whitespace_only_stack_id_raises(self, tmp_db):
        """A whitespace-only stack_id should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SQLiteStorage(stack_id="   ", db_path=tmp_db)

    def test_forward_slash_in_stack_id_raises(self, tmp_db):
        """A stack_id with forward slash should raise ValueError."""
        with pytest.raises(ValueError, match="path separators"):
            SQLiteStorage(stack_id="foo/bar", db_path=tmp_db)

    def test_backslash_in_stack_id_raises(self, tmp_db):
        """A stack_id with backslash should raise ValueError."""
        with pytest.raises(ValueError, match="path separators"):
            SQLiteStorage(stack_id="foo\\bar", db_path=tmp_db)

    def test_dot_dot_traversal_raises(self, tmp_db):
        """A stack_id of '..' should raise ValueError."""
        with pytest.raises(ValueError, match="relative path"):
            SQLiteStorage(stack_id="..", db_path=tmp_db)

    def test_single_dot_raises(self, tmp_db):
        """A stack_id of '.' should raise ValueError."""
        with pytest.raises(ValueError, match="relative path"):
            SQLiteStorage(stack_id=".", db_path=tmp_db)

    def test_consecutive_dots_allowed(self, tmp_db):
        """A stack_id with consecutive dots is allowed because split('.')
        never yields '..' as a segment — it yields empty strings instead.

        This verifies that the path traversal guard on line 235 of sqlite.py
        does not false-positive on double dots within a stack_id.
        """
        s = SQLiteStorage(stack_id="foo..bar", db_path=tmp_db)
        assert s.stack_id == "foo..bar"
        s.close()

    def test_valid_stack_id_with_dots_works(self, tmp_db):
        """A normal stack_id with dots (not traversal) should work fine."""
        # "foo.bar" does not have ".." as a segment
        s = SQLiteStorage(stack_id="foo.bar", db_path=tmp_db)
        assert s.stack_id == "foo.bar"
        s.close()


# =============================================================================
# _make_fallback_embedder
# =============================================================================


class TestMakeFallbackEmbedder:
    """Tests for the static _make_fallback_embedder method."""

    def test_hash_embedder_returns_none(self):
        """When the preferred embedder IS a HashEmbedder, fallback should be None."""
        # If we're already using the simplest embedder, there's no simpler fallback.
        hash_emb = HashEmbedder()
        result = SQLiteStorage._make_fallback_embedder(hash_emb)
        assert result is None

    def test_non_hash_embedder_returns_hash_fallback(self):
        """When the preferred embedder is NOT HashEmbedder, fallback should be a HashEmbedder."""
        # Create a mock embedding provider that is NOT a HashEmbedder
        mock_embedder = MagicMock(spec=EmbeddingProvider)
        mock_embedder.dimension = 768

        result = SQLiteStorage._make_fallback_embedder(mock_embedder)

        # Should return a HashEmbedder with matching dimension
        assert result is not None
        assert isinstance(result, HashEmbedder)
        assert result.dimension == 768


# =============================================================================
# _embed_text fallback cascade
# =============================================================================


class TestEmbedTextFallback:
    """Tests for the _embed_text method and its fallback behavior."""

    def test_successful_embedding_returns_vector(self, storage):
        """When the embedder succeeds, _embed_text returns the embedding."""
        result = storage._embed_text("hello world", context="test")

        # HashEmbedder always succeeds, so we should get a vector
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_hash_embedder_failure_returns_none(self, storage, caplog):
        """When HashEmbedder itself fails, _embed_text returns None and logs a warning."""
        # Force the hash embedder to raise an exception
        storage._embedder = storage._preferred_embedder
        with patch.object(storage._embedder, "embed", side_effect=RuntimeError("hash boom")):
            with caplog.at_level(logging.WARNING):
                result = storage._embed_text("test", context="hash-fail")

        assert result is None
        assert "Hash embedder failed" in caplog.text

    def test_preferred_fails_falls_back_to_hash(self, tmp_db, caplog):
        """When a non-hash preferred embedder fails, _embed_text falls back to HashEmbedder."""
        # Create a mock preferred embedder that will fail
        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("API unreachable")
        # Make isinstance check return False for HashEmbedder
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-fallback", db_path=tmp_db, embedder=mock_preferred)
        try:
            with caplog.at_level(logging.WARNING):
                result = s._embed_text("test text", context="fallback-test")

            # Should get a result from the fallback HashEmbedder
            assert result is not None
            assert isinstance(result, list)

            # Should have switched to fallback
            assert s._embedder is s._embedder_fallback

            # Should have set a retry time
            assert s._embedder_retry_at is not None

            assert "falling back to hash" in caplog.text
        finally:
            s.close()

    def test_both_preferred_and_fallback_fail_returns_none(self, tmp_db, caplog):
        """When both preferred and fallback embedders fail, _embed_text returns None."""
        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("API down")
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-double-fail", db_path=tmp_db, embedder=mock_preferred)
        try:
            # Make the fallback also fail
            s._embedder_fallback = MagicMock(spec=EmbeddingProvider)
            s._embedder_fallback.embed.side_effect = RuntimeError("fallback also broken")

            with caplog.at_level(logging.WARNING):
                result = s._embed_text("test text", context="double-fail")

            assert result is None
            assert "Fallback embedding failed" in caplog.text
        finally:
            s.close()

    def test_no_fallback_configured_returns_none(self, tmp_db, caplog):
        """When preferred fails and no fallback is configured, _embed_text returns None."""
        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("oops")
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-no-fallback", db_path=tmp_db, embedder=mock_preferred)
        try:
            # Remove the fallback
            s._embedder_fallback = None

            with caplog.at_level(logging.WARNING):
                result = s._embed_text("test text", context="no-fallback")

            assert result is None
            assert "no fallback is configured" in caplog.text
        finally:
            s.close()

    def test_fallback_logs_model_status_with_degraded(self, tmp_db, caplog):
        """When preferred embedder fails and fallback is used, ModelStatus is logged with degraded=True."""
        from kernle.protocols import ModelStatus

        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("API timeout")
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-status-degraded", db_path=tmp_db, embedder=mock_preferred)
        try:
            with caplog.at_level(logging.WARNING, logger="kernle.storage.sqlite"):
                s._embed_text("test text", context="status-test")

            # Verify ModelStatus was constructed and passed in log extra
            assert len(caplog.records) >= 1
            record = caplog.records[0]
            status = record.model_status
            assert isinstance(status, ModelStatus)
            assert status.provider == "MockEmbedder"
            assert status.available is False
            assert status.degraded is True
            assert status.error_message == "API timeout"
        finally:
            s.close()

    def test_no_fallback_logs_model_status_without_degraded(self, tmp_db, caplog):
        """When preferred fails with no fallback, ModelStatus.degraded is False."""
        from kernle.protocols import ModelStatus

        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("API error")
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(
            stack_id="test-status-no-fallback", db_path=tmp_db, embedder=mock_preferred
        )
        try:
            s._embedder_fallback = None

            with caplog.at_level(logging.WARNING, logger="kernle.storage.sqlite"):
                s._embed_text("test text", context="no-fallback-status")

            assert len(caplog.records) >= 1
            record = caplog.records[0]
            status = record.model_status
            assert isinstance(status, ModelStatus)
            assert status.degraded is False
            assert status.available is False
        finally:
            s.close()


# =============================================================================
# _maybe_restore_preferred_embedder
# =============================================================================


class TestMaybeRestorePreferredEmbedder:
    """Tests for the retry logic that restores the preferred embedder."""

    def test_no_fallback_is_noop(self, storage):
        """When _embedder_fallback is None, _maybe_restore is a no-op."""
        storage._embedder_fallback = None

        # Save original embedder reference
        original = storage._embedder
        storage._maybe_restore_preferred_embedder()
        assert storage._embedder is original

    def test_not_in_fallback_mode_is_noop(self, storage):
        """When not currently using the fallback, _maybe_restore is a no-op."""
        # Storage uses the preferred embedder by default
        storage._embedder_fallback = MagicMock()
        original = storage._embedder

        storage._maybe_restore_preferred_embedder()
        assert storage._embedder is original

    def test_retry_time_not_reached_stays_on_fallback(self, tmp_db):
        """When retry time hasn't been reached, stays on fallback."""
        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.side_effect = RuntimeError("still down")
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-no-restore", db_path=tmp_db, embedder=mock_preferred)
        try:
            # Force into fallback mode with future retry time
            s._embedder = s._embedder_fallback
            s._embedder_retry_at = datetime.now(timezone.utc) + timedelta(minutes=5)

            s._maybe_restore_preferred_embedder()

            # Should still be on fallback since retry time hasn't passed
            assert s._embedder is s._embedder_fallback
        finally:
            s.close()

    def test_retry_time_reached_restores_preferred(self, tmp_db, caplog):
        """When retry time has passed, restores the preferred embedder."""
        mock_preferred = MagicMock(spec=EmbeddingProvider)
        mock_preferred.dimension = 384
        mock_preferred.embed.return_value = [0.1] * 384
        type(mock_preferred).__name__ = "MockEmbedder"

        s = SQLiteStorage(stack_id="test-restore", db_path=tmp_db, embedder=mock_preferred)
        try:
            # Force into fallback mode with past retry time
            s._embedder = s._embedder_fallback
            s._embedder_retry_at = datetime.now(timezone.utc) - timedelta(minutes=1)

            with caplog.at_level(logging.INFO, logger="kernle.storage.sqlite"):
                s._maybe_restore_preferred_embedder()

            # Should have restored the preferred embedder
            assert s._embedder is s._preferred_embedder
            assert "restore preferred" in caplog.text.lower()
        finally:
            s.close()


# =============================================================================
# _normalize_source_type
# =============================================================================


class TestNormalizeSourceType:
    """Tests for the _normalize_source_type helper function."""

    def test_valid_string_normalizes(self):
        """A valid source_type string is normalized to lowercase."""
        # "direct_experience" is a valid source type
        result = _normalize_source_type("direct_experience")
        assert result == "direct_experience"

    def test_non_string_non_enum_raises(self):
        """A non-string, non-SourceType input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            _normalize_source_type(12345)

    def test_empty_string_raises(self):
        """An empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _normalize_source_type("  ")

    def test_invalid_source_type_raises(self):
        """An invalid source_type string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source_type"):
            _normalize_source_type("completely_bogus_type")

    def test_source_type_enum_value(self):
        """A SourceType enum value is converted to its string value."""
        from kernle.types import SourceType

        result = _normalize_source_type(SourceType.DIRECT_EXPERIENCE)
        assert result == "direct_experience"
