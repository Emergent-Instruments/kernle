"""Tests for embedding fallback observability (#704, #712).

Verifies that:
- embedding_meta records include provider and fallback_used columns
- get_embedding_stats returns provider breakdown and degradation status
- EmbeddingComponent.on_maintenance reports degradation from storage
- Fallback embeddings are tracked correctly
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from kernle.stack.components.embedding import EmbeddingComponent
from kernle.storage.embeddings import EmbeddingProvider
from kernle.storage.sqlite import SQLiteStorage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage(tmp_path):
    """SQLiteStorage with default HashEmbedder."""
    s = SQLiteStorage(stack_id="test-embed-obs", db_path=tmp_path / "embed-obs.db")
    yield s
    s.close()


@pytest.fixture
def storage_with_mock_embedder(tmp_path):
    """SQLiteStorage with a mock non-hash embedder for fallback testing."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.dimension = 384
    mock_embedder.embed.return_value = [0.1] * 384
    type(mock_embedder).__name__ = "MockEmbedder"

    s = SQLiteStorage(
        stack_id="test-embed-fallback",
        db_path=tmp_path / "embed-fallback.db",
        embedder=mock_embedder,
    )
    yield s
    s.close()


# =============================================================================
# embedding_meta schema columns
# =============================================================================


class TestEmbeddingMetaSchema:
    """Verify embedding_meta table has provider and fallback columns."""

    def test_embedding_meta_has_provider_column(self, storage):
        """embedding_meta table includes embedding_provider column."""
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        with storage._connect() as conn:
            cols = conn.execute("PRAGMA table_info(embedding_meta)").fetchall()
            col_names = {c[1] for c in cols}

        assert "embedding_provider" in col_names

    def test_embedding_meta_has_fallback_column(self, storage):
        """embedding_meta table includes fallback_used column."""
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        with storage._connect() as conn:
            cols = conn.execute("PRAGMA table_info(embedding_meta)").fetchall()
            col_names = {c[1] for c in cols}

        assert "fallback_used" in col_names


# =============================================================================
# Provider tracking in _save_embedding
# =============================================================================


class TestProviderTracking:
    """Verify _save_embedding records provider info in embedding_meta."""

    def test_save_embedding_records_provider_name(self, storage):
        """_save_embedding writes the provider class name to embedding_meta."""
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        from kernle.storage import Note

        note = Note(id="n-prov-1", stack_id="test-embed-obs", content="Test provider tracking")
        storage.save_note(note)

        with storage._connect() as conn:
            meta = conn.execute(
                "SELECT embedding_provider, fallback_used FROM embedding_meta WHERE record_id = ?",
                ("n-prov-1",),
            ).fetchone()

        assert meta is not None
        assert meta["embedding_provider"] == "HashEmbedder"
        assert meta["fallback_used"] == 0

    def test_fallback_embedding_records_fallback_flag(self, storage_with_mock_embedder):
        """When fallback is used, fallback_used=1 is recorded."""
        storage = storage_with_mock_embedder
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        # Force the mock embedder to fail, triggering fallback
        storage._preferred_embedder.embed.side_effect = RuntimeError("API down")

        from kernle.storage import Note

        note = Note(id="n-fb-1", stack_id="test-embed-fallback", content="Test fallback tracking")
        storage.save_note(note)

        with storage._connect() as conn:
            meta = conn.execute(
                "SELECT embedding_provider, fallback_used FROM embedding_meta WHERE record_id = ?",
                ("n-fb-1",),
            ).fetchone()

        assert meta is not None
        assert meta["embedding_provider"] == "HashEmbedder"
        assert meta["fallback_used"] == 1


# =============================================================================
# get_embedding_stats
# =============================================================================


class TestGetEmbeddingStats:
    """Verify get_embedding_stats returns correct observability data."""

    def test_stats_returns_provider_breakdown(self, storage):
        """get_embedding_stats includes per-provider counts."""
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        from kernle.storage import Note

        storage.save_note(
            Note(id="n-s1", stack_id="test-embed-obs", content="First note for stats")
        )
        storage.save_note(
            Note(id="n-s2", stack_id="test-embed-obs", content="Second note for stats")
        )

        stats = storage.get_embedding_stats()

        assert stats["total"] >= 2
        assert "HashEmbedder" in stats["by_provider"]
        assert stats["by_provider"]["HashEmbedder"] >= 2

    def test_stats_reports_current_provider(self, storage):
        """get_embedding_stats includes current active provider."""
        stats = storage.get_embedding_stats()

        assert stats["current_provider"] == "HashEmbedder"
        assert stats["is_degraded"] is False

    def test_stats_reports_degraded_when_on_fallback(self, storage_with_mock_embedder):
        """get_embedding_stats reports is_degraded=True when on fallback."""
        storage = storage_with_mock_embedder

        # Force into fallback mode
        storage._embedder = storage._embedder_fallback
        storage._embedder_retry_at = datetime.now(timezone.utc) + timedelta(minutes=5)

        stats = storage.get_embedding_stats()

        assert stats["is_degraded"] is True
        assert stats["current_provider"] == "HashEmbedder"

    def test_stats_counts_fallback_embeddings(self, storage_with_mock_embedder):
        """get_embedding_stats counts fallback-generated embeddings."""
        storage = storage_with_mock_embedder
        if not storage._has_vec:
            pytest.skip("sqlite-vec not available")

        # Force fallback
        storage._preferred_embedder.embed.side_effect = RuntimeError("API down")

        from kernle.storage import Note

        storage.save_note(
            Note(id="n-fc1", stack_id="test-embed-fallback", content="Fallback count test")
        )

        stats = storage.get_embedding_stats()

        assert stats["fallback_count"] >= 1

    def test_stats_without_vec_returns_defaults(self, tmp_path):
        """get_embedding_stats returns sane defaults when vec is unavailable."""
        # Create storage without vec
        with patch.object(SQLiteStorage, "_check_sqlite_vec", return_value=False):
            s = SQLiteStorage(stack_id="no-vec", db_path=tmp_path / "no-vec.db")
            try:
                stats = s.get_embedding_stats()

                assert stats["total"] == 0
                assert stats["by_provider"] == {}
                assert stats["fallback_count"] == 0
                assert stats["is_degraded"] is False
            finally:
                s.close()


# =============================================================================
# EmbeddingComponent.on_maintenance degradation reporting
# =============================================================================


class TestEmbeddingComponentMaintenance:
    """Verify EmbeddingComponent.on_maintenance reports degradation."""

    def test_maintenance_includes_degradation_status(self, storage):
        """on_maintenance includes is_degraded when storage is set."""
        comp = EmbeddingComponent()
        comp.attach(stack_id="test-comp")
        comp.set_storage(storage)

        result = comp.on_maintenance()

        assert "is_degraded" in result
        assert result["is_degraded"] is False

    def test_maintenance_includes_current_provider(self, storage):
        """on_maintenance includes current_provider from storage."""
        comp = EmbeddingComponent()
        comp.attach(stack_id="test-comp")
        comp.set_storage(storage)

        result = comp.on_maintenance()

        assert "current_provider" in result
        assert result["current_provider"] == "HashEmbedder"

    def test_maintenance_includes_fallback_count(self, storage):
        """on_maintenance includes fallback_count from storage."""
        comp = EmbeddingComponent()
        comp.attach(stack_id="test-comp")
        comp.set_storage(storage)

        result = comp.on_maintenance()

        assert "fallback_count" in result

    def test_maintenance_without_storage_omits_stats(self):
        """on_maintenance without storage set omits storage-level stats."""
        comp = EmbeddingComponent()
        comp.attach(stack_id="test-no-storage")

        result = comp.on_maintenance()

        assert "provider" in result
        assert "dimension" in result
        assert "is_degraded" not in result

    def test_maintenance_handles_storage_error_gracefully(self):
        """on_maintenance handles storage errors without crashing."""
        comp = EmbeddingComponent()
        comp.attach(stack_id="test-error")

        mock_storage = MagicMock()
        mock_storage.get_embedding_stats.side_effect = RuntimeError("DB locked")
        comp.set_storage(mock_storage)

        result = comp.on_maintenance()

        # Should still return base stats without crashing
        assert "provider" in result
        assert "dimension" in result
        assert "is_degraded" not in result  # Not set because stats call failed

    def test_set_storage_stores_reference(self):
        """set_storage stores the storage reference for later use."""
        comp = EmbeddingComponent()
        mock_storage = MagicMock()
        comp.set_storage(mock_storage)

        assert comp._storage is mock_storage


# =============================================================================
# Migration test
# =============================================================================


class TestEmbeddingMetaMigration:
    """Verify schema migration adds new columns to existing databases."""

    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        """Opening an existing DB triggers migration that adds new columns."""
        db_path = tmp_path / "migrate.db"

        # Create initial DB
        s1 = SQLiteStorage(stack_id="migrate-test", db_path=db_path)
        s1.close()

        # Reopen — migration should handle existing table
        s2 = SQLiteStorage(stack_id="migrate-test", db_path=db_path)
        try:
            if not s2._has_vec:
                pytest.skip("sqlite-vec not available")

            with s2._connect() as conn:
                cols = conn.execute("PRAGMA table_info(embedding_meta)").fetchall()
                col_names = {c[1] for c in cols}

            assert "embedding_provider" in col_names
            assert "fallback_used" in col_names
        finally:
            s2.close()
