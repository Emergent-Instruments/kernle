"""Tests for kernle.mcp.handlers.seed."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.mcp.handlers.seed import (
    _sanitize_string_list,
    handle_memory_seed_docs,
    handle_memory_seed_repo,
    handle_memory_seed_status,
    validate_memory_seed_docs,
    validate_memory_seed_repo,
    validate_memory_seed_status,
)


class TestSeedValidators:
    def test_sanitize_string_list(self):
        assert _sanitize_string_list(None) is None
        assert _sanitize_string_list("py, js") == ["py", "js"]
        assert _sanitize_string_list(["py", 12, ""]) == ["py", "12"]
        assert _sanitize_string_list(42) is None

    def test_validate_repo_defaults_for_invalid_types(self):
        result = validate_memory_seed_repo(
            {
                "path": "/tmp/repo",
                "extensions": ["py", "ts"],
                "exclude": "node_modules,.git",
                "max_chunk_size": "invalid",
                "dry_run": "nope",
            }
        )
        assert result["extensions"] == ["py", "ts"]
        assert result["exclude"] == ["node_modules", ".git"]
        assert result["max_chunk_size"] == 2000
        assert result["dry_run"] is False

    def test_validate_docs_defaults_for_invalid_types(self):
        result = validate_memory_seed_docs(
            {
                "path": "/tmp/docs",
                "extensions": "md,txt",
                "max_chunk_size": "invalid",
                "dry_run": "nope",
            }
        )
        assert result["extensions"] == ["md", "txt"]
        assert result["max_chunk_size"] == 2000
        assert result["dry_run"] is False

    def test_validate_status_returns_empty_dict(self):
        assert validate_memory_seed_status({"ignored": True}) == {}

    def test_validate_repo_requires_path(self):
        with pytest.raises(ValueError, match="path must be a string"):
            validate_memory_seed_repo({})


class TestSeedHandlers:
    def test_handle_repo_formats_output_with_errors(self):
        k = MagicMock()
        args = {
            "path": "/tmp/repo",
            "extensions": ["py"],
            "exclude": [".git"],
            "max_chunk_size": 500,
            "dry_run": True,
        }
        result = SimpleNamespace(
            files_scanned=5,
            chunks_created=9,
            chunks_skipped=1,
            raw_entries_created=0,
            errors=["failed to read foo.py"],
        )

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_repo.return_value = result
            output = handle_memory_seed_repo(args, k)

        mock_ingestor.ingest_repo.assert_called_once_with(
            "/tmp/repo",
            extensions=["py"],
            exclude=[".git"],
            max_chunk_size=500,
            dry_run=True,
        )
        assert "Corpus ingestion complete (dry run):" in output
        assert "Errors: 1" in output
        assert "failed to read foo.py" in output

    def test_handle_docs_formats_output(self):
        k = MagicMock()
        args = {
            "path": "/tmp/docs",
            "extensions": ["md"],
            "max_chunk_size": 1000,
            "dry_run": False,
        }
        result = SimpleNamespace(
            files_scanned=2,
            chunks_created=4,
            chunks_skipped=0,
            raw_entries_created=4,
            errors=[],
        )

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_docs.return_value = result
            output = handle_memory_seed_docs(args, k)

        mock_ingestor.ingest_docs.assert_called_once_with(
            "/tmp/docs",
            extensions=["md"],
            max_chunk_size=1000,
            dry_run=False,
        )
        assert "Docs ingestion complete:" in output
        assert "Files scanned: 2" in output

    def test_handle_status_formats_output(self):
        k = MagicMock()
        status = {"total_corpus_entries": 10, "repo_entries": 6, "docs_entries": 4}

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.get_status.return_value = status
            output = handle_memory_seed_status({}, k)

        assert "Corpus Ingestion Status" in output
        assert "Total corpus entries: 10" in output
        assert "Repo entries: 6" in output
