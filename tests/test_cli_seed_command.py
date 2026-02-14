"""Tests for kernle.cli.commands.seed."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kernle.cli.commands.seed import cmd_seed


def _ingest_result(**overrides):
    data = {
        "files_scanned": 4,
        "chunks_created": 10,
        "chunks_skipped": 2,
        "raw_entries_created": 8,
        "errors": [],
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class TestCmdSeed:
    def test_repo_json_output_and_arg_parsing(self, capsys):
        args = SimpleNamespace(
            seed_action="repo",
            path="/tmp/repo",
            extensions="py, md",
            exclude=".git, node_modules",
            max_chunk_size=512,
            dry_run=True,
            json=True,
        )
        k = MagicMock()

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_repo.return_value = _ingest_result()
            cmd_seed(args, k)

        mock_ingestor.ingest_repo.assert_called_once_with(
            "/tmp/repo",
            extensions=["py", "md"],
            exclude=[".git", "node_modules"],
            max_chunk_size=512,
            dry_run=True,
        )
        output = json.loads(capsys.readouterr().out)
        assert output["files_scanned"] == 4
        assert output["dry_run"] is True

    def test_repo_text_output_includes_errors(self, capsys):
        args = SimpleNamespace(
            seed_action="repo",
            path="/tmp/repo",
            extensions=None,
            exclude=None,
            max_chunk_size=2000,
            dry_run=False,
            json=False,
        )
        k = MagicMock()
        errors = ["first error", "second error"]

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_repo.return_value = _ingest_result(errors=errors)
            cmd_seed(args, k)

        out = capsys.readouterr().out
        assert "Corpus ingestion complete" in out
        assert "Errors: 2" in out
        assert "first error" in out

    def test_repo_text_output_without_errors(self, capsys):
        args = SimpleNamespace(
            seed_action="repo",
            path="/tmp/repo",
            extensions=None,
            exclude=None,
            max_chunk_size=2000,
            dry_run=False,
            json=False,
        )
        k = MagicMock()

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_repo.return_value = _ingest_result(errors=[])
            cmd_seed(args, k)

        out = capsys.readouterr().out
        assert "Corpus ingestion complete:" in out
        assert "Errors:" not in out

    def test_docs_json_output(self, capsys):
        args = SimpleNamespace(
            seed_action="docs",
            path="/tmp/docs",
            extensions="md,txt",
            max_chunk_size=1000,
            dry_run=False,
            json=True,
        )
        k = MagicMock()

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_docs.return_value = _ingest_result(raw_entries_created=5)
            cmd_seed(args, k)

        mock_ingestor.ingest_docs.assert_called_once_with(
            "/tmp/docs",
            extensions=["md", "txt"],
            max_chunk_size=1000,
            dry_run=False,
        )
        output = json.loads(capsys.readouterr().out)
        assert output["raw_entries_created"] == 5

    def test_docs_text_output_with_errors(self, capsys):
        args = SimpleNamespace(
            seed_action="docs",
            path="/tmp/docs",
            extensions=None,
            max_chunk_size=1500,
            dry_run=True,
            json=False,
        )
        k = MagicMock()

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.ingest_docs.return_value = _ingest_result(errors=["docs read failed"])
            cmd_seed(args, k)

        mock_ingestor.ingest_docs.assert_called_once_with(
            "/tmp/docs",
            extensions=None,
            max_chunk_size=1500,
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "Docs ingestion complete (dry run):" in out
        assert "Errors: 1" in out
        assert "docs read failed" in out

    def test_status_text_and_json_output(self, capsys):
        k = MagicMock()
        status = {"total_corpus_entries": 12, "repo_entries": 7, "docs_entries": 5}

        with patch("kernle.corpus.CorpusIngestor") as mock_ingestor_cls:
            mock_ingestor = mock_ingestor_cls.return_value
            mock_ingestor.get_status.return_value = status

            cmd_seed(SimpleNamespace(seed_action="status", json=False), k)
            text_out = capsys.readouterr().out
            assert "Corpus Ingestion Status" in text_out
            assert "Total corpus entries: 12" in text_out

            cmd_seed(SimpleNamespace(seed_action="status", json=True), k)
            json_out = json.loads(capsys.readouterr().out)
            assert json_out["status"] == status
            assert json_out["status_snapshot"] == status
            assert "status_snapshot_sha256" in json_out

    def test_unknown_action_prints_usage(self, capsys):
        cmd_seed(SimpleNamespace(seed_action="unknown"), MagicMock())
        assert "Usage: kernle seed {repo|docs|status}" in capsys.readouterr().out
