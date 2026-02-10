"""Tests for kernle CLI summary command — covering uncovered branches."""

import argparse
import json
from io import StringIO
from unittest.mock import patch

import pytest

from kernle.cli.commands.summary import cmd_summary
from kernle.core import Kernle
from kernle.storage import SQLiteStorage


@pytest.fixture
def cli_kernle(tmp_path):
    """Create a real Kernle instance with SQLite storage for CLI testing."""
    db_path = tmp_path / "summary_test.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="summary_test_agent", db_path=db_path)
    k = Kernle(
        stack_id="summary_test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )

    yield k, storage
    storage.close()


class TestSummaryWriteJson:
    """Test summary write with JSON output."""

    def test_write_json_output(self, cli_kernle):
        """Write summary with --json returns structured JSON."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            summary_action="write",
            scope="month",
            content="Monthly review of project progress.",
            period_start="2025-01-01",
            period_end="2025-01-31",
            theme=["progress", "review"],
            epoch_id=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        data = json.loads(out.getvalue())
        assert "summary_id" in data
        assert data["scope"] == "month"
        assert len(data["summary_id"]) == 36


class TestSummaryWriteError:
    """Test summary write error handling."""

    def test_write_invalid_scope_prints_error(self, cli_kernle, capsys):
        """Invalid scope raises ValueError which is caught and printed."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            summary_action="write",
            scope="invalid_scope",
            content="Some content",
            period_start="2025-01-01",
            period_end="2025-01-31",
            theme=None,
            epoch_id=None,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Error:" in captured.out


class TestSummaryListJson:
    """Test summary list with JSON output."""

    def test_list_json_output(self, cli_kernle):
        """List summaries with --json returns structured array."""
        k, _storage = cli_kernle
        k.summary_save(
            content="First month done.",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
            key_themes=["testing"],
        )
        k.summary_save(
            content="Second month done.",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )

        args = argparse.Namespace(
            summary_action="list",
            scope=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        data = json.loads(out.getvalue())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_json_fields(self, cli_kernle):
        """JSON list includes all expected fields."""
        k, _storage = cli_kernle
        k.summary_save(
            content="Quarter review content for testing all the expected fields in JSON output.",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            key_themes=["theme1"],
        )

        args = argparse.Namespace(
            summary_action="list",
            scope=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        data = json.loads(out.getvalue())
        s = data[0]
        for field in ("id", "scope", "period_start", "period_end", "content", "key_themes"):
            assert field in s, f"Missing field: {field}"

    def test_list_json_truncates_long_content(self, cli_kernle):
        """JSON list truncates content over 200 chars."""
        k, _storage = cli_kernle
        long_content = "x" * 300
        k.summary_save(
            content=long_content,
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )

        args = argparse.Namespace(
            summary_action="list",
            scope=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        data = json.loads(out.getvalue())
        assert data[0]["content"].endswith("...")
        assert len(data[0]["content"]) == 203  # 200 chars + "..."


class TestSummaryListEmpty:
    """Test summary list when no summaries exist."""

    def test_list_empty_text(self, cli_kernle, capsys):
        """No summaries found message for empty list."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            summary_action="list",
            scope=None,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "No summaries found." in captured.out


class TestSummaryListError:
    """Test summary list error handling."""

    def test_list_invalid_scope_prints_error(self, cli_kernle, capsys):
        """Invalid scope filter raises ValueError which is caught."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            summary_action="list",
            scope="bad_scope",
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Error:" in captured.out


class TestSummaryListTextWithSupersedes:
    """Test summary list text with supersedes display."""

    def test_list_shows_supersedes_count(self, cli_kernle, capsys):
        """List text shows supersedes count when present."""
        k, _storage = cli_kernle
        sid1 = k.summary_save(
            content="Month 1 summary.",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        sid2 = k.summary_save(
            content="Month 2 summary.",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )
        k.summary_save(
            content="Quarter summary covering month 1 and 2.",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            supersedes=[sid1, sid2],
        )

        args = argparse.Namespace(
            summary_action="list",
            scope="quarter",
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Supersedes: 2 summaries" in captured.out


class TestSummaryShowNotFound:
    """Test summary show with non-existent ID."""

    def test_show_not_found(self, cli_kernle, capsys):
        """Show with unknown ID prints not found."""
        k, _storage = cli_kernle
        fake_id = "00000000-0000-0000-0000-000000000000"

        args = argparse.Namespace(
            summary_action="show",
            id=fake_id,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestSummaryShowJson:
    """Test summary show with JSON output."""

    def test_show_json_full_fields(self, cli_kernle):
        """Show summary with --json returns all fields."""
        k, _storage = cli_kernle
        sid = k.summary_save(
            content="Detailed quarter summary.",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            key_themes=["theme_a", "theme_b"],
        )

        args = argparse.Namespace(
            summary_action="show",
            id=sid,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_summary(args, k)

        data = json.loads(out.getvalue())
        assert data["id"] == sid
        assert data["scope"] == "quarter"
        assert data["content"] == "Detailed quarter summary."
        assert data["key_themes"] == ["theme_a", "theme_b"]
        assert data["is_protected"] is True
        for field in (
            "period_start",
            "period_end",
            "supersedes",
            "epoch_id",
            "created_at",
            "updated_at",
        ):
            assert field in data, f"Missing field: {field}"


class TestSummaryShowTextDetailed:
    """Test summary show text with all optional fields populated."""

    def test_show_text_with_epoch(self, cli_kernle, capsys):
        """Show text output displays epoch_id when set."""
        k, _storage = cli_kernle
        eid = k.epoch_create(name="Test Epoch", trigger_type="declared")
        sid = k.summary_save(
            content="Summary tied to an epoch.",
            scope="epoch",
            period_start="2025-01-01",
            period_end="2025-03-31",
            epoch_id=eid,
        )

        args = argparse.Namespace(
            summary_action="show",
            id=sid,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Epoch:" in captured.out
        assert eid[:8] in captured.out

    def test_show_text_with_themes(self, cli_kernle, capsys):
        """Show text output displays themes when set."""
        k, _storage = cli_kernle
        sid = k.summary_save(
            content="Summary with themes.",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
            key_themes=["infrastructure", "testing"],
        )

        args = argparse.Namespace(
            summary_action="show",
            id=sid,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Themes:" in captured.out
        assert "infrastructure" in captured.out

    def test_show_text_with_supersedes(self, cli_kernle, capsys):
        """Show text output lists superseded summary IDs."""
        k, _storage = cli_kernle
        sid1 = k.summary_save(
            content="Month 1.",
            scope="month",
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        sid2 = k.summary_save(
            content="Month 2.",
            scope="month",
            period_start="2025-02-01",
            period_end="2025-02-28",
        )
        sid3 = k.summary_save(
            content="Quarter rollup.",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            supersedes=[sid1, sid2],
        )

        args = argparse.Namespace(
            summary_action="show",
            id=sid3,
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Supersedes: 2 summaries" in captured.out
        assert sid1[:8] in captured.out
        assert sid2[:8] in captured.out


class TestSummaryUnknownAction:
    """Test summary with unknown action."""

    def test_unknown_action_prints_usage(self, cli_kernle, capsys):
        """Unknown action prints usage information."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            summary_action="unknown_action",
            json=False,
        )

        cmd_summary(args, k)
        captured = capsys.readouterr()
        assert "Usage: kernle summary" in captured.out
