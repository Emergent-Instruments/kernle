"""Tests for kernle CLI summary commands (kernle/cli/commands/summary.py).

Tests the cmd_summary function covering write, list, show subcommands
and the fallback usage message, in both human-readable and JSON output modes.
"""

import json
from argparse import Namespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

from kernle.cli.commands.summary import cmd_summary
from kernle.types import Summary


def _make_summary(**overrides):
    """Create a realistic Summary dataclass for testing."""
    defaults = dict(
        id="sum-abc12345",
        stack_id="test-agent",
        scope="quarter",
        period_start="2025-01-01",
        period_end="2025-03-31",
        content="Q1 summary: foundation phase completed with strong progress",
        epoch_id=None,
        key_themes=None,
        supersedes=None,
        is_protected=True,
        created_at=datetime(2025, 4, 1, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2025, 4, 1, 12, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Summary(**defaults)


# === Write ===


class TestSummaryWrite:
    def test_write_text_output(self, capsys):
        k = MagicMock()
        k.summary_save.return_value = "sum-new12345"

        args = Namespace(
            summary_action="write",
            scope="quarter",
            content="Q1 went well",
            period_start="2025-01-01",
            period_end="2025-03-31",
            json=False,
        )
        # theme and epoch_id may or may not be present
        args.theme = ["growth", "learning"]
        args.epoch_id = None

        cmd_summary(args, k)

        k.summary_save.assert_called_once_with(
            content="Q1 went well",
            scope="quarter",
            period_start="2025-01-01",
            period_end="2025-03-31",
            key_themes=["growth", "learning"],
            epoch_id=None,
        )
        out = capsys.readouterr().out
        assert "Summary created (quarter)" in out
        assert "sum-new1" in out  # ID[:8]
        assert "2025-01-01" in out
        assert "2025-03-31" in out
        assert "growth" in out
        assert "learning" in out

    def test_write_json_output(self, capsys):
        k = MagicMock()
        k.summary_save.return_value = "sum-json1234"

        args = Namespace(
            summary_action="write",
            scope="month",
            content="January summary",
            period_start="2025-01-01",
            period_end="2025-01-31",
            json=True,
        )
        args.theme = None
        args.epoch_id = None

        cmd_summary(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["summary_id"] == "sum-json1234"
        assert data["scope"] == "month"

    def test_write_no_themes(self, capsys):
        k = MagicMock()
        k.summary_save.return_value = "sum-nothem12"

        args = Namespace(
            summary_action="write",
            scope="year",
            content="Annual summary content",
            period_start="2025-01-01",
            period_end="2025-12-31",
            json=False,
        )
        # No theme attribute at all
        args.epoch_id = None

        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "Summary created (year)" in out
        # "Themes" line should not appear
        assert "Themes" not in out

    def test_write_value_error(self, capsys):
        k = MagicMock()
        k.summary_save.side_effect = ValueError("scope must be one of ...")

        args = Namespace(
            summary_action="write",
            scope="invalid",
            content="Bad scope",
            period_start="2025-01-01",
            period_end="2025-01-31",
            json=False,
        )
        args.theme = None
        args.epoch_id = None

        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "Error:" in out

    def test_write_with_epoch_id(self, capsys):
        k = MagicMock()
        k.summary_save.return_value = "sum-epoch123"

        args = Namespace(
            summary_action="write",
            scope="epoch",
            content="Epoch summary",
            period_start="2025-01-01",
            period_end="2025-06-30",
            json=False,
        )
        args.theme = None
        args.epoch_id = "epoch-123"

        cmd_summary(args, k)

        k.summary_save.assert_called_once_with(
            content="Epoch summary",
            scope="epoch",
            period_start="2025-01-01",
            period_end="2025-06-30",
            key_themes=None,
            epoch_id="epoch-123",
        )


# === List ===


class TestSummaryList:
    def test_list_text_with_summaries(self, capsys):
        summaries = [
            _make_summary(
                id="sum-11111111",
                scope="quarter",
                period_start="2025-01-01",
                period_end="2025-03-31",
                content="Q1 quarterly summary with enough text to see in the output",
                key_themes=["growth", "learning"],
                supersedes=["sum-a", "sum-b", "sum-c"],
            ),
            _make_summary(
                id="sum-22222222",
                scope="month",
                period_start="2025-01-01",
                period_end="2025-01-31",
                content="January monthly summary",
                key_themes=None,
                supersedes=None,
            ),
        ]
        k = MagicMock()
        k.summary_list.return_value = summaries

        args = Namespace(summary_action="list", scope=None, json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "Summaries" in out
        assert "quarter" in out
        assert "month" in out
        assert "2025-01-01" in out
        assert "growth" in out
        assert "Supersedes: 3 summaries" in out

    def test_list_text_no_summaries(self, capsys):
        k = MagicMock()
        k.summary_list.return_value = []

        args = Namespace(summary_action="list", scope=None, json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "No summaries found." in out

    def test_list_json(self, capsys):
        summary = _make_summary(
            content="A short summary",
            key_themes=["testing"],
            supersedes=["old-1"],
        )
        k = MagicMock()
        k.summary_list.return_value = [summary]

        args = Namespace(summary_action="list", scope="quarter", json=True)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 1
        assert data[0]["scope"] == "quarter"
        assert data[0]["key_themes"] == ["testing"]
        assert data[0]["supersedes"] == ["old-1"]
        assert data[0]["created_at"] is not None

    def test_list_passes_scope(self):
        k = MagicMock()
        k.summary_list.return_value = []

        args = Namespace(summary_action="list", scope="year", json=False)
        cmd_summary(args, k)

        k.summary_list.assert_called_once_with(scope="year")

    def test_list_value_error(self, capsys):
        k = MagicMock()
        k.summary_list.side_effect = ValueError("scope must be one of ...")

        args = Namespace(summary_action="list", scope="invalid", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "Error:" in out

    def test_list_long_content_truncated(self, capsys):
        """Content > 80 chars gets truncated in text output."""
        long_content = "A" * 100
        summary = _make_summary(content=long_content)
        k = MagicMock()
        k.summary_list.return_value = [summary]

        args = Namespace(summary_action="list", scope=None, json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "..." in out
        # Should only show first 80 chars + "..."
        assert "A" * 81 not in out

    def test_list_json_long_content_truncated(self, capsys):
        """Content > 200 chars gets truncated in JSON list output."""
        long_content = "B" * 250
        summary = _make_summary(content=long_content)
        k = MagicMock()
        k.summary_list.return_value = [summary]

        args = Namespace(summary_action="list", scope=None, json=True)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["content"].endswith("...")
        assert len(data[0]["content"]) == 203  # 200 + "..."

    def test_list_no_created_at(self, capsys):
        """Summary with created_at=None shows 'unknown'."""
        summary = _make_summary(created_at=None)
        k = MagicMock()
        k.summary_list.return_value = [summary]

        args = Namespace(summary_action="list", scope=None, json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "unknown" in out


# === Show ===


class TestSummaryShow:
    def test_show_text_full(self, capsys):
        summary = _make_summary(
            epoch_id="epoch-123",
            key_themes=["testing", "coverage"],
            supersedes=["sum-old1234", "sum-old5678"],
        )
        k = MagicMock()
        k.summary_get.return_value = summary

        args = Namespace(summary_action="show", id="sum-abc12345", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "quarter" in out
        assert "2025-01-01" in out
        assert "2025-03-31" in out
        assert "sum-abc12345" in out
        assert "yes" in out  # is_protected
        assert "epoch-12" in out  # epoch_id[:8]
        assert "testing" in out
        assert "coverage" in out
        assert "Supersedes: 2 summaries" in out
        assert "sum-old1" in out

    def test_show_text_minimal(self, capsys):
        """Show with no optional fields."""
        summary = _make_summary(
            epoch_id=None,
            key_themes=None,
            supersedes=None,
            is_protected=False,
        )
        k = MagicMock()
        k.summary_get.return_value = summary

        args = Namespace(summary_action="show", id="sum-abc12345", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "no" in out  # is_protected=False
        assert "Epoch" not in out
        assert "Themes" not in out
        assert "Supersedes" not in out

    def test_show_not_found(self, capsys):
        k = MagicMock()
        k.summary_get.return_value = None

        args = Namespace(summary_action="show", id="sum-missing1", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "not found" in out

    def test_show_json(self, capsys):
        summary = _make_summary(
            epoch_id="epoch-456",
            key_themes=["quality"],
            supersedes=["old-1"],
        )
        k = MagicMock()
        k.summary_get.return_value = summary

        args = Namespace(summary_action="show", id="sum-abc12345", json=True)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "sum-abc12345"
        assert data["scope"] == "quarter"
        assert data["content"].startswith("Q1 summary")
        assert data["epoch_id"] == "epoch-456"
        assert data["is_protected"] is True
        assert data["key_themes"] == ["quality"]
        assert data["supersedes"] == ["old-1"]
        assert data["created_at"] is not None
        assert data["updated_at"] is not None

    def test_show_content_displayed(self, capsys):
        """Full content is displayed in show (not truncated like list)."""
        content = "Full summary content that should appear in its entirety"
        summary = _make_summary(content=content)
        k = MagicMock()
        k.summary_get.return_value = summary

        args = Namespace(summary_action="show", id="sum-abc12345", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert content in out


# === Unknown action ===


class TestSummaryUnknown:
    def test_unknown_action_shows_usage(self, capsys):
        k = MagicMock()

        args = Namespace(summary_action="unknown", json=False)
        cmd_summary(args, k)

        out = capsys.readouterr().out
        assert "Usage:" in out
        assert "write" in out
        assert "list" in out
        assert "show" in out
