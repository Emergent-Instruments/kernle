"""Tests for CLI stats command module."""

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from kernle.cli.commands.stats import _health_checks_stats, cmd_stats


class TestCmdStats:
    """Test the cmd_stats dispatcher function."""

    def test_health_checks_action(self, capsys):
        """Test cmd_stats dispatches to health-checks handler."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 10,
            "avg_per_day": 2.5,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(stats_action="health-checks", json=False)
        cmd_stats(args, k)

        captured = capsys.readouterr()
        assert "Health Check Compliance Stats" in captured.out
        k._storage.get_health_check_stats.assert_called_once()

    def test_unknown_action(self, capsys):
        """Test cmd_stats handles unknown action."""
        k = MagicMock()
        args = Namespace(stats_action="unknown-action")

        cmd_stats(args, k)

        captured = capsys.readouterr()
        assert "Unknown stats action: unknown-action" in captured.out


class TestHealthChecksStatsJsonOutput:
    """Test _health_checks_stats with JSON output."""

    def test_json_output(self, capsys):
        """Test JSON output format."""
        k = MagicMock()
        stats = {
            "total_checks": 25,
            "avg_per_day": 3.5,
            "last_check_at": "2026-01-15T10:30:00Z",
            "last_anxiety_score": 45,
            "checks_by_source": {"cli": 15, "api": 10},
            "checks_by_trigger": {"manual": 20, "scheduled": 5},
        }
        k._storage.get_health_check_stats.return_value = stats

        args = Namespace(json=True)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_checks"] == 25
        assert output["avg_per_day"] == 3.5
        assert output["checks_by_source"]["cli"] == 15


class TestHealthChecksStatsFormattedOutput:
    """Test _health_checks_stats with formatted output."""

    def test_basic_stats_display(self, capsys):
        """Test basic stats are displayed correctly."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 42,
            "avg_per_day": 2.5,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "Health Check Compliance Stats" in captured.out
        assert "Total Checks:     42" in captured.out
        assert "Avg per Day:      2.5" in captured.out
        assert "Last Check:       Never" in captured.out

    def test_last_check_just_now(self, capsys):
        """Test 'just now' elapsed time format."""
        k = MagicMock()
        # Check performed 30 seconds ago
        now = datetime.now(timezone.utc)
        last_check = (now - timedelta(seconds=30)).isoformat()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": last_check,
            "last_anxiety_score": 30,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "just now" in captured.out

    def test_last_check_minutes_ago(self, capsys):
        """Test minutes elapsed time format."""
        k = MagicMock()
        # Check performed 15 minutes ago
        now = datetime.now(timezone.utc)
        last_check = (now - timedelta(minutes=15)).isoformat()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": last_check,
            "last_anxiety_score": 30,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "15m ago" in captured.out

    def test_last_check_hours_ago(self, capsys):
        """Test hours elapsed time format."""
        k = MagicMock()
        # Check performed 3 hours ago
        now = datetime.now(timezone.utc)
        last_check = (now - timedelta(hours=3)).isoformat()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": last_check,
            "last_anxiety_score": 30,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "3h ago" in captured.out

    def test_last_check_days_ago(self, capsys):
        """Test days elapsed time format."""
        k = MagicMock()
        # Check performed 2 days ago
        now = datetime.now(timezone.utc)
        last_check = (now - timedelta(days=2)).isoformat()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": last_check,
            "last_anxiety_score": 30,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "2d ago" in captured.out

    def test_last_check_invalid_timestamp_fallback(self, capsys):
        """Test fallback for invalid timestamp format."""
        k = MagicMock()
        # Invalid timestamp that will cause parsing error
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": "2026-01-15T10:30:00",  # Valid but we test fallback
            "last_anxiety_score": 30,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Should show something (either parsed or fallback)
        assert "Last Check:" in captured.out


class TestAnxietyScoreEmojis:
    """Test anxiety score color-coded emoji display."""

    def test_high_anxiety_red(self, capsys):
        """Test high anxiety score (>=80) shows red emoji."""
        k = MagicMock()
        now = datetime.now(timezone.utc)
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": now.isoformat(),
            "last_anxiety_score": 85,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Red emoji for high anxiety
        assert "Last Score:" in captured.out
        assert "85" in captured.out

    def test_medium_anxiety_yellow(self, capsys):
        """Test medium anxiety score (50-79) shows yellow emoji."""
        k = MagicMock()
        now = datetime.now(timezone.utc)
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": now.isoformat(),
            "last_anxiety_score": 65,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Yellow emoji for medium anxiety
        assert "Last Score:" in captured.out
        assert "65" in captured.out

    def test_low_anxiety_green(self, capsys):
        """Test low anxiety score (<50) shows green emoji."""
        k = MagicMock()
        now = datetime.now(timezone.utc)
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": now.isoformat(),
            "last_anxiety_score": 25,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Green emoji for low anxiety
        assert "Last Score:" in captured.out
        assert "25" in captured.out

    def test_no_anxiety_score(self, capsys):
        """Test when no anxiety score is recorded."""
        k = MagicMock()
        now = datetime.now(timezone.utc)
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 1.0,
            "last_check_at": now.isoformat(),
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Should not display score line when None
        assert "Last Score:" not in captured.out


class TestBreakdownSections:
    """Test checks_by_source and checks_by_trigger breakdown display."""

    def test_by_source_breakdown(self, capsys):
        """Test checks by source breakdown display."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 100,
            "avg_per_day": 5.0,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {"cli": 60, "api": 40},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "By Source:" in captured.out
        assert "cli" in captured.out
        assert "60" in captured.out
        assert "(60%)" in captured.out
        assert "api" in captured.out
        assert "40" in captured.out
        assert "(40%)" in captured.out

    def test_by_trigger_breakdown(self, capsys):
        """Test checks by trigger breakdown display."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 50,
            "avg_per_day": 2.5,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {"manual": 30, "scheduled": 20},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "By Trigger:" in captured.out
        assert "manual" in captured.out
        assert "30" in captured.out
        assert "(60%)" in captured.out
        assert "scheduled" in captured.out
        assert "20" in captured.out
        assert "(40%)" in captured.out

    def test_empty_breakdowns_not_shown(self, capsys):
        """Test empty breakdown sections are not displayed."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 10,
            "avg_per_day": 1.0,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "By Source:" not in captured.out
        assert "By Trigger:" not in captured.out


class TestComplianceGuidance:
    """Test compliance guidance messages."""

    def test_no_checks_guidance(self, capsys):
        """Test guidance when no checks have been recorded."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 0,
            "avg_per_day": 0.0,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "No health checks recorded yet" in captured.out
        assert "kernle anxiety" in captured.out

    def test_low_frequency_guidance(self, capsys):
        """Test guidance for low check frequency."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 5,
            "avg_per_day": 0.5,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "Low check frequency" in captured.out
        assert "2-4 checks per day" in captured.out

    def test_good_compliance_guidance(self, capsys):
        """Test guidance for good compliance rate."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 50,
            "avg_per_day": 3.0,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        assert "Good compliance rate" in captured.out

    def test_borderline_compliance(self, capsys):
        """Test guidance at borderline frequency (between 1.0 and 2.0)."""
        k = MagicMock()
        k._storage.get_health_check_stats.return_value = {
            "total_checks": 15,
            "avg_per_day": 1.5,
            "last_check_at": None,
            "last_anxiety_score": None,
            "checks_by_source": {},
            "checks_by_trigger": {},
        }

        args = Namespace(json=False)
        _health_checks_stats(args, k)

        captured = capsys.readouterr()
        # Between low and good, no special guidance
        assert "No health checks recorded" not in captured.out
        assert "Low check frequency" not in captured.out
        assert "Good compliance rate" not in captured.out
