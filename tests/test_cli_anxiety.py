"""Tests for CLI anxiety command module."""

from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.anxiety import cmd_anxiety


class TestCmdAnxiety:
    """Test anxiety command."""

    def test_basic_report(self, capsys):
        """Basic anxiety report should display dimensions."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 35,
            "overall_level": "Aware",
            "overall_emoji": "🟡",
            "dimensions": {
                "context_pressure": {"score": 30, "emoji": "🟢", "detail": "low"},
                "unsaved_work": {"score": 40, "emoji": "🟡", "detail": "15 min"},
                "consolidation_debt": {"score": 20, "emoji": "🟢", "detail": "2 episodes"},
                "raw_aging": {"score": 10, "emoji": "🟢", "detail": "fresh"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 30, "emoji": "🟢", "detail": "2 beliefs"},
            },
            "timestamp": "2026-01-28T12:00:00Z",
            "agent_id": "test",
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=False,
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Memory Anxiety Report" in captured.out
        assert "Aware" in captured.out
        assert "35/100" in captured.out

    def test_emergency_save(self, capsys):
        """Emergency save should trigger emergency_save method."""
        k = MagicMock()
        k.emergency_save.return_value = {
            "checkpoint_saved": True,
            "episodes_consolidated": 3,
            "identity_synthesized": True,
            "sync_attempted": False,
            "sync_success": False,
            "errors": [],
            "success": True,
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=True,
            summary=None,
            json=False,
            detailed=False,
            actions=False,
            auto=False,
        )

        cmd_anxiety(args, k)

        k.emergency_save.assert_called_once()
        captured = capsys.readouterr()
        assert "EMERGENCY SAVE" in captured.out
        assert "Checkpoint saved: ✓" in captured.out

    def test_json_output(self, capsys):
        """JSON flag should output JSON."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 35,
            "overall_level": "Aware",
            "overall_emoji": "🟡",
            "dimensions": {},
            "timestamp": "2026-01-28T12:00:00Z",
            "agent_id": "test",
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=True,
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert '"overall_score": 35' in captured.out

    def test_auto_mode_calls_methods(self, capsys):
        """Auto mode should execute recommended actions."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 60, "emoji": "🟠", "detail": "45 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "5 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "fresh"},
                "identity_coherence": {"score": 70, "emoji": "🟠", "detail": "weak"},
                "memory_uncertainty": {"score": 50, "emoji": "🟡", "detail": "4 beliefs"},
            },
            "timestamp": "2026-01-28T12:00:00Z",
            "agent_id": "test",
        }
        k.get_recommended_actions.return_value = [
            {
                "priority": "high",
                "description": "Checkpoint",
                "method": "checkpoint",
                "command": "kernle checkpoint",
            },
        ]
        k.checkpoint.return_value = {"task": "test"}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
        )

        cmd_anxiety(args, k)

        k.checkpoint.assert_called_once()
        captured = capsys.readouterr()
        assert "Auto-execution complete" in captured.out
