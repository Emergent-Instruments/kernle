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


class TestEmergencyMode:
    """Test emergency save mode."""

    def test_emergency_json_output(self, capsys):
        """Emergency save with JSON output."""
        k = MagicMock()
        k.emergency_save.return_value = {
            "checkpoint_saved": True,
            "episodes_consolidated": 2,
            "identity_synthesized": True,
            "sync_attempted": True,
            "sync_success": True,
            "errors": [],
            "success": True,
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=True,
            summary="Test summary",
            json=True,
            detailed=False,
            actions=False,
            auto=False,
        )

        cmd_anxiety(args, k)

        k.emergency_save.assert_called_with(summary="Test summary")
        captured = capsys.readouterr()
        assert '"checkpoint_saved": true' in captured.out

    def test_emergency_with_sync(self, capsys):
        """Emergency save shows sync status when attempted."""
        k = MagicMock()
        k.emergency_save.return_value = {
            "checkpoint_saved": True,
            "episodes_consolidated": 2,
            "identity_synthesized": True,
            "sync_attempted": True,
            "sync_success": True,
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

        captured = capsys.readouterr()
        assert "Sync attempted: ✓" in captured.out
        assert "Sync success: ✓" in captured.out

    def test_emergency_with_errors(self, capsys):
        """Emergency save shows errors when present."""
        k = MagicMock()
        k.emergency_save.return_value = {
            "checkpoint_saved": True,
            "episodes_consolidated": 0,
            "identity_synthesized": False,
            "sync_attempted": False,
            "sync_success": False,
            "errors": ["Connection failed", "Timeout"],
            "success": False,
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

        captured = capsys.readouterr()
        assert "Errors:" in captured.out
        assert "Connection failed" in captured.out
        assert "Partial save" in captured.out


class TestBriefMode:
    """Test brief mode output."""

    def test_brief_critical(self, capsys):
        """Brief mode shows critical when score >= 80."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 85,
            "overall_level": "Critical",
            "overall_emoji": "🔴",
            "dimensions": {},
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=False,
            brief=True,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "CRITICAL (85)" in captured.out
        assert "immediate action needed" in captured.out

    def test_brief_warn(self, capsys):
        """Brief mode shows warning when score 50-79."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 65,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {},
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=False,
            brief=True,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "WARN (65)" in captured.out
        assert "consider consolidation" in captured.out

    def test_brief_ok(self, capsys):
        """Brief mode shows OK when score < 50."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 25,
            "overall_level": "Calm",
            "overall_emoji": "🟢",
            "dimensions": {},
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=False,
            brief=True,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "OK (25)" in captured.out


class TestDetailedAndActions:
    """Test detailed and actions modes."""

    def test_detailed_mode(self, capsys):
        """Detailed mode shows dimension details."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 45,
            "overall_level": "Aware",
            "overall_emoji": "🟡",
            "dimensions": {
                "context_pressure": {"score": 30, "emoji": "🟢", "detail": "20k tokens"},
                "unsaved_work": {"score": 40, "emoji": "🟡", "detail": "15 min"},
                "consolidation_debt": {"score": 50, "emoji": "🟡", "detail": "5 episodes"},
                "raw_aging": {"score": 10, "emoji": "🟢", "detail": "1 day"},
                "identity_coherence": {"score": 60, "emoji": "🟠", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "3 beliefs"},
            },
            "recommendations": [
                {
                    "priority": "medium",
                    "description": "Run consolidation",
                    "command": "kernle consolidate",
                }
            ],
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=True,
            actions=False,
            auto=False,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "20k tokens" in captured.out
        assert "15 min" in captured.out
        assert "5 episodes" in captured.out
        assert "Recommended Actions:" in captured.out
        assert "kernle consolidate" in captured.out

    def test_actions_mode(self, capsys):
        """Actions mode shows recommended actions."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 60,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 60, "emoji": "🟠", "detail": "30 min"},
                "consolidation_debt": {"score": 70, "emoji": "🟠", "detail": "8 episodes"},
                "raw_aging": {"score": 20, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 60, "emoji": "🟠", "detail": "developing"},
                "memory_uncertainty": {"score": 50, "emoji": "🟡", "detail": "5 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "high", "description": "Checkpoint current state"},
            {"priority": "medium", "description": "Consolidate episodes"},
            {"priority": "low", "description": "Review uncertain memories"},
        ]

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=True,
            auto=False,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Recommended Actions:" in captured.out
        assert "Checkpoint current state" in captured.out
        assert "HIGH" in captured.out


class TestAutoMode:
    """Test auto-execution mode."""

    def test_auto_no_actions_needed(self, capsys):
        """Auto mode with no actions needed."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 20,
            "overall_level": "Calm",
            "overall_emoji": "🟢",
            "dimensions": {
                "context_pressure": {"score": 10, "emoji": "🟢", "detail": "low"},
                "unsaved_work": {"score": 15, "emoji": "🟢", "detail": "5 min"},
                "consolidation_debt": {"score": 20, "emoji": "🟢", "detail": "1 episode"},
                "raw_aging": {"score": 5, "emoji": "🟢", "detail": "fresh"},
                "identity_coherence": {"score": 30, "emoji": "🟢", "detail": "stable"},
                "memory_uncertainty": {"score": 20, "emoji": "🟢", "detail": "1 belief"},
            },
        }
        k.get_recommended_actions.return_value = []

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "No actions needed" in captured.out

    def test_auto_consolidate_action(self, capsys):
        """Auto mode executes consolidate action."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 60, "emoji": "🟠", "detail": "6 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "medium", "description": "Consolidate", "method": "consolidate"}
        ]
        k.consolidate.return_value = {"consolidated": 5}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        k.consolidate.assert_called_once()
        captured = capsys.readouterr()
        assert "Consolidated 5 episodes" in captured.out

    def test_auto_synthesize_identity_action(self, capsys):
        """Auto mode executes synthesize identity action."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 70, "emoji": "🟠", "detail": "weak"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {
                "priority": "medium",
                "description": "Synthesize identity",
                "method": "synthesize_identity",
            }
        ]
        k.synthesize_identity.return_value = {"confidence": 0.75}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        k.synthesize_identity.assert_called_once()
        captured = capsys.readouterr()
        assert "Identity synthesized" in captured.out
        assert "75%" in captured.out

    def test_auto_sync_action_success(self, capsys):
        """Auto mode executes sync action successfully."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "medium", "description": "Sync memories", "method": "sync"}
        ]
        k.sync.return_value = {"success": True, "pushed": 5, "pulled": 2}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        k.sync.assert_called_once()
        captured = capsys.readouterr()
        assert "Synced (pushed: 5, pulled: 2)" in captured.out

    def test_auto_sync_action_failure(self, capsys):
        """Auto mode handles sync action failure."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "medium", "description": "Sync memories", "method": "sync"}
        ]
        k.sync.return_value = {"success": False, "errors": ["Connection timeout"]}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Sync had issues" in captured.out

    def test_auto_emergency_save_action(self, capsys):
        """Auto mode executes emergency save action."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 90,
            "overall_level": "Critical",
            "overall_emoji": "🔴",
            "dimensions": {
                "context_pressure": {"score": 90, "emoji": "🔴", "detail": "high"},
                "unsaved_work": {"score": 85, "emoji": "🔴", "detail": "2 hours"},
                "consolidation_debt": {"score": 80, "emoji": "🔴", "detail": "20 episodes"},
                "raw_aging": {"score": 70, "emoji": "🟠", "detail": "stale"},
                "identity_coherence": {"score": 80, "emoji": "🔴", "detail": "fragmented"},
                "memory_uncertainty": {"score": 90, "emoji": "🔴", "detail": "many beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "emergency", "description": "Emergency save", "method": "emergency_save"}
        ]
        k.emergency_save.return_value = {"success": True, "errors": []}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        k.emergency_save.assert_called_once()
        captured = capsys.readouterr()
        assert "Emergency save completed" in captured.out

    def test_auto_emergency_save_with_errors(self, capsys):
        """Auto mode handles emergency save errors."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 90,
            "overall_level": "Critical",
            "overall_emoji": "🔴",
            "dimensions": {
                "context_pressure": {"score": 90, "emoji": "🔴", "detail": "high"},
                "unsaved_work": {"score": 85, "emoji": "🔴", "detail": "2 hours"},
                "consolidation_debt": {"score": 80, "emoji": "🔴", "detail": "20 episodes"},
                "raw_aging": {"score": 70, "emoji": "🟠", "detail": "stale"},
                "identity_coherence": {"score": 80, "emoji": "🔴", "detail": "fragmented"},
                "memory_uncertainty": {"score": 90, "emoji": "🔴", "detail": "many beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "emergency", "description": "Emergency save", "method": "emergency_save"}
        ]
        k.emergency_save.return_value = {"success": False, "errors": ["Disk full"]}

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Emergency save had errors" in captured.out

    def test_auto_get_uncertain_memories_action(self, capsys):
        """Auto mode executes get_uncertain_memories action."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 70, "emoji": "🟠", "detail": "many beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {
                "priority": "low",
                "description": "Review uncertain memories",
                "method": "get_uncertain_memories",
            }
        ]
        k.get_uncertain_memories.return_value = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        k.get_uncertain_memories.assert_called_once()
        captured = capsys.readouterr()
        assert "Found 3 uncertain memories" in captured.out

    def test_auto_unknown_method(self, capsys):
        """Auto mode skips unknown methods."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "low", "description": "Unknown action", "method": "unknown_method"}
        ]

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Skipping: Unknown method" in captured.out

    def test_auto_manual_action_skipped(self, capsys):
        """Auto mode skips manual actions."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "medium", "description": "Manual review needed"}
        ]

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Skipping: Manual review needed (manual action)" in captured.out

    def test_auto_action_failure(self, capsys):
        """Auto mode handles action execution failures."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 55,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 50, "emoji": "🟡", "detail": "medium"},
                "unsaved_work": {"score": 50, "emoji": "🟡", "detail": "25 min"},
                "consolidation_debt": {"score": 40, "emoji": "🟡", "detail": "3 episodes"},
                "raw_aging": {"score": 30, "emoji": "🟢", "detail": "recent"},
                "identity_coherence": {"score": 50, "emoji": "🟡", "detail": "developing"},
                "memory_uncertainty": {"score": 40, "emoji": "🟡", "detail": "4 beliefs"},
            },
        }
        k.get_recommended_actions.return_value = [
            {"priority": "high", "description": "Checkpoint", "method": "checkpoint"}
        ]
        k.checkpoint.side_effect = Exception("Database error")

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=True,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "Failed: Database error" in captured.out


class TestHighAnxietySuggestion:
    """Test suggestion for high anxiety."""

    def test_high_anxiety_suggests_auto(self, capsys):
        """High anxiety suggests running --auto."""
        k = MagicMock()
        k.get_anxiety_report.return_value = {
            "overall_score": 65,
            "overall_level": "Elevated",
            "overall_emoji": "🟠",
            "dimensions": {
                "context_pressure": {"score": 60, "emoji": "🟠", "detail": "medium"},
                "unsaved_work": {"score": 70, "emoji": "🟠", "detail": "45 min"},
                "consolidation_debt": {"score": 60, "emoji": "🟠", "detail": "6 episodes"},
                "raw_aging": {"score": 50, "emoji": "🟡", "detail": "aging"},
                "identity_coherence": {"score": 70, "emoji": "🟠", "detail": "developing"},
                "memory_uncertainty": {"score": 60, "emoji": "🟠", "detail": "5 beliefs"},
            },
        }

        args = Namespace(
            context=None,
            limit=200000,
            emergency=False,
            detailed=False,
            actions=False,
            auto=False,
            json=False,
            brief=False,
            source="cli",
            triggered_by="manual",
        )

        cmd_anxiety(args, k)

        captured = capsys.readouterr()
        assert "kernle anxiety --auto" in captured.out
