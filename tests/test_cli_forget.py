"""Tests for CLI forget command module."""

from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.forget import cmd_forget


class TestCmdForgetCandidates:
    """Test forget candidates command."""

    def test_no_candidates(self, capsys):
        """No candidates should show message."""
        k = MagicMock()
        k.get_forgetting_candidates.return_value = []

        args = Namespace(
            forget_action="candidates",
            threshold=0.3,
            limit=20,
            json=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "No forgetting candidates" in captured.out

    def test_with_candidates(self, capsys):
        """Candidates should be displayed."""
        k = MagicMock()
        k.get_forgetting_candidates.return_value = [
            {
                "type": "episode",
                "id": "abc123",
                "salience": 0.15,
                "summary": "Test episode",
                "confidence": 0.8,
                "times_accessed": 2,
                "last_accessed": "2026-01-20T00:00:00Z",
                "created_at": "2026-01-01",
            }
        ]

        args = Namespace(
            forget_action="candidates",
            threshold=0.3,
            limit=20,
            json=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "Forgetting Candidates" in captured.out
        assert "abc123" in captured.out
        assert "0.15" in captured.out

    def test_candidates_json(self, capsys):
        """Candidates JSON output."""
        k = MagicMock()
        k.get_forgetting_candidates.return_value = [
            {"type": "episode", "id": "abc123", "salience": 0.1}
        ]

        args = Namespace(
            forget_action="candidates",
            threshold=0.3,
            limit=20,
            json=True,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert '"type"' in captured.out
        assert '"episode"' in captured.out


class TestCmdForgetRun:
    """Test forget run command."""

    def test_dry_run(self, capsys):
        """Dry run should not actually forget."""
        k = MagicMock()
        k.run_forgetting_cycle.return_value = {
            "threshold": 0.3,
            "candidates": [{"type": "episode", "id": "abc123", "summary": "test"}],
            "candidate_count": 1,
            "forgotten": 0,
            "protected": 0,
            "dry_run": True,
            "timestamp": "2026-01-28T00:00:00Z",
        }

        args = Namespace(
            forget_action="run",
            threshold=0.3,
            limit=10,
            dry_run=True,
            json=False,
        )

        cmd_forget(args, k)

        k.run_forgetting_cycle.assert_called_with(threshold=0.3, limit=10, dry_run=True)
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out

    def test_actual_run(self, capsys):
        """Actual run should forget memories."""
        k = MagicMock()
        k.run_forgetting_cycle.return_value = {
            "threshold": 0.3,
            "candidates": [{"type": "episode", "id": "abc123", "summary": "test"}],
            "candidate_count": 1,
            "forgotten": 1,
            "protected": 0,
            "dry_run": False,
            "timestamp": "2026-01-28T00:00:00Z",
        }

        args = Namespace(
            forget_action="run",
            threshold=0.3,
            limit=10,
            dry_run=False,
            json=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "LIVE" in captured.out
        assert "Forgotten: 1" in captured.out

    def test_run_json(self, capsys):
        """Run JSON output."""
        k = MagicMock()
        k.run_forgetting_cycle.return_value = {
            "threshold": 0.3,
            "candidates": [],
            "candidate_count": 0,
            "forgotten": 0,
            "protected": 0,
            "dry_run": True,
        }

        args = Namespace(
            forget_action="run",
            threshold=0.3,
            limit=10,
            dry_run=True,
            json=True,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert '"threshold"' in captured.out


class TestCmdForgetProtect:
    """Test forget protect command."""

    def test_protect_success(self, capsys):
        """Successful protection."""
        k = MagicMock()
        k.protect.return_value = True

        args = Namespace(
            forget_action="protect",
            type="episode",
            id="abc123",
            unprotect=False,
        )

        cmd_forget(args, k)

        k.protect.assert_called_with("episode", "abc123", protected=True)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Protected" in captured.out

    def test_unprotect(self, capsys):
        """Unprotect should remove protection."""
        k = MagicMock()
        k.protect.return_value = True

        args = Namespace(
            forget_action="protect",
            type="episode",
            id="abc123",
            unprotect=True,
        )

        cmd_forget(args, k)

        k.protect.assert_called_with("episode", "abc123", protected=False)
        captured = capsys.readouterr()
        assert "Removed protection" in captured.out

    def test_protect_not_found(self, capsys):
        """Protection of non-existent memory."""
        k = MagicMock()
        k.protect.return_value = False

        args = Namespace(
            forget_action="protect",
            type="episode",
            id="nonexistent",
            unprotect=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "Memory not found" in captured.out


class TestCmdForgetRecover:
    """Test forget recover command."""

    def test_recover_success(self, capsys):
        """Successful recovery."""
        k = MagicMock()
        k.recover.return_value = True

        args = Namespace(
            forget_action="recover",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        k.recover.assert_called_with("episode", "abc123")
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Recovered" in captured.out

    def test_recover_not_found(self, capsys):
        """Recovery of non-existent memory."""
        k = MagicMock()
        k.recover.return_value = False

        args = Namespace(
            forget_action="recover",
            type="episode",
            id="nonexistent",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestCmdForgetList:
    """Test forget list command."""

    def test_list_empty(self, capsys):
        """Empty list of forgotten memories."""
        k = MagicMock()
        k.get_forgotten_memories.return_value = []

        args = Namespace(
            forget_action="list",
            limit=50,
            json=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "No forgotten memories" in captured.out

    def test_list_with_items(self, capsys):
        """List with forgotten memories."""
        k = MagicMock()
        k.get_forgotten_memories.return_value = [
            {
                "type": "episode",
                "id": "abc123",
                "summary": "Forgotten episode",
                "forgotten_at": "2026-01-28T00:00:00Z",
                "forgotten_reason": "Low salience",
                "created_at": "2026-01-01",
            }
        ]

        args = Namespace(
            forget_action="list",
            limit=50,
            json=False,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "Forgotten Memories" in captured.out
        assert "abc123" in captured.out

    def test_list_json(self, capsys):
        """List JSON output."""
        k = MagicMock()
        k.get_forgotten_memories.return_value = [
            {"type": "episode", "id": "abc123", "summary": "test"}
        ]

        args = Namespace(
            forget_action="list",
            limit=50,
            json=True,
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert '"type"' in captured.out
        assert '"episode"' in captured.out


class TestCmdForgetSalience:
    """Test forget salience command."""

    def test_salience_found(self, capsys):
        """Salience calculation for existing memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.45
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.8,
            times_accessed=5,
            is_protected=False,
            last_accessed=None,
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "Salience Analysis" in captured.out
        assert "0.45" in captured.out

    def test_salience_not_found(self, capsys):
        """Salience for non-existent memory."""
        k = MagicMock()
        k.calculate_salience.return_value = -1.0

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="nonexistent",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_salience_protected(self, capsys):
        """Salience for protected memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.1
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.5,
            times_accessed=0,
            is_protected=True,
            last_accessed=MagicMock(isoformat=lambda: "2026-01-28T00:00:00Z"),
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "PROTECTED" in captured.out
        assert "🛡️" in captured.out

    def test_salience_critical(self, capsys):
        """Salience for critical (very low) memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.05
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.3,
            times_accessed=0,
            is_protected=False,
            last_accessed=None,
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "CRITICAL" in captured.out
        assert "🔴" in captured.out

    def test_salience_low(self, capsys):
        """Salience for low memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.2
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.5,
            times_accessed=1,
            is_protected=False,
            last_accessed=None,
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "LOW" in captured.out
        assert "🟠" in captured.out

    def test_salience_moderate(self, capsys):
        """Salience for moderate memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.4
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.6,
            times_accessed=2,
            is_protected=False,
            last_accessed=None,
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "MODERATE" in captured.out
        assert "🟡" in captured.out

    def test_salience_high(self, capsys):
        """Salience for high memory."""
        k = MagicMock()
        k.calculate_salience.return_value = 0.8
        k._storage.get_memory.return_value = MagicMock(
            confidence=0.9,
            times_accessed=10,
            is_protected=False,
            last_accessed=MagicMock(isoformat=lambda: "2026-01-28T00:00:00Z"),
            created_at=MagicMock(isoformat=lambda: "2026-01-01T00:00:00Z"),
        )

        args = Namespace(
            forget_action="salience",
            type="episode",
            id="abc123",
        )

        cmd_forget(args, k)

        captured = capsys.readouterr()
        assert "HIGH" in captured.out
        assert "🟢" in captured.out
