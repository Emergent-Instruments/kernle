"""Tests for CLI playbook command module."""

import json
from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.playbook import cmd_playbook


class TestPlaybookCreate:
    """Test playbook create command."""

    def test_create_basic(self, capsys):
        """Create playbook with minimal args."""
        k = MagicMock()
        k.playbook.return_value = "pb-12345678-abcd"

        args = Namespace(
            playbook_action="create",
            name="Test Playbook",
            description=None,
            steps="Step 1,Step 2,Step 3",
            step=None,
            triggers=None,
            trigger=None,
            failure_mode=None,
            recovery=None,
            tag=None,
        )

        cmd_playbook(args, k)

        k.playbook.assert_called_once()
        call_kwargs = k.playbook.call_args[1]
        assert call_kwargs["name"] == "Test Playbook"
        assert len(call_kwargs["steps"]) == 3

        captured = capsys.readouterr()
        assert "Playbook created:" in captured.out
        assert "Steps: 3" in captured.out

    def test_create_with_single_step(self, capsys):
        """Create playbook with single step (no comma)."""
        k = MagicMock()
        k.playbook.return_value = "pb-12345678-abcd"

        args = Namespace(
            playbook_action="create",
            name="Single Step",
            description="A simple playbook",
            steps="Just one step",
            step=None,
            triggers=None,
            trigger=None,
            failure_mode=None,
            recovery=None,
            tag=None,
        )

        cmd_playbook(args, k)

        call_kwargs = k.playbook.call_args[1]
        assert call_kwargs["steps"] == ["Just one step"]

    def test_create_with_step_flags(self, capsys):
        """Create playbook with --step flags."""
        k = MagicMock()
        k.playbook.return_value = "pb-12345678-abcd"

        args = Namespace(
            playbook_action="create",
            name="Multi Step",
            description=None,
            steps=None,
            step=["Step A", "Step B"],
            triggers=None,
            trigger=None,
            failure_mode=None,
            recovery=None,
            tag=None,
        )

        cmd_playbook(args, k)

        call_kwargs = k.playbook.call_args[1]
        assert call_kwargs["steps"] == ["Step A", "Step B"]

    def test_create_with_triggers(self, capsys):
        """Create playbook with triggers."""
        k = MagicMock()
        k.playbook.return_value = "pb-12345678-abcd"

        args = Namespace(
            playbook_action="create",
            name="Triggered Playbook",
            description=None,
            steps="Step 1",
            step=None,
            triggers="Trigger A, Trigger B",
            trigger=["Trigger C"],
            failure_mode=None,
            recovery=None,
            tag=None,
        )

        cmd_playbook(args, k)

        call_kwargs = k.playbook.call_args[1]
        assert len(call_kwargs["triggers"]) == 3

        captured = capsys.readouterr()
        assert "Triggers: 3" in captured.out

    def test_create_with_failure_modes(self, capsys):
        """Create playbook with failure modes."""
        k = MagicMock()
        k.playbook.return_value = "pb-12345678-abcd"

        args = Namespace(
            playbook_action="create",
            name="Robust Playbook",
            description=None,
            steps="Step 1",
            step=None,
            triggers=None,
            trigger=None,
            failure_mode=["Network timeout", "Authentication error"],
            recovery=["Retry", "Log out and back in"],
            tag=["important"],
        )

        cmd_playbook(args, k)

        call_kwargs = k.playbook.call_args[1]
        assert len(call_kwargs["failure_modes"]) == 2
        assert len(call_kwargs["recovery_steps"]) == 2
        assert call_kwargs["tags"] == ["important"]

        captured = capsys.readouterr()
        assert "Failure modes: 2" in captured.out


class TestPlaybookList:
    """Test playbook list command."""

    def test_list_empty(self, capsys):
        """List with no playbooks."""
        k = MagicMock()
        k.load_playbooks.return_value = []

        args = Namespace(
            playbook_action="list",
            limit=20,
            tag=None,
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "No playbooks found" in captured.out

    def test_list_with_playbooks(self, capsys):
        """List playbooks."""
        k = MagicMock()
        k.load_playbooks.return_value = [
            {
                "id": "pb-12345678",
                "name": "Test Playbook",
                "description": "A test playbook for testing",
                "mastery_level": "competent",
                "times_used": 5,
                "success_rate": 0.8,
                "tags": ["test", "example"],
            },
            {
                "id": "pb-87654321",
                "name": "Another Playbook",
                "description": "Another test",
                "mastery_level": "novice",
                "times_used": 0,
                "success_rate": 0.0,
                "tags": None,
            },
        ]

        args = Namespace(
            playbook_action="list",
            limit=20,
            tag=None,
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "Playbooks (2 total)" in captured.out
        assert "Test Playbook" in captured.out
        assert "competent" in captured.out
        assert "5x" in captured.out
        assert "test, example" in captured.out
        assert "n/a" in captured.out  # No uses for second playbook

    def test_list_json(self, capsys):
        """List playbooks as JSON."""
        k = MagicMock()
        k.load_playbooks.return_value = [
            {
                "id": "pb-12345678",
                "name": "Test",
                "description": "Test",
                "mastery_level": "novice",
                "times_used": 0,
                "success_rate": 0.0,
            }
        ]

        args = Namespace(
            playbook_action="list",
            limit=20,
            tag=None,
            json=True,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert len(output) == 1
        assert output[0]["name"] == "Test"


class TestPlaybookSearch:
    """Test playbook search command."""

    def test_search_not_found(self, capsys):
        """Search with no results."""
        k = MagicMock()
        k.search_playbooks.return_value = []

        args = Namespace(
            playbook_action="search",
            query="nonexistent",
            limit=10,
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "No playbooks found for 'nonexistent'" in captured.out

    def test_search_found(self, capsys):
        """Search with results."""
        k = MagicMock()
        k.search_playbooks.return_value = [
            {
                "id": "pb-12345678",
                "name": "Deployment Process",
                "description": "How to deploy to production safely",
                "mastery_level": "proficient",
                "times_used": 20,
                "success_rate": 0.95,
            }
        ]

        args = Namespace(
            playbook_action="search",
            query="deploy",
            limit=10,
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "Found 1 playbook(s)" in captured.out
        assert "Deployment Process" in captured.out
        assert "proficient" in captured.out

    def test_search_json(self, capsys):
        """Search with JSON output."""
        k = MagicMock()
        k.search_playbooks.return_value = [{"id": "pb-1", "name": "Test"}]

        args = Namespace(
            playbook_action="search",
            query="test",
            limit=10,
            json=True,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output[0]["name"] == "Test"


class TestPlaybookShow:
    """Test playbook show command."""

    def test_show_not_found(self, capsys):
        """Show playbook not found."""
        k = MagicMock()
        k.get_playbook.return_value = None

        args = Namespace(
            playbook_action="show",
            id="pb-nonexistent",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_show_playbook(self, capsys):
        """Show playbook details."""
        k = MagicMock()
        k.get_playbook.return_value = {
            "id": "pb-12345678",
            "name": "Full Playbook",
            "description": "A complete playbook with all fields",
            "mastery_level": "expert",
            "times_used": 50,
            "success_rate": 0.92,
            "confidence": 0.95,
            "triggers": ["When deploying", "Before releases"],
            "steps": ["Step 1", "Step 2"],
            "failure_modes": ["Timeout", "Auth error"],
            "recovery_steps": ["Retry", "Escalate"],
            "tags": ["deployment", "production"],
            "last_used": "2026-01-15T10:30:00Z",
            "created_at": "2025-06-01T00:00:00Z",
        }

        args = Namespace(
            playbook_action="show",
            id="pb-12345678",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "Full Playbook" in captured.out
        assert "Triggers (when to use)" in captured.out
        assert "When deploying" in captured.out
        assert "Steps" in captured.out
        assert "1. Step 1" in captured.out
        assert "Failure Modes" in captured.out
        assert "Timeout" in captured.out
        assert "Recovery Steps" in captured.out
        assert "expert" in captured.out
        assert "92%" in captured.out

    def test_show_playbook_with_dict_steps(self, capsys):
        """Show playbook with dict steps."""
        k = MagicMock()
        k.get_playbook.return_value = {
            "id": "pb-12345678",
            "name": "Complex Playbook",
            "description": "Steps with details",
            "mastery_level": "competent",
            "times_used": 10,
            "success_rate": 0.8,
            "confidence": 0.7,
            "triggers": None,
            "steps": [
                {"action": "First action", "details": "Extra info", "adaptations": "Varies"},
                {"action": "Second action"},
            ],
            "failure_modes": None,
            "recovery_steps": None,
            "tags": None,
            "last_used": None,
            "created_at": None,
        }

        args = Namespace(
            playbook_action="show",
            id="pb-12345678",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "First action" in captured.out
        assert "Extra info" in captured.out
        assert "Adaptations:" in captured.out
        assert "(none specified)" in captured.out

    def test_show_json(self, capsys):
        """Show playbook as JSON."""
        k = MagicMock()
        k.get_playbook.return_value = {
            "id": "pb-12345678",
            "name": "Test",
            "description": "Test desc",
            "mastery_level": "novice",
            "times_used": 0,
            "success_rate": 0.0,
            "confidence": 0.5,
            "steps": ["Step 1"],
        }

        args = Namespace(
            playbook_action="show",
            id="pb-12345678",
            json=True,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["name"] == "Test"


class TestPlaybookFind:
    """Test playbook find command."""

    def test_find_not_found(self, capsys):
        """Find with no relevant playbook."""
        k = MagicMock()
        k.find_playbook.return_value = None

        args = Namespace(
            playbook_action="find",
            situation="How do I fly to the moon?",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "No relevant playbook found" in captured.out

    def test_find_playbook(self, capsys):
        """Find a relevant playbook."""
        k = MagicMock()
        k.find_playbook.return_value = {
            "id": "pb-12345678",
            "name": "Deploy Process",
            "description": "How to safely deploy to production",
            "mastery_level": "proficient",
            "success_rate": 0.9,
            "steps": ["Build", "Test", "Deploy"],
        }

        args = Namespace(
            playbook_action="find",
            situation="I need to deploy my app",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "Recommended Playbook: Deploy Process" in captured.out
        assert "1. Build" in captured.out
        assert "proficient" in captured.out
        assert "90%" in captured.out
        assert "kernle playbook record" in captured.out

    def test_find_with_dict_steps(self, capsys):
        """Find playbook with dict steps."""
        k = MagicMock()
        k.find_playbook.return_value = {
            "id": "pb-12345678",
            "name": "Complex Process",
            "description": "Process with detailed steps",
            "mastery_level": "competent",
            "success_rate": 0.8,
            "steps": [{"action": "Step A"}, "Step B"],
        }

        args = Namespace(
            playbook_action="find",
            situation="complex task",
            json=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "Step A" in captured.out
        assert "Step B" in captured.out

    def test_find_json(self, capsys):
        """Find playbook with JSON output."""
        k = MagicMock()
        k.find_playbook.return_value = {"id": "pb-1", "name": "Test", "steps": []}

        args = Namespace(
            playbook_action="find",
            situation="test",
            json=True,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["name"] == "Test"


class TestPlaybookRecord:
    """Test playbook record command."""

    def test_record_success(self, capsys):
        """Record successful playbook usage."""
        k = MagicMock()
        k.record_playbook_use.return_value = True

        args = Namespace(
            playbook_action="record",
            id="pb-12345678",
            failure=False,
        )

        cmd_playbook(args, k)

        k.record_playbook_use.assert_called_with("pb-12345678", True)
        captured = capsys.readouterr()
        assert "success" in captured.out

    def test_record_failure(self, capsys):
        """Record failed playbook usage."""
        k = MagicMock()
        k.record_playbook_use.return_value = True

        args = Namespace(
            playbook_action="record",
            id="pb-12345678",
            failure=True,
        )

        cmd_playbook(args, k)

        k.record_playbook_use.assert_called_with("pb-12345678", False)
        captured = capsys.readouterr()
        assert "failure" in captured.out

    def test_record_not_found(self, capsys):
        """Record for non-existent playbook."""
        k = MagicMock()
        k.record_playbook_use.return_value = False

        args = Namespace(
            playbook_action="record",
            id="pb-nonexistent",
            failure=False,
        )

        cmd_playbook(args, k)

        captured = capsys.readouterr()
        assert "not found" in captured.out
