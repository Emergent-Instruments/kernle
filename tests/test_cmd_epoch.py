"""Tests for kernle CLI epoch commands (kernle/cli/commands/epoch.py).

Tests the cmd_epoch function covering create, close, list, show, and current
subcommands in both human-readable and JSON output modes.
"""

import json
from argparse import Namespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

from kernle.cli.commands.epoch import cmd_epoch
from kernle.types import Epoch


def _make_epoch(**overrides):
    """Create a realistic Epoch dataclass for testing."""
    defaults = dict(
        id="epoch-abc12345",
        stack_id="test-agent",
        epoch_number=1,
        name="foundation",
        started_at=datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc),
        ended_at=None,
        trigger_type="declared",
        trigger_description=None,
        summary=None,
        key_belief_ids=None,
        key_relationship_ids=None,
        key_goal_ids=None,
        dominant_drive_ids=None,
    )
    defaults.update(overrides)
    return Epoch(**defaults)


# === Create ===


class TestEpochCreate:
    def test_create_text_output(self, capsys):
        k = MagicMock()
        k.epoch_create.return_value = "epoch-new12345"

        args = Namespace(
            epoch_action="create",
            name="new-era",
            trigger="declared",
            trigger_description=None,
            json=False,
        )
        cmd_epoch(args, k)

        k.epoch_create.assert_called_once_with(
            name="new-era", trigger_type="declared", trigger_description=None
        )
        out = capsys.readouterr().out
        assert "Epoch created: new-era" in out
        assert "epoch-ne" in out  # ID[:8]
        assert "Trigger: declared" in out

    def test_create_json_output(self, capsys):
        k = MagicMock()
        k.epoch_create.return_value = "epoch-json1234"

        args = Namespace(
            epoch_action="create",
            name="json-era",
            trigger="detected",
            trigger_description="Major shift",
            json=True,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["epoch_id"] == "epoch-json1234"
        assert data["name"] == "json-era"

    def test_create_with_trigger_description(self, capsys):
        k = MagicMock()
        k.epoch_create.return_value = "epoch-trig1234"

        args = Namespace(
            epoch_action="create",
            name="triggered",
            trigger="detected",
            trigger_description="Role change detected",
            json=False,
        )
        cmd_epoch(args, k)

        k.epoch_create.assert_called_once_with(
            name="triggered",
            trigger_type="detected",
            trigger_description="Role change detected",
        )

    def test_create_default_trigger(self, capsys):
        """When trigger is None or empty, defaults to 'declared'."""
        k = MagicMock()
        k.epoch_create.return_value = "epoch-def12345"

        args = Namespace(
            epoch_action="create",
            name="default-trigger",
            json=False,
        )
        # trigger attr missing — getattr with default handles it
        cmd_epoch(args, k)

        _, kwargs = k.epoch_create.call_args
        assert kwargs["trigger_type"] == "declared"

    def test_create_value_error(self, capsys):
        """ValueError from epoch_create is caught and printed."""
        k = MagicMock()
        k.epoch_create.side_effect = ValueError("Invalid trigger type")

        args = Namespace(
            epoch_action="create",
            name="valid-name",
            trigger="bad-trigger",
            trigger_description=None,
            json=False,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "Error:" in out
        assert "Invalid trigger type" in out


# === Close ===


class TestEpochClose:
    def test_close_text_success(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = _make_epoch(id="epoch-cur12345")
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {"scaffold": "Reflection step 1..."}

        args = Namespace(
            epoch_action="close",
            id=None,
            summary="Finished phase one",
            json=False,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "Epoch closed." in out
        assert "Finished" in out
        assert "Reflection step 1..." in out

    def test_close_with_explicit_id(self, capsys):
        k = MagicMock()
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {"scaffold": "Done."}

        args = Namespace(
            epoch_action="close",
            id="epoch-explicit1",
            summary=None,
            json=False,
        )
        cmd_epoch(args, k)

        k.epoch_close.assert_called_once_with(epoch_id="epoch-explicit1", summary=None)

    def test_close_no_open_epoch(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = None
        k.epoch_close.return_value = False

        args = Namespace(
            epoch_action="close",
            id=None,
            summary=None,
            json=False,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "No open epoch to close." in out

    def test_close_json_with_consolidation(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = _make_epoch(id="epoch-j1234567")
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {"scaffold": "JSON scaffold"}

        args = Namespace(
            epoch_action="close",
            id=None,
            summary="Summary text",
            json=True,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["closed"] is True
        assert data["consolidation"]["scaffold"] == "JSON scaffold"

    def test_close_json_no_epoch(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = None
        k.epoch_close.return_value = False

        args = Namespace(
            epoch_action="close",
            id=None,
            summary=None,
            json=True,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["closed"] is False
        assert data["consolidation"] is None

    def test_close_text_no_summary_no_consolidation_id(self, capsys):
        """Close succeeds but no closing_epoch_id resolved."""
        k = MagicMock()
        k.get_current_epoch.return_value = None
        k.epoch_close.return_value = True

        args = Namespace(
            epoch_action="close",
            id=None,
            summary=None,
            json=False,
        )
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "Epoch closed." in out
        # No consolidation should be triggered
        k.consolidate_epoch_closing.assert_not_called()


# === List ===


class TestEpochList:
    def test_list_text_with_epochs(self, capsys):
        epochs = [
            _make_epoch(
                id="epoch-22222222",
                epoch_number=2,
                name="growth",
                started_at=datetime(2025, 4, 1, tzinfo=timezone.utc),
                ended_at=None,
            ),
            _make_epoch(
                id="epoch-11111111",
                epoch_number=1,
                name="onboarding",
                started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                ended_at=datetime(2025, 3, 31, tzinfo=timezone.utc),
                summary="Completed initial setup and learned the system",
            ),
        ]
        k = MagicMock()
        k.get_epochs.return_value = epochs

        args = Namespace(epoch_action="list", limit=20, json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "Epochs" in out
        assert "growth" in out
        assert "ACTIVE" in out
        assert "onboarding" in out
        assert "closed" in out
        assert "Completed initial" in out

    def test_list_text_no_epochs(self, capsys):
        k = MagicMock()
        k.get_epochs.return_value = []

        args = Namespace(epoch_action="list", limit=20, json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "No epochs found." in out

    def test_list_json(self, capsys):
        epoch = _make_epoch(
            ended_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            summary="Test summary",
        )
        k = MagicMock()
        k.get_epochs.return_value = [epoch]

        args = Namespace(epoch_action="list", limit=5, json=True)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 1
        assert data[0]["name"] == "foundation"
        assert data[0]["summary"] == "Test summary"
        assert data[0]["started_at"] is not None
        assert data[0]["ended_at"] is not None

    def test_list_passes_limit(self):
        k = MagicMock()
        k.get_epochs.return_value = []

        args = Namespace(epoch_action="list", limit=10, json=False)
        cmd_epoch(args, k)

        k.get_epochs.assert_called_once_with(limit=10)

    def test_list_epoch_no_started_at(self, capsys):
        """Epoch with started_at=None displays 'unknown'."""
        epoch = _make_epoch(started_at=None)
        k = MagicMock()
        k.get_epochs.return_value = [epoch]

        args = Namespace(epoch_action="list", limit=20, json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "unknown" in out


# === Show ===


class TestEpochShow:
    def test_show_text_active_epoch(self, capsys):
        epoch = _make_epoch(
            trigger_description="User declared new era",
            key_belief_ids=["b1", "b2"],
            key_relationship_ids=["r1"],
            key_goal_ids=["g1", "g2", "g3"],
            dominant_drive_ids=["d1"],
        )
        k = MagicMock()
        k.get_epoch.return_value = epoch

        args = Namespace(epoch_action="show", id="epoch-abc12345", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "foundation" in out
        assert "ACTIVE" in out
        assert "epoch-abc12345" in out
        assert "User declared new era" in out
        assert "Key beliefs: 2" in out
        assert "Key relationships: 1" in out
        assert "Key goals: 3" in out
        assert "Dominant drives: 1" in out

    def test_show_text_closed_epoch_with_summary(self, capsys):
        epoch = _make_epoch(
            ended_at=datetime(2025, 6, 1, 14, 0, tzinfo=timezone.utc),
            summary="Completed the foundation phase",
        )
        k = MagicMock()
        k.get_epoch.return_value = epoch

        args = Namespace(epoch_action="show", id="epoch-abc12345", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "closed" in out
        assert "Completed the foundation phase" in out

    def test_show_not_found(self, capsys):
        k = MagicMock()
        k.get_epoch.return_value = None

        args = Namespace(epoch_action="show", id="epoch-missing1", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "not found" in out

    def test_show_json(self, capsys):
        epoch = _make_epoch(
            key_belief_ids=["b1"],
            key_goal_ids=["g1"],
            trigger_description="Auto-detected",
        )
        k = MagicMock()
        k.get_epoch.return_value = epoch

        args = Namespace(epoch_action="show", id="epoch-abc12345", json=True)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "epoch-abc12345"
        assert data["name"] == "foundation"
        assert data["key_belief_ids"] == ["b1"]
        assert data["key_goal_ids"] == ["g1"]
        assert data["trigger_description"] == "Auto-detected"

    def test_show_minimal_epoch(self, capsys):
        """Show epoch with no optional fields populated."""
        epoch = _make_epoch(
            started_at=None,
            trigger_description=None,
            summary=None,
            key_belief_ids=None,
            key_relationship_ids=None,
            key_goal_ids=None,
            dominant_drive_ids=None,
        )
        k = MagicMock()
        k.get_epoch.return_value = epoch

        args = Namespace(epoch_action="show", id="epoch-abc12345", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "foundation" in out
        assert "unknown" in out  # started_at is None
        # Should NOT print optional fields when None
        assert "Key beliefs" not in out
        assert "Key relationships" not in out
        assert "Key goals" not in out
        assert "Dominant drives" not in out
        assert "Trigger description" not in out
        assert "Summary" not in out


# === Current ===


class TestEpochCurrent:
    def test_current_text_with_epoch(self, capsys):
        epoch = _make_epoch(epoch_number=3, name="maturity")
        k = MagicMock()
        k.get_current_epoch.return_value = epoch

        args = Namespace(epoch_action="current", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "Current epoch: #3 - maturity" in out
        assert "epoch-ab" in out  # ID[:8]

    def test_current_text_no_epoch(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = None

        args = Namespace(epoch_action="current", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "No active epoch." in out

    def test_current_json_with_epoch(self, capsys):
        epoch = _make_epoch()
        k = MagicMock()
        k.get_current_epoch.return_value = epoch

        args = Namespace(epoch_action="current", json=True)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "epoch-abc12345"
        assert data["epoch_number"] == 1
        assert data["name"] == "foundation"

    def test_current_json_no_epoch(self, capsys):
        k = MagicMock()
        k.get_current_epoch.return_value = None

        args = Namespace(epoch_action="current", json=True)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert json.loads(out) is None

    def test_current_no_started_at(self, capsys):
        """Current epoch with started_at=None shows 'unknown'."""
        epoch = _make_epoch(started_at=None)
        k = MagicMock()
        k.get_current_epoch.return_value = epoch

        args = Namespace(epoch_action="current", json=False)
        cmd_epoch(args, k)

        out = capsys.readouterr().out
        assert "unknown" in out
