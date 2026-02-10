"""Tests for kernle CLI epoch command — covering uncovered branches."""

import argparse
import json
import uuid
from datetime import datetime, timezone
from io import StringIO
from unittest.mock import patch

import pytest

from kernle.cli.commands.epoch import cmd_epoch
from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from kernle.types import Epoch


@pytest.fixture
def cli_kernle(tmp_path):
    """Create a real Kernle instance with SQLite storage for CLI testing."""
    db_path = tmp_path / "epoch_test.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    storage = SQLiteStorage(stack_id="epoch_test_agent", db_path=db_path)
    k = Kernle(
        stack_id="epoch_test_agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )

    yield k, storage
    storage.close()


class TestEpochCreateJson:
    """Test epoch create with JSON output."""

    def test_create_json_output(self, cli_kernle):
        """Create epoch with --json returns structured JSON."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            epoch_action="create",
            name="JSON Era",
            trigger="declared",
            trigger_description=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data["name"] == "JSON Era"
        assert "epoch_id" in data
        assert len(data["epoch_id"]) == 36  # UUID format

    def test_create_json_trigger_type(self, cli_kernle):
        """Create epoch with custom trigger in JSON output."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            epoch_action="create",
            name="System Era",
            trigger="system",
            trigger_description="Auto detected",
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data["name"] == "System Era"


class TestEpochCreateError:
    """Test epoch create error handling."""

    def test_create_invalid_trigger_prints_error(self, cli_kernle, capsys):
        """Invalid trigger_type prints error (ValueError caught)."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            epoch_action="create",
            name="Bad Trigger Era",
            trigger="invalid_trigger_type",
            trigger_description=None,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Error:" in captured.out


class TestEpochCloseWithSummary:
    """Test epoch close with a summary string."""

    def test_close_with_summary_text_output(self, cli_kernle, capsys):
        """Close epoch with summary shows summary snippet in text output."""
        k, _storage = cli_kernle
        k.epoch_create(name="Closeable Era", trigger_type="declared")

        args = argparse.Namespace(
            epoch_action="close",
            id=None,
            summary="This era was about building the foundation for the project and testing all the things we need",
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Epoch closed." in captured.out
        assert "Summary:" in captured.out


class TestEpochCloseJson:
    """Test epoch close with JSON output."""

    def test_close_json_output_with_consolidation(self, cli_kernle):
        """Close epoch with --json returns structured data."""
        k, _storage = cli_kernle
        k.epoch_create(name="JSON Close Era", trigger_type="declared")

        args = argparse.Namespace(
            epoch_action="close",
            id=None,
            summary=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data["closed"] is True
        # consolidation should be present
        assert "consolidation" in data

    def test_close_json_no_open_epoch(self, cli_kernle):
        """Close epoch with --json when no open epoch returns closed: false."""
        k, _storage = cli_kernle

        args = argparse.Namespace(
            epoch_action="close",
            id=None,
            summary=None,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data["closed"] is False


class TestEpochListJson:
    """Test epoch list with JSON output."""

    def test_list_json_output(self, cli_kernle):
        """List epochs with --json returns structured array."""
        k, _storage = cli_kernle
        k.epoch_create(name="JSON List Era 1", trigger_type="declared")
        k.epoch_create(name="JSON List Era 2", trigger_type="declared")

        args = argparse.Namespace(
            epoch_action="list",
            limit=20,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert isinstance(data, list)
        assert len(data) == 2
        names = [e["name"] for e in data]
        assert "JSON List Era 2" in names

    def test_list_json_fields(self, cli_kernle):
        """JSON list includes all expected fields."""
        k, _storage = cli_kernle
        k.epoch_create(name="Fields Era", trigger_type="system")

        args = argparse.Namespace(
            epoch_action="list",
            limit=20,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        epoch = data[0]
        for field in ("id", "epoch_number", "name", "started_at", "trigger_type"):
            assert field in epoch, f"Missing field: {field}"


class TestEpochListEmpty:
    """Test epoch list when no epochs exist."""

    def test_list_empty_text(self, cli_kernle, capsys):
        """No epochs found message for empty list."""
        k, _storage = cli_kernle
        args = argparse.Namespace(
            epoch_action="list",
            limit=20,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "No epochs found." in captured.out


class TestEpochListWithSummary:
    """Test epoch list text with summary display."""

    def test_list_shows_summary_snippet(self, cli_kernle, capsys):
        """List text output shows summary snippet when epoch has one."""
        k, _storage = cli_kernle
        eid = k.epoch_create(name="Summarized Era", trigger_type="declared")
        k.epoch_close(
            epoch_id=eid, summary="This era was about exploration and discovery of new approaches"
        )

        args = argparse.Namespace(
            epoch_action="list",
            limit=20,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Summary:" in captured.out
        assert "exploration" in captured.out


class TestEpochShow:
    """Test epoch show action (text and JSON)."""

    def test_show_text_basic(self, cli_kernle, capsys):
        """Show epoch in text mode displays all fields."""
        k, _storage = cli_kernle
        eid = k.epoch_create(
            name="Show Era", trigger_type="detected", trigger_description="Big change"
        )

        args = argparse.Namespace(
            epoch_action="show",
            id=eid,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Show Era" in captured.out
        assert "ACTIVE" in captured.out
        assert "detected" in captured.out
        assert "Big change" in captured.out

    def test_show_text_closed_epoch(self, cli_kernle, capsys):
        """Show closed epoch displays ended date and summary."""
        k, _storage = cli_kernle
        eid = k.epoch_create(name="Closed Show Era", trigger_type="declared")
        k.epoch_close(epoch_id=eid, summary="All done here")

        args = argparse.Namespace(
            epoch_action="show",
            id=eid,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "closed" in captured.out
        assert "All done here" in captured.out

    def test_show_not_found(self, cli_kernle, capsys):
        """Show epoch with non-existent ID prints not found."""
        k, _storage = cli_kernle
        fake_id = "00000000-0000-0000-0000-000000000000"

        args = argparse.Namespace(
            epoch_action="show",
            id=fake_id,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_show_json_output(self, cli_kernle):
        """Show epoch with --json returns full structured data."""
        k, _storage = cli_kernle
        eid = k.epoch_create(name="JSON Show Era", trigger_type="system")

        args = argparse.Namespace(
            epoch_action="show",
            id=eid,
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data["name"] == "JSON Show Era"
        assert data["id"] == eid
        assert data["trigger_type"] == "system"
        assert data["ended_at"] is None
        for field in (
            "epoch_number",
            "key_belief_ids",
            "key_relationship_ids",
            "key_goal_ids",
            "dominant_drive_ids",
        ):
            assert field in data, f"Missing field: {field}"


class TestEpochCurrentText:
    """Test epoch current with text output."""

    def test_current_text_active(self, cli_kernle, capsys):
        """Current epoch text output shows name and start date."""
        k, _storage = cli_kernle
        k.epoch_create(name="Active Era", trigger_type="declared")

        args = argparse.Namespace(
            epoch_action="current",
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Current epoch:" in captured.out
        assert "Active Era" in captured.out
        assert "Started:" in captured.out

    def test_current_text_none(self, cli_kernle, capsys):
        """Current epoch text output when none active."""
        k, _storage = cli_kernle

        args = argparse.Namespace(
            epoch_action="current",
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "No active epoch." in captured.out

    def test_current_json_none(self, cli_kernle):
        """Current epoch JSON output when none active returns null."""
        k, _storage = cli_kernle

        args = argparse.Namespace(
            epoch_action="current",
            json=True,
        )

        with patch("sys.stdout", new=StringIO()) as out:
            cmd_epoch(args, k)

        data = json.loads(out.getvalue())
        assert data is None


class TestEpochShowWithKeyIds:
    """Test epoch show text output with key_belief_ids etc. populated."""

    def test_show_text_with_all_key_ids(self, cli_kernle, capsys):
        """Show epoch displays counts for all key ID lists."""
        k, storage = cli_kernle

        # Create an epoch directly with key IDs populated
        epoch = Epoch(
            id=str(uuid.uuid4()),
            stack_id="epoch_test_agent",
            epoch_number=1,
            name="Rich Era",
            started_at=datetime.now(timezone.utc),
            trigger_type="declared",
            key_belief_ids=["b1", "b2", "b3"],
            key_relationship_ids=["r1"],
            key_goal_ids=["g1", "g2"],
            dominant_drive_ids=["d1", "d2", "d3", "d4"],
        )
        storage.save_epoch(epoch)

        args = argparse.Namespace(
            epoch_action="show",
            id=epoch.id,
            json=False,
        )

        cmd_epoch(args, k)
        captured = capsys.readouterr()
        assert "Key beliefs: 3" in captured.out
        assert "Key relationships: 1" in captured.out
        assert "Key goals: 2" in captured.out
        assert "Dominant drives: 4" in captured.out
