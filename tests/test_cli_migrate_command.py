"""Tests for kernle.cli.commands.migrate — CLI boundary hardening.

Exercises negative-path scenarios for cmd_migrate: unknown actions,
missing arguments, and dry-run behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from kernle.cli.commands.migrate import cmd_migrate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal args namespace for cmd_migrate.

    Returns a SimpleNamespace with sensible defaults that can be overridden
    by the caller via keyword arguments.
    """
    defaults = {
        "migrate_action": None,
        "dry_run": False,
        "force": False,
        "tier": None,
        "list": False,
        "level": "minimal",
        "json": False,
        "window": 30,
        "link_all": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_kernle(stack_id="test-stack"):
    """Create a mock Kernle instance with the storage methods used by migrate."""
    k = MagicMock()
    k.stack_id = stack_id

    # _storage is used directly by migrate commands
    k._storage.get_beliefs.return_value = []
    k._storage.get_episodes.return_value = []
    k._storage.get_notes.return_value = []
    k._storage.get_values.return_value = []
    k._storage.get_goals.return_value = []
    k._storage.get_drives.return_value = []
    k._storage.get_relationships.return_value = []
    k._storage.list_raw.return_value = []

    return k


# ===========================================================================
# Test: Unknown or missing migrate action
# ===========================================================================


class TestMigrateUnknownAction:
    """cmd_migrate should handle unknown or None actions gracefully."""

    def test_none_action_prints_usage(self, capsys):
        """When migrate_action is None, cmd_migrate prints available actions."""
        k = _make_kernle()
        args = _make_args(migrate_action=None)

        # Should not raise
        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "Unknown migrate action" in output
        assert "seed-beliefs" in output
        assert "backfill-provenance" in output
        assert "link-raw" in output

    def test_unknown_action_prints_usage(self, capsys):
        """An unrecognized action string prints available actions."""
        k = _make_kernle()
        args = _make_args(migrate_action="frobnicate")

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "Unknown migrate action: frobnicate" in output
        assert "seed-beliefs" in output


# ===========================================================================
# Test: seed-beliefs with empty storage
# ===========================================================================


class TestMigrateSeedBeliefs:
    """Test seed-beliefs action boundary conditions."""

    def test_seed_beliefs_dry_run_no_existing(self, capsys):
        """Dry run with no existing beliefs shows what would be added."""
        k = _make_kernle()
        args = _make_args(migrate_action="seed-beliefs", dry_run=True)

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        # Should show dry run message
        assert "DRY RUN" in output

    def test_seed_beliefs_list_minimal(self, capsys):
        """List mode for minimal level shows the seed beliefs."""
        k = _make_kernle()
        args = _make_args(migrate_action="seed-beliefs", **{"list": True}, level="minimal")

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "Minimal Seed Beliefs" in output

    def test_seed_beliefs_list_full(self, capsys):
        """List mode for full level shows all seed beliefs."""
        k = _make_kernle()
        args = _make_args(migrate_action="seed-beliefs", **{"list": True}, level="full")

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "Full Seed Beliefs" in output

    def test_seed_beliefs_tier_ignored_for_minimal(self, capsys):
        """--tier flag is ignored for the minimal level, with a warning."""
        k = _make_kernle()
        args = _make_args(
            migrate_action="seed-beliefs",
            **{"list": True},
            level="minimal",
            tier=1,
        )

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        # Should warn that --tier is only for full
        assert "--tier is only valid with 'full' level" in output


# ===========================================================================
# Test: backfill-provenance with empty storage
# ===========================================================================


class TestMigrateBackfillProvenance:
    """Test backfill-provenance with empty or no memories."""

    def test_backfill_no_memories_prints_complete(self, capsys):
        """When all memory types return empty, prints all-done message."""
        k = _make_kernle()
        args = _make_args(migrate_action="backfill-provenance")

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "already have provenance metadata" in output

    def test_backfill_dry_run_empty(self, capsys):
        """Dry run with no memories needing updates prints all-done."""
        k = _make_kernle()
        args = _make_args(migrate_action="backfill-provenance", dry_run=True)

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "already have provenance metadata" in output


# ===========================================================================
# Test: link-raw with empty storage
# ===========================================================================


class TestMigrateLinkRaw:
    """Test link-raw with empty raw entries."""

    def test_link_raw_no_raw_entries(self, capsys):
        """When there are no raw entries, prints a helpful message."""
        k = _make_kernle()
        args = _make_args(migrate_action="link-raw")

        cmd_migrate(args, k)

        output = capsys.readouterr().out
        assert "No raw entries found" in output

    def test_link_raw_no_raw_entries_json(self, capsys):
        """JSON output mode with no raw entries returns error JSON."""
        k = _make_kernle()
        args = _make_args(migrate_action="link-raw", json=True)

        cmd_migrate(args, k)

        import json

        output = json.loads(capsys.readouterr().out)
        assert "error" in output
        assert "No raw entries found" in output["error"]
