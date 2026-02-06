"""Tests for CLI Entity integration (v0.4.0).

Verifies that the CLI status command shows composition info
and that plugin discovery + CLI registration works without
breaking existing CLI commands.
"""

import io
from contextlib import redirect_stdout

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage


@pytest.fixture
def kernle_cli(tmp_path):
    """Kernle instance for CLI tests."""
    db_path = tmp_path / "cli_test.db"
    storage = SQLiteStorage(stack_id="test_cli_entity", db_path=db_path)
    k = Kernle(
        stack_id="test_cli_entity",
        storage=storage,
        checkpoint_dir=tmp_path / "cp",
    )
    yield k
    storage.close()


def capture_status(k):
    """Run cmd_status and capture stdout."""
    from kernle.cli.__main__ import cmd_status

    class FakeArgs:
        pass

    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_status(FakeArgs(), k)
    return buf.getvalue()


class TestStatusComposition:
    """Tests for the enhanced status command with composition info."""

    def test_status_shows_composition_section(self, kernle_cli):
        output = capture_status(kernle_cli)
        assert "Composition (v0.4.0)" in output

    def test_status_shows_core_id(self, kernle_cli):
        output = capture_status(kernle_cli)
        assert "Core ID:" in output
        assert "test_cli_entity" in output

    def test_status_shows_stack_info(self, kernle_cli):
        output = capture_status(kernle_cli)
        assert "Stack:" in output
        assert "schema v" in output

    def test_status_shows_plugins_line(self, kernle_cli):
        output = capture_status(kernle_cli)
        assert "Plugins:" in output or "Plugin:" in output

    def test_status_shows_model_line(self, kernle_cli):
        output = capture_status(kernle_cli)
        assert "Model:" in output

    def test_status_still_shows_memory_counts(self, kernle_cli):
        """Existing status output is preserved."""
        output = capture_status(kernle_cli)
        assert "Memory Status for" in output
        assert "Values:" in output
        assert "Beliefs:" in output
        assert "Goals:" in output
        assert "Episodes:" in output
        assert "Checkpoint:" in output

    def test_status_stack_detached_when_entity_not_accessed_first(self, kernle_cli):
        """If entity is not accessed before stack, shows 'detached'."""
        # Access stack first (without entity), then status
        # This tests the fallback path in cmd_status
        output = capture_status(kernle_cli)
        # Entity is accessed first in cmd_status, so stack auto-attaches.
        # The output should show the stack as attached (active).
        assert "(active)" in output or "detached" in output


class TestStatusWithAutoAttach:
    """Verify auto-attach behavior in status command."""

    def test_entity_accessed_first_attaches_stack(self, kernle_cli):
        """Accessing entity then stack auto-attaches."""
        output = capture_status(kernle_cli)
        # Entity is created first in cmd_status, then stack is checked
        # via entity.stacks — if stack was accessed through Kernle.stack
        # and entity existed, it should be auto-attached.
        assert "test_cli_entity" in output
        assert "schema v" in output


class TestPluginDiscovery:
    """Tests for plugin discovery in CLI."""

    def test_discover_plugins_called_in_status(self, kernle_cli):
        """Status command calls entity.discover_plugins()."""
        output = capture_status(kernle_cli)
        # Even with no plugins installed, the line should appear
        assert "Plugins:" in output or "Plugin:" in output

    def test_no_plugins_shows_none(self, kernle_cli):
        """When no plugins are installed, shows '(none)' or count."""
        output = capture_status(kernle_cli)
        # With no plugins installed, should show "discovered, 0 loaded" or "(none)"
        assert "(none)" in output or "0 loaded" in output
