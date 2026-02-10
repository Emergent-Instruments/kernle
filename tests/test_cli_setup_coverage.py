"""Tests for kernle CLI setup command — covering uncovered branches.

Targets setup_openclaw(), _enable_openclaw_hook(), and get_hooks_dir().
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from kernle.cli.commands.setup import (
    _enable_openclaw_hook,
    get_hooks_dir,
    setup_openclaw,
)


@pytest.fixture
def hooks_source(tmp_path):
    """Create a fake openclaw hooks source directory."""
    hooks_dir = tmp_path / "kernle_pkg" / "hooks" / "openclaw"
    hooks_dir.mkdir(parents=True)
    # Put a real file in there so copytree works
    (hooks_dir / "hook.sh").write_text("#!/bin/bash\necho kernle")
    return hooks_dir


@pytest.fixture
def home_dir(tmp_path):
    """Fake home directory for Path.home()."""
    home = tmp_path / "home"
    home.mkdir()
    return home


class TestGetHooksDir:
    """Test get_hooks_dir returns the correct path."""

    def test_returns_path_relative_to_package(self):
        """get_hooks_dir returns kernle/hooks/ path."""
        result = get_hooks_dir()
        assert isinstance(result, Path)
        assert result.name == "hooks"
        # The parent should be the kernle package root
        assert "kernle" in str(result)


class TestSetupOpenclawSourceMissing:
    """Test setup_openclaw when source directory does not exist."""

    def test_missing_source_prints_error(self, tmp_path, capsys):
        """When hook source files are missing, print error."""
        # Patch get_hooks_dir to return a dir without openclaw subfolder
        fake_hooks = tmp_path / "no_hooks"
        fake_hooks.mkdir()

        with patch("kernle.cli.commands.setup.get_hooks_dir", return_value=fake_hooks):
            setup_openclaw("test-stack")

        captured = capsys.readouterr()
        assert "hook files not found" in captured.out


class TestSetupOpenclawInstallUser:
    """Test setup_openclaw installs to user hooks directory."""

    def test_installs_to_user_hooks_when_bundled_missing(self, hooks_source, home_dir, capsys):
        """When bundled hooks dir doesn't exist, installs to user hooks."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "Installed OpenClaw hook to user hooks" in captured.out

        target = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        assert target.exists()
        assert (target / "hook.sh").exists()

    def test_installs_to_bundled_hooks_when_parent_exists(self, hooks_source, home_dir, capsys):
        """When bundled hooks parent dir exists, installs there."""
        bundled_parent = home_dir / "openclaw" / "src" / "hooks" / "bundled"
        bundled_parent.mkdir(parents=True)

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "Installed OpenClaw hook to bundled hooks" in captured.out

        target = bundled_parent / "kernle-load"
        assert target.exists()


class TestSetupOpenclawAlreadyInstalled:
    """Test setup_openclaw when hook already exists."""

    def test_already_installed_without_force(self, hooks_source, home_dir, capsys):
        """Already installed prints warning without --force."""
        # Pre-install
        target = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        target.mkdir(parents=True)
        (target / "hook.sh").write_text("existing")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "already installed" in captured.out
        assert "--force" in captured.out

    def test_already_installed_with_enable(self, hooks_source, home_dir, capsys):
        """Already installed with --enable still calls _enable_openclaw_hook."""
        target = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        target.mkdir(parents=True)
        (target / "hook.sh").write_text("existing")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("kernle.cli.commands.setup._enable_openclaw_hook") as mock_enable,
        ):
            setup_openclaw("test-stack", force=False, enable=True)

        mock_enable.assert_called_once_with("test-stack")

    def test_force_overwrites_existing(self, hooks_source, home_dir, capsys):
        """--force replaces existing hook files."""
        target = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        target.mkdir(parents=True)
        (target / "old_file.txt").write_text("old")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=True, enable=False)

        captured = capsys.readouterr()
        assert "Installed OpenClaw hook" in captured.out
        # Old file should be gone, new file should exist
        assert not (target / "old_file.txt").exists()
        assert (target / "hook.sh").exists()


class TestSetupOpenclawConfigStatus:
    """Test setup_openclaw shows config status when not enabling."""

    def test_shows_hook_enabled_in_config(self, hooks_source, home_dir, capsys):
        """When hook is already enabled in config, prints status."""
        # Don't pre-install so it does a fresh install

        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config = {"hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}}}
        config_path.write_text(json.dumps(config))

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "already enabled" in captured.out

    def test_shows_hook_not_enabled(self, hooks_source, home_dir, capsys):
        """When hook exists in config but disabled, tells user."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config = {"hooks": {"internal": {"entries": {"kernle-load": {"enabled": False}}}}}
        config_path.write_text(json.dumps(config))

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "not enabled" in captured.out

    def test_shows_config_not_found(self, hooks_source, home_dir, capsys):
        """When openclaw config doesn't exist, tells user."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "config not found" in captured.out

    def test_shows_next_steps(self, hooks_source, home_dir, capsys):
        """After install, shows next steps with stack_id."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("my-agent", force=False, enable=False)

        captured = capsys.readouterr()
        assert "Next steps:" in captured.out
        assert "my-agent" in captured.out

    def test_handles_corrupt_config(self, hooks_source, home_dir, capsys):
        """When config file contains invalid JSON, warns user."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not valid json {{{")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "Could not read config" in captured.out


class TestSetupOpenclawWithEnable:
    """Test setup_openclaw with --enable."""

    def test_install_and_enable(self, hooks_source, home_dir, capsys):
        """Fresh install with --enable creates config and enables hook."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False, enable=True)

        captured = capsys.readouterr()
        assert "Installed OpenClaw hook" in captured.out

        # Config should have been created
        config_path = home_dir / ".openclaw" / "openclaw.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["hooks"]["internal"]["entries"]["kernle-load"]["enabled"] is True


class TestEnableOpenclawHook:
    """Test _enable_openclaw_hook function."""

    def test_creates_new_config(self, home_dir, capsys):
        """Creates openclaw.json when it doesn't exist."""
        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        config_path = home_dir / ".openclaw" / "openclaw.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert config["hooks"]["internal"]["enabled"] is True
        assert config["hooks"]["internal"]["entries"]["kernle-load"]["enabled"] is True
        assert config["agents"]["defaults"]["compaction"]["memoryFlush"]["enabled"] is True

        captured = capsys.readouterr()
        assert "Updated openclaw.json" in captured.out
        assert "Enabled session start hook" in captured.out
        assert "Configured pre-compaction memory flush" in captured.out

    def test_merges_with_existing_config(self, home_dir, capsys):
        """Merges kernle config into existing openclaw.json."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        existing = {"some_key": "preserved", "hooks": {"external": True}}
        config_path.write_text(json.dumps(existing))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        config = json.loads(config_path.read_text())
        # Existing key preserved
        assert config["some_key"] == "preserved"
        # External hook preserved
        assert config["hooks"]["external"] is True
        # Kernle config added
        assert config["hooks"]["internal"]["entries"]["kernle-load"]["enabled"] is True

    def test_already_fully_configured(self, home_dir, capsys):
        """When already fully configured, reports status."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config = {
            "hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}},
            "agents": {"defaults": {"compaction": {"memoryFlush": {"enabled": True}}}},
        }
        config_path.write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        captured = capsys.readouterr()
        assert "already fully configured" in captured.out

    def test_partially_configured_hook_only(self, home_dir, capsys):
        """When only hook enabled, adds flush config."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config = {
            "hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}},
        }
        config_path.write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        captured = capsys.readouterr()
        assert "Configured pre-compaction memory flush" in captured.out
        # Hook was already enabled, should NOT say "Enabled session start hook"
        assert "Enabled session start hook" not in captured.out

    def test_partially_configured_flush_only(self, home_dir, capsys):
        """When only flush configured, adds hook config."""
        config_path = home_dir / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config = {
            "agents": {"defaults": {"compaction": {"memoryFlush": {"enabled": True}}}},
        }
        config_path.write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        captured = capsys.readouterr()
        assert "Enabled session start hook" in captured.out
        # Flush was already configured, should NOT say "Configured pre-compaction"
        assert "Configured pre-compaction memory flush" not in captured.out

    def test_prints_restart_instructions(self, home_dir, capsys):
        """After enable, prints restart and stack info."""
        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            _enable_openclaw_hook("my-agent")

        captured = capsys.readouterr()
        assert "Restart OpenClaw gateway" in captured.out
        assert "my-agent" in captured.out
        assert "Kernle Setup Complete" in captured.out

    def test_handles_write_error(self, home_dir, capsys):
        """Returns False and prints error when config write fails."""
        # Make the parent directory unwritable by patching open to raise
        with (
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("builtins.open", side_effect=PermissionError("denied")),
        ):
            result = _enable_openclaw_hook("test-stack")

        assert result is False
        captured = capsys.readouterr()
        assert "Failed to enable hook" in captured.out


class TestSetupOpenclawCopyError:
    """Test setup_openclaw when copy fails."""

    def test_copy_failure_prints_error(self, hooks_source, home_dir, capsys):
        """When copytree fails, prints error."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source.parent),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("kernle.cli.commands.setup.shutil.copytree", side_effect=OSError("disk full")),
        ):
            setup_openclaw("test-stack", force=False, enable=False)

        captured = capsys.readouterr()
        assert "Failed to copy hook files" in captured.out
