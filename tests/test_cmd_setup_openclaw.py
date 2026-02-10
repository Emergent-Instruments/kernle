"""Tests for kernle setup openclaw CLI commands (kernle/cli/commands/setup.py).

Covers setup_openclaw and _enable_openclaw_hook which are the uncovered portions:
lines 55-131 (setup_openclaw) and 143-225 (_enable_openclaw_hook).
"""

import json
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
    hooks_dir = tmp_path / "kernle" / "hooks" / "openclaw"
    hooks_dir.mkdir(parents=True)
    (hooks_dir / "hook.sh").write_text("#!/bin/bash\necho ok")
    return tmp_path / "kernle" / "hooks"


@pytest.fixture
def home_dir(tmp_path):
    """Create a fake home directory."""
    return tmp_path / "home"


class TestGetHooksDir:
    def test_returns_hooks_path(self):
        result = get_hooks_dir()
        # Should be kernle/hooks relative to the package
        assert result.name == "hooks"
        assert "kernle" in str(result)


class TestSetupOpenClaw:
    def test_source_not_found(self, tmp_path, capsys):
        """When hook source files don't exist, prints error."""
        fake_hooks = tmp_path / "empty_hooks"
        fake_hooks.mkdir()

        with patch("kernle.cli.commands.setup.get_hooks_dir", return_value=fake_hooks):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "hook files not found" in out

    def test_installs_to_user_hooks(self, hooks_source, home_dir, capsys):
        """Installs to ~/.config/openclaw/hooks/ when bundled dir doesn't exist."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        target = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        assert target.exists()

        out = capsys.readouterr().out
        assert "Installed OpenClaw hook to user hooks" in out

    def test_installs_to_bundled_hooks(self, hooks_source, home_dir, capsys):
        """Installs to ~/openclaw/src/hooks/bundled/ when it exists."""
        bundled_parent = home_dir / "openclaw" / "src" / "hooks" / "bundled"
        bundled_parent.mkdir(parents=True)

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        target = bundled_parent / "kernle-load"
        assert target.exists()

        out = capsys.readouterr().out
        assert "Installed OpenClaw hook to bundled hooks" in out

    def test_already_exists_no_force(self, hooks_source, home_dir, capsys):
        """When target already exists without --force, warns user."""
        user_hooks = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        user_hooks.mkdir(parents=True)

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=False)

        out = capsys.readouterr().out
        assert "already installed" in out
        assert "--force" in out

    def test_already_exists_with_enable(self, hooks_source, home_dir, capsys):
        """When target exists, still tries to enable if --enable is set."""
        user_hooks = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        user_hooks.mkdir(parents=True)

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("kernle.cli.commands.setup._enable_openclaw_hook") as mock_enable,
        ):
            setup_openclaw("test-stack", force=False, enable=True)

        mock_enable.assert_called_once_with("test-stack")

    def test_force_overwrites_existing(self, hooks_source, home_dir, capsys):
        """--force overwrites existing hook directory."""
        user_hooks = home_dir / ".config" / "openclaw" / "hooks" / "kernle-load"
        user_hooks.mkdir(parents=True)
        (user_hooks / "old_file.txt").write_text("old")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack", force=True)

        out = capsys.readouterr().out
        assert "Installed OpenClaw hook" in out
        assert (user_hooks / "hook.sh").exists()
        assert not (user_hooks / "old_file.txt").exists()

    def test_install_with_enable(self, hooks_source, home_dir, capsys):
        """--enable calls _enable_openclaw_hook after install."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("kernle.cli.commands.setup._enable_openclaw_hook") as mock_enable,
        ):
            setup_openclaw("test-stack", enable=True)

        mock_enable.assert_called_once_with("test-stack")

    def test_install_without_enable_shows_instructions(self, hooks_source, home_dir, capsys):
        """Without --enable, shows next steps instructions."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "Next steps:" in out
        assert "kernle setup openclaw --enable" in out

    def test_install_config_exists_hook_enabled(self, hooks_source, home_dir, capsys):
        """Config exists and hook already enabled — shows status."""
        # Create config with hook enabled
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config = {"hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}}}
        (config_dir / "openclaw.json").write_text(json.dumps(config))

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "already enabled" in out

    def test_install_config_exists_hook_not_enabled(self, hooks_source, home_dir, capsys):
        """Config exists but hook not enabled — shows warning."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config = {"hooks": {"internal": {"entries": {}}}}
        (config_dir / "openclaw.json").write_text(json.dumps(config))

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "not enabled" in out

    def test_install_config_read_error(self, hooks_source, home_dir, capsys):
        """Corrupt config file — shows warning."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        (config_dir / "openclaw.json").write_text("{invalid json")

        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "Could not read config" in out

    def test_install_no_config_file(self, hooks_source, home_dir, capsys):
        """No config file — shows warning about missing config."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "config not found" in out

    def test_copy_error(self, hooks_source, home_dir, capsys):
        """Copy failure is handled gracefully."""
        with (
            patch("kernle.cli.commands.setup.get_hooks_dir", return_value=hooks_source),
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch(
                "kernle.cli.commands.setup.shutil.copytree",
                side_effect=OSError("Permission denied"),
            ),
        ):
            setup_openclaw("test-stack")

        out = capsys.readouterr().out
        assert "Failed to copy hook files" in out


class TestEnableOpenClawHook:
    def test_creates_new_config(self, home_dir, capsys):
        """Creates config file when it doesn't exist."""
        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        config_path = home_dir / ".openclaw" / "openclaw.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert config["hooks"]["internal"]["enabled"] is True
        assert config["hooks"]["internal"]["entries"]["kernle-load"]["enabled"] is True
        assert config["agents"]["defaults"]["compaction"]["memoryFlush"]["enabled"] is True

        out = capsys.readouterr().out
        assert "Updated openclaw.json" in out
        assert "Enabled session start hook" in out
        assert "Configured pre-compaction memory flush" in out
        assert "Kernle Setup Complete" in out

    def test_merges_with_existing_config(self, home_dir, capsys):
        """Merges into existing config preserving other settings."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        existing = {"gateway": {"port": 8080}, "hooks": {"custom": True}}
        (config_dir / "openclaw.json").write_text(json.dumps(existing))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        config = json.loads((config_dir / "openclaw.json").read_text())
        # Existing config preserved
        assert config["gateway"]["port"] == 8080
        # New config merged in
        assert config["hooks"]["internal"]["entries"]["kernle-load"]["enabled"] is True

    def test_already_fully_configured(self, home_dir, capsys):
        """Detects when already fully configured."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config = {
            "hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}},
            "agents": {"defaults": {"compaction": {"memoryFlush": {"enabled": True}}}},
        }
        (config_dir / "openclaw.json").write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        out = capsys.readouterr().out
        assert "already fully configured" in out

    def test_hook_enabled_flush_not(self, home_dir, capsys):
        """Hook enabled but flush not configured — only configures flush."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config = {
            "hooks": {"internal": {"entries": {"kernle-load": {"enabled": True}}}},
        }
        (config_dir / "openclaw.json").write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        out = capsys.readouterr().out
        assert "Configured pre-compaction memory flush" in out
        # Should NOT say "Enabled session start hook" since it's already enabled
        assert "Enabled session start hook" not in out

    def test_flush_configured_hook_not(self, home_dir, capsys):
        """Flush configured but hook not enabled — only enables hook."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config = {
            "hooks": {"internal": {"entries": {}}},
            "agents": {"defaults": {"compaction": {"memoryFlush": {"enabled": True}}}},
        }
        (config_dir / "openclaw.json").write_text(json.dumps(config))

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            result = _enable_openclaw_hook("test-stack")

        assert result is True
        out = capsys.readouterr().out
        assert "Enabled session start hook" in out
        assert "Configured pre-compaction memory flush" not in out

    def test_stack_id_in_prompt(self, home_dir, capsys):
        """Stack ID appears in the memory flush prompt stored in config."""
        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            _enable_openclaw_hook("my-unique-stack")

        config_path = home_dir / ".openclaw" / "openclaw.json"
        config = json.loads(config_path.read_text())
        prompt = config["agents"]["defaults"]["compaction"]["memoryFlush"]["prompt"]
        assert "my-unique-stack" in prompt

    def test_restart_message(self, home_dir, capsys):
        """Output includes restart gateway message."""
        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            _enable_openclaw_hook("test-stack")

        out = capsys.readouterr().out
        assert "openclaw gateway restart" in out

    def test_config_write_error(self, home_dir, capsys):
        """Handles config write errors gracefully."""
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "openclaw.json"
        config_path.write_text("{}")

        with (
            patch("kernle.cli.commands.setup.Path.home", return_value=home_dir),
            patch("builtins.open", side_effect=OSError("disk full")),
        ):
            result = _enable_openclaw_hook("test-stack")

        assert result is False
        out = capsys.readouterr().out
        assert "Failed to enable hook" in out
