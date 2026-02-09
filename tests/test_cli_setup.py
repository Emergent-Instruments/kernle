"""Tests for kernle setup CLI commands."""

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

from kernle.cli.commands.setup import (
    _build_claude_code_hooks,
    _deep_merge,
    _get_memory_flush_prompt,
    cmd_setup,
    setup_claude_code,
)

# --- Helpers ---


def make_args(**kwargs):
    """Create an argparse Namespace with defaults."""
    defaults = {"platform": None, "force": False, "enable": False}
    defaults.update(kwargs)
    # Use setattr for 'global' since it's a keyword
    ns = Namespace(**{k: v for k, v in defaults.items() if k != "global"})
    setattr(ns, "global", defaults.get("global", False))
    return ns


# --- _deep_merge tests ---


class TestDeepMerge:
    def test_merge_flat_dicts(self):
        base = {"a": 1, "b": 2}
        updates = {"c": 3}
        result = _deep_merge(base, updates)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_overwrites_flat_value(self):
        base = {"a": 1}
        updates = {"a": 2}
        result = _deep_merge(base, updates)
        assert result == {"a": 2}

    def test_merge_nested_dicts(self):
        base = {"a": {"x": 1, "y": 2}}
        updates = {"a": {"z": 3}}
        result = _deep_merge(base, updates)
        assert result == {"a": {"x": 1, "y": 2, "z": 3}}

    def test_merge_deeply_nested(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        updates = {"a": {"b": {"e": 3}}}
        result = _deep_merge(base, updates)
        assert result == {"a": {"b": {"c": 1, "d": 2, "e": 3}}}

    def test_merge_does_not_mutate_base(self):
        base = {"a": 1}
        updates = {"b": 2}
        _deep_merge(base, updates)
        assert base == {"a": 1}

    def test_merge_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_empty_updates(self):
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_both_empty(self):
        result = _deep_merge({}, {})
        assert result == {}

    def test_merge_overwrites_non_dict_with_dict(self):
        base = {"a": 1}
        updates = {"a": {"nested": True}}
        result = _deep_merge(base, updates)
        assert result == {"a": {"nested": True}}

    def test_merge_overwrites_dict_with_non_dict(self):
        base = {"a": {"nested": True}}
        updates = {"a": "flat"}
        result = _deep_merge(base, updates)
        assert result == {"a": "flat"}


# --- _build_claude_code_hooks tests ---


class TestBuildClaudeCodeHooks:
    def test_with_stack_id(self):
        result = _build_claude_code_hooks("my-stack")
        assert "hooks" in result
        hooks = result["hooks"]
        assert set(hooks.keys()) == {
            "SessionStart",
            "PreToolUse",
            "PreCompact",
            "SessionEnd",
        }

    def test_stack_id_in_commands(self):
        result = _build_claude_code_hooks("my-stack")
        hooks = result["hooks"]
        # Each event should have the -s flag with the stack_id
        for event in ("SessionStart", "PreToolUse", "PreCompact", "SessionEnd"):
            command = hooks[event][0]["hooks"][0]["command"]
            assert "-s my-stack" in command

    def test_without_stack_id_omits_flag(self):
        result = _build_claude_code_hooks(None)
        hooks = result["hooks"]
        for event in ("SessionStart", "PreToolUse", "PreCompact", "SessionEnd"):
            command = hooks[event][0]["hooks"][0]["command"]
            assert "-s " not in command
            assert command.startswith("kernle hook ")

    def test_all_hooks_are_command_type(self):
        result = _build_claude_code_hooks("test")
        hooks = result["hooks"]
        for event in hooks:
            for entry in hooks[event]:
                for hook in entry["hooks"]:
                    assert hook["type"] == "command"

    def test_all_hooks_have_timeout(self):
        result = _build_claude_code_hooks("test")
        hooks = result["hooks"]
        for event in hooks:
            for entry in hooks[event]:
                for hook in entry["hooks"]:
                    assert hook["timeout"] == 10

    def test_pre_tool_use_has_matcher(self):
        result = _build_claude_code_hooks("test")
        pre_tool = result["hooks"]["PreToolUse"][0]
        assert pre_tool["matcher"] == "Write|Edit|NotebookEdit"

    def test_session_start_no_matcher(self):
        result = _build_claude_code_hooks("test")
        session_start = result["hooks"]["SessionStart"][0]
        assert "matcher" not in session_start

    def test_hook_commands_contain_event_names(self):
        result = _build_claude_code_hooks("test")
        hooks = result["hooks"]
        assert "session-start" in hooks["SessionStart"][0]["hooks"][0]["command"]
        assert "pre-tool-use" in hooks["PreToolUse"][0]["hooks"][0]["command"]
        assert "pre-compact" in hooks["PreCompact"][0]["hooks"][0]["command"]
        assert "session-end" in hooks["SessionEnd"][0]["hooks"][0]["command"]


# --- _get_memory_flush_prompt tests ---


class TestGetMemoryFlushPrompt:
    def test_contains_stack_id(self):
        prompt = _get_memory_flush_prompt("my-stack")
        assert "my-stack" in prompt

    def test_contains_kernle_command(self):
        prompt = _get_memory_flush_prompt("test-stack")
        assert "kernle -s test-stack checkpoint" in prompt

    def test_returns_string(self):
        result = _get_memory_flush_prompt("abc")
        assert isinstance(result, str)
        assert len(result) > 0


# --- setup_claude_code tests ---


class TestSetupClaudeCode:
    def test_creates_new_settings_file(self, tmp_path):
        """Creates .claude/settings.json when it doesn't exist."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("kernle.cli.commands.setup.Path") as mock_path_cls:
            mock_path_cls.cwd.return_value = project_dir
            mock_path_cls.home.return_value = tmp_path / "home"
            # Make Path() constructor work normally for everything else
            mock_path_cls.side_effect = lambda *a, **kw: type(project_dir)(*a, **kw)

        # Instead of mocking Path, we directly call with mocked cwd
        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("test-stack")

        settings_file = project_dir / ".claude" / "settings.json"
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert "hooks" in data
        assert "SessionStart" in data["hooks"]

    def test_creates_global_settings_file(self, tmp_path):
        """Creates ~/.claude/settings.json when global=True."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        with patch("kernle.cli.commands.setup.Path.home", return_value=home_dir):
            setup_claude_code("test-stack", global_install=True)

        settings_file = home_dir / ".claude" / "settings.json"
        assert settings_file.exists()
        data = json.loads(settings_file.read_text())
        assert "hooks" in data

    def test_merges_with_existing_settings(self, tmp_path):
        """Merges hooks into existing settings.json without clobbering."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        settings_file = claude_dir / "settings.json"
        existing = {"allowedTools": ["Read", "Write"], "hooks": {}}
        settings_file.write_text(json.dumps(existing))

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("test-stack")

        data = json.loads(settings_file.read_text())
        # Original key preserved
        assert data["allowedTools"] == ["Read", "Write"]
        # Hooks merged in
        assert "SessionStart" in data["hooks"]

    def test_detects_already_configured(self, tmp_path, capsys):
        """Detects existing kernle hooks and exits without force."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        settings_file = claude_dir / "settings.json"

        hooks_config = _build_claude_code_hooks("test-stack")
        settings_file.write_text(json.dumps(hooks_config))

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("test-stack")

        captured = capsys.readouterr()
        assert "already configured" in captured.out

    def test_force_overwrites_existing(self, tmp_path, capsys):
        """--force overwrites existing kernle hooks."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        settings_file = claude_dir / "settings.json"

        hooks_config = _build_claude_code_hooks("old-stack")
        settings_file.write_text(json.dumps(hooks_config))

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("new-stack", force=True)

        data = json.loads(settings_file.read_text())
        # Should have new-stack in the commands
        cmd = data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
        assert "-s new-stack" in cmd

        captured = capsys.readouterr()
        assert "already configured" not in captured.out

    def test_handles_corrupt_json(self, tmp_path, capsys):
        """Handles corrupt settings.json gracefully by writing fresh."""
        project_dir = tmp_path / "project"
        claude_dir = project_dir / ".claude"
        claude_dir.mkdir(parents=True)
        settings_file = claude_dir / "settings.json"
        settings_file.write_text("not valid json {{{")

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("test-stack")

        # Should have written valid JSON despite corrupt original
        data = json.loads(settings_file.read_text())
        assert "hooks" in data

    def test_prints_verify_hint(self, tmp_path, capsys):
        """Output includes the verify hint."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("test-stack")

        captured = capsys.readouterr()
        assert "kernle doctor --full" in captured.out

    def test_prints_stack_id_in_output(self, tmp_path, capsys):
        """Output mentions the stack id."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        with patch("kernle.cli.commands.setup.Path.cwd", return_value=project_dir):
            setup_claude_code("my-fancy-stack")

        captured = capsys.readouterr()
        assert "my-fancy-stack" in captured.out


# --- cmd_setup tests ---


class TestCmdSetup:
    def _make_kernle_mock(self, stack_id="default"):
        k = MagicMock()
        k.stack_id = stack_id
        return k

    def test_no_platform_shows_usage(self, capsys):
        """No platform argument shows available platforms."""
        args = make_args(platform=None)
        k = self._make_kernle_mock()
        cmd_setup(args, k)
        captured = capsys.readouterr()
        assert "Available platforms:" in captured.out
        assert "openclaw" in captured.out
        assert "claude-code" in captured.out

    def test_unknown_platform_shows_error(self, capsys):
        """Unknown platform shows error message."""
        args = make_args(platform="unknown-thing")
        k = self._make_kernle_mock()
        cmd_setup(args, k)
        captured = capsys.readouterr()
        assert "Unknown platform" in captured.out

    @patch("kernle.cli.commands.setup.setup_claude_code")
    def test_dispatches_claude_code(self, mock_setup_cc):
        """Platform 'claude-code' dispatches to setup_claude_code."""
        args = make_args(platform="claude-code")
        setattr(args, "global", False)
        k = self._make_kernle_mock("my-stack")
        cmd_setup(args, k)
        mock_setup_cc.assert_called_once_with("my-stack", False, False)

    @patch("kernle.cli.commands.setup.setup_claude_code")
    def test_dispatches_cowork(self, mock_setup_cc):
        """Platform 'cowork' dispatches to setup_claude_code (same as claude-code)."""
        args = make_args(platform="cowork")
        setattr(args, "global", True)
        k = self._make_kernle_mock("cowork-stack")
        cmd_setup(args, k)
        mock_setup_cc.assert_called_once_with("cowork-stack", False, True)

    @patch("kernle.cli.commands.setup.setup_openclaw")
    def test_dispatches_openclaw(self, mock_setup_oc):
        """Platform 'openclaw' dispatches to setup_openclaw."""
        args = make_args(platform="openclaw", force=True, enable=True)
        k = self._make_kernle_mock("my-stack")
        cmd_setup(args, k)
        mock_setup_oc.assert_called_once_with("my-stack", True, True)

    @patch("kernle.cli.commands.setup.setup_claude_code")
    def test_passes_force_flag(self, mock_setup_cc):
        """Force flag is passed through to platform setup."""
        args = make_args(platform="claude-code", force=True)
        setattr(args, "global", False)
        k = self._make_kernle_mock("s")
        cmd_setup(args, k)
        mock_setup_cc.assert_called_once_with("s", True, False)

    @patch("kernle.cli.commands.setup.setup_claude_code")
    def test_passes_global_flag(self, mock_setup_cc):
        """Global flag is passed through for claude-code."""
        args = make_args(platform="claude-code")
        setattr(args, "global", True)
        k = self._make_kernle_mock("s")
        cmd_setup(args, k)
        mock_setup_cc.assert_called_once_with("s", False, True)
