"""Tests for hook fail-closed behavior.

The PreToolUse hook must deny memory writes even when internal errors occur.
A bare `except: pass` that silently permits writes is a safety bug.
"""

import io
import json
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

from kernle.cli.commands.hook import cmd_hook_pre_tool_use


def make_args(**kwargs):
    """Create an argparse Namespace with defaults."""
    defaults = {"stack": None, "hook_event": None}
    defaults.update(kwargs)
    return Namespace(**defaults)


def run_hook(func, stdin_data, args=None):
    """Run a hook function with mocked stdin/stdout, return (output_dict, exit_code)."""
    if args is None:
        args = make_args()

    old_stdin = sys.stdin
    old_stdout = sys.stdout

    sys.stdin = io.StringIO(json.dumps(stdin_data))
    captured = io.StringIO()
    sys.stdout = captured

    exit_code = None
    try:
        func(args)
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout

    output = captured.getvalue()
    if output:
        try:
            return json.loads(output), exit_code
        except json.JSONDecodeError:
            return output, exit_code
    return None, exit_code


DENY_SCHEMA_KEYS = {"hookEventName", "permissionDecision", "permissionDecisionReason"}


class TestHookFailClosed:
    """PreToolUse hook must deny on internal errors (fail-closed)."""

    @patch("kernle.Kernle")
    def test_hook_denies_on_internal_error(self, mock_kernle_cls):
        """If Kernle constructor raises, output must still deny."""
        mock_kernle_cls.side_effect = RuntimeError("DB init failed")

        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )

        assert exit_code == 0
        assert output is not None
        hook_output = output["hookSpecificOutput"]
        assert hook_output["permissionDecision"] == "deny"
        assert hook_output["hookEventName"] == "PreToolUse"

    @patch("kernle.Kernle")
    def test_hook_emits_deny_on_capture_failure(self, mock_kernle_cls):
        """If k.raw() throws, output must still deny."""
        mock_k = MagicMock()
        mock_k.raw.side_effect = Exception("Storage write failed")
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "data"},
            },
        )

        assert exit_code == 0
        assert output is not None
        hook_output = output["hookSpecificOutput"]
        assert hook_output["permissionDecision"] == "deny"
        assert hook_output["hookEventName"] == "PreToolUse"

    @patch("kernle.Kernle")
    def test_hook_deny_schema_matches_normal_deny(self, mock_kernle_cls):
        """Error deny and normal deny must have the same hookSpecificOutput keys."""
        # Normal deny
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        normal_output, _ = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )

        # Error deny
        mock_kernle_cls.side_effect = RuntimeError("boom")

        error_output, _ = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )

        normal_keys = set(normal_output["hookSpecificOutput"].keys())
        error_keys = set(error_output["hookSpecificOutput"].keys())

        assert normal_keys == DENY_SCHEMA_KEYS
        assert error_keys == DENY_SCHEMA_KEYS

    @patch("kernle.Kernle")
    def test_hook_normal_deny_still_works(self, mock_kernle_cls):
        """Memory-file write produces deny with correct hookSpecificOutput."""
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )

        assert exit_code == 0
        assert output is not None
        hook_output = output["hookSpecificOutput"]
        assert hook_output["hookEventName"] == "PreToolUse"
        assert hook_output["permissionDecision"] == "deny"
        assert "permissionDecisionReason" in hook_output

    def test_hook_normal_allow_still_works(self):
        """Non-memory-file write produces no deny output."""
        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {"tool_input": {"file_path": "src/main.py", "content": "code"}},
        )

        assert exit_code == 0
        assert output is None

    @patch("kernle.Kernle")
    def test_hook_always_exits_zero(self, mock_kernle_cls):
        """Both error and success paths must exit(0)."""
        # Success path
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        _, exit_code_success = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )
        assert exit_code_success == 0

        # Error path
        mock_kernle_cls.side_effect = RuntimeError("crash")

        _, exit_code_error = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/notes.md", "content": "hello"},
            },
        )
        assert exit_code_error == 0

        # Allow path (non-memory file)
        mock_kernle_cls.side_effect = None
        _, exit_code_allow = run_hook(
            cmd_hook_pre_tool_use,
            {"tool_input": {"file_path": "src/main.py", "content": "code"}},
        )
        assert exit_code_allow == 0
