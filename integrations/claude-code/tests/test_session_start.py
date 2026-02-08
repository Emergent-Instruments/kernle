"""Tests for session_start.py — SessionStart hook."""

import subprocess

from session_start import main


class TestSessionStart:
    def test_outputs_additional_context_when_memory_loads(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="# Kernle Memory\n\nValues: be helpful\n", stderr=""
        )

        output, exit_code = capture_hook_output(main, make_hook_input())

        assert exit_code == 0
        assert output is not None
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert "be helpful" in output["hookSpecificOutput"]["additionalContext"]

    def test_outputs_nothing_when_load_fails(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )

        output, exit_code = capture_hook_output(main, make_hook_input())

        assert exit_code == 0
        assert output is None

    def test_resolves_stack_id_from_cwd(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="memory\n", stderr=""
        )

        capture_hook_output(main, make_hook_input(cwd="/Users/test/cool-project"))

        call_args = mock_subprocess.call_args[0][0]
        assert "-s" in call_args
        assert "cool-project" in call_args

    def test_handles_invalid_stdin(self, mock_subprocess, capture_hook_output):
        """When stdin is invalid JSON, exit 0 silently."""
        from io import StringIO
        from unittest.mock import patch as p

        stdout = StringIO()
        with p("sys.stdin", StringIO("not json")), p("sys.stdout", stdout):
            try:
                main()
            except SystemExit as e:
                assert e.code == 0

    def test_exits_zero_on_any_error(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.side_effect = Exception("unexpected")

        output, exit_code = capture_hook_output(main, make_hook_input())

        assert exit_code == 0
