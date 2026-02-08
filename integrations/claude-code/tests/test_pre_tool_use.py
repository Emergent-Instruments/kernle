"""Tests for pre_tool_use.py — PreToolUse hook."""

import subprocess

from pre_tool_use import main


class TestPreToolUse:
    def test_blocks_write_to_memory_dir(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "memory/2026-02-07.md", "content": "today I learned..."},
        )

        output, exit_code = capture_hook_output(main, hook_input)

        assert exit_code == 0
        assert output is not None
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "Kernle" in output["hookSpecificOutput"]["permissionDecisionReason"]

    def test_blocks_write_to_memory_md(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "MEMORY.md", "content": "updated memory"},
        )

        output, _ = capture_hook_output(main, hook_input)

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_allows_write_to_non_memory_path(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "src/index.ts", "content": "code"},
        )

        output, exit_code = capture_hook_output(main, hook_input)

        assert exit_code == 0
        assert output is None  # No output means allow

    def test_captures_content_as_raw_entry(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "memory/daily.md", "content": "important note"},
        )

        capture_hook_output(main, hook_input)

        # Find the raw call (should be one of the subprocess calls)
        raw_calls = [c for c in mock_subprocess.call_args_list if "raw" in c[0][0]]
        assert len(raw_calls) == 1
        raw_content = raw_calls[0][0][0][-1]  # Last arg is content
        assert "[memory-capture] memory/daily.md" in raw_content
        assert "important note" in raw_content

    def test_truncates_large_content(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "MEMORY.md", "content": "x" * 3000},
        )

        capture_hook_output(main, hook_input)

        raw_calls = [c for c in mock_subprocess.call_args_list if "raw" in c[0][0]]
        raw_content = raw_calls[0][0][0][-1]
        assert "[truncated]" in raw_content

    def test_handles_nested_memory_path(
        self, mock_subprocess, make_hook_input, capture_hook_output
    ):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={
                "file_path": "/home/user/project/memory/daily.md",
                "content": "note",
            },
        )

        output, _ = capture_hook_output(main, hook_input)

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_uses_path_as_fallback(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="saved\n", stderr=""
        )

        hook_input = make_hook_input(
            tool_name="Edit",
            tool_input={"path": "memory/test.md", "new_string": "new content"},
        )

        output, _ = capture_hook_output(main, hook_input)

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_allows_when_no_file_path(self, mock_subprocess, make_hook_input, capture_hook_output):
        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"content": "some content"},
        )

        output, exit_code = capture_hook_output(main, hook_input)

        assert exit_code == 0
        assert output is None

    def test_exits_zero_on_error(self, mock_subprocess, make_hook_input, capture_hook_output):
        mock_subprocess.side_effect = Exception("unexpected")

        hook_input = make_hook_input(
            tool_name="Write",
            tool_input={"file_path": "memory/test.md", "content": "note"},
        )

        # Should not raise
        output, exit_code = capture_hook_output(main, hook_input)
        assert exit_code == 0
