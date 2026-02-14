"""Tests for kernle hook CLI commands."""

import io
import json
import os
import sys
import tempfile
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.commands.hook import (
    _extract_text,
    _is_memory_path,
    _read_last_messages,
    _resolve_hook_stack_id,
    _truncate,
    _validate_hook_input,
    cmd_hook_pre_compact,
    cmd_hook_pre_tool_use,
    cmd_hook_session_end,
    cmd_hook_session_start,
)

# --- Fixtures ---


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


def make_transcript(messages):
    """Create a temporary JSONL transcript file."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for msg in messages:
        f.write(json.dumps(msg) + "\n")
    f.close()
    return f.name


# --- TestResolveHookStackId ---


class TestResolveHookStackId:
    def test_explicit_stack_takes_priority(self):
        assert _resolve_hook_stack_id("explicit", "/some/project") == "explicit"

    def test_env_var_second_priority(self, monkeypatch):
        monkeypatch.setenv("KERNLE_STACK_ID", "from-env")
        assert _resolve_hook_stack_id(None, "/some/project") == "from-env"

    def test_cwd_third_priority(self, monkeypatch):
        monkeypatch.delenv("KERNLE_STACK_ID", raising=False)
        assert _resolve_hook_stack_id(None, "/Users/test/my-project") == "my-project"

    def test_skips_workspace_dir(self, monkeypatch):
        monkeypatch.delenv("KERNLE_STACK_ID", raising=False)
        with patch("kernle.utils.resolve_stack_id", return_value="auto"):
            result = _resolve_hook_stack_id(None, "/home/user/workspace")
        assert result == "auto"

    def test_skips_home_dir(self, monkeypatch):
        monkeypatch.delenv("KERNLE_STACK_ID", raising=False)
        with patch("kernle.utils.resolve_stack_id", return_value="auto"):
            result = _resolve_hook_stack_id(None, "/home")
        assert result == "auto"

    def test_falls_back_to_auto_resolve(self, monkeypatch):
        monkeypatch.delenv("KERNLE_STACK_ID", raising=False)
        with patch("kernle.utils.resolve_stack_id", return_value="auto"):
            result = _resolve_hook_stack_id(None, None)
        assert result == "auto"


# --- TestIsMemoryPath ---


class TestIsMemoryPath:
    def test_matches_memory_dir(self):
        assert _is_memory_path("memory/test.md")

    def test_matches_nested_memory_dir(self):
        assert _is_memory_path("/home/user/.claude/memory/notes.md")

    def test_matches_memory_md(self):
        assert _is_memory_path("/home/user/.claude/MEMORY.md")

    def test_does_not_match_normal_file(self):
        assert not _is_memory_path("src/main.py")

    def test_does_not_match_similar_names(self):
        assert not _is_memory_path("memorable.py")
        assert not _is_memory_path("in-memory-db.py")


# --- TestReadLastMessages ---


class TestReadLastMessages:
    def test_reads_last_user_message(self):
        path = make_transcript(
            [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"},
            ]
        )
        task, context = _read_last_messages(path)
        assert task == "Second message"
        os.unlink(path)

    def test_reads_last_assistant_message(self):
        path = make_transcript(
            [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        )
        task, context = _read_last_messages(path)
        assert context == "Answer"
        os.unlink(path)

    def test_handles_content_blocks(self):
        path = make_transcript(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        )
        task, context = _read_last_messages(path)
        assert task == "Hello World"
        os.unlink(path)

    def test_truncates_long_user_message(self):
        path = make_transcript([{"role": "user", "content": "x" * 300}])
        task, context = _read_last_messages(path, max_user=50)
        assert len(task) == 50
        assert task.endswith("...")
        os.unlink(path)

    def test_returns_fallback_when_no_file(self):
        task, context = _read_last_messages("/nonexistent/path.jsonl")
        assert task == "Session ended"
        assert context is None

    def test_returns_fallback_when_none(self):
        task, context = _read_last_messages(None)
        assert task == "Session ended"
        assert context is None

    def test_handles_empty_transcript(self):
        path = make_transcript([])
        task, context = _read_last_messages(path)
        assert task == "Session ended"
        os.unlink(path)

    def test_skips_invalid_json_lines(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write("not json\n")
        f.write(json.dumps({"role": "user", "content": "valid"}) + "\n")
        f.close()
        task, context = _read_last_messages(f.name)
        assert task == "valid"
        os.unlink(f.name)


# --- TestExtractText ---


class TestExtractText:
    def test_string_content(self):
        assert _extract_text({"content": "hello"}) == "hello"

    def test_list_content(self):
        entry = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        assert _extract_text(entry) == "a b"

    def test_empty_content(self):
        assert _extract_text({}) == ""


# --- TestTruncate ---


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("x" * 100, 50)
        assert len(result) == 50
        assert result.endswith("...")


# --- TestHookSessionStart ---


class TestHookSessionStart:
    @patch("kernle.Kernle")
    def test_outputs_additional_context(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_k.load.return_value = {"checkpoint": {}}
        mock_k.format_memory.return_value = "# Working Memory"
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_session_start,
            {"cwd": "/tmp/my-project"},
        )

        assert exit_code == 0
        assert output is not None
        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert output["hookSpecificOutput"]["additionalContext"] == "# Working Memory"

    @patch("kernle.Kernle")
    def test_outputs_nothing_when_empty(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_k.load.return_value = {}
        mock_k.format_memory.return_value = ""
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_session_start,
            {"cwd": "/tmp/my-project"},
        )

        assert exit_code == 0
        assert output is None

    def test_exits_zero_on_error(self):
        """Hook exits 0 even with invalid stdin."""
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO("not json")
        sys.stdout = io.StringIO()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cmd_hook_session_start(make_args())
            assert exc_info.value.code == 0
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout

    @patch("kernle.Kernle")
    def test_uses_explicit_stack_flag(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_k.load.return_value = {}
        mock_k.format_memory.return_value = ""
        mock_kernle_cls.return_value = mock_k

        run_hook(
            cmd_hook_session_start,
            {"cwd": "/tmp/other"},
            args=make_args(stack="my-stack"),
        )

        mock_kernle_cls.assert_called_once_with(stack_id="my-stack")

    @patch("kernle.Kernle")
    def test_reads_token_budget_from_env(self, mock_kernle_cls, monkeypatch):
        monkeypatch.setenv("KERNLE_TOKEN_BUDGET", "4000")
        mock_k = MagicMock()
        mock_k.load.return_value = {}
        mock_k.format_memory.return_value = ""
        mock_kernle_cls.return_value = mock_k

        run_hook(cmd_hook_session_start, {"cwd": "/tmp/test"})

        mock_k.load.assert_called_once_with(budget=4000)

    @patch("kernle.Kernle")
    def test_resolves_stack_from_cwd(self, mock_kernle_cls, monkeypatch):
        monkeypatch.delenv("KERNLE_STACK_ID", raising=False)
        mock_k = MagicMock()
        mock_k.load.return_value = {}
        mock_k.format_memory.return_value = ""
        mock_kernle_cls.return_value = mock_k

        run_hook(cmd_hook_session_start, {"cwd": "/Users/test/cool-project"})

        mock_kernle_cls.assert_called_once_with(stack_id="cool-project")


# --- TestHookPreToolUse ---


class TestHookPreToolUse:
    @patch("kernle.Kernle")
    def test_blocks_write_to_memory_dir(self, mock_kernle_cls):
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
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    @patch("kernle.Kernle")
    def test_blocks_write_to_memory_md(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {
                    "file_path": "/home/user/.claude/MEMORY.md",
                    "content": "note",
                },
            },
        )

        assert exit_code == 0
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_allows_write_to_non_memory_path(self):
        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {"cwd": "/tmp/project", "tool_input": {"file_path": "src/main.py", "content": "code"}},
        )

        assert exit_code == 0
        assert output is None

    @patch("kernle.Kernle")
    def test_captures_content_as_raw_entry(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/test.md", "content": "captured"},
            },
        )

        mock_k.raw.assert_called_once()
        call_args = mock_k.raw.call_args
        assert "[memory-capture]" in call_args[0][0]
        assert "captured" in call_args[0][0]

    @patch("kernle.Kernle")
    def test_truncates_large_content(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"file_path": "memory/big.md", "content": "x" * 3000},
            },
        )

        call_args = mock_k.raw.call_args[0][0]
        assert "[truncated]" in call_args

    def test_allows_when_no_file_path(self):
        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {"cwd": "/tmp/project", "tool_input": {"content": "hello"}},
        )
        assert exit_code == 0
        assert output is None

    @patch("kernle.Kernle")
    def test_uses_path_as_fallback(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {
                "cwd": "/tmp/project",
                "tool_input": {"path": "memory/notes.md", "content": "hello"},
            },
        )

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_exits_zero_on_error(self):
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO("not json")
        sys.stdout = io.StringIO()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cmd_hook_pre_tool_use(make_args())
            assert exc_info.value.code == 0
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


# --- TestHookPreCompact ---


class TestHookPreCompact:
    @patch("kernle.Kernle")
    def test_saves_checkpoint_with_transcript(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        transcript = make_transcript(
            [
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "I'll investigate"},
            ]
        )

        run_hook(
            cmd_hook_pre_compact,
            {"cwd": "/tmp/project", "transcript_path": transcript},
        )

        mock_k.checkpoint.assert_called_once()
        call_args = mock_k.checkpoint.call_args
        assert "[pre-compact]" in call_args[0][0]
        assert "Fix the bug" in call_args[0][0]
        os.unlink(transcript)

    @patch("kernle.Kernle")
    def test_handles_missing_transcript(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_pre_compact,
            {"cwd": "/tmp/project", "transcript_path": None},
        )

        assert exit_code == 0
        mock_k.checkpoint.assert_called_once()
        assert "Session ended" in mock_k.checkpoint.call_args[0][0]

    def test_exits_zero_on_error(self):
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO("not json")
        sys.stdout = io.StringIO()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cmd_hook_pre_compact(make_args())
            assert exc_info.value.code == 0
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


# --- TestHookSessionEnd ---


class TestHookSessionEnd:
    @patch("kernle.Kernle")
    def test_saves_checkpoint_and_raw(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        transcript = make_transcript(
            [
                {"role": "user", "content": "Deploy v2"},
                {"role": "assistant", "content": "Done"},
            ]
        )

        run_hook(
            cmd_hook_session_end,
            {"cwd": "/tmp/project", "transcript_path": transcript},
        )

        mock_k.checkpoint.assert_called_once()
        mock_k.raw.assert_called_once()
        assert "Deploy v2" in mock_k.checkpoint.call_args[0][0]
        assert "Session ended" in mock_k.raw.call_args[0][0]
        os.unlink(transcript)

    @patch("kernle.Kernle")
    def test_continues_if_checkpoint_fails(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_k.checkpoint.side_effect = Exception("DB error")
        mock_kernle_cls.return_value = mock_k

        output, exit_code = run_hook(
            cmd_hook_session_end,
            {"cwd": "/tmp/project", "transcript_path": None},
        )

        assert exit_code == 0
        # raw() should still be called even if checkpoint fails
        mock_k.raw.assert_called_once()

    @patch("kernle.Kernle")
    def test_uses_fallback_when_no_transcript(self, mock_kernle_cls):
        mock_k = MagicMock()
        mock_kernle_cls.return_value = mock_k

        run_hook(
            cmd_hook_session_end,
            {"cwd": "/tmp/project", "transcript_path": None},
        )

        assert "Session ended" in mock_k.checkpoint.call_args[0][0]

    def test_exits_zero_on_error(self):
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        sys.stdin = io.StringIO("not json")
        sys.stdout = io.StringIO()
        try:
            with pytest.raises(SystemExit) as exc_info:
                cmd_hook_session_end(make_args())
            assert exc_info.value.code == 0
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


# --- TestValidateHookInput (#717: Payload Validation) ---


class TestValidateHookInput:
    """Validate that hook payloads contain required keys for each hook type."""

    def test_session_start_rejects_missing_cwd(self):
        with pytest.raises(ValueError, match="cwd"):
            _validate_hook_input({}, "SessionStart")

    def test_session_start_accepts_valid_payload(self):
        result = _validate_hook_input({"cwd": "/tmp/project"}, "SessionStart")
        assert result["cwd"] == "/tmp/project"

    def test_pre_tool_use_rejects_missing_tool_input(self):
        with pytest.raises(ValueError, match="tool_input"):
            _validate_hook_input({"cwd": "/tmp/project"}, "PreToolUse")

    def test_pre_tool_use_rejects_missing_cwd(self):
        with pytest.raises(ValueError, match="cwd"):
            _validate_hook_input({"tool_input": {"file_path": "x"}}, "PreToolUse")

    def test_pre_tool_use_with_valid_payload_succeeds(self):
        data = {"cwd": "/tmp/project", "tool_input": {"file_path": "src/main.py"}}
        result = _validate_hook_input(data, "PreToolUse")
        assert result["cwd"] == "/tmp/project"
        assert result["tool_input"]["file_path"] == "src/main.py"

    def test_pre_compact_rejects_missing_cwd(self):
        with pytest.raises(ValueError, match="cwd"):
            _validate_hook_input({}, "PreCompact")

    def test_pre_compact_accepts_valid_payload(self):
        result = _validate_hook_input({"cwd": "/tmp/project"}, "PreCompact")
        assert result["cwd"] == "/tmp/project"

    def test_session_end_rejects_missing_cwd(self):
        with pytest.raises(ValueError, match="cwd"):
            _validate_hook_input({}, "SessionEnd")

    def test_session_end_accepts_valid_payload(self):
        result = _validate_hook_input({"cwd": "/tmp/project"}, "SessionEnd")
        assert result["cwd"] == "/tmp/project"

    def test_unknown_hook_type_passes_through(self):
        """Unknown hook types should not raise -- forward compatibility."""
        result = _validate_hook_input({"anything": True}, "FutureHook")
        assert result == {"anything": True}


class TestPreToolUseValidationFailsDenyClosed:
    """PreToolUse validation failure must produce deny output, not crash (#717)."""

    def test_pre_tool_use_denies_on_missing_cwd(self):
        """Missing cwd triggers validation error, which must still deny."""
        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {"tool_input": {"file_path": "memory/notes.md", "content": "data"}},
        )
        assert exit_code == 0
        assert output is not None
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_pre_tool_use_denies_on_missing_tool_input(self):
        """Missing tool_input triggers validation error, which must still deny."""
        output, exit_code = run_hook(
            cmd_hook_pre_tool_use,
            {"cwd": "/tmp/project"},
        )
        assert exit_code == 0
        assert output is not None
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
