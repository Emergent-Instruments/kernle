"""Shared fixtures for Claude Code plugin tests."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture
def mock_subprocess():
    """Patch subprocess.run and return the mock."""
    with patch("_bridge.subprocess.run") as mock:
        yield mock


@pytest.fixture
def make_hook_input():
    """Factory for creating hook stdin JSON."""

    def _make(**kwargs):
        base = {
            "session_id": "test-session",
            "transcript_path": "/tmp/test-transcript.jsonl",
            "cwd": "/Users/test/my-project",
            "hook_event_name": "SessionStart",
        }
        base.update(kwargs)
        return base

    return _make


@pytest.fixture
def make_transcript(tmp_path):
    """Factory for creating temporary JSONL transcript files."""

    def _make(messages: list[dict]) -> str:
        path = tmp_path / "transcript.jsonl"
        lines = [json.dumps(msg) for msg in messages]
        path.write_text("\n".join(lines))
        return str(path)

    return _make


@pytest.fixture
def capture_hook_output():
    """Run a hook script's main() with mocked stdin/stdout.

    Returns (stdout_content, exit_code_or_none).
    """

    def _capture(main_func, hook_input: dict):
        stdin = StringIO(json.dumps(hook_input))
        stdout = StringIO()
        exit_code = None

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            try:
                main_func()
            except SystemExit as e:
                exit_code = e.code

        stdout_content = stdout.getvalue()
        try:
            return json.loads(stdout_content) if stdout_content else None, exit_code
        except json.JSONDecodeError:
            return stdout_content, exit_code

    return _capture
