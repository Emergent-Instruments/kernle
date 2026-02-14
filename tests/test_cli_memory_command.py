"""Tests for memory-related CLI commands boundary validation.

Covers the gaps identified by codex audit for v0.13.06:
- cmd_raw with empty content should fail validation
- cmd_note with missing required fields should fail
- cmd_episode with empty objective/outcome should fail
- A list command on an empty stack should return empty output

All tests focus on boundary validation at the CLI command layer,
not core logic. Uses argparse.Namespace for args and mock Kernle
instances where appropriate.
"""

import argparse
from unittest.mock import patch

import pytest

from kernle import Kernle
from kernle.cli.commands.memory import cmd_episode, cmd_note
from kernle.cli.commands.raw import cmd_raw

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-mem-cli", db_path=tmp_path / "mem-cli.db")


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-mem-cli", storage=storage, strict=False)
    yield inst


# ============================================================================
# cmd_raw boundary validation
# ============================================================================


class TestCmdRawBoundary:
    """Boundary validation for the raw capture command."""

    def test_raw_capture_empty_content_prints_error(self, k, capsys):
        """cmd_raw with empty/None content prints an error, does not save."""
        args = argparse.Namespace(
            command="raw",
            raw_action="capture",
            content=None,
            tags="",
            source="cli",
            quiet=False,
            stdin=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "Content is required" in captured

    def test_raw_capture_empty_string_content_prints_error(self, k, capsys):
        """cmd_raw with empty string content prints an error, does not save."""
        args = argparse.Namespace(
            command="raw",
            raw_action="capture",
            content="",
            tags="",
            source="cli",
            quiet=False,
            stdin=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "Content is required" in captured

    def test_raw_capture_stdin_empty_prints_error(self, k, capsys):
        """cmd_raw in stdin mode with empty input prints an error."""
        args = argparse.Namespace(
            command="raw",
            raw_action="capture",
            content=None,
            tags="",
            source="cli",
            quiet=False,
            stdin=True,
        )
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read.return_value = "   \n  "
            cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "No content received" in captured

    def test_raw_capture_quiet_mode_empty_produces_no_output(self, k, capsys):
        """cmd_raw in quiet mode with empty content produces no output."""
        args = argparse.Namespace(
            command="raw",
            raw_action="capture",
            content=None,
            tags="",
            source="cli",
            quiet=True,
            stdin=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert captured == ""

    def test_raw_capture_valid_content_saves(self, k, capsys):
        """cmd_raw with valid content saves successfully."""
        args = argparse.Namespace(
            command="raw",
            raw_action="capture",
            content="A meaningful observation",
            tags="",
            source="cli",
            quiet=False,
            stdin=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "Raw entry captured" in captured


# ============================================================================
# cmd_note boundary validation
# ============================================================================


class TestCmdNoteBoundary:
    """Boundary validation for the note command."""

    def test_note_empty_content_raises(self, k):
        """cmd_note with empty content raises ValueError from validate_input."""
        args = argparse.Namespace(
            command="note",
            content="",
            type="note",
            speaker=None,
            reason=None,
            tag=[],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
        )
        with pytest.raises(ValueError, match="content cannot be empty"):
            cmd_note(args, k)

    def test_note_none_content_raises(self, k):
        """cmd_note with None content raises ValueError (not a string)."""
        args = argparse.Namespace(
            command="note",
            content=None,
            type="note",
            speaker=None,
            reason=None,
            tag=[],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
        )
        with pytest.raises(ValueError, match="must be a string"):
            cmd_note(args, k)

    def test_note_whitespace_only_content_raises(self, k):
        """cmd_note with whitespace-only content raises ValueError."""
        args = argparse.Namespace(
            command="note",
            content="   \t\n   ",
            type="note",
            speaker=None,
            reason=None,
            tag=[],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
        )
        with pytest.raises(ValueError, match="content cannot be empty"):
            cmd_note(args, k)

    def test_note_valid_content_saves(self, k, capsys):
        """cmd_note with valid content saves successfully."""
        args = argparse.Namespace(
            command="note",
            content="An important observation about the system",
            type="note",
            speaker=None,
            reason=None,
            tag=[],
            protect=False,
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
        )
        cmd_note(args, k)

        captured = capsys.readouterr().out
        assert "Note saved" in captured


# ============================================================================
# cmd_episode boundary validation
# ============================================================================


class TestCmdEpisodeBoundary:
    """Boundary validation for the episode command."""

    def test_episode_empty_objective_raises(self, k):
        """cmd_episode with empty objective raises ValueError."""
        args = argparse.Namespace(
            command="episode",
            objective="",
            outcome="something happened",
            lesson=[],
            tag=[],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            emotion=None,
            valence=None,
            arousal=None,
            auto_emotion=False,
        )
        with pytest.raises(ValueError, match="objective cannot be empty"):
            cmd_episode(args, k)

    def test_episode_empty_outcome_raises(self, k):
        """cmd_episode with empty outcome raises ValueError."""
        args = argparse.Namespace(
            command="episode",
            objective="tried something",
            outcome="",
            lesson=[],
            tag=[],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            emotion=None,
            valence=None,
            arousal=None,
            auto_emotion=False,
        )
        with pytest.raises(ValueError, match="outcome cannot be empty"):
            cmd_episode(args, k)

    def test_episode_none_objective_raises(self, k):
        """cmd_episode with None objective raises ValueError."""
        args = argparse.Namespace(
            command="episode",
            objective=None,
            outcome="something happened",
            lesson=[],
            tag=[],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            emotion=None,
            valence=None,
            arousal=None,
            auto_emotion=False,
        )
        with pytest.raises(ValueError, match="must be a string"):
            cmd_episode(args, k)

    def test_episode_whitespace_only_objective_raises(self, k):
        """cmd_episode with whitespace-only objective raises ValueError."""
        args = argparse.Namespace(
            command="episode",
            objective="   \n   ",
            outcome="something happened",
            lesson=[],
            tag=[],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            emotion=None,
            valence=None,
            arousal=None,
            auto_emotion=False,
        )
        with pytest.raises(ValueError, match="objective cannot be empty"):
            cmd_episode(args, k)

    def test_episode_valid_inputs_saves(self, k, capsys):
        """cmd_episode with valid objective and outcome saves successfully."""
        args = argparse.Namespace(
            command="episode",
            objective="Deployed the new feature",
            outcome="Users reported fewer errors",
            lesson=["Always test in staging first"],
            tag=[],
            derived_from=None,
            source=None,
            context=None,
            context_tag=None,
            emotion=None,
            valence=None,
            arousal=None,
            auto_emotion=False,
        )
        cmd_episode(args, k)

        captured = capsys.readouterr().out
        assert "Episode saved" in captured


# ============================================================================
# List commands on empty stack
# ============================================================================


class TestListOnEmptyStack:
    """List commands on an empty stack should return empty output cleanly."""

    def test_raw_list_empty_stack(self, k, capsys):
        """raw list on empty stack prints 'No raw entries found'."""
        args = argparse.Namespace(
            command="raw",
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=50,
            json=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "No raw entries found" in captured

    def test_raw_list_empty_stack_json(self, k, capsys):
        """raw list --json on empty stack prints empty JSON array."""
        args = argparse.Namespace(
            command="raw",
            raw_action="list",
            unprocessed=False,
            processed=False,
            limit=50,
            json=True,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        # Empty stack should print "No raw entries found" without JSON
        # (the JSON branch only triggers when entries exist)
        assert "No raw entries found" in captured

    def test_raw_review_empty_stack(self, k, capsys):
        """raw review on empty stack prints 'no unprocessed' message."""
        args = argparse.Namespace(
            command="raw",
            raw_action="review",
            limit=10,
            json=False,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "No unprocessed raw entries" in captured

    def test_raw_triage_empty_stack(self, k, capsys):
        """raw triage on empty stack prints 'no unprocessed' message."""
        args = argparse.Namespace(
            command="raw",
            raw_action="triage",
            limit=10,
        )
        cmd_raw(args, k)

        captured = capsys.readouterr().out
        assert "No unprocessed raw entries" in captured
