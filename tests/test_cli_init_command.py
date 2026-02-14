"""Tests for kernle.cli.commands.init — CLI boundary hardening.

Exercises negative-path scenarios for cmd_init: invalid stack IDs,
idempotent re-runs, and write-failure handling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kernle.cli.commands.init import (
    cmd_init,
    detect_instruction_file,
    generate_section,
    has_kernle_section,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Build a minimal args namespace for cmd_init.

    Returns a SimpleNamespace with sensible defaults that can be overridden
    by the caller via keyword arguments.
    """
    defaults = {
        "style": "standard",
        "no_per_message": False,
        "output": None,
        "force": False,
        "print": False,
        "non_interactive": True,  # avoid stdin prompts in tests
        "seed_values": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_kernle(stack_id="test-stack"):
    """Create a mock Kernle instance with a given stack_id.

    The mock includes a storage attribute with a get_values method
    and a checkpoint method, both needed by cmd_init.
    """
    k = MagicMock()
    k.stack_id = stack_id

    # storage.get_values is called by _snapshot_values
    k.storage.get_values.return_value = []

    # checkpoint() is called to save an initial checkpoint
    k.checkpoint.return_value = {
        "current_task": "Kernle initialized",
        "pending": ["Configure instruction file", "Test memory persistence"],
    }

    # seed_trust() is called to seed trust layer
    k.seed_trust.return_value = 0

    return k


# ===========================================================================
# Test: Invalid stack IDs should be caught by the Kernle constructor,
#       but cmd_init itself uses k.stack_id — so we verify that the
#       stack_id validation lives upstream in ValidationMixin.
# ===========================================================================


class TestStackIdValidation:
    """Verify that ValidationMixin._validate_stack_id rejects dangerous inputs."""

    def test_stack_id_with_forward_slash_raises(self):
        """Stack IDs containing '/' are rejected as path traversal."""
        from kernle.core.validation import ValidationMixin

        mixin = ValidationMixin()
        with pytest.raises(ValueError, match="path separators"):
            mixin._validate_stack_id("../../etc/passwd")

    def test_stack_id_with_backslash_raises(self):
        """Stack IDs containing '\\' are rejected as path traversal."""
        from kernle.core.validation import ValidationMixin

        mixin = ValidationMixin()
        with pytest.raises(ValueError, match="path separators"):
            mixin._validate_stack_id("foo\\bar")

    def test_stack_id_empty_raises(self):
        """Empty stack IDs are rejected."""
        from kernle.core.validation import ValidationMixin

        mixin = ValidationMixin()
        with pytest.raises(ValueError, match="cannot be empty"):
            mixin._validate_stack_id("")

    def test_stack_id_dot_dot_raises(self):
        """The '..' relative path component is rejected."""
        from kernle.core.validation import ValidationMixin

        mixin = ValidationMixin()
        with pytest.raises(ValueError, match="relative path component"):
            mixin._validate_stack_id("..")

    def test_stack_id_valid_passes(self):
        """A normal alphanumeric stack ID with dashes is accepted."""
        from kernle.core.validation import ValidationMixin

        mixin = ValidationMixin()
        result = mixin._validate_stack_id("my-agent-01")
        assert result == "my-agent-01"


# ===========================================================================
# Test: Idempotent init — running init twice should not error
# ===========================================================================


class TestInitIdempotency:
    """cmd_init should handle already-initialized instruction files gracefully."""

    def test_init_already_present_returns_not_success(self, tmp_path, capsys):
        """When the target file already has Kernle instructions, init
        returns success=False with status='already_present' (unless --force).
        """
        # Pre-populate a CLAUDE.md with Kernle instructions
        target_file = tmp_path / "CLAUDE.md"
        target_file.write_text("## Memory (Kernle)\nkernle -s test load\n")

        k = _make_kernle()
        args = _make_args(output=str(target_file))

        result = cmd_init(args, k)

        # Should indicate already present, not an error
        assert result["success"] is False
        assert result["status"] == "already_present"
        assert "already_present" in result["status"]

    def test_init_force_overwrites_existing(self, tmp_path, capsys):
        """With --force, init appends to a file that already has Kernle content."""
        target_file = tmp_path / "CLAUDE.md"
        original_content = "# Existing\n\n## Memory (Kernle)\nold content\n"
        target_file.write_text(original_content)

        k = _make_kernle()
        args = _make_args(output=str(target_file), force=True)

        result = cmd_init(args, k)

        # Force should succeed
        assert result["success"] is True
        assert result["action"] == "append"

        # The original content should be preserved
        final_content = target_file.read_text()
        assert "# Existing" in final_content

    def test_init_creates_new_file_then_detects_existing(self, tmp_path, capsys, monkeypatch):
        """First init creates the file; second init detects it as already present."""
        # Change to tmp_path so detect_instruction_file works
        monkeypatch.chdir(tmp_path)

        target_file = tmp_path / "CLAUDE.md"
        k = _make_kernle()

        # First init: creates the file
        args_first = _make_args(output=str(target_file))
        result_first = cmd_init(args_first, k)
        assert result_first["success"] is True
        assert result_first["action"] == "create"

        # Second init: detects existing Kernle section
        args_second = _make_args(output=str(target_file))
        result_second = cmd_init(args_second, k)
        assert result_second["success"] is False
        assert result_second["status"] == "already_present"


# ===========================================================================
# Test: Print-only mode
# ===========================================================================


class TestInitPrintMode:
    """The --print flag should output the section without writing any file."""

    def test_print_mode_outputs_section(self, capsys):
        """Print mode writes the Kernle section to stdout."""
        k = _make_kernle(stack_id="my-stack")
        args = _make_args(**{"print": True})

        result = cmd_init(args, k)

        output = capsys.readouterr().out
        assert "my-stack" in output
        assert result["status"] == "printed"
        assert result["success"] is True


# ===========================================================================
# Test: Write failure handling
# ===========================================================================


class TestInitWriteFailure:
    """cmd_init should handle write failures gracefully."""

    def test_write_to_readonly_dir_returns_failure(self, tmp_path, capsys):
        """If the target file cannot be written, cmd_init returns a failure result."""
        # Point to a path inside a non-existent directory
        bad_path = tmp_path / "nonexistent_dir" / "subdir" / "CLAUDE.md"

        k = _make_kernle()
        args = _make_args(output=str(bad_path))

        result = cmd_init(args, k)

        assert result["success"] is False
        assert result["status"] == "write_failed"


# ===========================================================================
# Test: Helper functions
# ===========================================================================


class TestDetectInstructionFile:
    """detect_instruction_file should find existing instruction files."""

    def test_detects_claude_md(self, tmp_path, monkeypatch):
        """Finds CLAUDE.md in the current directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Instructions")
        result = detect_instruction_file()
        assert result is not None
        assert result.name == "CLAUDE.md"

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        """Returns None when no instruction file is present."""
        monkeypatch.chdir(tmp_path)
        result = detect_instruction_file()
        assert result is None


class TestHasKernleSection:
    """has_kernle_section should detect Kernle content in markdown."""

    def test_detects_memory_header(self):
        assert has_kernle_section("## Memory (Kernle)\nSome content")

    def test_detects_kernle_header(self):
        assert has_kernle_section("## Kernle\nSome content")

    def test_detects_load_command(self):
        assert has_kernle_section("kernle -s my-stack load")

    def test_returns_false_for_unrelated(self):
        assert not has_kernle_section("# My Project\nNothing about kernle here")


class TestGenerateSection:
    """generate_section should produce valid sections for each style."""

    def test_standard_style_includes_stack_id(self):
        section = generate_section("test-agent", style="standard")
        assert "test-agent" in section
        assert "kernle -s test-agent load" in section

    def test_minimal_style(self):
        section = generate_section("test-agent", style="minimal")
        assert "test-agent" in section
        # Minimal style is shorter
        assert len(section) < len(generate_section("test-agent", style="standard"))

    def test_combined_style(self):
        section = generate_section("test-agent", style="combined")
        assert "test-agent" in section
        assert "Every Session" in section

    def test_no_per_message_omits_health_check(self):
        with_pm = generate_section("agent", style="standard", include_per_message=True)
        without_pm = generate_section("agent", style="standard", include_per_message=False)
        assert "Every Message" in with_pm
        assert "Every Message" not in without_pm
