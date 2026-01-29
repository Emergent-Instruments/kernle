"""
Tests for the Kernle init and doctor CLI commands.
"""

import argparse
import json
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kernle.cli.commands.doctor import (
    ComplianceCheck,
    check_checkpoint_instruction,
    check_kernle_anxiety,
    check_kernle_load,
    check_memory_section,
    check_per_message_health,
    cmd_doctor,
    find_instruction_file,
    run_all_checks,
)
from kernle.cli.commands.init import (
    cmd_init,
    detect_instruction_file,
    generate_section,
    has_kernle_section,
)


class TestGenerateSection:
    """Test section generation."""

    def test_generate_standard_section(self):
        """Test generating standard section."""
        section = generate_section("claire", style="standard", include_per_message=True)

        assert "## Memory (Kernle)" in section
        assert "kernle -a claire load" in section
        assert "kernle -a claire anxiety" in section
        assert "## Memory Health (Every Message)" in section
        assert "kernle -a claire anxiety -b" in section

    def test_generate_minimal_section(self):
        """Test generating minimal section."""
        section = generate_section("test-agent", style="minimal", include_per_message=True)

        assert "## Kernle" in section
        assert "kernle -a test-agent load" in section
        assert "kernle -a test-agent anxiety -b" in section
        assert "## Memory Health (Every Message)" in section

    def test_generate_combined_section(self):
        """Test generating combined section."""
        section = generate_section("myagent", style="combined", include_per_message=True)

        assert "## Memory (Kernle)" in section
        assert "kernle -a myagent load" in section
        assert "Every Message" in section

    def test_generate_without_per_message(self):
        """Test generating section without per-message health check."""
        section = generate_section("agent1", style="standard", include_per_message=False)

        assert "## Memory (Kernle)" in section
        assert "kernle -a agent1 load" in section
        # Should NOT have per-message section
        assert "## Memory Health (Every Message)" not in section


class TestHasKernleSection:
    """Test Kernle section detection."""

    def test_detect_memory_kernle_header(self):
        """Test detecting ## Memory (Kernle) header."""
        content = """# My Project

## Memory (Kernle)

Some instructions here.
"""
        assert has_kernle_section(content) is True

    def test_detect_kernle_header(self):
        """Test detecting ## Kernle header."""
        content = """# AGENTS.md

## Kernle

At session start...
"""
        assert has_kernle_section(content) is True

    def test_detect_kernle_load_command(self):
        """Test detecting kernle load command."""
        content = """# Instructions

Run `kernle -a myagent load` at startup.
"""
        assert has_kernle_section(content) is True

    def test_detect_kernle_anxiety_command(self):
        """Test detecting kernle anxiety command."""
        content = """# Boot Sequence

Check health with kernle anxiety -b
"""
        assert has_kernle_section(content) is True

    def test_no_kernle_content(self):
        """Test file with no Kernle content."""
        content = """# My Project

## Setup

Some other instructions.
"""
        assert has_kernle_section(content) is False


class TestDetectInstructionFile:
    """Test instruction file detection."""

    def test_detect_claude_md(self, tmp_path, monkeypatch):
        """Test detecting CLAUDE.md."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Instructions")

        result = detect_instruction_file()
        assert result == Path("CLAUDE.md")

    def test_detect_agents_md(self, tmp_path, monkeypatch):
        """Test detecting AGENTS.md."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# Agents")

        result = detect_instruction_file()
        assert result == Path("AGENTS.md")

    def test_no_instruction_file(self, tmp_path, monkeypatch):
        """Test when no instruction file exists."""
        monkeypatch.chdir(tmp_path)

        result = detect_instruction_file()
        assert result is None


class TestCmdInit:
    """Test the init command."""

    @pytest.fixture
    def mock_kernle(self):
        """Mock Kernle instance."""
        k = Mock()
        k.agent_id = "test-agent"
        return k

    def test_init_print_only(self, mock_kernle, tmp_path, monkeypatch):
        """Test init with --print flag."""
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            style="standard",
            output=None,
            print=True,
            force=False,
            no_per_message=False,
            non_interactive=False,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_init(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Kernle Instructions for CLAUDE.md" in output
        assert "kernle -a test-agent load" in output

    def test_init_creates_file(self, mock_kernle, tmp_path, monkeypatch):
        """Test init creates CLAUDE.md when no file exists."""
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(
            style="standard",
            output=None,
            print=False,
            force=False,
            no_per_message=False,
            non_interactive=True,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_init(args, mock_kernle)

        assert (tmp_path / "CLAUDE.md").exists()
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "kernle -a test-agent load" in content

    def test_init_appends_to_existing(self, mock_kernle, tmp_path, monkeypatch):
        """Test init appends to existing file."""
        monkeypatch.chdir(tmp_path)

        # Create existing file
        existing = "# My Project\n\nSome existing content."
        (tmp_path / "CLAUDE.md").write_text(existing)

        args = argparse.Namespace(
            style="standard",
            output=None,
            print=False,
            force=False,
            no_per_message=False,
            non_interactive=True,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_init(args, mock_kernle)

        content = (tmp_path / "CLAUDE.md").read_text()
        assert "My Project" in content
        assert "Some existing content" in content
        assert "kernle -a test-agent load" in content

    def test_init_detects_existing_section(self, mock_kernle, tmp_path, monkeypatch):
        """Test init detects existing Kernle section."""
        monkeypatch.chdir(tmp_path)

        # Create file with Kernle section
        existing = "# My Project\n\n## Memory (Kernle)\n\nkernle load instructions"
        (tmp_path / "CLAUDE.md").write_text(existing)

        args = argparse.Namespace(
            style="standard",
            output=None,
            print=False,
            force=False,
            no_per_message=False,
            non_interactive=True,
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_init(args, mock_kernle)

        output = fake_out.getvalue()
        assert "already contains Kernle instructions" in output

    def test_init_force_overwrites(self, mock_kernle, tmp_path, monkeypatch):
        """Test init with --force appends even if section exists."""
        monkeypatch.chdir(tmp_path)

        existing = "# My Project\n\n## Memory (Kernle)\n\nOld instructions"
        (tmp_path / "CLAUDE.md").write_text(existing)

        args = argparse.Namespace(
            style="standard",
            output=None,
            print=False,
            force=True,
            no_per_message=False,
            non_interactive=True,
        )

        with patch("sys.stdout", new=StringIO()):
            cmd_init(args, mock_kernle)

        content = (tmp_path / "CLAUDE.md").read_text()
        # Should have both old and new
        assert "Old instructions" in content
        assert "kernle -a test-agent load" in content


class TestComplianceChecks:
    """Test individual compliance checks."""

    def test_check_kernle_load_found(self):
        """Test detecting load instruction."""
        content = "Run `kernle -a myagent load` at session start."
        result = check_kernle_load(content, "myagent")

        assert result.passed is True
        assert "✓" in result.message

    def test_check_kernle_load_missing(self):
        """Test missing load instruction."""
        content = "Some other content without load."
        result = check_kernle_load(content, "myagent")

        assert result.passed is False
        assert "✗" in result.message
        assert result.fix is not None

    def test_check_anxiety_found(self):
        """Test detecting anxiety instruction."""
        content = "Check health: kernle -a agent1 anxiety"
        result = check_kernle_anxiety(content, "agent1")

        assert result.passed is True

    def test_check_anxiety_missing(self):
        """Test missing anxiety instruction."""
        content = "Just some random content without the required commands."
        result = check_kernle_anxiety(content, "agent1")

        assert result.passed is False

    def test_check_per_message_found(self):
        """Test detecting per-message health check."""
        content = """
## Every Message

Run kernle anxiety -b before processing.
"""
        result = check_per_message_health(content, "agent1")

        assert result.passed is True

    def test_check_per_message_missing(self):
        """Test missing per-message health check."""
        content = "Only session start instructions here."
        result = check_per_message_health(content, "agent1")

        assert result.passed is False

    def test_check_checkpoint_found(self):
        """Test detecting checkpoint instruction."""
        content = "Before ending: kernle checkpoint save 'state'"
        result = check_checkpoint_instruction(content, "agent1")

        assert result.passed is True

    def test_check_memory_section_found(self):
        """Test detecting memory section."""
        content = "## Memory (Kernle)\n\nInstructions here."
        result = check_memory_section(content)

        assert result.passed is True


class TestCmdDoctor:
    """Test the doctor command."""

    @pytest.fixture
    def mock_kernle(self):
        """Mock Kernle instance."""
        k = Mock()
        k.agent_id = "test-agent"
        return k

    def test_doctor_no_file(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor when no instruction file exists."""
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(json=False, verbose=False, fix=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor(args, mock_kernle)

        output = fake_out.getvalue()
        assert "No instruction file found" in output

    def test_doctor_json_no_file(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor JSON output when no file."""
        monkeypatch.chdir(tmp_path)

        args = argparse.Namespace(json=True, verbose=False, fix=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor(args, mock_kernle)

        output = json.loads(fake_out.getvalue())
        assert output["status"] == "no_file"

    def test_doctor_excellent_compliance(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor with excellent compliance."""
        monkeypatch.chdir(tmp_path)

        # Create file with all requirements
        content = """# Instructions

## Memory (Kernle)

## Every Session

1. Run `kernle -a test-agent load`
2. Run `kernle -a test-agent anxiety`

## Every Message

Run `kernle anxiety -b` before processing any request.

## Before Session Ends

Run `kernle checkpoint save "state"`
"""
        (tmp_path / "CLAUDE.md").write_text(content)

        args = argparse.Namespace(json=False, verbose=False, fix=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Excellent" in output or "All checks passed" in output

    def test_doctor_needs_work(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor with missing required instructions."""
        monkeypatch.chdir(tmp_path)

        # Create file without Kernle instructions
        content = """# My Project

## Setup

Some setup instructions.
"""
        (tmp_path / "CLAUDE.md").write_text(content)

        args = argparse.Namespace(json=False, verbose=False, fix=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor(args, mock_kernle)

        output = fake_out.getvalue()
        assert "Needs work" in output or "missing" in output.lower()

    def test_doctor_json_output(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor JSON output with checks."""
        monkeypatch.chdir(tmp_path)

        content = "# Instructions\n\nkernle -a test-agent load"
        (tmp_path / "CLAUDE.md").write_text(content)

        args = argparse.Namespace(json=True, verbose=False, fix=False)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            cmd_doctor(args, mock_kernle)

        output = json.loads(fake_out.getvalue())
        assert "status" in output
        assert "checks" in output
        assert "file" in output

    def test_doctor_fix_mode(self, mock_kernle, tmp_path, monkeypatch):
        """Test doctor --fix auto-fixes missing instructions."""
        monkeypatch.chdir(tmp_path)

        # Create file without Kernle instructions
        content = "# My Project\n\nSome content."
        (tmp_path / "CLAUDE.md").write_text(content)

        args = argparse.Namespace(json=False, verbose=False, fix=True)

        with patch("sys.stdout", new=StringIO()):
            cmd_doctor(args, mock_kernle)

        # File should now have Kernle instructions
        new_content = (tmp_path / "CLAUDE.md").read_text()
        assert "kernle" in new_content.lower()


class TestRunAllChecks:
    """Test running all compliance checks."""

    def test_all_checks_pass(self):
        """Test when all checks pass."""
        content = """
## Memory (Kernle)

At session start:
1. kernle -a myagent load
2. kernle -a myagent anxiety

Every message: kernle anxiety -b

Before ending: kernle checkpoint save
"""
        checks = run_all_checks(content, "myagent")

        passed = sum(1 for c in checks if c.passed)
        assert passed == 5  # All 5 checks should pass

    def test_partial_compliance(self):
        """Test with partial compliance."""
        content = """
## Memory

kernle -a myagent load
"""
        checks = run_all_checks(content, "myagent")

        # Should have load, maybe memory section, but not all
        passed = sum(1 for c in checks if c.passed)
        assert passed >= 2  # At least load and memory section
        assert passed < 5  # Not all checks


class TestComplianceCheckDataClass:
    """Test ComplianceCheck data class."""

    def test_to_dict(self):
        """Test serialization to dict."""
        check = ComplianceCheck(
            name="test_check", passed=True, message="Test message", fix="Run this to fix"
        )

        result = check.to_dict()

        assert result["name"] == "test_check"
        assert result["passed"] is True
        assert result["message"] == "Test message"
        assert result["fix"] == "Run this to fix"

    def test_to_dict_no_fix(self):
        """Test serialization when no fix provided."""
        check = ComplianceCheck(name="passing_check", passed=True, message="All good")

        result = check.to_dict()

        assert result["fix"] is None


class TestFindInstructionFile:
    """Test finding instruction files in various locations."""

    def test_find_claude_md(self, tmp_path, monkeypatch):
        """Test finding CLAUDE.md in current directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Test")

        result = find_instruction_file()

        assert result is not None
        assert result[0] == Path("CLAUDE.md")
        assert result[1] == "claude"

    def test_find_agents_md(self, tmp_path, monkeypatch):
        """Test finding AGENTS.md in current directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "AGENTS.md").write_text("# Test")

        result = find_instruction_file()

        assert result is not None
        assert result[0] == Path("AGENTS.md")
        assert result[1] == "agents"

    def test_find_cursorrules(self, tmp_path, monkeypatch):
        """Test finding .cursorrules in current directory."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".cursorrules").write_text("# Test")

        result = find_instruction_file()

        assert result is not None
        assert result[0] == Path(".cursorrules")
        assert result[1] == "cursor"

    def test_find_priority_order(self, tmp_path, monkeypatch):
        """Test that CLAUDE.md is found before AGENTS.md."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Claude")
        (tmp_path / "AGENTS.md").write_text("# Agents")

        result = find_instruction_file()

        assert result is not None
        assert result[0] == Path("CLAUDE.md")  # CLAUDE.md first

    def test_find_none(self, tmp_path, monkeypatch):
        """Test when no instruction file exists."""
        monkeypatch.chdir(tmp_path)

        result = find_instruction_file()

        assert result is None
