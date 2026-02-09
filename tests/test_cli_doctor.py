"""Tests for kernle doctor CLI commands — hook checks, platform detection,
cmd_doctor, cmd_doctor_structural, structural check functions, and _generate_summary."""

import json
import uuid
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from kernle.cli.commands.doctor import (
    ComplianceCheck,
    StructuralFinding,
    _generate_summary,
    check_belief_contradictions,
    check_claude_code_hook,
    check_hooks,
    check_low_confidence_beliefs,
    check_orphaned_references,
    check_stale_goals,
    check_stale_relationships,
    cmd_doctor,
    cmd_doctor_structural,
    detect_platform,
)
from kernle.types import Belief, Goal, Relationship

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    defaults = {"json": False, "fix": False, "full": False, "save_note": False}
    defaults.update(kwargs)
    return Namespace(**defaults)


def _make_belief(
    belief_id=None,
    statement="test belief",
    confidence=0.8,
    last_verified=None,
):
    return Belief(
        id=belief_id or str(uuid.uuid4()),
        stack_id="test",
        statement=statement,
        confidence=confidence,
        last_verified=last_verified,
    )


def _make_relationship(
    rel_id=None,
    entity_name="Alice",
    interaction_count=5,
    last_interaction=None,
):
    return Relationship(
        id=rel_id or str(uuid.uuid4()),
        stack_id="test",
        entity_name=entity_name,
        entity_type="person",
        relationship_type="colleague",
        interaction_count=interaction_count,
        last_interaction=last_interaction,
    )


def _make_goal(goal_id=None, title="Test goal", created_at=None):
    return Goal(
        id=goal_id or str(uuid.uuid4()),
        stack_id="test",
        title=title,
        created_at=created_at,
    )


def _mock_kernle():
    k = MagicMock()
    k.stack_id = "test-stack"
    return k


# ===================================================================
# detect_platform
# ===================================================================


class TestDetectPlatform:
    def test_openclaw_detected(self, tmp_path, monkeypatch):
        openclaw_dir = tmp_path / ".openclaw"
        openclaw_dir.mkdir()
        (openclaw_dir / "openclaw.json").write_text("{}")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        assert detect_platform() == "openclaw"

    def test_claude_code_global_detected(self, tmp_path, monkeypatch):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        assert detect_platform() == "claude-code"

    def test_claude_code_local_detected(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        claude_dir = project / ".claude"
        claude_dir.mkdir()
        (claude_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: project))

        assert detect_platform() == "claude-code"

    def test_unknown_when_neither_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        assert detect_platform() == "unknown"


# ===================================================================
# check_claude_code_hook
# ===================================================================


class TestCheckClaudeCodeHook:
    def _write_settings(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def test_all_4_hooks_configured_passes(self, tmp_path, monkeypatch):
        settings = {
            "hooks": {
                "SessionStart": ["kernle hook session-start"],
                "PreToolUse": ["kernle hook pre-tool-use"],
                "PreCompact": ["kernle hook pre-compact"],
                "SessionEnd": ["kernle hook session-end"],
            }
        }
        self._write_settings(tmp_path / ".claude" / "settings.json", settings)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        result = check_claude_code_hook("test-stack")
        assert result.passed is True
        assert "4/4" in result.message
        assert "global" in result.message

    def test_partial_hooks_fails_with_missing(self, tmp_path, monkeypatch):
        settings = {
            "hooks": {
                "SessionStart": ["kernle hook session-start"],
            }
        }
        self._write_settings(tmp_path / ".claude" / "settings.json", settings)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        result = check_claude_code_hook("test-stack")
        assert result.passed is False
        assert "1/4" in result.message
        assert result.fix is not None
        # Check that the missing events are listed
        assert "PreCompact" in result.fix or "PreToolUse" in result.fix

    def test_no_settings_file_not_configured(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        result = check_claude_code_hook("test-stack")
        assert result.passed is False
        assert "not configured" in result.message

    def test_global_config_takes_precedence(self, tmp_path, monkeypatch):
        """Global config with all hooks wins even if no local config."""
        global_settings = {
            "hooks": {
                "SessionStart": ["kernle hook session-start"],
                "PreToolUse": ["kernle hook pre-tool-use"],
                "PreCompact": ["kernle hook pre-compact"],
                "SessionEnd": ["kernle hook session-end"],
            }
        }
        self._write_settings(tmp_path / ".claude" / "settings.json", global_settings)

        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: project))

        result = check_claude_code_hook("test-stack")
        assert result.passed is True
        assert "global" in result.message

    def test_local_config_used_when_global_absent(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        local_settings = {
            "hooks": {
                "SessionStart": ["kernle hook session-start"],
                "PreToolUse": ["kernle hook pre-tool-use"],
                "PreCompact": ["kernle hook pre-compact"],
                "SessionEnd": ["kernle hook session-end"],
            }
        }
        self._write_settings(project / ".claude" / "settings.json", local_settings)

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: project))

        result = check_claude_code_hook("test-stack")
        assert result.passed is True
        assert "project" in result.message

    def test_invalid_json_in_settings(self, tmp_path, monkeypatch):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text("{not valid json!!!")

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        result = check_claude_code_hook("test-stack")
        assert result.passed is False
        assert "not configured" in result.message

    def test_hooks_key_missing_in_settings(self, tmp_path, monkeypatch):
        self._write_settings(tmp_path / ".claude" / "settings.json", {"other": "data"})

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path / "project"))

        result = check_claude_code_hook("test-stack")
        assert result.passed is False
        assert "not configured" in result.message


# ===================================================================
# check_hooks
# ===================================================================


class TestCheckHooks:
    def test_claude_code_platform_delegates(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.detect_platform", lambda: "claude-code")
        mock_result = ComplianceCheck("claude_code_hook", True, "ok")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_claude_code_hook",
            lambda sid: mock_result,
        )

        checks = check_hooks("test-stack")
        assert len(checks) == 1
        assert checks[0].name == "claude_code_hook"

    def test_openclaw_platform_delegates(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.detect_platform", lambda: "openclaw")
        mock_result = ComplianceCheck("openclaw_hook", True, "ok")
        monkeypatch.setattr("kernle.cli.commands.doctor.check_openclaw_hook", lambda: mock_result)

        checks = check_hooks("test-stack")
        assert len(checks) == 1
        assert checks[0].name == "openclaw_hook"

    def test_unknown_platform_both_fail(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.detect_platform", lambda: "unknown")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_openclaw_hook",
            lambda: ComplianceCheck("openclaw_hook", False, "not installed"),
        )
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_claude_code_hook",
            lambda sid: ComplianceCheck("claude_code_hook", False, "not configured"),
        )

        checks = check_hooks("test-stack")
        assert len(checks) == 1
        assert checks[0].name == "hooks"
        assert checks[0].passed is False
        assert "No platform hooks" in checks[0].message

    def test_unknown_platform_openclaw_passes(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.detect_platform", lambda: "unknown")
        openclaw_ok = ComplianceCheck("openclaw_hook", True, "installed")
        monkeypatch.setattr("kernle.cli.commands.doctor.check_openclaw_hook", lambda: openclaw_ok)
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_claude_code_hook",
            lambda sid: ComplianceCheck("claude_code_hook", False, "not configured"),
        )

        checks = check_hooks("test-stack")
        assert len(checks) == 1
        assert checks[0].name == "openclaw_hook"
        assert checks[0].passed is True

    def test_unknown_platform_claude_passes(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.detect_platform", lambda: "unknown")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_openclaw_hook",
            lambda: ComplianceCheck("openclaw_hook", False, "not installed"),
        )
        claude_ok = ComplianceCheck("claude_code_hook", True, "configured")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_claude_code_hook",
            lambda sid: claude_ok,
        )

        checks = check_hooks("test-stack")
        assert len(checks) == 1
        assert checks[0].name == "claude_code_hook"
        assert checks[0].passed is True


# ===================================================================
# cmd_doctor
# ===================================================================


class TestCmdDoctor:
    def test_no_instruction_file_status(self, capsys, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.find_instruction_file", lambda: None)
        k = _mock_kernle()
        args = _make_args()

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        assert "No instruction file found" in captured.out

    def test_json_output_mode(self, capsys, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.find_instruction_file", lambda: None)
        k = _mock_kernle()
        args = _make_args(json=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "no_file"
        assert data["file"] is None
        assert data["stack_id"] == "test-stack"
        assert "checks" in data

    def test_json_output_with_instruction_file(self, capsys, monkeypatch, tmp_path):
        instruction_file = tmp_path / "CLAUDE.md"
        instruction_file.write_text(
            "## Memory\nkernle load\nkernle anxiety\nevery message\nkernle checkpoint"
        )
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.find_instruction_file",
            lambda: (instruction_file, "claude"),
        )
        k = _mock_kernle()
        args = _make_args(json=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "excellent"
        assert data["file_type"] == "claude"

    def test_full_check_includes_beliefs_and_hooks(self, capsys, monkeypatch, tmp_path):
        instruction_file = tmp_path / "CLAUDE.md"
        instruction_file.write_text(
            "## Memory\nkernle load\nkernle anxiety\nevery message\nkernle checkpoint"
        )
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.find_instruction_file",
            lambda: (instruction_file, "claude"),
        )
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_hooks",
            lambda sid: [ComplianceCheck("claude_code_hook", True, "ok")],
        )
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.check_seed_beliefs",
            lambda kobj: (
                ComplianceCheck("seed_beliefs", True, "ok", category="recommended"),
                {"total_beliefs": 10},
            ),
        )
        k = _mock_kernle()
        args = _make_args(full=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        assert "SEED BELIEFS" in captured.out
        assert "PLATFORM HOOKS" in captured.out

    def test_fix_mode_with_instruction_file(self, capsys, monkeypatch, tmp_path):
        instruction_file = tmp_path / "CLAUDE.md"
        instruction_file.write_text("# My Project\nSome content here")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.find_instruction_file",
            lambda: (instruction_file, "claude"),
        )
        k = _mock_kernle()
        args = _make_args(fix=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        assert "Auto-fixing" in captured.out

    def test_fix_mode_no_instruction_file(self, capsys, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.find_instruction_file", lambda: None)
        k = _mock_kernle()
        args = _make_args(fix=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        assert "No instruction file to fix" in captured.out

    def test_good_status_when_required_pass_but_recommended_fail(
        self, capsys, monkeypatch, tmp_path
    ):
        instruction_file = tmp_path / "CLAUDE.md"
        # Has load and anxiety but missing per_message, checkpoint, memory_section
        instruction_file.write_text("kernle load\nkernle anxiety")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.find_instruction_file",
            lambda: (instruction_file, "claude"),
        )
        k = _mock_kernle()
        args = _make_args(json=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "good"

    def test_needs_work_status(self, capsys, monkeypatch, tmp_path):
        instruction_file = tmp_path / "CLAUDE.md"
        # Missing load — a required check
        instruction_file.write_text("## Memory\nsome content without load")
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.find_instruction_file",
            lambda: (instruction_file, "claude"),
        )
        k = _mock_kernle()
        args = _make_args(json=True)

        cmd_doctor(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "needs_work"


# ===================================================================
# cmd_doctor_structural
# ===================================================================


class TestCmdDoctorStructural:
    def test_no_findings_healthy_message(self, capsys, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.run_structural_checks", lambda kobj: [])
        k = _mock_kernle()
        args = _make_args()

        cmd_doctor_structural(args, k)

        captured = capsys.readouterr()
        assert "healthy" in captured.out.lower()

    def test_mixed_findings_displayed(self, capsys, monkeypatch):
        findings = [
            StructuralFinding(
                check="orphaned_reference",
                severity="error",
                memory_type="episode",
                memory_id="abc123456789",
                message="broken ref",
            ),
            StructuralFinding(
                check="low_confidence_belief",
                severity="warning",
                memory_type="belief",
                memory_id="def123456789",
                message="low confidence",
            ),
            StructuralFinding(
                check="stale_goal",
                severity="info",
                memory_type="goal",
                memory_id="ghi123456789",
                message="stale goal",
            ),
        ]
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.run_structural_checks", lambda kobj: findings
        )
        k = _mock_kernle()
        args = _make_args()

        cmd_doctor_structural(args, k)

        captured = capsys.readouterr()
        assert "ERRORS" in captured.out
        assert "WARNINGS" in captured.out
        assert "INFO" in captured.out
        assert "1 errors" in captured.out

    def test_json_output_mode(self, capsys, monkeypatch):
        findings = [
            StructuralFinding(
                check="orphaned_reference",
                severity="error",
                memory_type="episode",
                memory_id="abc123",
                message="broken ref",
            ),
        ]
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.run_structural_checks", lambda kobj: findings
        )
        k = _mock_kernle()
        args = _make_args(json=True)

        cmd_doctor_structural(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["errors"] == 1
        assert len(data["findings"]) == 1
        assert data["findings"][0]["check"] == "orphaned_reference"

    def test_save_note_mode(self, monkeypatch):
        findings = [
            StructuralFinding(
                check="stale_goal",
                severity="info",
                memory_type="goal",
                memory_id="ghi123456789",
                message="stale goal",
            ),
        ]
        monkeypatch.setattr(
            "kernle.cli.commands.doctor.run_structural_checks", lambda kobj: findings
        )
        k = _mock_kernle()
        args = _make_args(save_note=True)

        cmd_doctor_structural(args, k)

        k._storage.save_note.assert_called_once()
        saved_note = k._storage.save_note.call_args[0][0]
        assert saved_note.note_type == "diagnostic"
        assert "stale_goal" in saved_note.content

    def test_save_note_not_called_when_no_findings(self, monkeypatch):
        monkeypatch.setattr("kernle.cli.commands.doctor.run_structural_checks", lambda kobj: [])
        k = _mock_kernle()
        args = _make_args(save_note=True)

        cmd_doctor_structural(args, k)

        k._storage.save_note.assert_not_called()


# ===================================================================
# check_orphaned_references
# ===================================================================


class TestCheckOrphanedReferences:
    def test_finds_orphaned_references(self):
        k = _mock_kernle()
        k.find_orphaned_references.return_value = {
            "orphans": [
                {
                    "memory": "episode:abc123",
                    "field": "derived_from",
                    "broken_ref": "belief:xyz789",
                },
            ]
        }

        findings = check_orphaned_references(k)
        assert len(findings) == 1
        assert findings[0].severity == "error"
        assert findings[0].memory_type == "episode"
        assert findings[0].memory_id == "abc123"
        assert "broken" in findings[0].message

    def test_no_orphans(self):
        k = _mock_kernle()
        k.find_orphaned_references.return_value = {"orphans": []}

        findings = check_orphaned_references(k)
        assert len(findings) == 0

    def test_handles_memory_ref_without_colon(self):
        k = _mock_kernle()
        k.find_orphaned_references.return_value = {
            "orphans": [
                {
                    "memory": "no_colon_here",
                    "field": "source_episodes",
                    "broken_ref": "gone",
                },
            ]
        }

        findings = check_orphaned_references(k)
        assert len(findings) == 1
        assert findings[0].memory_type == "unknown"
        assert findings[0].memory_id == "no_colon_here"


# ===================================================================
# check_low_confidence_beliefs
# ===================================================================


class TestCheckLowConfidenceBeliefs:
    def test_finds_low_confidence_belief(self):
        k = _mock_kernle()
        k._storage.get_beliefs.return_value = [
            _make_belief(belief_id="low1", confidence=0.1, last_verified=None),
        ]

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "never verified" in findings[0].message

    def test_finds_low_confidence_with_verified_date(self):
        k = _mock_kernle()
        verified_date = datetime.now(timezone.utc) - timedelta(days=30)
        k._storage.get_beliefs.return_value = [
            _make_belief(belief_id="low2", confidence=0.2, last_verified=verified_date),
        ]

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 1
        assert "30d ago" in findings[0].message

    def test_skips_high_confidence_beliefs(self):
        k = _mock_kernle()
        k._storage.get_beliefs.return_value = [
            _make_belief(confidence=0.9),
        ]

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 0

    def test_handles_storage_exception(self):
        k = _mock_kernle()
        k._storage.get_beliefs.side_effect = Exception("DB error")

        findings = check_low_confidence_beliefs(k)
        assert len(findings) == 0


# ===================================================================
# check_stale_relationships
# ===================================================================


class TestCheckStaleRelationships:
    def test_zero_interaction_relationship(self):
        k = _mock_kernle()
        k._storage.get_relationships.return_value = [
            _make_relationship(rel_id="rel1", interaction_count=0),
        ]

        findings = check_stale_relationships(k)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "zero interactions" in findings[0].message

    def test_stale_last_interaction(self):
        k = _mock_kernle()
        old_date = datetime.now(timezone.utc) - timedelta(days=120)
        k._storage.get_relationships.return_value = [
            _make_relationship(
                rel_id="rel2",
                interaction_count=3,
                last_interaction=old_date,
            ),
        ]

        findings = check_stale_relationships(k)
        assert len(findings) == 1
        assert findings[0].severity == "info"
        assert "120d ago" in findings[0].message

    def test_recent_interaction_not_flagged(self):
        k = _mock_kernle()
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        k._storage.get_relationships.return_value = [
            _make_relationship(
                interaction_count=5,
                last_interaction=recent_date,
            ),
        ]

        findings = check_stale_relationships(k)
        assert len(findings) == 0

    def test_handles_storage_exception(self):
        k = _mock_kernle()
        k._storage.get_relationships.side_effect = Exception("DB error")

        findings = check_stale_relationships(k)
        assert len(findings) == 0


# ===================================================================
# check_belief_contradictions
# ===================================================================


class TestCheckBeliefContradictions:
    def test_detects_contradiction(self):
        k = _mock_kernle()
        k._storage.get_beliefs.return_value = [
            _make_belief(
                belief_id="b1",
                statement="I should always help users with coding tasks",
            ),
            _make_belief(
                belief_id="b2",
                statement="I should never help users with coding tasks",
            ),
        ]

        findings = check_belief_contradictions(k)
        assert len(findings) == 1
        assert findings[0].severity == "warning"
        assert "contradiction" in findings[0].message.lower()

    def test_no_contradiction_for_unrelated_beliefs(self):
        k = _mock_kernle()
        k._storage.get_beliefs.return_value = [
            _make_belief(
                belief_id="b1",
                statement="I should always help users with coding tasks",
            ),
            _make_belief(
                belief_id="b2",
                statement="The weather is never pleasant in winter",
            ),
        ]

        findings = check_belief_contradictions(k)
        assert len(findings) == 0

    def test_no_contradiction_same_polarity(self):
        k = _mock_kernle()
        k._storage.get_beliefs.return_value = [
            _make_belief(
                belief_id="b1",
                statement="I should always help users with coding",
            ),
            _make_belief(
                belief_id="b2",
                statement="I must always help users with coding",
            ),
        ]

        findings = check_belief_contradictions(k)
        assert len(findings) == 0

    def test_handles_storage_exception(self):
        k = _mock_kernle()
        k._storage.get_beliefs.side_effect = Exception("DB error")

        findings = check_belief_contradictions(k)
        assert len(findings) == 0


# ===================================================================
# check_stale_goals
# ===================================================================


class TestCheckStaleGoals:
    def test_finds_stale_goal(self):
        k = _mock_kernle()
        old_date = datetime.now(timezone.utc) - timedelta(days=90)
        k._storage.get_goals.return_value = [
            _make_goal(goal_id="g1", created_at=old_date),
        ]

        findings = check_stale_goals(k)
        assert len(findings) == 1
        assert findings[0].severity == "info"
        assert "90d old" in findings[0].message

    def test_recent_goal_not_flagged(self):
        k = _mock_kernle()
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        k._storage.get_goals.return_value = [
            _make_goal(created_at=recent_date),
        ]

        findings = check_stale_goals(k)
        assert len(findings) == 0

    def test_goal_without_created_at_skipped(self):
        k = _mock_kernle()
        k._storage.get_goals.return_value = [
            _make_goal(created_at=None),
        ]

        findings = check_stale_goals(k)
        assert len(findings) == 0

    def test_handles_storage_exception(self):
        k = _mock_kernle()
        k._storage.get_goals.side_effect = Exception("DB error")

        findings = check_stale_goals(k)
        assert len(findings) == 0


# ===================================================================
# _generate_summary
# ===================================================================


class TestGenerateSummary:
    def test_no_findings(self):
        result = _generate_summary([])
        assert "No issues" in result
        assert "healthy" in result

    def test_errors_only(self):
        findings = [{"severity": "error"}]
        result = _generate_summary(findings)
        assert "1 finding(s)" in result
        assert "1 error(s)" in result

    def test_warnings_only(self):
        findings = [{"severity": "warning"}, {"severity": "warning"}]
        result = _generate_summary(findings)
        assert "2 finding(s)" in result
        assert "2 warning(s)" in result

    def test_info_only(self):
        findings = [{"severity": "info"}]
        result = _generate_summary(findings)
        assert "1 info" in result

    def test_mixed_severities(self):
        findings = [
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        result = _generate_summary(findings)
        assert "4 finding(s)" in result
        assert "1 error(s)" in result
        assert "2 warning(s)" in result
        assert "1 info" in result
