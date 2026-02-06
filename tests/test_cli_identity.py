"""Tests for CLI identity command module."""

import json
from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.identity import cmd_consolidate, cmd_identity, cmd_promote


class TestCmdPromote:
    """Test the cmd_promote function (episode -> belief promotion)."""

    def _make_promote_args(self, **overrides):
        defaults = dict(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=False,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_no_patterns(self, capsys):
        """Test promote with no patterns found."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 5,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
        }

        cmd_promote(self._make_promote_args(), k)

        captured = capsys.readouterr()
        assert "Promotion Results" in captured.out
        assert "Episodes scanned: 5" in captured.out
        assert "No recurring patterns found" in captured.out

    def test_with_suggestions(self, capsys):
        """Test promote with suggestions found."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 10,
            "patterns_found": 2,
            "suggestions": [
                {
                    "lesson": "Always test first",
                    "count": 3,
                    "source_episodes": ["ep1", "ep2", "ep3"],
                },
                {
                    "lesson": "Document changes",
                    "count": 2,
                    "source_episodes": ["ep1", "ep4"],
                },
            ],
            "beliefs_created": 0,
        }

        cmd_promote(self._make_promote_args(), k)

        captured = capsys.readouterr()
        assert "Episodes scanned: 10" in captured.out
        assert "Patterns found: 2" in captured.out
        assert "Always test first" in captured.out
        assert "Document changes" in captured.out

    def test_auto_mode(self, capsys):
        """Test promote in auto mode creates beliefs."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 10,
            "patterns_found": 1,
            "suggestions": [
                {
                    "lesson": "Always test first",
                    "count": 3,
                    "source_episodes": ["ep1", "ep2", "ep3"],
                    "promoted": True,
                    "belief_id": "belief-abc123",
                },
            ],
            "beliefs_created": 1,
        }

        cmd_promote(self._make_promote_args(auto=True), k)

        captured = capsys.readouterr()
        assert "Beliefs created: 1" in captured.out
        k.promote.assert_called_once_with(
            auto=True,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
        )

    def test_json_output(self, capsys):
        """Test promote with JSON output."""
        result = {
            "episodes_scanned": 10,
            "patterns_found": 1,
            "suggestions": [],
            "beliefs_created": 0,
        }
        k = MagicMock()
        k.promote.return_value = result

        cmd_promote(self._make_promote_args(json=True), k)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["episodes_scanned"] == 10

    def test_not_enough_episodes(self, capsys):
        """Test promote when not enough episodes."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 1,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
            "message": "Need at least 3 episodes (found 1)",
        }

        cmd_promote(self._make_promote_args(), k)

        captured = capsys.readouterr()
        assert "Need at least 3 episodes" in captured.out

    def test_manual_promote_hint(self, capsys):
        """Test that manual promote hint is shown when not in auto mode."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 10,
            "patterns_found": 1,
            "suggestions": [
                {
                    "lesson": "Always test",
                    "count": 2,
                    "source_episodes": ["ep1", "ep2"],
                },
            ],
            "beliefs_created": 0,
        }

        cmd_promote(self._make_promote_args(), k)

        captured = capsys.readouterr()
        assert "kernle" in captured.out
        assert "promote --auto" in captured.out


class TestCmdConsolidateDeprecated:
    """Test that cmd_consolidate is a deprecated alias for cmd_promote."""

    def test_deprecation_warning(self, capsys):
        """Test consolidate prints deprecation warning to stderr."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 5,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
        }

        args = Namespace(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=False,
        )

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()
        assert "promote" in captured.err

    def test_delegates_to_promote(self, capsys):
        """Test consolidate delegates to promote and produces same output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 10,
            "patterns_found": 1,
            "suggestions": [
                {
                    "lesson": "Always test first",
                    "count": 3,
                    "source_episodes": ["ep1", "ep2", "ep3"],
                },
            ],
            "beliefs_created": 0,
        }

        args = Namespace(
            auto=False,
            min_occurrences=2,
            min_episodes=3,
            confidence=0.7,
            limit=50,
            json=False,
        )

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        # Should produce promote output
        assert "Promotion Results" in captured.out
        assert "Always test first" in captured.out
        # k.promote should have been called
        k.promote.assert_called_once()


class TestCmdIdentityShow:
    """Test cmd_identity show action."""

    def test_show_text_output(self, capsys):
        """Test identity show with text output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.synthesize_identity.return_value = {
            "narrative": "A diligent agent focused on quality.",
            "core_values": [{"name": "quality", "priority": 1, "statement": "Quality over speed"}],
            "key_beliefs": [
                {"statement": "Testing is essential", "confidence": 0.9, "foundational": True}
            ],
            "active_goals": [{"title": "Complete project", "priority": "high"}],
            "drives": {"curiosity": 0.8, "achievement": 0.6},
            "significant_episodes": [
                {"objective": "Shipped v1.0", "outcome": "success", "lessons": ["Plan ahead"]}
            ],
            "confidence": 0.75,
        }

        args = Namespace(identity_action="show", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert f"Identity Synthesis for {k.stack_id}" in captured.out
        assert "Narrative" in captured.out
        assert "diligent agent" in captured.out
        assert "Core Values" in captured.out
        assert "quality" in captured.out
        assert "Key Beliefs" in captured.out
        assert "Testing is essential" in captured.out
        assert "[foundational]" in captured.out
        assert "Active Goals" in captured.out
        assert "Complete project" in captured.out
        assert "Drives" in captured.out
        assert "curiosity" in captured.out
        assert "Formative Experiences" in captured.out
        assert "Shipped v1.0" in captured.out
        assert "Plan ahead" in captured.out
        assert "Identity Confidence: 75%" in captured.out

    def test_show_json_output(self, capsys):
        """Test identity show with JSON output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        identity_data = {
            "narrative": "Test narrative",
            "core_values": [],
            "key_beliefs": [],
            "active_goals": [],
            "drives": {},
            "significant_episodes": [],
            "confidence": 0.5,
        }
        k.synthesize_identity.return_value = identity_data

        args = Namespace(identity_action="show", json=True)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["narrative"] == "Test narrative"
        assert output["confidence"] == 0.5

    def test_show_none_action_defaults_to_show(self, capsys):
        """Test that None identity_action defaults to show."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.synthesize_identity.return_value = {
            "narrative": "Default show",
            "core_values": [],
            "key_beliefs": [],
            "active_goals": [],
            "drives": {},
            "significant_episodes": [],
            "confidence": 0.5,
        }

        args = Namespace(identity_action=None, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Default show" in captured.out

    def test_show_empty_sections(self, capsys):
        """Test show with empty optional sections."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.synthesize_identity.return_value = {
            "narrative": "Minimal identity",
            "core_values": [],
            "key_beliefs": [],
            "active_goals": [],
            "drives": {},
            "significant_episodes": [],
            "confidence": 0.2,
        }

        args = Namespace(identity_action="show", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Minimal identity" in captured.out
        # Empty sections should not have headers printed
        assert "Core Values" not in captured.out
        assert "Key Beliefs" not in captured.out
        assert "Active Goals" not in captured.out
        assert "Drives" not in captured.out
        assert "Formative Experiences" not in captured.out


class TestCmdIdentityConfidence:
    """Test cmd_identity confidence action."""

    def test_confidence_text_output(self, capsys):
        """Test confidence with text output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.get_identity_confidence.return_value = 0.75

        args = Namespace(identity_action="confidence", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Identity Confidence:" in captured.out
        assert "75%" in captured.out
        # Should show a progress bar
        assert "[" in captured.out
        assert "]" in captured.out

    def test_confidence_json_output(self, capsys):
        """Test confidence with JSON output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.get_identity_confidence.return_value = 0.85

        args = Namespace(identity_action="confidence", json=True)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["stack_id"] == "test-agent"
        assert output["confidence"] == 0.85

    def test_confidence_zero(self, capsys):
        """Test confidence at zero."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.get_identity_confidence.return_value = 0.0

        args = Namespace(identity_action="confidence", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "0%" in captured.out

    def test_confidence_full(self, capsys):
        """Test confidence at 100%."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.get_identity_confidence.return_value = 1.0

        args = Namespace(identity_action="confidence", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "100%" in captured.out


class TestCmdIdentityDrift:
    """Test cmd_identity drift action."""

    def test_drift_text_output_stable(self, capsys):
        """Test drift with stable interpretation."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.1,  # Low = stable
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Identity Drift Analysis" in captured.out
        assert "past 30 days" in captured.out
        assert "Drift Score:" in captured.out
        assert "10%" in captured.out
        assert "stable" in captured.out

    def test_drift_text_output_evolving(self, capsys):
        """Test drift with evolving interpretation."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.35,  # evolving
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "evolving" in captured.out

    def test_drift_text_output_significant_change(self, capsys):
        """Test drift with significant change interpretation."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.6,  # significant change
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "significant change" in captured.out

    def test_drift_text_output_transformational(self, capsys):
        """Test drift with transformational interpretation."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.9,  # transformational
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "transformational" in captured.out

    def test_drift_json_output(self, capsys):
        """Test drift with JSON output."""
        k = MagicMock()
        k.stack_id = "test-agent"
        drift_data = {
            "period_days": 14,
            "drift_score": 0.25,
            "changed_values": [
                {"name": "efficiency", "change": "new", "statement": "Work smarter"}
            ],
            "evolved_beliefs": [{"statement": "Code review helps", "confidence": 0.8}],
            "new_experiences": [
                {
                    "objective": "Launched feature",
                    "outcome": "success",
                    "date": "2026-01-15",
                    "lessons": ["Ship early"],
                }
            ],
        }
        k.detect_identity_drift.return_value = drift_data

        args = Namespace(identity_action="drift", days=14, json=True)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["period_days"] == 14
        assert output["drift_score"] == 0.25
        assert len(output["changed_values"]) == 1

    def test_drift_with_changed_values(self, capsys):
        """Test drift showing changed values."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.4,
            "changed_values": [
                {
                    "name": "collaboration",
                    "change": "new",
                    "statement": "Work with others when possible",
                },
                {"name": "speed", "change": "modified", "statement": "Balance speed with quality"},
            ],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Changed Values" in captured.out
        assert "collaboration" in captured.out
        assert "speed" in captured.out

    def test_drift_with_evolved_beliefs(self, capsys):
        """Test drift showing evolved beliefs."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.3,
            "changed_values": [],
            "evolved_beliefs": [
                {"statement": "TDD leads to better code", "confidence": 0.85},
                {"statement": "Documentation is crucial", "confidence": 0.7},
            ],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "New/Evolved Beliefs" in captured.out
        assert "TDD leads to better code" in captured.out
        assert "85%" in captured.out

    def test_drift_with_new_experiences(self, capsys):
        """Test drift showing new experiences."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.45,
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [
                {
                    "objective": "Deployed to production",
                    "outcome": "success",
                    "date": "2026-01-20",
                    "lessons": ["Always have rollback plan"],
                },
                {
                    "objective": "Fixed critical bug",
                    "outcome": "partial",
                    "date": "2026-01-18",
                    "lessons": [],
                },
            ],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Recent Significant Experiences" in captured.out
        assert "Deployed to production" in captured.out
        assert "2026-01-20" in captured.out
        assert "Always have rollback plan" in captured.out

    def test_drift_empty_sections_not_shown(self, capsys):
        """Test that empty drift sections are not shown."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 30,
            "drift_score": 0.05,
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=30, json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "Changed Values" not in captured.out
        assert "New/Evolved Beliefs" not in captured.out
        assert "Recent Significant Experiences" not in captured.out

    def test_drift_custom_days(self, capsys):
        """Test drift with custom days parameter."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.detect_identity_drift.return_value = {
            "period_days": 7,
            "drift_score": 0.15,
            "changed_values": [],
            "evolved_beliefs": [],
            "new_experiences": [],
        }

        args = Namespace(identity_action="drift", days=7, json=False)

        cmd_identity(args, k)

        k.detect_identity_drift.assert_called_with(7)
        captured = capsys.readouterr()
        assert "past 7 days" in captured.out
