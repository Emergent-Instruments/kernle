"""Tests for CLI identity command module."""

import json
from argparse import Namespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

from kernle.cli.commands.identity import cmd_consolidate, cmd_identity


class TestCmdConsolidate:
    """Test the cmd_consolidate function."""

    def test_no_episodes(self, capsys):
        """Test consolidation prompt with no episodes."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []
        k._storage.get_beliefs.return_value = []

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Memory Consolidation - Reflection Prompt" in captured.out
        assert "0 recent episodes" in captured.out
        assert "No episodes recorded yet" in captured.out
        assert "No beliefs recorded yet" in captured.out
        assert "Reflection Questions:" in captured.out

    def test_with_episodes(self, capsys):
        """Test consolidation prompt with episodes."""
        k = MagicMock()
        k.agent_id = "test-agent"

        # Create mock episodes
        ep1 = MagicMock()
        ep1.is_forgotten = False
        ep1.objective = "Complete task A"
        ep1.outcome = "Successfully completed"
        ep1.outcome_type = "success"
        ep1.lessons = ["Lesson 1", "Lesson 2"]
        ep1.emotional_valence = 0.5
        ep1.emotional_arousal = 0.7
        ep1.emotional_tags = ["satisfaction", "pride"]
        ep1.created_at = datetime(2026, 1, 15, tzinfo=timezone.utc)

        ep2 = MagicMock()
        ep2.is_forgotten = False
        ep2.objective = "Complete task B"
        ep2.outcome = "Partially completed"
        ep2.outcome_type = "partial"
        ep2.lessons = []
        ep2.emotional_valence = 0
        ep2.emotional_arousal = 0
        ep2.emotional_tags = []
        ep2.created_at = datetime(2026, 1, 14, tzinfo=timezone.utc)

        k._storage.get_episodes.return_value = [ep1, ep2]
        k._storage.get_beliefs.return_value = []

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "2 recent episodes" in captured.out
        assert "Complete task A" in captured.out
        assert "Complete task B" in captured.out
        assert "Lesson 1" in captured.out
        assert "positive" in captured.out  # emotional valence
        assert "high" in captured.out  # emotional arousal
        assert "satisfaction" in captured.out

    def test_with_beliefs(self, capsys):
        """Test consolidation prompt with existing beliefs."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []

        # Create mock beliefs
        belief1 = MagicMock()
        belief1.is_active = True
        belief1.is_forgotten = False
        belief1.statement = "Testing is important"
        belief1.confidence = 0.85

        belief2 = MagicMock()
        belief2.is_active = True
        belief2.is_forgotten = False
        belief2.statement = "Simple code is better"
        belief2.confidence = 0.9

        k._storage.get_beliefs.return_value = [belief1, belief2]

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Current Beliefs (for context):" in captured.out
        assert "Testing is important" in captured.out
        assert "0.85" in captured.out
        assert "Simple code is better" in captured.out

    def test_with_repeated_lessons(self, capsys):
        """Test consolidation prompt highlights repeated lessons."""
        k = MagicMock()
        k.agent_id = "test-agent"

        # Create episodes with repeated lessons
        ep1 = MagicMock()
        ep1.is_forgotten = False
        ep1.objective = "Task 1"
        ep1.outcome = "Done"
        ep1.outcome_type = "success"
        ep1.lessons = ["Always test", "Document code"]
        ep1.emotional_valence = 0
        ep1.emotional_arousal = 0
        ep1.emotional_tags = []
        ep1.created_at = datetime(2026, 1, 15, tzinfo=timezone.utc)

        ep2 = MagicMock()
        ep2.is_forgotten = False
        ep2.objective = "Task 2"
        ep2.outcome = "Done"
        ep2.outcome_type = "success"
        ep2.lessons = ["Always test", "Review PRs"]
        ep2.emotional_valence = 0
        ep2.emotional_arousal = 0
        ep2.emotional_tags = []
        ep2.created_at = datetime(2026, 1, 14, tzinfo=timezone.utc)

        ep3 = MagicMock()
        ep3.is_forgotten = False
        ep3.objective = "Task 3"
        ep3.outcome = "Done"
        ep3.outcome_type = "success"
        ep3.lessons = ["Always test"]
        ep3.emotional_valence = 0
        ep3.emotional_arousal = 0
        ep3.emotional_tags = []
        ep3.created_at = datetime(2026, 1, 13, tzinfo=timezone.utc)

        k._storage.get_episodes.return_value = [ep1, ep2, ep3]
        k._storage.get_beliefs.return_value = []

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Patterns Detected:" in captured.out
        assert "Always test" in captured.out
        assert "appears 3 times" in captured.out

    def test_filters_forgotten_episodes(self, capsys):
        """Test that forgotten episodes are filtered out."""
        k = MagicMock()
        k.agent_id = "test-agent"

        # Create a forgotten episode
        forgotten_ep = MagicMock()
        forgotten_ep.is_forgotten = True

        # Create an active episode
        active_ep = MagicMock()
        active_ep.is_forgotten = False
        active_ep.objective = "Active task"
        active_ep.outcome = "Done"
        active_ep.outcome_type = "success"
        active_ep.lessons = []
        active_ep.emotional_valence = 0
        active_ep.emotional_arousal = 0
        active_ep.emotional_tags = []
        active_ep.created_at = datetime(2026, 1, 15, tzinfo=timezone.utc)

        k._storage.get_episodes.return_value = [forgotten_ep, active_ep]
        k._storage.get_beliefs.return_value = []

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "1 recent episodes" in captured.out  # Only active episode counted

    def test_filters_forgotten_beliefs(self, capsys):
        """Test that forgotten beliefs are filtered out."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []

        # Create a forgotten belief
        forgotten_belief = MagicMock()
        forgotten_belief.is_active = True
        forgotten_belief.is_forgotten = True

        # Create an active belief
        active_belief = MagicMock()
        active_belief.is_active = True
        active_belief.is_forgotten = False
        active_belief.statement = "Active belief"
        active_belief.confidence = 0.8

        k._storage.get_beliefs.return_value = [forgotten_belief, active_belief]

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Active belief" in captured.out
        # Only 1 belief should be shown

    def test_outcome_icons(self, capsys):
        """Test outcome type icons are correct."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_beliefs.return_value = []

        # Create episodes with different outcome types
        success = MagicMock()
        success.is_forgotten = False
        success.objective = "Success task"
        success.outcome = "Done"
        success.outcome_type = "success"
        success.lessons = []
        success.emotional_valence = 0
        success.emotional_arousal = 0
        success.emotional_tags = []
        success.created_at = datetime(2026, 1, 15, tzinfo=timezone.utc)

        failure = MagicMock()
        failure.is_forgotten = False
        failure.objective = "Failed task"
        failure.outcome = "Failed"
        failure.outcome_type = "failure"
        failure.lessons = []
        failure.emotional_valence = 0
        failure.emotional_arousal = 0
        failure.emotional_tags = []
        failure.created_at = datetime(2026, 1, 14, tzinfo=timezone.utc)

        k._storage.get_episodes.return_value = [success, failure]

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        # Check that success/failure markers are present
        lines = captured.out.split("\n")
        assert any("Success task" in line for line in lines)
        assert any("Failed task" in line for line in lines)
        # The actual icons depend on the terminal, but the logic should differentiate

    def test_action_suggestions(self, capsys):
        """Test that action suggestions are shown."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []
        k._storage.get_beliefs.return_value = []

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Actions:" in captured.out
        assert f"kernle -a {k.agent_id} belief add" in captured.out
        assert f"kernle -a {k.agent_id} belief reinforce" in captured.out
        assert f"kernle -a {k.agent_id} belief revise" in captured.out

    def test_emotional_context_negative_valence(self, capsys):
        """Test negative emotional valence display."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_beliefs.return_value = []

        ep = MagicMock()
        ep.is_forgotten = False
        ep.objective = "Frustrating task"
        ep.outcome = "Failed"
        ep.outcome_type = "failure"
        ep.lessons = []
        ep.emotional_valence = -0.5  # Negative
        ep.emotional_arousal = 0.2  # Low
        ep.emotional_tags = ["frustration"]
        ep.created_at = datetime(2026, 1, 15, tzinfo=timezone.utc)

        k._storage.get_episodes.return_value = [ep]

        args = Namespace(limit=20)

        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "negative" in captured.out
        assert "low" in captured.out
        assert "frustration" in captured.out


class TestCmdIdentityShow:
    """Test cmd_identity show action."""

    def test_show_text_output(self, capsys):
        """Test identity show with text output."""
        k = MagicMock()
        k.agent_id = "test-agent"
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
        assert f"Identity Synthesis for {k.agent_id}" in captured.out
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
        k.get_identity_confidence.return_value = 0.85

        args = Namespace(identity_action="confidence", json=True)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["agent_id"] == "test-agent"
        assert output["confidence"] == 0.85

    def test_confidence_zero(self, capsys):
        """Test confidence at zero."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k.get_identity_confidence.return_value = 0.0

        args = Namespace(identity_action="confidence", json=False)

        cmd_identity(args, k)

        captured = capsys.readouterr()
        assert "0%" in captured.out

    def test_confidence_full(self, capsys):
        """Test confidence at 100%."""
        k = MagicMock()
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
        k.agent_id = "test-agent"
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
