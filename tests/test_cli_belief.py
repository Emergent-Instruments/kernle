"""Tests for CLI belief command module."""

import pytest
from unittest.mock import MagicMock
from argparse import Namespace

from kernle.cli.commands.belief import cmd_belief


class TestCmdBeliefRevise:
    """Test belief revise command."""

    def test_revise_success(self, capsys):
        """Successful belief revision."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [
                {"statement": "Test works", "belief_id": "b123"}
            ],
            "contradicted": [],
            "suggested_new": [],
        }
        
        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=False,
        )
        
        cmd_belief(args, k)
        
        k.revise_beliefs_from_episode.assert_called_with("ep123")
        captured = capsys.readouterr()
        assert "Reinforced" in captured.out
        assert "Test works" in captured.out

    def test_revise_error(self, capsys):
        """Revision with error."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "error": "Episode not found",
        }
        
        args = Namespace(
            belief_action="revise",
            episode_id="nonexistent",
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "✗" in captured.out


class TestCmdBeliefContradictions:
    """Test belief contradictions command."""

    def test_no_contradictions(self, capsys):
        """No contradictions found."""
        k = MagicMock()
        k.find_contradictions.return_value = []
        
        args = Namespace(
            belief_action="contradictions",
            statement="The sky is blue",
            limit=10,
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "No contradictions found" in captured.out

    def test_with_contradictions(self, capsys):
        """Contradictions found."""
        k = MagicMock()
        k.find_contradictions.return_value = [
            {
                "belief_id": "b123",
                "statement": "Contradicting belief",
                "contradiction_confidence": 0.8,
                "contradiction_type": "semantic",
                "is_active": True,
                "times_reinforced": 2,
                "explanation": "Direct conflict",
            }
        ]
        
        args = Namespace(
            belief_action="contradictions",
            statement="Test statement",
            limit=10,
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "Potential Contradictions" in captured.out
        assert "Contradicting belief" in captured.out


class TestCmdBeliefHistory:
    """Test belief history command."""

    def test_history_not_found(self, capsys):
        """History for non-existent belief."""
        k = MagicMock()
        k.get_belief_history.return_value = []
        
        args = Namespace(
            belief_action="history",
            id="nonexistent",
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "No history found" in captured.out

    def test_history_found(self, capsys):
        """History for existing belief."""
        k = MagicMock()
        k.get_belief_history.return_value = [
            {
                "id": "b123",
                "statement": "Original belief",
                "confidence": 0.7,
                "is_current": False,
                "is_active": False,
                "times_reinforced": 1,
                "created_at": "2026-01-01",
                "supersession_reason": "Updated understanding",
                "superseded_by": "b456",
            },
            {
                "id": "b456",
                "statement": "Updated belief",
                "confidence": 0.9,
                "is_current": True,
                "is_active": True,
                "times_reinforced": 3,
                "created_at": "2026-01-15",
                "supersession_reason": None,
                "superseded_by": None,
            }
        ]
        
        args = Namespace(
            belief_action="history",
            id="b123",
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "Belief Revision History" in captured.out
        assert "Original belief" in captured.out
        assert "Updated belief" in captured.out


class TestCmdBeliefReinforce:
    """Test belief reinforce command."""

    def test_reinforce_success(self, capsys):
        """Successful reinforcement."""
        k = MagicMock()
        k.reinforce_belief.return_value = True
        
        args = Namespace(
            belief_action="reinforce",
            id="b123",
        )
        
        cmd_belief(args, k)
        
        k.reinforce_belief.assert_called_with("b123")
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "reinforced" in captured.out

    def test_reinforce_not_found(self, capsys):
        """Reinforcement of non-existent belief."""
        k = MagicMock()
        k.reinforce_belief.return_value = False
        
        args = Namespace(
            belief_action="reinforce",
            id="nonexistent",
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "not found" in captured.out


class TestCmdBeliefSupersede:
    """Test belief supersede command."""

    def test_supersede_success(self, capsys):
        """Successful supersession."""
        k = MagicMock()
        k.supersede_belief.return_value = "new-b456"
        
        args = Namespace(
            belief_action="supersede",
            old_id="b123",
            new_statement="New understanding",
            confidence=0.85,
            reason="Better evidence",
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "superseded" in captured.out
        assert "new-b456" in captured.out

    def test_supersede_error(self, capsys):
        """Supersession with error."""
        k = MagicMock()
        k.supersede_belief.side_effect = ValueError("Belief not found")
        
        args = Namespace(
            belief_action="supersede",
            old_id="nonexistent",
            new_statement="New statement",
            confidence=0.8,
            reason="Test",
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "✗" in captured.out


class TestCmdBeliefList:
    """Test belief list command."""

    def test_list_empty(self, capsys):
        """Empty belief list."""
        k = MagicMock()
        k._storage.get_beliefs.return_value = []
        
        args = Namespace(
            belief_action="list",
            limit=20,
            all=False,
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "Beliefs" in captured.out
        assert "0 total" in captured.out

    def test_list_with_beliefs(self, capsys):
        """List with beliefs."""
        belief1 = MagicMock()
        belief1.id = "b123"
        belief1.statement = "Test belief"
        belief1.confidence = 0.8
        belief1.times_reinforced = 2
        belief1.is_active = True
        belief1.supersedes = None
        belief1.superseded_by = None
        belief1.created_at = MagicMock()
        belief1.created_at.isoformat.return_value = "2026-01-01T00:00:00Z"
        
        k = MagicMock()
        k._storage.get_beliefs.return_value = [belief1]
        
        args = Namespace(
            belief_action="list",
            limit=20,
            all=False,
            json=False,
        )
        
        cmd_belief(args, k)
        
        captured = capsys.readouterr()
        assert "Test belief" in captured.out
        assert "80%" in captured.out
