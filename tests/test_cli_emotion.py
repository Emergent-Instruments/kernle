"""Tests for CLI emotion command module."""

import pytest
from unittest.mock import MagicMock
from argparse import Namespace

from kernle.cli.commands.emotion import cmd_emotion


class TestCmdEmotionSummary:
    """Test emotion summary command."""

    def test_summary_no_data(self, capsys):
        """Summary with no data should show message."""
        k = MagicMock()
        k.get_emotional_summary.return_value = {
            "average_valence": 0.0,
            "average_arousal": 0.0,
            "dominant_emotions": [],
            "emotional_trajectory": [],
            "episode_count": 0,
        }
        
        args = Namespace(
            emotion_action="summary",
            days=7,
            json=False,
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "No emotional data" in captured.out

    def test_summary_with_data(self, capsys):
        """Summary with data should display properly."""
        k = MagicMock()
        k.get_emotional_summary.return_value = {
            "average_valence": 0.5,
            "average_arousal": 0.6,
            "dominant_emotions": ["joy", "excitement"],
            "emotional_trajectory": [
                {"date": "2026-01-27", "valence": 0.4, "arousal": 0.5},
                {"date": "2026-01-28", "valence": 0.6, "arousal": 0.7},
            ],
            "episode_count": 10,
        }
        
        args = Namespace(
            emotion_action="summary",
            days=7,
            json=False,
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "Emotional Summary" in captured.out
        assert "positive" in captured.out
        assert "joy" in captured.out


class TestCmdEmotionDetect:
    """Test emotion detection command."""

    def test_detect_no_emotion(self, capsys):
        """Detect with no emotional signals."""
        k = MagicMock()
        k.detect_emotion.return_value = {
            "valence": 0.0,
            "arousal": 0.0,
            "tags": [],
            "confidence": 0.0,
        }
        
        args = Namespace(
            emotion_action="detect",
            text="The sky is blue.",
            json=False,
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "No emotional signals" in captured.out

    def test_detect_positive_emotion(self, capsys):
        """Detect positive emotion."""
        k = MagicMock()
        k.detect_emotion.return_value = {
            "valence": 0.8,
            "arousal": 0.6,
            "tags": ["joy", "excitement"],
            "confidence": 0.8,
        }
        
        args = Namespace(
            emotion_action="detect",
            text="I'm so happy and excited!",
            json=False,
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "positive" in captured.out
        assert "joy" in captured.out


class TestCmdEmotionSearch:
    """Test emotion search command."""

    def test_search_positive_filter(self):
        """Search with positive filter."""
        k = MagicMock()
        k.search_by_emotion.return_value = []
        
        args = Namespace(
            emotion_action="search",
            positive=True,
            negative=False,
            valence_min=None,
            valence_max=None,
            calm=False,
            intense=False,
            arousal_min=None,
            arousal_max=None,
            tag=None,
            limit=10,
            json=False,
        )
        
        cmd_emotion(args, k)
        
        # Should pass valence_range (0.3, 1.0) for positive
        call_args = k.search_by_emotion.call_args
        assert call_args[1]["valence_range"] == (0.3, 1.0)

    def test_search_negative_filter(self):
        """Search with negative filter."""
        k = MagicMock()
        k.search_by_emotion.return_value = []
        
        args = Namespace(
            emotion_action="search",
            positive=False,
            negative=True,
            valence_min=None,
            valence_max=None,
            calm=False,
            intense=False,
            arousal_min=None,
            arousal_max=None,
            tag=None,
            limit=10,
            json=False,
        )
        
        cmd_emotion(args, k)
        
        call_args = k.search_by_emotion.call_args
        assert call_args[1]["valence_range"] == (-1.0, -0.3)


class TestCmdEmotionTag:
    """Test emotion tagging command."""

    def test_tag_success(self, capsys):
        """Successful tagging."""
        k = MagicMock()
        k.add_emotional_association.return_value = True
        
        args = Namespace(
            emotion_action="tag",
            episode_id="abc123",
            valence=0.5,
            arousal=0.6,
            tag=["happy"],
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "abc123" in captured.out

    def test_tag_not_found(self, capsys):
        """Tagging non-existent episode."""
        k = MagicMock()
        k.add_emotional_association.return_value = False
        
        args = Namespace(
            emotion_action="tag",
            episode_id="nonexistent",
            valence=0.5,
            arousal=0.6,
            tag=None,
        )
        
        cmd_emotion(args, k)
        
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "not found" in captured.out
