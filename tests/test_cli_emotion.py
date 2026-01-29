"""Tests for CLI emotion command module."""

from argparse import Namespace
from unittest.mock import MagicMock

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

    def test_search_calm_filter(self):
        """Search with calm filter."""
        k = MagicMock()
        k.search_by_emotion.return_value = []

        args = Namespace(
            emotion_action="search",
            positive=False,
            negative=False,
            valence_min=None,
            valence_max=None,
            calm=True,
            intense=False,
            arousal_min=None,
            arousal_max=None,
            tag=None,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        call_args = k.search_by_emotion.call_args
        assert call_args[1]["arousal_range"] == (0.0, 0.3)

    def test_search_intense_filter(self):
        """Search with intense filter."""
        k = MagicMock()
        k.search_by_emotion.return_value = []

        args = Namespace(
            emotion_action="search",
            positive=False,
            negative=False,
            valence_min=None,
            valence_max=None,
            calm=False,
            intense=True,
            arousal_min=None,
            arousal_max=None,
            tag=None,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        call_args = k.search_by_emotion.call_args
        assert call_args[1]["arousal_range"] == (0.7, 1.0)

    def test_search_with_ranges(self):
        """Search with custom valence and arousal ranges."""
        k = MagicMock()
        k.search_by_emotion.return_value = []

        args = Namespace(
            emotion_action="search",
            positive=False,
            negative=False,
            valence_min=-0.5,
            valence_max=0.5,
            calm=False,
            intense=False,
            arousal_min=0.3,
            arousal_max=0.7,
            tag=None,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        call_args = k.search_by_emotion.call_args
        assert call_args[1]["valence_range"] == (-0.5, 0.5)
        assert call_args[1]["arousal_range"] == (0.3, 0.7)

    def test_search_with_results(self, capsys):
        """Search with results should display them."""
        k = MagicMock()
        k.search_by_emotion.return_value = [
            {
                "id": "ep123",
                "objective": "Test episode objective",
                "emotional_valence": 0.8,
                "emotional_arousal": 0.6,
                "emotional_tags": ["joy", "excitement"],
                "created_at": "2026-01-28T10:00:00Z",
            },
            {
                "id": "ep456",
                "objective": "Another episode",
                "emotional_valence": -0.5,
                "emotional_arousal": 0.3,
                "emotional_tags": ["sadness"],
                "created_at": "2026-01-27T09:00:00Z",
            },
        ]

        args = Namespace(
            emotion_action="search",
            positive=False,
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

        captured = capsys.readouterr()
        assert "2 matching episode" in captured.out
        assert "Test episode objective" in captured.out
        assert "joy, excitement" in captured.out
        assert "😊" in captured.out  # Positive emoji
        assert "😢" in captured.out  # Negative emoji

    def test_search_json(self, capsys):
        """Search JSON output."""
        k = MagicMock()
        k.search_by_emotion.return_value = [{"id": "ep123", "emotional_valence": 0.5}]

        args = Namespace(
            emotion_action="search",
            positive=False,
            negative=False,
            valence_min=None,
            valence_max=None,
            calm=False,
            intense=False,
            arousal_min=None,
            arousal_max=None,
            tag=None,
            limit=10,
            json=True,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert '"emotional_valence"' in captured.out


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


class TestCmdEmotionMood:
    """Test emotion mood command."""

    def test_mood_no_results(self, capsys):
        """Mood with no relevant memories."""
        k = MagicMock()
        k.get_mood_relevant_memories.return_value = []

        args = Namespace(
            emotion_action="mood",
            valence=0.5,
            arousal=0.5,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        k.get_mood_relevant_memories.assert_called_with(
            current_valence=0.5,
            current_arousal=0.5,
            limit=10,
        )
        captured = capsys.readouterr()
        assert "Memories relevant to mood" in captured.out
        assert "No mood-relevant memories" in captured.out

    def test_mood_with_results(self, capsys):
        """Mood with relevant memories."""
        k = MagicMock()
        k.get_mood_relevant_memories.return_value = [
            {
                "id": "ep123",
                "objective": "Happy memory",
                "outcome_description": "It went well",
                "emotional_valence": 0.8,
                "emotional_arousal": 0.6,
                "lessons_learned": ["Be grateful"],
                "created_at": "2026-01-28",
            },
        ]

        args = Namespace(
            emotion_action="mood",
            valence=0.7,
            arousal=0.5,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert "😊" in captured.out  # Happy mood
        assert "Happy memory" in captured.out
        assert "Be grateful" in captured.out

    def test_mood_negative(self, capsys):
        """Mood with negative valence."""
        k = MagicMock()
        k.get_mood_relevant_memories.return_value = [
            {
                "id": "ep456",
                "objective": "Sad memory",
                "outcome_description": "It was difficult",
                "emotional_valence": -0.5,
                "emotional_arousal": 0.4,
                "lessons_learned": [],
                "created_at": "2026-01-27",
            },
        ]

        args = Namespace(
            emotion_action="mood",
            valence=-0.5,
            arousal=0.4,
            limit=10,
            json=False,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert "😢" in captured.out  # Sad mood

    def test_mood_json(self, capsys):
        """Mood JSON output."""
        k = MagicMock()
        k.get_mood_relevant_memories.return_value = [{"id": "ep123", "objective": "test"}]

        args = Namespace(
            emotion_action="mood",
            valence=0.0,
            arousal=0.5,
            limit=10,
            json=True,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out


class TestCmdEmotionSummaryJson:
    """Test emotion summary JSON output."""

    def test_summary_json(self, capsys):
        """Summary JSON output."""
        k = MagicMock()
        k.get_emotional_summary.return_value = {
            "average_valence": 0.5,
            "average_arousal": 0.6,
            "dominant_emotions": ["joy"],
            "emotional_trajectory": [],
            "episode_count": 5,
        }

        args = Namespace(
            emotion_action="summary",
            days=7,
            json=True,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert '"average_valence"' in captured.out
        assert "0.5" in captured.out


class TestCmdEmotionDetectJson:
    """Test emotion detect JSON output."""

    def test_detect_json(self, capsys):
        """Detect JSON output."""
        k = MagicMock()
        k.detect_emotion.return_value = {
            "valence": 0.7,
            "arousal": 0.5,
            "tags": ["joy"],
            "confidence": 0.8,
        }

        args = Namespace(
            emotion_action="detect",
            text="I'm so happy!",
            json=True,
        )

        cmd_emotion(args, k)

        captured = capsys.readouterr()
        assert '"valence"' in captured.out
        assert '"tags"' in captured.out
