"""Tests for consolidation scaffold: emotional weighting and drive emergence."""

from argparse import Namespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from kernle.cli.commands.identity import _print_drive_pattern_analysis, cmd_consolidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    objective="Task",
    outcome="Done",
    outcome_type="success",
    lessons=None,
    tags=None,
    emotional_valence=0.0,
    emotional_arousal=0.0,
    emotional_tags=None,
    created_at=None,
):
    """Create a mock episode with sensible defaults."""
    ep = MagicMock()
    ep.is_forgotten = False
    ep.objective = objective
    ep.outcome = outcome
    ep.outcome_type = outcome_type
    ep.lessons = lessons or []
    ep.tags = tags or []
    ep.emotional_valence = emotional_valence
    ep.emotional_arousal = emotional_arousal
    ep.emotional_tags = emotional_tags or []
    ep.created_at = created_at or datetime.now(timezone.utc)
    return ep


def _make_drive(drive_type, intensity=0.5, focus_areas=None):
    """Create a mock drive."""
    d = MagicMock()
    d.drive_type = drive_type
    d.intensity = intensity
    d.focus_areas = focus_areas or []
    return d


# ---------------------------------------------------------------------------
# High-Arousal Episodes
# ---------------------------------------------------------------------------


class TestHighArousalEpisodes:
    def test_high_arousal_section_shown(self, capsys):
        """High-arousal episodes section appears when arousal > 0.6."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Intense experience",
                emotional_arousal=0.8,
                emotional_valence=0.5,
                emotional_tags=["excitement"],
            ),
        ]
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "HIGH-AROUSAL EPISODES" in captured.out
        assert "Intense experience" in captured.out
        assert "0.80" in captured.out
        assert "excitement" in captured.out

    def test_no_high_arousal_section_when_none(self, capsys):
        """No high-arousal section when all episodes have low arousal."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Calm task",
                emotional_arousal=0.2,
                emotional_valence=0.1,
            ),
        ]
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "HIGH-AROUSAL EPISODES" not in captured.out

    def test_high_arousal_sorted_by_arousal(self, capsys):
        """High-arousal episodes are sorted by arousal (highest first)."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Moderate intensity",
                emotional_arousal=0.7,
                emotional_valence=0.0,
            ),
            _make_episode(
                objective="Maximum intensity",
                emotional_arousal=0.95,
                emotional_valence=-0.3,
            ),
        ]
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        # Extract just the high-arousal section
        ha_start = captured.out.index("HIGH-AROUSAL EPISODES")
        ha_section = captured.out[ha_start:]
        lines = ha_section.split("\n")
        # Maximum intensity should appear before Moderate intensity within section
        max_idx = next(i for i, line in enumerate(lines) if "Maximum intensity" in line)
        mod_idx = next(i for i, line in enumerate(lines) if "Moderate intensity" in line)
        assert max_idx < mod_idx

    def test_high_arousal_valence_labels(self, capsys):
        """Valence labels are correct (positive, negative, neutral)."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Positive high",
                emotional_arousal=0.8,
                emotional_valence=0.5,
            ),
            _make_episode(
                objective="Negative high",
                emotional_arousal=0.9,
                emotional_valence=-0.5,
            ),
            _make_episode(
                objective="Neutral high",
                emotional_arousal=0.7,
                emotional_valence=0.0,
            ),
        ]
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        # Check all three valence labels appear in the high-arousal section
        assert "positive" in captured.out
        assert "negative" in captured.out
        assert "neutral" in captured.out

    def test_high_arousal_boundary(self, capsys):
        """Episodes with arousal exactly 0.6 are NOT included (> not >=)."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Boundary case",
                emotional_arousal=0.6,
                emotional_valence=0.0,
            ),
        ]
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "HIGH-AROUSAL EPISODES" not in captured.out


# ---------------------------------------------------------------------------
# Drive Pattern Analysis
# ---------------------------------------------------------------------------


class TestDrivePatternAnalysis:
    def test_no_output_when_no_recent_episodes(self, capsys):
        """No drive analysis when no recent episodes."""
        k = MagicMock()
        k._storage.get_drives.return_value = []

        # Episode from 60 days ago (outside 30-day window)
        old_ep = _make_episode(
            tags=["coding"],
            created_at=datetime.now(timezone.utc) - timedelta(days=60),
        )
        _print_drive_pattern_analysis([old_ep], k)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_output_when_no_tags(self, capsys):
        """No drive analysis when episodes have no tags."""
        k = MagicMock()
        k._storage.get_drives.return_value = []

        ep = _make_episode(tags=[], emotional_tags=[])
        _print_drive_pattern_analysis([ep], k)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_shows_declared_drives(self, capsys):
        """Drive analysis shows declared drives."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("curiosity", intensity=0.8, focus_areas=["coding"]),
        ]

        ep1 = _make_episode(tags=["coding", "research"])
        ep2 = _make_episode(tags=["coding", "research"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        assert "DRIVE PATTERN ANALYSIS" in captured.out
        assert "curiosity" in captured.out
        assert "80%" in captured.out

    def test_detects_unmatched_patterns(self, capsys):
        """Detects recurring tags that don't match any declared drive."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("curiosity", focus_areas=["coding"]),
        ]

        # "mentoring" appears 3 times but is not a declared drive or focus area
        ep1 = _make_episode(tags=["mentoring", "coding"])
        ep2 = _make_episode(tags=["mentoring"])
        ep3 = _make_episode(tags=["mentoring"])
        _print_drive_pattern_analysis([ep1, ep2, ep3], k)

        captured = capsys.readouterr()
        assert "Potential undeclared drives" in captured.out
        assert "mentoring" in captured.out
        assert "3 episodes" in captured.out

    def test_matched_tags_not_flagged(self, capsys):
        """Tags matching declared drives are not flagged as unmatched."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("curiosity", focus_areas=["coding", "research"]),
        ]

        ep1 = _make_episode(tags=["coding", "research"])
        ep2 = _make_episode(tags=["coding", "research"])
        ep3 = _make_episode(tags=["coding"])
        _print_drive_pattern_analysis([ep1, ep2, ep3], k)

        captured = capsys.readouterr()
        assert "All recurring themes align with declared drives" in captured.out
        assert "Potential undeclared drives" not in captured.out

    def test_emotional_tags_counted(self, capsys):
        """Emotional tags are also counted in the pattern analysis."""
        k = MagicMock()
        k._storage.get_drives.return_value = []

        ep1 = _make_episode(emotional_tags=["pride", "satisfaction"])
        ep2 = _make_episode(emotional_tags=["pride"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        assert "DRIVE PATTERN ANALYSIS" in captured.out
        assert "pride" in captured.out
        assert "2 occurrences" in captured.out

    def test_case_insensitive_matching(self, capsys):
        """Tag matching is case-insensitive."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("Curiosity", focus_areas=["Coding"]),
        ]

        ep1 = _make_episode(tags=["coding", "CODING"])
        ep2 = _make_episode(tags=["coding"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        # "coding" should match the declared focus area "Coding"
        assert "Potential undeclared drives" not in captured.out

    def test_single_occurrence_not_flagged(self, capsys):
        """Tags appearing only once are not flagged as potential drives."""
        k = MagicMock()
        k._storage.get_drives.return_value = []

        # "rare-tag" appears only once
        ep1 = _make_episode(tags=["common", "rare-tag"])
        ep2 = _make_episode(tags=["common"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        # "rare-tag" should not appear in the undeclared drives section
        if "Potential undeclared drives" in captured.out:
            undeclared_section = captured.out.split("Potential undeclared drives")[1]
            assert "rare-tag" not in undeclared_section

    def test_episode_count_in_header(self, capsys):
        """Shows count of recent episodes in the header."""
        k = MagicMock()
        k._storage.get_drives.return_value = []

        ep1 = _make_episode(tags=["coding"])
        ep2 = _make_episode(tags=["coding"])
        ep3 = _make_episode(tags=["coding"])
        _print_drive_pattern_analysis([ep1, ep2, ep3], k)

        captured = capsys.readouterr()
        assert "3 episodes" in captured.out

    def test_drive_type_matched(self, capsys):
        """Tags matching drive_type (not just focus_areas) are considered matched."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("achievement"),
        ]

        ep1 = _make_episode(tags=["achievement", "growth"])
        ep2 = _make_episode(tags=["achievement", "growth"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        # "achievement" matches the drive type, should not be flagged
        # But "growth" should be unmatched
        if "Potential undeclared drives" in captured.out:
            assert "achievement" not in captured.out.split("Potential undeclared drives")[1]

    def test_star_marker_for_unmatched(self, capsys):
        """Unmatched tags in the themes list get a * marker."""
        k = MagicMock()
        k._storage.get_drives.return_value = [
            _make_drive("curiosity", focus_areas=["coding"]),
        ]

        ep1 = _make_episode(tags=["coding", "teaching"])
        ep2 = _make_episode(tags=["coding", "teaching"])
        _print_drive_pattern_analysis([ep1, ep2], k)

        captured = capsys.readouterr()
        assert "no matching declared drive" in captured.out


# ---------------------------------------------------------------------------
# Integration: cmd_consolidate with both new sections
# ---------------------------------------------------------------------------


class TestConsolidateIntegration:
    def test_both_sections_appear(self, capsys):
        """Both high-arousal and drive analysis appear in consolidation output."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = [
            _make_drive("curiosity"),
        ]

        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="Exciting discovery",
                emotional_arousal=0.9,
                emotional_valence=0.7,
                tags=["research", "breakthrough"],
                emotional_tags=["excitement"],
            ),
            _make_episode(
                objective="Follow-up research",
                emotional_arousal=0.3,
                tags=["research"],
            ),
        ]

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "HIGH-AROUSAL EPISODES" in captured.out
        assert "DRIVE PATTERN ANALYSIS" in captured.out
        assert "Reflection Questions" in captured.out

    def test_sections_order(self, capsys):
        """Sections appear in correct order: episodes, beliefs, patterns,
        high-arousal, drive analysis, reflection questions, actions."""
        k = MagicMock()
        k.agent_id = "test-agent"

        belief = MagicMock()
        belief.is_active = True
        belief.is_forgotten = False
        belief.statement = "Test belief"
        belief.confidence = 0.8
        k._storage.get_beliefs.return_value = [belief]
        k._storage.get_drives.return_value = []

        k._storage.get_episodes.return_value = [
            _make_episode(
                objective="High energy task",
                emotional_arousal=0.8,
                tags=["coding", "coding"],
                lessons=["repeated lesson"],
            ),
            _make_episode(
                objective="Another task",
                emotional_arousal=0.2,
                tags=["coding"],
                lessons=["repeated lesson"],
            ),
        ]

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        out = captured.out

        # Check ordering
        episodes_pos = out.index("Recent Episodes:")
        beliefs_pos = out.index("Current Beliefs")
        patterns_pos = out.index("Patterns Detected:")
        arousal_pos = out.index("HIGH-AROUSAL EPISODES")
        drive_pos = out.index("DRIVE PATTERN ANALYSIS")
        questions_pos = out.index("Reflection Questions:")
        actions_pos = out.index("Actions:")

        assert episodes_pos < beliefs_pos < patterns_pos
        assert patterns_pos < arousal_pos < drive_pos
        assert drive_pos < questions_pos < actions_pos

    def test_no_episodes_no_new_sections(self, capsys):
        """With no episodes, neither new section appears."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "HIGH-AROUSAL EPISODES" not in captured.out
        assert "DRIVE PATTERN ANALYSIS" not in captured.out

    def test_existing_sections_preserved(self, capsys):
        """Existing consolidation sections (reflection questions, actions) still appear."""
        k = MagicMock()
        k.agent_id = "test-agent"
        k._storage.get_episodes.return_value = []
        k._storage.get_beliefs.return_value = []
        k._storage.get_drives.return_value = []

        args = Namespace(limit=20)
        cmd_consolidate(args, k)

        captured = capsys.readouterr()
        assert "Memory Consolidation - Reflection Prompt" in captured.out
        assert "Reflection Questions:" in captured.out
        assert "Actions:" in captured.out
        assert "the agent" in captured.out.lower() or "you (the agent)" in captured.out.lower()
