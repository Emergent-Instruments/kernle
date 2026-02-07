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
    ep.strength = 1.0
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
    """These tests previously tested the old cmd_consolidate guided-reflection
    output. Since cmd_consolidate now delegates to cmd_promote, we test the
    _print_drive_pattern_analysis helper directly for drive analysis, and
    verify cmd_consolidate produces promotion output with deprecation warning.
    """

    def test_cmd_consolidate_is_deprecated_alias(self, capsys):
        """cmd_consolidate prints deprecation warning and delegates to cmd_promote."""
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
        assert "Promotion Results" in captured.out


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
# Integration: cmd_consolidate is now a deprecated alias for cmd_promote
# ---------------------------------------------------------------------------


class TestConsolidateIntegration:
    """Test that cmd_consolidate properly delegates to cmd_promote.

    The old guided-reflection behavior has been replaced by the promote
    command. cmd_consolidate now prints a deprecation warning and
    delegates to cmd_promote.
    """

    def _make_consolidate_args(self, **overrides):
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

    def test_consolidate_delegates_to_promote(self, capsys):
        """cmd_consolidate calls k.promote() and outputs promotion results."""
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
            ],
            "beliefs_created": 0,
        }

        cmd_consolidate(self._make_consolidate_args(), k)

        captured = capsys.readouterr()
        assert "Promotion Results" in captured.out
        assert "Always test first" in captured.out
        k.promote.assert_called_once()

    def test_consolidate_warns_deprecated(self, capsys):
        """cmd_consolidate prints deprecation warning to stderr."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 0,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
        }

        cmd_consolidate(self._make_consolidate_args(), k)

        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()
        assert "promote" in captured.err

    def test_consolidate_json_delegates(self, capsys):
        """cmd_consolidate --json delegates JSON output to cmd_promote."""
        result = {
            "episodes_scanned": 5,
            "patterns_found": 0,
            "suggestions": [],
            "beliefs_created": 0,
        }
        k = MagicMock()
        k.promote.return_value = result

        cmd_consolidate(self._make_consolidate_args(json=True), k)

        captured = capsys.readouterr()
        import json

        parsed = json.loads(captured.out)
        assert parsed["episodes_scanned"] == 5
