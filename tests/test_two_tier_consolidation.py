"""Tests for two-tier consolidation: regular + epoch-closing."""

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from kernle.cli.commands.epoch import cmd_epoch
from kernle.cli.commands.identity import cmd_consolidate
from kernle.features.consolidation import build_epoch_closing_scaffold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    id="ep-001",
    objective="Task",
    outcome="Done",
    outcome_type="success",
    lessons=None,
    tags=None,
    emotional_valence=0.0,
    emotional_arousal=0.0,
    emotional_tags=None,
    created_at=None,
    epoch_id=None,
):
    ep = MagicMock()
    ep.id = id
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
    ep.epoch_id = epoch_id
    return ep


def _make_epoch(
    id="epoch-001",
    name="Test Epoch",
    epoch_number=1,
    started_at=None,
    ended_at=None,
):
    epoch = MagicMock()
    epoch.id = id
    epoch.name = name
    epoch.epoch_number = epoch_number
    epoch.started_at = started_at or datetime.now(timezone.utc) - timedelta(days=30)
    epoch.ended_at = ended_at
    epoch.trigger_type = "declared"
    epoch.trigger_description = None
    epoch.summary = None
    epoch.key_belief_ids = None
    epoch.key_relationship_ids = None
    epoch.key_goal_ids = None
    epoch.dominant_drive_ids = None
    return epoch


def _make_belief(id="b-001", statement="Test belief", confidence=0.8):
    b = MagicMock()
    b.id = id
    b.statement = statement
    b.confidence = confidence
    b.is_active = True
    b.strength = 1.0
    b.times_reinforced = 0
    b.superseded_by = None
    b.source_domain = None
    b.cross_domain_applications = None
    b.belief_type = "pattern"
    b.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    return b


def _make_relationship(id="r-001", entity_name="Alice"):
    r = MagicMock()
    r.id = id
    r.entity_name = entity_name
    return r


def _make_goal(id="g-001", title="Test goal", status="active"):
    g = MagicMock()
    g.id = id
    g.title = title
    g.status = status
    return g


def _make_drive(id="d-001", drive_type="curiosity", intensity=0.7, focus_areas=None):
    d = MagicMock()
    d.id = id
    d.drive_type = drive_type
    d.intensity = intensity
    d.focus_areas = focus_areas or []
    return d


def _make_narrative(content="I am a curious entity.", key_themes=None, tensions=None):
    n = MagicMock()
    n.content = content
    n.key_themes = key_themes or ["curiosity"]
    n.unresolved_tensions = tensions
    return n


def _setup_kernle_mock(
    epoch=None,
    episodes=None,
    beliefs=None,
    relationships=None,
    goals=None,
    drives=None,
    narrative=None,
):
    """Create a Kernle mock with standard storage returns."""
    k = MagicMock()
    k.stack_id = "test-agent"
    k._storage.get_epoch.return_value = epoch or _make_epoch()
    k._storage.get_episodes.return_value = episodes or []
    k._storage.get_beliefs.return_value = beliefs or []
    k._storage.get_relationships.return_value = relationships or []
    k._storage.get_goals.return_value = goals or []
    k._storage.get_drives.return_value = drives or []
    k.narrative_get_active.return_value = narrative

    # scaffold_belief_to_value is a method on the mixin, so mock it
    k.scaffold_belief_to_value.return_value = {
        "beliefs_scanned": 0,
        "candidates": [],
        "scaffold": "## Belief-to-Value Promotion Analysis\n\nNo beliefs currently meet value-promotion criteria.\n",
    }
    return k


# ---------------------------------------------------------------------------
# Regular consolidation still works
# ---------------------------------------------------------------------------


class TestRegularConsolidation:
    """cmd_consolidate is now a deprecated alias for cmd_promote.

    It prints a deprecation warning and delegates to the promote command.
    """

    def test_regular_consolidation_delegates_to_promote(self, capsys):
        """cmd_consolidate delegates to cmd_promote with deprecation warning."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 5,
            "patterns_found": 1,
            "suggestions": [
                {
                    "lesson": "Testing is important",
                    "count": 2,
                    "source_episodes": ["ep1", "ep2"],
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
        assert "deprecated" in captured.err.lower()
        assert "Promotion Results" in captured.out
        assert "Testing is important" in captured.out

    def test_regular_consolidation_calls_promote(self, capsys):
        """cmd_consolidate calls k.promote() under the hood."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.promote.return_value = {
            "episodes_scanned": 0,
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
        k.promote.assert_called_once()


# ---------------------------------------------------------------------------
# Epoch-closing consolidation scaffold
# ---------------------------------------------------------------------------


class TestEpochClosingScaffold:
    """build_epoch_closing_scaffold produces all 6 steps."""

    def test_returns_all_six_steps(self):
        """Epoch-closing scaffold returns exactly 6 steps."""
        epoch = _make_epoch()
        episodes = [_make_episode(id=f"ep-{i}", epoch_id="epoch-001") for i in range(5)]
        k = _setup_kernle_mock(
            epoch=epoch,
            episodes=episodes,
            beliefs=[_make_belief()],
            relationships=[_make_relationship()],
            goals=[_make_goal()],
            drives=[_make_drive()],
            narrative=_make_narrative(),
        )

        result = build_epoch_closing_scaffold(k, "epoch-001")

        assert result["epoch_id"] == "epoch-001"
        assert len(result["steps"]) == 6

        step_numbers = [s["number"] for s in result["steps"]]
        assert step_numbers == [1, 2, 3, 4, 5, 6]

        step_names = [s["name"] for s in result["steps"]]
        assert "Epoch Summary" in step_names
        assert "Reference Snapshots" in step_names
        assert "Self-Narrative Update" in step_names
        assert "Belief-to-Value Promotion" in step_names
        assert "Drive Emergence Analysis" in step_names
        assert "Aggressive Archival" in step_names

    def test_combined_scaffold_text(self):
        """Combined scaffold contains all step headers."""
        epoch = _make_epoch()
        k = _setup_kernle_mock(epoch=epoch)

        result = build_epoch_closing_scaffold(k, "epoch-001")

        scaffold = result["scaffold"]
        assert "Epoch-Closing Consolidation" in scaffold
        assert "Step 1:" in scaffold
        assert "Step 2:" in scaffold
        assert "Step 3:" in scaffold
        assert "Step 4:" in scaffold
        assert "Step 5:" in scaffold
        assert "Step 6:" in scaffold

    def test_epoch_not_found(self):
        """Handles missing epoch gracefully."""
        k = MagicMock()
        k._storage.get_epoch.return_value = None

        result = build_epoch_closing_scaffold(k, "nonexistent")

        assert result["steps"] == []
        assert "not found" in result["scaffold"]

    def test_step1_epoch_summary_content(self):
        """Step 1 provides material for writing an epoch summary."""
        epoch = _make_epoch(name="Growth Phase")
        episodes = [
            _make_episode(
                id="ep-1",
                objective="Built the feature",
                outcome_type="success",
                lessons=["Incremental is better"],
                tags=["engineering"],
                epoch_id="epoch-001",
            ),
            _make_episode(
                id="ep-2",
                objective="Debugging session",
                outcome_type="failure",
                tags=["engineering"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(epoch=epoch, episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step1 = result["steps"][0]

        assert step1["name"] == "Epoch Summary"
        assert "Growth Phase" in step1["scaffold"]
        assert "Episodes in epoch" in step1["scaffold"]
        assert "success" in step1["scaffold"]
        assert "summary" in step1["scaffold"].lower()

    def test_step2_reference_snapshots(self):
        """Step 2 includes belief, relationship, goal, and drive IDs."""
        k = _setup_kernle_mock(
            beliefs=[_make_belief(id="b-1", statement="Testing matters")],
            relationships=[_make_relationship(id="r-1", entity_name="Bob")],
            goals=[_make_goal(id="g-1", title="Ship v2")],
            drives=[_make_drive(id="d-1", drive_type="mastery")],
        )

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step2 = result["steps"][1]

        assert step2["name"] == "Reference Snapshots"
        assert "b-1" in step2["data"]["key_belief_ids"]
        assert "r-1" in step2["data"]["key_relationship_ids"]
        assert "g-1" in step2["data"]["key_goal_ids"]
        assert "d-1" in step2["data"]["dominant_drive_ids"]

    def test_step3_narrative_update_with_existing(self):
        """Step 3 shows existing narrative and prompts for update."""
        narrative = _make_narrative(
            content="I learn through building.",
            key_themes=["building", "learning"],
        )
        k = _setup_kernle_mock(narrative=narrative)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step3 = result["steps"][2]

        assert step3["name"] == "Self-Narrative Update"
        assert "I learn through building" in step3["scaffold"]
        assert "still hold" in step3["scaffold"]
        assert step3["data"]["has_narrative"] is True

    def test_step3_narrative_update_without_existing(self):
        """Step 3 handles missing narrative gracefully."""
        k = _setup_kernle_mock(narrative=None)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step3 = result["steps"][2]

        assert "No active self-narrative" in step3["scaffold"]
        assert step3["data"]["has_narrative"] is False

    def test_step4_belief_to_value_promotion(self):
        """Step 4 delegates to scaffold_belief_to_value."""
        k = _setup_kernle_mock()
        k.scaffold_belief_to_value.return_value = {
            "beliefs_scanned": 10,
            "candidates": [],
            "scaffold": "## Belief-to-Value Promotion Analysis\n\nScanned 10 beliefs.",
        }

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step4 = result["steps"][3]

        assert step4["name"] == "Belief-to-Value Promotion"
        assert "10 beliefs" in step4["scaffold"]
        k.scaffold_belief_to_value.assert_called_once()

    def test_step5_drive_emergence(self):
        """Step 5 analyzes behavioral patterns for undeclared drives."""
        drives = [_make_drive(drive_type="curiosity")]
        episodes = [
            _make_episode(
                id="ep-1",
                tags=["coding", "reading"],
                epoch_id="epoch-001",
            ),
            _make_episode(
                id="ep-2",
                tags=["coding", "writing"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(drives=drives, episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step5 = result["steps"][4]

        assert step5["name"] == "Drive Emergence Analysis"
        # "coding" appears twice and is not a declared drive
        assert "coding" in step5["scaffold"]

    def test_step6_archive_candidates(self):
        """Step 6 identifies low-salience episodes for archival."""
        episodes = [
            # Low-salience: neutral, low arousal, no lessons
            _make_episode(
                id="ep-low",
                objective="Routine task",
                emotional_valence=0.0,
                emotional_arousal=0.1,
                lessons=[],
                epoch_id="epoch-001",
            ),
            # High-salience: positive, high arousal, has lessons
            _make_episode(
                id="ep-high",
                objective="Big discovery",
                emotional_valence=0.8,
                emotional_arousal=0.9,
                lessons=["Discovery changes everything"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step6 = result["steps"][5]

        assert step6["name"] == "Aggressive Archival"
        assert step6["data"]["candidate_count"] == 1
        assert "ep-low" in step6["data"]["candidate_ids"]
        assert "ep-high" not in step6["data"]["candidate_ids"]

    def test_step6_no_archive_candidates(self):
        """Step 6 handles no candidates gracefully."""
        episodes = [
            _make_episode(
                id="ep-1",
                emotional_valence=0.5,
                emotional_arousal=0.5,
                lessons=["learned something"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step6 = result["steps"][5]

        assert step6["data"]["candidate_count"] == 0
        assert "No obvious archive candidates" in step6["scaffold"]


# ---------------------------------------------------------------------------
# Epoch-closing integrates with CLI epoch close
# ---------------------------------------------------------------------------


class TestEpochCloseIntegration:
    """Epoch close CLI triggers epoch-closing consolidation."""

    def test_epoch_close_triggers_consolidation(self, capsys):
        """Closing an epoch via CLI outputs the consolidation scaffold."""
        k = MagicMock()
        k.stack_id = "test-agent"

        epoch = _make_epoch(id="epoch-x", name="Test Era")
        k.get_current_epoch.return_value = epoch
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {
            "epoch_id": "epoch-x",
            "steps": [{"number": 1, "name": "Test", "scaffold": "test scaffold"}],
            "scaffold": "# Epoch-Closing Consolidation: Test Era\n\nConsolidation output here",
        }

        args = Namespace(epoch_action="close", id=None, summary=None, json=False)
        cmd_epoch(args, k)

        captured = capsys.readouterr()
        assert "Epoch closed." in captured.out
        assert "Epoch-Closing Consolidation" in captured.out
        k.consolidate_epoch_closing.assert_called_once_with("epoch-x")

    def test_epoch_close_json_includes_consolidation(self, capsys):
        """JSON output includes consolidation data."""
        k = MagicMock()
        k.stack_id = "test-agent"

        epoch = _make_epoch(id="epoch-y")
        k.get_current_epoch.return_value = epoch
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {
            "epoch_id": "epoch-y",
            "steps": [],
            "scaffold": "test",
        }

        args = Namespace(epoch_action="close", id=None, summary=None, json=True)
        cmd_epoch(args, k)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["closed"] is True
        assert data["consolidation"]["epoch_id"] == "epoch-y"

    def test_epoch_close_no_epoch_skips_consolidation(self, capsys):
        """When no epoch to close, consolidation is not triggered."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.get_current_epoch.return_value = None
        k.epoch_close.return_value = False

        args = Namespace(epoch_action="close", id=None, summary=None, json=False)
        cmd_epoch(args, k)

        captured = capsys.readouterr()
        assert "No open epoch to close" in captured.out
        k.consolidate_epoch_closing.assert_not_called()

    def test_epoch_close_with_explicit_id(self, capsys):
        """Closing an epoch with explicit ID works."""
        k = MagicMock()
        k.stack_id = "test-agent"
        k.epoch_close.return_value = True
        k.consolidate_epoch_closing.return_value = {
            "epoch_id": "epoch-z",
            "steps": [],
            "scaffold": "# Epoch-Closing Consolidation: Explicit\n\nDone",
        }

        args = Namespace(epoch_action="close", id="epoch-z", summary=None, json=False)
        cmd_epoch(args, k)

        captured = capsys.readouterr()
        assert "Epoch closed." in captured.out
        k.consolidate_epoch_closing.assert_called_once_with("epoch-z")


# ---------------------------------------------------------------------------
# Core API: consolidate_epoch_closing
# ---------------------------------------------------------------------------


class TestCoreConsolidateEpochClosing:
    """Test the Kernle.consolidate_epoch_closing method."""

    def test_consolidate_epoch_closing_delegates(self):
        """consolidate_epoch_closing delegates to build_epoch_closing_scaffold."""
        k = _setup_kernle_mock()
        k._validate_string_input = lambda s, n, m: s

        with patch("kernle.features.consolidation.build_epoch_closing_scaffold") as mock_build:
            mock_build.return_value = {
                "epoch_id": "epoch-1",
                "steps": [],
                "scaffold": "test",
            }

            # Call the real method
            from kernle.core import Kernle

            result = Kernle.consolidate_epoch_closing(k, "epoch-1")

            mock_build.assert_called_once_with(k, "epoch-1")
            assert result["epoch_id"] == "epoch-1"


# ---------------------------------------------------------------------------
# Each scaffold step produces meaningful content
# ---------------------------------------------------------------------------


class TestScaffoldStepContent:
    """Each scaffold step produces non-trivial, meaningful content."""

    def test_step1_includes_lesson_frequencies(self):
        """Step 1 shows repeated lessons with counts."""
        episodes = [
            _make_episode(
                id="ep-1",
                lessons=["Patience pays off", "Test first"],
                epoch_id="epoch-001",
            ),
            _make_episode(
                id="ep-2",
                lessons=["Patience pays off"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step1 = result["steps"][0]

        assert "Patience pays off" in step1["scaffold"]
        assert "x2" in step1["scaffold"]

    def test_step1_includes_tag_themes(self):
        """Step 1 shows top tags as themes."""
        episodes = [
            _make_episode(id="ep-1", tags=["python", "testing"], epoch_id="epoch-001"),
            _make_episode(id="ep-2", tags=["python", "debugging"], epoch_id="epoch-001"),
        ]
        k = _setup_kernle_mock(episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step1 = result["steps"][0]

        assert "python" in step1["scaffold"]

    def test_step2_includes_belief_statements(self):
        """Step 2 shows belief statements in snapshots."""
        beliefs = [_make_belief(id="b-1", statement="Code reviews catch bugs")]
        k = _setup_kernle_mock(beliefs=beliefs)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step2 = result["steps"][1]

        assert "Code reviews catch bugs" in step2["scaffold"]

    def test_step3_shows_epoch_stats(self):
        """Step 3 shows episode stats since the epoch."""
        episodes = [
            _make_episode(id="ep-1", outcome_type="success", epoch_id="epoch-001"),
            _make_episode(id="ep-2", outcome_type="failure", epoch_id="epoch-001"),
            _make_episode(id="ep-3", outcome_type="success", epoch_id="epoch-001"),
        ]
        k = _setup_kernle_mock(episodes=episodes, narrative=_make_narrative())

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step3 = result["steps"][2]

        assert "3 episodes" in step3["scaffold"]
        assert "2 successes" in step3["scaffold"]
        assert "1 failures" in step3["scaffold"]

    def test_step5_identifies_unmatched_themes(self):
        """Step 5 flags themes without matching declared drives."""
        drives = [_make_drive(drive_type="curiosity", focus_areas=["learning"])]
        episodes = [
            _make_episode(
                id="ep-1",
                tags=["cooking", "cooking"],
                epoch_id="epoch-001",
            ),
            _make_episode(
                id="ep-2",
                tags=["cooking"],
                epoch_id="epoch-001",
            ),
        ]
        k = _setup_kernle_mock(drives=drives, episodes=episodes)

        result = build_epoch_closing_scaffold(k, "epoch-001")
        step5 = result["steps"][4]

        assert "cooking" in step5["scaffold"]
        assert len(step5["data"]["unmatched_themes"]) > 0

    def test_empty_epoch_produces_valid_scaffold(self):
        """An epoch with no episodes still produces a valid scaffold."""
        k = _setup_kernle_mock(episodes=[])

        result = build_epoch_closing_scaffold(k, "epoch-001")

        assert len(result["steps"]) == 6
        assert result["scaffold"]  # non-empty
        # Step 1 should note no episodes
        step1 = result["steps"][0]
        assert "0" in step1["scaffold"] or "No episodes" in step1["scaffold"]
