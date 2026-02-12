"""Tests for confidence propagation through derived memory chains.

Tests both the core BFS propagation in MetaMemoryMixin.propagate_confidence()
and the CLI output formatting for the propagate action.
"""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest

from kernle.cli.commands.meta import cmd_meta
from kernle.core import Kernle
from kernle.storage import Belief, Episode, SQLiteStorage

# --- Fixtures ---


@pytest.fixture
def k(tmp_path):
    """Create a Kernle instance for testing."""
    db_path = tmp_path / "test_confidence.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    storage = SQLiteStorage(stack_id="test-agent", db_path=db_path)
    return Kernle(
        stack_id="test-agent",
        storage=storage,
        checkpoint_dir=checkpoint_dir,
    )


# --- Core propagation tests ---


class TestPropagateConfidence:
    """Tests for propagate_confidence BFS logic."""

    def test_propagate_caps_derived_at_source_confidence(self, k):
        """Episode E1 (conf=0.5) -> derived belief B1 (conf=0.8) -> B1 capped to 0.5."""
        # Create episode with confidence 0.5
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="test",
            outcome="ok",
            confidence=0.5,
        )
        k._storage.save_episode(ep)

        # Create belief derived from the episode, with higher confidence
        b = Belief(
            id="b1",
            stack_id="test-agent",
            statement="derived belief",
            confidence=0.8,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b)

        result = k.propagate_confidence("episode", "ep1")

        assert result["updated"] == 1
        assert result["source_confidence"] == 0.5

        # Verify belief confidence was capped
        updated_belief = k._storage.get_memory("belief", "b1")
        assert updated_belief.confidence == 0.5

    def test_propagate_no_derived_returns_zero(self, k):
        """Source with no derivations returns updated=0."""
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="test",
            outcome="ok",
            confidence=0.7,
        )
        k._storage.save_episode(ep)

        result = k.propagate_confidence("episode", "ep1")

        assert result["updated"] == 0
        assert result["source_confidence"] == 0.7

    def test_propagate_cascades_to_transitive(self, k):
        """E1 -> B1 -> B2 chain: propagating E1 also caps B2."""
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="test",
            outcome="ok",
            confidence=0.5,
        )
        k._storage.save_episode(ep)

        b1 = Belief(
            id="b1",
            stack_id="test-agent",
            statement="first derived",
            confidence=0.8,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b1)

        b2 = Belief(
            id="b2",
            stack_id="test-agent",
            statement="second derived",
            confidence=0.9,
            derived_from=["belief:b1"],
        )
        k._storage.save_belief(b2)

        result = k.propagate_confidence("episode", "ep1")

        assert result["updated"] == 2

        updated_b1 = k._storage.get_memory("belief", "b1")
        assert updated_b1.confidence == 0.5

        updated_b2 = k._storage.get_memory("belief", "b2")
        assert updated_b2.confidence == 0.5

    def test_propagate_grandchild_higher_than_source(self, k):
        """E1(0.5) -> B1(0.4) -> B2(0.9): B1 untouched, B2 capped to 0.4 (effective bound)."""
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="test",
            outcome="ok",
            confidence=0.5,
        )
        k._storage.save_episode(ep)

        b1 = Belief(
            id="b1",
            stack_id="test-agent",
            statement="lower than source",
            confidence=0.4,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b1)

        b2 = Belief(
            id="b2",
            stack_id="test-agent",
            statement="too high",
            confidence=0.9,
            derived_from=["belief:b1"],
        )
        k._storage.save_belief(b2)

        result = k.propagate_confidence("episode", "ep1")

        # B1 should not be updated (0.4 < 0.5)
        # B2 should be capped to min(0.4, 0.9) = 0.4 (effective bound from B1)
        updated_b1 = k._storage.get_memory("belief", "b1")
        assert updated_b1.confidence == 0.4  # unchanged

        updated_b2 = k._storage.get_memory("belief", "b2")
        assert updated_b2.confidence == 0.4

        # Only B2 was updated
        assert result["updated"] == 1

    def test_propagate_missing_memory_returns_error(self, k):
        """Non-existent ID returns error dict."""
        result = k.propagate_confidence("episode", "nonexistent-id")

        assert "error" in result
        assert "not found" in result["error"]

    def test_propagate_does_not_increase_confidence(self, k):
        """Source conf=0.9, derived conf=0.5 -> no change."""
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="test",
            outcome="ok",
            confidence=0.9,
        )
        k._storage.save_episode(ep)

        b = Belief(
            id="b1",
            stack_id="test-agent",
            statement="lower derived",
            confidence=0.5,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b)

        result = k.propagate_confidence("episode", "ep1")

        assert result["updated"] == 0

        updated_belief = k._storage.get_memory("belief", "b1")
        assert updated_belief.confidence == 0.5  # unchanged

    def test_propagate_diamond_does_not_double_count(self, k):
        """Diamond pattern A->B, A->C, B->D, C->D: D updated only once."""
        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="root",
            outcome="ok",
            confidence=0.5,
        )
        k._storage.save_episode(ep)

        b1 = Belief(
            id="b1",
            stack_id="test-agent",
            statement="branch one",
            confidence=0.8,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b1)

        b2 = Belief(
            id="b2",
            stack_id="test-agent",
            statement="branch two",
            confidence=0.7,
            derived_from=["episode:ep1"],
        )
        k._storage.save_belief(b2)

        # D derives from both B and C — diamond merge
        b3 = Belief(
            id="b3",
            stack_id="test-agent",
            statement="merge point",
            confidence=0.9,
            derived_from=["belief:b1", "belief:b2"],
        )
        k._storage.save_belief(b3)

        result = k.propagate_confidence("episode", "ep1")

        # b1 (0.8 > 0.5) capped, b2 (0.7 > 0.5) capped, b3 (0.9 > 0.5) capped
        assert result["updated"] == 3
        assert "error" not in result

        updated_b3 = k._storage.get_memory("belief", "b3")
        assert updated_b3.confidence == 0.5

    def test_propagate_depth_limit(self, k):
        """Chain of 5 with max_depth=3 stops early."""
        prev_type = "episode"
        prev_id = "ep1"

        ep = Episode(
            id="ep1",
            stack_id="test-agent",
            objective="root",
            outcome="ok",
            confidence=0.3,
        )
        k._storage.save_episode(ep)

        for i in range(5):
            b = Belief(
                id=f"b{i}",
                stack_id="test-agent",
                statement=f"belief {i}",
                confidence=0.9,
                derived_from=[f"{prev_type}:{prev_id}"],
            )
            k._storage.save_belief(b)
            prev_type = "belief"
            prev_id = f"b{i}"

        result = k.propagate_confidence("episode", "ep1", max_depth=3)

        # Should update at most 3 (depths 1, 2, 3 — b0, b1, b2)
        assert result["updated"] <= 3

        # Beliefs beyond depth 3 should NOT be updated
        deep_belief = k._storage.get_memory("belief", "b4")
        assert deep_belief.confidence == 0.9  # untouched


# --- CLI output tests ---


class TestCLIPropagateOutput:
    """Tests for CLI propagate action output formatting."""

    def test_cli_propagate_zero_updated_no_checkmark(self, capsys):
        """When updated=0, CLI prints info message, not checkmark."""
        k = MagicMock()
        k.propagate_confidence.return_value = {
            "source_confidence": 0.7,
            "source_ref": "episode:abc123",
            "updated": 0,
        }

        args = Namespace(
            meta_action="propagate",
            type="episode",
            id="abc12345",
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "\u2713" not in captured.out  # no checkmark
        assert "No derived memories needed updating" in captured.out
        assert "70%" in captured.out

    def test_cli_propagate_shows_updated_count(self, capsys):
        """When updated>0, CLI prints checkmark with count."""
        k = MagicMock()
        k.propagate_confidence.return_value = {
            "source_confidence": 0.9,
            "source_ref": "episode:abc123",
            "updated": 3,
        }

        args = Namespace(
            meta_action="propagate",
            type="episode",
            id="abc12345",
        )

        cmd_meta(args, k)

        captured = capsys.readouterr()
        assert "\u2713" in captured.out  # checkmark present
        assert "90%" in captured.out
        assert "3" in captured.out
