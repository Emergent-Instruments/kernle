"""Tests for derived_from cycle detection.

Verifies that circular derivation references are detected and rejected
before being persisted, preventing infinite loops in lineage traversal.
"""

import pytest

from kernle.storage import Belief, Episode, Note
from kernle.storage.lineage import MAX_DERIVATION_DEPTH, check_derived_from_cycle
from kernle.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    """Create a fresh SQLiteStorage for testing."""
    db_path = tmp_path / "test.db"
    s = SQLiteStorage(stack_id="test-agent", db_path=db_path)
    yield s
    s.close()


class TestCycleDetectionUnit:
    """Unit tests for the check_derived_from_cycle function."""

    def test_no_derived_from(self, storage):
        """None or empty derived_from should not raise."""
        check_derived_from_cycle(storage, "episode", "ep1", None)
        check_derived_from_cycle(storage, "episode", "ep1", [])

    def test_direct_self_reference(self, storage):
        """A memory referencing itself should be detected."""
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            check_derived_from_cycle(storage, "episode", "ep1", ["episode:ep1"])

    def test_simple_cycle_a_b_a(self, storage):
        """A->B->A cycle should be detected when saving A with derived_from=[B]."""
        # Create episode B that derives from A
        ep_b = Episode(
            id="ep-b",
            stack_id="test-agent",
            objective="B",
            outcome="outcome-b",
            derived_from=["episode:ep-a"],
        )
        storage.save_episode(ep_b)

        # Now try to create episode A that derives from B -> should detect cycle
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            check_derived_from_cycle(storage, "episode", "ep-a", ["episode:ep-b"])

    def test_longer_cycle_a_b_c_a(self, storage):
        """A->B->C->A cycle should be detected."""
        # Create C -> derives from nothing yet
        ep_c = Episode(
            id="ep-c",
            stack_id="test-agent",
            objective="C",
            outcome="outcome-c",
            derived_from=["episode:ep-a"],
        )
        storage.save_episode(ep_c)

        # Create B -> derives from C
        ep_b = Episode(
            id="ep-b",
            stack_id="test-agent",
            objective="B",
            outcome="outcome-b",
            derived_from=["episode:ep-c"],
        )
        storage.save_episode(ep_b)

        # Now try to save A deriving from B -> should detect A->B->C->A cycle
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            check_derived_from_cycle(storage, "episode", "ep-a", ["episode:ep-b"])

    def test_no_cycle_valid_chain(self, storage):
        """A valid chain A->B->C should not raise."""
        # Create C (no parents)
        ep_c = Episode(
            id="ep-c",
            stack_id="test-agent",
            objective="C",
            outcome="outcome-c",
        )
        storage.save_episode(ep_c)

        # Create B -> derives from C
        ep_b = Episode(
            id="ep-b",
            stack_id="test-agent",
            objective="B",
            outcome="outcome-b",
            derived_from=["episode:ep-c"],
        )
        storage.save_episode(ep_b)

        # A derives from B -> valid chain, should not raise
        check_derived_from_cycle(storage, "episode", "ep-a", ["episode:ep-b"])

    def test_deep_chain_within_limit(self, storage):
        """A chain within the depth limit should not raise false positive."""
        # Build a chain: ep-0 <- ep-1 <- ep-2 <- ... <- ep-9 (depth 9)
        prev_id = None
        for i in range(10):
            ep = Episode(
                id=f"ep-{i}",
                stack_id="test-agent",
                objective=f"Episode {i}",
                outcome=f"outcome-{i}",
                derived_from=[f"episode:{prev_id}"] if prev_id else None,
            )
            storage.save_episode(ep)
            prev_id = f"ep-{i}"

        # Adding ep-new deriving from ep-9 should be fine (depth 10, within limit)
        check_derived_from_cycle(storage, "episode", "ep-new", ["episode:ep-9"])

    def test_depth_limit_exceeded(self, storage):
        """A chain exceeding the depth limit should be treated as potential cycle."""
        # Build a chain deeper than MAX_DERIVATION_DEPTH by saving without
        # derived_from first, then updating each one's derived_from directly
        # via SQL to bypass the cycle check during construction.
        from kernle.storage.sqlite import SQLiteStorage

        assert isinstance(storage, SQLiteStorage)
        for i in range(MAX_DERIVATION_DEPTH + 2):
            ep = Episode(
                id=f"ep-{i}",
                stack_id="test-agent",
                objective=f"Episode {i}",
                outcome=f"outcome-{i}",
            )
            storage.save_episode(ep)

        # Now wire up the chain via direct SQL to bypass cycle check
        import json

        with storage._connect() as conn:
            for i in range(1, MAX_DERIVATION_DEPTH + 2):
                conn.execute(
                    "UPDATE episodes SET derived_from = ? WHERE id = ?",
                    (json.dumps([f"episode:ep-{i-1}"]), f"ep-{i}"),
                )
            conn.commit()

        # Adding another link should exceed the depth limit
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            check_derived_from_cycle(
                storage,
                "episode",
                "ep-new",
                [f"episode:ep-{MAX_DERIVATION_DEPTH + 1}"],
            )

    def test_context_refs_skipped(self, storage):
        """context: and kernle: refs should be skipped during cycle check."""
        # These are not actual memory references, should not cause issues
        check_derived_from_cycle(storage, "episode", "ep-1", ["context:cli", "kernle:system"])

    def test_missing_ref_no_error(self, storage):
        """References to non-existent memories should not raise."""
        # ep-nonexistent doesn't exist in storage; traversal should stop gracefully
        check_derived_from_cycle(storage, "episode", "ep-1", ["episode:ep-nonexistent"])

    def test_cross_type_cycle(self, storage):
        """Cycles across memory types (episode->belief->episode) should be detected."""
        # Create a belief derived from episode ep-a
        belief = Belief(
            id="bel-1",
            stack_id="test-agent",
            statement="test belief",
            derived_from=["episode:ep-a"],
        )
        storage.save_belief(belief)

        # Now try to create episode ep-a derived from belief bel-1 -> cycle
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            check_derived_from_cycle(storage, "episode", "ep-a", ["belief:bel-1"])


class TestCycleDetectionIntegration:
    """Integration tests verifying cycle detection is enforced during save/update."""

    def test_save_episode_rejects_self_reference(self, storage):
        """save_episode should reject self-referencing derived_from."""
        ep = Episode(
            id="ep-self",
            stack_id="test-agent",
            objective="self-ref",
            outcome="outcome",
            derived_from=["episode:ep-self"],
        )
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            storage.save_episode(ep)

    def test_save_belief_rejects_cycle(self, storage):
        """save_belief should reject circular derived_from."""
        # Create belief A derived from belief B (B doesn't exist yet, so no cycle)
        bel_b = Belief(
            id="bel-b",
            stack_id="test-agent",
            statement="belief B",
            derived_from=["belief:bel-a"],
        )
        storage.save_belief(bel_b)

        # Now try to save belief A derived from belief B -> cycle
        bel_a = Belief(
            id="bel-a",
            stack_id="test-agent",
            statement="belief A",
            derived_from=["belief:bel-b"],
        )
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            storage.save_belief(bel_a)

    def test_save_note_rejects_cycle(self, storage):
        """save_note should reject circular derived_from."""
        note = Note(
            id="note-self",
            stack_id="test-agent",
            content="self-referencing note",
            derived_from=["note:note-self"],
        )
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            storage.save_note(note)

    def test_update_memory_meta_rejects_cycle(self, storage):
        """update_memory_meta should reject circular derived_from."""
        # Create two episodes
        ep_a = Episode(
            id="ep-a",
            stack_id="test-agent",
            objective="A",
            outcome="outcome-a",
        )
        storage.save_episode(ep_a)

        ep_b = Episode(
            id="ep-b",
            stack_id="test-agent",
            objective="B",
            outcome="outcome-b",
            derived_from=["episode:ep-a"],
        )
        storage.save_episode(ep_b)

        # Try to update ep-a's derived_from to point to ep-b -> cycle
        with pytest.raises(ValueError, match="Circular derived_from reference detected"):
            storage.update_memory_meta("episode", "ep-a", derived_from=["episode:ep-b"])

    def test_update_memory_meta_allows_valid_update(self, storage):
        """update_memory_meta should allow valid derived_from updates."""
        ep_a = Episode(
            id="ep-a",
            stack_id="test-agent",
            objective="A",
            outcome="outcome-a",
        )
        storage.save_episode(ep_a)

        ep_b = Episode(
            id="ep-b",
            stack_id="test-agent",
            objective="B",
            outcome="outcome-b",
        )
        storage.save_episode(ep_b)

        # ep-a derives from ep-b, no cycle
        result = storage.update_memory_meta("episode", "ep-a", derived_from=["episode:ep-b"])
        assert result is True

    def test_save_episode_allows_valid_derived_from(self, storage):
        """save_episode should allow valid (non-cyclic) derived_from."""
        ep_parent = Episode(
            id="ep-parent",
            stack_id="test-agent",
            objective="parent",
            outcome="outcome",
        )
        storage.save_episode(ep_parent)

        ep_child = Episode(
            id="ep-child",
            stack_id="test-agent",
            objective="child",
            outcome="outcome",
            derived_from=["episode:ep-parent"],
        )
        # Should succeed without error
        result_id = storage.save_episode(ep_child)
        assert result_id == "ep-child"
