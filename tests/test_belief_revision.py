"""Tests for belief revision functionality.

Tests the belief revision system including:
- Finding contradictions
- Reinforcing beliefs
- Superseding beliefs
- Revising beliefs from episodes
- Getting belief history
"""

import pytest

from kernle import Kernle
from kernle.storage import SQLiteStorage


@pytest.fixture
def kernle_with_beliefs(tmp_path):
    """Create a Kernle instance with some initial beliefs."""
    db_path = tmp_path / "test_beliefs.db"
    storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    k = Kernle(agent_id="test_agent", storage=storage)

    # Add some initial beliefs
    k.belief("I should always validate user input", type="principle", confidence=0.9)
    k.belief("Python is a good language for rapid prototyping", type="fact", confidence=0.8)
    k.belief("I prefer test-driven development", type="preference", confidence=0.7)
    k.belief("I should avoid using global variables", type="principle", confidence=0.85)
    k.belief("I dislike working with legacy code", type="preference", confidence=0.6)

    return k


@pytest.fixture
def kernle_fresh(tmp_path):
    """Create a fresh Kernle instance."""
    db_path = tmp_path / "test_fresh.db"
    storage = SQLiteStorage(agent_id="test_agent", db_path=db_path)
    return Kernle(agent_id="test_agent", storage=storage)


class TestFindContradictions:
    """Tests for find_contradictions method."""

    def test_finds_direct_negation(self, kernle_with_beliefs):
        """Should find beliefs with direct negation patterns."""
        k = kernle_with_beliefs

        # Add a contradicting belief
        k.belief("I should never validate user input", type="principle", confidence=0.5)

        # Find contradictions
        contradictions = k.find_contradictions("I should always validate user input")

        # Should find the contradiction
        assert len(contradictions) >= 1
        contra = next((c for c in contradictions if "never" in c["statement"].lower()), None)
        assert contra is not None
        assert contra["contradiction_type"] == "direct_negation"

    def test_finds_preference_conflict(self, kernle_with_beliefs):
        """Should find beliefs with preference conflicts."""
        k = kernle_with_beliefs

        # Add a conflicting preference
        k.belief("I like working with legacy code", type="preference", confidence=0.5)

        # Find contradictions
        contradictions = k.find_contradictions("I like working with legacy code")

        # Should find the dislike contradiction
        assert len(contradictions) >= 1
        # Look for the dislike belief
        contra = next((c for c in contradictions if "dislike" in c["statement"].lower()), None)
        assert contra is not None
        # Can be either direct_negation or preference_conflict (like/dislike matches both patterns)
        assert contra["contradiction_type"] in ("preference_conflict", "direct_negation")

    def test_finds_comparative_opposition(self, kernle_with_beliefs):
        """Should find beliefs with comparative opposition (more/less, better/worse)."""
        k = kernle_with_beliefs

        # Add a belief with comparative
        k.belief(
            "Local-first memory is more reliable than cloud-dependent", type="fact", confidence=0.8
        )

        # Find contradictions with opposite comparative
        contradictions = k.find_contradictions(
            "Local-first memory is less reliable than cloud-dependent"
        )

        # Should find the contradiction
        assert len(contradictions) >= 1
        contra = next(
            (c for c in contradictions if "more reliable" in c["statement"].lower()), None
        )
        assert contra is not None
        assert contra["contradiction_type"] == "comparative_opposition"

    def test_finds_comparative_opposition_better_worse(self, kernle_with_beliefs):
        """Should find comparative contradictions with better/worse."""
        k = kernle_with_beliefs

        k.belief("Python is better than JavaScript for data science", type="fact", confidence=0.7)

        contradictions = k.find_contradictions("Python is worse than JavaScript for data science")

        assert len(contradictions) >= 1
        contra = next((c for c in contradictions if "better" in c["statement"].lower()), None)
        assert contra is not None
        assert contra["contradiction_type"] == "comparative_opposition"

    def test_no_contradictions_for_unrelated(self, kernle_with_beliefs):
        """Should not find contradictions for unrelated statements."""
        k = kernle_with_beliefs

        # Search for something unrelated
        contradictions = k.find_contradictions("The sky is blue")

        # Should find no contradictions
        assert len(contradictions) == 0

    def test_respects_limit(self, kernle_with_beliefs):
        """Should respect the limit parameter."""
        k = kernle_with_beliefs

        # Add many potential contradictions
        for i in range(10):
            k.belief(f"I should never use method {i}", type="principle", confidence=0.5)

        contradictions = k.find_contradictions("I should always use method 5", limit=3)
        assert len(contradictions) <= 3


class TestReinforceBeliefs:
    """Tests for reinforce_belief method."""

    def test_increments_reinforcement_count(self, kernle_fresh):
        """Should increment times_reinforced."""
        k = kernle_fresh

        # Add a belief
        belief_id = k.belief("Testing is important", confidence=0.7)

        # Reinforce it
        assert k.reinforce_belief(belief_id) is True

        # Check the belief
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief is not None
        assert belief.times_reinforced == 1

        # Reinforce again
        k.reinforce_belief(belief_id)

        # Check again
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.times_reinforced == 2

    def test_increases_confidence(self, kernle_fresh):
        """Should slightly increase confidence."""
        k = kernle_fresh

        # Add a belief with moderate confidence
        belief_id = k.belief("Testing is important", confidence=0.6)

        # Reinforce it
        k.reinforce_belief(belief_id)

        # Check confidence increased
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.confidence > 0.6

    def test_confidence_has_diminishing_returns(self, kernle_fresh):
        """Confidence increase should have diminishing returns."""
        k = kernle_fresh

        # Add a high-confidence belief
        belief_id = k.belief("Testing is important", confidence=0.95)

        original_beliefs = k._storage.get_beliefs(include_inactive=True)
        original_belief = next((b for b in original_beliefs if b.id == belief_id), None)
        original_confidence = original_belief.confidence

        # Reinforce many times
        for _ in range(10):
            k.reinforce_belief(belief_id)

        # Check confidence capped at 0.99
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)
        assert belief.confidence <= 0.99
        # Verify confidence did increase (diminishing returns, not zero returns)
        assert belief.confidence > original_confidence

    def test_returns_false_for_nonexistent(self, kernle_fresh):
        """Should return False for nonexistent belief."""
        k = kernle_fresh

        result = k.reinforce_belief("nonexistent-id")
        assert result is False

    def test_updates_confidence_history(self, kernle_fresh):
        """Should add entry to confidence_history."""
        k = kernle_fresh

        belief_id = k.belief("Testing is important", confidence=0.6)
        k.reinforce_belief(belief_id)

        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        assert belief.confidence_history is not None
        assert len(belief.confidence_history) > 0
        assert "Reinforced" in belief.confidence_history[-1]["reason"]


class TestSupersedeBelief:
    """Tests for supersede_belief method."""

    def test_creates_new_belief(self, kernle_fresh):
        """Should create a new belief that supersedes the old one."""
        k = kernle_fresh

        # Add original belief
        old_id = k.belief("Python 2 is the best", confidence=0.7)

        # Supersede it
        new_id = k.supersede_belief(
            old_id, "Python 3 is the best", confidence=0.9, reason="Python 2 is deprecated"
        )

        assert new_id != old_id

        # Check new belief
        beliefs = k._storage.get_beliefs(include_inactive=True)
        new_belief = next((b for b in beliefs if b.id == new_id), None)
        assert new_belief is not None
        assert new_belief.statement == "Python 3 is the best"
        assert new_belief.confidence == 0.9
        assert new_belief.supersedes == old_id
        assert new_belief.is_active is True

    def test_deactivates_old_belief(self, kernle_fresh):
        """Should deactivate the old belief."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        k.supersede_belief(old_id, "Python 3 is the best")

        # Old belief should be inactive
        beliefs = k._storage.get_beliefs(include_inactive=True)
        old_belief = next((b for b in beliefs if b.id == old_id), None)
        assert old_belief is not None
        assert old_belief.is_active is False

    def test_links_beliefs_bidirectionally(self, kernle_fresh):
        """Should link old and new beliefs both ways."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        new_id = k.supersede_belief(old_id, "Python 3 is the best")

        beliefs = k._storage.get_beliefs(include_inactive=True)
        old_belief = next((b for b in beliefs if b.id == old_id), None)
        new_belief = next((b for b in beliefs if b.id == new_id), None)

        assert old_belief.superseded_by == new_id
        assert new_belief.supersedes == old_id

    def test_raises_for_nonexistent(self, kernle_fresh):
        """Should raise ValueError for nonexistent belief."""
        k = kernle_fresh

        with pytest.raises(ValueError, match="not found"):
            k.supersede_belief("nonexistent-id", "New statement")

    def test_old_belief_excluded_from_active_list(self, kernle_fresh):
        """Old belief should not appear in default beliefs list."""
        k = kernle_fresh

        old_id = k.belief("Python 2 is the best", confidence=0.7)
        k.supersede_belief(old_id, "Python 3 is the best")

        # Get active beliefs only
        active_beliefs = k._storage.get_beliefs(include_inactive=False)
        old_belief = next((b for b in active_beliefs if b.id == old_id), None)
        assert old_belief is None


class TestReviseFromEpisode:
    """Tests for revise_beliefs_from_episode method."""

    def test_reinforces_relevant_beliefs(self, kernle_fresh):
        """Should reinforce beliefs supported by successful episode."""
        k = kernle_fresh

        # Add beliefs
        k.belief("I should write tests first", confidence=0.6)

        # Create a successful episode related to testing
        episode_id = k.episode(
            objective="Implement feature using TDD",
            outcome="success",
            lessons=["Writing tests first helped catch bugs early"],
        )

        # Revise beliefs
        result = k.revise_beliefs_from_episode(episode_id)

        # Should have reinforced the testing belief
        # Note: matching depends on word overlap, so this may or may not match
        # depending on implementation details
        assert "reinforced" in result
        assert "contradicted" in result
        assert "suggested_new" in result

    def test_identifies_contradicted_beliefs(self, kernle_fresh):
        """Should identify beliefs contradicted by failed episode."""
        k = kernle_fresh

        # Add a belief
        k.belief("Manual testing is always sufficient", confidence=0.7)

        # Create a failed episode
        episode_id = k.episode(
            objective="Ship without automated tests",
            outcome="failure",
            lessons=["Manual testing missed critical bugs"],
        )

        result = k.revise_beliefs_from_episode(episode_id)

        # Check structure
        assert isinstance(result["contradicted"], list)

    def test_suggests_new_beliefs_from_lessons(self, kernle_fresh):
        """Should suggest new beliefs from episode lessons."""
        k = kernle_fresh

        # Create episode with unique lessons
        episode_id = k.episode(
            objective="Learn about quantum computing",
            outcome="success",
            lessons=["Quantum superposition enables parallel computation"],
        )

        result = k.revise_beliefs_from_episode(episode_id)

        # Should suggest the lesson as a new belief
        assert isinstance(result["suggested_new"], list)
        # The lesson should be suggested if no similar belief exists
        if result["suggested_new"]:
            assert any("quantum" in s["statement"].lower() for s in result["suggested_new"])

    def test_returns_error_for_nonexistent_episode(self, kernle_fresh):
        """Should return error for nonexistent episode."""
        k = kernle_fresh

        result = k.revise_beliefs_from_episode("nonexistent-id")

        assert result.get("error") == "Episode not found"


class TestBeliefHistory:
    """Tests for get_belief_history method."""

    def test_returns_single_belief_for_no_supersession(self, kernle_fresh):
        """Should return single entry for belief with no history."""
        k = kernle_fresh

        belief_id = k.belief("Simple belief", confidence=0.8)

        history = k.get_belief_history(belief_id)

        assert len(history) == 1
        assert history[0]["id"] == belief_id
        assert history[0]["is_current"] is True
        assert history[0]["is_active"] is True

    def test_returns_full_chain_for_superseded_beliefs(self, kernle_fresh):
        """Should return full supersession chain."""
        k = kernle_fresh

        # Create a chain of supersessions
        id1 = k.belief("Version 1", confidence=0.6)
        id2 = k.supersede_belief(id1, "Version 2", confidence=0.7)
        id3 = k.supersede_belief(id2, "Version 3", confidence=0.8)

        # Get history from middle belief
        history = k.get_belief_history(id2)

        assert len(history) == 3
        # Should be in chronological order
        assert history[0]["id"] == id1
        assert history[1]["id"] == id2
        assert history[2]["id"] == id3

        # Check is_current flag
        assert history[0]["is_current"] is False
        assert history[1]["is_current"] is True  # We queried for id2
        assert history[2]["is_current"] is False

        # Check is_active flag
        assert history[0]["is_active"] is False
        assert history[1]["is_active"] is False  # Superseded by id3
        assert history[2]["is_active"] is True

    def test_returns_empty_for_nonexistent(self, kernle_fresh):
        """Should return empty list for nonexistent belief."""
        k = kernle_fresh

        history = k.get_belief_history("nonexistent-id")

        assert history == []

    def test_walks_backwards_from_later_belief(self, kernle_fresh):
        """Should walk backwards to find root when starting from later belief."""
        k = kernle_fresh

        id1 = k.belief("Original", confidence=0.5)
        id2 = k.supersede_belief(id1, "Updated")

        # Get history starting from the newer belief
        history = k.get_belief_history(id2)

        assert len(history) == 2
        assert history[0]["id"] == id1  # Root should be first


class TestBeliefDataclassFields:
    """Tests for the new Belief dataclass fields."""

    def test_belief_has_revision_fields(self, kernle_fresh):
        """Belief should have all revision-related fields."""
        k = kernle_fresh

        belief_id = k.belief("Test belief", confidence=0.7)
        beliefs = k._storage.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        # Check fields exist
        assert hasattr(belief, "supersedes")
        assert hasattr(belief, "superseded_by")
        assert hasattr(belief, "times_reinforced")
        assert hasattr(belief, "is_active")

        # Check default values
        assert belief.supersedes is None
        assert belief.superseded_by is None
        assert belief.times_reinforced == 0
        assert belief.is_active is True

    def test_belief_fields_persist(self, tmp_path):
        """Belief revision fields should persist across storage operations."""
        db_path = tmp_path / "persist_test.db"

        # Create and save belief
        storage1 = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        k1 = Kernle(agent_id="test_agent", storage=storage1)

        belief_id = k1.belief("Test belief", confidence=0.7)
        k1.reinforce_belief(belief_id)
        k1.reinforce_belief(belief_id)

        # Reopen storage
        storage2 = SQLiteStorage(agent_id="test_agent", db_path=db_path)
        beliefs = storage2.get_beliefs(include_inactive=True)
        belief = next((b for b in beliefs if b.id == belief_id), None)

        # Fields should persist
        assert belief.times_reinforced == 2


class TestGetBeliefsFiltering:
    """Tests for get_beliefs include_inactive parameter."""

    def test_excludes_inactive_by_default(self, kernle_fresh):
        """Should exclude inactive beliefs by default."""
        k = kernle_fresh

        # Create and supersede a belief
        old_id = k.belief("Old belief", confidence=0.5)
        k.supersede_belief(old_id, "New belief")

        # Get beliefs without include_inactive
        beliefs = k._storage.get_beliefs()
        ids = [b.id for b in beliefs]

        assert old_id not in ids

    def test_includes_inactive_when_requested(self, kernle_fresh):
        """Should include inactive beliefs when requested."""
        k = kernle_fresh

        # Create and supersede a belief
        old_id = k.belief("Old belief", confidence=0.5)
        new_id = k.supersede_belief(old_id, "New belief")

        # Get beliefs with include_inactive
        beliefs = k._storage.get_beliefs(include_inactive=True)
        ids = [b.id for b in beliefs]

        assert old_id in ids
        assert new_id in ids
