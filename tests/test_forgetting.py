"""
Tests for controlled forgetting functionality.

Controlled forgetting is the mechanism by which low-salience memories
decay over time, but are tombstoned (not deleted) for potential recovery.
"""

from datetime import datetime, timedelta, timezone

from kernle.storage.base import Episode


class TestSalienceCalculation:
    """Test salience calculation for memories."""

    def test_salience_new_memory(self, kernle_instance):
        """New memory with no access should have moderate salience."""
        kernle, storage = kernle_instance

        # Create an episode
        ep_id = kernle.episode(
            objective="Test salience calculation",
            outcome="success",
        )

        # Calculate salience - should be based on confidence and age
        salience = kernle.calculate_salience("episode", ep_id)

        # New memory should have reasonable salience
        assert salience > 0
        assert salience < 1.0  # Not super high without reinforcement

    def test_salience_accessed_memory(self, kernle_instance):
        """Accessed memory should have higher salience."""
        kernle, storage = kernle_instance

        # Create an episode
        ep_id = kernle.episode(
            objective="Test accessed memory",
            outcome="success",
        )

        # Record multiple accesses
        for _ in range(5):
            kernle.record_access("episode", ep_id)

        # Calculate salience
        salience = kernle.calculate_salience("episode", ep_id)

        # Should be higher due to access reinforcement
        assert salience > 0.1  # Some reinforcement

    def test_salience_old_memory_decays(self, kernle_instance):
        """Old memories without access should have lower salience."""
        kernle, storage = kernle_instance

        # Create an old episode by manipulating created_at
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=90)  # 3 months ago

        episode = Episode(
            id="test-old-episode",
            agent_id=kernle.agent_id,
            objective="Old test episode",
            outcome="success",
            outcome_type="success",
            created_at=old_date,
            confidence=0.8,
            times_accessed=0,
        )
        storage.save_episode(episode)

        salience = kernle.calculate_salience("episode", "test-old-episode")

        # Old memory should have low salience
        assert salience < 0.3  # Below typical threshold

    def test_salience_returns_negative_for_missing(self, kernle_instance):
        """Should return -1.0 for missing memories."""
        kernle, _ = kernle_instance

        salience = kernle.calculate_salience("episode", "nonexistent-id")
        assert salience == -1.0


class TestForgettingCandidates:
    """Test finding memories eligible for forgetting."""

    def test_get_candidates_empty(self, kernle_instance):
        """Should return empty list when no low-salience memories."""
        kernle, _ = kernle_instance

        # Create high-confidence, recent memory
        kernle.episode(
            objective="High salience memory",
            outcome="success",
        )

        # Get candidates with very low threshold
        candidates = kernle.get_forgetting_candidates(threshold=0.01)

        # New memory shouldn't be a candidate
        assert len(candidates) == 0 or all(c["salience"] < 0.01 for c in candidates)

    def test_get_candidates_finds_old_memories(self, kernle_instance):
        """Should find old, unaccessed memories."""
        kernle, storage = kernle_instance

        # Create an old, low-confidence episode
        old_date = datetime.now(timezone.utc) - timedelta(days=120)

        episode = Episode(
            id="old-forgettable",
            agent_id=kernle.agent_id,
            objective="Old forgettable episode",
            outcome="meh",
            outcome_type="partial",
            created_at=old_date,
            confidence=0.3,  # Low confidence
            times_accessed=0,
        )
        storage.save_episode(episode)

        candidates = kernle.get_forgetting_candidates(threshold=0.5)

        # Should find our old episode
        candidate_ids = [c["id"] for c in candidates]
        assert "old-forgettable" in candidate_ids

    def test_candidates_exclude_protected(self, kernle_instance):
        """Protected memories should not be candidates."""
        kernle, storage = kernle_instance

        # Create old but protected episode
        old_date = datetime.now(timezone.utc) - timedelta(days=120)

        episode = Episode(
            id="protected-episode",
            agent_id=kernle.agent_id,
            objective="Protected episode",
            outcome="important",
            outcome_type="success",
            created_at=old_date,
            confidence=0.3,
            times_accessed=0,
            is_protected=True,
        )
        storage.save_episode(episode)

        candidates = kernle.get_forgetting_candidates(threshold=0.5)

        # Protected memory should not be in candidates
        candidate_ids = [c["id"] for c in candidates]
        assert "protected-episode" not in candidate_ids


class TestForgetting:
    """Test the actual forgetting process."""

    def test_forget_memory(self, kernle_instance):
        """Should tombstone a memory, not delete it."""
        kernle, storage = kernle_instance

        # Create episode
        ep_id = kernle.episode(
            objective="To be forgotten",
            outcome="partial",
        )

        # Verify it exists
        episode = storage.get_episode(ep_id)
        assert episode is not None
        assert not episode.is_forgotten

        # Forget it
        success = kernle.forget("episode", ep_id, reason="Test forgetting")
        assert success

        # Should still exist but be marked forgotten
        episode = storage.get_episode(ep_id)
        assert episode is not None
        assert episode.is_forgotten
        assert episode.forgotten_reason == "Test forgetting"
        assert episode.forgotten_at is not None

    def test_forget_protected_fails(self, kernle_instance):
        """Cannot forget protected memories."""
        kernle, storage = kernle_instance

        # Create and protect episode
        ep_id = kernle.episode(
            objective="Protected memory",
            outcome="success",
        )
        kernle.protect("episode", ep_id)

        # Try to forget
        success = kernle.forget("episode", ep_id)
        assert not success

        # Should not be forgotten
        episode = storage.get_episode(ep_id)
        assert not episode.is_forgotten

    def test_forget_already_forgotten(self, kernle_instance):
        """Forgetting already forgotten memory returns False."""
        kernle, _ = kernle_instance

        ep_id = kernle.episode(
            objective="Forget me twice",
            outcome="partial",
        )

        # First forget succeeds
        assert kernle.forget("episode", ep_id)

        # Second forget fails
        assert not kernle.forget("episode", ep_id)


class TestRecovery:
    """Test recovering forgotten memories."""

    def test_recover_memory(self, kernle_instance):
        """Should restore a forgotten memory."""
        kernle, storage = kernle_instance

        # Create and forget
        ep_id = kernle.episode(
            objective="Recover me",
            outcome="partial",
        )
        kernle.forget("episode", ep_id)

        # Verify forgotten
        episode = storage.get_episode(ep_id)
        assert episode.is_forgotten

        # Recover
        success = kernle.recover("episode", ep_id)
        assert success

        # Should no longer be forgotten
        episode = storage.get_episode(ep_id)
        assert not episode.is_forgotten
        assert episode.forgotten_at is None
        assert episode.forgotten_reason is None

    def test_recover_not_forgotten_fails(self, kernle_instance):
        """Cannot recover memory that isn't forgotten."""
        kernle, _ = kernle_instance

        ep_id = kernle.episode(
            objective="Not forgotten",
            outcome="success",
        )

        # Try to recover
        success = kernle.recover("episode", ep_id)
        assert not success

    def test_recover_nonexistent_fails(self, kernle_instance):
        """Cannot recover nonexistent memory."""
        kernle, _ = kernle_instance

        success = kernle.recover("episode", "nonexistent-id")
        assert not success


class TestProtection:
    """Test memory protection."""

    def test_protect_memory(self, kernle_instance):
        """Should mark memory as protected."""
        kernle, storage = kernle_instance

        ep_id = kernle.episode(
            objective="Protect me",
            outcome="success",
        )

        # Protect
        success = kernle.protect("episode", ep_id)
        assert success

        # Verify protected
        episode = storage.get_episode(ep_id)
        assert episode.is_protected

    def test_unprotect_memory(self, kernle_instance):
        """Should remove protection."""
        kernle, storage = kernle_instance

        ep_id = kernle.episode(
            objective="Unprotect me",
            outcome="success",
        )

        # Protect then unprotect
        kernle.protect("episode", ep_id)
        success = kernle.protect("episode", ep_id, protected=False)
        assert success

        # Should not be protected
        episode = storage.get_episode(ep_id)
        assert not episode.is_protected

    def test_values_protected_by_default(self, kernle_instance):
        """Values should be protected by default."""
        kernle, storage = kernle_instance

        value_id = kernle.value(
            name="Test Value",
            statement="This is important",
        )

        values = storage.get_values(limit=100)
        value = next((v for v in values if v.id == value_id), None)

        assert value is not None
        assert value.is_protected


class TestForgettingCycle:
    """Test the forgetting cycle operation."""

    def test_dry_run_does_not_forget(self, kernle_instance):
        """Dry run should not actually forget anything."""
        kernle, storage = kernle_instance

        # Create old, low-salience episode
        old_date = datetime.now(timezone.utc) - timedelta(days=120)

        episode = Episode(
            id="dry-run-test",
            agent_id=kernle.agent_id,
            objective="Dry run test",
            outcome="meh",
            outcome_type="partial",
            created_at=old_date,
            confidence=0.2,
            times_accessed=0,
        )
        storage.save_episode(episode)

        # Run dry cycle
        result = kernle.run_forgetting_cycle(threshold=0.5, dry_run=True)

        assert result["dry_run"] is True
        assert result["forgotten"] == 0

        # Episode should still not be forgotten
        episode = storage.get_episode("dry-run-test")
        assert not episode.is_forgotten

    def test_live_run_forgets(self, kernle_instance):
        """Live run should actually forget low-salience memories."""
        kernle, storage = kernle_instance

        # Create old, low-salience episode
        old_date = datetime.now(timezone.utc) - timedelta(days=120)

        episode = Episode(
            id="live-run-test",
            agent_id=kernle.agent_id,
            objective="Live run test",
            outcome="meh",
            outcome_type="partial",
            created_at=old_date,
            confidence=0.1,  # Very low confidence
            times_accessed=0,
        )
        storage.save_episode(episode)

        # Run live cycle with high threshold to catch it
        result = kernle.run_forgetting_cycle(threshold=0.5, dry_run=False, limit=10)

        assert result["dry_run"] is False
        assert result["forgotten"] >= 1

        # Episode should be forgotten
        episode = storage.get_episode("live-run-test")
        assert episode.is_forgotten


class TestAccessTracking:
    """Test memory access tracking for salience."""

    def test_record_access(self, kernle_instance):
        """Should track memory access."""
        kernle, storage = kernle_instance

        ep_id = kernle.episode(
            objective="Track my access",
            outcome="success",
        )

        # Record accesses
        kernle.record_access("episode", ep_id)
        kernle.record_access("episode", ep_id)

        # Verify access count
        episode = storage.get_episode(ep_id)
        assert episode.times_accessed == 2
        assert episode.last_accessed is not None

    def test_record_access_nonexistent(self, kernle_instance):
        """Recording access for nonexistent memory returns False."""
        kernle, _ = kernle_instance

        success = kernle.record_access("episode", "nonexistent-id")
        assert not success


class TestAutomaticAccessTracking:
    """Test that load() and search() automatically track access for salience."""

    def test_load_tracks_access_by_default(self, kernle_instance):
        """load() should record access for all loaded memories."""
        kernle, storage = kernle_instance

        # Create some memories
        ep_id = kernle.episode(objective="Test episode", outcome="success")
        _belief_id = kernle.belief(statement="Test belief")
        _note_id = kernle.note(content="Test note")

        # Verify initial access counts are 0
        ep = storage.get_episode(ep_id)
        assert ep.times_accessed == 0 or ep.times_accessed is None

        # Load working memory (should track access)
        kernle.load()

        # Verify access was tracked
        ep = storage.get_episode(ep_id)
        assert ep.times_accessed >= 1
        assert ep.last_accessed is not None

    def test_load_respects_track_access_false(self, kernle_instance):
        """load(track_access=False) should not record access."""
        kernle, storage = kernle_instance

        # Create an episode
        ep_id = kernle.episode(objective="No tracking episode", outcome="success")

        # Verify initial access count
        ep = storage.get_episode(ep_id)
        initial_access = ep.times_accessed or 0

        # Load without tracking
        kernle.load(track_access=False)

        # Access count should remain the same
        ep = storage.get_episode(ep_id)
        assert (ep.times_accessed or 0) == initial_access

    def test_search_tracks_access_by_default(self, kernle_instance):
        """search() should record access for returned results."""
        kernle, storage = kernle_instance

        # Create an episode with searchable content
        ep_id = kernle.episode(
            objective="Testing automatic access tracking feature",
            outcome="success",
        )

        # Verify initial access count
        ep = storage.get_episode(ep_id)
        initial_access = ep.times_accessed or 0

        # Search (should track access)
        kernle.search("automatic access tracking")

        # Verify access was tracked
        ep = storage.get_episode(ep_id)
        assert (ep.times_accessed or 0) > initial_access

    def test_search_respects_track_access_false(self, kernle_instance):
        """search(track_access=False) should not record access."""
        kernle, storage = kernle_instance

        # Create an episode
        ep_id = kernle.episode(
            objective="No tracking search test",
            outcome="success",
        )

        # Verify initial access count
        ep = storage.get_episode(ep_id)
        initial_access = ep.times_accessed or 0

        # Search without tracking
        kernle.search("No tracking search", track_access=False)

        # Access count should remain the same
        ep = storage.get_episode(ep_id)
        assert (ep.times_accessed or 0) == initial_access

    def test_record_access_batch(self, kernle_instance):
        """Batch access recording should update multiple memories efficiently."""
        kernle, storage = kernle_instance

        # Create multiple episodes
        ep_ids = [kernle.episode(objective=f"Batch test {i}", outcome="success") for i in range(3)]

        # Record batch access
        accesses = [("episode", ep_id) for ep_id in ep_ids]
        count = storage.record_access_batch(accesses)

        # Should have updated all episodes
        assert count == 3

        # Verify each episode was updated
        for ep_id in ep_ids:
            ep = storage.get_episode(ep_id)
            assert ep.times_accessed >= 1
            assert ep.last_accessed is not None

    def test_record_access_batch_mixed_types(self, kernle_instance):
        """Batch access recording should work with mixed memory types."""
        kernle, storage = kernle_instance

        # Create different memory types
        ep_id = kernle.episode(objective="Mixed batch test", outcome="success")
        belief_id = kernle.belief(statement="Mixed batch belief")
        note_id = kernle.note(content="Mixed batch note")

        # Record batch access
        accesses = [
            ("episode", ep_id),
            ("belief", belief_id),
            ("note", note_id),
        ]
        count = storage.record_access_batch(accesses)

        # Should have updated all memories
        assert count == 3

    def test_record_access_batch_empty(self, kernle_instance):
        """Batch access recording with empty list should return 0."""
        _, storage = kernle_instance

        count = storage.record_access_batch([])
        assert count == 0

    def test_record_access_batch_invalid_type(self, kernle_instance):
        """Batch access recording should skip invalid memory types."""
        kernle, storage = kernle_instance

        ep_id = kernle.episode(objective="Valid episode", outcome="success")

        # Mix valid and invalid types
        accesses = [
            ("episode", ep_id),
            ("invalid_type", "some-id"),
        ]
        count = storage.record_access_batch(accesses)

        # Should only count the valid one
        assert count == 1


class TestGetForgottenMemories:
    """Test listing forgotten memories."""

    def test_get_forgotten_empty(self, kernle_instance):
        """Should return empty list when no forgotten memories."""
        kernle, _ = kernle_instance

        forgotten = kernle.get_forgotten_memories()
        assert forgotten == []

    def test_get_forgotten_returns_tombstoned(self, kernle_instance):
        """Should return all forgotten memories."""
        kernle, _ = kernle_instance

        # Create and forget some episodes
        ep1 = kernle.episode(objective="Forget 1", outcome="partial")
        ep2 = kernle.episode(objective="Forget 2", outcome="partial")
        kernle.episode(objective="Keep", outcome="success")  # Not forgotten

        kernle.forget("episode", ep1, reason="Reason 1")
        kernle.forget("episode", ep2, reason="Reason 2")

        forgotten = kernle.get_forgotten_memories()

        assert len(forgotten) == 2
        forgotten_ids = [f["id"] for f in forgotten]
        assert ep1 in forgotten_ids
        assert ep2 in forgotten_ids


class TestMultipleMemoryTypes:
    """Test forgetting works across different memory types."""

    def test_forget_belief(self, kernle_instance):
        """Should be able to forget beliefs."""
        kernle, storage = kernle_instance

        belief_id = kernle.belief(
            statement="Test belief",
            confidence=0.3,
        )

        success = kernle.forget("belief", belief_id)
        assert success

        # Verify in forgotten list
        forgotten = kernle.get_forgotten_memories(memory_types=["belief"])
        assert any(f["id"] == belief_id for f in forgotten)

    def test_forget_note(self, kernle_instance):
        """Should be able to forget notes."""
        kernle, storage = kernle_instance

        note_id = kernle.note(content="Test note")

        success = kernle.forget("note", note_id)
        assert success

    def test_forget_goal(self, kernle_instance):
        """Should be able to forget goals."""
        kernle, storage = kernle_instance

        goal_id = kernle.goal(title="Test goal")

        success = kernle.forget("goal", goal_id)
        assert success


# Note: Fixtures (kernle_instance, temp_db_path, temp_checkpoint_dir) are
# defined in conftest.py and shared across all test files.
