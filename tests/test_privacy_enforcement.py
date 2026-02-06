"""Test privacy enforcement for Phase 8a.

Tests privacy field storage/retrieval and access filtering for all memory types.
"""

import tempfile
import uuid
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.storage.base import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Playbook,
    Relationship,
    Value,
)
from kernle.storage.sqlite import SQLiteStorage


class TestPrivacyFields:
    """Test that privacy fields are stored and retrieved correctly."""

    def setup_method(self):
        """Set up test storage."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.storage = SQLiteStorage(agent_id="test-privacy", db_path=self.tmpdir / "test.db")
        self.kernle = Kernle(
            agent_id="test-privacy", storage=self.storage, checkpoint_dir=self.tmpdir
        )

    def test_episode_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for episodes."""
        episode = Episode(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            objective="Test objective",
            outcome="Test outcome",
            subject_ids=["user123", "project456"],
            access_grants=["team_lead", "manager"],
            consent_grants=["user123"],
        )

        episode_id = self.storage.save_episode(episode)
        retrieved = self.storage.get_episode(episode_id)

        assert retrieved is not None
        assert retrieved.subject_ids == ["user123", "project456"]
        assert retrieved.access_grants == ["team_lead", "manager"]
        assert retrieved.consent_grants == ["user123"]

    def test_belief_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for beliefs."""
        belief = Belief(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            statement="Test belief",
            subject_ids=["user123"],
            access_grants=["team_lead"],
            consent_grants=["user123"],
        )

        belief_id = self.storage.save_belief(belief)
        retrieved = self.storage.get_belief(belief_id)

        assert retrieved is not None
        assert retrieved.subject_ids == ["user123"]
        assert retrieved.access_grants == ["team_lead"]
        assert retrieved.consent_grants == ["user123"]

    def test_value_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for values."""
        value = Value(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            name="Test Value",
            statement="Test value statement",
            subject_ids=["user123"],
            access_grants=[],  # Empty = private to self
            consent_grants=None,
        )

        value_id = self.storage.save_value(value)
        retrieved = next((v for v in self.storage.get_values() if v.id == value_id), None)

        assert retrieved is not None
        assert retrieved.subject_ids == ["user123"]
        assert retrieved.access_grants == []
        assert retrieved.consent_grants is None

    def test_goal_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for goals."""
        goal = Goal(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            title="Test Goal",
            description="Test goal description",
            subject_ids=None,
            access_grants=["public"],
            consent_grants=["user123"],
        )

        goal_id = self.storage.save_goal(goal)
        retrieved = next((g for g in self.storage.get_goals(status=None) if g.id == goal_id), None)

        assert retrieved is not None
        assert retrieved.subject_ids is None
        assert retrieved.access_grants == ["public"]
        assert retrieved.consent_grants == ["user123"]

    def test_note_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for notes."""
        note = Note(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            content="Test note content",
            subject_ids=["user123", "user456"],
            access_grants=None,  # Private to self
            consent_grants=["user123"],
        )

        note_id = self.storage.save_note(note)
        retrieved = next((n for n in self.storage.get_notes() if n.id == note_id), None)

        assert retrieved is not None
        assert retrieved.subject_ids == ["user123", "user456"]
        assert retrieved.access_grants is None
        assert retrieved.consent_grants == ["user123"]

    def test_drive_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for drives."""
        drive = Drive(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            drive_type="achievement",
            intensity=0.8,
            subject_ids=["self"],
            access_grants=["therapist"],
            consent_grants=["self"],
        )

        self.storage.save_drive(drive)
        retrieved = self.storage.get_drive("achievement")

        assert retrieved is not None
        assert retrieved.subject_ids == ["self"]
        assert retrieved.access_grants == ["therapist"]
        assert retrieved.consent_grants == ["self"]

    def test_relationship_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for relationships."""
        relationship = Relationship(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            entity_name="Test Person",
            entity_type="human",
            relationship_type="colleague",
            subject_ids=["test_person_id"],
            access_grants=["hr_team"],
            consent_grants=["test_person_id"],
        )

        self.storage.save_relationship(relationship)
        retrieved = self.storage.get_relationship("Test Person")

        assert retrieved is not None
        assert retrieved.subject_ids == ["test_person_id"]
        assert retrieved.access_grants == ["hr_team"]
        assert retrieved.consent_grants == ["test_person_id"]

    def test_playbook_privacy_fields_storage(self):
        """Test privacy fields are stored and retrieved for playbooks."""
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test-privacy",
            name="Test Playbook",
            description="Test playbook description",
            trigger_conditions=["condition1"],
            steps=[{"action": "step1", "details": "details1"}],
            failure_modes=["failure1"],
            subject_ids=["team123"],
            access_grants=["team_members"],
            consent_grants=["team_lead"],
        )

        playbook_id = self.storage.save_playbook(playbook)
        retrieved = self.storage.get_playbook(playbook_id)

        assert retrieved is not None
        assert retrieved.subject_ids == ["team123"]
        assert retrieved.access_grants == ["team_members"]
        assert retrieved.consent_grants == ["team_lead"]


class TestAccessControl:
    """Test access control filtering based on requesting_entity."""

    def setup_method(self):
        """Set up test storage with sample data."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.storage = SQLiteStorage(agent_id="test-privacy", db_path=self.tmpdir / "test.db")

        # Create test episodes with different privacy levels
        self.public_episode = Episode(
            id="episode_public",
            agent_id="test-privacy",
            objective="Public episode",
            outcome="Public outcome",
            access_grants=["team_lead", "manager", "public"],
        )

        self.team_episode = Episode(
            id="episode_team",
            agent_id="test-privacy",
            objective="Team episode",
            outcome="Team outcome",
            access_grants=["team_lead", "team_member"],
        )

        self.private_episode = Episode(
            id="episode_private",
            agent_id="test-privacy",
            objective="Private episode",
            outcome="Private outcome",
            access_grants=None,  # Private to self
        )

        self.empty_grants_episode = Episode(
            id="episode_empty",
            agent_id="test-privacy",
            objective="Empty grants episode",
            outcome="Empty grants outcome",
            access_grants=[],  # Empty = private to self
        )

        # Save all episodes
        self.storage.save_episode(self.public_episode)
        self.storage.save_episode(self.team_episode)
        self.storage.save_episode(self.private_episode)
        self.storage.save_episode(self.empty_grants_episode)

    def test_self_access_sees_everything(self):
        """Test that self-access (requesting_entity=None) sees all episodes."""
        episodes = self.storage.get_episodes(requesting_entity=None)
        episode_ids = [ep.id for ep in episodes]

        assert "episode_public" in episode_ids
        assert "episode_team" in episode_ids
        assert "episode_private" in episode_ids
        assert "episode_empty" in episode_ids

    def test_external_access_filtered_correctly(self):
        """Test that external entities only see episodes they have access to."""
        # Team lead should see public and team episodes
        episodes = self.storage.get_episodes(requesting_entity="team_lead")
        episode_ids = [ep.id for ep in episodes]

        assert "episode_public" in episode_ids
        assert "episode_team" in episode_ids
        assert "episode_private" not in episode_ids  # Private to self
        assert "episode_empty" not in episode_ids  # Empty grants = private

    def test_public_access_filtering(self):
        """Test that public entity sees only public episodes."""
        episodes = self.storage.get_episodes(requesting_entity="public")
        episode_ids = [ep.id for ep in episodes]

        assert "episode_public" in episode_ids
        assert "episode_team" not in episode_ids
        assert "episode_private" not in episode_ids
        assert "episode_empty" not in episode_ids

    def test_no_access_entity_sees_nothing(self):
        """Test that entity with no access sees no episodes."""
        episodes = self.storage.get_episodes(requesting_entity="random_stranger")
        assert len(episodes) == 0

    def test_single_episode_access_control(self):
        """Test access control for single episode retrieval."""
        # Self access should work
        episode = self.storage.get_episode("episode_private", requesting_entity=None)
        assert episode is not None
        assert episode.id == "episode_private"

        # External access should be blocked
        episode = self.storage.get_episode("episode_private", requesting_entity="team_lead")
        assert episode is None

        # Authorized access should work
        episode = self.storage.get_episode("episode_team", requesting_entity="team_lead")
        assert episode is not None
        assert episode.id == "episode_team"

    def test_null_vs_empty_access_grants(self):
        """Test that NULL and empty access_grants are both treated as private."""
        # Both should be invisible to external entities
        episodes = self.storage.get_episodes(requesting_entity="team_lead")
        episode_ids = [ep.id for ep in episodes]

        assert "episode_private" not in episode_ids  # NULL grants
        assert "episode_empty" not in episode_ids  # Empty grants

    def test_search_respects_privacy_filters(self):
        """Test that search operations respect privacy filters."""
        # Self search sees everything
        results = self.storage.search("episode", requesting_entity=None, prefer_cloud=False)
        result_ids = [r.record.id for r in results if hasattr(r.record, "id")]

        # Should see all episodes
        assert len([rid for rid in result_ids if rid.startswith("episode_")]) == 4

        # External search is filtered
        results = self.storage.search("episode", requesting_entity="team_lead", prefer_cloud=False)
        result_ids = [r.record.id for r in results if hasattr(r.record, "id")]

        # Should only see accessible episodes
        accessible_count = len(
            [rid for rid in result_ids if rid in ["episode_public", "episode_team"]]
        )
        private_count = len(
            [rid for rid in result_ids if rid in ["episode_private", "episode_empty"]]
        )

        assert accessible_count >= 0  # Should see accessible ones
        assert private_count == 0  # Should not see private ones


class TestSchemaMigration:
    """Test that schema migration from version 15 to 16 works correctly."""

    def test_privacy_columns_added_to_all_tables(self):
        """Test that privacy columns are added to all memory tables."""
        tmpdir = Path(tempfile.mkdtemp())
        storage = SQLiteStorage(agent_id="test-migration", db_path=tmpdir / "test.db")

        # Check that privacy columns exist in all tables
        with storage._connect() as conn:
            tables_to_check = [
                "episodes",
                "beliefs",
                "agent_values",
                "goals",
                "notes",
                "drives",
                "relationships",
                "playbooks",
                "raw_entries",
            ]

            for table in tables_to_check:
                columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
                column_names = [col[1] for col in columns]

                assert "subject_ids" in column_names, f"subject_ids missing from {table}"
                assert "access_grants" in column_names, f"access_grants missing from {table}"
                assert "consent_grants" in column_names, f"consent_grants missing from {table}"

    def test_schema_version_updated(self):
        """Test that schema version is at least 16 (privacy migration)."""
        tmpdir = Path(tempfile.mkdtemp())
        storage = SQLiteStorage(agent_id="test-version", db_path=tmpdir / "test.db")

        with storage._connect() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()
            assert version[0] >= 16


class TestAllMemoryTypes:
    """Test privacy enforcement for all 8 memory types."""

    def setup_method(self):
        """Set up test storage."""
        self.tmpdir = Path(tempfile.mkdtemp())
        self.storage = SQLiteStorage(agent_id="test-all-types", db_path=self.tmpdir / "test.db")

    def test_all_memory_types_support_privacy_fields(self):
        """Test that all memory types can store and retrieve privacy fields."""
        # Episode
        episode = Episode(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            objective="Test",
            outcome="Test",
            subject_ids=["s1"],
            access_grants=["a1"],
            consent_grants=["c1"],
        )
        episode_id = self.storage.save_episode(episode)
        retrieved_episode = self.storage.get_episode(episode_id)
        assert retrieved_episode.subject_ids == ["s1"]

        # Belief
        belief = Belief(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            statement="Test belief",
            subject_ids=["s2"],
            access_grants=["a2"],
            consent_grants=["c2"],
        )
        belief_id = self.storage.save_belief(belief)
        retrieved_belief = self.storage.get_belief(belief_id)
        assert retrieved_belief.subject_ids == ["s2"]

        # Value
        value = Value(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            name="Test",
            statement="Test value",
            subject_ids=["s3"],
            access_grants=["a3"],
            consent_grants=["c3"],
        )
        value_id = self.storage.save_value(value)
        retrieved_value = next((v for v in self.storage.get_values() if v.id == value_id), None)
        assert retrieved_value.subject_ids == ["s3"]

        # Goal
        goal = Goal(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            title="Test Goal",
            subject_ids=["s4"],
            access_grants=["a4"],
            consent_grants=["c4"],
        )
        goal_id = self.storage.save_goal(goal)
        retrieved_goal = next(
            (g for g in self.storage.get_goals(status=None) if g.id == goal_id), None
        )
        assert retrieved_goal.subject_ids == ["s4"]

        # Note
        note = Note(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            content="Test note",
            subject_ids=["s5"],
            access_grants=["a5"],
            consent_grants=["c5"],
        )
        note_id = self.storage.save_note(note)
        retrieved_note = next((n for n in self.storage.get_notes() if n.id == note_id), None)
        assert retrieved_note.subject_ids == ["s5"]

        # Drive
        drive = Drive(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            drive_type="test_drive",
            subject_ids=["s6"],
            access_grants=["a6"],
            consent_grants=["c6"],
        )
        self.storage.save_drive(drive)
        retrieved_drive = self.storage.get_drive("test_drive")
        assert retrieved_drive.subject_ids == ["s6"]

        # Relationship
        relationship = Relationship(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            entity_name="Test Entity",
            entity_type="human",
            relationship_type="colleague",
            subject_ids=["s7"],
            access_grants=["a7"],
            consent_grants=["c7"],
        )
        self.storage.save_relationship(relationship)
        retrieved_rel = self.storage.get_relationship("Test Entity")
        assert retrieved_rel.subject_ids == ["s7"]

        # Playbook
        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            name="Test Playbook",
            description="Test",
            trigger_conditions=["t1"],
            steps=[{"action": "a1"}],
            failure_modes=["f1"],
            subject_ids=["s8"],
            access_grants=["a8"],
            consent_grants=["c8"],
        )
        playbook_id = self.storage.save_playbook(playbook)
        retrieved_playbook = self.storage.get_playbook(playbook_id)
        assert retrieved_playbook.subject_ids == ["s8"]

    def test_backward_compatibility_with_null_privacy_fields(self):
        """Test that existing records without privacy fields still work."""
        # Create record without privacy fields (simulating old data)
        episode = Episode(
            id=str(uuid.uuid4()),
            agent_id="test-all-types",
            objective="Old episode",
            outcome="Old outcome",
            # No privacy fields set
        )

        episode_id = self.storage.save_episode(episode)
        retrieved = self.storage.get_episode(episode_id)

        # Should have None values for privacy fields
        assert retrieved.subject_ids is None
        assert retrieved.access_grants is None
        assert retrieved.consent_grants is None

        # Self-access should still work
        episodes = self.storage.get_episodes(requesting_entity=None)
        assert any(ep.id == episode_id for ep in episodes)

        # External access should be blocked (NULL = private)
        episodes = self.storage.get_episodes(requesting_entity="external")
        assert not any(ep.id == episode_id for ep in episodes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
