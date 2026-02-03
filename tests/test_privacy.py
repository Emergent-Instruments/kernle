"""Tests for Phase 8a privacy field enforcement.

Tests access_grants filtering at query time, context management,
and privacy-aware memory creation.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.storage import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    Relationship,
    SQLiteStorage,
    Value,
)


@pytest.fixture
def temp_db():
    """Create a temporary database path."""
    path = Path(tempfile.mktemp(suffix=".db"))
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def storage(temp_db):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    yield s
    s.close()


@pytest.fixture
def kernle_instance(temp_db):
    """Create a Kernle instance with SQLiteStorage for testing."""
    s = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    checkpoint_dir = Path(tempfile.mkdtemp())
    k = Kernle(agent_id="test-agent", storage=s, checkpoint_dir=checkpoint_dir)
    yield k
    s.close()


def _make_episode(storage, episode_id, access_grants=None, objective="Test episode"):
    """Helper to create an episode with specific access_grants."""
    ep = Episode(
        id=episode_id,
        agent_id="test-agent",
        objective=objective,
        outcome="Test outcome",
        created_at=datetime.now(timezone.utc),
        access_grants=access_grants,
    )
    storage.save_episode(ep)
    return ep


def _make_belief(storage, belief_id, access_grants=None, statement="Test belief"):
    """Helper to create a belief with specific access_grants."""
    b = Belief(
        id=belief_id,
        agent_id="test-agent",
        statement=statement,
        created_at=datetime.now(timezone.utc),
        access_grants=access_grants,
    )
    storage.save_belief(b)
    return b


def _make_note(storage, note_id, access_grants=None, content="Test note"):
    """Helper to create a note with specific access_grants."""
    n = Note(
        id=note_id,
        agent_id="test-agent",
        content=content,
        created_at=datetime.now(timezone.utc),
        access_grants=access_grants,
    )
    storage.save_note(n)
    return n


def _make_value(storage, value_id, access_grants=None, name="test-value"):
    """Helper to create a value with specific access_grants."""
    v = Value(
        id=value_id,
        agent_id="test-agent",
        name=name,
        statement="Test value statement",
        created_at=datetime.now(timezone.utc),
        access_grants=access_grants,
    )
    storage.save_value(v)
    return v


def _make_goal(storage, goal_id, access_grants=None, title="Test goal"):
    """Helper to create a goal with specific access_grants."""
    g = Goal(
        id=goal_id,
        agent_id="test-agent",
        title=title,
        status="active",
        created_at=datetime.now(timezone.utc),
        access_grants=access_grants,
    )
    storage.save_goal(g)
    return g


class TestAccessGrantsFiltering:
    """Test that access_grants filtering works correctly at query time."""

    def test_no_requesting_entity_sees_everything(self, storage):
        """Backward compatibility: no requesting_entity = see all."""
        _make_episode(storage, "ep-private", access_grants=[])
        _make_episode(storage, "ep-public", access_grants=["*"])
        _make_episode(storage, "ep-granted", access_grants=["si:other-agent"])

        # No requesting_entity = see everything (backward compatible)
        episodes = storage.get_episodes()
        assert len(episodes) == 3

    def test_empty_access_grants_private_to_self(self, storage):
        """Empty access_grants = private to self only."""
        _make_episode(storage, "ep-private", access_grants=[])
        _make_episode(storage, "ep-null", access_grants=None)

        # Self-access (si:test-agent) sees private memories
        episodes = storage.get_episodes(requesting_entity="si:test-agent")
        assert len(episodes) == 2

        # Other entity cannot see private memories
        episodes = storage.get_episodes(requesting_entity="si:other-agent")
        assert len(episodes) == 0

    def test_public_access_grants(self, storage):
        """access_grants containing "*" = visible to everyone."""
        _make_episode(storage, "ep-public", access_grants=["*"])
        _make_episode(storage, "ep-private", access_grants=[])

        # Any entity sees public memories
        episodes = storage.get_episodes(requesting_entity="si:random-entity")
        assert len(episodes) == 1
        assert episodes[0].id == "ep-public"

    def test_specific_entity_grant(self, storage):
        """access_grants containing specific entity = visible to that entity."""
        _make_episode(storage, "ep-granted", access_grants=["si:friend-agent"])
        _make_episode(storage, "ep-private", access_grants=[])

        # Granted entity sees the memory
        episodes = storage.get_episodes(requesting_entity="si:friend-agent")
        assert len(episodes) == 1
        assert episodes[0].id == "ep-granted"

        # Non-granted entity does not
        episodes = storage.get_episodes(requesting_entity="si:stranger")
        assert len(episodes) == 0

    def test_multiple_grants(self, storage):
        """access_grants with multiple entities."""
        _make_episode(storage, "ep-multi", access_grants=["si:agent-a", "si:agent-b", "human:sean"])

        assert len(storage.get_episodes(requesting_entity="si:agent-a")) == 1
        assert len(storage.get_episodes(requesting_entity="si:agent-b")) == 1
        assert len(storage.get_episodes(requesting_entity="human:sean")) == 1
        assert len(storage.get_episodes(requesting_entity="si:agent-c")) == 0

    def test_context_grant(self, storage):
        """access_grants containing context ID."""
        _make_episode(storage, "ep-ctx", access_grants=["ctx:bella_health"])

        assert len(storage.get_episodes(requesting_entity="ctx:bella_health")) == 1
        assert len(storage.get_episodes(requesting_entity="ctx:other")) == 0

    def test_mixed_grants(self, storage):
        """Multiple memories with different grant levels."""
        _make_episode(storage, "ep-private", access_grants=[])
        _make_episode(storage, "ep-public", access_grants=["*"])
        _make_episode(storage, "ep-sean", access_grants=["human:sean"])
        _make_episode(storage, "ep-ctx", access_grants=["ctx:work"])

        # human:sean sees public + their own grant
        episodes = storage.get_episodes(requesting_entity="human:sean")
        ids = {e.id for e in episodes}
        assert ids == {"ep-public", "ep-sean"}

        # Self sees private + all others
        episodes = storage.get_episodes(requesting_entity="si:test-agent")
        ids = {e.id for e in episodes}
        assert ids == {"ep-private", "ep-public", "ep-sean", "ep-ctx"}


class TestAccessGrantsAllMemoryTypes:
    """Test that access_grants filtering works for all memory types."""

    def test_beliefs_filtering(self, storage):
        _make_belief(storage, "b-private", access_grants=[])
        _make_belief(storage, "b-public", access_grants=["*"])
        _make_belief(storage, "b-granted", access_grants=["si:friend"])

        assert len(storage.get_beliefs(requesting_entity="si:friend")) == 2  # public + granted
        assert len(storage.get_beliefs(requesting_entity="si:stranger")) == 1  # public only
        assert len(storage.get_beliefs()) == 3  # backward compat

    def test_values_filtering(self, storage):
        _make_value(storage, "v-private", access_grants=[])
        _make_value(storage, "v-public", access_grants=["*"], name="pub-value")

        assert len(storage.get_values(requesting_entity="si:other")) == 1
        assert len(storage.get_values()) == 2

    def test_goals_filtering(self, storage):
        _make_goal(storage, "g-private", access_grants=[])
        _make_goal(storage, "g-public", access_grants=["*"])

        assert len(storage.get_goals(requesting_entity="si:other")) == 1
        assert len(storage.get_goals()) == 2

    def test_notes_filtering(self, storage):
        _make_note(storage, "n-private", access_grants=[])
        _make_note(storage, "n-public", access_grants=["*"])
        _make_note(storage, "n-friend", access_grants=["si:friend"])

        assert len(storage.get_notes(requesting_entity="si:friend")) == 2
        assert len(storage.get_notes(requesting_entity="si:other")) == 1
        assert len(storage.get_notes()) == 3

    def test_drives_filtering(self, storage):
        d1 = Drive(
            id="d-private", agent_id="test-agent", drive_type="curiosity",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
            access_grants=[],
        )
        d2 = Drive(
            id="d-public", agent_id="test-agent", drive_type="growth",
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
            access_grants=["*"],
        )
        storage.save_drive(d1)
        storage.save_drive(d2)

        assert len(storage.get_drives(requesting_entity="si:other")) == 1
        assert len(storage.get_drives()) == 2

    def test_relationships_filtering(self, storage):
        r1 = Relationship(
            id="r-private", agent_id="test-agent", entity_name="secret-friend",
            entity_type="human", relationship_type="friend",
            created_at=datetime.now(timezone.utc), access_grants=[],
        )
        r2 = Relationship(
            id="r-public", agent_id="test-agent", entity_name="public-friend",
            entity_type="human", relationship_type="colleague",
            created_at=datetime.now(timezone.utc), access_grants=["*"],
        )
        storage.save_relationship(r1)
        storage.save_relationship(r2)

        assert len(storage.get_relationships(requesting_entity="si:other")) == 1
        assert len(storage.get_relationships()) == 2


class TestSearchAccessGrants:
    """Test that search respects access_grants filtering."""

    def test_text_search_respects_access_grants(self, storage):
        """Text search should filter by access_grants."""
        _make_note(storage, "n-private", access_grants=[], content="secret health data")
        _make_note(storage, "n-public", access_grants=["*"], content="public health info")
        _make_note(storage, "n-friend", access_grants=["si:friend"], content="shared health note")

        # No requesting_entity = see all
        results = storage.search("health", prefer_cloud=False)
        assert len(results) == 3

        # Friend sees public + granted
        results = storage.search("health", prefer_cloud=False, requesting_entity="si:friend")
        assert len(results) == 2
        ids = {r.record.id for r in results}
        assert ids == {"n-public", "n-friend"}

        # Stranger sees only public
        results = storage.search("health", prefer_cloud=False, requesting_entity="si:stranger")
        assert len(results) == 1
        assert results[0].record.id == "n-public"

    def test_text_search_beliefs_access(self, storage):
        """Search across beliefs respects access_grants."""
        _make_belief(storage, "b-secret", access_grants=[], statement="I secretly love cats")
        _make_belief(storage, "b-public", access_grants=["*"], statement="I love dogs publicly")

        results = storage.search("love", prefer_cloud=False, requesting_entity="si:other")
        assert len(results) == 1
        assert results[0].record.id == "b-public"


class TestContextManagement:
    """Test Kernle context management for privacy."""

    def test_enter_exit_context(self, kernle_instance):
        """Test entering and exiting a context."""
        k = kernle_instance

        # No context initially
        assert k.current_context is None

        # Enter context
        ctx = k.enter_context(
            "ctx:bella_health",
            participants=["human:sean", "si:bella_agent"],
            role="role:care_agent",
        )
        assert k.current_context is not None
        assert k.current_context["context_id"] == "ctx:bella_health"
        assert "human:sean" in k.current_context["participants"]
        assert k.current_context["role"] == "role:care_agent"

        # Exit context
        k.exit_context()
        assert k.current_context is None

    def test_context_auto_inherits_access_grants(self, kernle_instance):
        """Memories created in context should auto-inherit access_grants."""
        k = kernle_instance

        # Create memory without context — no access_grants
        ep_id_before = k.episode(
            objective="No context memory",
            outcome="Created without context",
        )
        ep_before = k.storage.get_episode(ep_id_before)
        assert ep_before.access_grants is None

        # Enter context
        k.enter_context(
            "ctx:test_project",
            participants=["human:alice", "si:helper"],
        )

        # Create memory in context — should inherit participants + context
        ep_id_in = k.episode(
            objective="In-context memory",
            outcome="Created with context",
        )
        ep_in = k.storage.get_episode(ep_id_in)
        assert ep_in.access_grants is not None
        assert "human:alice" in ep_in.access_grants
        assert "si:helper" in ep_in.access_grants
        assert "ctx:test_project" in ep_in.access_grants
        assert "si:test-agent" in ep_in.access_grants  # self always included

        # Exit context
        k.exit_context()

        # Memory created after context — no access_grants again
        ep_id_after = k.episode(
            objective="After context memory",
            outcome="Created after exit",
        )
        ep_after = k.storage.get_episode(ep_id_after)
        assert ep_after.access_grants is None

    def test_note_inherits_context_access_grants(self, kernle_instance):
        """Notes created in context should also inherit access_grants."""
        k = kernle_instance

        k.enter_context(
            "ctx:meeting",
            participants=["human:bob"],
        )

        note_id = k.note(content="Meeting discussion point")
        note = k.storage.get_notes()[0]
        assert note.access_grants is not None
        assert "human:bob" in note.access_grants
        assert "ctx:meeting" in note.access_grants

        k.exit_context()


class TestAccessGrantsPersistence:
    """Test that access_grants are correctly persisted and retrieved."""

    def test_episode_access_grants_round_trip(self, storage):
        """access_grants survive save/load cycle."""
        grants = ["human:sean", "si:friend", "ctx:work"]
        _make_episode(storage, "ep-roundtrip", access_grants=grants)

        ep = storage.get_episode("ep-roundtrip")
        assert ep.access_grants == grants

    def test_belief_privacy_fields_round_trip(self, storage):
        """All privacy fields survive save/load cycle."""
        b = Belief(
            id="b-full-privacy",
            agent_id="test-agent",
            statement="Private belief",
            created_at=datetime.now(timezone.utc),
            subject_ids=["human:student_123"],
            access_grants=["human:parent", "ctx:academic"],
            consent_grants=["human:parent"],
        )
        storage.save_belief(b)

        loaded = storage.get_beliefs()
        found = [x for x in loaded if x.id == "b-full-privacy"][0]
        assert found.subject_ids == ["human:student_123"]
        assert found.access_grants == ["human:parent", "ctx:academic"]
        assert found.consent_grants == ["human:parent"]

    def test_note_privacy_fields_round_trip(self, storage):
        """Note privacy fields survive save/load."""
        n = Note(
            id="n-privacy",
            agent_id="test-agent",
            content="Private note",
            created_at=datetime.now(timezone.utc),
            subject_ids=["dog:bella"],
            access_grants=["human:sean", "*"],
            consent_grants=["human:sean"],
        )
        storage.save_note(n)

        loaded = storage.get_notes()
        found = [x for x in loaded if x.id == "n-privacy"][0]
        assert found.subject_ids == ["dog:bella"]
        assert found.access_grants == ["human:sean", "*"]
        assert found.consent_grants == ["human:sean"]


class TestBackwardCompatibility:
    """Ensure backward compatibility is maintained."""

    def test_existing_memories_visible_without_requesting_entity(self, storage):
        """Memories without access_grants are visible when no requesting_entity."""
        # Simulate pre-Phase-8a memories (no privacy fields set)
        ep = Episode(
            id="ep-legacy",
            agent_id="test-agent",
            objective="Legacy episode",
            outcome="From before privacy",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep)

        # No requesting_entity = see everything
        episodes = storage.get_episodes()
        assert len(episodes) == 1

    def test_self_access_sees_legacy_memories(self, storage):
        """Self-access (si:agent_id) sees legacy memories with no grants."""
        ep = Episode(
            id="ep-legacy",
            agent_id="test-agent",
            objective="Legacy episode",
            outcome="From before privacy",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep)

        # Self sees them (empty access_grants = private to self)
        episodes = storage.get_episodes(requesting_entity="si:test-agent")
        assert len(episodes) == 1

    def test_other_entity_cannot_see_legacy_memories(self, storage):
        """Other entities cannot see legacy memories (private by default)."""
        ep = Episode(
            id="ep-legacy",
            agent_id="test-agent",
            objective="Legacy episode",
            outcome="From before privacy",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep)

        # Other entity cannot see legacy memories
        episodes = storage.get_episodes(requesting_entity="si:other")
        assert len(episodes) == 0

    def test_storage_methods_work_without_requesting_entity(self, storage):
        """All get methods work without requesting_entity (backward compat)."""
        _make_episode(storage, "ep1")
        _make_belief(storage, "b1")
        _make_note(storage, "n1")
        _make_value(storage, "v1")
        _make_goal(storage, "g1")

        # All should return results without requesting_entity
        assert len(storage.get_episodes()) >= 1
        assert len(storage.get_beliefs()) >= 1
        assert len(storage.get_notes()) >= 1
        assert len(storage.get_values()) >= 1
        assert len(storage.get_goals()) >= 1
        assert len(storage.get_drives()) >= 0
        assert len(storage.get_relationships()) >= 0
