"""Tests for stack lifecycle states and provenance validation.

Tests cover:
- Stack lifecycle state transitions (INITIALIZING -> ACTIVE -> MAINTENANCE)
- Provenance enforcement when enforce_provenance=True
- Provenance bypass when enforce_provenance=False (default)
- Hierarchy rules: each type must cite allowed source types
- ProvenanceError for missing, malformed, or invalid references
- MaintenanceModeError during maintenance mode
- memory_exists() in SQLiteStorage
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.protocols import (
    MaintenanceModeError,
    ProvenanceError,
    StackState,
)
from kernle.stack import SQLiteStack
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Goal,
    Note,
    RawEntry,
    Relationship,
    Value,
)


def _uid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_provenance.db"


@pytest.fixture
def stack(tmp_db):
    """Stack with enforce_provenance=False (default), bare components."""
    return SQLiteStack(stack_id="test-stack", db_path=tmp_db, components=[])


@pytest.fixture
def enforced_stack(tmp_db):
    """Stack with enforce_provenance=True, bare components."""
    return SQLiteStack(
        stack_id="test-stack",
        db_path=tmp_db,
        components=[],
        enforce_provenance=True,
    )


# ---- Stack Lifecycle State Tests ----


class TestStackLifecycle:
    def test_initial_state_is_initializing(self, stack):
        assert stack.state == StackState.INITIALIZING

    def test_on_attach_transitions_to_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        assert stack.state == StackState.ACTIVE

    def test_enter_maintenance_from_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        assert stack.state == StackState.MAINTENANCE

    def test_exit_maintenance_returns_to_active(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.exit_maintenance()
        assert stack.state == StackState.ACTIVE

    def test_enter_maintenance_idempotent(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.enter_maintenance()
        stack.enter_maintenance()  # Should not raise
        assert stack.state == StackState.MAINTENANCE

    def test_exit_maintenance_noop_when_not_in_maintenance(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        stack.exit_maintenance()  # Should not raise or change state
        assert stack.state == StackState.ACTIVE

    def test_writes_allowed_in_initializing(self, enforced_stack):
        """Even with provenance enforcement, writes are allowed in INITIALIZING."""
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Seed objective",
            outcome="Seed outcome",
            source_type="seed",
            created_at=_now(),
        )
        ep_id = enforced_stack.save_episode(ep)
        assert ep_id

    def test_maintenance_mode_blocks_writes(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        enforced_stack.enter_maintenance()

        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Should fail",
            outcome="Never saved",
            source_type="observation",
            created_at=_now(),
        )
        with pytest.raises(MaintenanceModeError):
            enforced_stack.save_episode(ep)


# ---- Provenance Enforcement OFF (Default) ----


class TestProvenanceOff:
    """With enforce_provenance=False, provenance is not checked."""

    def test_episode_without_derived_from_allowed(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="No provenance needed",
            outcome="Still works",
            source_type="observation",
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        assert ep_id

    def test_belief_without_derived_from_allowed(self, stack):
        stack.on_attach(core_id="core-1", inference=None)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test belief",
            source_type="inferred",
            created_at=_now(),
        )
        belief_id = stack.save_belief(belief)
        assert belief_id


# ---- Provenance Enforcement ON ----


class TestProvenanceEnforced:
    """With enforce_provenance=True in ACTIVE state, provenance is enforced."""

    def _save_raw(self, stack):
        raw = RawEntry(
            id=_uid(),
            stack_id="test-stack",
            blob="Test raw content",
            source="test",
            captured_at=_now(),
        )
        return stack.save_raw(raw)

    def _save_episode(self, stack, raw_id):
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Test objective",
            outcome="Test outcome",
            source_type="observation",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        return stack.save_episode(ep)

    def _save_note(self, stack, raw_id):
        note = Note(
            id=_uid(),
            stack_id="test-stack",
            content="Test note",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        return stack.save_note(note)

    def _save_belief(self, stack, episode_id):
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test belief",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{episode_id}"],
        )
        return stack.save_belief(belief)

    # -- Raw entries need no provenance --

    def test_raw_no_provenance_needed(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        assert raw_id

    # -- Episode requires raw --

    def test_episode_with_valid_provenance(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        assert ep_id

    def test_episode_without_derived_from_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="No provenance",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_episode(ep)

    def test_episode_with_wrong_source_type_raises(self, enforced_stack):
        """Episode must cite raw, not belief."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Wrong source",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["belief:fake-id"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_episode(ep)

    def test_episode_with_nonexistent_raw_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Cites ghost",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["raw:nonexistent-id"],
        )
        with pytest.raises(ProvenanceError, match="does not exist"):
            enforced_stack.save_episode(ep)

    # -- Note requires raw --

    def test_note_with_valid_provenance(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        note_id = self._save_note(enforced_stack, raw_id)
        assert note_id

    def test_note_without_provenance_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        note = Note(
            id=_uid(),
            stack_id="test-stack",
            content="No provenance",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_note(note)

    # -- Belief requires episode or note --

    def test_belief_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)
        assert belief_id

    def test_belief_from_note(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        note_id = self._save_note(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From note",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"note:{note_id}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_belief_from_raw_raises(self, enforced_stack):
        """Belief cannot cite raw directly."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Skip a layer",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"raw:{raw_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_belief(belief)

    def test_belief_without_provenance_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="No provenance",
            source_type="inferred",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_belief(belief)

    # -- Value requires belief --

    def test_value_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="honesty",
            statement="Value honesty",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        value_id = enforced_stack.save_value(value)
        assert value_id

    def test_value_from_episode_raises(self, enforced_stack):
        """Value must cite belief, not episode."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="honesty",
            statement="Value honesty",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_value(value)

    # -- Goal requires episode or belief --

    def test_goal_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        goal = Goal(
            id=_uid(),
            stack_id="test-stack",
            title="Test goal",
            source_type="stated",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        goal_id = enforced_stack.save_goal(goal)
        assert goal_id

    def test_goal_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        goal = Goal(
            id=_uid(),
            stack_id="test-stack",
            title="Test goal",
            source_type="stated",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        goal_id = enforced_stack.save_goal(goal)
        assert goal_id

    # -- Relationship requires episode --

    def test_relationship_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        rel = Relationship(
            id=_uid(),
            stack_id="test-stack",
            entity_name="Alice",
            entity_type="human",
            relationship_type="colleague",
            source_type="observed",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        rel_id = enforced_stack.save_relationship(rel)
        assert rel_id

    def test_relationship_from_belief_raises(self, enforced_stack):
        """Relationship must cite episode, not belief."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        rel = Relationship(
            id=_uid(),
            stack_id="test-stack",
            entity_name="Bob",
            entity_type="human",
            relationship_type="colleague",
            source_type="observed",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_relationship(rel)

    # -- Drive requires episode or belief --

    def test_drive_from_episode(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        drive = Drive(
            id=_uid(),
            stack_id="test-stack",
            drive_type="curiosity",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}"],
        )
        drive_id = enforced_stack.save_drive(drive)
        assert drive_id

    def test_drive_from_belief(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        belief_id = self._save_belief(enforced_stack, ep_id)

        drive = Drive(
            id=_uid(),
            stack_id="test-stack",
            drive_type="curiosity",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"belief:{belief_id}"],
        )
        drive_id = enforced_stack.save_drive(drive)
        assert drive_id

    # -- Malformed provenance references --

    def test_malformed_reference_no_colon_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Bad ref",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=["no-colon-here"],
        )
        with pytest.raises(ProvenanceError, match="Expected format"):
            enforced_stack.save_episode(ep)

    def test_empty_derived_from_list_raises(self, enforced_stack):
        enforced_stack.on_attach(core_id="core-1", inference=None)
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Empty list",
            outcome="Should fail",
            source_type="observation",
            created_at=_now(),
            derived_from=[],
        )
        with pytest.raises(ProvenanceError, match="derived_from must cite"):
            enforced_stack.save_episode(ep)

    # -- Multiple valid sources --

    def test_multiple_valid_sources(self, enforced_stack):
        """A belief can cite multiple episodes."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id1 = self._save_episode(enforced_stack, raw_id)
        ep_id2 = self._save_episode(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From two episodes",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{ep_id1}", f"episode:{ep_id2}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_mixed_valid_sources(self, enforced_stack):
        """A belief can cite both an episode and a note."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)
        note_id = self._save_note(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="From episode and note",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}", f"note:{note_id}"],
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

    def test_one_invalid_in_list_raises(self, enforced_stack):
        """If any reference is invalid, the whole save fails."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        raw_id = self._save_raw(enforced_stack)
        ep_id = self._save_episode(enforced_stack, raw_id)

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="One bad ref",
            source_type="inferred",
            created_at=_now(),
            derived_from=[f"episode:{ep_id}", "raw:some-raw"],
        )
        with pytest.raises(ProvenanceError, match="not an allowed source"):
            enforced_stack.save_belief(belief)


# ---- memory_exists Tests ----


class TestMemoryExists:
    def test_existing_raw_found(self, stack):
        raw = RawEntry(
            id=_uid(),
            stack_id="test-stack",
            blob="Test",
            source="test",
            captured_at=_now(),
        )
        raw_id = stack.save_raw(raw)
        assert stack._backend.memory_exists("raw", raw_id)

    def test_nonexistent_raw_not_found(self, stack):
        assert not stack._backend.memory_exists("raw", "nonexistent")

    def test_existing_episode_found(self, stack):
        ep = Episode(
            id=_uid(),
            stack_id="test-stack",
            objective="Test",
            outcome="Test",
            source_type="observation",
            created_at=_now(),
        )
        ep_id = stack.save_episode(ep)
        assert stack._backend.memory_exists("episode", ep_id)

    def test_unknown_type_returns_false(self, stack):
        assert not stack._backend.memory_exists("unknown_type", "some-id")

    def test_existing_belief_found(self, stack):
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Test",
            source_type="inferred",
            created_at=_now(),
        )
        belief_id = stack.save_belief(belief)
        assert stack._backend.memory_exists("belief", belief_id)


# ---- INITIALIZING State Allows Seed Writes ----


class TestSeedWrites:
    def test_seed_writes_bypass_provenance(self, enforced_stack):
        """In INITIALIZING state, all writes are allowed without provenance."""
        assert enforced_stack.state == StackState.INITIALIZING

        # Save a belief directly -- no episode needed
        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Seed belief",
            source_type="seed",
            created_at=_now(),
        )
        belief_id = enforced_stack.save_belief(belief)
        assert belief_id

        # Save a value directly -- no belief needed
        value = Value(
            id=_uid(),
            stack_id="test-stack",
            name="core-value",
            statement="Seed value",
            source_type="seed",
            created_at=_now(),
        )
        value_id = enforced_stack.save_value(value)
        assert value_id

    def test_after_attach_provenance_enforced(self, enforced_stack):
        """After on_attach, provenance must be provided."""
        enforced_stack.on_attach(core_id="core-1", inference=None)
        assert enforced_stack.state == StackState.ACTIVE

        belief = Belief(
            id=_uid(),
            stack_id="test-stack",
            statement="Post-seed belief",
            source_type="inferred",
            created_at=_now(),
        )
        with pytest.raises(ProvenanceError):
            enforced_stack.save_belief(belief)
