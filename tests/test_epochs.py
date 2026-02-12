"""Tests for temporal epoch (era tracking) functionality.

Tests epoch CRUD operations at both storage and core API levels,
including epoch_id propagation to memory types and epoch-filtered loading.
"""

import pytest

from kernle import Kernle
from kernle.storage import Belief, Episode, Epoch, Note, SQLiteStorage


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def storage(temp_db, monkeypatch):
    """Create a SQLiteStorage instance for testing."""
    s = SQLiteStorage(stack_id="test-agent", db_path=temp_db)
    monkeypatch.setattr(s, "has_cloud_credentials", lambda: False)
    yield s
    s.close()


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a Kernle instance with temp storage."""
    db_path = tmp_path / "test.db"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    storage = SQLiteStorage(stack_id="test-agent", db_path=db_path)
    return Kernle(
        stack_id="test-agent", storage=storage, checkpoint_dir=checkpoint_dir, strict=False
    )


class TestEpochStorage:
    """Test epoch CRUD at the storage layer."""

    def test_save_and_get_epoch(self, storage):
        """Save an epoch and retrieve it by ID."""
        epoch = Epoch(
            id="epoch-1",
            stack_id="test-agent",
            epoch_number=1,
            name="onboarding",
            trigger_type="declared",
        )
        result_id = storage.save_epoch(epoch)
        assert result_id == "epoch-1"

        retrieved = storage.get_epoch("epoch-1")
        assert retrieved is not None
        assert retrieved.name == "onboarding"
        assert retrieved.epoch_number == 1
        assert retrieved.trigger_type == "declared"

    def test_get_epoch_not_found(self, storage):
        """Getting a nonexistent epoch returns None."""
        result = storage.get_epoch("nonexistent")
        assert result is None

    def test_get_epochs_ordered(self, storage):
        """Epochs should be returned in reverse epoch_number order."""
        for i in range(1, 4):
            epoch = Epoch(
                id=f"epoch-{i}",
                stack_id="test-agent",
                epoch_number=i,
                name=f"era-{i}",
            )
            storage.save_epoch(epoch)

        epochs = storage.get_epochs(limit=10)
        assert len(epochs) == 3
        assert epochs[0].epoch_number == 3
        assert epochs[1].epoch_number == 2
        assert epochs[2].epoch_number == 1

    def test_get_epochs_limit(self, storage):
        """get_epochs respects the limit parameter."""
        for i in range(1, 6):
            storage.save_epoch(
                Epoch(
                    id=f"epoch-{i}",
                    stack_id="test-agent",
                    epoch_number=i,
                    name=f"era-{i}",
                )
            )

        epochs = storage.get_epochs(limit=2)
        assert len(epochs) == 2

    def test_get_current_epoch(self, storage):
        """get_current_epoch returns the open epoch (ended_at is None)."""
        storage.save_epoch(
            Epoch(
                id="epoch-1",
                stack_id="test-agent",
                epoch_number=1,
                name="closed-era",
                ended_at=storage._utc_now() if hasattr(storage, "_utc_now") else None,
            )
        )
        storage.save_epoch(
            Epoch(
                id="epoch-2",
                stack_id="test-agent",
                epoch_number=2,
                name="open-era",
            )
        )

        current = storage.get_current_epoch()
        assert current is not None
        assert current.id == "epoch-2"
        assert current.name == "open-era"

    def test_get_current_epoch_none(self, storage):
        """get_current_epoch returns None when no open epoch exists."""
        current = storage.get_current_epoch()
        assert current is None

    def test_close_epoch(self, storage):
        """close_epoch sets ended_at and optional summary."""
        storage.save_epoch(
            Epoch(
                id="epoch-1",
                stack_id="test-agent",
                epoch_number=1,
                name="to-close",
            )
        )

        result = storage.close_epoch("epoch-1", summary="Completed onboarding")
        assert result is True

        closed = storage.get_epoch("epoch-1")
        assert closed is not None
        assert closed.ended_at is not None
        assert closed.summary == "Completed onboarding"

    def test_close_epoch_not_found(self, storage):
        """close_epoch returns False for nonexistent epoch."""
        result = storage.close_epoch("nonexistent")
        assert result is False

    def test_close_already_closed_epoch(self, storage):
        """close_epoch returns False if epoch is already closed."""
        storage.save_epoch(
            Epoch(
                id="epoch-1",
                stack_id="test-agent",
                epoch_number=1,
                name="already-closed",
            )
        )
        storage.close_epoch("epoch-1")
        result = storage.close_epoch("epoch-1")
        assert result is False

    def test_epoch_with_key_ids(self, storage):
        """Epoch stores key_belief_ids, key_relationship_ids, etc."""
        epoch = Epoch(
            id="epoch-rich",
            stack_id="test-agent",
            epoch_number=1,
            name="rich-epoch",
            key_belief_ids=["b1", "b2"],
            key_relationship_ids=["r1"],
            key_goal_ids=["g1", "g2", "g3"],
            dominant_drive_ids=["d1"],
        )
        storage.save_epoch(epoch)

        retrieved = storage.get_epoch("epoch-rich")
        assert retrieved is not None
        assert retrieved.key_belief_ids == ["b1", "b2"]
        assert retrieved.key_relationship_ids == ["r1"]
        assert retrieved.key_goal_ids == ["g1", "g2", "g3"]
        assert retrieved.dominant_drive_ids == ["d1"]

    def test_epoch_trigger_description(self, storage):
        """Epoch stores trigger_description."""
        epoch = Epoch(
            id="epoch-trigger",
            stack_id="test-agent",
            epoch_number=1,
            name="triggered-epoch",
            trigger_type="detected",
            trigger_description="Major role change detected",
        )
        storage.save_epoch(epoch)

        retrieved = storage.get_epoch("epoch-trigger")
        assert retrieved is not None
        assert retrieved.trigger_type == "detected"
        assert retrieved.trigger_description == "Major role change detected"

    def test_epoch_trigger_description_none(self, storage):
        """Epoch with no trigger_description defaults to None."""
        epoch = Epoch(
            id="epoch-no-desc",
            stack_id="test-agent",
            epoch_number=1,
            name="no-desc",
        )
        storage.save_epoch(epoch)

        retrieved = storage.get_epoch("epoch-no-desc")
        assert retrieved is not None
        assert retrieved.trigger_description is None


class TestEpochCore:
    """Test epoch operations via the Kernle core API."""

    def test_epoch_create(self, kernle_instance):
        """epoch_create returns an ID and stores the epoch."""
        epoch_id = kernle_instance.epoch_create(name="v1-launch")
        assert epoch_id is not None
        assert len(epoch_id) > 0

        epoch = kernle_instance.get_epoch(epoch_id)
        assert epoch is not None
        assert epoch.name == "v1-launch"
        assert epoch.epoch_number == 1
        assert epoch.ended_at is None

    def test_epoch_create_with_trigger(self, kernle_instance):
        """epoch_create accepts a trigger_type."""
        epoch_id = kernle_instance.epoch_create(name="detected-shift", trigger_type="detected")
        epoch = kernle_instance.get_epoch(epoch_id)
        assert epoch.trigger_type == "detected"

    def test_epoch_create_with_trigger_description(self, kernle_instance):
        """epoch_create accepts a trigger_description."""
        epoch_id = kernle_instance.epoch_create(
            name="system-epoch",
            trigger_type="system",
            trigger_description="Automated consolidation threshold reached",
        )
        epoch = kernle_instance.get_epoch(epoch_id)
        assert epoch.trigger_type == "system"
        assert epoch.trigger_description == "Automated consolidation threshold reached"

    def test_epoch_create_invalid_trigger(self, kernle_instance):
        """epoch_create rejects invalid trigger types."""
        with pytest.raises(ValueError):
            kernle_instance.epoch_create(name="bad", trigger_type="invalid")

    def test_epoch_create_auto_closes_previous(self, kernle_instance):
        """Creating a new epoch auto-closes the previous one."""
        id1 = kernle_instance.epoch_create(name="era-1")
        id2 = kernle_instance.epoch_create(name="era-2")

        epoch1 = kernle_instance.get_epoch(id1)
        epoch2 = kernle_instance.get_epoch(id2)

        assert epoch1.ended_at is not None
        assert epoch2.ended_at is None

    def test_epoch_numbering(self, kernle_instance):
        """Epochs should have incrementing epoch_number."""
        id1 = kernle_instance.epoch_create(name="first")
        id2 = kernle_instance.epoch_create(name="second")
        id3 = kernle_instance.epoch_create(name="third")

        assert kernle_instance.get_epoch(id1).epoch_number == 1
        assert kernle_instance.get_epoch(id2).epoch_number == 2
        assert kernle_instance.get_epoch(id3).epoch_number == 3

    def test_epoch_close(self, kernle_instance):
        """epoch_close closes the current epoch."""
        epoch_id = kernle_instance.epoch_create(name="to-close")
        result = kernle_instance.epoch_close(summary="Done")
        assert result is True

        epoch = kernle_instance.get_epoch(epoch_id)
        assert epoch.ended_at is not None
        assert epoch.summary == "Done"

    def test_epoch_close_no_current(self, kernle_instance):
        """epoch_close returns False when no open epoch."""
        result = kernle_instance.epoch_close()
        assert result is False

    def test_get_current_epoch(self, kernle_instance):
        """get_current_epoch returns the active epoch."""
        epoch_id = kernle_instance.epoch_create(name="current")
        current = kernle_instance.get_current_epoch()
        assert current is not None
        assert current.id == epoch_id

    def test_get_epochs(self, kernle_instance):
        """get_epochs returns all epochs."""
        kernle_instance.epoch_create(name="a")
        kernle_instance.epoch_create(name="b")

        epochs = kernle_instance.get_epochs()
        assert len(epochs) == 2
        # Most recent first
        assert epochs[0].name == "b"
        assert epochs[1].name == "a"


class TestEpochIdPropagation:
    """Test that epoch_id is stored and retrieved on memory types."""

    def test_episode_epoch_id(self, storage):
        """Episodes should store and retrieve epoch_id."""
        episode = Episode(
            id="ep-1",
            stack_id="test-agent",
            objective="Test",
            outcome="Done",
            outcome_type="success",
            epoch_id="epoch-1",
        )
        storage.save_episode(episode)

        retrieved = storage.get_episode("ep-1")
        assert retrieved is not None
        assert retrieved.epoch_id == "epoch-1"

    def test_episode_epoch_id_none(self, storage):
        """Episodes without epoch_id should have None."""
        episode = Episode(
            id="ep-2",
            stack_id="test-agent",
            objective="No epoch",
            outcome="Done",
            outcome_type="success",
        )
        storage.save_episode(episode)

        retrieved = storage.get_episode("ep-2")
        assert retrieved is not None
        assert retrieved.epoch_id is None

    def test_belief_epoch_id(self, storage):
        """Beliefs should store and retrieve epoch_id."""
        belief = Belief(
            id="bel-1",
            stack_id="test-agent",
            statement="Testing is important",
            confidence=0.9,
            belief_type="principle",
            epoch_id="epoch-1",
        )
        storage.save_belief(belief)

        retrieved = storage.get_belief("bel-1")
        assert retrieved is not None
        assert retrieved.epoch_id == "epoch-1"

    def test_note_epoch_id(self, storage):
        """Notes should store and retrieve epoch_id."""
        note = Note(
            id="note-1",
            stack_id="test-agent",
            content="A note in an epoch",
            note_type="observation",
            epoch_id="epoch-2",
        )
        storage.save_note(note)

        notes = storage.get_notes(limit=10)
        retrieved = next((n for n in notes if n.id == "note-1"), None)
        assert retrieved is not None
        assert retrieved.epoch_id == "epoch-2"


class TestEpochFilteredLoading:
    """Test load_all with epoch_id filtering."""

    def test_load_all_filters_by_epoch_id(self, storage):
        """load_all with epoch_id only returns memories from that epoch."""
        # Create episodes in different epochs
        storage.save_episode(
            Episode(
                id="ep-e1",
                stack_id="test-agent",
                objective="Epoch 1 task",
                outcome="Done",
                outcome_type="success",
                epoch_id="epoch-1",
            )
        )
        storage.save_episode(
            Episode(
                id="ep-e2",
                stack_id="test-agent",
                objective="Epoch 2 task",
                outcome="Done",
                outcome_type="success",
                epoch_id="epoch-2",
            )
        )
        storage.save_episode(
            Episode(
                id="ep-none",
                stack_id="test-agent",
                objective="Pre-epoch task",
                outcome="Done",
                outcome_type="success",
            )
        )

        # Load all without filter - should get all 3
        all_data = storage.load_all()
        assert len(all_data["episodes"]) == 3

        # Load with epoch_id filter - should get only epoch-1
        filtered = storage.load_all(epoch_id="epoch-1")
        assert len(filtered["episodes"]) == 1
        assert filtered["episodes"][0].id == "ep-e1"

    def test_load_all_epoch_filter_beliefs(self, storage):
        """load_all epoch_id filtering works on beliefs."""
        storage.save_belief(
            Belief(
                id="bel-e1",
                stack_id="test-agent",
                statement="Epoch 1 belief",
                epoch_id="epoch-1",
            )
        )
        storage.save_belief(
            Belief(
                id="bel-e2",
                stack_id="test-agent",
                statement="Epoch 2 belief",
                epoch_id="epoch-2",
            )
        )

        filtered = storage.load_all(epoch_id="epoch-1")
        assert len(filtered["beliefs"]) == 1
        assert filtered["beliefs"][0].id == "bel-e1"

    def test_load_all_no_epoch_filter_returns_all(self, storage):
        """load_all without epoch_id returns everything."""
        storage.save_episode(
            Episode(
                id="ep-1",
                stack_id="test-agent",
                objective="Task",
                outcome="Done",
                outcome_type="success",
                epoch_id="epoch-1",
            )
        )
        storage.save_episode(
            Episode(
                id="ep-2",
                stack_id="test-agent",
                objective="Task 2",
                outcome="Done",
                outcome_type="success",
            )
        )

        data = storage.load_all()
        assert len(data["episodes"]) == 2

    def test_load_with_epoch_id(self, kernle_instance):
        """Kernle.load(epoch_id=...) filters memory to that epoch."""
        k = kernle_instance

        # Create an epoch and some memories
        epoch_id = k.epoch_create(name="test-epoch")

        # Record an episode in the epoch
        ep_id = k.episode(
            objective="Epoch task",
            outcome="Done",
            lessons=["test"],
        )

        # Manually set epoch_id on the stored episode
        ep = k._storage.get_episode(ep_id)
        ep.epoch_id = epoch_id
        k._storage.save_episode(ep)

        # Record another episode without epoch
        k.episode(
            objective="No epoch task",
            outcome="Done",
            lessons=["test"],
        )

        # Load without epoch filter
        full = k.load()
        full_episodes = full.get("episodes", [])

        # Load with epoch filter
        filtered = k.load(epoch_id=epoch_id)
        filtered_episodes = filtered.get("episodes", [])

        # Full load should have more episodes than filtered
        assert len(full_episodes) >= len(filtered_episodes)
        # Filtered should only have epoch-tagged episodes
        for ep_data in filtered_episodes:
            if isinstance(ep_data, dict):
                assert ep_data.get("epoch_id") == epoch_id
