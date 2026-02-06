"""Tests for goal_type field (KEP v3, Issue #165).

Tests cover:
- Goal dataclass default and custom goal_type values
- SQLite storage round-trip for goal_type
- Forgetting behavior based on goal_type (half-life, protection)
- Core API goal_type validation
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.storage.base import Goal

# === Dataclass Tests ===


class TestGoalTypeDataclass:
    """Test Goal dataclass goal_type field."""

    def test_default_goal_type_is_task(self):
        """Goal.goal_type defaults to 'task'."""
        goal = Goal(id="g1", stack_id="a1", title="Do something")
        assert goal.goal_type == "task"

    def test_goal_type_aspiration(self):
        """Can create a goal with goal_type='aspiration'."""
        goal = Goal(id="g1", stack_id="a1", title="Be kind", goal_type="aspiration")
        assert goal.goal_type == "aspiration"

    def test_goal_type_commitment(self):
        """Can create a goal with goal_type='commitment'."""
        goal = Goal(id="g1", stack_id="a1", title="Ship v1", goal_type="commitment")
        assert goal.goal_type == "commitment"

    def test_goal_type_exploration(self):
        """Can create a goal with goal_type='exploration'."""
        goal = Goal(id="g1", stack_id="a1", title="Try Rust", goal_type="exploration")
        assert goal.goal_type == "exploration"


# === SQLite Storage Tests ===


class TestGoalTypeSQLiteStorage:
    """Test goal_type persistence in SQLite."""

    def test_save_and_retrieve_default_goal_type(self, sqlite_storage):
        """Saving a goal without goal_type stores 'task' and retrieves it."""
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Write tests",
            created_at=datetime.now(timezone.utc),
        )
        sqlite_storage.save_goal(goal)

        goals = sqlite_storage.get_goals(status="active")
        assert len(goals) >= 1
        saved = [g for g in goals if g.id == goal.id][0]
        assert saved.goal_type == "task"

    def test_save_and_retrieve_aspiration(self, sqlite_storage):
        """Saving an aspiration goal round-trips correctly."""
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Become a better communicator",
            goal_type="aspiration",
            created_at=datetime.now(timezone.utc),
        )
        sqlite_storage.save_goal(goal)

        goals = sqlite_storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal.id][0]
        assert saved.goal_type == "aspiration"

    def test_save_and_retrieve_commitment(self, sqlite_storage):
        """Saving a commitment goal round-trips correctly."""
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Deliver project by Friday",
            goal_type="commitment",
            created_at=datetime.now(timezone.utc),
        )
        sqlite_storage.save_goal(goal)

        goals = sqlite_storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal.id][0]
        assert saved.goal_type == "commitment"

    def test_save_and_retrieve_exploration(self, sqlite_storage):
        """Saving an exploration goal round-trips correctly."""
        goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Try out new testing framework",
            goal_type="exploration",
            created_at=datetime.now(timezone.utc),
        )
        sqlite_storage.save_goal(goal)

        goals = sqlite_storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal.id][0]
        assert saved.goal_type == "exploration"

    def test_multiple_goal_types_coexist(self, sqlite_storage):
        """Goals of different types can be saved and retrieved together."""
        types = ["task", "aspiration", "commitment", "exploration"]
        ids = []
        for gt in types:
            goal = Goal(
                id=str(uuid.uuid4()),
                stack_id="test_agent",
                title=f"Goal of type {gt}",
                goal_type=gt,
                created_at=datetime.now(timezone.utc),
            )
            sqlite_storage.save_goal(goal)
            ids.append(goal.id)

        goals = sqlite_storage.get_goals(status="active")
        saved_types = {g.id: g.goal_type for g in goals if g.id in ids}
        for i, gt in enumerate(types):
            assert saved_types[ids[i]] == gt


# === Core API Tests ===


class TestGoalTypeCoreAPI:
    """Test goal_type through the Kernle core API."""

    def test_core_goal_default_type(self, kernle_instance):
        """Kernle.goal() creates task by default."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Write documentation")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.goal_type == "task"

    def test_core_goal_with_type(self, kernle_instance):
        """Kernle.goal() accepts and stores goal_type."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Be more empathetic", goal_type="aspiration")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.goal_type == "aspiration"

    def test_core_goal_invalid_type_raises(self, kernle_instance):
        """Kernle.goal() raises ValueError for invalid goal_type."""
        kernle, _ = kernle_instance
        with pytest.raises(ValueError, match="Invalid goal_type"):
            kernle.goal("Bad goal", goal_type="invalid")

    def test_aspiration_goal_is_protected(self, kernle_instance):
        """Aspiration goals are created with is_protected=True."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Long-term growth", goal_type="aspiration")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.is_protected is True

    def test_commitment_goal_is_protected(self, kernle_instance):
        """Commitment goals are created with is_protected=True."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Ship feature", goal_type="commitment")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.is_protected is True

    def test_task_goal_not_protected(self, kernle_instance):
        """Task goals are not automatically protected."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Fix bug", goal_type="task")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.is_protected is False

    def test_exploration_goal_not_protected(self, kernle_instance):
        """Exploration goals are not automatically protected."""
        kernle, storage = kernle_instance
        goal_id = kernle.goal("Try Rust", goal_type="exploration")
        goals = storage.get_goals(status="active")
        saved = [g for g in goals if g.id == goal_id][0]
        assert saved.is_protected is False


# === Forgetting Behavior Tests ===


class TestGoalTypeForgetting:
    """Test that forgetting respects goal_type."""

    def test_aspiration_has_slow_decay(self, kernle_instance):
        """Aspiration goals use a longer half-life (180 days)."""
        from kernle.features.forgetting import ForgettingMixin

        assert ForgettingMixin.GOAL_TYPE_HALF_LIVES["aspiration"] == 180.0

    def test_commitment_has_very_slow_decay(self, kernle_instance):
        """Commitment goals use a very long half-life (365 days)."""
        from kernle.features.forgetting import ForgettingMixin

        assert ForgettingMixin.GOAL_TYPE_HALF_LIVES["commitment"] == 365.0

    def test_task_has_normal_decay(self, kernle_instance):
        """Task goals use normal half-life (30 days)."""
        from kernle.features.forgetting import ForgettingMixin

        assert ForgettingMixin.GOAL_TYPE_HALF_LIVES["task"] == 30.0

    def test_exploration_has_normal_decay(self, kernle_instance):
        """Exploration goals use normal half-life (30 days)."""
        from kernle.features.forgetting import ForgettingMixin

        assert ForgettingMixin.GOAL_TYPE_HALF_LIVES["exploration"] == 30.0

    def test_aspiration_salience_higher_than_task(self, kernle_instance):
        """An aspiration goal should have higher salience than an identical task goal.

        With the same age and access pattern, the aspiration's slower decay
        should result in a higher salience score.
        """
        kernle, storage = kernle_instance
        from datetime import timedelta

        # Create both goals with the same timestamp 60 days ago
        old_time = datetime.now(timezone.utc) - timedelta(days=60)

        task_goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Task goal",
            goal_type="task",
            created_at=old_time,
            confidence=0.8,
        )
        aspiration_goal = Goal(
            id=str(uuid.uuid4()),
            stack_id="test_agent",
            title="Aspiration goal",
            goal_type="aspiration",
            created_at=old_time,
            confidence=0.8,
        )

        storage.save_goal(task_goal)
        storage.save_goal(aspiration_goal)

        task_salience = kernle.calculate_salience("goal", task_goal.id)
        aspiration_salience = kernle.calculate_salience("goal", aspiration_goal.id)

        # Aspiration should decay slower, hence higher salience
        assert aspiration_salience > task_salience

    def test_get_half_life_for_non_goal_record(self, kernle_instance):
        """Non-goal records use the default half-life."""
        kernle, storage = kernle_instance

        episode_id = str(uuid.uuid4())
        from kernle.storage.base import Episode

        ep = Episode(
            id=episode_id,
            stack_id="test_agent",
            objective="test",
            outcome="test",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_episode(ep)

        # For non-goal types, half-life should be the default
        from kernle.features.forgetting import ForgettingMixin

        assert kernle._get_half_life_for_record("episode", ep) == ForgettingMixin.DEFAULT_HALF_LIFE


# === Schema Migration Tests ===


class TestGoalTypeMigration:
    """Test that schema migration adds goal_type column."""

    def test_goal_type_column_exists(self, sqlite_storage):
        """The goals table should have a goal_type column after migration."""

        with sqlite_storage._connect() as conn:
            cols = conn.execute("PRAGMA table_info(goals)").fetchall()
            col_names = {c[1] for c in cols}
            assert "goal_type" in col_names

    def test_goal_type_defaults_to_task_in_db(self, sqlite_storage):
        """Inserting a goal without goal_type gets DEFAULT 'task'."""
        with sqlite_storage._connect() as conn:
            conn.execute(
                """INSERT INTO goals (id, stack_id, title, created_at, local_updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "test-id",
                    "test_agent",
                    "Raw insert",
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                ),
            )
            conn.commit()
            row = conn.execute("SELECT goal_type FROM goals WHERE id = 'test-id'").fetchone()
            assert row[0] == "task"
