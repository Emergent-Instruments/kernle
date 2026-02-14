"""Tests for flat file permissions (Security-02).

All flat files (beliefs, values, goals, relationships) must be written
with 0o600 permissions to prevent other users from reading memory data.
"""

import os
import stat
from datetime import datetime, timezone

from kernle.storage import Belief, Goal, Relationship, Value
from kernle.storage.flat_files import (
    sync_beliefs_to_file,
    sync_goals_to_file,
    sync_relationships_to_file,
    sync_values_to_file,
)

NOW = "2025-06-01T00:00:00+00:00"


def _make_belief(**overrides):
    defaults = {
        "id": "test-belief-1",
        "stack_id": "test",
        "statement": "Test belief",
        "belief_type": "fact",
        "confidence": 0.8,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Belief(**defaults)


def _make_value(**overrides):
    defaults = {
        "id": "test-value-1",
        "stack_id": "test",
        "name": "Test Value",
        "statement": "A test value",
        "priority": 5,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Value(**defaults)


def _make_goal(**overrides):
    defaults = {
        "id": "test-goal-1",
        "stack_id": "test",
        "title": "Test Goal",
        "description": "A test goal",
        "status": "active",
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Goal(**defaults)


def _make_relationship(**overrides):
    defaults = {
        "id": "test-rel-1",
        "stack_id": "test",
        "entity_name": "Test Entity",
        "entity_type": "person",
        "relationship_type": "acquaintance",
        "sentiment": 0.5,
        "interaction_count": 3,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Relationship(**defaults)


def _file_mode(path):
    """Return file permission bits (e.g. 0o600)."""
    return stat.S_IMODE(os.stat(path).st_mode)


class TestFlatFilePermissions:
    def test_beliefs_file_has_0o600(self, tmp_path):
        path = tmp_path / "beliefs.md"
        sync_beliefs_to_file(path, [_make_belief()], NOW)
        assert _file_mode(path) == 0o600

    def test_values_file_has_0o600(self, tmp_path):
        path = tmp_path / "values.md"
        sync_values_to_file(path, [_make_value()], NOW)
        assert _file_mode(path) == 0o600

    def test_goals_file_has_0o600(self, tmp_path):
        path = tmp_path / "goals.md"
        sync_goals_to_file(path, [_make_goal()], NOW)
        assert _file_mode(path) == 0o600

    def test_relationships_file_has_0o600(self, tmp_path):
        path = tmp_path / "relationships.md"
        sync_relationships_to_file(path, [_make_relationship()], NOW)
        assert _file_mode(path) == 0o600

    def test_empty_lists_still_set_permissions(self, tmp_path):
        """Even empty files should have restrictive permissions."""
        for fn, name in [
            (sync_beliefs_to_file, "beliefs.md"),
            (sync_values_to_file, "values.md"),
            (sync_goals_to_file, "goals.md"),
            (sync_relationships_to_file, "relationships.md"),
        ]:
            path = tmp_path / name
            fn(path, [], NOW)
            assert _file_mode(path) == 0o600, f"{name} should have 0o600 permissions"
