"""Tests for suggestion type resolution — all produced types must be accepted.

Verifies that accept_suggestion() handles every memory type that the
processing pipeline can produce, and that a shared constant ensures
producer and resolver can't drift apart.
"""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.stack import SQLiteStack
from kernle.types import (
    SUGGESTION_MEMORY_TYPES,
    MemorySuggestion,
)

STACK_ID = "type-resolution-test"


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def stack(tmp_path):
    db_path = tmp_path / "type_resolution.db"
    return SQLiteStack(stack_id=STACK_ID, db_path=db_path, components=[], enforce_provenance=False)


@pytest.fixture
def strict_stack(tmp_path):
    db_path = tmp_path / "type_resolution_strict.db"
    return SQLiteStack(stack_id=STACK_ID, db_path=db_path, components=[], enforce_provenance=True)


def _make_suggestion(memory_type: str, content: dict) -> MemorySuggestion:
    return MemorySuggestion(
        id=_uid(),
        stack_id=STACK_ID,
        memory_type=memory_type,
        content=content,
        confidence=0.8,
        source_raw_ids=["raw-1"],
        created_at=_now(),
    )


class TestSuggestionTypeResolution:
    """Verify that accept_suggestion handles all types the producer emits."""

    def test_accept_goal_suggestion_creates_goal(self, stack):
        s = _make_suggestion(
            "goal",
            {
                "title": "Ship v2.0",
                "description": "Release the next version",
                "goal_type": "task",
                "priority": "high",
                "status": "active",
            },
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        goals = stack.get_goals(limit=100)
        assert any(g.id == memory_id for g in goals)

    def test_accept_value_suggestion_creates_value(self, stack):
        s = _make_suggestion(
            "value",
            {
                "name": "Reliability",
                "statement": "Systems should be dependable",
                "priority": 75,
            },
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        values = stack.get_values(limit=100)
        assert any(v.id == memory_id for v in values)

    def test_accept_relationship_suggestion_creates_relationship(self, stack):
        s = _make_suggestion(
            "relationship",
            {
                "entity_name": "alice",
                "entity_type": "human",
                "relationship_type": "collaborator",
                "notes": "Good partner",
            },
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        rels = stack.get_relationships()
        assert any(r.id == memory_id for r in rels)

    def test_accept_drive_suggestion_creates_drive(self, stack):
        s = _make_suggestion(
            "drive",
            {
                "drive_type": "curiosity",
                "intensity": 0.7,
                "focus_areas": ["learning"],
            },
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        drives = stack.get_drives()
        assert any(d.id == memory_id for d in drives)

    def test_accept_unknown_type_raises_value_error(self, stack):
        s = _make_suggestion("widget", {"foo": "bar"})
        stack.save_suggestion(s)
        with pytest.raises(ValueError, match="[Uu]nsupported.*widget"):
            stack.accept_suggestion(s.id)

    def test_produced_types_exactly_match_resolved_types(self):
        """Hard assertion: the shared constant covers all 7 types."""
        expected = {"episode", "belief", "note", "goal", "relationship", "value", "drive"}
        assert SUGGESTION_MEMORY_TYPES == expected


class TestSuggestionTypeResolutionStrictMode:
    """Verify accept_suggestion works with enforce_provenance=True.

    The source_entity='kernle:suggestion-promotion' bypass must be set
    on all type handlers, otherwise strict mode raises ProvenanceError.
    """

    def test_accept_goal_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "goal",
            {
                "title": "Ship v2.0",
                "description": "Release the next version",
                "goal_type": "task",
                "priority": "high",
                "status": "active",
            },
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_value_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "value",
            {
                "name": "Reliability",
                "statement": "Systems should be dependable",
                "priority": 75,
            },
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_relationship_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "relationship",
            {
                "entity_name": "alice",
                "entity_type": "human",
                "relationship_type": "collaborator",
                "notes": "Good partner",
            },
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None

    def test_accept_drive_strict_mode(self, strict_stack):
        s = _make_suggestion(
            "drive",
            {
                "drive_type": "curiosity",
                "intensity": 0.7,
                "focus_areas": ["learning"],
            },
        )
        strict_stack.save_suggestion(s)
        memory_id = strict_stack.accept_suggestion(s.id)
        assert memory_id is not None
