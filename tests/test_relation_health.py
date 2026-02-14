"""Tests for check_relation_table_health (#705)."""

import uuid
from datetime import datetime, timezone

import pytest

from kernle.storage.health import check_relation_table_health
from kernle.storage.sqlite import SQLiteStorage
from kernle.types import Relationship, RelationshipHistoryEntry


@pytest.fixture
def storage(tmp_path):
    s = SQLiteStorage(stack_id="health-test", db_path=tmp_path / "health.db")
    yield s
    s.close()


class TestRelationTableHealth:
    """Tests for cross-table relationship health checks."""

    def test_empty_tables_are_healthy(self, storage):
        """No rows means healthy."""
        result = check_relation_table_health(storage._connect, "health-test")
        assert result["healthy"] is True
        assert result["total_relationships"] == 0
        assert result["total_history"] == 0
        assert result["orphaned_history_count"] == 0

    def test_relationship_with_matching_history(self, storage):
        """History entries linked to existing relationships are healthy."""
        rel = Relationship(
            id="rel-1",
            stack_id="health-test",
            entity_name="Alice",
            entity_type="person",
            relationship_type="collaborator",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship(rel)

        entry = RelationshipHistoryEntry(
            id=str(uuid.uuid4()),
            stack_id="health-test",
            relationship_id="rel-1",
            entity_name="Alice",
            event_type="interaction",
            notes="Met for coffee",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship_history(entry)

        result = check_relation_table_health(storage._connect, "health-test")
        assert result["healthy"] is True
        assert result["total_relationships"] == 1
        assert result["total_history"] == 1
        assert result["orphaned_history_count"] == 0

    def test_orphaned_history_detected(self, storage):
        """History entry with no matching relationship is detected as orphan."""
        # Insert a history entry with a relationship_id that does not exist
        entry = RelationshipHistoryEntry(
            id=str(uuid.uuid4()),
            stack_id="health-test",
            relationship_id="rel-nonexistent",
            entity_name="Ghost",
            event_type="trust_change",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship_history(entry)

        result = check_relation_table_health(storage._connect, "health-test")
        assert result["healthy"] is False
        assert result["orphaned_history_count"] == 1
        assert result["total_history"] == 1
        assert result["total_relationships"] == 0

    def test_stack_id_isolation(self, tmp_path):
        """Health check only considers records for the given stack_id."""
        s = SQLiteStorage(stack_id="stack-a", db_path=tmp_path / "isolation.db")
        try:
            rel = Relationship(
                id="rel-a",
                stack_id="stack-a",
                entity_name="Bob",
                entity_type="person",
                relationship_type="friend",
                created_at=datetime.now(timezone.utc),
            )
            s.save_relationship(rel)

            entry = RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="stack-a",
                relationship_id="rel-a",
                entity_name="Bob",
                event_type="interaction",
                created_at=datetime.now(timezone.utc),
            )
            s.save_relationship_history(entry)

            # Check for a different stack_id -- should see nothing
            result = check_relation_table_health(s._connect, "stack-b")
            assert result["healthy"] is True
            assert result["total_relationships"] == 0
            assert result["total_history"] == 0
        finally:
            s.close()

    def test_mixed_healthy_and_orphaned(self, storage):
        """Mix of valid and orphaned history entries."""
        rel = Relationship(
            id="rel-ok",
            stack_id="health-test",
            entity_name="Valid",
            entity_type="person",
            relationship_type="peer",
            created_at=datetime.now(timezone.utc),
        )
        storage.save_relationship(rel)

        # Valid history
        storage.save_relationship_history(
            RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="health-test",
                relationship_id="rel-ok",
                entity_name="Valid",
                event_type="interaction",
                created_at=datetime.now(timezone.utc),
            )
        )
        # Orphaned history
        storage.save_relationship_history(
            RelationshipHistoryEntry(
                id=str(uuid.uuid4()),
                stack_id="health-test",
                relationship_id="rel-deleted",
                entity_name="Orphan",
                event_type="note",
                created_at=datetime.now(timezone.utc),
            )
        )

        result = check_relation_table_health(storage._connect, "health-test")
        assert result["healthy"] is False
        assert result["total_relationships"] == 1
        assert result["total_history"] == 2
        assert result["orphaned_history_count"] == 1
