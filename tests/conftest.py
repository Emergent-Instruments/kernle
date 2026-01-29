"""
Pytest fixtures and test configuration for Kernle tests.

Updated to work with the storage abstraction layer.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from kernle.core import Kernle
from kernle.storage import SQLiteStorage
from kernle.storage.base import Belief, Drive, Episode, Goal, Note, Value


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary SQLite database path."""
    return tmp_path / "test_memories.db"


@pytest.fixture
def sqlite_storage(temp_db_path):
    """SQLite storage instance for testing."""
    storage = SQLiteStorage(
        agent_id="test_agent",
        db_path=temp_db_path,
    )
    yield storage
    storage.close()


@pytest.fixture
def kernle_instance(temp_checkpoint_dir, temp_db_path):
    """Kernle instance with SQLite storage for testing."""
    storage = SQLiteStorage(
        agent_id="test_agent",
        db_path=temp_db_path,
    )

    kernle = Kernle(agent_id="test_agent", storage=storage, checkpoint_dir=temp_checkpoint_dir)

    yield kernle, storage
    storage.close()


@pytest.fixture
def sample_episode():
    """Sample episode for testing."""
    return Episode(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        objective="Complete unit tests for Kernle",
        outcome="All tests passing with good coverage",
        outcome_type="success",
        lessons=["Always test edge cases", "Mock external dependencies"],
        tags=["testing", "development"],
        created_at=datetime.now(timezone.utc),
        confidence=0.9,
    )


@pytest.fixture
def sample_note():
    """Sample note for testing."""
    return Note(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        content="**Decision**: Use pytest for testing framework",
        note_type="decision",
        reason="Industry standard with good plugin ecosystem",
        tags=["testing"],
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_belief():
    """Sample belief for testing."""
    return Belief(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        statement="Comprehensive testing leads to more reliable software",
        belief_type="fact",
        confidence=0.9,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_value():
    """Sample value for testing."""
    return Value(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        name="Quality",
        statement="Software should be thoroughly tested and reliable",
        priority=80,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_goal():
    """Sample goal for testing."""
    return Goal(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        title="Achieve 80%+ test coverage",
        description="Write comprehensive tests for the entire Kernle system",
        priority="high",
        status="active",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_drive():
    """Sample drive for testing."""
    return Drive(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        drive_type="growth",
        intensity=0.7,
        focus_areas=["learning", "improvement"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def populated_storage(
    kernle_instance,
    sample_episode,
    sample_note,
    sample_belief,
    sample_value,
    sample_goal,
    sample_drive,
):
    """Populate the kernle_instance storage with sample data.

    This fixture depends on kernle_instance and populates its storage.
    Use both fixtures together: (kernle_instance, populated_storage)
    """
    kernle, storage = kernle_instance

    # Save sample data
    storage.save_episode(sample_episode)
    storage.save_note(sample_note)
    storage.save_belief(sample_belief)
    storage.save_value(sample_value)
    storage.save_goal(sample_goal)
    storage.save_drive(sample_drive)

    # Add some additional test data
    # Episode without lessons (not reflected)
    unreflected_episode = Episode(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        objective="Debug memory leak",
        outcome="Could not reproduce the issue",
        outcome_type="failure",
        lessons=["Need better monitoring tools"],
        tags=["debugging"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_episode(unreflected_episode)

    # Checkpoint episode (should be filtered from recent work)
    checkpoint_episode = Episode(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        objective="Implement caching",
        outcome="Basic caching implemented, optimization needed",
        outcome_type="partial",
        lessons=["Start simple, then optimize"],
        tags=["checkpoint"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_episode(checkpoint_episode)

    # Additional note
    insight_note = Note(
        id=str(uuid.uuid4()),
        agent_id="test_agent",
        content="**Insight**: Mocking is crucial for isolated testing",
        note_type="insight",
        tags=["testing"],
        created_at=datetime.now(timezone.utc),
    )
    storage.save_note(insight_note)

    return storage


# Legacy fixtures for backwards compatibility with old test patterns
# These mock the Supabase client interface


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client that simulates database operations.

    DEPRECATED: Use sqlite_storage fixture instead for new tests.
    Kept for backwards compatibility with existing tests.
    """
    client = Mock()

    # In-memory storage for different tables
    storage = {
        "agent_values": [],
        "agent_beliefs": [],
        "agent_goals": [],
        "agent_episodes": [],
        "agent_drives": [],
        "agent_relationships": [],
        "memories": [],
    }

    def create_table_mock(table_name: str):
        """Create a mock table interface."""
        table_mock = Mock()
        table_data = storage[table_name]

        def select_mock(fields="*", count=None):
            result = Mock()
            result.data = table_data.copy()
            result.count = len(table_data) if count == "exact" else None

            # Chain methods
            def eq_mock(field, value):
                filtered_data = [item for item in table_data if item.get(field) == value]
                result.data = filtered_data
                result.count = len(filtered_data) if count == "exact" else None
                return result

            def ilike_mock(field, value):
                pattern = value.replace("%", "")
                filtered_data = [
                    item
                    for item in table_data
                    if pattern.lower() in str(item.get(field, "")).lower()
                ]
                result.data = filtered_data
                return result

            def gte_mock(field, value):
                result.data = [item for item in result.data if item.get(field, "") >= value]
                return result

            def lte_mock(field, value):
                result.data = [item for item in result.data if item.get(field, "") <= value]
                return result

            def order_mock(field, desc=False):
                if result.data:
                    reverse = desc
                    try:
                        result.data.sort(key=lambda x: x.get(field, ""), reverse=reverse)
                    except (TypeError, KeyError):
                        pass  # Skip sorting if comparison fails
                return result

            def limit_mock(count):
                result.data = result.data[:count]
                return result

            def execute_mock():
                return result

            # Attach chaining methods
            result.eq = eq_mock
            result.ilike = ilike_mock
            result.gte = gte_mock
            result.lte = lte_mock
            result.order = order_mock
            result.limit = limit_mock
            result.execute = execute_mock

            return result

        def insert_mock(data):
            if isinstance(data, list):
                for item in data:
                    if "id" not in item:
                        item["id"] = str(uuid.uuid4())
                    item["created_at"] = datetime.now(timezone.utc).isoformat()
                    table_data.append(item)
            else:
                if "id" not in data:
                    data["id"] = str(uuid.uuid4())
                data["created_at"] = datetime.now(timezone.utc).isoformat()
                table_data.append(data)

            result = Mock()
            result.data = [data] if not isinstance(data, list) else data
            result.execute = lambda: result
            return result

        def upsert_mock(data):
            return insert_mock(data)

        def update_mock(data):
            # Returns an object that can be chained with .eq()
            update_result = Mock()

            def eq_update_mock(field, value):
                for item in table_data:
                    if item.get(field) == value:
                        item.update(data)
                        break

                result = Mock()
                result.data = [data]
                result.execute = lambda: result
                return result

            update_result.eq = eq_update_mock
            return update_result

        table_mock.select = select_mock
        table_mock.insert = insert_mock
        table_mock.upsert = upsert_mock
        table_mock.update = update_mock

        return table_mock

    # Set up table method
    client.table = create_table_mock

    return client, storage


@pytest.fixture
def sample_episode_data():
    """Sample episode data as dict for testing.

    DEPRECATED: Use sample_episode fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "objective": "Complete unit tests for Kernle",
        "outcome_type": "success",
        "outcome_description": "All tests passing with good coverage",
        "lessons_learned": ["Always test edge cases", "Mock external dependencies"],
        "patterns_to_repeat": ["Comprehensive test coverage"],
        "patterns_to_avoid": ["Tautological tests"],
        "tags": ["testing", "development"],
        "is_reflected": True,
        "confidence": 0.9,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory/note data as dict for testing.

    DEPRECATED: Use sample_note fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "owner_id": "test_agent",
        "owner_type": "agent",
        "content": "**Decision**: Use pytest for testing framework",
        "source": "curated",
        "metadata": {
            "note_type": "decision",
            "tags": ["testing"],
            "reason": "Industry standard with good plugin ecosystem",
        },
        "visibility": "private",
        "is_curated": True,
        "is_protected": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_belief_data():
    """Sample belief data as dict for testing.

    DEPRECATED: Use sample_belief fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "statement": "Comprehensive testing leads to more reliable software",
        "belief_type": "fact",
        "confidence": 0.9,
        "is_active": True,
        "is_foundational": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_value_data():
    """Sample value data as dict for testing.

    DEPRECATED: Use sample_value fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "name": "Quality",
        "statement": "Software should be thoroughly tested and reliable",
        "priority": 80,
        "value_type": "core_value",
        "is_active": True,
        "is_foundational": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_goal_data():
    """Sample goal data as dict for testing.

    DEPRECATED: Use sample_goal fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "title": "Achieve 80%+ test coverage",
        "description": "Write comprehensive tests for the entire Kernle system",
        "priority": "high",
        "status": "active",
        "visibility": "public",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_drive_data():
    """Sample drive data as dict for testing.

    DEPRECATED: Use sample_drive fixture instead.
    """
    return {
        "id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "drive_type": "growth",
        "intensity": 0.7,
        "focus_areas": ["learning", "improvement"],
        "satisfaction_decay_hours": 24,
        "last_satisfied_at": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
