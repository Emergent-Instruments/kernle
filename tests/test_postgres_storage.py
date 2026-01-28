"""
Tests for PostgreSQL/Supabase storage backend.

These tests mock the Supabase client to test the SupabaseStorage class
without requiring actual cloud infrastructure.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.storage.base import Belief, Drive, Episode, Goal, Note, Value
from kernle.storage.postgres import SupabaseStorage

# === Initialization Tests ===

class TestSupabaseStorageInit:
    """Tests for SupabaseStorage initialization."""

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicit URL and key."""
        storage = SupabaseStorage(
            agent_id="test_agent",
            supabase_url="https://example.supabase.co",
            supabase_key="test-key-12345"
        )
        assert storage.agent_id == "test_agent"
        assert storage.supabase_url == "https://example.supabase.co"
        assert storage.supabase_key == "test-key-12345"
        assert storage._client is None  # Lazy loaded

    def test_init_with_env_vars(self, monkeypatch):
        """Test initialization using environment variables."""
        monkeypatch.setenv("KERNLE_SUPABASE_URL", "https://env.supabase.co")
        monkeypatch.setenv("KERNLE_SUPABASE_KEY", "env-key-67890")

        storage = SupabaseStorage(agent_id="test_agent")
        assert storage.supabase_url == "https://env.supabase.co"
        assert storage.supabase_key == "env-key-67890"

    def test_init_with_fallback_env_vars(self, monkeypatch):
        """Test initialization using fallback environment variables."""
        monkeypatch.setenv("SUPABASE_URL", "https://fallback.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fallback-key")

        storage = SupabaseStorage(agent_id="test_agent")
        assert storage.supabase_url == "https://fallback.supabase.co"
        assert storage.supabase_key == "fallback-key"

    def test_client_lazy_load_missing_url(self, monkeypatch):
        """Test that accessing client raises error when URL is missing."""
        monkeypatch.delenv("KERNLE_SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("KERNLE_SUPABASE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

        storage = SupabaseStorage(agent_id="test_agent")

        with pytest.raises(ValueError, match="Supabase credentials required"):
            _ = storage.client

    def test_client_lazy_load_invalid_url_format(self):
        """Test that accessing client raises error with invalid URL format."""
        storage = SupabaseStorage(
            agent_id="test_agent",
            supabase_url="not-a-valid-url",
            supabase_key="test-key"
        )

        with pytest.raises(ValueError, match="(Invalid|must use HTTPS)"):
            _ = storage.client

    def test_client_lazy_load_empty_key(self):
        """Test that accessing client raises error with empty key."""
        storage = SupabaseStorage(
            agent_id="test_agent",
            supabase_url="https://example.supabase.co",
            supabase_key="   "  # Whitespace only
        )

        with pytest.raises(ValueError, match="Supabase key cannot be empty"):
            _ = storage.client


# === Mock Client Fixture ===

@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client with common behaviors."""
    client = MagicMock()

    # Storage for simulating database
    storage = {
        "agent_episodes": [],
        "agent_beliefs": [],
        "agent_values": [],
        "agent_goals": [],
        "memories": [],
        "agent_drives": [],
        "agent_relationships": [],
    }

    def create_table_mock(table_name):
        """Create a chainable table mock."""
        table = MagicMock()

        def select_mock(*args, **kwargs):
            chain = MagicMock()
            chain._data = storage.get(table_name, []).copy()
            chain._count = len(chain._data)

            def eq_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field) == value]
                chain._count = len(chain._data)
                return chain

            def gte_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field, '') >= value]
                return chain

            def lte_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field, '') <= value]
                return chain

            def lt_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field, 0) < value]
                return chain

            def order_mock(field, desc=False):
                try:
                    chain._data.sort(key=lambda x: x.get(field, ''), reverse=desc)
                except TypeError:
                    pass
                return chain

            def limit_mock(n):
                chain._data = chain._data[:n]
                return chain

            def execute_mock():
                result = MagicMock()
                result.data = chain._data
                result.count = chain._count if kwargs.get('count') == 'exact' else None
                return result

            chain.eq = eq_filter
            chain.gte = gte_filter
            chain.lte = lte_filter
            chain.lt = lt_filter
            chain.order = order_mock
            chain.limit = limit_mock
            chain.execute = execute_mock
            return chain

        def upsert_mock(data):
            chain = MagicMock()
            # Add to storage
            if table_name in storage:
                # Remove existing record with same ID
                storage[table_name] = [r for r in storage[table_name] if r.get('id') != data.get('id')]
                data['created_at'] = data.get('created_at') or datetime.now(timezone.utc).isoformat()
                storage[table_name].append(data)

            def execute_mock():
                result = MagicMock()
                result.data = [data]
                return result

            chain.execute = execute_mock
            return chain

        def insert_mock(data):
            chain = MagicMock()
            if table_name in storage:
                data['created_at'] = data.get('created_at') or datetime.now(timezone.utc).isoformat()
                storage[table_name].append(data)

            def execute_mock():
                result = MagicMock()
                result.data = [data]
                return result

            chain.execute = execute_mock
            return chain

        def update_mock(data):
            chain = MagicMock()
            chain._update_data = data

            def eq_filter(field, value):
                # Apply update to matching records
                for record in storage.get(table_name, []):
                    if record.get(field) == value:
                        record.update(chain._update_data)

                def execute_mock():
                    result = MagicMock()
                    result.data = [r for r in storage.get(table_name, []) if r.get(field) == value]
                    return result

                inner_chain = MagicMock()
                inner_chain.eq = eq_filter  # Allow chaining multiple eq
                inner_chain.execute = execute_mock
                return inner_chain

            chain.eq = eq_filter
            return chain

        table.select = select_mock
        table.upsert = upsert_mock
        table.insert = insert_mock
        table.update = update_mock
        return table

    client.table = create_table_mock
    return client, storage


@pytest.fixture
def supabase_storage(mock_supabase_client):
    """Create a SupabaseStorage instance with mocked client."""
    client, storage = mock_supabase_client

    supabase = SupabaseStorage(
        agent_id="test_agent",
        supabase_url="https://test.supabase.co",
        supabase_key="test-key"
    )
    # Inject the mock client directly
    supabase._client = client

    return supabase, storage


# === Episode Tests ===

class TestSupabaseEpisodes:
    """Tests for episode operations."""

    def test_save_episode(self, supabase_storage):
        """Test saving an episode."""
        storage, db = supabase_storage

        episode = Episode(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            objective="Test objective",
            outcome="Test outcome",
            outcome_type="success",
            lessons=["Lesson 1"],
            tags=["test"],
        )

        episode_id = storage.save_episode(episode)
        assert episode_id is not None
        assert len(db["agent_episodes"]) == 1

        saved = db["agent_episodes"][0]
        assert saved["objective"] == "Test objective"
        assert saved["outcome_description"] == "Test outcome"

    def test_get_episodes(self, supabase_storage):
        """Test retrieving episodes."""
        storage, db = supabase_storage

        # Add test data directly to mock storage
        db["agent_episodes"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "objective": "First task",
            "outcome_description": "Completed",
            "outcome_type": "success",
            "lessons_learned": [],
            "tags": ["work"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.9,
        })

        episodes = storage.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "First task"

    def test_get_episode_by_id(self, supabase_storage):
        """Test retrieving a specific episode by ID."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append({
            "id": ep_id,
            "agent_id": "test_agent",
            "objective": "Specific episode",
            "outcome_description": "Done",
            "outcome_type": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        episode = storage.get_episode(ep_id)
        assert episode is not None
        assert episode.objective == "Specific episode"

    def test_update_episode_emotion(self, supabase_storage):
        """Test updating episode emotional data."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append({
            "id": ep_id,
            "agent_id": "test_agent",
            "objective": "Emotional episode",
            "outcome_description": "Felt good",
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        result = storage.update_episode_emotion(
            ep_id, valence=0.8, arousal=0.5, tags=["joy"]
        )
        assert result is True


# === Belief Tests ===

class TestSupabaseBeliefs:
    """Tests for belief operations."""

    def test_save_belief(self, supabase_storage):
        """Test saving a belief."""
        storage, db = supabase_storage

        belief = Belief(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            statement="Testing is valuable",
            belief_type="value",
            confidence=0.9,
        )

        belief_id = storage.save_belief(belief)
        assert belief_id is not None
        assert len(db["agent_beliefs"]) == 1

    def test_get_beliefs(self, supabase_storage):
        """Test retrieving beliefs."""
        storage, db = supabase_storage

        db["agent_beliefs"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "statement": "Code should be tested",
            "belief_type": "fact",
            "confidence": 0.85,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].statement == "Code should be tested"

    def test_find_belief(self, supabase_storage):
        """Test finding a belief by statement."""
        storage, db = supabase_storage

        db["agent_beliefs"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "statement": "Unique statement",
            "belief_type": "fact",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        found = storage.find_belief("Unique statement")
        assert found is not None
        assert found.statement == "Unique statement"


# === Value Tests ===

class TestSupabaseValues:
    """Tests for value operations."""

    def test_save_value(self, supabase_storage):
        """Test saving a value."""
        storage, db = supabase_storage

        value = Value(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Quality",
            statement="Quality over quantity",
            priority=80,
        )

        value_id = storage.save_value(value)
        assert value_id is not None
        assert len(db["agent_values"]) == 1

    def test_get_values(self, supabase_storage):
        """Test retrieving values."""
        storage, db = supabase_storage

        db["agent_values"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "name": "Integrity",
            "statement": "Be honest and transparent",
            "priority": 90,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        values = storage.get_values()
        assert len(values) == 1
        assert values[0].name == "Integrity"


# === Goal Tests ===

class TestSupabaseGoals:
    """Tests for goal operations."""

    def test_save_goal(self, supabase_storage):
        """Test saving a goal."""
        storage, db = supabase_storage

        goal = Goal(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            title="Write tests",
            description="Achieve good test coverage",
            priority="high",
            status="active",
        )

        goal_id = storage.save_goal(goal)
        assert goal_id is not None
        assert len(db["agent_goals"]) == 1

    def test_get_goals(self, supabase_storage):
        """Test retrieving goals."""
        storage, db = supabase_storage

        db["agent_goals"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "title": "Ship feature",
            "description": "Complete the feature",
            "priority": "high",
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        goals = storage.get_goals(status="active")
        assert len(goals) == 1
        assert goals[0].title == "Ship feature"


# === Note Tests ===

class TestSupabaseNotes:
    """Tests for note operations."""

    def test_save_note(self, supabase_storage):
        """Test saving a note."""
        storage, db = supabase_storage

        note = Note(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            content="Important insight",
            note_type="insight",
            tags=["important"],
        )

        note_id = storage.save_note(note)
        assert note_id is not None
        assert len(db["memories"]) == 1

    def test_get_notes(self, supabase_storage):
        """Test retrieving notes."""
        storage, db = supabase_storage

        db["memories"].append({
            "id": str(uuid.uuid4()),
            "owner_id": "test_agent",
            "content": "A curated memory",
            "source": "curated",
            "metadata": {"note_type": "note", "tags": []},
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        notes = storage.get_notes()
        assert len(notes) == 1
        assert notes[0].content == "A curated memory"


# === Drive Tests ===

class TestSupabaseDrives:
    """Tests for drive operations."""

    def test_save_drive(self, supabase_storage):
        """Test saving a drive."""
        storage, db = supabase_storage

        drive = Drive(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            drive_type="curiosity",
            intensity=0.7,
            focus_areas=["learning", "exploration"],
        )

        drive_id = storage.save_drive(drive)
        assert drive_id is not None

    def test_get_drives(self, supabase_storage):
        """Test retrieving drives."""
        storage, db = supabase_storage

        db["agent_drives"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "drive_type": "growth",
            "intensity": 0.8,
            "focus_areas": ["improvement"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        drives = storage.get_drives()
        assert len(drives) == 1
        assert drives[0].drive_type == "growth"


# === Sync Tests ===

class TestSupabaseSync:
    """Tests for sync-related operations."""

    def test_sync_returns_empty_result(self, supabase_storage):
        """Test that sync returns empty result for cloud storage."""
        storage, _ = supabase_storage
        result = storage.sync()
        assert result is not None

    def test_pull_changes_returns_empty_result(self, supabase_storage):
        """Test that pull_changes returns empty result."""
        storage, _ = supabase_storage
        result = storage.pull_changes()
        assert result is not None

    def test_get_pending_sync_count_is_zero(self, supabase_storage):
        """Test that pending sync count is always 0 for cloud storage."""
        storage, _ = supabase_storage
        count = storage.get_pending_sync_count()
        assert count == 0


# === Stats Tests ===

class TestSupabaseStats:
    """Tests for statistics operations."""

    def test_get_stats(self, supabase_storage):
        """Test retrieving storage statistics."""
        storage, db = supabase_storage

        # Add some test data
        db["agent_episodes"].append({"id": "1", "agent_id": "test_agent"})
        db["agent_beliefs"].append({"id": "1", "agent_id": "test_agent", "is_active": True})
        db["agent_values"].append({"id": "1", "agent_id": "test_agent", "is_active": True})
        db["agent_goals"].append({"id": "1", "agent_id": "test_agent", "status": "active"})
        db["memories"].append({"id": "1", "owner_id": "test_agent", "source": "curated"})

        stats = storage.get_stats()
        assert "episodes" in stats
        assert "beliefs" in stats
        assert "values" in stats
        assert "goals" in stats
        assert "notes" in stats


# === Search Tests ===

class TestSupabaseSearch:
    """Tests for search operations."""

    def test_search_episodes(self, supabase_storage):
        """Test searching episodes by text."""
        storage, db = supabase_storage

        db["agent_episodes"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "objective": "Implement feature X",
            "outcome_description": "Successfully deployed",
            "lessons_learned": ["Plan ahead"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        results = storage.search("feature", record_types=["episode"])
        assert len(results) >= 1

    def test_search_notes(self, supabase_storage):
        """Test searching notes by content."""
        storage, db = supabase_storage

        db["memories"].append({
            "id": str(uuid.uuid4()),
            "owner_id": "test_agent",
            "content": "Important discovery about testing",
            "source": "curated",
            "metadata": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        results = storage.search("discovery", record_types=["note"])
        assert len(results) >= 1


# === Meta-Memory Tests ===

class TestSupabaseMetaMemory:
    """Tests for meta-memory operations."""

    def test_get_memory_by_type_and_id(self, supabase_storage):
        """Test retrieving a specific memory by type and ID."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append({
            "id": ep_id,
            "agent_id": "test_agent",
            "objective": "Test memory",
            "outcome_description": "Found",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        memory = storage.get_memory("episode", ep_id)
        assert memory is not None
        assert memory.objective == "Test memory"

    def test_get_memory_invalid_type(self, supabase_storage):
        """Test that invalid memory type returns None."""
        storage, _ = supabase_storage
        memory = storage.get_memory("invalid_type", "some-id")
        assert memory is None
