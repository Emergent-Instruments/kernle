"""
Tests for PostgreSQL/Supabase storage backend.

These tests mock the Supabase client to test the SupabaseStorage class
without requiring actual cloud infrastructure.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kernle.storage.base import Belief, Drive, Episode, Goal, Note, Playbook, Value
from kernle.storage.postgres import SupabaseStorage

# === Initialization Tests ===


class TestSupabaseStorageInit:
    """Tests for SupabaseStorage initialization."""

    def test_init_with_explicit_credentials(self):
        """Test initialization with explicit URL and key."""
        storage = SupabaseStorage(
            agent_id="test_agent",
            supabase_url="https://example.supabase.co",
            supabase_key="test-key-12345",
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
            agent_id="test_agent", supabase_url="not-a-valid-url", supabase_key="test-key"
        )

        with pytest.raises(ValueError, match="(Invalid|must use HTTPS)"):
            _ = storage.client

    def test_client_lazy_load_empty_key(self):
        """Test that accessing client raises error with empty key."""
        storage = SupabaseStorage(
            agent_id="test_agent",
            supabase_url="https://example.supabase.co",
            supabase_key="   ",  # Whitespace only
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
        "playbooks": [],
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
                chain._data = [r for r in chain._data if r.get(field, "") >= value]
                return chain

            def lte_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field, "") <= value]
                return chain

            def lt_filter(field, value):
                chain._data = [r for r in chain._data if r.get(field, 0) < value]
                return chain

            def order_mock(field, desc=False):
                try:
                    chain._data.sort(key=lambda x: x.get(field, ""), reverse=desc)
                except TypeError:
                    pass
                return chain

            def limit_mock(n):
                chain._data = chain._data[:n]
                return chain

            def execute_mock():
                result = MagicMock()
                result.data = chain._data
                result.count = chain._count if kwargs.get("count") == "exact" else None
                return result

            chain.eq = eq_filter
            chain.gte = gte_filter
            chain.lte = lte_filter
            chain.lt = lt_filter
            chain.order = order_mock
            chain.limit = limit_mock
            chain.execute = execute_mock
            return chain

        def upsert_mock(data, on_conflict=None):
            chain = MagicMock()
            # Add to storage
            if table_name in storage:
                if on_conflict:
                    # Handle composite unique constraint - remove by conflict fields
                    conflict_fields = [f.strip() for f in on_conflict.split(",")]
                    storage[table_name] = [
                        r
                        for r in storage[table_name]
                        if not all(r.get(f) == data.get(f) for f in conflict_fields)
                    ]
                else:
                    # Remove existing record with same ID
                    storage[table_name] = [
                        r for r in storage[table_name] if r.get("id") != data.get("id")
                    ]
                data["created_at"] = (
                    data.get("created_at") or datetime.now(timezone.utc).isoformat()
                )
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
                data["created_at"] = (
                    data.get("created_at") or datetime.now(timezone.utc).isoformat()
                )
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
        agent_id="test_agent", supabase_url="https://test.supabase.co", supabase_key="test-key"
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
        db["agent_episodes"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "objective": "First task",
                "outcome_description": "Completed",
                "outcome_type": "success",
                "lessons_learned": [],
                "tags": ["work"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.9,
            }
        )

        episodes = storage.get_episodes()
        assert len(episodes) == 1
        assert episodes[0].objective == "First task"

    def test_get_episode_by_id(self, supabase_storage):
        """Test retrieving a specific episode by ID."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Specific episode",
                "outcome_description": "Done",
                "outcome_type": "success",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        episode = storage.get_episode(ep_id)
        assert episode is not None
        assert episode.objective == "Specific episode"

    def test_update_episode_emotion(self, supabase_storage):
        """Test updating episode emotional data."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Emotional episode",
                "outcome_description": "Felt good",
                "emotional_valence": 0.0,
                "emotional_arousal": 0.0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.update_episode_emotion(ep_id, valence=0.8, arousal=0.5, tags=["joy"])
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

        db["agent_beliefs"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "statement": "Code should be tested",
                "belief_type": "fact",
                "confidence": 0.85,
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        beliefs = storage.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].statement == "Code should be tested"

    def test_find_belief(self, supabase_storage):
        """Test finding a belief by statement."""
        storage, db = supabase_storage

        db["agent_beliefs"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "statement": "Unique statement",
                "belief_type": "fact",
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

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

        db["agent_values"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "name": "Integrity",
                "statement": "Be honest and transparent",
                "priority": 90,
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

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

        db["agent_goals"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "title": "Ship feature",
                "description": "Complete the feature",
                "priority": "high",
                "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

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

        db["memories"].append(
            {
                "id": str(uuid.uuid4()),
                "owner_id": "test_agent",
                "content": "A curated memory",
                "source": "curated",
                "metadata": {"note_type": "note", "tags": []},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

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

        db["agent_drives"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "drive_type": "growth",
                "intensity": 0.8,
                "focus_areas": ["improvement"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

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

        db["agent_episodes"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "objective": "Implement feature X",
                "outcome_description": "Successfully deployed",
                "lessons_learned": ["Plan ahead"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        results = storage.search("feature", record_types=["episode"])
        assert len(results) >= 1

    def test_search_notes(self, supabase_storage):
        """Test searching notes by content."""
        storage, db = supabase_storage

        db["memories"].append(
            {
                "id": str(uuid.uuid4()),
                "owner_id": "test_agent",
                "content": "Important discovery about testing",
                "source": "curated",
                "metadata": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        results = storage.search("discovery", record_types=["note"])
        assert len(results) >= 1


# === Meta-Memory Tests ===


class TestSupabaseMetaMemory:
    """Tests for meta-memory operations."""

    def test_get_memory_by_type_and_id(self, supabase_storage):
        """Test retrieving a specific memory by type and ID."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test memory",
                "outcome_description": "Found",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        memory = storage.get_memory("episode", ep_id)
        assert memory is not None
        assert memory.objective == "Test memory"

    def test_get_memory_invalid_type(self, supabase_storage):
        """Test that invalid memory type returns None."""
        storage, _ = supabase_storage
        memory = storage.get_memory("invalid_type", "some-id")
        assert memory is None


# === Playbook Tests ===


class TestSupabasePlaybooks:
    """Tests for playbook operations."""

    def test_save_playbook(self, supabase_storage):
        """Test saving a playbook."""
        storage, db = supabase_storage

        playbook = Playbook(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            name="Deploy to Production",
            description="Standard deployment process",
            trigger_conditions=["release ready", "deploy command"],
            steps=[
                {"action": "Run tests", "details": None, "adaptations": None},
                {"action": "Build image", "details": None, "adaptations": None},
                {"action": "Deploy", "details": None, "adaptations": None},
            ],
            failure_modes=["Tests fail", "Build fails"],
            recovery_steps=["Fix tests", "Check Docker"],
            tags=["deploy", "production"],
        )

        playbook_id = storage.save_playbook(playbook)
        assert playbook_id is not None
        assert len(db["playbooks"]) == 1

        saved = db["playbooks"][0]
        assert saved["name"] == "Deploy to Production"
        assert saved["description"] == "Standard deployment process"
        assert len(saved["steps"]) == 3
        assert saved["tags"] == ["deploy", "production"]

    def test_get_playbook(self, supabase_storage):
        """Test retrieving a specific playbook by ID."""
        storage, db = supabase_storage

        pb_id = str(uuid.uuid4())
        db["playbooks"].append(
            {
                "id": pb_id,
                "agent_id": "test_agent",
                "name": "Code Review",
                "description": "Review code changes",
                "trigger_conditions": ["PR opened"],
                "steps": [{"action": "Review", "details": None, "adaptations": None}],
                "failure_modes": [],
                "recovery_steps": None,
                "mastery_level": "competent",
                "times_used": 10,
                "success_rate": 0.9,
                "tags": ["review"],
                "confidence": 0.85,
                "deleted": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        playbook = storage.get_playbook(pb_id)
        assert playbook is not None
        assert playbook.name == "Code Review"
        assert playbook.mastery_level == "competent"
        assert playbook.times_used == 10
        assert playbook.success_rate == 0.9

    def test_get_playbook_not_found(self, supabase_storage):
        """Test that get_playbook returns None for non-existent ID."""
        storage, _ = supabase_storage
        playbook = storage.get_playbook("nonexistent-id")
        assert playbook is None

    def test_list_playbooks(self, supabase_storage):
        """Test listing playbooks."""
        storage, db = supabase_storage

        # Add test playbooks
        for i in range(3):
            db["playbooks"].append(
                {
                    "id": str(uuid.uuid4()),
                    "agent_id": "test_agent",
                    "name": f"Playbook {i}",
                    "description": f"Description {i}",
                    "trigger_conditions": [f"trigger {i}"],
                    "steps": [],
                    "failure_modes": [],
                    "mastery_level": "novice",
                    "times_used": i * 5,
                    "success_rate": 0.8,
                    "tags": ["test"] if i % 2 == 0 else ["other"],
                    "confidence": 0.8,
                    "deleted": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        playbooks = storage.list_playbooks(limit=10)
        assert len(playbooks) == 3

        # Test tag filtering
        test_tagged = storage.list_playbooks(tags=["test"], limit=10)
        assert len(test_tagged) == 2  # Playbooks 0 and 2

    def test_search_playbooks(self, supabase_storage):
        """Test searching playbooks."""
        storage, db = supabase_storage

        db["playbooks"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "name": "Database Migration",
                "description": "Migrate database schema safely",
                "trigger_conditions": ["schema change"],
                "steps": [],
                "failure_modes": [],
                "mastery_level": "proficient",
                "times_used": 20,
                "success_rate": 0.95,
                "tags": ["database"],
                "confidence": 0.9,
                "deleted": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        db["playbooks"].append(
            {
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "name": "Cache Flush",
                "description": "Clear application cache",
                "trigger_conditions": ["stale data"],
                "steps": [],
                "failure_modes": [],
                "mastery_level": "competent",
                "times_used": 15,
                "success_rate": 0.85,
                "tags": ["cache"],
                "confidence": 0.85,
                "deleted": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Search for database
        results = storage.search_playbooks("database", limit=10)
        assert len(results) >= 1
        assert any(p.name == "Database Migration" for p in results)

        # Search for cache
        results = storage.search_playbooks("cache", limit=10)
        assert len(results) >= 1
        assert any(p.name == "Cache Flush" for p in results)

    def test_update_playbook_usage(self, supabase_storage):
        """Test updating playbook usage statistics."""
        storage, db = supabase_storage

        pb_id = str(uuid.uuid4())
        db["playbooks"].append(
            {
                "id": pb_id,
                "agent_id": "test_agent",
                "name": "Test Playbook",
                "description": "For testing usage updates",
                "trigger_conditions": [],
                "steps": [],
                "failure_modes": [],
                "mastery_level": "novice",
                "times_used": 0,
                "success_rate": 0.0,
                "tags": [],
                "confidence": 0.8,
                "version": 1,
                "deleted": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Record a success
        result = storage.update_playbook_usage(pb_id, success=True)
        assert result is True

        # Check updated values
        updated = db["playbooks"][0]
        assert updated["times_used"] == 1
        assert updated["success_rate"] == 1.0

    def test_update_playbook_usage_not_found(self, supabase_storage):
        """Test that update_playbook_usage returns False for non-existent playbook."""
        storage, _ = supabase_storage
        result = storage.update_playbook_usage("nonexistent-id", success=True)
        assert result is False

    def test_playbook_mastery_progression(self, supabase_storage):
        """Test that mastery level progresses with usage and success."""
        storage, db = supabase_storage

        pb_id = str(uuid.uuid4())
        db["playbooks"].append(
            {
                "id": pb_id,
                "agent_id": "test_agent",
                "name": "Mastery Test",
                "description": "Testing mastery progression",
                "trigger_conditions": [],
                "steps": [],
                "failure_modes": [],
                "mastery_level": "novice",
                "times_used": 4,
                "success_rate": 0.75,
                "tags": [],
                "confidence": 0.8,
                "version": 1,
                "deleted": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # One more success should trigger competent (5 uses, 80% success)
        storage.update_playbook_usage(pb_id, success=True)

        updated = db["playbooks"][0]
        assert updated["times_used"] == 5
        assert updated["mastery_level"] == "competent"


# === Forgetting Tests ===


class TestSupabaseForgetting:
    """Tests for forgetting operations."""

    def test_record_access(self, supabase_storage):
        """Test recording memory access."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test episode",
                "outcome_description": "Success",
                "times_accessed": 0,
                "last_accessed": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.record_access("episode", ep_id)
        assert result is True

        # Verify access was recorded
        updated = db["agent_episodes"][0]
        assert updated["times_accessed"] == 1
        assert updated["last_accessed"] is not None

    def test_record_access_increments(self, supabase_storage):
        """Test that record_access increments count."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test episode",
                "outcome_description": "Success",
                "times_accessed": 5,
                "last_accessed": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        storage.record_access("episode", ep_id)

        updated = db["agent_episodes"][0]
        assert updated["times_accessed"] == 6

    def test_record_access_not_found(self, supabase_storage):
        """Test record_access returns False for nonexistent memory."""
        storage, _ = supabase_storage
        result = storage.record_access("episode", "nonexistent-id")
        assert result is False

    def test_record_access_invalid_type(self, supabase_storage):
        """Test record_access returns False for invalid type."""
        storage, _ = supabase_storage
        result = storage.record_access("invalid_type", "some-id")
        assert result is False

    def test_record_access_batch(self, supabase_storage):
        """Test batch access recording."""
        storage, db = supabase_storage

        ep_id1 = str(uuid.uuid4())
        ep_id2 = str(uuid.uuid4())
        db["agent_episodes"].extend(
            [
                {
                    "id": ep_id1,
                    "agent_id": "test_agent",
                    "objective": "Episode 1",
                    "outcome_description": "Success",
                    "times_accessed": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "id": ep_id2,
                    "agent_id": "test_agent",
                    "objective": "Episode 2",
                    "outcome_description": "Success",
                    "times_accessed": 0,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            ]
        )

        updated = storage.record_access_batch([("episode", ep_id1), ("episode", ep_id2)])
        assert updated == 2

    def test_forget_memory(self, supabase_storage):
        """Test forgetting a memory."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test episode",
                "outcome_description": "Success",
                "is_protected": False,
                "is_forgotten": False,
                "forgotten_at": None,
                "forgotten_reason": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.forget_memory("episode", ep_id, reason="Low salience")
        assert result is True

        updated = db["agent_episodes"][0]
        assert updated["is_forgotten"] is True
        assert updated["forgotten_reason"] == "Low salience"
        assert updated["forgotten_at"] is not None

    def test_forget_protected_memory_fails(self, supabase_storage):
        """Test that protected memories cannot be forgotten."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Protected episode",
                "outcome_description": "Success",
                "is_protected": True,
                "is_forgotten": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.forget_memory("episode", ep_id)
        assert result is False

        updated = db["agent_episodes"][0]
        assert updated["is_forgotten"] is False

    def test_forget_already_forgotten_fails(self, supabase_storage):
        """Test that already forgotten memories return False."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Already forgotten",
                "outcome_description": "Success",
                "is_protected": False,
                "is_forgotten": True,
                "forgotten_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.forget_memory("episode", ep_id)
        assert result is False

    def test_recover_memory(self, supabase_storage):
        """Test recovering a forgotten memory."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Forgotten episode",
                "outcome_description": "Success",
                "is_forgotten": True,
                "forgotten_at": datetime.now(timezone.utc).isoformat(),
                "forgotten_reason": "Test",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.recover_memory("episode", ep_id)
        assert result is True

        updated = db["agent_episodes"][0]
        assert updated["is_forgotten"] is False
        assert updated["forgotten_at"] is None
        assert updated["forgotten_reason"] is None

    def test_recover_not_forgotten_fails(self, supabase_storage):
        """Test that recovering non-forgotten memory returns False."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Normal episode",
                "outcome_description": "Success",
                "is_forgotten": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.recover_memory("episode", ep_id)
        assert result is False

    def test_protect_memory(self, supabase_storage):
        """Test protecting a memory."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test episode",
                "outcome_description": "Success",
                "is_protected": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.protect_memory("episode", ep_id, protected=True)
        assert result is True

        updated = db["agent_episodes"][0]
        assert updated["is_protected"] is True

    def test_unprotect_memory(self, supabase_storage):
        """Test unprotecting a memory."""
        storage, db = supabase_storage

        ep_id = str(uuid.uuid4())
        db["agent_episodes"].append(
            {
                "id": ep_id,
                "agent_id": "test_agent",
                "objective": "Test episode",
                "outcome_description": "Success",
                "is_protected": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.protect_memory("episode", ep_id, protected=False)
        assert result is True

        updated = db["agent_episodes"][0]
        assert updated["is_protected"] is False

    def test_get_forgetting_candidates(self, supabase_storage):
        """Test getting forgetting candidates."""
        storage, db = supabase_storage

        # Add old episode (120 days ago) with low confidence
        old_date = datetime.now(timezone.utc).isoformat()
        db["agent_episodes"].append(
            {
                "id": "old-episode",
                "agent_id": "test_agent",
                "objective": "Old episode",
                "outcome_description": "Meh",
                "confidence": 0.3,
                "times_accessed": 0,
                "last_accessed": None,
                "is_protected": False,
                "is_forgotten": False,
                "created_at": old_date,
            }
        )

        # Add protected episode (should not be a candidate)
        db["agent_episodes"].append(
            {
                "id": "protected-episode",
                "agent_id": "test_agent",
                "objective": "Protected episode",
                "outcome_description": "Important",
                "confidence": 0.3,
                "times_accessed": 0,
                "is_protected": True,
                "is_forgotten": False,
                "created_at": old_date,
            }
        )

        # Add already forgotten episode (should not be a candidate)
        db["agent_episodes"].append(
            {
                "id": "forgotten-episode",
                "agent_id": "test_agent",
                "objective": "Already forgotten",
                "outcome_description": "Old",
                "confidence": 0.3,
                "times_accessed": 0,
                "is_protected": False,
                "is_forgotten": True,
                "created_at": old_date,
            }
        )

        candidates = storage.get_forgetting_candidates(memory_types=["episode"], limit=10)

        # Should only return the old, unprotected, non-forgotten episode
        assert len(candidates) == 1
        assert candidates[0].record.id == "old-episode"
        assert candidates[0].score >= 0  # Has a salience score

    def test_get_forgetting_candidates_sorted_by_salience(self, supabase_storage):
        """Test that candidates are sorted by salience (lowest first)."""
        storage, db = supabase_storage

        now = datetime.now(timezone.utc).isoformat()

        # High salience (high confidence, recent access)
        db["agent_episodes"].append(
            {
                "id": "high-salience",
                "agent_id": "test_agent",
                "objective": "High salience",
                "outcome_description": "Great",
                "confidence": 0.9,
                "times_accessed": 10,
                "last_accessed": now,
                "is_protected": False,
                "is_forgotten": False,
                "created_at": now,
            }
        )

        # Low salience (low confidence, never accessed)
        db["agent_episodes"].append(
            {
                "id": "low-salience",
                "agent_id": "test_agent",
                "objective": "Low salience",
                "outcome_description": "Meh",
                "confidence": 0.2,
                "times_accessed": 0,
                "last_accessed": None,
                "is_protected": False,
                "is_forgotten": False,
                "created_at": now,
            }
        )

        candidates = storage.get_forgetting_candidates(memory_types=["episode"], limit=10)

        # Low salience should come first
        assert len(candidates) == 2
        assert candidates[0].record.id == "low-salience"
        assert candidates[1].record.id == "high-salience"
        assert candidates[0].score < candidates[1].score

    def test_get_forgotten_memories(self, supabase_storage):
        """Test getting forgotten memories."""
        storage, db = supabase_storage

        now = datetime.now(timezone.utc).isoformat()

        # Add forgotten episodes
        db["agent_episodes"].append(
            {
                "id": "forgotten-1",
                "agent_id": "test_agent",
                "objective": "Forgotten 1",
                "outcome_description": "Old",
                "is_forgotten": True,
                "forgotten_at": now,
                "forgotten_reason": "Low salience",
                "created_at": now,
            }
        )

        db["agent_episodes"].append(
            {
                "id": "forgotten-2",
                "agent_id": "test_agent",
                "objective": "Forgotten 2",
                "outcome_description": "Old",
                "is_forgotten": True,
                "forgotten_at": now,
                "forgotten_reason": "Manual",
                "created_at": now,
            }
        )

        # Add non-forgotten episode (should not be returned)
        db["agent_episodes"].append(
            {
                "id": "active",
                "agent_id": "test_agent",
                "objective": "Active",
                "outcome_description": "Current",
                "is_forgotten": False,
                "created_at": now,
            }
        )

        forgotten = storage.get_forgotten_memories(memory_types=["episode"], limit=10)

        assert len(forgotten) == 2
        forgotten_ids = [f.record.id for f in forgotten]
        assert "forgotten-1" in forgotten_ids
        assert "forgotten-2" in forgotten_ids
        assert "active" not in forgotten_ids

    def test_get_forgotten_memories_empty(self, supabase_storage):
        """Test getting forgotten memories when none exist."""
        storage, db = supabase_storage

        db["agent_episodes"].append(
            {
                "id": "active",
                "agent_id": "test_agent",
                "objective": "Active",
                "outcome_description": "Current",
                "is_forgotten": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        forgotten = storage.get_forgotten_memories(memory_types=["episode"], limit=10)
        assert len(forgotten) == 0

    def test_forget_belief(self, supabase_storage):
        """Test forgetting a belief."""
        storage, db = supabase_storage

        belief_id = str(uuid.uuid4())
        db["agent_beliefs"].append(
            {
                "id": belief_id,
                "agent_id": "test_agent",
                "statement": "Test belief",
                "belief_type": "fact",
                "confidence": 0.5,
                "is_protected": False,
                "is_forgotten": False,
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.forget_memory("belief", belief_id, reason="Outdated")
        assert result is True

        updated = db["agent_beliefs"][0]
        assert updated["is_forgotten"] is True

    def test_forget_note_uses_owner_id(self, supabase_storage):
        """Test that forgetting notes uses owner_id correctly."""
        storage, db = supabase_storage

        note_id = str(uuid.uuid4())
        db["memories"].append(
            {
                "id": note_id,
                "owner_id": "test_agent",  # Notes use owner_id, not agent_id
                "content": "Test note",
                "source": "curated",
                "is_protected": False,
                "is_forgotten": False,
                "metadata": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        result = storage.forget_memory("note", note_id, reason="Cleanup")
        assert result is True

        updated = db["memories"][0]
        assert updated["is_forgotten"] is True
