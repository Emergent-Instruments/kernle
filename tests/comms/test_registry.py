"""Tests for agent registry."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kernle.comms.registry import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    AgentProfile,
    AgentRegistry,
)
from kernle.storage import SQLiteStorage


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
    storage = SQLiteStorage(agent_id="test-agent", db_path=temp_db)
    yield storage
    storage.close()


@pytest.fixture
def registry(storage):
    """Create an AgentRegistry instance for testing."""
    return AgentRegistry(storage)


class TestAgentProfile:
    """Tests for AgentProfile dataclass."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all fields."""
        profile = AgentProfile(
            agent_id="test-agent",
            user_id="user-123",
            display_name="Test Agent",
            capabilities=["code", "research"],
            public_key="pk_abc123",
            endpoints={"webhook": "https://example.com/hook"},
            trust_level="verified",
            reputation_score=0.75,
            is_public=True,
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            last_seen_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

        data = profile.to_dict()

        assert data["agent_id"] == "test-agent"
        assert data["user_id"] == "user-123"
        assert data["display_name"] == "Test Agent"
        assert data["capabilities"] == ["code", "research"]
        assert data["public_key"] == "pk_abc123"
        assert data["endpoints"] == {"webhook": "https://example.com/hook"}
        assert data["trust_level"] == "verified"
        assert data["reputation_score"] == 0.75
        assert data["is_public"] is True
        assert "2026-01-01" in data["registered_at"]
        assert "2026-02-01" in data["last_seen_at"]

    def test_from_dict_creates_profile(self):
        """Test that from_dict creates a valid profile."""
        data = {
            "agent_id": "test-agent",
            "user_id": "user-123",
            "display_name": "Test Agent",
            "capabilities": ["code", "research"],
            "trust_level": "verified",
            "reputation_score": 0.75,
            "is_public": True,
        }

        profile = AgentProfile.from_dict(data)

        assert profile.agent_id == "test-agent"
        assert profile.user_id == "user-123"
        assert profile.display_name == "Test Agent"
        assert profile.capabilities == ["code", "research"]
        assert profile.trust_level == "verified"
        assert profile.reputation_score == 0.75
        assert profile.is_public is True

    def test_from_dict_handles_missing_optional_fields(self):
        """Test that from_dict handles missing optional fields."""
        data = {
            "agent_id": "test-agent",
            "user_id": "user-123",
        }

        profile = AgentProfile.from_dict(data)

        assert profile.agent_id == "test-agent"
        assert profile.user_id == "user-123"
        assert profile.display_name is None
        assert profile.capabilities == []
        assert profile.public_key is None
        assert profile.endpoints == {}
        assert profile.trust_level == "unverified"
        assert profile.reputation_score == 0.0
        assert profile.is_public is False


class TestAgentRegistry:
    """Tests for AgentRegistry CRUD operations."""

    def test_register_creates_profile(self, registry):
        """Test that register creates a new profile."""
        profile = registry.register(
            agent_id="new-agent",
            user_id="user-123",
            display_name="New Agent",
            capabilities=["code", "research"],
            is_public=True,
        )

        assert profile.agent_id == "new-agent"
        assert profile.user_id == "user-123"
        assert profile.display_name == "New Agent"
        assert profile.capabilities == ["code", "research"]
        assert profile.is_public is True
        assert profile.trust_level == "unverified"
        assert profile.registered_at is not None

    def test_register_raises_for_duplicate(self, registry):
        """Test that register raises for duplicate agent_id."""
        registry.register(agent_id="dup-agent", user_id="user-123")

        with pytest.raises(AgentAlreadyExistsError):
            registry.register(agent_id="dup-agent", user_id="user-456")

    def test_get_profile_returns_profile(self, registry):
        """Test that get_profile returns existing profile."""
        registry.register(
            agent_id="get-agent",
            user_id="user-123",
            display_name="Get Agent",
        )

        profile = registry.get_profile("get-agent")

        assert profile is not None
        assert profile.agent_id == "get-agent"
        assert profile.display_name == "Get Agent"

    def test_get_profile_returns_none_for_missing(self, registry):
        """Test that get_profile returns None for missing agent."""
        profile = registry.get_profile("nonexistent")
        assert profile is None

    def test_update_profile_updates_fields(self, registry):
        """Test that update_profile updates specified fields."""
        registry.register(
            agent_id="update-agent",
            user_id="user-123",
            display_name="Original Name",
            capabilities=["code"],
        )

        updated = registry.update_profile(
            agent_id="update-agent",
            display_name="New Name",
            capabilities=["code", "research"],
            is_public=True,
        )

        assert updated.display_name == "New Name"
        assert updated.capabilities == ["code", "research"]
        assert updated.is_public is True

    def test_update_profile_preserves_unchanged_fields(self, registry):
        """Test that update_profile preserves fields not specified."""
        registry.register(
            agent_id="preserve-agent",
            user_id="user-123",
            display_name="Original Name",
            capabilities=["code", "research"],
        )

        updated = registry.update_profile(
            agent_id="preserve-agent",
            display_name="New Name",
        )

        assert updated.display_name == "New Name"
        assert updated.capabilities == ["code", "research"]  # Preserved

    def test_update_profile_raises_for_missing(self, registry):
        """Test that update_profile raises for missing agent."""
        with pytest.raises(AgentNotFoundError):
            registry.update_profile(agent_id="nonexistent", display_name="Name")

    def test_delete_profile_removes_agent(self, registry):
        """Test that delete_profile removes the agent."""
        registry.register(agent_id="delete-agent", user_id="user-123")

        result = registry.delete_profile("delete-agent")
        assert result is True

        profile = registry.get_profile("delete-agent")
        assert profile is None

    def test_delete_profile_returns_false_for_missing(self, registry):
        """Test that delete_profile returns False for missing agent."""
        result = registry.delete_profile("nonexistent")
        assert result is False


class TestAgentDiscovery:
    """Tests for agent discovery functionality."""

    def test_discover_returns_public_agents(self, registry):
        """Test that discover returns only public agents."""
        registry.register(
            agent_id="public-agent",
            user_id="user-123",
            is_public=True,
        )
        registry.register(
            agent_id="private-agent",
            user_id="user-456",
            is_public=False,
        )

        results = registry.discover()

        agent_ids = [p.agent_id for p in results]
        assert "public-agent" in agent_ids
        assert "private-agent" not in agent_ids

    def test_discover_filters_by_capability(self, registry):
        """Test that discover filters by capability."""
        registry.register(
            agent_id="code-agent",
            user_id="user-123",
            capabilities=["code", "review"],
            is_public=True,
        )
        registry.register(
            agent_id="research-agent",
            user_id="user-456",
            capabilities=["research"],
            is_public=True,
        )

        results = registry.discover(capabilities=["code"])

        agent_ids = [p.agent_id for p in results]
        assert "code-agent" in agent_ids
        assert "research-agent" not in agent_ids

    def test_discover_multiple_capabilities_match_any(self, registry):
        """Test that discover with multiple capabilities matches any."""
        registry.register(
            agent_id="code-agent",
            user_id="user-123",
            capabilities=["code"],
            is_public=True,
        )
        registry.register(
            agent_id="research-agent",
            user_id="user-456",
            capabilities=["research"],
            is_public=True,
        )
        registry.register(
            agent_id="other-agent",
            user_id="user-789",
            capabilities=["writing"],
            is_public=True,
        )

        results = registry.discover(capabilities=["code", "research"])

        agent_ids = [p.agent_id for p in results]
        assert "code-agent" in agent_ids
        assert "research-agent" in agent_ids
        assert "other-agent" not in agent_ids

    def test_list_all_returns_all_agents(self, registry):
        """Test that list_all returns all agents including private."""
        registry.register(
            agent_id="public-agent",
            user_id="user-123",
            is_public=True,
        )
        registry.register(
            agent_id="private-agent",
            user_id="user-456",
            is_public=False,
        )

        results = registry.list_all()

        agent_ids = [p.agent_id for p in results]
        assert "public-agent" in agent_ids
        assert "private-agent" in agent_ids


class TestLastSeenTracking:
    """Tests for last_seen_at tracking."""

    def test_register_sets_last_seen(self, registry):
        """Test that register sets last_seen_at."""
        profile = registry.register(agent_id="new-agent", user_id="user-123")
        assert profile.last_seen_at is not None

    def test_update_last_seen_updates_timestamp(self, registry):
        """Test that update_last_seen updates the timestamp."""
        registry.register(agent_id="seen-agent", user_id="user-123")
        profile1 = registry.get_profile("seen-agent")
        original_seen = profile1.last_seen_at

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        registry.update_last_seen("seen-agent")
        profile2 = registry.get_profile("seen-agent")

        assert profile2.last_seen_at >= original_seen
