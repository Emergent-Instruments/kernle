"""
Comprehensive tests for the Kernle core functionality.
"""

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kernle.core import Kernle


class TestKernleInitialization:
    """Test Kernle class initialization and setup."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        kernle = Kernle()
        assert kernle.agent_id == "default"  # Default when no env var
        assert kernle.checkpoint_dir == Path.home() / ".kernle" / "checkpoints"
        assert kernle._client is None  # Lazy loading
    
    def test_init_with_explicit_params(self, temp_checkpoint_dir):
        """Test initialization with explicit parameters."""
        kernle = Kernle(
            agent_id="test_agent",
            supabase_url="http://test.url",
            supabase_key="test_key",
            checkpoint_dir=temp_checkpoint_dir
        )
        assert kernle.agent_id == "test_agent"
        assert kernle.supabase_url == "http://test.url"
        assert kernle.supabase_key == "test_key"
        assert kernle.checkpoint_dir == temp_checkpoint_dir
    
    def test_init_with_env_vars(self, temp_checkpoint_dir):
        """Test initialization with environment variables."""
        with patch.dict(os.environ, {
            "KERNLE_AGENT_ID": "env_agent",
            "KERNLE_SUPABASE_URL": "http://env.url",
            "KERNLE_SUPABASE_KEY": "env_key",
        }):
            kernle = Kernle(checkpoint_dir=temp_checkpoint_dir)
            assert kernle.agent_id == "env_agent"
            assert kernle.supabase_url == "http://env.url"
            assert kernle.supabase_key == "env_key"
    
    def test_client_property_missing_credentials(self):
        """Test that client property raises error with missing credentials."""
        kernle = Kernle(agent_id="test")
        
        with pytest.raises(ValueError, match="Supabase credentials required"):
            _ = kernle.client
    
    @patch('kernle.core.create_client')
    def test_client_property_lazy_loading(self, mock_create_client):
        """Test that client is lazily loaded."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        kernle = Kernle(
            agent_id="test",
            supabase_url="http://test.url",
            supabase_key="test_key"
        )
        
        # Client not created yet
        assert kernle._client is None
        mock_create_client.assert_not_called()
        
        # Access client - should create it
        client = kernle.client
        assert client is mock_client
        assert kernle._client is mock_client
        mock_create_client.assert_called_once_with("http://test.url", "test_key")
        
        # Second access - should reuse existing client
        client2 = kernle.client
        assert client2 is mock_client
        mock_create_client.assert_called_once()  # Still only called once


class TestLoadMethods:
    """Test various load methods."""
    
    def test_load_full_context(self, kernle_instance, populated_storage):
        """Test loading full working memory context."""
        kernle, storage = kernle_instance
        
        memory = kernle.load()
        
        assert "values" in memory
        assert "beliefs" in memory
        assert "goals" in memory
        assert "drives" in memory
        assert "lessons" in memory
        assert "recent_work" in memory
        assert "recent_notes" in memory
        assert "relationships" in memory
        assert "checkpoint" in memory
    
    def test_load_values(self, kernle_instance, populated_storage):
        """Test loading agent values."""
        kernle, storage = kernle_instance
        
        values = kernle.load_values(limit=5)
        
        assert len(values) == 1  # One value in sample data
        assert values[0]["name"] == "Quality"
        assert values[0]["statement"] == "Software should be thoroughly tested and reliable"
        assert values[0]["priority"] == 80
        assert values[0]["value_type"] == "core_value"
    
    def test_load_beliefs(self, kernle_instance, populated_storage):
        """Test loading agent beliefs.""" 
        kernle, storage = kernle_instance
        
        beliefs = kernle.load_beliefs(limit=10)
        
        assert len(beliefs) == 1
        assert beliefs[0]["statement"] == "Comprehensive testing leads to more reliable software"
        assert beliefs[0]["belief_type"] == "fact"
        assert beliefs[0]["confidence"] == 0.9
    
    def test_load_goals(self, kernle_instance, populated_storage):
        """Test loading active goals."""
        kernle, storage = kernle_instance
        
        goals = kernle.load_goals(limit=5)
        
        assert len(goals) == 1
        assert goals[0]["title"] == "Achieve 80%+ test coverage"
        assert goals[0]["status"] == "active"
        assert goals[0]["priority"] == "high"
    
    def test_load_lessons(self, kernle_instance, populated_storage):
        """Test extracting lessons from reflected episodes."""
        kernle, storage = kernle_instance
        
        lessons = kernle.load_lessons(limit=10)
        
        # Should extract lessons from reflected episodes
        assert "Always test edge cases" in lessons
        assert "Mock external dependencies" in lessons
        # Should not include lessons from unreflected episodes
        assert "Need better monitoring tools" not in lessons
    
    def test_load_recent_work_filters_checkpoints(self, kernle_instance, populated_storage):
        """Test that recent work excludes checkpoint episodes."""
        kernle, storage = kernle_instance
        
        recent_work = kernle.load_recent_work(limit=5)
        
        # Should have 2 episodes (excluding the one tagged with "checkpoint")
        assert len(recent_work) == 2
        
        # Verify no checkpoint episodes
        for episode in recent_work:
            tags = episode.get("tags", [])
            assert "checkpoint" not in tags
    
    def test_load_recent_notes(self, kernle_instance, populated_storage):
        """Test loading recent curated notes."""
        kernle, storage = kernle_instance
        
        notes = kernle.load_recent_notes(limit=5)
        
        assert len(notes) == 2  # Two notes in sample data
        assert any("Decision" in note["content"] for note in notes)
        assert any("Insight" in note["content"] for note in notes)
    
    def test_load_drives(self, kernle_instance, populated_storage):
        """Test loading drive states."""
        kernle, storage = kernle_instance
        
        drives = kernle.load_drives()
        
        assert len(drives) == 1
        assert drives[0]["drive_type"] == "growth"
        assert drives[0]["intensity"] == 0.7
        assert drives[0]["focus_areas"] == ["learning", "improvement"]


class TestCheckpoints:
    """Test checkpoint save/load/clear functionality."""
    
    def test_checkpoint_save_basic(self, kernle_instance):
        """Test basic checkpoint saving."""
        kernle, storage = kernle_instance
        
        checkpoint_data = kernle.checkpoint(
            task="Write tests",
            pending=["Test CLI", "Test edge cases"],
            context="Working on comprehensive test suite"
        )
        
        assert checkpoint_data["agent_id"] == "test_agent"
        assert checkpoint_data["current_task"] == "Write tests"
        assert checkpoint_data["pending"] == ["Test CLI", "Test edge cases"]
        assert checkpoint_data["context"] == "Working on comprehensive test suite"
        assert "timestamp" in checkpoint_data
        
        # Should also save to file
        checkpoint_file = kernle.checkpoint_dir / "test_agent.json"
        assert checkpoint_file.exists()
        
        with open(checkpoint_file) as f:
            saved_data = json.load(f)
        
        assert isinstance(saved_data, list)
        assert len(saved_data) == 1
        assert saved_data[0]["current_task"] == "Write tests"
    
    def test_checkpoint_multiple_saves(self, kernle_instance):
        """Test that multiple checkpoints are stored as list."""
        kernle, storage = kernle_instance
        
        # Save multiple checkpoints
        kernle.checkpoint("Task 1")
        kernle.checkpoint("Task 2") 
        kernle.checkpoint("Task 3")
        
        checkpoint_file = kernle.checkpoint_dir / "test_agent.json"
        with open(checkpoint_file) as f:
            saved_data = json.load(f)
        
        assert isinstance(saved_data, list)
        assert len(saved_data) == 3
        assert saved_data[0]["current_task"] == "Task 1"
        assert saved_data[2]["current_task"] == "Task 3"
    
    def test_checkpoint_history_limit(self, kernle_instance):
        """Test that checkpoint history is limited to last 10."""
        kernle, storage = kernle_instance
        
        # Save 12 checkpoints
        for i in range(12):
            kernle.checkpoint(f"Task {i}")
        
        checkpoint_file = kernle.checkpoint_dir / "test_agent.json"
        with open(checkpoint_file) as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 10  # Should keep only last 10
        assert saved_data[0]["current_task"] == "Task 2"  # First two dropped
        assert saved_data[9]["current_task"] == "Task 11"
    
    def test_load_checkpoint_exists(self, kernle_instance):
        """Test loading existing checkpoint."""
        kernle, storage = kernle_instance
        
        # Save a checkpoint first
        original = kernle.checkpoint("Test task", ["item1", "item2"], "Test context")
        
        # Load it back
        loaded = kernle.load_checkpoint()
        
        assert loaded is not None
        assert loaded["current_task"] == "Test task"
        assert loaded["pending"] == ["item1", "item2"]
        assert loaded["context"] == "Test context"
    
    def test_load_checkpoint_not_exists(self, kernle_instance):
        """Test loading checkpoint when none exists."""
        kernle, storage = kernle_instance
        
        loaded = kernle.load_checkpoint()
        assert loaded is None
    
    def test_load_checkpoint_corrupted_file(self, kernle_instance):
        """Test loading checkpoint with corrupted file."""
        kernle, storage = kernle_instance
        
        # Create corrupted checkpoint file
        checkpoint_file = kernle.checkpoint_dir / "test_agent.json"
        with open(checkpoint_file, "w") as f:
            f.write("invalid json{")
        
        # Should return None for corrupted file
        loaded = kernle.load_checkpoint()
        assert loaded is None
    
    def test_clear_checkpoint_exists(self, kernle_instance):
        """Test clearing existing checkpoint."""
        kernle, storage = kernle_instance
        
        # Create checkpoint first
        kernle.checkpoint("Test task")
        checkpoint_file = kernle.checkpoint_dir / "test_agent.json"
        assert checkpoint_file.exists()
        
        # Clear it
        result = kernle.clear_checkpoint()
        assert result is True
        assert not checkpoint_file.exists()
    
    def test_clear_checkpoint_not_exists(self, kernle_instance):
        """Test clearing checkpoint when none exists."""
        kernle, storage = kernle_instance
        
        result = kernle.clear_checkpoint()
        assert result is False


class TestEpisodes:
    """Test episode recording and management."""
    
    def test_episode_basic(self, kernle_instance):
        """Test basic episode recording."""
        kernle, storage = kernle_instance
        
        episode_id = kernle.episode(
            objective="Fix bug in authentication",
            outcome="success",
            lessons=["Always validate user input", "Check edge cases"],
            repeat=["Thorough testing"],
            avoid=["Quick fixes"],
            tags=["bug-fix", "auth"]
        )
        
        assert len(episode_id) > 0
        
        # Check that episode was saved
        episodes = storage["agent_episodes"]
        saved_episode = next(ep for ep in episodes if ep["id"] == episode_id)
        
        assert saved_episode["agent_id"] == "test_agent"
        assert saved_episode["objective"] == "Fix bug in authentication"
        assert saved_episode["outcome_type"] == "success"
        assert saved_episode["outcome_description"] == "success"
        assert saved_episode["lessons_learned"] == ["Always validate user input", "Check edge cases"]
        assert saved_episode["patterns_to_repeat"] == ["Thorough testing"]
        assert saved_episode["patterns_to_avoid"] == ["Quick fixes"]
        assert saved_episode["tags"] == ["bug-fix", "auth"]
        assert saved_episode["is_reflected"] is True
        assert saved_episode["confidence"] == 0.8
    
    def test_episode_outcome_type_detection(self, kernle_instance):
        """Test automatic outcome type detection."""
        kernle, storage = kernle_instance
        
        # Test success detection
        success_id = kernle.episode("Task 1", "completed")
        success_episode = next(ep for ep in storage["agent_episodes"] if ep["id"] == success_id)
        assert success_episode["outcome_type"] == "success"
        
        # Test failure detection
        failure_id = kernle.episode("Task 2", "failed")
        failure_episode = next(ep for ep in storage["agent_episodes"] if ep["id"] == failure_id)
        assert failure_episode["outcome_type"] == "failure"
        
        # Test partial/other
        partial_id = kernle.episode("Task 3", "in progress")
        partial_episode = next(ep for ep in storage["agent_episodes"] if ep["id"] == partial_id)
        assert partial_episode["outcome_type"] == "partial"
    
    def test_episode_minimal(self, kernle_instance):
        """Test episode with minimal data."""
        kernle, storage = kernle_instance
        
        episode_id = kernle.episode("Simple task", "done")
        
        saved_episode = next(ep for ep in storage["agent_episodes"] if ep["id"] == episode_id)
        assert saved_episode["lessons_learned"] == []
        assert saved_episode["patterns_to_repeat"] == []
        assert saved_episode["patterns_to_avoid"] == []
        assert saved_episode["tags"] == ["manual"]


class TestNotes:
    """Test note capture functionality."""
    
    def test_note_basic(self, kernle_instance):
        """Test basic note capture."""
        kernle, storage = kernle_instance
        
        note_id = kernle.note(
            content="This is a test note",
            type="note",
            tags=["test", "example"]
        )
        
        assert len(note_id) > 0
        
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["owner_id"] == "test_agent"
        assert saved_note["content"] == "This is a test note"
        assert saved_note["source"] == "curated"
        assert saved_note["metadata"]["note_type"] == "note"
        assert saved_note["metadata"]["tags"] == ["test", "example"]
        assert saved_note["is_curated"] is True
        assert saved_note["is_protected"] is False
    
    def test_note_decision(self, kernle_instance):
        """Test decision note with reason."""
        kernle, storage = kernle_instance
        
        note_id = kernle.note(
            content="Use PostgreSQL for the database",
            type="decision",
            reason="Better performance for complex queries",
            tags=["architecture"]
        )
        
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["content"] == "**Decision**: Use PostgreSQL for the database\n**Reason**: Better performance for complex queries"
        assert saved_note["metadata"]["note_type"] == "decision"
        assert saved_note["metadata"]["reason"] == "Better performance for complex queries"
    
    def test_note_quote(self, kernle_instance):
        """Test quote note with speaker."""
        kernle, storage = kernle_instance
        
        note_id = kernle.note(
            content="The best code is no code at all",
            type="quote",
            speaker="Jeff Atwood",
            tags=["wisdom"]
        )
        
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["content"] == '> "The best code is no code at all"\n> — Jeff Atwood'
        assert saved_note["metadata"]["speaker"] == "Jeff Atwood"
    
    def test_note_insight(self, kernle_instance):
        """Test insight note."""
        kernle, storage = kernle_instance
        
        note_id = kernle.note(
            content="Testing first prevents bugs later",
            type="insight"
        )
        
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["content"] == "**Insight**: Testing first prevents bugs later"
        assert saved_note["metadata"]["note_type"] == "insight"
    
    def test_note_protected(self, kernle_instance):
        """Test protected note."""
        kernle, storage = kernle_instance
        
        note_id = kernle.note("Important info", protect=True)
        
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["is_protected"] is True


class TestBeliefValueGoal:
    """Test belief, value, and goal creation."""
    
    def test_belief_creation(self, kernle_instance):
        """Test creating a belief."""
        kernle, storage = kernle_instance
        
        belief_id = kernle.belief(
            statement="Unit tests improve code quality",
            type="principle",
            confidence=0.95,
            foundational=True
        )
        
        saved_belief = next(b for b in storage["agent_beliefs"] if b["id"] == belief_id)
        assert saved_belief["agent_id"] == "test_agent"
        assert saved_belief["statement"] == "Unit tests improve code quality"
        assert saved_belief["belief_type"] == "principle"
        assert saved_belief["confidence"] == 0.95
        assert saved_belief["is_foundational"] is True
        assert saved_belief["is_active"] is True
    
    def test_belief_defaults(self, kernle_instance):
        """Test belief creation with defaults."""
        kernle, storage = kernle_instance
        
        belief_id = kernle.belief("Simple fact")
        
        saved_belief = next(b for b in storage["agent_beliefs"] if b["id"] == belief_id)
        assert saved_belief["belief_type"] == "fact"
        assert saved_belief["confidence"] == 0.8
        assert saved_belief["is_foundational"] is False
    
    def test_value_creation(self, kernle_instance):
        """Test creating a value."""
        kernle, storage = kernle_instance
        
        value_id = kernle.value(
            name="Excellence",
            statement="Strive for the highest quality in all work",
            priority=90,
            type="core_principle",
            foundational=True
        )
        
        saved_value = next(v for v in storage["agent_values"] if v["id"] == value_id)
        assert saved_value["agent_id"] == "test_agent"
        assert saved_value["name"] == "Excellence"
        assert saved_value["statement"] == "Strive for the highest quality in all work"
        assert saved_value["priority"] == 90
        assert saved_value["value_type"] == "core_principle"
        assert saved_value["is_foundational"] is True
    
    def test_value_defaults(self, kernle_instance):
        """Test value creation with defaults."""
        kernle, storage = kernle_instance
        
        value_id = kernle.value("Simplicity", "Keep things simple")
        
        saved_value = next(v for v in storage["agent_values"] if v["id"] == value_id)
        assert saved_value["priority"] == 50
        assert saved_value["value_type"] == "core_value"
        assert saved_value["is_foundational"] is False
    
    def test_goal_creation(self, kernle_instance):
        """Test creating a goal."""
        kernle, storage = kernle_instance
        
        goal_id = kernle.goal(
            title="Complete comprehensive testing",
            description="Write and run all tests for the Kernle system",
            priority="high"
        )
        
        saved_goal = next(g for g in storage["agent_goals"] if g["id"] == goal_id)
        assert saved_goal["agent_id"] == "test_agent"
        assert saved_goal["title"] == "Complete comprehensive testing"
        assert saved_goal["description"] == "Write and run all tests for the Kernle system"
        assert saved_goal["priority"] == "high"
        assert saved_goal["status"] == "active"
    
    def test_goal_defaults(self, kernle_instance):
        """Test goal creation with defaults."""
        kernle, storage = kernle_instance
        
        goal_id = kernle.goal("Simple goal")
        
        saved_goal = next(g for g in storage["agent_goals"] if g["id"] == goal_id)
        assert saved_goal["description"] == "Simple goal"  # Defaults to title
        assert saved_goal["priority"] == "medium"


class TestSearch:
    """Test search functionality across different memory types."""
    
    def test_search_episodes(self, kernle_instance, populated_storage):
        """Test searching in episodes."""
        kernle, storage = kernle_instance
        
        results = kernle.search("unit tests")
        
        # Should find episode with "unit tests" in objective
        episode_results = [r for r in results if r["type"] == "episode"]
        assert len(episode_results) >= 1
        
        episode_result = episode_results[0]
        assert "unit tests" in episode_result["title"].lower()
        assert episode_result["type"] == "episode"
        assert "lessons" in episode_result
        assert "date" in episode_result
    
    def test_search_notes(self, kernle_instance, populated_storage):
        """Test searching in notes."""
        kernle, storage = kernle_instance
        
        results = kernle.search("testing framework")
        
        # Should find note about pytest
        note_results = [r for r in results if r["type"] in ["note", "decision", "insight"]]
        assert len(note_results) >= 1
        
        # Should find the decision about pytest
        decision_result = next((r for r in note_results if "pytest" in r["content"]), None)
        assert decision_result is not None
        assert "tags" in decision_result
    
    def test_search_beliefs(self, kernle_instance, populated_storage):
        """Test searching in beliefs."""
        kernle, storage = kernle_instance
        
        results = kernle.search("testing")
        
        # Should find belief about testing
        belief_results = [r for r in results if r["type"] == "belief"]
        assert len(belief_results) >= 1
        
        belief_result = belief_results[0]
        assert "testing" in belief_result["content"].lower()
        assert "confidence" in belief_result
    
    def test_search_no_results(self, kernle_instance, populated_storage):
        """Test search with no matching results."""
        kernle, storage = kernle_instance
        
        results = kernle.search("nonexistent_query_12345")
        assert len(results) == 0
    
    def test_search_case_insensitive(self, kernle_instance, populated_storage):
        """Test that search is case insensitive."""
        kernle, storage = kernle_instance
        
        results_lower = kernle.search("testing")
        results_upper = kernle.search("TESTING")
        results_mixed = kernle.search("TeStInG")
        
        # Should return same results regardless of case
        assert len(results_lower) == len(results_upper) == len(results_mixed)
    
    def test_search_limit(self, kernle_instance, populated_storage):
        """Test search result limiting."""
        kernle, storage = kernle_instance
        
        # Add more data to test limit
        for i in range(15):
            storage["memories"].append({
                "id": str(uuid.uuid4()),
                "owner_id": "test_agent",
                "content": f"Test content {i} with search term",
                "source": "curated",
                "metadata": {"note_type": "note"},
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        
        results = kernle.search("search term", limit=5)
        assert len(results) <= 5


class TestStatus:
    """Test status reporting."""
    
    def test_status_with_data(self, kernle_instance, populated_storage):
        """Test status with existing data."""
        kernle, storage = kernle_instance
        
        # Add a checkpoint
        kernle.checkpoint("Test task")
        
        status = kernle.status()
        
        assert status["agent_id"] == "test_agent"
        assert status["values"] == 1
        assert status["beliefs"] == 1
        assert status["goals"] == 1
        assert status["episodes"] == 3  # From populated storage
        assert status["checkpoint"] is True
    
    def test_status_empty(self, kernle_instance):
        """Test status with no data."""
        kernle, storage = kernle_instance
        
        status = kernle.status()
        
        assert status["agent_id"] == "test_agent"
        assert status["values"] == 0
        assert status["beliefs"] == 0
        assert status["goals"] == 0
        assert status["episodes"] == 0
        assert status["checkpoint"] is False


class TestDrives:
    """Test drive system functionality."""
    
    def test_drive_creation(self, kernle_instance):
        """Test creating a new drive."""
        kernle, storage = kernle_instance
        
        drive_id = kernle.drive(
            drive_type="curiosity",
            intensity=0.8,
            focus_areas=["machine learning", "AI safety"],
            decay_hours=48
        )
        
        saved_drive = next(d for d in storage["agent_drives"] if d["id"] == drive_id)
        assert saved_drive["agent_id"] == "test_agent"
        assert saved_drive["drive_type"] == "curiosity"
        assert saved_drive["intensity"] == 0.8
        assert saved_drive["focus_areas"] == ["machine learning", "AI safety"]
        assert saved_drive["satisfaction_decay_hours"] == 48
    
    def test_drive_update_existing(self, kernle_instance, populated_storage):
        """Test updating an existing drive."""
        kernle, storage = kernle_instance
        
        # Update existing growth drive
        drive_id = kernle.drive("growth", 0.9, ["testing", "development"])
        
        # Should update existing drive, not create new one
        growth_drives = [d for d in storage["agent_drives"] if d["drive_type"] == "growth"]
        assert len(growth_drives) == 1
        
        # Check updated values
        updated_drive = growth_drives[0]
        assert updated_drive["intensity"] == 0.9
        assert updated_drive["focus_areas"] == ["testing", "development"]
    
    def test_drive_invalid_type(self, kernle_instance):
        """Test that invalid drive type raises error."""
        kernle, storage = kernle_instance
        
        with pytest.raises(ValueError, match="Invalid drive type"):
            kernle.drive("invalid_drive", 0.5)
    
    def test_drive_intensity_bounds(self, kernle_instance):
        """Test that drive intensity is clamped to valid bounds."""
        kernle, storage = kernle_instance
        
        # Test upper bound
        drive_id1 = kernle.drive("existence", 1.5)  # Should clamp to 1.0
        saved_drive1 = next(d for d in storage["agent_drives"] if d["id"] == drive_id1)
        assert saved_drive1["intensity"] == 1.0
        
        # Test lower bound
        drive_id2 = kernle.drive("connection", -0.5)  # Should clamp to 0.0
        saved_drive2 = next(d for d in storage["agent_drives"] if d["id"] == drive_id2)
        assert saved_drive2["intensity"] == 0.0
    
    def test_satisfy_drive_existing(self, kernle_instance, populated_storage):
        """Test satisfying an existing drive."""
        kernle, storage = kernle_instance
        
        result = kernle.satisfy_drive("growth", 0.3)
        
        assert result is True
        
        # Check that intensity was reduced
        growth_drive = next(d for d in storage["agent_drives"] if d["drive_type"] == "growth")
        assert growth_drive["intensity"] == 0.4  # 0.7 - 0.3
    
    def test_satisfy_drive_minimum_intensity(self, kernle_instance, populated_storage):
        """Test that drive satisfaction respects minimum intensity."""
        kernle, storage = kernle_instance
        
        # Try to satisfy by a large amount
        kernle.satisfy_drive("growth", 0.8)
        
        # Should not go below 0.1
        growth_drive = next(d for d in storage["agent_drives"] if d["drive_type"] == "growth")
        assert growth_drive["intensity"] == 0.1
    
    def test_satisfy_drive_nonexistent(self, kernle_instance):
        """Test satisfying a nonexistent drive."""
        kernle, storage = kernle_instance
        
        result = kernle.satisfy_drive("nonexistent", 0.2)
        assert result is False


class TestRelationships:
    """Test relational memory functionality."""
    
    def test_relationship_creation(self, kernle_instance):
        """Test creating a new relationship."""
        kernle, storage = kernle_instance
        
        # Mock successful table insert
        rel_id = kernle.relationship(
            other_agent_id="other_agent",
            trust_level=0.8,
            notes="Collaborative AI researcher",
            interaction_type="professional"
        )
        
        # Note: Since agent_relationships table might not exist, 
        # this might fall back to storing as a note
        # Check if stored in relationships or as note
        if storage.get("agent_relationships"):
            saved_rel = next(r for r in storage["agent_relationships"] if r["id"] == rel_id)
            assert saved_rel["agent_id"] == "test_agent"
            assert saved_rel["other_agent_id"] == "other_agent"
            assert saved_rel["trust_level"] == 0.8
            assert saved_rel["notes"] == "Collaborative AI researcher"
    
    def test_load_relationships_empty(self, kernle_instance):
        """Test loading relationships when none exist or table doesn't exist."""
        kernle, storage = kernle_instance
        
        relationships = kernle.load_relationships()
        
        # Should return empty list if table doesn't exist or is empty
        assert isinstance(relationships, list)
        assert len(relationships) == 0


class TestTemporal:
    """Test temporal memory queries."""
    
    def test_load_temporal_default_range(self, kernle_instance, populated_storage):
        """Test loading temporal memories with default range (today)."""
        kernle, storage = kernle_instance
        
        result = kernle.load_temporal()
        
        assert "range" in result
        assert "episodes" in result
        assert "notes" in result
        assert isinstance(result["episodes"], list)
        assert isinstance(result["notes"], list)
        
        # Check date range format
        assert result["range"]["start"] is not None
        assert result["range"]["end"] is not None
    
    def test_load_temporal_custom_range(self, kernle_instance, populated_storage):
        """Test loading temporal memories with custom date range."""
        kernle, storage = kernle_instance
        
        start = datetime.now(timezone.utc) - timedelta(days=1)
        end = datetime.now(timezone.utc)
        
        result = kernle.load_temporal(start, end, limit=5)
        
        assert result["range"]["start"] == start.isoformat()
        assert result["range"]["end"] == end.isoformat()
    
    def test_what_happened_today(self, kernle_instance, populated_storage):
        """Test 'what happened today' query."""
        kernle, storage = kernle_instance
        
        result = kernle.what_happened("today")
        
        assert "range" in result
        assert "episodes" in result
        assert "notes" in result
    
    def test_what_happened_yesterday(self, kernle_instance, populated_storage):
        """Test 'what happened yesterday' query."""
        kernle, storage = kernle_instance
        
        result = kernle.what_happened("yesterday")
        
        # Should return data for yesterday
        start_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        result_start_date = datetime.fromisoformat(result["range"]["start"].replace('Z', '+00:00')).date()
        assert result_start_date == start_date
    
    def test_what_happened_invalid_period(self, kernle_instance, populated_storage):
        """Test 'what happened' with invalid period falls back to today."""
        kernle, storage = kernle_instance
        
        result = kernle.what_happened("invalid_period")
        
        # Should default to today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        result_start = datetime.fromisoformat(result["range"]["start"].replace('Z', '+00:00'))
        assert result_start.date() == today_start.date()


class TestSignalDetection:
    """Test automatic significance detection."""
    
    def test_detect_significance_positive(self, kernle_instance):
        """Test detecting positive significance signals."""
        kernle, storage = kernle_instance
        
        detection = kernle.detect_significance("The task completed successfully and works perfectly")
        
        assert detection["significant"] is True
        assert detection["score"] >= 0.6
        assert len(detection["signals"]) > 0
        
        # Should detect success signal
        success_signals = [s for s in detection["signals"] if s["signal"] == "success"]
        assert len(success_signals) > 0
    
    def test_detect_significance_failure(self, kernle_instance):
        """Test detecting failure significance signals."""
        kernle, storage = kernle_instance
        
        detection = kernle.detect_significance("The code failed with an error and doesn't work")
        
        assert detection["significant"] is True
        assert detection["score"] >= 0.6
        
        # Should detect failure signal
        failure_signals = [s for s in detection["signals"] if s["signal"] == "failure"]
        assert len(failure_signals) > 0
    
    def test_detect_significance_decision(self, kernle_instance):
        """Test detecting decision significance signals."""
        kernle, storage = kernle_instance
        
        detection = kernle.detect_significance("I decided to use React for the frontend")
        
        assert detection["significant"] is True
        assert detection["score"] >= 0.6
        
        # Should detect decision signal  
        decision_signals = [s for s in detection["signals"] if s["signal"] == "decision"]
        assert len(decision_signals) > 0
    
    def test_detect_significance_lesson(self, kernle_instance):
        """Test detecting lesson significance signals."""
        kernle, storage = kernle_instance
        
        detection = kernle.detect_significance("I learned that testing early prevents bugs")
        
        assert detection["significant"] is True
        assert detection["score"] >= 0.6
        
        # Should detect lesson signal with highest weight
        lesson_signals = [s for s in detection["signals"] if s["signal"] == "lesson"]
        assert len(lesson_signals) > 0
        assert detection["score"] == 0.9  # Lesson has weight 0.9
    
    def test_detect_significance_insignificant(self, kernle_instance):
        """Test detecting insignificant content."""
        kernle, storage = kernle_instance
        
        detection = kernle.detect_significance("Just a normal conversation about the weather")
        
        assert detection["significant"] is False
        assert detection["score"] < 0.6
        assert len(detection["signals"]) == 0
    
    def test_auto_capture_significant_decision(self, kernle_instance):
        """Test auto-capturing significant decision."""
        kernle, storage = kernle_instance
        
        note_id = kernle.auto_capture(
            "I decided to use PostgreSQL for better query performance",
            context="Database selection"
        )
        
        assert note_id is not None
        
        # Should create a decision note
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["metadata"]["note_type"] == "decision"
        assert "auto-captured" in saved_note["metadata"]["tags"]
    
    def test_auto_capture_significant_lesson(self, kernle_instance):
        """Test auto-capturing significant lesson."""
        kernle, storage = kernle_instance
        
        note_id = kernle.auto_capture(
            "I learned that mocking external services makes tests more reliable",
            context="Testing practices"
        )
        
        assert note_id is not None
        
        # Should create an insight note
        saved_note = next(note for note in storage["memories"] if note["id"] == note_id)
        assert saved_note["metadata"]["note_type"] == "insight"
    
    def test_auto_capture_insignificant(self, kernle_instance):
        """Test that insignificant content is not auto-captured."""
        kernle, storage = kernle_instance
        
        result = kernle.auto_capture("Just a regular comment with no special significance")
        
        assert result is None


class TestConsolidation:
    """Test memory consolidation functionality."""
    
    def test_consolidate_insufficient_episodes(self, kernle_instance):
        """Test consolidation with insufficient episodes."""
        kernle, storage = kernle_instance
        
        # Add only 1 unreflected episode
        storage["agent_episodes"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent",
            "objective": "Test task",
            "outcome_description": "Test outcome",
            "lessons_learned": ["Test lesson"],
            "is_reflected": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        
        result = kernle.consolidate(min_episodes=3)
        
        assert result["consolidated"] == 0
        assert "Need 3 episodes" in result["message"]
    
    def test_consolidate_creates_beliefs(self, kernle_instance):
        """Test that consolidation creates beliefs from repeated lessons."""
        kernle, storage = kernle_instance
        
        # Add multiple episodes with repeated lessons
        repeated_lesson = "Always validate user input"
        for i in range(3):
            storage["agent_episodes"].append({
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "objective": f"Task {i}",
                "outcome_description": "Success",
                "lessons_learned": [repeated_lesson, f"Specific lesson {i}"],
                "is_reflected": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        
        result = kernle.consolidate(min_episodes=3)
        
        assert result["consolidated"] == 3
        assert result["new_beliefs"] >= 1  # Should create belief from repeated lesson
        assert result["lessons_found"] == 6  # 2 lessons per episode * 3 episodes
        
        # Check that episodes are marked as reflected
        unreflected = [ep for ep in storage["agent_episodes"] if not ep.get("is_reflected", True)]
        assert len(unreflected) == 0
        
        # Check that belief was created
        new_beliefs = [b for b in storage["agent_beliefs"] 
                      if repeated_lesson in b.get("statement", "")]
        assert len(new_beliefs) >= 1
    
    def test_consolidate_avoids_duplicate_beliefs(self, kernle_instance):
        """Test that consolidation doesn't create duplicate beliefs."""
        kernle, storage = kernle_instance
        
        lesson = "Test-driven development improves code quality"
        
        # Add existing belief
        storage["agent_beliefs"].append({
            "id": str(uuid.uuid4()),
            "agent_id": "test_agent", 
            "statement": lesson,
            "belief_type": "learned",
            "confidence": 0.8,
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        
        # Add episodes with same lesson
        for i in range(3):
            storage["agent_episodes"].append({
                "id": str(uuid.uuid4()),
                "agent_id": "test_agent",
                "objective": f"Task {i}",
                "lessons_learned": [lesson],
                "is_reflected": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        
        beliefs_before = len([b for b in storage["agent_beliefs"] if lesson in b.get("statement", "")])
        
        result = kernle.consolidate()
        
        beliefs_after = len([b for b in storage["agent_beliefs"] if lesson in b.get("statement", "")])
        
        # Should not create duplicate belief
        assert beliefs_after == beliefs_before
        assert result["new_beliefs"] == 0


class TestFormatMemory:
    """Test memory formatting for context injection."""
    
    def test_format_memory_full(self, kernle_instance, populated_storage):
        """Test formatting full memory context."""
        kernle, storage = kernle_instance
        
        # Add a checkpoint
        kernle.checkpoint("Current task", ["item1", "item2"], "Working context")
        
        memory = kernle.load()
        formatted = kernle.format_memory(memory)
        
        # Check that all sections are present
        assert "# Working Memory (test_agent)" in formatted
        assert "## Working State" in formatted
        assert "## Values" in formatted
        assert "## Goals" in formatted
        assert "## Beliefs" in formatted
        assert "## Recent Work" in formatted
        assert "Current task" in formatted
        assert "Quality" in formatted  # From sample value
        
        # Check that checkpoint info is formatted properly
        assert "**Task**: Current task" in formatted
        assert "**Pending**:" in formatted
        assert "  - item1" in formatted
        assert "  - item2" in formatted
        assert "**Context**: Working context" in formatted
    
    def test_format_memory_empty_sections(self, kernle_instance):
        """Test formatting memory with empty sections."""
        kernle, storage = kernle_instance
        
        memory = kernle.load()
        formatted = kernle.format_memory(memory)
        
        # Should still have header and basic structure
        assert "# Working Memory (test_agent)" in formatted
        assert "_Loaded at" in formatted
        
        # Empty sections should not appear
        assert "## Values" not in formatted
        assert "## Goals" not in formatted
        assert "## Beliefs" not in formatted
    
    def test_format_memory_with_none(self, kernle_instance):
        """Test formatting with None input (should load fresh)."""
        kernle, storage = kernle_instance
        
        formatted = kernle.format_memory(None)
        
        # Should load fresh memory and format it
        assert "# Working Memory (test_agent)" in formatted
        assert "_Loaded at" in formatted