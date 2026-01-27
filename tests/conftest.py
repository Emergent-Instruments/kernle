"""
Pytest fixtures and test configuration for Kernle tests.
"""

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock

import pytest

from kernle.core import Kernle


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client that simulates database operations."""
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
                pattern = value.replace('%', '')
                filtered_data = [
                    item for item in table_data 
                    if pattern.lower() in str(item.get(field, '')).lower()
                ]
                result.data = filtered_data
                return result
            
            def gte_mock(field, value):
                result.data = [item for item in result.data if item.get(field, '') >= value]
                return result
            
            def lte_mock(field, value):
                result.data = [item for item in result.data if item.get(field, '') <= value]
                return result
            
            def order_mock(field, desc=False):
                if result.data:
                    reverse = desc
                    try:
                        result.data.sort(key=lambda x: x.get(field, ''), reverse=reverse)
                    except:
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
                    if 'id' not in item:
                        item['id'] = str(uuid.uuid4())
                    item['created_at'] = datetime.now(timezone.utc).isoformat()
                    table_data.append(item)
            else:
                if 'id' not in data:
                    data['id'] = str(uuid.uuid4())
                data['created_at'] = datetime.now(timezone.utc).isoformat()
                table_data.append(data)
            
            result = Mock()
            result.data = [data] if not isinstance(data, list) else data
            result.execute = lambda: result
            return result
        
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
        table_mock.update = update_mock
        
        return table_mock
    
    # Set up table method
    client.table = create_table_mock
    
    return client, storage


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for checkpoint files."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def kernle_instance(mock_supabase_client, temp_checkpoint_dir):
    """Kernle instance with mocked dependencies."""
    client_mock, storage = mock_supabase_client
    
    kernle = Kernle(
        agent_id="test_agent",
        supabase_url="http://test.url",
        supabase_key="test_key",
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Override the client property to return our mock
    kernle._client = client_mock
    
    return kernle, storage


@pytest.fixture
def sample_episode_data():
    """Sample episode data for testing."""
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
    """Sample memory/note data for testing."""
    return {
        "id": str(uuid.uuid4()),
        "owner_id": "test_agent",
        "owner_type": "agent",
        "content": "**Decision**: Use pytest for testing framework",
        "source": "curated",
        "metadata": {
            "note_type": "decision",
            "tags": ["testing"],
            "reason": "Industry standard with good plugin ecosystem"
        },
        "visibility": "private",
        "is_curated": True,
        "is_protected": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_belief_data():
    """Sample belief data for testing."""
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
    """Sample value data for testing."""
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
    """Sample goal data for testing."""
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
    """Sample drive data for testing."""
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


@pytest.fixture
def populated_storage(mock_supabase_client, sample_episode_data, sample_memory_data, 
                     sample_belief_data, sample_value_data, sample_goal_data, sample_drive_data):
    """Storage populated with sample data."""
    client_mock, storage = mock_supabase_client
    
    # Populate storage
    storage["agent_episodes"].append(sample_episode_data)
    storage["memories"].append(sample_memory_data)
    storage["agent_beliefs"].append(sample_belief_data)
    storage["agent_values"].append(sample_value_data)
    storage["agent_goals"].append(sample_goal_data)
    storage["agent_drives"].append(sample_drive_data)
    
    # Add some additional test data
    storage["agent_episodes"].extend([
        {
            **sample_episode_data,
            "id": str(uuid.uuid4()),
            "objective": "Debug memory leak",
            "outcome_type": "failure",
            "outcome_description": "Could not reproduce the issue",
            "lessons_learned": ["Need better monitoring tools"],
            "is_reflected": False,
        },
        {
            **sample_episode_data,
            "id": str(uuid.uuid4()),
            "objective": "Implement caching",
            "outcome_type": "partial", 
            "outcome_description": "Basic caching implemented, optimization needed",
            "lessons_learned": ["Start simple, then optimize"],
            "tags": ["checkpoint"],  # This should be filtered from recent work
        }
    ])
    
    storage["memories"].extend([
        {
            **sample_memory_data,
            "id": str(uuid.uuid4()),
            "content": "**Insight**: Mocking is crucial for isolated testing",
            "metadata": {"note_type": "insight", "tags": ["testing"]},
        }
    ])
    
    return client_mock, storage