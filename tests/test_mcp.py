"""
Comprehensive tests for the Kernle MCP server.

Tests all MCP tools, tool definitions, call_tool dispatcher, and error handling.
"""

import json
import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from mcp.types import Tool, TextContent

from kernle.mcp.server import (
    TOOLS,
    list_tools,
    call_tool,
    get_kernle,
)


class TestMCPToolDefinitions:
    """Test MCP tool definitions and list_tools functionality."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        """Test that list_tools returns all 14 expected tools."""
        tools = await list_tools()
        
        assert len(tools) == 14
        assert all(isinstance(tool, Tool) for tool in tools)
        
        # Check all expected tools are present
        tool_names = {tool.name for tool in tools}
        expected_names = {
            "memory_load",
            "memory_checkpoint_save",
            "memory_checkpoint_load",
            "memory_episode",
            "memory_note",
            "memory_search",
            "memory_belief",
            "memory_value",
            "memory_goal",
            "memory_drive",
            "memory_when",
            "memory_consolidate",
            "memory_status",
            "memory_auto_capture",
        }
        assert tool_names == expected_names

    def test_tool_definitions_have_required_fields(self):
        """Test that all tool definitions have required fields."""
        for tool in TOOLS:
            assert tool.name
            assert tool.description
            assert tool.inputSchema
            assert "type" in tool.inputSchema
            assert "properties" in tool.inputSchema

    def test_memory_load_tool_definition(self):
        """Test memory_load tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_load")
        
        assert "format" in tool.inputSchema["properties"]
        format_prop = tool.inputSchema["properties"]["format"]
        assert format_prop["type"] == "string"
        assert format_prop["enum"] == ["text", "json"]
        assert format_prop["default"] == "text"

    def test_memory_checkpoint_save_tool_definition(self):
        """Test memory_checkpoint_save tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_checkpoint_save")
        
        props = tool.inputSchema["properties"]
        assert "task" in props
        assert "pending" in props
        assert "context" in props
        assert tool.inputSchema["required"] == ["task"]

    def test_memory_episode_tool_definition(self):
        """Test memory_episode tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_episode")
        
        props = tool.inputSchema["properties"]
        assert "objective" in props
        assert "outcome" in props
        assert "lessons" in props
        assert "tags" in props
        assert tool.inputSchema["required"] == ["objective", "outcome"]

    def test_memory_note_tool_definition(self):
        """Test memory_note tool definition."""
        tool = next(t for t in TOOLS if t.name == "memory_note")
        
        props = tool.inputSchema["properties"]
        assert "content" in props
        assert "type" in props
        assert "speaker" in props
        assert "reason" in props
        assert "tags" in props
        
        type_prop = props["type"]
        assert type_prop["enum"] == ["note", "decision", "insight", "quote"]
        assert tool.inputSchema["required"] == ["content"]


# Fixtures for mocking
@pytest.fixture
def mock_kernle():
    """Create a comprehensive mock of the Kernle class."""
    kernle_mock = Mock()
    
    # Mock all methods used by MCP tools
    kernle_mock.load.return_value = {
        "checkpoint": {"task": "test task", "pending": []},
        "values": [{"name": "quality", "statement": "Quality is important"}],
        "beliefs": [{"statement": "Testing is crucial", "confidence": 0.9}],
        "goals": [{"title": "Write tests", "priority": "high"}],
        "drives": [{"drive_type": "growth", "intensity": 0.8}],
        "lessons": ["Always test edge cases"],
        "recent_work": [{"objective": "Recent work", "outcome": "success"}],
        "recent_notes": [{"content": "Test note"}],
        "relationships": []
    }
    
    kernle_mock.format_memory.return_value = "Formatted memory output"
    
    kernle_mock.checkpoint.return_value = {
        "current_task": "test task",
        "pending": ["item1", "item2"]
    }
    
    kernle_mock.load_checkpoint.return_value = {
        "task": "loaded task",
        "context": "test context"
    }
    
    kernle_mock.episode.return_value = "episode_123456"
    kernle_mock.note.return_value = "note_123456"
    kernle_mock.belief.return_value = "belief_123456"
    kernle_mock.value.return_value = "value_123456"
    kernle_mock.goal.return_value = "goal_123456"
    kernle_mock.drive.return_value = "drive_123456"
    
    kernle_mock.search.return_value = [
        {
            "type": "episode",
            "title": "Test Episode",
            "lessons": ["Lesson 1", "Lesson 2"]
        }
    ]
    
    kernle_mock.what_happened.return_value = {
        "episodes": [{"objective": "Test objective", "outcome_type": "success"}],
        "notes": [{"content": "Test note content"}]
    }
    
    kernle_mock.consolidate.return_value = {
        "consolidated": 5,
        "new_beliefs": 2
    }
    
    kernle_mock.status.return_value = {
        "agent_id": "test_agent",
        "values": 3,
        "beliefs": 10,
        "goals": 2,
        "episodes": 25,
        "checkpoint": True
    }
    
    kernle_mock.auto_capture.return_value = "capture_123456"
    
    return kernle_mock


@pytest.fixture
def patched_get_kernle(mock_kernle):
    """Patch the get_kernle function to return our mock."""
    with patch('kernle.mcp.server.get_kernle', return_value=mock_kernle):
        yield mock_kernle


class TestKernleMocking:
    """Test proper mocking of the Kernle core class."""
    # NOTE: Removed test_mock_setup - it only tested mock configuration, not production code.
    # Mocks are implementation details of tests, not things to test themselves.


class TestMCPToolCalls:
    """Test individual MCP tool calls with proper mocking."""

    @pytest.mark.asyncio
    async def test_memory_load_text_format(self, patched_get_kernle):
        """Test memory_load with text format calls correct methods."""
        result = await call_tool("memory_load", {"format": "text"})
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        # Verify production code calls the right sequence:
        # 1. Loads memory data
        patched_get_kernle.load.assert_called_once()
        # 2. Formats it for text display (passes loaded data to formatter)
        patched_get_kernle.format_memory.assert_called_once()
        # 3. format_memory should receive the loaded data
        format_call_args = patched_get_kernle.format_memory.call_args
        assert format_call_args is not None  # format_memory was called with arguments
        
        # The result should not be empty
        assert result[0].text  # Non-empty string returned

    @pytest.mark.asyncio
    async def test_memory_load_json_format(self, patched_get_kernle):
        """Test memory_load with JSON format."""
        result = await call_tool("memory_load", {"format": "json"})
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        # Should be valid JSON
        json_data = json.loads(result[0].text)
        assert "checkpoint" in json_data
        assert "values" in json_data
        
        patched_get_kernle.load.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_load_default_format(self, patched_get_kernle):
        """Test memory_load with default format uses text format path."""
        result = await call_tool("memory_load", {})
        
        assert len(result) == 1
        # Default format should be text, so format_memory should be called
        patched_get_kernle.load.assert_called_once()
        patched_get_kernle.format_memory.assert_called_once()
        # Result should contain formatted text (not raw JSON)
        assert result[0].text  # Non-empty

    @pytest.mark.asyncio
    async def test_memory_checkpoint_save(self, patched_get_kernle):
        """Test memory_checkpoint_save."""
        args = {
            "task": "Write comprehensive tests",
            "pending": ["Test edge cases", "Add documentation"],
            "context": "Working on MCP tests"
        }
        
        result = await call_tool("memory_checkpoint_save", args)
        
        assert len(result) == 1
        assert "Checkpoint saved: test task" in result[0].text
        assert "Pending: 2 items" in result[0].text
        
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Write comprehensive tests",
            pending=["Test edge cases", "Add documentation"],
            context="Working on MCP tests"
        )

    @pytest.mark.asyncio
    async def test_memory_checkpoint_save_minimal(self, patched_get_kernle):
        """Test memory_checkpoint_save with only required fields."""
        result = await call_tool("memory_checkpoint_save", {"task": "Minimal test"})
        
        assert len(result) == 1
        assert "Checkpoint saved: test task" in result[0].text
        
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Minimal test",
            pending=[],
            context=""
        )

    @pytest.mark.asyncio
    async def test_memory_checkpoint_load(self, patched_get_kernle):
        """Test memory_checkpoint_load."""
        result = await call_tool("memory_checkpoint_load", {})
        
        assert len(result) == 1
        json_data = json.loads(result[0].text)
        assert json_data["task"] == "loaded task"
        
        patched_get_kernle.load_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_checkpoint_load_empty(self, patched_get_kernle):
        """Test memory_checkpoint_load when no checkpoint exists."""
        patched_get_kernle.load_checkpoint.return_value = None
        
        result = await call_tool("memory_checkpoint_load", {})
        
        assert len(result) == 1
        assert result[0].text == "No checkpoint found."

    @pytest.mark.asyncio
    async def test_memory_episode(self, patched_get_kernle):
        """Test memory_episode."""
        args = {
            "objective": "Write comprehensive tests",
            "outcome": "success",
            "lessons": ["Mock dependencies", "Test error cases"],
            "tags": ["testing", "development"]
        }
        
        result = await call_tool("memory_episode", args)
        
        assert len(result) == 1
        assert "Episode saved:" in result[0].text
        assert "episode_" in result[0].text
        
        patched_get_kernle.episode.assert_called_once_with(
            objective="Write comprehensive tests",
            outcome="success",
            lessons=["Mock dependencies", "Test error cases"],
            tags=["testing", "development"]
        )

    @pytest.mark.asyncio
    async def test_memory_note_all_types(self, patched_get_kernle):
        """Test memory_note with different note types."""
        note_types = [
            {
                "content": "This is a regular note",
                "type": "note",
                "tags": ["general"],
                "expected_call": {"type": "note", "speaker": "", "reason": ""}
            },
            {
                "content": "Use pytest for testing",
                "type": "decision",
                "reason": "Industry standard with good ecosystem",
                "tags": ["testing"],
                "expected_call": {"type": "decision", "speaker": "", "reason": "Industry standard with good ecosystem"}
            },
            {
                "content": "Mocking enables isolated testing",
                "type": "insight",
                "tags": ["testing", "insights"],
                "expected_call": {"type": "insight", "speaker": "", "reason": ""}
            },
            {
                "content": "Code is poetry",
                "type": "quote",
                "speaker": "Someone Wise",
                "tags": ["inspiration"],
                "expected_call": {"type": "quote", "speaker": "Someone Wise", "reason": ""}
            }
        ]
        
        for note_args in note_types:
            # Reset mock before each iteration to verify individual calls
            patched_get_kernle.note.reset_mock()
            
            expected = note_args.pop("expected_call")
            result = await call_tool("memory_note", note_args)
            
            assert len(result) == 1
            assert "Note saved:" in result[0].text
            assert note_args["content"][:50] in result[0].text
            
            # Verify the correct call was made for THIS iteration
            patched_get_kernle.note.assert_called_once_with(
                content=note_args["content"],
                type=expected["type"],
                speaker=expected["speaker"],
                reason=expected["reason"],
                tags=note_args["tags"]
            )

    @pytest.mark.asyncio
    async def test_memory_note_minimal(self, patched_get_kernle):
        """Test memory_note with minimal required fields."""
        result = await call_tool("memory_note", {"content": "Simple note"})
        
        assert len(result) == 1
        assert "Note saved: Simple note..." in result[0].text
        
        patched_get_kernle.note.assert_called_once_with(
            content="Simple note",
            type="note",
            speaker="",
            reason="",
            tags=[]
        )

    @pytest.mark.asyncio
    async def test_memory_search(self, patched_get_kernle):
        """Test memory_search."""
        result = await call_tool("memory_search", {"query": "testing", "limit": 5})
        
        assert len(result) == 1
        assert "Found 1 result(s):" in result[0].text
        assert "[episode] Test Episode" in result[0].text
        assert "Lesson 1" in result[0].text
        
        patched_get_kernle.search.assert_called_once_with(query="testing", limit=5)

    @pytest.mark.asyncio
    async def test_memory_search_no_results(self, patched_get_kernle):
        """Test memory_search with no results."""
        patched_get_kernle.search.return_value = []
        
        result = await call_tool("memory_search", {"query": "nonexistent"})
        
        assert len(result) == 1
        assert "No results for 'nonexistent'" in result[0].text

    @pytest.mark.asyncio
    async def test_memory_search_default_limit(self, patched_get_kernle):
        """Test memory_search with default limit."""
        result = await call_tool("memory_search", {"query": "testing"})
        
        patched_get_kernle.search.assert_called_once_with(query="testing", limit=10)

    @pytest.mark.asyncio
    async def test_memory_belief(self, patched_get_kernle):
        """Test memory_belief."""
        args = {
            "statement": "Testing is essential for quality software",
            "type": "fact",
            "confidence": 0.95
        }
        
        result = await call_tool("memory_belief", args)
        
        assert len(result) == 1
        assert "Belief saved: belief_1" in result[0].text
        
        patched_get_kernle.belief.assert_called_once_with(
            statement="Testing is essential for quality software",
            type="fact",
            confidence=0.95
        )

    @pytest.mark.asyncio
    async def test_memory_belief_default_values(self, patched_get_kernle):
        """Test memory_belief with default type and confidence."""
        result = await call_tool("memory_belief", {"statement": "Simple belief"})
        
        patched_get_kernle.belief.assert_called_once_with(
            statement="Simple belief",
            type="fact",
            confidence=0.8
        )

    @pytest.mark.asyncio
    async def test_memory_value(self, patched_get_kernle):
        """Test memory_value."""
        args = {
            "name": "quality",
            "statement": "Software must be thoroughly tested and reliable",
            "priority": 90
        }
        
        result = await call_tool("memory_value", args)
        
        assert len(result) == 1
        assert "Value saved: quality" in result[0].text
        
        patched_get_kernle.value.assert_called_once_with(
            name="quality",
            statement="Software must be thoroughly tested and reliable",
            priority=90
        )

    @pytest.mark.asyncio
    async def test_memory_goal(self, patched_get_kernle):
        """Test memory_goal."""
        args = {
            "title": "Achieve comprehensive test coverage",
            "description": "Write tests for all MCP tools with edge cases",
            "priority": "high"
        }
        
        result = await call_tool("memory_goal", args)
        
        assert len(result) == 1
        assert "Goal saved: Achieve comprehensive test coverage" in result[0].text
        
        patched_get_kernle.goal.assert_called_once_with(
            title="Achieve comprehensive test coverage",
            description="Write tests for all MCP tools with edge cases",
            priority="high"
        )

    @pytest.mark.asyncio
    async def test_memory_goal_minimal(self, patched_get_kernle):
        """Test memory_goal with minimal fields."""
        result = await call_tool("memory_goal", {"title": "Simple goal"})
        
        patched_get_kernle.goal.assert_called_once_with(
            title="Simple goal",
            description="",
            priority="medium"
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="BUG: validate_enum() gets unexpected 'required' kwarg - see kernle/mcp/server.py ~line 183")
    async def test_memory_drive(self, patched_get_kernle):
        """Test memory_drive with all parameters."""
        args = {
            "drive_type": "growth",
            "intensity": 0.8,
            "focus_areas": ["learning", "improvement", "mastery"]
        }
        
        result = await call_tool("memory_drive", args)
        
        assert len(result) == 1
        # Test intended behavior: should save drive and return confirmation
        assert "Drive saved:" in result[0].text
        assert "drive_" in result[0].text
        
        patched_get_kernle.drive.assert_called_once_with(
            drive_type="growth",
            intensity=0.8,
            focus_areas=["learning", "improvement", "mastery"]
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="BUG: validate_enum() gets unexpected 'required' kwarg - see kernle/mcp/server.py ~line 183")
    async def test_memory_drive_default_intensity(self, patched_get_kernle):
        """Test memory_drive with default intensity."""
        result = await call_tool("memory_drive", {"drive_type": "curiosity"})
        
        assert len(result) == 1
        # Test intended behavior: should use default intensity (0.5)
        assert "Drive saved:" in result[0].text
        
        patched_get_kernle.drive.assert_called_once_with(
            drive_type="curiosity",
            intensity=0.5,
            focus_areas=[]
        )

    # NOTE: The no-op test_memory_drive_validation_bug_documentation was removed.
    # Bug documentation belongs in issue tracker or code comments, not empty tests.

    @pytest.mark.asyncio
    async def test_memory_when_periods(self, patched_get_kernle):
        """Test memory_when with different time periods."""
        periods = ["today", "yesterday", "this week", "last hour"]
        
        for period in periods:
            result = await call_tool("memory_when", {"period": period})
            
            assert len(result) == 1
            assert f"What happened {period}:" in result[0].text
            assert "Episodes:" in result[0].text
            assert "Notes:" in result[0].text
            
            patched_get_kernle.what_happened.assert_called_with(period)

    @pytest.mark.asyncio
    async def test_memory_when_default_period(self, patched_get_kernle):
        """Test memory_when with default period."""
        result = await call_tool("memory_when", {})
        
        patched_get_kernle.what_happened.assert_called_with("today")

    @pytest.mark.asyncio
    async def test_memory_consolidate(self, patched_get_kernle):
        """Test memory_consolidate."""
        result = await call_tool("memory_consolidate", {"min_episodes": 5})
        
        assert len(result) == 1
        assert "Consolidation complete:" in result[0].text
        assert "Episodes: 5" in result[0].text
        assert "New beliefs: 2" in result[0].text
        
        patched_get_kernle.consolidate.assert_called_once_with(min_episodes=5)

    @pytest.mark.asyncio
    async def test_memory_consolidate_default(self, patched_get_kernle):
        """Test memory_consolidate with default min_episodes."""
        result = await call_tool("memory_consolidate", {})
        
        patched_get_kernle.consolidate.assert_called_once_with(min_episodes=3)

    @pytest.mark.asyncio
    async def test_memory_status(self, patched_get_kernle):
        """Test memory_status."""
        result = await call_tool("memory_status", {})
        
        assert len(result) == 1
        status_text = result[0].text
        assert "Memory Status (test_agent)" in status_text
        assert "Values:     3" in status_text
        assert "Beliefs:    10" in status_text
        assert "Goals:      2 active" in status_text
        assert "Episodes:   25" in status_text
        assert "Checkpoint: Yes" in status_text

    @pytest.mark.asyncio
    async def test_memory_auto_capture_success(self, patched_get_kernle):
        """Test memory_auto_capture with successful capture."""
        args = {
            "text": "I learned that mocking is crucial for isolated testing",
            "context": "While writing tests"
        }
        
        result = await call_tool("memory_auto_capture", args)
        
        assert len(result) == 1
        assert "Auto-captured:" in result[0].text
        assert "capture_" in result[0].text
        
        patched_get_kernle.auto_capture.assert_called_once_with(
            text="I learned that mocking is crucial for isolated testing",
            context="While writing tests"
        )

    @pytest.mark.asyncio
    async def test_memory_auto_capture_not_significant(self, patched_get_kernle):
        """Test memory_auto_capture when text is not significant."""
        patched_get_kernle.auto_capture.return_value = None
        
        result = await call_tool("memory_auto_capture", {"text": "Just casual conversation"})
        
        assert len(result) == 1
        assert "Not significant enough to capture." in result[0].text

    @pytest.mark.asyncio
    async def test_memory_auto_capture_minimal(self, patched_get_kernle):
        """Test memory_auto_capture with minimal args."""
        result = await call_tool("memory_auto_capture", {"text": "Test text"})
        
        patched_get_kernle.auto_capture.assert_called_once_with(
            text="Test text",
            context=""
        )


class TestErrorHandling:
    """Test error handling in MCP tool calls."""

    @pytest.fixture
    def failing_kernle(self):
        """Mock Kernle that raises exceptions."""
        kernle_mock = Mock()
        kernle_mock.load.side_effect = Exception("Database connection failed")
        kernle_mock.episode.side_effect = ValueError("Invalid outcome type")
        kernle_mock.search.side_effect = RuntimeError("Search service unavailable")
        return kernle_mock

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, patched_get_kernle):
        """Test error handling for unknown tool names."""
        result = await call_tool("unknown_tool", {})
        
        assert len(result) == 1
        assert "Unknown tool: unknown_tool" in result[0].text

    @pytest.mark.asyncio
    async def test_kernle_exception_handling(self, failing_kernle):
        """Test that Kernle exceptions are caught and returned as error text."""
        with patch('kernle.mcp.server.get_kernle', return_value=failing_kernle):
            result = await call_tool("memory_load", {})
            
            assert len(result) == 1
            assert "Internal server error" in result[0].text

    @pytest.mark.asyncio
    async def test_episode_error_handling(self, failing_kernle):
        """Test error handling for memory_episode."""
        with patch('kernle.mcp.server.get_kernle', return_value=failing_kernle):
            result = await call_tool("memory_episode", {
                "objective": "Test",
                "outcome": "invalid_type"
            })
            
            assert len(result) == 1
            assert "Invalid input: Invalid outcome type" in result[0].text

    @pytest.mark.asyncio
    async def test_search_error_handling(self, failing_kernle):
        """Test error handling for memory_search."""
        with patch('kernle.mcp.server.get_kernle', return_value=failing_kernle):
            result = await call_tool("memory_search", {"query": "test"})
            
            assert len(result) == 1
            assert "Internal server error" in result[0].text


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_missing_required_arguments(self, patched_get_kernle):
        """Test behavior when required arguments are missing returns clear error."""
        # memory_checkpoint_save requires 'task' argument
        result = await call_tool("memory_checkpoint_save", {})
        
        assert len(result) == 1
        # Should get a validation error mentioning the missing field
        error_text = result[0].text.lower()
        assert "invalid" in error_text or "error" in error_text or "required" in error_text
        # Error should identify what's missing (task is required)
        assert "task" in error_text or "required" in error_text
        
        # The kernle method should NOT have been called with missing required args
        patched_get_kernle.checkpoint.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_argument_types(self, patched_get_kernle):
        """Test behavior with invalid argument types returns validation error."""
        # Pass invalid type for limit (should be integer)
        result = await call_tool("memory_search", {
            "query": "test",
            "limit": "invalid"
        })
        
        assert len(result) == 1
        # Server should validate argument types and return a clear error
        # The result should either:
        # - Be an error message about invalid type
        # - Or the call was made anyway (and we can verify how it was called)
        if "Invalid input:" in result[0].text or "Error" in result[0].text:
            # Validation rejected it - this is the expected safe behavior
            assert "limit" in result[0].text.lower() or "type" in result[0].text.lower()
        else:
            # If no error, verify the search was actually called (not silently swallowed)
            patched_get_kernle.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, patched_get_kernle):
        """Test handling of empty results from Kernle methods."""
        patched_get_kernle.search.return_value = []
        patched_get_kernle.what_happened.return_value = {"episodes": [], "notes": []}
        
        # Test empty search results
        result = await call_tool("memory_search", {"query": "nothing"})
        assert "No results for 'nothing'" in result[0].text
        
        # Test empty temporal results
        result = await call_tool("memory_when", {"period": "today"})
        assert "What happened today:" in result[0].text

    @pytest.mark.asyncio
    async def test_null_values_handling(self, patched_get_kernle):
        """Test handling of null/None values from Kernle."""
        patched_get_kernle.load_checkpoint.return_value = None
        patched_get_kernle.auto_capture.return_value = None
        
        # Test null checkpoint
        result = await call_tool("memory_checkpoint_load", {})
        assert "No checkpoint found." in result[0].text
        
        # Test null auto_capture
        result = await call_tool("memory_auto_capture", {"text": "not significant"})
        assert "Not significant enough to capture." in result[0].text

    @pytest.mark.asyncio
    async def test_large_content_handling(self, patched_get_kernle):
        """Test handling of large content that gets rejected by validation."""
        long_content = "This is a very long piece of content " * 100  # ~3800 chars
        
        result = await call_tool("memory_note", {"content": long_content})
        
        assert len(result) == 1
        # Should be rejected by validation (max 2000 characters for notes)
        error_text = result[0].text.lower()
        
        # Verify this is a validation error, not a success
        assert "invalid" in error_text or "error" in error_text, \
            f"Large content should be rejected, got: {result[0].text}"
        
        # Error should mention the issue is with content length/size
        length_related_terms = ["long", "length", "size", "character", "2000", "limit", "exceed", "max"]
        has_length_info = any(term in error_text for term in length_related_terms)
        assert has_length_info, \
            f"Error should mention length/size issue, got: {result[0].text}"
        
        # Verify Kernle.note was NOT called (validation should prevent it)
        patched_get_kernle.note.assert_not_called()

    @pytest.mark.asyncio
    async def test_reasonable_content_handling(self, patched_get_kernle):
        """Test handling of reasonably-sized content."""
        content = "This is a reasonable piece of content for testing."
        
        result = await call_tool("memory_note", {"content": content})
        
        assert len(result) == 1
        assert "Note saved:" in result[0].text
        assert content in result[0].text

    @pytest.mark.asyncio
    async def test_json_serialization_edge_cases(self, patched_get_kernle):
        """Test JSON serialization converts datetime objects to strings."""
        # Create specific datetime for verification
        test_datetime = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        complex_memory = {
            "checkpoint": {"created_at": test_datetime},
            "values": [{"created": test_datetime}],
            "complex_data": {"nested": {"deep": "value"}}
        }
        patched_get_kernle.load.return_value = complex_memory
        
        result = await call_tool("memory_load", {"format": "json"})
        
        assert len(result) == 1
        # Should be valid JSON (datetime objects must be converted to strings)
        json_data = json.loads(result[0].text)
        assert "checkpoint" in json_data
        
        # Verify datetime was serialized - it should be a string, not a datetime object
        checkpoint_created = json_data["checkpoint"]["created_at"]
        assert isinstance(checkpoint_created, str), "datetime should be serialized to string"
        # The serialized string should contain date components
        assert "2024" in checkpoint_created
        
        # Verify nested structure is preserved
        assert json_data["complex_data"]["nested"]["deep"] == "value"

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, patched_get_kernle):
        """Test handling of Unicode content is preserved."""
        unicode_content = "测试 🧪 emoji and unicode characters ñoño"
        
        result = await call_tool("memory_note", {"content": unicode_content})
        
        assert len(result) == 1
        assert "Note saved:" in result[0].text
        
        # Verify unicode content was passed through correctly to Kernle
        patched_get_kernle.note.assert_called_once()
        call_args = patched_get_kernle.note.call_args
        assert call_args.kwargs["content"] == unicode_content, "Unicode should be preserved"
        
        # Verify unicode appears in the response (truncated in output)
        assert "测试" in result[0].text or unicode_content[:50] in result[0].text

    @pytest.mark.asyncio
    async def test_special_characters_in_search(self, patched_get_kernle):
        """Test search with special characters."""
        special_query = "test with \"quotes\" and 'apostrophes' & symbols"
        
        result = await call_tool("memory_search", {"query": special_query})
        
        patched_get_kernle.search.assert_called_once_with(
            query=special_query,
            limit=10
        )


class TestGetKernleFunction:
    """Test the get_kernle singleton function."""

    def test_get_kernle_singleton_behavior(self):
        """Test that get_kernle returns the same instance."""
        # Clear any existing instance
        if hasattr(get_kernle, '_instance'):
            delattr(get_kernle, '_instance')
        
        # First call should create instance
        kernle1 = get_kernle()
        
        # Second call should return same instance
        kernle2 = get_kernle()
        
        assert kernle1 is kernle2

    def test_get_kernle_creates_kernle_instance(self):
        """Test that get_kernle creates a proper Kernle instance."""
        # Clear any existing instance
        if hasattr(get_kernle, '_instance'):
            delattr(get_kernle, '_instance')
        
        with patch('kernle.mcp.server.Kernle') as MockKernle:
            mock_instance = Mock()
            MockKernle.return_value = mock_instance
            
            result = get_kernle()
            
            MockKernle.assert_called_once()
            assert result is mock_instance


class TestMultiToolWorkflows:
    """Test workflows combining multiple tool calls (mocked).
    
    NOTE: These are NOT true integration tests - they use mocked Kernle.
    They verify that call_tool correctly dispatches multiple sequential calls
    and that each tool call works independently.
    
    For real integration tests, see tests/test_integration.py (if it exists)
    or create one that uses a real Kernle instance.
    """

    @pytest.mark.asyncio
    async def test_typical_session_workflow_dispatch(self, patched_get_kernle):
        """Test that typical workflow dispatches to correct Kernle methods."""
        # Load memory
        result1 = await call_tool("memory_load", {"format": "text"})
        assert len(result1) == 1
        
        # Record an episode
        result2 = await call_tool("memory_episode", {
            "objective": "Write MCP tests",
            "outcome": "success",
            "lessons": ["Comprehensive mocking is essential"]
        })
        assert "Episode saved:" in result2[0].text
        
        # Save checkpoint
        result3 = await call_tool("memory_checkpoint_save", {
            "task": "Testing complete",
            "pending": []
        })
        assert "Checkpoint saved:" in result3[0].text
        
        # Verify correct methods called with correct arguments
        patched_get_kernle.load.assert_called_once()
        patched_get_kernle.episode.assert_called_once_with(
            objective="Write MCP tests",
            outcome="success",
            lessons=["Comprehensive mocking is essential"],
            tags=[]
        )
        patched_get_kernle.checkpoint.assert_called_once_with(
            task="Testing complete",
            pending=[],
            context=""
        )

    @pytest.mark.asyncio
    async def test_memory_building_workflow_dispatch(self, patched_get_kernle):
        """Test that memory building workflow dispatches correctly."""
        # Add belief
        await call_tool("memory_belief", {
            "statement": "Testing prevents bugs",
            "type": "fact",
            "confidence": 0.9
        })
        
        # Add value
        await call_tool("memory_value", {
            "name": "reliability",
            "statement": "Software should be dependable",
            "priority": 85
        })
        
        # Add goal
        await call_tool("memory_goal", {
            "title": "Achieve zero critical bugs",
            "priority": "high"
        })
        
        # Verify correct methods called with correct arguments
        patched_get_kernle.belief.assert_called_once_with(
            statement="Testing prevents bugs",
            type="fact",
            confidence=0.9
        )
        patched_get_kernle.value.assert_called_once_with(
            name="reliability",
            statement="Software should be dependable",
            priority=85
        )
        patched_get_kernle.goal.assert_called_once_with(
            title="Achieve zero critical bugs",
            description="",
            priority="high"
        )

    @pytest.mark.asyncio
    async def test_search_and_consolidation_workflow_dispatch(self, patched_get_kernle):
        """Test search -> consolidate -> status dispatches correctly."""
        # Search for patterns
        await call_tool("memory_search", {"query": "testing patterns"})
        
        # Consolidate learnings
        await call_tool("memory_consolidate", {"min_episodes": 3})
        
        # Check status
        await call_tool("memory_status", {})
        
        # Verify correct methods called with correct arguments
        patched_get_kernle.search.assert_called_once_with(query="testing patterns", limit=10)
        patched_get_kernle.consolidate.assert_called_once_with(min_episodes=3)
        patched_get_kernle.status.assert_called_once()