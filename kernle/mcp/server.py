"""
Kernle MCP Server - Memory operations for Claude Code and other MCP clients.

This exposes Kernle's memory operations as MCP tools, enabling AI agents
to manage their stratified memory through the Model Context Protocol.

Security Features:
- Comprehensive input validation and sanitization
- Secure error handling with no information disclosure
- Type safety and schema validation
- Structured logging for debugging

Usage:
    kernle mcp  # Start MCP server (stdio transport)
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from kernle.core import Kernle

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = Server("kernle")


# Global agent_id for MCP session
_mcp_agent_id: str = "default"


def set_agent_id(agent_id: str) -> None:
    """Set the agent ID for this MCP session."""
    global _mcp_agent_id
    _mcp_agent_id = agent_id
    # Clear cached instance so next get_kernle uses new agent_id
    if hasattr(get_kernle, "_instance"):
        delattr(get_kernle, "_instance")


def get_kernle() -> Kernle:
    """Get or create Kernle instance."""
    if not hasattr(get_kernle, "_instance"):
        get_kernle._instance = Kernle(_mcp_agent_id)  # type: ignore[attr-defined]
    return get_kernle._instance  # type: ignore[attr-defined]


# =============================================================================
# INPUT VALIDATION & SANITIZATION
# =============================================================================


def sanitize_string(
    value: Any, field_name: str, max_length: int = 1000, required: bool = True
) -> str:
    """Sanitize and validate string inputs at MCP layer."""
    if value is None and not required:
        return ""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if required and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters, got {len(value)})")

    # Remove null bytes and control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


def sanitize_array(
    value: Any, field_name: str, item_max_length: int = 500, max_items: int = 100
) -> List[str]:
    """Sanitize and validate array inputs."""
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")

    if len(value) > max_items:
        raise ValueError(f"{field_name} too many items (max {max_items}, got {len(value)})")

    sanitized = []
    for i, item in enumerate(value):
        sanitized_item = sanitize_string(
            item, f"{field_name}[{i}]", item_max_length, required=False
        )
        if sanitized_item:  # Only add non-empty items
            sanitized.append(sanitized_item)

    return sanitized


def validate_enum(
    value: Any,
    field_name: str,
    valid_values: List[str],
    default: Optional[str] = None,
    required: bool = False,
) -> str:
    """Validate enum values.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages
        valid_values: List of valid enum values
        default: Default value if value is None
        required: If True, value must be provided (no default)
    """
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if value not in valid_values:
        raise ValueError(f"{field_name} must be one of {valid_values}, got '{value}'")

    return value


def validate_number(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Validate numeric values."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(value).__name__}")

    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}, got {value}")

    return float(value)


def validate_tool_input(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize MCP tool inputs."""
    sanitized: Dict[str, Any] = {}

    try:
        if name == "memory_load":
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )
            from kernle.core import MAX_TOKEN_BUDGET, MIN_TOKEN_BUDGET

            sanitized["budget"] = int(
                validate_number(
                    arguments.get("budget"), "budget", MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET, 8000
                )
            )
            sanitized["truncate"] = arguments.get("truncate", True)
            if not isinstance(sanitized["truncate"], bool):
                sanitized["truncate"] = True

        elif name == "memory_checkpoint_save":
            sanitized["task"] = sanitize_string(arguments.get("task"), "task", 500, required=True)
            sanitized["pending"] = sanitize_array(arguments.get("pending"), "pending", 200, 20)
            sanitized["context"] = sanitize_string(
                arguments.get("context"), "context", 1000, required=False
            )

        elif name == "memory_checkpoint_load":
            # No parameters to validate
            pass

        elif name == "memory_episode":
            sanitized["objective"] = sanitize_string(
                arguments.get("objective"), "objective", 1000, required=True
            )
            sanitized["outcome"] = sanitize_string(
                arguments.get("outcome"), "outcome", 1000, required=True
            )
            sanitized["lessons"] = sanitize_array(arguments.get("lessons"), "lessons", 500, 20)
            sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
            sanitized["context"] = (
                sanitize_string(arguments.get("context"), "context", 500, required=False) or None
            )
            sanitized["context_tags"] = (
                sanitize_array(arguments.get("context_tags"), "context_tags", 100, 20) or None
            )

        elif name == "memory_note":
            sanitized["content"] = sanitize_string(
                arguments.get("content"), "content", 2000, required=True
            )
            sanitized["type"] = validate_enum(
                arguments.get("type"), "type", ["note", "decision", "insight", "quote"], "note"
            )
            sanitized["speaker"] = sanitize_string(
                arguments.get("speaker"), "speaker", 200, required=False
            )
            sanitized["reason"] = sanitize_string(
                arguments.get("reason"), "reason", 1000, required=False
            )
            sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
            sanitized["context"] = (
                sanitize_string(arguments.get("context"), "context", 500, required=False) or None
            )
            sanitized["context_tags"] = (
                sanitize_array(arguments.get("context_tags"), "context_tags", 100, 20) or None
            )

        elif name == "memory_search":
            sanitized["query"] = sanitize_string(
                arguments.get("query"), "query", 500, required=True
            )
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))

        elif name == "memory_belief":
            sanitized["statement"] = sanitize_string(
                arguments.get("statement"), "statement", 1000, required=True
            )
            sanitized["type"] = validate_enum(
                arguments.get("type"),
                "type",
                ["fact", "rule", "preference", "constraint", "learned"],
                "fact",
            )
            sanitized["confidence"] = validate_number(
                arguments.get("confidence"), "confidence", 0.0, 1.0, 0.8
            )
            sanitized["context"] = (
                sanitize_string(arguments.get("context"), "context", 500, required=False) or None
            )
            sanitized["context_tags"] = (
                sanitize_array(arguments.get("context_tags"), "context_tags", 100, 20) or None
            )

        elif name == "memory_value":
            sanitized["name"] = sanitize_string(arguments.get("name"), "name", 100, required=True)
            sanitized["statement"] = sanitize_string(
                arguments.get("statement"), "statement", 1000, required=True
            )
            sanitized["priority"] = int(
                validate_number(arguments.get("priority"), "priority", 0, 100, 50)
            )
            sanitized["context"] = (
                sanitize_string(arguments.get("context"), "context", 500, required=False) or None
            )
            sanitized["context_tags"] = (
                sanitize_array(arguments.get("context_tags"), "context_tags", 100, 20) or None
            )

        elif name == "memory_goal":
            sanitized["title"] = sanitize_string(
                arguments.get("title"), "title", 200, required=True
            )
            sanitized["description"] = sanitize_string(
                arguments.get("description"), "description", 1000, required=False
            )
            sanitized["priority"] = validate_enum(
                arguments.get("priority"), "priority", ["low", "medium", "high"], "medium"
            )
            sanitized["context"] = (
                sanitize_string(arguments.get("context"), "context", 500, required=False) or None
            )
            sanitized["context_tags"] = (
                sanitize_array(arguments.get("context_tags"), "context_tags", 100, 20) or None
            )

        elif name == "memory_drive":
            sanitized["drive_type"] = validate_enum(
                arguments.get("drive_type"),
                "drive_type",
                ["existence", "growth", "curiosity", "connection", "reproduction"],
                default=None,
                required=True,
            )
            sanitized["intensity"] = validate_number(
                arguments.get("intensity"), "intensity", 0.0, 1.0, 0.5
            )
            sanitized["focus_areas"] = sanitize_array(
                arguments.get("focus_areas"), "focus_areas", 200, 10
            )

        elif name == "memory_when":
            sanitized["period"] = validate_enum(
                arguments.get("period"),
                "period",
                ["today", "yesterday", "this week", "last hour"],
                "today",
            )

        elif name == "memory_consolidate":
            sanitized["min_episodes"] = int(
                validate_number(arguments.get("min_episodes"), "min_episodes", 1, 100, 3)
            )

        elif name == "memory_status":
            # No parameters to validate
            pass

        elif name == "memory_auto_capture":
            sanitized["text"] = sanitize_string(arguments.get("text"), "text", 10000, required=True)
            sanitized["context"] = sanitize_string(
                arguments.get("context"), "context", 1000, required=False
            )
            sanitized["source"] = (
                sanitize_string(arguments.get("source"), "source", 100, required=False) or "auto"
            )
            sanitized["extract_suggestions"] = arguments.get("extract_suggestions", False)
            if not isinstance(sanitized["extract_suggestions"], bool):
                sanitized["extract_suggestions"] = False

        elif name == "memory_belief_list":
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 20))
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )

        elif name == "memory_value_list":
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )

        elif name == "memory_goal_list":
            sanitized["status"] = validate_enum(
                arguments.get("status"),
                "status",
                ["active", "completed", "paused", "all"],
                "active",
            )
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )

        elif name == "memory_drive_list":
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )

        elif name == "memory_episode_update":
            sanitized["episode_id"] = sanitize_string(
                arguments.get("episode_id"), "episode_id", 100, required=True
            )
            sanitized["outcome"] = sanitize_string(
                arguments.get("outcome"), "outcome", 1000, required=False
            )
            sanitized["lessons"] = sanitize_array(arguments.get("lessons"), "lessons", 500, 20)
            sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)

        elif name == "memory_goal_update":
            sanitized["goal_id"] = sanitize_string(
                arguments.get("goal_id"), "goal_id", 100, required=True
            )
            sanitized["status"] = (
                validate_enum(
                    arguments.get("status"), "status", ["active", "completed", "paused"], None
                )
                if arguments.get("status")
                else None
            )
            sanitized["priority"] = (
                validate_enum(
                    arguments.get("priority"), "priority", ["low", "medium", "high"], None
                )
                if arguments.get("priority")
                else None
            )
            sanitized["description"] = sanitize_string(
                arguments.get("description"), "description", 1000, required=False
            )

        elif name == "memory_belief_update":
            sanitized["belief_id"] = sanitize_string(
                arguments.get("belief_id"), "belief_id", 100, required=True
            )
            sanitized["confidence"] = (
                validate_number(arguments.get("confidence"), "confidence", 0.0, 1.0, None)
                if arguments.get("confidence") is not None
                else None
            )
            sanitized["is_active"] = arguments.get("is_active")  # Boolean, can be None

        elif name == "memory_sync":
            # No parameters to validate
            pass

        elif name == "memory_note_search":
            sanitized["query"] = sanitize_string(
                arguments.get("query"), "query", 500, required=True
            )
            sanitized["note_type"] = validate_enum(
                arguments.get("note_type"),
                "note_type",
                ["note", "decision", "insight", "quote", "all"],
                "all",
            )
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))

        elif name == "memory_suggestions_list":
            status = arguments.get("status", "pending")
            if status == "all":
                sanitized["status"] = None
            else:
                sanitized["status"] = validate_enum(
                    status, "status", ["pending", "promoted", "rejected"], "pending"
                )
            memory_type = arguments.get("memory_type")
            if memory_type:
                sanitized["memory_type"] = validate_enum(
                    memory_type, "memory_type", ["episode", "belief", "note"], None
                )
            else:
                sanitized["memory_type"] = None
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 20))
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )

        elif name == "memory_suggestions_promote":
            sanitized["suggestion_id"] = sanitize_string(
                arguments.get("suggestion_id"), "suggestion_id", 100, required=True
            )
            sanitized["objective"] = (
                sanitize_string(arguments.get("objective"), "objective", 1000, required=False)
                or None
            )
            sanitized["outcome"] = (
                sanitize_string(arguments.get("outcome"), "outcome", 1000, required=False) or None
            )
            sanitized["statement"] = (
                sanitize_string(arguments.get("statement"), "statement", 1000, required=False)
                or None
            )
            sanitized["content"] = (
                sanitize_string(arguments.get("content"), "content", 2000, required=False) or None
            )

        elif name == "memory_suggestions_reject":
            sanitized["suggestion_id"] = sanitize_string(
                arguments.get("suggestion_id"), "suggestion_id", 100, required=True
            )
            sanitized["reason"] = (
                sanitize_string(arguments.get("reason"), "reason", 500, required=False) or None
            )

        elif name == "memory_suggestions_extract":
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 200, 50))

        elif name == "memory_raw":
            # Blob has no length limit - raw capture is intentionally permissive
            # Still sanitize to remove null bytes and control characters
            sanitized["blob"] = sanitize_string(
                arguments.get("blob"), "blob", max_length=1_000_000, required=True
            )

        elif name == "memory_raw_search":
            sanitized["query"] = sanitize_string(
                arguments.get("query"), "query", max_length=1000, required=True
            )
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 500, 50))

        else:
            raise ValueError(f"Unknown tool: {name}")

        return sanitized

    except (ValueError, TypeError) as e:
        logger.warning(f"Input validation failed for tool {name}: {e}")
        raise ValueError(f"Invalid input: {str(e)}")


def handle_tool_error(e: Exception, tool_name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool errors securely."""
    if isinstance(e, ValueError):
        # Input validation or business logic error
        logger.warning(f"Invalid input for tool {tool_name}: {e}")
        return [TextContent(type="text", text=f"Invalid input: {str(e)}")]

    elif isinstance(e, PermissionError):
        logger.warning(f"Permission denied for tool {tool_name}")
        return [TextContent(type="text", text="Access denied")]

    elif isinstance(e, FileNotFoundError):
        logger.warning(f"Resource not found for tool {tool_name}")
        return [TextContent(type="text", text="Resource not found")]

    elif isinstance(e, ConnectionError):
        logger.error(f"Database connection error for tool {tool_name}")
        return [TextContent(type="text", text="Service temporarily unavailable")]

    else:
        # Unknown error - log full details but return generic message
        logger.error(
            f"Internal error in tool {tool_name}",
            extra={
                "tool_name": tool_name,
                "arguments_keys": list(arguments.keys()) if arguments else [],
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        return [TextContent(type="text", text="Internal server error")]


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    Tool(
        name="memory_load",
        description="Load working memory context including checkpoint, values, beliefs, goals, drives, lessons, and recent work. Uses priority-based budget loading to prevent context overflow. Call at session start.",
        inputSchema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format (default: text)",
                    "default": "text",
                },
                "budget": {
                    "type": "integer",
                    "description": "Token budget for memory loading (default: 8000, range: 100-50000). Higher values load more memories.",
                    "default": 8000,
                    "minimum": 100,
                    "maximum": 50000,
                },
                "truncate": {
                    "type": "boolean",
                    "description": "Truncate long content to fit more items in budget (default: true)",
                    "default": True,
                },
            },
        },
    ),
    Tool(
        name="memory_checkpoint_save",
        description="Save current working state. Use before session end or major context changes.",
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Current task description",
                },
                "pending": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of pending items",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context",
                },
            },
            "required": ["task"],
        },
    ),
    Tool(
        name="memory_checkpoint_load",
        description="Load the most recent checkpoint.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_episode",
        description="Record an episodic experience with lessons learned.",
        inputSchema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "What was the objective?",
                },
                "outcome": {
                    "type": "string",
                    "description": "What was the outcome? (success/failure/partial)",
                },
                "lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lessons learned from this experience",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
            },
            "required": ["objective", "outcome"],
        },
    ),
    Tool(
        name="memory_note",
        description="Capture a quick note (decision, insight, or quote).",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Note content",
                },
                "type": {
                    "type": "string",
                    "enum": ["note", "decision", "insight", "quote"],
                    "description": "Type of note",
                    "default": "note",
                },
                "speaker": {
                    "type": "string",
                    "description": "Speaker (for quotes)",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason (for decisions)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_search",
        description="Search across episodes, notes, and beliefs.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_belief",
        description="Add or update a belief.",
        inputSchema={
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Belief statement",
                },
                "type": {
                    "type": "string",
                    "enum": ["fact", "rule", "preference", "constraint", "learned"],
                    "description": "Type of belief",
                    "default": "fact",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level (0.0-1.0)",
                    "default": 0.8,
                },
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
            },
            "required": ["statement"],
        },
    ),
    Tool(
        name="memory_value",
        description="Add or affirm a core value.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Value name (snake_case)",
                },
                "statement": {
                    "type": "string",
                    "description": "Value statement",
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority (0-100, higher = more important)",
                    "default": 50,
                },
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
            },
            "required": ["name", "statement"],
        },
    ),
    Tool(
        name="memory_goal",
        description="Add a goal.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Goal title",
                },
                "description": {
                    "type": "string",
                    "description": "Goal description",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Priority level",
                    "default": "medium",
                },
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
            },
            "required": ["title"],
        },
    ),
    Tool(
        name="memory_drive",
        description="Set or update a drive (motivation).",
        inputSchema={
            "type": "object",
            "properties": {
                "drive_type": {
                    "type": "string",
                    "enum": ["existence", "growth", "curiosity", "connection", "reproduction"],
                    "description": "Type of drive",
                },
                "intensity": {
                    "type": "number",
                    "description": "Intensity (0.0-1.0)",
                    "default": 0.5,
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Areas of focus for this drive",
                },
            },
            "required": ["drive_type"],
        },
    ),
    Tool(
        name="memory_when",
        description="Query memories by time period.",
        inputSchema={
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["today", "yesterday", "this week", "last hour"],
                    "description": "Time period to query",
                    "default": "today",
                },
            },
        },
    ),
    Tool(
        name="memory_consolidate",
        description="Get a reflection scaffold with recent episodes and beliefs. Returns structured prompts to guide you through pattern recognition and belief updates. Kernle provides the data; you do the reasoning.",
        inputSchema={
            "type": "object",
            "properties": {
                "min_episodes": {
                    "type": "integer",
                    "description": "Minimum episodes required for full consolidation (default: 3)",
                    "default": 3,
                },
            },
        },
    ),
    Tool(
        name="memory_status",
        description="Get memory statistics.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_raw",
        description="Capture raw brain dump to memory. Zero-friction capture - dump whatever you want without structure or validation. The blob can contain thoughts, code snippets, context, emotions - anything. Process it later.",
        inputSchema={
            "type": "object",
            "properties": {
                "blob": {
                    "type": "string",
                    "description": "Raw brain dump content - no structure, no validation, no length limits",
                },
            },
            "required": ["blob"],
        },
    ),
    Tool(
        name="memory_raw_search",
        description="Search raw entries using keyword search (FTS5). Safety net for when raw entry backlogs accumulate.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "FTS5 search query (supports AND, OR, NOT, phrases in quotes)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 50)",
                    "default": 50,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_auto_capture",
        description="(DEPRECATED: Use memory_raw instead) Capture text to raw memory layer with optional suggestion extraction.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to capture",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the capture",
                },
                "source": {
                    "type": "string",
                    "description": "Source identifier (e.g., 'hook-session-end', 'hook-post-tool', 'conversation')",
                    "default": "auto",
                },
                "extract_suggestions": {
                    "type": "boolean",
                    "description": "If true, analyze text and return promotion suggestions (episode/note/belief)",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    ),
    # List tools
    Tool(
        name="memory_belief_list",
        description="List all active beliefs with their confidence levels.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum beliefs to return (default: 20)",
                    "default": 20,
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format. Use 'json' to get IDs for updates (default: text)",
                    "default": "text",
                },
            },
        },
    ),
    Tool(
        name="memory_value_list",
        description="List all core values ordered by priority.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum values to return (default: 10)",
                    "default": 10,
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format. Use 'json' to get IDs for updates (default: text)",
                    "default": "text",
                },
            },
        },
    ),
    Tool(
        name="memory_goal_list",
        description="List goals filtered by status.",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "completed", "paused", "all"],
                    "description": "Filter by status (default: active)",
                    "default": "active",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum goals to return (default: 10)",
                    "default": 10,
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format. Use 'json' to get IDs for updates (default: text)",
                    "default": "text",
                },
            },
        },
    ),
    Tool(
        name="memory_drive_list",
        description="List all drives/motivations with their current intensities.",
        inputSchema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format. Use 'json' to get IDs for updates (default: text)",
                    "default": "text",
                },
            },
        },
    ),
    # Update tools
    Tool(
        name="memory_episode_update",
        description="Update an existing episode (add lessons, change outcome, add tags).",
        inputSchema={
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": "ID of the episode to update",
                },
                "outcome": {
                    "type": "string",
                    "description": "New outcome description",
                },
                "lessons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional lessons to add",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional tags to add",
                },
            },
            "required": ["episode_id"],
        },
    ),
    Tool(
        name="memory_goal_update",
        description="Update a goal's status, priority, or description.",
        inputSchema={
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "string",
                    "description": "ID of the goal to update",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "completed", "paused"],
                    "description": "New status",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "New priority",
                },
                "description": {
                    "type": "string",
                    "description": "New description",
                },
            },
            "required": ["goal_id"],
        },
    ),
    Tool(
        name="memory_belief_update",
        description="Update a belief's confidence or deactivate it.",
        inputSchema={
            "type": "object",
            "properties": {
                "belief_id": {
                    "type": "string",
                    "description": "ID of the belief to update",
                },
                "confidence": {
                    "type": "number",
                    "description": "New confidence level (0.0-1.0)",
                },
                "is_active": {
                    "type": "boolean",
                    "description": "Whether the belief is still active",
                },
            },
            "required": ["belief_id"],
        },
    ),
    Tool(
        name="memory_sync",
        description="Trigger synchronization with cloud storage. Pushes local changes and pulls remote updates.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_note_search",
        description="Search notes by content and optionally filter by type.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "note_type": {
                    "type": "string",
                    "enum": ["note", "decision", "insight", "quote", "all"],
                    "description": "Filter by note type (default: all)",
                    "default": "all",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    # Suggestion tools
    Tool(
        name="memory_suggestions_list",
        description="List memory suggestions extracted from raw entries. Suggestions are auto-extracted patterns that may be promoted to structured memories after review.",
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "promoted", "rejected", "all"],
                    "description": "Filter by status (default: pending)",
                    "default": "pending",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episode", "belief", "note"],
                    "description": "Filter by suggested memory type",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum suggestions to return (default: 20)",
                    "default": 20,
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Output format (default: text)",
                    "default": "text",
                },
            },
        },
    ),
    Tool(
        name="memory_suggestions_promote",
        description="Approve and promote a suggestion to a structured memory. Optionally modify the content before promotion.",
        inputSchema={
            "type": "object",
            "properties": {
                "suggestion_id": {
                    "type": "string",
                    "description": "ID of the suggestion to promote",
                },
                "objective": {
                    "type": "string",
                    "description": "Override objective (for episode suggestions)",
                },
                "outcome": {
                    "type": "string",
                    "description": "Override outcome (for episode suggestions)",
                },
                "statement": {
                    "type": "string",
                    "description": "Override statement (for belief suggestions)",
                },
                "content": {
                    "type": "string",
                    "description": "Override content (for note suggestions)",
                },
            },
            "required": ["suggestion_id"],
        },
    ),
    Tool(
        name="memory_suggestions_reject",
        description="Reject a suggestion (it will not be promoted to a memory).",
        inputSchema={
            "type": "object",
            "properties": {
                "suggestion_id": {
                    "type": "string",
                    "description": "ID of the suggestion to reject",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for rejection",
                },
            },
            "required": ["suggestion_id"],
        },
    ),
    Tool(
        name="memory_suggestions_extract",
        description="Extract suggestions from unprocessed raw entries. Analyzes raw captures and creates pending suggestions for review.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum raw entries to process (default: 50)",
                    "default": 50,
                },
            },
        },
    ),
]


@mcp.list_tools()
async def list_tools() -> list[Tool]:
    """List available memory tools."""
    return TOOLS


@mcp.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls with comprehensive validation and error handling."""
    try:
        # Validate and sanitize all inputs
        sanitized_args = validate_tool_input(name, arguments)

        k = get_kernle()

        if name == "memory_load":
            format_type = sanitized_args.get("format", "text")
            budget = sanitized_args.get("budget", 8000)
            truncate = sanitized_args.get("truncate", True)
            memory = k.load(budget=budget, truncate=truncate)
            if format_type == "json":
                result = json.dumps(memory, indent=2, default=str)
            else:
                result = k.format_memory(memory)

        elif name == "memory_checkpoint_save":
            checkpoint = k.checkpoint(
                task=sanitized_args["task"],
                pending=sanitized_args.get("pending"),
                context=sanitized_args.get("context"),
            )
            result = f"Checkpoint saved: {checkpoint['current_task']}"
            if checkpoint.get("pending"):
                result += f"\nPending: {len(checkpoint['pending'])} items"

        elif name == "memory_checkpoint_load":
            loaded_checkpoint = k.load_checkpoint()
            if loaded_checkpoint:
                result = json.dumps(loaded_checkpoint, indent=2, default=str)
            else:
                result = "No checkpoint found."

        elif name == "memory_episode":
            episode_id = k.episode(
                objective=sanitized_args["objective"],
                outcome=sanitized_args["outcome"],
                lessons=sanitized_args.get("lessons"),
                tags=sanitized_args.get("tags"),
                context=sanitized_args.get("context"),
                context_tags=sanitized_args.get("context_tags"),
            )
            result = f"Episode saved: {episode_id[:8]}..."

        elif name == "memory_note":
            k.note(
                content=sanitized_args["content"],
                type=sanitized_args.get("type", "note"),
                speaker=sanitized_args.get("speaker"),
                reason=sanitized_args.get("reason"),
                tags=sanitized_args.get("tags"),
                context=sanitized_args.get("context"),
                context_tags=sanitized_args.get("context_tags"),
            )
            result = f"Note saved: {sanitized_args['content'][:50]}..."

        elif name == "memory_search":
            results = k.search(
                query=sanitized_args["query"],
                limit=sanitized_args.get("limit", 10),
            )
            if not results:
                result = f"No results for '{sanitized_args['query']}'"
            else:
                lines = [f"Found {len(results)} result(s):\n"]
                for i, r in enumerate(results, 1):
                    lines.append(f"{i}. [{r['type']}] {r['title']}")
                    if r.get("lessons"):
                        for lesson in r["lessons"][:2]:
                            lines.append(f"   → {lesson[:60]}...")
                result = "\n".join(lines)

        elif name == "memory_belief":
            belief_id = k.belief(
                statement=sanitized_args["statement"],
                type=sanitized_args.get("type", "fact"),
                confidence=sanitized_args.get("confidence", 0.8),
                context=sanitized_args.get("context"),
                context_tags=sanitized_args.get("context_tags"),
            )
            result = f"Belief saved: {belief_id[:8]}..."

        elif name == "memory_value":
            k.value(
                name=sanitized_args["name"],
                statement=sanitized_args["statement"],
                priority=sanitized_args.get("priority", 50),
                context=sanitized_args.get("context"),
                context_tags=sanitized_args.get("context_tags"),
            )
            result = f"Value saved: {sanitized_args['name']}"

        elif name == "memory_goal":
            goal_id = k.goal(
                title=sanitized_args["title"],
                description=sanitized_args.get("description"),
                priority=sanitized_args.get("priority", "medium"),
                context=sanitized_args.get("context"),
                context_tags=sanitized_args.get("context_tags"),
            )
            result = f"Goal saved: {sanitized_args['title']}"

        elif name == "memory_drive":
            k.drive(
                drive_type=sanitized_args["drive_type"],
                intensity=sanitized_args.get("intensity", 0.5),
                focus_areas=sanitized_args.get("focus_areas"),
            )
            result = f"Drive '{sanitized_args['drive_type']}' set to {sanitized_args.get('intensity', 0.5):.0%}"

        elif name == "memory_when":
            period = sanitized_args.get("period", "today")
            temporal = k.what_happened(period)
            lines = [f"What happened {period}:\n"]
            if temporal.get("episodes"):
                lines.append("Episodes:")
                for ep in temporal["episodes"][:5]:
                    lines.append(f"  - {ep['objective'][:60]} [{ep.get('outcome_type', '?')}]")
            if temporal.get("notes"):
                lines.append("Notes:")
                for n in temporal["notes"][:5]:
                    lines.append(f"  - {n['content'][:60]}...")
            result = "\n".join(lines)

        elif name == "memory_consolidate":
            min_episodes = sanitized_args.get("min_episodes", 3)

            # Fetch recent episodes
            episodes = k._storage.get_episodes(limit=20)

            # Fetch existing beliefs
            beliefs = k.load_beliefs(limit=15)

            # Build reflection scaffold
            lines = []
            lines.append("# Memory Consolidation: Reflection Scaffold")
            lines.append("")
            lines.append("Kernle has gathered your recent experiences and current beliefs.")
            lines.append(
                "Your task: reason about patterns, extract insights, decide on belief updates."
            )
            lines.append("")

            # Episodes section
            lines.append("## Recent Experiences")
            lines.append("")
            if len(episodes) < min_episodes:
                lines.append(
                    f"Only {len(episodes)} episode(s) recorded (minimum {min_episodes} for consolidation)."
                )
                lines.append("Continue capturing experiences before consolidating.")
            else:
                for i, ep in enumerate(episodes[:10], 1):
                    outcome_emoji = {"success": "✓", "failure": "✗", "partial": "~"}.get(
                        ep.outcome_type or "", "?"
                    )
                    lines.append(
                        f"**{i}. {ep.objective}** [{outcome_emoji} {ep.outcome_type or 'unknown'}]"
                    )
                    lines.append(f"   Outcome: {ep.outcome}")
                    if ep.lessons:
                        for lesson in ep.lessons:
                            lines.append(f"   → Lesson: {lesson}")
                    if ep.tags:
                        lines.append(f"   Tags: {', '.join(ep.tags)}")
                    lines.append("")

            # Beliefs section
            lines.append("## Current Beliefs")
            lines.append("")
            if beliefs:
                for b in beliefs[:10]:
                    conf = f" ({b['confidence']:.0%})" if b.get("confidence") else ""
                    btype = f"[{b.get('belief_type', 'fact')}]" if b.get("belief_type") else ""
                    lines.append(f"- {btype} {b['statement']}{conf}")
                lines.append("")
            else:
                lines.append("No beliefs recorded yet.")
                lines.append("")

            # Extract lesson patterns
            all_lessons = []
            for ep in episodes:
                if ep.lessons:
                    all_lessons.extend(ep.lessons)

            if all_lessons:
                from collections import Counter

                lesson_counts = Counter(all_lessons)
                recurring = [(lesson, cnt) for lesson, cnt in lesson_counts.items() if cnt >= 2]
                if recurring:
                    lines.append("## Recurring Patterns")
                    lines.append("")
                    for lesson, count in sorted(recurring, key=lambda x: -x[1])[:5]:
                        lines.append(f"- ({count}x) {lesson}")
                    lines.append("")

            # Reflection prompts
            lines.append("---")
            lines.append("## Your Reflection Task")
            lines.append("")
            lines.append("Consider the experiences above and ask yourself:")
            lines.append("")
            lines.append(
                "1. **Pattern Recognition**: What themes appear across multiple experiences?"
            )
            lines.append("   - Are there repeated successes or failures?")
            lines.append("   - What approaches consistently work (or don't)?")
            lines.append("")
            lines.append(
                "2. **Belief Validation**: Do your current beliefs match your experiences?"
            )
            lines.append("   - Any beliefs that should increase in confidence?")
            lines.append("   - Any beliefs contradicted by recent outcomes?")
            lines.append("")
            lines.append("3. **New Insights**: What have you learned that isn't captured yet?")
            lines.append("   - Consider adding new beliefs with `memory_belief`")
            lines.append("   - Update existing beliefs with `memory_belief_update`")
            lines.append("")
            lines.append("**Kernle provides the data. You do the reasoning.**")
            lines.append("")
            lines.append(f"Episodes reviewed: {len(episodes)} | Beliefs on file: {len(beliefs)}")

            result = "\n".join(lines)

        elif name == "memory_status":
            status = k.status()
            result = f"""Memory Status ({status["agent_id"]})
=====================================
Values:     {status["values"]}
Beliefs:    {status["beliefs"]}
Goals:      {status["goals"]} active
Episodes:   {status["episodes"]}
Checkpoint: {"Yes" if status["checkpoint"] else "No"}"""

        elif name == "memory_raw":
            # New simplified raw capture - just blob, source is auto-detected as "mcp"
            blob = sanitized_args["blob"]
            capture_id = k.raw(blob=blob, source="mcp")
            result = json.dumps(
                {
                    "captured": True,
                    "id": capture_id[:8],
                    "full_id": capture_id,
                    "source": "mcp",
                },
                indent=2,
            )

        elif name == "memory_raw_search":
            # FTS5 keyword search on raw entries
            query = sanitized_args["query"]
            limit = sanitized_args.get("limit", 50)
            entries = k.search_raw(query, limit=limit)

            if not entries:
                result = json.dumps({"results": [], "query": query, "total": 0}, indent=2)
            else:
                results = []
                for e in entries:
                    results.append(
                        {
                            "id": e["id"][:8],
                            "blob_preview": (
                                e["blob"][:200] + "..." if len(e["blob"]) > 200 else e["blob"]
                            ),
                            "captured_at": e.get("captured_at"),
                            "source": e.get("source"),
                            "processed": e.get("processed"),
                        }
                    )
                result = json.dumps(
                    {
                        "results": results,
                        "query": query,
                        "total": len(results),
                    },
                    indent=2,
                )

        elif name == "memory_auto_capture":
            source = sanitized_args.get("source", "auto")
            # Normalize source to valid enum values
            if source not in {"cli", "mcp", "sdk", "import", "unknown"}:
                if "auto" in source.lower():
                    source = "mcp"  # MCP tool is auto-capture
                else:
                    source = "mcp"
            extract_suggestions = sanitized_args.get("extract_suggestions", False)

            # Get the blob content (was 'text', mapping to new 'blob' parameter)
            blob = sanitized_args["text"]
            if sanitized_args.get("context"):
                # Append context to blob if provided
                blob = f"{blob}\n\n[Context: {sanitized_args['context']}]"

            # Capture to raw layer using new blob parameter
            # Note: tags parameter is deprecated but kept for backward compatibility
            capture_id = k.raw(
                blob=blob,
                source=source,
            )

            if extract_suggestions:
                # Analyze text and suggest promotion type
                text_lower = sanitized_args["text"].lower()
                suggestions = []

                # Check for episode indicators
                if any(
                    word in text_lower
                    for word in [
                        "session",
                        "completed",
                        "shipped",
                        "implemented",
                        "built",
                        "fixed",
                        "deployed",
                        "finished",
                    ]
                ):
                    suggestions.append("episode")
                # Check for note/insight indicators
                if any(
                    word in text_lower
                    for word in ["insight", "decision", "realized", "learned", "important", "noted"]
                ):
                    suggestions.append("note")
                # Check for belief indicators
                if any(
                    word in text_lower
                    for word in [
                        "believe",
                        "think that",
                        "seems like",
                        "pattern",
                        "always",
                        "never",
                        "should",
                    ]
                ):
                    suggestions.append("belief")

                result_data = {
                    "captured": True,
                    "id": capture_id[:8],
                    "source": source,
                    "suggestions": suggestions or ["review"],
                    "promote_command": f"kernle raw promote {capture_id[:8]} --type <episode|note|belief>",
                }
                result = json.dumps(result_data, indent=2)
            else:
                result = f"Auto-captured: {capture_id[:8]}... (source: {source})"

        elif name == "memory_belief_list":
            beliefs = k.load_beliefs(limit=sanitized_args.get("limit", 20))
            format_type = sanitized_args.get("format", "text")
            if not beliefs:
                result = "No beliefs found." if format_type == "text" else json.dumps([], indent=2)
            elif format_type == "json":
                result = json.dumps(beliefs, indent=2, default=str)
            else:
                lines = [f"Found {len(beliefs)} belief(s):\n"]
                for i, b in enumerate(beliefs, 1):
                    conf = f" ({b['confidence']:.0%})" if b.get("confidence") else ""
                    btype = f"[{b.get('belief_type', 'fact')}]" if b.get("belief_type") else ""
                    lines.append(f"{i}. {btype} {b['statement']}{conf}")
                result = "\n".join(lines)

        elif name == "memory_value_list":
            values = k.load_values(limit=sanitized_args.get("limit", 10))
            format_type = sanitized_args.get("format", "text")
            if not values:
                result = "No values found." if format_type == "text" else json.dumps([], indent=2)
            elif format_type == "json":
                result = json.dumps(values, indent=2, default=str)
            else:
                lines = [f"Found {len(values)} value(s):\n"]
                for i, v in enumerate(values, 1):
                    priority = f" (priority: {v.get('priority', 50)})" if v.get("priority") else ""
                    lines.append(f"{i}. **{v['name']}**: {v['statement']}{priority}")
                result = "\n".join(lines)

        elif name == "memory_goal_list":
            status = sanitized_args.get("status", "active")
            format_type = sanitized_args.get("format", "text")
            # Pass status directly to load_goals - it now handles filtering
            goals = k.load_goals(limit=sanitized_args.get("limit", 10), status=status)

            if not goals:
                result = (
                    f"No {status} goals found."
                    if format_type == "text"
                    else json.dumps([], indent=2)
                )
            elif format_type == "json":
                result = json.dumps(goals, indent=2, default=str)
            else:
                lines = [f"Found {len(goals)} goal(s):\n"]
                for i, g in enumerate(goals, 1):
                    priority = f" [{g.get('priority', 'medium')}]" if g.get("priority") else ""
                    status_str = (
                        f" ({g.get('status', 'active')})" if g.get("status") != "active" else ""
                    )
                    lines.append(f"{i}. {g['title']}{priority}{status_str}")
                    if g.get("description"):
                        lines.append(f"   {g['description'][:60]}...")
                result = "\n".join(lines)

        elif name == "memory_drive_list":
            drives = k.load_drives()
            format_type = sanitized_args.get("format", "text")
            if not drives:
                result = (
                    "No drives configured." if format_type == "text" else json.dumps([], indent=2)
                )
            elif format_type == "json":
                result = json.dumps(drives, indent=2, default=str)
            else:
                lines = ["Current drives:\n"]
                for d in drives:
                    focus = (
                        f" → {', '.join(d.get('focus_areas', []))}" if d.get("focus_areas") else ""
                    )
                    lines.append(f"- **{d['drive_type']}**: {d['intensity']:.0%}{focus}")
                result = "\n".join(lines)

        elif name == "memory_episode_update":
            episode_id = sanitized_args["episode_id"]
            updated = k.update_episode(
                episode_id=episode_id,
                outcome=sanitized_args.get("outcome"),
                lessons=sanitized_args.get("lessons"),
                tags=sanitized_args.get("tags"),
            )
            if updated:
                result = f"Episode {episode_id[:8]}... updated successfully."
            else:
                result = f"Episode {episode_id[:8]}... not found."

        elif name == "memory_goal_update":
            goal_id = sanitized_args["goal_id"]
            updated = k.update_goal(
                goal_id=goal_id,
                status=sanitized_args.get("status"),
                priority=sanitized_args.get("priority"),
                description=sanitized_args.get("description"),
            )
            if updated:
                result = f"Goal {goal_id[:8]}... updated successfully."
            else:
                result = f"Goal {goal_id[:8]}... not found."

        elif name == "memory_belief_update":
            belief_id = sanitized_args["belief_id"]
            updated = k.update_belief(
                belief_id=belief_id,
                confidence=sanitized_args.get("confidence"),
                is_active=sanitized_args.get("is_active"),
            )
            if updated:
                result = f"Belief {belief_id[:8]}... updated successfully."
            else:
                result = f"Belief {belief_id[:8]}... not found."

        elif name == "memory_sync":
            sync_result = k.sync()
            lines = ["Sync complete:"]
            lines.append(f"  Pushed: {sync_result.get('pushed', 0)}")
            lines.append(f"  Pulled: {sync_result.get('pulled', 0)}")
            if sync_result.get("conflicts"):
                lines.append(f"  Conflicts: {sync_result['conflicts']}")
            if sync_result.get("errors"):
                lines.append(f"  Errors: {len(sync_result['errors'])}")
                for err in sync_result["errors"][:3]:
                    lines.append(f"    - {err}")
            result = "\n".join(lines)

        elif name == "memory_note_search":
            query = sanitized_args["query"]
            note_type = sanitized_args.get("note_type", "all")
            limit = sanitized_args.get("limit", 10)

            # Use the general search and filter by type
            results = k.search(query=query, limit=limit * 2)  # Get extra in case we filter

            # Filter for note types only
            note_results = [
                r for r in results if r.get("type") in ["note", "decision", "insight", "quote"]
            ]

            # Further filter by specific type if not "all"
            if note_type != "all":
                note_results = [r for r in note_results if r.get("type") == note_type]

            note_results = note_results[:limit]

            if not note_results:
                result = f"No notes found for '{query}'"
            else:
                lines = [f"Found {len(note_results)} note(s):\n"]
                for i, n in enumerate(note_results, 1):
                    lines.append(f"{i}. [{n['type']}] {n['title']}")
                    if n.get("date"):
                        lines.append(f"   {n['date']}")
                result = "\n".join(lines)

        elif name == "memory_suggestions_list":
            status = sanitized_args.get("status")
            memory_type = sanitized_args.get("memory_type")
            limit = sanitized_args.get("limit", 20)
            format_type = sanitized_args.get("format", "text")

            suggestions = k.get_suggestions(
                status=status,
                memory_type=memory_type,
                limit=limit,
            )

            if not suggestions:
                status_str = f" {status}" if status else ""
                result = f"No{status_str} suggestions found."
            elif format_type == "json":
                result = json.dumps(suggestions, indent=2, default=str)
            else:
                # Group counts
                pending = sum(1 for s in suggestions if s["status"] == "pending")
                promoted = sum(1 for s in suggestions if s["status"] in ("promoted", "modified"))
                rejected = sum(1 for s in suggestions if s["status"] == "rejected")

                lines = [f"Memory Suggestions ({len(suggestions)} total)\n"]
                if not status:  # Show breakdown only for unfiltered list
                    lines.append(
                        f"Pending: {pending} | Approved: {promoted} | Rejected: {rejected}\n"
                    )

                for s in suggestions:
                    status_icons = {
                        "pending": "?",
                        "promoted": "+",
                        "modified": "*",
                        "rejected": "x",
                    }
                    icon = status_icons.get(s["status"], "?")
                    type_label = s["memory_type"][:3].upper()

                    # Get preview
                    content = s.get("content", {})
                    if s["memory_type"] == "episode":
                        preview = content.get("objective", "")[:50]
                    elif s["memory_type"] == "belief":
                        preview = content.get("statement", "")[:50]
                    else:
                        preview = content.get("content", "")[:50]

                    lines.append(
                        f"[{icon}] {s['id'][:8]} [{type_label}] {s['confidence']:.0%}: {preview}..."
                    )

                    if s.get("promoted_to"):
                        lines.append(f"    -> {s['promoted_to']}")

                result = "\n".join(lines)

        elif name == "memory_suggestions_promote":
            suggestion_id = sanitized_args["suggestion_id"]

            # Build modifications dict
            modifications = {}
            if sanitized_args.get("objective"):
                modifications["objective"] = sanitized_args["objective"]
            if sanitized_args.get("outcome"):
                modifications["outcome"] = sanitized_args["outcome"]
            if sanitized_args.get("statement"):
                modifications["statement"] = sanitized_args["statement"]
            if sanitized_args.get("content"):
                modifications["content"] = sanitized_args["content"]

            memory_id = k.promote_suggestion(
                suggestion_id,
                modifications if modifications else None,
            )

            if memory_id:
                status = "modified" if modifications else "promoted"
                result = f"Suggestion {suggestion_id[:8]}... {status} to memory {memory_id[:8]}..."
            else:
                result = f"Could not promote suggestion {suggestion_id[:8]}... (not found or not pending)"

        elif name == "memory_suggestions_reject":
            suggestion_id = sanitized_args["suggestion_id"]
            reason = sanitized_args.get("reason")

            if k.reject_suggestion(suggestion_id, reason):
                result = f"Suggestion {suggestion_id[:8]}... rejected"
                if reason:
                    result += f" (reason: {reason})"
            else:
                result = (
                    f"Could not reject suggestion {suggestion_id[:8]}... (not found or not pending)"
                )

        elif name == "memory_suggestions_extract":
            limit = sanitized_args.get("limit", 50)
            suggestions = k.extract_suggestions_from_unprocessed(limit=limit)

            if not suggestions:
                result = "No suggestions extracted from raw entries."
            else:
                # Group by type
                by_type = {}
                for s in suggestions:
                    t = s["memory_type"]
                    by_type[t] = by_type.get(t, 0) + 1

                lines = [f"Extracted {len(suggestions)} suggestion(s):\n"]
                for t, count in by_type.items():
                    lines.append(f"  {t}: {count}")
                lines.append("\nUse memory_suggestions_list to review pending suggestions.")
                result = "\n".join(lines)

        else:
            # This should never happen due to validation, but handle gracefully
            logger.error(f"Unexpected tool name after validation: {name}")
            result = f"Tool '{name}' is not available"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return handle_tool_error(e, name, arguments)


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options(),
        )


def main(agent_id: str = "default"):
    """Entry point for MCP server.

    Agent ID resolution (in order):
    1. Explicit agent_id argument (if not "default")
    2. KERNLE_AGENT_ID environment variable
    3. Auto-generated from machine + project path
    """
    from kernle.utils import resolve_agent_id

    # Use resolve_agent_id for consistent fallback logic
    resolved_id = resolve_agent_id(agent_id if agent_id != "default" else None)

    set_agent_id(resolved_id)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
