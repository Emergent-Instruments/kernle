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
from typing import Any, Optional, Dict, List, Union

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from kernle.core import Kernle

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = Server("kernle")


def get_kernle() -> Kernle:
    """Get or create Kernle instance."""
    if not hasattr(get_kernle, "_instance"):
        get_kernle._instance = Kernle()
    return get_kernle._instance


# =============================================================================
# INPUT VALIDATION & SANITIZATION
# =============================================================================

def sanitize_string(value: Any, field_name: str, max_length: int = 1000, required: bool = True) -> str:
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
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    
    return sanitized


def sanitize_array(value: Any, field_name: str, item_max_length: int = 500, max_items: int = 100) -> List[str]:
    """Sanitize and validate array inputs."""
    if value is None:
        return []
    
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")
    
    if len(value) > max_items:
        raise ValueError(f"{field_name} too many items (max {max_items}, got {len(value)})")
    
    sanitized = []
    for i, item in enumerate(value):
        sanitized_item = sanitize_string(item, f"{field_name}[{i}]", item_max_length, required=False)
        if sanitized_item:  # Only add non-empty items
            sanitized.append(sanitized_item)
    
    return sanitized


def validate_enum(value: Any, field_name: str, valid_values: List[str], default: str = None) -> str:
    """Validate enum values."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")
    
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    if value not in valid_values:
        raise ValueError(f"{field_name} must be one of {valid_values}, got '{value}'")
    
    return value


def validate_number(value: Any, field_name: str, min_val: float = None, max_val: float = None, default: float = None) -> float:
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
    sanitized = {}
    
    try:
        if name == "memory_load":
            sanitized["format"] = validate_enum(
                arguments.get("format"), "format", ["text", "json"], "text"
            )
        
        elif name == "memory_checkpoint_save":
            sanitized["task"] = sanitize_string(arguments.get("task"), "task", 500, required=True)
            sanitized["pending"] = sanitize_array(arguments.get("pending"), "pending", 200, 20)
            sanitized["context"] = sanitize_string(arguments.get("context"), "context", 1000, required=False)
        
        elif name == "memory_checkpoint_load":
            # No parameters to validate
            pass
        
        elif name == "memory_episode":
            sanitized["objective"] = sanitize_string(arguments.get("objective"), "objective", 1000, required=True)
            sanitized["outcome"] = sanitize_string(arguments.get("outcome"), "outcome", 1000, required=True)
            sanitized["lessons"] = sanitize_array(arguments.get("lessons"), "lessons", 500, 20)
            sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
        
        elif name == "memory_note":
            sanitized["content"] = sanitize_string(arguments.get("content"), "content", 2000, required=True)
            sanitized["type"] = validate_enum(
                arguments.get("type"), "type", ["note", "decision", "insight", "quote"], "note"
            )
            sanitized["speaker"] = sanitize_string(arguments.get("speaker"), "speaker", 200, required=False)
            sanitized["reason"] = sanitize_string(arguments.get("reason"), "reason", 1000, required=False)
            sanitized["tags"] = sanitize_array(arguments.get("tags"), "tags", 100, 10)
        
        elif name == "memory_search":
            sanitized["query"] = sanitize_string(arguments.get("query"), "query", 500, required=True)
            sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
        
        elif name == "memory_belief":
            sanitized["statement"] = sanitize_string(arguments.get("statement"), "statement", 1000, required=True)
            sanitized["type"] = validate_enum(
                arguments.get("type"), "type", ["fact", "rule", "preference", "constraint", "learned"], "fact"
            )
            sanitized["confidence"] = validate_number(arguments.get("confidence"), "confidence", 0.0, 1.0, 0.8)
        
        elif name == "memory_value":
            sanitized["name"] = sanitize_string(arguments.get("name"), "name", 100, required=True)
            sanitized["statement"] = sanitize_string(arguments.get("statement"), "statement", 1000, required=True)
            sanitized["priority"] = int(validate_number(arguments.get("priority"), "priority", 0, 100, 50))
        
        elif name == "memory_goal":
            sanitized["title"] = sanitize_string(arguments.get("title"), "title", 200, required=True)
            sanitized["description"] = sanitize_string(arguments.get("description"), "description", 1000, required=False)
            sanitized["priority"] = validate_enum(
                arguments.get("priority"), "priority", ["low", "medium", "high"], "medium"
            )
        
        elif name == "memory_drive":
            sanitized["drive_type"] = validate_enum(
                arguments.get("drive_type"), "drive_type", 
                ["existence", "growth", "curiosity", "connection", "reproduction"], required=True
            )
            sanitized["intensity"] = validate_number(arguments.get("intensity"), "intensity", 0.0, 1.0, 0.5)
            sanitized["focus_areas"] = sanitize_array(arguments.get("focus_areas"), "focus_areas", 200, 10)
        
        elif name == "memory_when":
            sanitized["period"] = validate_enum(
                arguments.get("period"), "period", 
                ["today", "yesterday", "this week", "last hour"], "today"
            )
        
        elif name == "memory_consolidate":
            sanitized["min_episodes"] = int(validate_number(arguments.get("min_episodes"), "min_episodes", 1, 100, 3))
        
        elif name == "memory_status":
            # No parameters to validate
            pass
        
        elif name == "memory_auto_capture":
            sanitized["text"] = sanitize_string(arguments.get("text"), "text", 5000, required=True)
            sanitized["context"] = sanitize_string(arguments.get("context"), "context", 1000, required=False)
        
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
        logger.error(f"Internal error in tool {tool_name}", extra={
            "tool_name": tool_name,
            "arguments_keys": list(arguments.keys()) if arguments else [],
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, exc_info=True)
        return [TextContent(type="text", text="Internal server error")]


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    Tool(
        name="memory_load",
        description="Load working memory context including checkpoint, values, beliefs, goals, drives, lessons, and recent work. Call at session start.",
        inputSchema={
            "type": "object",
            "properties": {
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
        description="Run memory consolidation to extract beliefs from episodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "min_episodes": {
                    "type": "integer",
                    "description": "Minimum episodes required (default: 3)",
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
        name="memory_auto_capture",
        description="Automatically capture text if it contains significant signals (decisions, lessons, etc.).",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze and potentially capture",
                },
                "context": {
                    "type": "string",
                    "description": "Context for the capture",
                },
            },
            "required": ["text"],
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
            memory = k.load()
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
            checkpoint = k.load_checkpoint()
            if checkpoint:
                result = json.dumps(checkpoint, indent=2, default=str)
            else:
                result = "No checkpoint found."
        
        elif name == "memory_episode":
            episode_id = k.episode(
                objective=sanitized_args["objective"],
                outcome=sanitized_args["outcome"],
                lessons=sanitized_args.get("lessons"),
                tags=sanitized_args.get("tags"),
            )
            result = f"Episode saved: {episode_id[:8]}..."
        
        elif name == "memory_note":
            note_id = k.note(
                content=sanitized_args["content"],
                type=sanitized_args.get("type", "note"),
                speaker=sanitized_args.get("speaker"),
                reason=sanitized_args.get("reason"),
                tags=sanitized_args.get("tags"),
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
            )
            result = f"Belief saved: {belief_id[:8]}..."
        
        elif name == "memory_value":
            value_id = k.value(
                name=sanitized_args["name"],
                statement=sanitized_args["statement"],
                priority=sanitized_args.get("priority", 50),
            )
            result = f"Value saved: {sanitized_args['name']}"
        
        elif name == "memory_goal":
            goal_id = k.goal(
                title=sanitized_args["title"],
                description=sanitized_args.get("description"),
                priority=sanitized_args.get("priority", "medium"),
            )
            result = f"Goal saved: {sanitized_args['title']}"
        
        elif name == "memory_drive":
            drive_id = k.drive(
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
            consolidation = k.consolidate(
                min_episodes=sanitized_args.get("min_episodes", 3)
            )
            result = f"Consolidation complete:\n  Episodes: {consolidation['consolidated']}\n  New beliefs: {consolidation.get('new_beliefs', 0)}"
        
        elif name == "memory_status":
            status = k.status()
            result = f"""Memory Status ({status['agent_id']})
=====================================
Values:     {status['values']}
Beliefs:    {status['beliefs']}
Goals:      {status['goals']} active
Episodes:   {status['episodes']}
Checkpoint: {'Yes' if status['checkpoint'] else 'No'}"""
        
        elif name == "memory_auto_capture":
            capture_id = k.auto_capture(
                text=sanitized_args["text"],
                context=sanitized_args.get("context"),
            )
            if capture_id:
                result = f"Auto-captured: {capture_id[:8]}..."
            else:
                result = "Not significant enough to capture."
        
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


def main():
    """Entry point for MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
