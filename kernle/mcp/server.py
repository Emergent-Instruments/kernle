"""
Kernle MCP Server - Memory operations for Claude Code and other MCP clients.

This exposes Kernle's memory operations as MCP tools, enabling AI agents
to manage their stratified memory through the Model Context Protocol.

Usage:
    kernle mcp  # Start MCP server (stdio transport)
"""

import asyncio
import json
import logging
from typing import Any, Optional

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
    """Handle tool calls."""
    k = get_kernle()
    
    try:
        if name == "memory_load":
            format_type = arguments.get("format", "text")
            memory = k.load()
            if format_type == "json":
                result = json.dumps(memory, indent=2, default=str)
            else:
                result = k.format_memory(memory)
        
        elif name == "memory_checkpoint_save":
            checkpoint = k.checkpoint(
                task=arguments["task"],
                pending=arguments.get("pending"),
                context=arguments.get("context"),
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
                objective=arguments["objective"],
                outcome=arguments["outcome"],
                lessons=arguments.get("lessons"),
                tags=arguments.get("tags"),
            )
            result = f"Episode saved: {episode_id[:8]}..."
        
        elif name == "memory_note":
            note_id = k.note(
                content=arguments["content"],
                type=arguments.get("type", "note"),
                speaker=arguments.get("speaker"),
                reason=arguments.get("reason"),
                tags=arguments.get("tags"),
            )
            result = f"Note saved: {arguments['content'][:50]}..."
        
        elif name == "memory_search":
            results = k.search(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
            )
            if not results:
                result = f"No results for '{arguments['query']}'"
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
                statement=arguments["statement"],
                type=arguments.get("type", "fact"),
                confidence=arguments.get("confidence", 0.8),
            )
            result = f"Belief saved: {belief_id[:8]}..."
        
        elif name == "memory_value":
            value_id = k.value(
                name=arguments["name"],
                statement=arguments["statement"],
                priority=arguments.get("priority", 50),
            )
            result = f"Value saved: {arguments['name']}"
        
        elif name == "memory_goal":
            goal_id = k.goal(
                title=arguments["title"],
                description=arguments.get("description"),
                priority=arguments.get("priority", "medium"),
            )
            result = f"Goal saved: {arguments['title']}"
        
        elif name == "memory_drive":
            drive_id = k.drive(
                drive_type=arguments["drive_type"],
                intensity=arguments.get("intensity", 0.5),
                focus_areas=arguments.get("focus_areas"),
            )
            result = f"Drive '{arguments['drive_type']}' set to {arguments.get('intensity', 0.5):.0%}"
        
        elif name == "memory_when":
            period = arguments.get("period", "today")
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
                min_episodes=arguments.get("min_episodes", 3)
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
                text=arguments["text"],
                context=arguments.get("context"),
            )
            if capture_id:
                result = f"Auto-captured: {capture_id[:8]}..."
            else:
                result = "Not significant enough to capture."
        
        else:
            result = f"Unknown tool: {name}"
        
        return [TextContent(type="text", text=result)]
    
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


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
