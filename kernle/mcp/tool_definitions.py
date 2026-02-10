"""MCP tool schema definitions for Kernle memory operations.

Each Tool() defines the name, description, and JSON Schema for one MCP tool.
Validators and handlers live in kernle.mcp.handlers.
"""

from mcp.types import Tool

from kernle.types import VALID_SOURCE_TYPE_VALUES

# Re-export as a list for JSON Schema enum validation and backward compat.
# Single source of truth is SourceType enum in kernle.types.
VALID_SOURCE_TYPES = sorted(VALID_SOURCE_TYPE_VALUES)

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
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'session with Sean', 'heartbeat check')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'conversation with X', 'reading Y')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'consolidation', 'told by X', 'raw-processing')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'consolidation', 'told by X', 'raw-processing')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'consolidation', 'told by X', 'raw-processing')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
                "context": {
                    "type": "string",
                    "description": "Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')",
                },
                "context_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional context tags for filtering",
                },
                "source": {
                    "type": "string",
                    "description": "Source context (e.g., 'consolidation', 'told by X', 'raw-processing')",
                },
                "derived_from": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Memory IDs this was derived from (format: type:id, e.g., 'raw:abc123')",
                },
                "source_type": {
                    "type": "string",
                    "enum": [
                        "direct_experience",
                        "inference",
                        "consolidation",
                        "external",
                        "seed",
                        "observation",
                        "unknown",
                    ],
                    "description": "How this memory was acquired (default: auto-derived from source)",
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
        name="memory_consolidate_advanced",
        description="Get an advanced consolidation scaffold with cross-domain pattern analysis, belief-to-value promotion candidates, and entity model-to-belief generalizations. Surfaces deeper patterns across your memories for reflection. Kernle provides the analysis; you decide what to act on.",
        inputSchema={
            "type": "object",
            "properties": {
                "episode_limit": {
                    "type": "integer",
                    "description": "Maximum episodes to analyze for cross-domain patterns (default: 100)",
                    "default": 100,
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
    Tool(
        name="memory_process",
        description="Run memory processing to promote memories up the hierarchy using the bound model. Transitions: raw->episode, raw->note, episode->belief, episode->goal, episode->relationship, belief->value, episode->drive. Requires a bound model.",
        inputSchema={
            "type": "object",
            "properties": {
                "transition": {
                    "type": "string",
                    "description": "Specific layer transition to process (omit to check all)",
                    "enum": [
                        "raw_to_episode",
                        "raw_to_note",
                        "episode_to_belief",
                        "episode_to_goal",
                        "episode_to_relationship",
                        "belief_to_value",
                        "episode_to_drive",
                    ],
                },
                "force": {
                    "type": "boolean",
                    "description": "Process even if trigger thresholds aren't met (default: false)",
                    "default": False,
                },
            },
        },
    ),
    Tool(
        name="memory_process_status",
        description="Show unprocessed memory counts and which processing triggers would fire. No model required.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]
