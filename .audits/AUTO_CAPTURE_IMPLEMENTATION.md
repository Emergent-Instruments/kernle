# Auto-Capture Implementation Draft

This document outlines the implementation plan for hook-based auto-capture to the raw layer, plus the memory suggestion system for agent-controlled promotion.

---

## Overview

```
                         ┌─────────────────────────────────────┐
                         │         Claude Code Session         │
                         └──────────────┬──────────────────────┘
                                        │
                    PostToolUse/Stop hooks trigger
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │     kernle raw --quiet --source     │
                         │      (auto-capture to raw layer)    │
                         └──────────────┬──────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │          Raw Layer (SQLite)         │
                         │    + Flat files (human-readable)    │
                         └──────────────┬──────────────────────┘
                                        │
                   Periodic or on-demand extraction
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │       Suggestion Extraction         │
                         │  (pattern-based or LLM-powered)     │
                         └──────────────┬──────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │       Pending Suggestions           │
                         │   (agent reviews and approves)      │
                         └──────────────┬──────────────────────┘
                                        │
                         Agent: promote / modify / reject
                                        │
                                        ▼
                         ┌─────────────────────────────────────┐
                         │     Structured Memory Layers        │
                         │  (Episodes, Beliefs, Notes, etc.)   │
                         └─────────────────────────────────────┘
```

---

## Part 1: Claude Code Hooks Configuration

### File: `.claude/hooks.json`

```json
{
  "hooks": [
    {
      "event": "Stop",
      "command": "kernle -a ${KERNLE_AGENT_ID:-default} raw --quiet --source session-end",
      "stdin": true,
      "input": "Session ended. Last response summary:\n\n${LAST_ASSISTANT_MESSAGE}"
    },
    {
      "event": "PostToolUse",
      "matcher": {
        "tool": "Edit"
      },
      "command": "kernle -a ${KERNLE_AGENT_ID:-default} raw --quiet --source code-edit --tags code",
      "stdin": true,
      "input": "Code edited: ${TOOL_NAME}\nFile: ${TOOL_ARGS.file_path}\nChange: ${TOOL_ARGS.old_string} -> ${TOOL_ARGS.new_string}"
    },
    {
      "event": "PostToolUse",
      "matcher": {
        "tool": "Write"
      },
      "command": "kernle -a ${KERNLE_AGENT_ID:-default} raw --quiet --source code-write --tags code",
      "stdin": true,
      "input": "File created: ${TOOL_ARGS.file_path}"
    },
    {
      "event": "PostToolUse",
      "matcher": {
        "tool": "Bash"
      },
      "command": "kernle -a ${KERNLE_AGENT_ID:-default} raw --quiet --source bash-output --tags terminal",
      "stdin": true,
      "input": "Command: ${TOOL_ARGS.command}\nOutput: ${TOOL_OUTPUT}"
    }
  ]
}
```

### Alternative: Minimal Hooks (Session-End Only)

```json
{
  "hooks": [
    {
      "event": "Stop",
      "command": "kernle -a ${KERNLE_AGENT_ID:-default} raw --quiet --source session-end",
      "stdin": true,
      "input": "${LAST_ASSISTANT_MESSAGE}"
    }
  ]
}
```

---

## Part 2: CLI Enhancements for Hook Support

### Changes to `kernle/cli/commands/raw.py`

```python
"""Raw entry commands for Kernle CLI - with hook support."""

import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_raw(args, k: "Kernle"):
    """Handle raw entry subcommands."""

    if args.raw_action == "capture" or args.raw_action is None:
        # Get content from stdin if --stdin flag is set
        if getattr(args, 'stdin', False):
            content = sys.stdin.read().strip()
            if not content:
                if not getattr(args, 'quiet', False):
                    print("✗ No content provided via stdin")
                return
        else:
            content = validate_input(args.content, "content", 5000)

        tags = [validate_input(t, "tag", 100) for t in (args.tags.split(",") if args.tags else [])]
        tags = [t.strip() for t in tags if t.strip()]
        source = getattr(args, 'source', None) or "cli"

        raw_id = k.raw(content, tags=tags if tags else None, source=source)

        # Quiet mode for hooks - no output
        if not getattr(args, 'quiet', False):
            print(f"✓ Raw entry captured: {raw_id[:8]}...")
            if tags:
                print(f"  Tags: {', '.join(tags)}")
            if source and source != "cli":
                print(f"  Source: {source}")

    # ... rest of existing handlers ...
```

### Changes to `kernle/cli/__main__.py` - Argument Parser

```python
# In the raw subparser setup:

raw_parser = subparsers.add_parser("raw", help="Raw entry management")
raw_subparsers = raw_parser.add_subparsers(dest="raw_action")

# Capture subcommand (default)
capture_parser = raw_subparsers.add_parser("capture", help="Capture raw entry")
capture_parser.add_argument("content", nargs="?", help="Content to capture")
capture_parser.add_argument("--tags", "-t", help="Comma-separated tags")
capture_parser.add_argument("--source", "-s", help="Source identifier (cli, hook, etc.)")
capture_parser.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress output (for hooks)")
capture_parser.add_argument("--stdin", action="store_true",
                           help="Read content from stdin")

# Also support "kernle raw 'content'" directly
raw_parser.add_argument("content", nargs="?", help="Content to capture (shorthand)")
raw_parser.add_argument("--tags", "-t", help="Comma-separated tags")
raw_parser.add_argument("--source", "-s", help="Source identifier")
raw_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
raw_parser.add_argument("--stdin", action="store_true", help="Read from stdin")
```

---

## Part 3: MCP Tool Enhancement

### Changes to `kernle/mcp/server.py`

```python
# Add to TOOLS list:

Tool(
    name="memory_auto_capture",
    description="""Capture content to raw layer with optional suggestion extraction.

Use this for auto-capturing context that might contain memorable content.
The raw layer stores unprocessed content; suggestions can be extracted
for agent review and promotion to structured memories.

Best used via hooks or at session boundaries.""",
    inputSchema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to capture (conversation excerpt, tool output, etc.)",
            },
            "source": {
                "type": "string",
                "description": "Source identifier (hook-session-end, hook-tool-output, manual)",
                "default": "auto",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization",
            },
            "extract_suggestions": {
                "type": "boolean",
                "description": "If true, also extract suggested memories for review",
                "default": False,
            },
        },
        "required": ["content"],
    },
),

Tool(
    name="memory_suggestions_list",
    description="List pending memory suggestions for review and promotion.",
    inputSchema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum suggestions to return",
                "default": 10,
            },
            "memory_type": {
                "type": "string",
                "enum": ["episode", "belief", "note", "all"],
                "description": "Filter by suggested memory type",
                "default": "all",
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum extraction confidence (0-1)",
                "default": 0.0,
            },
        },
    },
),

Tool(
    name="memory_suggestions_promote",
    description="Promote a suggestion to a real memory, optionally with modifications.",
    inputSchema={
        "type": "object",
        "properties": {
            "suggestion_id": {
                "type": "string",
                "description": "ID of the suggestion to promote",
            },
            "modifications": {
                "type": "object",
                "description": "Optional modifications to apply before promotion",
            },
        },
        "required": ["suggestion_id"],
    },
),

Tool(
    name="memory_suggestions_reject",
    description="Reject a suggestion (won't be promoted).",
    inputSchema={
        "type": "object",
        "properties": {
            "suggestion_id": {
                "type": "string",
                "description": "ID of the suggestion to reject",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for rejection (helps improve future extraction)",
            },
        },
        "required": ["suggestion_id"],
    },
),
```

### Handler Implementation

```python
# In call_tool() handler:

elif name == "memory_auto_capture":
    content = sanitized_args["content"]
    source = sanitized_args.get("source", "auto")
    tags = sanitized_args.get("tags", [])
    extract = sanitized_args.get("extract_suggestions", False)

    # Capture to raw layer
    raw_id = k.raw(content, tags=tags, source=source)

    result = {
        "raw_id": raw_id,
        "captured": True,
    }

    # Optionally extract suggestions
    if extract:
        suggestions = k.extract_suggestions([raw_id])
        result["suggestions"] = [
            {
                "id": s.id,
                "type": s.memory_type,
                "preview": s.preview,
                "confidence": s.confidence,
            }
            for s in suggestions
        ]
        result["suggestion_count"] = len(suggestions)

    return [TextContent(type="text", text=json.dumps(result, indent=2))]

elif name == "memory_suggestions_list":
    limit = sanitized_args.get("limit", 10)
    memory_type = sanitized_args.get("memory_type", "all")
    min_conf = sanitized_args.get("min_confidence", 0.0)

    suggestions = k.list_suggestions(
        limit=limit,
        memory_type=None if memory_type == "all" else memory_type,
        min_confidence=min_conf,
    )

    if not suggestions:
        return [TextContent(type="text", text="No pending suggestions.")]

    result = []
    for s in suggestions:
        result.append({
            "id": s.id,
            "type": s.memory_type,
            "content": s.content,
            "confidence": s.confidence,
            "source_raw_ids": s.source_raw_ids,
            "created_at": s.created_at.isoformat(),
        })

    return [TextContent(type="text", text=json.dumps(result, indent=2))]

elif name == "memory_suggestions_promote":
    suggestion_id = sanitized_args["suggestion_id"]
    modifications = sanitized_args.get("modifications", {})

    memory_id = k.promote_suggestion(suggestion_id, modifications)

    return [TextContent(type="text", text=f"Promoted to memory: {memory_id}")]

elif name == "memory_suggestions_reject":
    suggestion_id = sanitized_args["suggestion_id"]
    reason = sanitized_args.get("reason")

    k.reject_suggestion(suggestion_id, reason)

    return [TextContent(type="text", text=f"Rejected suggestion: {suggestion_id}")]
```

---

## Part 4: Suggestion System Data Model

### File: `kernle/storage/base.py` - Add Dataclass

```python
@dataclass
class MemorySuggestion:
    """A suggested memory extracted from raw content."""

    id: str
    agent_id: str
    memory_type: str  # episode, belief, note
    content: Dict[str, Any]  # Structured content for the memory type
    confidence: float  # Extraction confidence (0-1)
    source_raw_ids: List[str]  # Raw entries this was extracted from
    status: str = "pending"  # pending, promoted, rejected
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    resolution_reason: Optional[str] = None

    @property
    def preview(self) -> str:
        """Short preview of the suggestion content."""
        if self.memory_type == "episode":
            return self.content.get("objective", "")[:60]
        elif self.memory_type == "belief":
            return self.content.get("statement", "")[:60]
        elif self.memory_type == "note":
            return self.content.get("content", "")[:60]
        return str(self.content)[:60]
```

### Schema Migration

```sql
-- Migration: Add suggestions table

CREATE TABLE IF NOT EXISTS suggestions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,  -- JSON
    confidence REAL NOT NULL DEFAULT 0.5,
    source_raw_ids TEXT,  -- JSON array
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_reason TEXT,
    promoted_to TEXT,  -- Memory ID if promoted

    -- Sync metadata
    local_updated_at TEXT,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

CREATE INDEX idx_suggestions_agent_status ON suggestions(agent_id, status);
CREATE INDEX idx_suggestions_type ON suggestions(memory_type);
CREATE INDEX idx_suggestions_confidence ON suggestions(confidence);
```

---

## Part 5: Extraction Logic

### File: `kernle/features/suggestions.py`

```python
"""Memory suggestion extraction and management."""

import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Any, Optional

from kernle.storage.base import MemorySuggestion

if TYPE_CHECKING:
    from kernle import Kernle


class SuggestionsMixin:
    """Mixin for memory suggestion extraction and management."""

    # Pattern-based extraction rules
    EPISODE_PATTERNS = [
        r"(?:completed|finished|shipped|deployed|built|implemented|fixed)\s+(.+)",
        r"(?:session|today|this week).*?(?:accomplished|did|completed)\s*:?\s*(.+)",
        r"(?:worked on|working on)\s+(.+?)(?:\.|$)",
    ]

    BELIEF_PATTERNS = [
        r"(?:I believe|I think|seems like|it appears|I've noticed)\s+(.+)",
        r"(?:always|never|should always|should never)\s+(.+)",
        r"(?:pattern|principle|rule):\s*(.+)",
    ]

    NOTE_PATTERNS = [
        r"(?:decided|decision):\s*(.+)",
        r"(?:important|note|remember):\s*(.+)",
        r"(?:insight|realization|learned):\s*(.+)",
    ]

    def extract_suggestions(
        self: "Kernle",
        raw_ids: List[str],
        use_llm: bool = False,
    ) -> List[MemorySuggestion]:
        """Extract memory suggestions from raw entries.

        Args:
            raw_ids: IDs of raw entries to analyze
            use_llm: If True, use LLM for extraction (requires API key)

        Returns:
            List of extracted suggestions
        """
        suggestions = []

        for raw_id in raw_ids:
            entry = self._storage.get_raw(raw_id)
            if not entry or entry.processed:
                continue

            content = entry.content

            if use_llm:
                suggestions.extend(self._extract_with_llm(raw_id, content))
            else:
                suggestions.extend(self._extract_with_patterns(raw_id, content))

        # Save suggestions to storage
        for suggestion in suggestions:
            self._storage.save_suggestion(suggestion)

        return suggestions

    def _extract_with_patterns(
        self: "Kernle",
        raw_id: str,
        content: str,
    ) -> List[MemorySuggestion]:
        """Extract suggestions using pattern matching."""
        suggestions = []
        content_lower = content.lower()

        # Try episode patterns
        for pattern in self.EPISODE_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                suggestions.append(MemorySuggestion(
                    id=f"sug_{uuid.uuid4().hex[:12]}",
                    agent_id=self.agent_id,
                    memory_type="episode",
                    content={
                        "objective": match.group(1).strip()[:200],
                        "outcome_type": "success",
                        "lessons": [],
                    },
                    confidence=0.6,
                    source_raw_ids=[raw_id],
                ))
                break  # One episode per raw entry

        # Try belief patterns
        for pattern in self.BELIEF_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                suggestions.append(MemorySuggestion(
                    id=f"sug_{uuid.uuid4().hex[:12]}",
                    agent_id=self.agent_id,
                    memory_type="belief",
                    content={
                        "statement": match.group(1).strip()[:500],
                        "belief_type": "fact",
                        "confidence": 0.7,
                    },
                    confidence=0.5,
                    source_raw_ids=[raw_id],
                ))

        # Try note patterns
        for pattern in self.NOTE_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                suggestions.append(MemorySuggestion(
                    id=f"sug_{uuid.uuid4().hex[:12]}",
                    agent_id=self.agent_id,
                    memory_type="note",
                    content={
                        "content": match.group(1).strip()[:1000],
                        "note_type": "insight",
                    },
                    confidence=0.5,
                    source_raw_ids=[raw_id],
                ))

        return suggestions

    def _extract_with_llm(
        self: "Kernle",
        raw_id: str,
        content: str,
    ) -> List[MemorySuggestion]:
        """Extract suggestions using LLM (optional, requires API key)."""
        # TODO: Implement LLM-based extraction
        # This would call an LLM with a prompt like:
        # "Extract memorable content from this text. Identify:
        #  - Episodes (significant experiences with lessons)
        #  - Beliefs (facts, patterns, principles learned)
        #  - Notes (decisions, insights, important observations)"
        return self._extract_with_patterns(raw_id, content)

    def list_suggestions(
        self: "Kernle",
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_confidence: float = 0.0,
        status: str = "pending",
    ) -> List[MemorySuggestion]:
        """List suggestions, optionally filtered."""
        return self._storage.list_suggestions(
            limit=limit,
            memory_type=memory_type,
            min_confidence=min_confidence,
            status=status,
        )

    def promote_suggestion(
        self: "Kernle",
        suggestion_id: str,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Promote a suggestion to a real memory.

        Args:
            suggestion_id: ID of suggestion to promote
            modifications: Optional changes to apply before promotion

        Returns:
            ID of created memory
        """
        suggestion = self._storage.get_suggestion(suggestion_id)
        if not suggestion:
            raise ValueError(f"Suggestion {suggestion_id} not found")

        if suggestion.status != "pending":
            raise ValueError(f"Suggestion {suggestion_id} already {suggestion.status}")

        # Apply modifications
        content = {**suggestion.content}
        if modifications:
            content.update(modifications)

        # Create the memory
        memory_id = None
        if suggestion.memory_type == "episode":
            memory_id = self.episode(
                objective=content.get("objective", ""),
                outcome=content.get("outcome_type", "success"),
                lessons=content.get("lessons", []),
                tags=["promoted-from-suggestion"],
            )
        elif suggestion.memory_type == "belief":
            memory_id = self.belief(
                statement=content.get("statement", ""),
                belief_type=content.get("belief_type", "fact"),
                confidence=content.get("confidence", 0.7),
            )
        elif suggestion.memory_type == "note":
            memory_id = self.note(
                content=content.get("content", ""),
                type=content.get("note_type", "note"),
                tags=["promoted-from-suggestion"],
            )
        else:
            raise ValueError(f"Unknown memory type: {suggestion.memory_type}")

        # Mark suggestion as promoted
        self._storage.update_suggestion(
            suggestion_id,
            status="promoted",
            resolved_at=datetime.now(timezone.utc),
            promoted_to=memory_id,
        )

        # Mark source raw entries as processed
        for raw_id in suggestion.source_raw_ids:
            self._storage.mark_raw_processed(
                raw_id,
                [f"{suggestion.memory_type}:{memory_id}"],
            )

        return memory_id

    def reject_suggestion(
        self: "Kernle",
        suggestion_id: str,
        reason: Optional[str] = None,
    ):
        """Reject a suggestion.

        Args:
            suggestion_id: ID of suggestion to reject
            reason: Optional reason (helps improve future extraction)
        """
        suggestion = self._storage.get_suggestion(suggestion_id)
        if not suggestion:
            raise ValueError(f"Suggestion {suggestion_id} not found")

        self._storage.update_suggestion(
            suggestion_id,
            status="rejected",
            resolved_at=datetime.now(timezone.utc),
            resolution_reason=reason,
        )

    def batch_promote(
        self: "Kernle",
        min_confidence: float = 0.8,
    ) -> Dict[str, int]:
        """Batch promote high-confidence suggestions.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with counts of promoted suggestions by type
        """
        suggestions = self.list_suggestions(
            limit=100,
            min_confidence=min_confidence,
            status="pending",
        )

        counts = {"episode": 0, "belief": 0, "note": 0}

        for suggestion in suggestions:
            try:
                self.promote_suggestion(suggestion.id)
                counts[suggestion.memory_type] += 1
            except Exception:
                continue

        return counts

    def batch_reject(
        self: "Kernle",
        older_than_days: int = 7,
    ) -> int:
        """Batch reject old pending suggestions.

        Args:
            older_than_days: Reject suggestions older than this

        Returns:
            Count of rejected suggestions
        """
        # TODO: Implement with date filtering in storage
        return 0
```

---

## Part 6: CLI Commands for Suggestions

### File: `kernle/cli/commands/suggestions.py`

```python
"""Suggestion management commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_suggestions(args, k: "Kernle"):
    """Handle suggestion subcommands."""

    if args.suggestion_action == "list" or args.suggestion_action is None:
        suggestions = k.list_suggestions(
            limit=getattr(args, 'limit', 10),
            memory_type=getattr(args, 'type', None),
            min_confidence=getattr(args, 'min_confidence', 0.0),
        )

        if not suggestions:
            print("No pending suggestions.")
            print("\nTo generate suggestions from raw entries:")
            print("  kernle suggestions extract")
            return

        if getattr(args, 'json', False):
            print(json.dumps([{
                "id": s.id,
                "type": s.memory_type,
                "content": s.content,
                "confidence": s.confidence,
            } for s in suggestions], indent=2))
            return

        print(f"Pending Suggestions ({len(suggestions)})")
        print("=" * 50)

        for s in suggestions:
            conf_bar = "█" * int(s.confidence * 5) + "░" * (5 - int(s.confidence * 5))
            print(f"\n[{s.id[:8]}] {s.memory_type.upper()} ({conf_bar} {s.confidence:.0%})")
            print(f"  {s.preview}...")
            print(f"  → kernle suggestions approve {s.id[:8]}")

    elif args.suggestion_action == "approve":
        suggestion_id = args.id

        # Resolve partial ID
        suggestions = k.list_suggestions(limit=100)
        matches = [s for s in suggestions if s.id.startswith(suggestion_id)]

        if len(matches) == 0:
            print(f"✗ Suggestion {suggestion_id} not found")
            return
        elif len(matches) > 1:
            print(f"✗ Ambiguous ID, matches: {[s.id[:8] for s in matches]}")
            return

        suggestion = matches[0]

        if getattr(args, 'edit', False):
            # TODO: Open in editor
            print("Editor mode not yet implemented")
            return

        memory_id = k.promote_suggestion(suggestion.id)
        print(f"✓ Promoted to {suggestion.memory_type}: {memory_id[:8]}...")

    elif args.suggestion_action == "reject":
        suggestion_id = args.id
        reason = getattr(args, 'reason', None)

        # Similar ID resolution...
        suggestions = k.list_suggestions(limit=100)
        matches = [s for s in suggestions if s.id.startswith(suggestion_id)]

        if len(matches) == 0:
            print(f"✗ Suggestion {suggestion_id} not found")
            return
        elif len(matches) > 1:
            print(f"✗ Ambiguous ID, matches: {[s.id[:8] for s in matches]}")
            return

        k.reject_suggestion(matches[0].id, reason)
        print(f"✓ Rejected suggestion {suggestion_id}")

    elif args.suggestion_action == "extract":
        # Extract suggestions from unprocessed raw entries
        raw_entries = k.list_raw(processed=False, limit=50)

        if not raw_entries:
            print("No unprocessed raw entries to analyze.")
            return

        print(f"Analyzing {len(raw_entries)} raw entries...")

        raw_ids = [e["id"] for e in raw_entries]
        suggestions = k.extract_suggestions(raw_ids)

        print(f"\n✓ Extracted {len(suggestions)} suggestions:")
        for s in suggestions:
            print(f"  [{s.memory_type}] {s.preview}...")

        print(f"\nReview with: kernle suggestions list")

    elif args.suggestion_action == "approve-all":
        min_conf = getattr(args, 'confidence_above', 0.8)
        counts = k.batch_promote(min_confidence=min_conf)

        total = sum(counts.values())
        print(f"✓ Promoted {total} suggestions:")
        for mem_type, count in counts.items():
            if count > 0:
                print(f"  {mem_type}: {count}")

    elif args.suggestion_action == "reject-all":
        older_than = getattr(args, 'older_than', 7)
        count = k.batch_reject(older_than_days=older_than)
        print(f"✓ Rejected {count} old suggestions")
```

---

## Part 7: Integration with Kernle Class

### Changes to `kernle/core.py`

```python
# Add import
from kernle.features.suggestions import SuggestionsMixin

# Update class definition
class Kernle(
    AnxietyMixin,
    EmotionsMixin,
    ForgettingMixin,
    KnowledgeMixin,
    MetaMemoryMixin,
    SuggestionsMixin,  # Add this
):
    """Main Kernle class with all features."""
    ...
```

---

## Part 8: Setup Instructions

### For Users

1. **Enable auto-capture hooks** (optional):

   Create `.claude/hooks.json` in your project with the hook configuration from Part 1.

2. **Set agent ID**:

   ```bash
   export KERNLE_AGENT_ID=my-agent
   ```

3. **Review suggestions periodically**:

   ```bash
   # Extract suggestions from captured raw content
   kernle suggestions extract

   # Review pending suggestions
   kernle suggestions list

   # Approve good ones
   kernle suggestions approve <id>

   # Reject irrelevant ones
   kernle suggestions reject <id>

   # Batch approve high-confidence
   kernle suggestions approve-all --confidence-above 0.8
   ```

### For Agents (CLAUDE.md)

Add to CLAUDE.md:

```markdown
## Memory (Kernle)

**At session start:**
```bash
kernle -a {AGENT_ID} load && kernle -a {AGENT_ID} anxiety -b
```

**Periodically during session:**
- Run `kernle suggestions list` to review pending memory suggestions
- Approve (`kernle suggestions approve <id>`) memories that are useful
- Reject (`kernle suggestions reject <id>`) irrelevant suggestions
- Create memories directly when appropriate

**Auto-capture is enabled:** Session content is automatically captured to the raw layer.
You control what gets promoted to permanent memory.
```

---

## Summary

This implementation provides:

1. **Hook-based auto-capture** - Claude Code hooks automatically capture session content to raw layer
2. **Quiet mode for CLI** - `--quiet` and `--stdin` flags for hook integration
3. **Suggestion extraction** - Pattern-based extraction with optional LLM upgrade path
4. **Agent-controlled promotion** - Agent reviews, modifies, and approves/rejects suggestions
5. **Preserves sovereignty** - System suggests, agent decides

The key insight: **auto-capture at raw layer + agent-controlled promotion preserves memory sovereignty while reducing friction**.
