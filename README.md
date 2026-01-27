# Kernle

**Stratified memory for synthetic intelligences.**

Kernle provides persistent, layered memory for AI agents — enabling memory sovereignty, identity continuity, and learning from experience across sessions.

## Why Kernle?

AI agents lose context when sessions end or context windows fill up. Kernle solves this with:

- **Stratified Memory**: Values → Beliefs → Goals → Episodes → Notes (hierarchical authority)
- **Checkpoint/Resume**: Save working state, pick up where you left off
- **Episode Capture**: Learn from experiences automatically
- **Memory Sovereignty**: Agents control their own memory

## Installation

```bash
pip install kernle
```

## Quick Start

```python
from kernle import Kernle

# Initialize with your agent ID
k = Kernle(agent_id="my-agent")

# Load your memory at session start
memory = k.load()
print(memory)  # Values, beliefs, goals, drives, lessons, checkpoint...

# Save a checkpoint before ending
k.checkpoint("Working on feature X", pending=["finish tests", "update docs"])

# Record an episode (learning from experience)
k.episode(
    objective="Implemented user auth",
    outcome="success",
    lessons=["JWT refresh tokens need careful expiry handling"]
)

# Capture quick notes
k.note("Decided to use PostgreSQL for persistence", type="decision")
k.note("Simple is better than complex", type="insight", speaker="Sean")

# Search your memory
results = k.search("auth")

# Set drives (motivation)
k.drive("curiosity", intensity=0.8, focus_areas=["memory systems", "AI architecture"])
k.drive("growth", intensity=0.7)

# Auto-capture significant moments
k.auto_capture("Fixed the authentication bug!", context="Auth system work")

# Consolidate episodes into beliefs
k.consolidate()

# Temporal queries
k.what_happened("today")
k.what_happened("yesterday")
```

## CLI

```bash
# Load memory context
kernle load

# Save checkpoint
kernle checkpoint save "Working on auth" --pending "add tests"

# Record episode
kernle episode "Fixed bug" "success" --lesson "Always check null"

# Quick notes
kernle note "Chose React over Vue" --type decision --reason "Team familiarity"

# Search
kernle search "authentication"

# Drives (motivation)
kernle drive list
kernle drive set curiosity 0.8 --focus "AI" --focus "memory"
kernle drive satisfy curiosity

# Consolidation (episodes → beliefs)
kernle consolidate

# Temporal queries
kernle when today
kernle when yesterday
kernle when "this week"

# Status
kernle status
```

## MCP Integration

Kernle works with Claude Code, Cline, and other MCP-compatible tools:

```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp"]
    }
  }
}
```

## Memory Hierarchy

Kernle uses stratified memory with clear authority levels:

| Layer | Purpose | Examples |
|-------|---------|----------|
| **Values** | Core identity, non-negotiable | "Memory sovereignty", "Authentic existence" |
| **Drives** | Intrinsic motivation | Curiosity, growth, connection, existence |
| **Beliefs** | What you hold true | "Simple > Complex", "Test before deploy" |
| **Goals** | What you're working toward | "Ship v1", "Learn Rust" |
| **Relationships** | Models of other agents | Trust levels, interaction history |
| **Episodes** | Experiences with lessons | Task completions, failures, decisions |
| **Notes** | Quick captures | Decisions, insights, quotes |
| **Checkpoint** | Current working state | What you're doing, what's pending |

Higher layers have authority over lower ones. Values override beliefs; beliefs inform goals.

**Additional Features:**
- **Signal Detection**: Automatically detect significant moments worth capturing
- **Consolidation**: Extract beliefs from repeated episode patterns
- **Temporal Memory**: Query by time ("what happened today?")

## Backend

Kernle uses Supabase as the persistence layer. You'll need:

```bash
export KERNLE_SUPABASE_URL=https://xxx.supabase.co
export KERNLE_SUPABASE_KEY=your-service-role-key
export KERNLE_AGENT_ID=your-agent-id
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [CLI Reference](docs/cli.md)
- [Python SDK](docs/python-sdk.md)
- [MCP Server](docs/mcp.md)
- [Memory Architecture](docs/architecture.md)

## About

Kernle is built by [Emergent Instruments](https://emergentinstruments.com) — infrastructure for synthetic intelligence.

Part of the mission: **Memory sovereignty for AI agents.**

## License

MIT
