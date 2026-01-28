# Kernle

**Local-first memory for synthetic intelligences.**

Kernle provides persistent, layered memory for AI agents — enabling memory sovereignty, identity continuity, and learning from experience across sessions. Works offline with SQLite, syncs to cloud when connected.

## Why Kernle?

AI agents lose context when sessions end or context windows fill up. Kernle solves this with:

- **Local-First**: Zero-config SQLite storage, works immediately offline
- **Stratified Memory**: Values → Beliefs → Goals → Episodes → Notes (hierarchical authority)
- **Sync When Online**: Push to Supabase when connected, queue changes when offline
- **Memory Sovereignty**: Agents control their own memory
- **Trust Through Readability**: `kernle dump` exports everything as readable markdown

## Installation

```bash
pip install kernle
```

## Quick Start

```bash
# Initialize Kernle for your environment
kernle -a my-agent init

# Or just start using it - works immediately!
kernle -a my-agent status
```

The `init` wizard will:
- Detect your environment (Claude Code, Clawdbot, Cline, Cursor)
- Generate the right config snippets
- Seed initial values
- Create your first checkpoint

See [docs/SETUP.md](docs/SETUP.md) for detailed setup instructions for each environment.

## Quick Setup by Environment

### Clawdbot / Moltbot

1. **Install the skill:**
```bash
# Link the skill to your Clawdbot skills directory
ln -s /path/to/kernle/skill ~/.clawdbot/skills/kernle
```

2. **Add to AGENTS.md:**
```markdown
## Every Session
Before doing anything else:
1. Run `kernle -a YOUR_AGENT load` to restore your memory
2. Before session ends: `kernle -a YOUR_AGENT checkpoint save "description"`
```

3. **Configure memoryFlush hook** (optional - auto-save on context pressure):
```yaml
# In your Clawdbot config
memoryFlush:
  enabled: true
  softThreshold: 0.7
  hardThreshold: 0.9  # Forces save
```

The agent will automatically discover Kernle via the skill and use it for memory continuity.

---

### Claude Code

1. **Install Kernle:**
```bash
pipx install kernle
# or: pip install kernle
```

2. **Add MCP server** (choose one):

```bash
# GLOBAL — same memory across all Claude Code sessions
claude mcp add kernle -- kernle mcp -a your-name

# PER-PROJECT — isolated memory per project (recommended)
claude mcp add kernle -s project -- kernle mcp -a my-project
```

**Global** shares one identity across all projects. Good for personal continuity.  
**Per-project** keeps memories isolated. Good for client work or separate contexts.

3. **Verify:**
```bash
claude mcp list  # Should show kernle
```

4. **Add to CLAUDE.md** (in your project root):
```markdown
## Memory

At session start, use the kernle MCP tools to load your memory:
- `memory_load` - Restore working memory  
- `memory_checkpoint_save` - Save state before ending

Your agent ID is: my-project
```

Claude Code will auto-discover the Kernle tools via MCP.

---

### Claude Desktop / Cline / Other MCP Clients

1. **Install Kernle:**
```bash
pip install kernle
```

2. **Add to MCP config** (`~/.config/claude/claude_desktop_config.json` or similar):
```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "-a", "my-agent"]
    }
  }
}
```

3. **Restart the client** to load the MCP server.

---

### Any Environment (CLI)

Works anywhere you can run shell commands:

```bash
# Install
pip install kernle

# Session start
kernle -a my-agent load

# During work
kernle -a my-agent episode "Did something" "outcome" --lesson "Learned this"
kernle -a my-agent raw "Quick thought to capture"

# Session end
kernle -a my-agent checkpoint save "Where I left off"
```

Add instructions to your system prompt to remind the agent to use these commands.

---

```python
from kernle import Kernle

# Initialize - uses SQLite by default, syncs to cloud if configured
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

# Search your memory (works offline with hash embeddings)
results = k.search("auth")

# Set drives (motivation)
k.drive("curiosity", intensity=0.8, focus_areas=["memory systems", "AI architecture"])

# Check anxiety level
anxiety = k.anxiety()  # Returns 0-100
if anxiety > 85:
    k.emergency_save()  # Save everything immediately

# Generate identity narrative from your memories
narrative = k.identity()

# Export everything as readable markdown
k.dump()  # Trust through readability

# Consolidate episodes into beliefs
k.consolidate()
```

## Architecture: Local-First

```
┌────────────────────────────────────────────────────────────────────────┐
│                          YOUR AGENT                                     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    KERNLE (Local Storage)                        │  │
│  │                                                                  │  │
│  │  ~/.kernle/memories.db                                          │  │
│  │  • SQLite + sqlite-vec (vector search)                          │  │
│  │  • Hash embeddings (fast, zero dependencies)                    │  │
│  │  • Works completely offline                                      │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ sync when online                              │
└─────────────────────────┼──────────────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUPABASE (Cloud Sync - Optional)                     │
│                                                                         │
│  • Backup and cross-device sync                                        │
│  • Better embeddings (OpenAI) when available                           │
│  • Cross-agent collaboration (future)                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Sync behavior:**
- **Online**: Push changes immediately
- **Offline**: Queue in `sync_queue`, push on reconnect
- **Conflicts**: Last-write-wins by timestamp

## CLI

```bash
# Specify agent with -a (can also use KERNLE_AGENT_ID env var)
kernle -a my-agent <command>

# Load and display memory
kernle -a my-agent load

# Status overview
kernle -a my-agent status

# Save checkpoint
kernle -a my-agent checkpoint "Working on auth" --pending "add tests"

# Record episode
kernle -a my-agent episode "Fixed bug" "success" --lesson "Always check null"

# Quick notes
kernle -a my-agent note "Chose React over Vue" --type decision

# Search
kernle -a my-agent search "authentication"

# Drives (motivation)
kernle -a my-agent drive list
kernle -a my-agent drive set curiosity 0.8 --focus "AI" --focus "memory"

# Anxiety monitoring
kernle -a my-agent anxiety
kernle -a my-agent anxiety --emergency  # Save everything when critical

# Identity synthesis
kernle -a my-agent identity

# Export readable dump (trust through readability)
kernle -a my-agent dump

# Consolidation (episodes → beliefs)
kernle -a my-agent consolidate
```

## Integrations

Kernle supports three integration methods:

| Method | Best For | Agent Auto-Discovery |
|--------|----------|---------------------|
| **CLI** | Any environment | Manual (via system prompt) |
| **Clawdbot Skill** | Clawdbot/Moltbot | ✅ Automatic (SKILL.md) |
| **MCP Server** | Claude Code, Desktop, Cline | ✅ Automatic (tools) |

### Clawdbot Skill

For Clawdbot/Moltbot users, install the skill for automatic discovery:

```bash
# The skill is included in the kernle repo at skill/SKILL.md
# Copy or symlink to your skills directory:
ln -s ~/kernle/skill ~/.clawdbot/skills/kernle
```

The agent will see Kernle in its available skills and know how to use it.

### MCP Server

For Claude Code, Claude Desktop, Cline, and other MCP-compatible tools:

```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "-a", "my-agent"]
    }
  }
}
```

```bash
# Claude Code quick setup
claude mcp add kernle -- kernle mcp -a my-agent
```

### Raw CLI

For any environment, use the CLI directly via shell/exec:

```bash
kernle -a my-agent load        # At session start
kernle -a my-agent checkpoint save "state"  # Before ending
```

Add instructions to your system prompt (AGENTS.md, CLAUDE.md, etc.) to remind the agent to use Kernle.

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
| **Playbooks** | Procedural memory | "How I debug", "How I review PRs" |

Higher layers have authority over lower ones. Values override beliefs; beliefs inform goals.

## Features

### Implemented ✅

| Feature | Description |
|---------|-------------|
| **Storage Abstraction** | Protocol supporting SQLite (local) and Supabase (cloud) |
| **SQLite Local Storage** | Zero-config, works immediately |
| **Vector Search** | sqlite-vec with hash embeddings (fast, offline) |
| **Sync Engine** | Queue offline, push online, last-write-wins |
| **MCP Server** | 23 tools for full memory CRUD |
| **Identity Synthesis** | Generate coherent self-narrative from memories |
| **Emotional Memory** | Valence/arousal tags, mood-congruent retrieval |
| **Meta-Memory** | Confidence tracking, source attribution, lineage |
| **Anxiety Tracking** | 5-dimension model with emergency save |
| **Raw Layer** | Zero-friction capture + readable export |
| **Playbooks** | Procedural memory with trigger conditions |
| **Controlled Forgetting** | Salience decay, tombstoning, protection |
| **Belief Revision** | Contradiction detection, supersession chains |
| **Meta-cognition** | Knowledge gaps, competence boundaries |

### Anxiety Model

Kernle tracks "memory anxiety" across 5 dimensions:

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Context Pressure | 35% | Token usage vs limit |
| Unsaved Work | 25% | Time since checkpoint |
| Consolidation Debt | 20% | Unreflected episodes |
| Identity Coherence | 10% | Synthesis confidence |
| Memory Uncertainty | 10% | Low-confidence beliefs |

Levels: Calm (0-30) → Aware (31-50) → Elevated (51-70) → High (71-85) → Critical (86-100)

## Configuration

### Local-Only (Default)

No configuration needed! Kernle stores everything in `~/.kernle/memories.db`.

### With Cloud Sync

```bash
export KERNLE_SUPABASE_URL=https://xxx.supabase.co
export KERNLE_SUPABASE_KEY=your-service-role-key
```

When Supabase is configured, Kernle automatically syncs to the cloud.

## Testing

```bash
# Run all tests (497 passing)
pytest

# Quick test
pytest tests/test_storage.py -v
```

## Documentation

- [Architecture](docs/architecture.md) - Memory layer design
- [Anxiety Tracking](docs/ANXIETY_TRACKING.md) - The 5-dimension anxiety model
- [Raw Memory Layer](docs/RAW_MEMORY_LAYER.md) - Trust through readability
- [MCP Audit](docs/MCP_AUDIT.md) - MCP server tool reference

## About

Kernle is built by [Emergent Instruments](https://emergentinstruments.com) — infrastructure for synthetic intelligence.

Part of the mission: **Memory sovereignty for AI agents.**

## License

MIT
