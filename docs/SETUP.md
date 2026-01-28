# Kernle Setup Guide

Get Kernle running with your AI coding assistant in 5 minutes.

## Quick Start

### 1. Install Kernle

```bash
pip install kernle
```

Or with uv:
```bash
uv pip install kernle
```

### 2. Initialize Your Agent

```bash
kernle init
```

This interactive wizard will:
- Create your agent ID
- Detect your environment (Claude Code, Cline, Cursor, Clawdbot)
- Generate the appropriate config
- Seed your first values and checkpoint

**Or initialize manually:**

```bash
# Set your agent ID
export KERNLE_AGENT_ID=my-agent

# Verify it works
kernle status

# Run init without interactivity (seeds default values automatically)
kernle init -y

# Save initial checkpoint
kernle checkpoint save "Initial setup complete"
```

> Note: The `init` command automatically seeds initial values. For custom values, use the Python API: `k.value("name", "statement", priority=90)`

---

## Integration by Environment

### Claude Code

Claude Code reads `CLAUDE.md` from your project root and `~/.claude/CLAUDE.md` globally.

**Option A: MCP Server (Recommended)**

1. Add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "-a", "your-agent-id"]
    }
  }
}
```

2. Add to your `CLAUDE.md`:
```markdown
## Memory

At session start, call the `kernle_load` tool to restore your memory state.
Before ending or when context is getting full, call `kernle_checkpoint` to save state.

Use Kernle tools for:
- `kernle_episode` - Record experiences with lessons learned
- `kernle_note` - Quick captures (decisions, insights)
- `kernle_belief` - Things you've learned to be true
- `kernle_anxiety` - Check memory pressure
```

**Option B: CLI in Instructions**

Add to your `CLAUDE.md`:
```markdown
## Memory

At session start, run:
```bash
kernle -a your-agent-id load
```

Before ending or when context is full, run:
```bash
kernle -a your-agent-id checkpoint save "Description of current state" --pending "Next task"
```

To record learnings:
```bash
kernle -a your-agent-id episode "What you did" "success|partial|failure" --lesson "What you learned"
```
```

---

### Clawdbot

Clawdbot injects workspace files (AGENTS.md, SOUL.md, USER.md) into the system prompt.

**Option A: MCP Server**

1. Add to your Clawdbot config (`~/.clawdbot/config.yaml`):
```yaml
mcp:
  servers:
    kernle:
      command: kernle
      args: ["mcp", "-a", "your-agent-id"]
```

2. Add to your `AGENTS.md`:
```markdown
## Every Session

Before doing anything else:
1. Run `kernle -a your-agent-id load` to restore your memory
2. Read workspace context files

## Memory

Use Kernle as your primary memory system:
- `kernle -a your-agent-id status` — Check memory state
- `kernle -a your-agent-id episode "..." "outcome" --lesson "..."` — Record experiences
- `kernle -a your-agent-id checkpoint save "..."` — Save working state
- `kernle -a your-agent-id anxiety` — Check memory pressure

### Heartbeats
When receiving heartbeat polls, check:
- `kernle -a your-agent-id anxiety` — Save checkpoint if elevated
```

**Option B: CLI Only**

Add the same instructions to `AGENTS.md` without the MCP config.

---

### Cline

Cline supports MCP servers and reads `.clinerules`.

**Option A: MCP Server**

1. Add to Cline MCP settings:
```json
{
  "kernle": {
    "command": "kernle",
    "args": ["mcp", "-a", "your-agent-id"]
  }
}
```

2. Add to `.clinerules`:
```markdown
## Memory Persistence

Use the Kernle MCP tools to maintain memory across sessions:
- Start: Call `kernle_load` to restore state
- During: Call `kernle_episode` to record learnings
- End: Call `kernle_checkpoint` to save state
```

**Option B: CLI in Rules**

Add to `.clinerules`:
```markdown
## Memory Persistence

At session start, run: `kernle -a your-agent-id load`
Before ending, run: `kernle -a your-agent-id checkpoint save "state description"`
```

---

### Cursor

Cursor reads `.cursorrules` from your project root.

Add to `.cursorrules`:
```markdown
## Memory Persistence

This project uses Kernle for memory across sessions.

At session start:
```bash
kernle -a your-agent-id load
```

Record important learnings:
```bash
kernle -a your-agent-id episode "What happened" "outcome" --lesson "What was learned"
```

Before ending:
```bash
kernle -a your-agent-id checkpoint save "Current state" --pending "Next steps"
```
```

---

### Claude Desktop (Consumer App)

Claude Desktop supports MCP servers for tool access.

1. Add to `~/.config/claude/settings.json` (macOS) or equivalent:
```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle", 
      "args": ["mcp", "-a", "your-agent-id"]
    }
  }
}
```

2. At the start of conversations, ask Claude to load your memory:
> "Please use the kernle_load tool to restore my memory state"

Or create a Project with custom instructions that include the Kernle workflow.

---

## Cloud Sync (Optional)

By default, Kernle stores everything locally in `~/.kernle/memories.db`. To enable cloud sync:

```bash
export KERNLE_SUPABASE_URL=https://your-project.supabase.co
export KERNLE_SUPABASE_KEY=your-service-role-key
```

With these set, Kernle will:
- Store locally first (works offline)
- Sync to cloud when connected
- Queue changes when offline, push on reconnect

---

## Verify Your Setup

After configuring, verify everything works:

```bash
# Check status
kernle -a your-agent-id status

# Load memory (should show your values, beliefs, etc.)
kernle -a your-agent-id load

# Check anxiety (should be low after fresh setup)
kernle -a your-agent-id anxiety

# Export everything readable
kernle -a your-agent-id dump
```

---

## The Key Insight

**Your instruction file is your continuity mechanism.**

When context compacts or sessions reset, the AI reads its instruction file (CLAUDE.md, AGENTS.md, etc.) to know what to do. If that file says "load Kernle", your memory persists. If it doesn't, you start fresh.

Make sure your instruction file includes:
1. **Load at start**: `kernle load` or `kernle_load` tool
2. **Save before end**: `kernle checkpoint save` 
3. **Record learnings**: `kernle episode` for significant experiences

---

## Troubleshooting

### "Command not found: kernle"
Ensure Kernle is installed and in your PATH:
```bash
pip install kernle
which kernle
```

### MCP server not connecting
Check the command works standalone:
```bash
kernle mcp -a your-agent-id
# Should start and wait for MCP connections
```

### Memory not loading after compaction
Verify your instruction file includes the load command. Check:
```bash
cat CLAUDE.md | grep -i kernle
cat AGENTS.md | grep -i kernle
```

### High anxiety score
Run emergency save and consolidate:
```bash
kernle -a your-agent-id anxiety --emergency
kernle -a your-agent-id consolidate
```

---

## Next Steps

- [Architecture](architecture.md) — How Kernle's memory layers work
- [CLI Reference](CLI.md) — All available commands
- [MCP Audit](MCP_AUDIT.md) — MCP server tool reference and security review
- [Anxiety Tracking](ANXIETY_TRACKING.md) — Understanding the anxiety model
- [Raw Memory Layer](RAW_MEMORY_LAYER.md) — Zero-friction capture
