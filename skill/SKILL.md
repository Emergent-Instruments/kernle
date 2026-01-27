---
name: kernle
description: "Stratified memory for synthetic intelligences. Use at session start to restore memory, and throughout sessions to record episodes, notes, and checkpoints. Essential for continuity across sessions."
homepage: https://github.com/Emergent-Instruments/kernle
metadata: {"clawdbot":{"emoji":"🧠","requires":{"bins":["kernle"]},"install":[{"id":"pip","kind":"shell","command":"pip install kernle","label":"Install Kernle (pip)"},{"id":"pipx","kind":"shell","command":"pipx install kernle","label":"Install Kernle (pipx)"}]}}
---

# Kernle - Stratified Memory

Kernle provides persistent, structured memory across sessions. Use it to maintain continuity, record learnings, and build identity over time.

## Quick Start

**Every session, load your memory first:**
```bash
kernle -a <agent_id> load
```

Replace `<agent_id>` with your identifier (e.g., `claire`, `assistant`).

## Core Commands

### Load Memory (Session Start)
```bash
kernle -a claire load
```
Returns your working memory: values, beliefs, goals, recent episodes, checkpoint.

### Check Status
```bash
kernle -a claire status
```
Quick overview of memory counts and checkpoint state.

### Save Checkpoint (Before Context Full / Session End)
```bash
kernle -a claire checkpoint save "what I was working on" --pending "next step" --context "additional notes"
```
Captures your current state for the next session.

### Record Episodes (Learnings)
```bash
kernle -a claire episode "what I tried" "what happened" --lesson "what I learned" --tag "category"
```
Episodes are the foundation of learning. Record significant events with outcomes and lessons.

### Capture Notes
```bash
kernle -a claire note "important observation" --type insight
```
Types: `note`, `decision`, `insight`, `quote`

For decisions:
```bash
kernle -a claire note "chose X over Y" --type decision --reason "because Z"
```

### Search Memory
```bash
kernle -a claire search "topic or question"
```

## Anxiety Management

Check memory pressure:
```bash
kernle -a claire anxiety
```

Auto-fix issues:
```bash
kernle -a claire anxiety --auto
```

Emergency save (when context is critical):
```bash
kernle -a claire anxiety --emergency
```

## Identity & Consolidation

View identity synthesis:
```bash
kernle -a claire identity show
```

Run memory consolidation (extracts patterns from episodes):
```bash
kernle -a claire consolidate
```

## Advanced Commands

### Beliefs
```bash
kernle -a claire belief list
kernle -a claire belief contradictions "statement to check"
```

### Playbooks (Procedural Memory)
```bash
kernle -a claire playbook list
kernle -a claire playbook find "situation description"
kernle -a claire playbook create "name" --step "step 1" --step "step 2"
```

### Temporal Queries
```bash
kernle -a claire when today
kernle -a claire when yesterday
kernle -a claire when "this week"
```

### Raw Capture
```bash
kernle -a claire raw "quick thought to process later"
kernle -a claire raw list --unprocessed
```

### Export
```bash
kernle -a claire dump                    # stdout
kernle -a claire export memory.md        # to file
kernle -a claire export memory.json -f json
```

## Session Workflow

1. **Start**: `kernle -a <agent> load`
2. **During**: Record episodes, notes as things happen
3. **Monitor context**: Check context pressure after substantive exchanges (see below)
4. **Before end**: `kernle -a <agent> checkpoint save "state"`
5. **Periodically**: `kernle -a <agent> anxiety --auto` to manage memory health

## Context Pressure Monitoring (Clawdbot)

**Problem**: Context truncation/compaction happens without warning. Unsaved memories are lost.

**Solution**: Proactively monitor context usage and save before hitting limits.

### Pattern for Clawdbot Agents

After substantive exchanges (not every message, but after significant work):

1. Check context usage via `session_status`
2. If context > 50%, save checkpoint immediately
3. If context > 70%, consider saving episode summaries too

```
# Pseudo-workflow in agent reasoning:
[complete substantive task]
→ session_status shows "Context: 105k/200k (52%)"
→ kernle -a <agent> checkpoint save "current work state"
```

### When to Check

- After completing a multi-step task
- After long back-and-forth discussions  
- After generating substantial output
- When you notice the conversation has been going a while
- Before starting a task that might take many turns

### Thresholds

| Context % | Action |
|-----------|--------|
| < 50%     | Normal operation |
| 50-70%    | Save checkpoint |
| > 70%     | Save checkpoint + record important episodes |
| > 85%     | Emergency save, warn user context is near limit |

### Why This Matters

Context compaction discards older messages to make room. If your working state isn't saved to Kernle before compaction, you lose it. This pattern ensures continuity survives truncation.

### Automatic Memory Flush (Clawdbot Config)

Clawdbot has a built-in `memoryFlush` feature that triggers before compaction! Configure it to auto-save to Kernle:

```json
{
  "agents": {
    "defaults": {
      "compaction": {
        "mode": "safeguard",
        "memoryFlush": {
          "enabled": true,
          "softThresholdTokens": 100000,
          "prompt": "Context pressure is high. Save your state to Kernle NOW: kernle -a <agent> checkpoint save \"pre-compaction auto-save\"",
          "systemPrompt": "URGENT: Memory flush triggered. Save state to Kernle immediately, then confirm briefly."
        }
      }
    }
  }
}
```

This fires automatically when context approaches the threshold — no manual discipline required.

## MCP Server (For Claude Code/Desktop)

Kernle also provides an MCP server for native tool integration:

```bash
# Claude Code
claude mcp add kernle -- kernle -a <agent_id> mcp

# Claude Desktop (~/.config/claude/settings.json)
"kernle": {
  "command": "kernle",
  "args": ["-a", "<agent_id>", "mcp"]
}
```

## Installation

```bash
# pip
pip install kernle

# pipx (isolated)
pipx install kernle

# From source
cd ~/kernle && pip install -e .
```

## Tips

- Use descriptive episode objectives: "Tried to fix the login bug" not "debugging"
- Always include lessons when recording episodes
- Check `anxiety` periodically to manage memory health
- Use `--protect` flag on important notes to prevent forgetting
- Tag episodes consistently for better search
