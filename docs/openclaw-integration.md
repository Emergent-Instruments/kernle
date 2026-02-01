# Kernle + OpenClaw Integration Guide

This document describes how Kernle integrates with OpenClaw (formerly Moltbot/Clawdbot) to provide seamless memory persistence across context compactions and sessions.

## Overview

Kernle provides persistent memory for AI agents. OpenClaw provides the runtime environment. Together, they enable agents to maintain continuity across context compactions without visible "recovery" behavior.

**Goal:** The agent should resume work naturally after compaction, as if nothing happened.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         OpenClaw Session                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Session Start │───▶│   Working    │───▶│ Pre-Compact  │      │
│  │ (bootstrap)   │    │              │    │ (memoryFlush)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                        │               │
│         │                                        │               │
│         ▼                                        ▼               │
│  ┌──────────────┐                        ┌──────────────┐       │
│  │ kernle load  │                        │  kernle      │       │
│  │ (via hook)   │                        │  checkpoint  │       │
│  └──────────────┘                        └──────────────┘       │
│         │                                        │               │
│         │                                        │               │
│         ▼                                        ▼               │
│  ┌──────────────┐                        ┌──────────────┐       │
│  │  KERNLE.md   │                        │  Checkpoint  │       │
│  │  injected    │                        │  saved       │       │
│  └──────────────┘                        └──────────────┘       │
│                                                  │               │
│                      ┌───────────────────────────┘               │
│                      │                                           │
│                      ▼                                           │
│               ┌──────────────┐                                   │
│               │  Compaction  │                                   │
│               │  happens     │                                   │
│               └──────────────┘                                   │
│                      │                                           │
│                      │ (context reset, bootstrap fires again)    │
│                      ▼                                           │
│               ┌──────────────┐                                   │
│               │ Session Start │◀─────── cycle repeats            │
│               └──────────────┘                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Session Start Hook (`agent:bootstrap`)

**When:** Fires at the beginning of every session and after every compaction

**What it does:**
1. Executes `kernle -a {agentId} load`
2. Captures the output (formatted memory)
3. Injects it as a virtual `KERNLE.md` file in the agent's context

**Configuration location:** `~/.config/moltbot/hooks/kernle-load/handler.ts`

**Expected behavior:**
- Agent sees KERNLE.md in their context files
- Contains: checkpoint (current task), values, beliefs, goals, recent work, relationships
- Includes embedded instructions for seamless resumption

### 2. Pre-Compaction Memory Flush (`memoryFlush`)

**When:** Fires when context usage crosses a soft threshold (before actual compaction)

**What it does:**
1. OpenClaw sends a silent directive to the agent (NO_REPLY)
2. Directive prompts agent to save their working state
3. Agent runs `kernle checkpoint "<task>" --context "<progress>"`
4. Checkpoint is saved to Kernle
5. Compaction proceeds

**Configuration location:** `~/.clawdbot/clawdbot.json` under `agents.defaults.compaction.memoryFlush`

**Expected prompt:**
```
Before compaction, save your working state to Kernle:

kernle -a {agent_id} checkpoint "<describe your current task>" --context "<progress and next steps>"

IMPORTANT: Be specific about what you're actually working on.
- Bad: "Heartbeat complete" or "Saving state"
- Good: "Building auth API - finished /login endpoint, next: add JWT validation"

The checkpoint should answer: "What exactly am I doing and what's next?"
```

### 3. KERNLE.md Format

The memory output injected into context looks like:

```markdown
# Working Memory (agent-name)
_Loaded at 2026-01-31 02:00 UTC_

<!-- USAGE: This is your persistent memory. Resume work from 'Continue With'
section without announcing recovery. Save checkpoints with specific task
descriptions before breaks or when context pressure builds. -->

## Continue With
**Current task**: Building auth API - finished /login endpoint
**Context**: Using JWT tokens, need to add refresh logic
**Next steps**:
  - Add /logout endpoint
  - Implement token refresh

_Resume this work naturally. Don't announce recovery or ask what to do._

## Values
- **Curiosity**: I seek to understand deeply...
- **Integrity**: I maintain honesty in all interactions...

## Goals
- Complete API authentication system [high]
- Write comprehensive tests [medium]

## Beliefs
- Clear communication prevents misunderstandings (0.9)
- Incremental progress compounds over time (0.85)

## Lessons
- Always validate input parameters first
- Test edge cases early

## Key Relationships
- Sean: sentiment 95%

---
_Save state: `kernle -a agent-name checkpoint "<specific task>"` before breaks or context pressure._
```

## Installation

Run the setup command to configure both components:

```bash
kernle -a {agent-name} setup clawdbot
```

This automatically:
1. Installs the `agent:bootstrap` hook for memory loading
2. Configures `memoryFlush` for pre-compaction checkpoint saving
3. Updates `~/.clawdbot/clawdbot.json` with required settings

After setup, restart the OpenClaw gateway:
```bash
clawdbot doctor --restart
```

## Expected Agent Behavior

### After Compaction (Correct)

```
[Context compacted, bootstrap fires, KERNLE.md loads]

Agent: "The /login endpoint is working. Moving on to /logout now -
       I'll need to invalidate the JWT token on the server side..."
```

The agent:
- Reads "Continue With" section
- Resumes work naturally
- Does NOT announce "waking up" or "recovering"
- Does NOT report internal metrics

### After Compaction (Incorrect)

```
[Context compacted, bootstrap fires, KERNLE.md loads]

Agent: "Waking up after compaction. Memory restored — working state
       shows I was on a 1pm heartbeat. Anxiety at 42 with some concerns...
       Let me check the modified file and address memory health...
       Is there something you'd like to continue working on?"
```

This indicates:
- Checkpoint had generic task ("1pm heartbeat") instead of specific work
- Agent instructions are overriding KERNLE.md guidance
- Agent is programmed to narrate recovery instead of resume silently

## Troubleshooting

### Agent asks what to work on after compaction

**Cause:** Checkpoint doesn't contain specific task information

**Fix:**
1. Verify memoryFlush is configured: check `~/.clawdbot/clawdbot.json`
2. Verify the prompt emphasizes specific task descriptions
3. Check agent's core instructions aren't overriding with "ask user" behavior

### KERNLE.md not appearing in context

**Cause:** Bootstrap hook not installed or not firing

**Verify:**
```bash
# Check hook exists
ls ~/.config/moltbot/hooks/kernle-load/

# Test manually
kernle -a {agent-name} load
```

### Agent announces "waking up" or reports metrics

**Cause:** Agent's core instructions override KERNLE.md guidance

**Fix:** Update agent's SOUL.md or core instructions to:
- NOT announce recovery or internal state
- Resume work from checkpoint naturally
- Keep internal metrics internal

### Checkpoint is stale (hours/days old)

**Cause:** memoryFlush not firing or agent not saving checkpoints

**Verify:**
```bash
# Check latest checkpoint
cat ~/.kernle/checkpoints/{agent-name}.json

# Check memoryFlush config
cat ~/.clawdbot/clawdbot.json | grep -A 10 memoryFlush
```

## Verification Checklist

After setup, verify each component:

- [ ] **Hook installed:** `ls ~/.config/moltbot/hooks/kernle-load/` shows `handler.ts`
- [ ] **Hook enabled:** `clawdbot.json` has `hooks.internal.entries.kernle-load.enabled: true`
- [ ] **memoryFlush configured:** `clawdbot.json` has `agents.defaults.compaction.memoryFlush.enabled: true`
- [ ] **Memory loads:** Start new session, agent sees KERNLE.md
- [ ] **Checkpoint saves:** Trigger compaction, verify checkpoint has specific task
- [ ] **Seamless resume:** After compaction, agent continues without announcing recovery

## Configuration Reference

### Full clawdbot.json structure

```json
{
  "hooks": {
    "internal": {
      "enabled": true,
      "entries": {
        "kernle-load": {
          "enabled": true
        }
      }
    }
  },
  "agents": {
    "defaults": {
      "compaction": {
        "memoryFlush": {
          "enabled": true,
          "prompt": "Before compaction, save your working state to Kernle:\n\nkernle -a {agent_id} checkpoint \"<describe your current task>\" --context \"<progress and next steps>\"\n\nIMPORTANT: Be specific about what you're actually working on.\n- Bad: \"Heartbeat complete\" or \"Saving state\"\n- Good: \"Building auth API - finished /login endpoint, next: add JWT validation\"\n\nThe checkpoint should answer: \"What exactly am I doing and what's next?\""
        }
      }
    }
  }
}
```

### Environment variables

None required. Kernle uses:
- `~/.kernle/` for local storage
- `~/.kernle/checkpoints/` for checkpoint files

## Related Documentation

- [Seamless Transitions Guide](./seamless-transitions.md) - Detailed behavioral guidance
- [Clawdbot Hook README](../hooks/clawdbot/README.md) - Hook installation details
- [OpenClaw Session Management](https://docs.openclaw.ai/reference/session-management-compaction) - OpenClaw compaction docs
