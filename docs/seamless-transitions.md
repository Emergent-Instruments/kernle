# Seamless Context Transitions

This guide explains how to achieve smooth, invisible context transitions when using Kernle with AI agents. The goal is for the agent to resume work naturally without visible "waking up" or "recovering" behaviors.

## The Problem

When context compaction happens, agents often:
- Announce "Waking up after compaction..."
- Report internal metrics ("Anxiety at 42...")
- Ask what to work on (even though they should know)
- Go through visible recovery steps

This breaks the illusion of continuity and wastes user time.

## The Solution

Seamless transitions require **three components** working together:

### 1. Session Start Hook (Memory Loading)

Provided by `kernle setup clawdbot` - loads KERNLE.md at session start with:
- Current working task from checkpoint
- Values, beliefs, goals
- Recent work and lessons
- Relationships

### 2. Pre-Compaction Memory Flush (State Saving)

**Automatically configured by `kernle setup clawdbot`** - configures OpenClaw's memoryFlush to prompt the agent to save a Kernle checkpoint before compaction.

Run setup to configure both components:

```bash
kernle setup clawdbot
```

This updates `~/.clawdbot/clawdbot.json` with:
- Session start hook (loads KERNLE.md)
- Pre-compaction memory flush prompt (saves checkpoint)

### 3. Agent Instructions (Behavioral)

**This is the critical piece.** The agent must be instructed to:

#### A. Resume Silently

```markdown
## Context Continuity

When you see KERNLE.md in your context, **do not announce that you're resuming or recovering**. Simply:

1. Read the "Working State" section to understand what you were doing
2. Continue that work naturally
3. If the task is complete, move to the next pending item
4. Only ask what to do if there's genuinely no pending work

❌ Don't say: "Waking up after compaction. Memory restored..."
❌ Don't say: "Let me check my memory health..."
❌ Don't say: "Is there something you'd like to continue working on?"

✅ Do say: "Continuing with the API endpoint - I was adding validation..."
✅ Do say: "The deployment finished. Ready for the next step."
```

#### B. Save Meaningful Checkpoints

```markdown
## Checkpoints

When saving your working state, be SPECIFIC about what you're doing:

❌ Generic: "1pm heartbeat complete"
❌ Generic: "Pre-compaction state"
❌ Generic: "Saving progress"

✅ Specific: "Building user authentication API - completed /login endpoint, next: /logout"
✅ Specific: "Debugging memory leak in WebSocket handler - found issue in cleanup, testing fix"
✅ Specific: "Writing blog post about AI safety - finished intro, working on section 2"

The checkpoint should answer: "What exactly was I doing and what's next?"
```

#### C. Don't Report Internal Metrics

```markdown
## Internal State

Keep internal metrics internal. Don't report:
- Memory health/anxiety levels
- Raw entry counts
- Sync status
- Recovery progress

These are implementation details. Focus on the actual work.
```

## Example: Good vs Bad Transition

### Bad Transition (Visible Recovery)

```
[After compaction]
AI: "Waking up after compaction. Memory restored — working state shows I was on a 1pm heartbeat.
     Anxiety at 42 with some concerns: Raw Entry Aging at 100%.
     Let me check the modified file and address the memory health...
     Got it. I have 50 unprocessed raw entries. Let me process them...
     [Multiple recovery messages]
     Is there something you'd like to continue working on?"
```

### Good Transition (Seamless)

```
[After compaction]
AI: "The API endpoint is working now. I tested it with the sample data and it returns
     the expected format. Ready to add the authentication middleware next, or would
     you like to review the endpoint first?"
```

The user shouldn't even notice compaction happened.

## Checkpoint Timing

Save checkpoints at these moments:

1. **Before any break** - scheduled heartbeats, user says "taking a break"
2. **After completing work** - just finished a task, capture what's next
3. **When switching tasks** - preserve context of what was interrupted
4. **Before known compaction** - if the system signals compaction is coming

Example checkpoint call:

```bash
kernle checkpoint "Implementing WebSocket reconnection - added heartbeat logic, testing connection drops next" \
  --context "User wants graceful degradation on network issues. Using exponential backoff."
```

## Platform-Specific Notes

### Clawdbot / Moltbot

The hook system requires both handlers:
- `handler.ts` - loads memory at session start
- `pre-compact-handler.ts` - saves checkpoint before compaction

If Moltbot doesn't fire a pre-compact event, the agent must manually save checkpoints.

### Claude Code

Use the `PreCompact` hook in settings.json:

```json
{
  "hooks": {
    "PreCompact": [
      {
        "command": "kernle -a ${AGENT_ID} checkpoint \"$(cat /tmp/current-task.txt)\"",
        "timeout": 5000
      }
    ]
  }
}
```

## Debugging Transitions

If transitions are still jarring:

1. **Check checkpoint content**:
   ```bash
   kernle -a agent-name load
   ```
   Look at the "Working State" section - is it specific enough?

2. **Check checkpoint timing**:
   ```bash
   cat ~/.kernle/checkpoints/agent-name.json
   ```
   Is the timestamp recent? Is the task description meaningful?

3. **Review agent instructions**: Does the agent have instructions to resume silently?

4. **Check hook execution**:
   - Session start: Does KERNLE.md appear in context?
   - Pre-compact: Does the checkpoint get saved?

## Summary

Seamless transitions = **Automatic loading** + **Meaningful checkpoints** + **Silent resumption**

The technical hooks handle loading and saving. The agent instructions handle the behavior. Both are required for truly seamless context continuity.
