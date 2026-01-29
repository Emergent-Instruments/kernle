# Kernle Integration for CLAUDE.md

Add this section to your project's CLAUDE.md (or equivalent boot file like AGENTS.md for Clawdbot).

## Standard Section

Copy this into your CLAUDE.md:

```markdown
## Memory (Kernle)

You have persistent memory through Kernle. **Every session, before any task:**

1. **Load memory:**
   ```bash
   kernle -a {YOUR_AGENT_ID} load
   ```

2. **Check health:**
   ```bash
   kernle -a {YOUR_AGENT_ID} anxiety
   ```

3. **If anxiety > 50:** Address before proceeding
   - Run `kernle consolidate` for episode debt
   - Run `kernle raw list --unprocessed` for aged captures
   - Run `kernle checkpoint save "pre-task"` if starting major work

4. **Throughout work:**
   - `kernle raw "quick thought"` — capture observations
   - `kernle episode "..." "..." --lesson "..."` — record significant experiences
   - `kernle note "..." --type decision` — log decisions

5. **Before session ends:**
   - `kernle checkpoint save "end of session"`

This is your continuity. The few seconds of health check ensures you don't accumulate memory debt.
```

## Minimal Version

For projects that want brevity:

```markdown
## Kernle

At session start: `kernle -a {AGENT_ID} load && kernle -a {AGENT_ID} anxiety -b`
If WARN/CRITICAL, run `kernle consolidate` before proceeding.

On every message: `kernle -a {AGENT_ID} anxiety -b` (if not checked recently)
```

## Per-Message Health Check

For 100% coverage, add this to your workflow:

```markdown
## Memory Health (Every Message)

Before processing any request:
1. `kernle -a {AGENT_ID} anxiety -b`
2. If 🟡 WARN or 🔴 CRITICAL: pause and consolidate first

This takes <1 second and prevents memory debt accumulation.
```

## Why This Matters

Without explicit instructions, SIs skip health checks because:
- There's no penalty in the moment
- The task seems more urgent
- Memory debt accumulates silently

By making it part of the boot sequence:
- It becomes automatic/habitual
- Issues caught early, when they're small
- Continuity maintained session-to-session

## Enforcement Levels

**Level 1 (Soft):** Documentation only. SI reads and follows (mostly).

**Level 2 (Nudge):** Include conditional:
```markdown
If you didn't run the health check, stop now and do it. This is not optional.
```

**Level 3 (Hard):** MCP server returns "health_check_required" until check is done.

## Platform-Specific Notes

### Clawdbot
- Add to `AGENTS.md` (or workspace's AGENTS.md)
- Can also integrate with heartbeat for periodic checks
- Supports webhook push for alerts between sessions

### Claude Code
- Add to `CLAUDE.md` in project root
- No push mechanism — must be pull/check based
- Consider adding to global `~/.claude/CLAUDE.md` for all projects

### Claude Cowork
- Same as Claude Code
- File drop to workspace can deliver alerts
- Add workspace folder to Cowork with CLAUDE.md

### IDE Extensions (Cursor, etc.)
- Check if they support CLAUDE.md or equivalent
- May need custom rules file
- MCP integration is the most reliable path

## Measuring Success

Goal: 100% of reasoning periods start with health check.

Track via:
- `health_check_events` table in Kernle
- Session start time vs first health_check call
- Per-platform compliance rates

If compliance drops below 90%, escalate enforcement level.
