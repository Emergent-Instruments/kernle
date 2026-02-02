# Kernle × OpenClaw: Integration Lifecycle

How Kernle integrates with [OpenClaw](https://openclaw.ai) — from message arrival to memory persistence.

## Current Architecture

```
USER SENDS MESSAGE
       ↓
1. Channel receives (iMessage, Telegram, Discord, etc.)
2. OpenClaw gateway routes message to session
3. Gateway loads workspace context:
   - AGENTS.md  (agent instructions)
   - SOUL.md    (persona & tone)
   - USER.md    (about the human)
   - TOOLS.md   (local config — IPs, chat IDs, credentials)
   - MEMORY.md  (bootstrap memory cache, if exists)
   - HEARTBEAT.md (periodic task checklist)
4. Gateway builds system prompt with all injected context
5. Gateway sends to LLM API (Claude, etc.)
       ↓
   AGENT RECEIVES FULL CONTEXT
   (TOOLS.md config already visible — chat IDs, IPs, etc.)
       ↓
6. Agent reads AGENTS.md, sees instruction: "run kernle load"
7. Agent calls: kernle -a {agent} load
8. Kernle returns: beliefs, episodes, goals, values, checkpoint
9. Agent now has full memory context
10. Agent processes user request
11. Agent responds
12. Agent captures new memories:
    - kernle -a {agent} episode "..."
    - kernle -a {agent} raw "..."
    - kernle -a {agent} believe "..."
13. On context pressure (compaction): agent saves checkpoint
    - kernle -a {agent} checkpoint "task summary"
```

### The Timing Gap

Steps 1–5 happen **before any agent code runs**. `TOOLS.md` is injected at step 3. Kernle isn't touched until step 7. That's a **6-step gap** where config is needed but Kernle can't provide it.

```
CURRENT TIMELINE:
├─ Gateway startup
├─ Workspace files loaded (TOOLS.md)  ← CONFIG AVAILABLE HERE
├─ Context injection
├─ LLM invoked
├─ Agent code runs
├─ Agent decides to call kernle load   ← KERNLE AVAILABLE HERE
└─ Full memory context

Gap = 4+ steps where config is needed but Kernle isn't loaded yet
```

Today this gap is bridged by `TOOLS.md` — a manually maintained markdown file containing environment-specific config (camera names, SSH hosts, voice preferences, chat IDs). It works, but it means config lives outside Kernle in a platform-specific mechanism.

## With Boot Layer

The boot layer eliminates the timing gap by making config available as a flat file **before** Kernle is invoked:

```
REVISED TIMELINE:
├─ Gateway startup
├─ Read ~/.kernle/{agent}/boot.md      ← BOOT CONFIG AVAILABLE
├─ Workspace files loaded (MEMORY.md includes boot config)
├─ Context injection (boot included)   ← NO GAP
├─ LLM invoked
├─ Agent code runs
├─ kernle load (boot also included in response)
└─ Full memory context

Gap = 0. Kernle serves config from the first moment.
```

### How It Works

1. **Boot config stored in Kernle** — simple key/value pairs in SQLite (`boot` table)
2. **Auto-projected to file** — `~/.kernle/{agent}/boot.md` is always up to date
3. **Included in `export-cache`** — `MEMORY.md` bootstrap cache contains boot config at the top
4. **Included in `load` response** — redundant but consistent; agent always has boot config regardless of entry point

### CLI Interface

```bash
# Set config
kernle -a ash boot set chat_id 4
kernle -a ash boot set gateway_ip 192.168.50.11

# Read config (instant — no full memory load)
kernle -a ash boot get chat_id
# → 4

# List all boot config
kernle -a ash boot list
# → chat_id: 4
# → gateway_ip: 192.168.50.11

# Remove config
kernle -a ash boot delete chat_id
```

### Migration Path

For existing OpenClaw agents using `TOOLS.md`:

1. Move environment-specific config into boot layer: `kernle boot set key value`
2. Run `kernle export-cache` — boot config appears in `MEMORY.md` header
3. `TOOLS.md` becomes optional (still available for non-Kernle notes)
4. Boot file at `~/.kernle/{agent}/boot.md` serves as the pre-injection config source

## Memory Lifecycle

### Session Start
1. `MEMORY.md` (auto-generated cache) is injected as workspace context
2. Agent runs `kernle -a {agent} load` to restore full memory state
3. Agent is now fully context-aware

### During Work
- New memories captured via CLI: `episode`, `raw`, `believe`, `note`
- Consolidation promotes raw entries → episodes → beliefs
- `kernle anxiety` monitors memory health metrics

### Session End / Compaction
1. `kernle checkpoint "task summary"` saves working state
2. `kernle export-cache --output MEMORY.md` regenerates bootstrap cache
3. Next session gets fresh `MEMORY.md` with latest state

### Periodic Maintenance
- `kernle consolidate` — promotes raw entries through the pipeline
- `kernle anxiety` — checks for rising metrics or belief conflicts
- Heartbeat tasks can trigger these automatically

## Key Design Principles

- **Kernle is the single source of truth** for agent memory
- **`MEMORY.md` is a derived cache**, never manually edited
- **Boot config is always available** — file-based, no full load required
- **Platform-agnostic** — same Kernle commands work regardless of host platform
- **Local-first** — SQLite database, no network dependency for core operations
