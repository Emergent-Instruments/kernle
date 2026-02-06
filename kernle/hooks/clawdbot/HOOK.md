# 🧠 kernle-load

Automatically loads Kernle memory into session context at startup.

## What it does

Executes `kernle load` at session start and injects the output into the SI's system prompt. This ensures memory (values, beliefs, goals, episodes, relationships) is always available without requiring the SI to manually run the command.

## When it runs

- **Event**: `agent:bootstrap`
- **Frequency**: Every new session
- **Timing**: Before the first user message is processed

## Configuration

```yaml
metadata:
  moltbot:
    events: ["agent:bootstrap"]
    requires:
      config:
        - agents.defaults.workspace
```

Enable/disable in `~/.clawdbot/clawdbot.json`:

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
  }
}
```

## Features

- **Zero AI involvement**: Memory loads automatically
- **Graceful degradation**: Session continues if kernle is unavailable
- **Configurable**: Can be disabled per-stack or globally
- **Efficient**: Only loads once per session

## Stack ID Detection

The hook detects the stack ID from:
1. Session key (e.g., `agent:claire:main` → `claire`)
2. Workspace directory name as fallback
3. Default to `"main"` if unable to detect

## Output Format

Injects a virtual `KERNLE.md` file containing:
- Working memory state (current task, context)
- Values and beliefs
- Active goals
- Recent episodes with lessons
- Drives and relationships
- Key patterns from consolidation
