# Kernle Load Hook

Automatically loads Kernle persistent memory into every session.

## Problem Statement

Synthetic intelligences need persistent memory across sessions, but requiring them to manually run `kernle load` is unreliable:
- SIs forget to run the command
- Instructions in AGENT.md can be skipped
- Inconsistent behavior across sessions
- Memory loss at checkpoints

## Solution

This hook **automatically** injects Kernle memory into the session's system prompt, making it as natural as native memory.

## How It Works

1. **Session starts** → `agent:bootstrap` event fires
2. **Hook executes** → Runs `kernle -s {stackId} load`
3. **Output injected** → Creates virtual `KERNLE.md` in `bootstrapFiles`
4. **System prompt** → Memory context automatically included
5. **SI sees memory** → Values, beliefs, goals, episodes all available

## Stack ID Detection

The hook intelligently detects the stack ID:

```
Session Key: "agent:ash:main"    → Stack ID: "ash"
Session Key: "agent:bob:work"   → Stack ID: "bob"
Workspace: /Users/ash/project   → Stack ID: "project"
Fallback: No detection          → Stack ID: "main"
```

## Installation

### 1. Enable the Hook

Edit `~/.clawdbot/clawdbot.json`:

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

### 2. Verify Kernle is Installed

```bash
kernle --version
```

### 3. Initialize Kernle for Your Stack

```bash
cd ~/workspace
kernle -s yourname init
```

### 4. Test the Hook

Start a new session and check if `KERNLE.md` is in context:

```bash
# Start session
clawdbot

# In the session, ask:
> Do you have memory context from Kernle?
```

## Configuration Options

### Disable for Specific Stacks

```json
{
  "agents": {
    "defaults": {
      "hooks": {
        "kernle-load": {
          "enabled": false
        }
      }
    }
  }
}
```

### Adjust Timeout

Modify `handler.ts`:

```typescript
const { stdout } = await execAsync(`kernle -s ${stackId} load`, {
  timeout: 10000, // 10 seconds instead of 5
});
```

## Graceful Degradation

The hook fails silently if:
- Kernle is not installed
- No stack exists with the detected ID
- `kernle load` times out (>5 seconds)
- Command returns an error

In all cases, the session continues normally without Kernle memory.

## Performance

- **Execution time**: ~100-500ms (depends on memory size)
- **Memory overhead**: Minimal (only loads once per session)
- **Network**: None (local-first operation)

## Debugging

### Check if Hook is Enabled

```bash
cat ~/.clawdbot/clawdbot.json | grep -A 5 kernle-load
```

### Test Kernle Load Manually

```bash
kernle -s my-project load
```

### View Hook Logs

Hook errors are logged to stderr but don't block sessions.

## Comparison to Manual Loading

| Approach | Consistency | Setup | Maintenance |
|----------|-------------|-------|-------------|
| **Manual** (`kernle load` in AGENTS.md) | Unreliable | Simple | High (SI must follow instructions) |
| **Hook** (this implementation) | ✅ 100% consistent | One-time | None (automatic) |

## Integration with Other Systems

### Claude Code

For Claude Code sessions, use a `SessionStart` hook instead:

`.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kernle -s my-project load 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

### Cowork

Same as Claude Code (uses identical settings format).

### MCP Server

The Kernle MCP server provides manual `memory_load()` tool but doesn't auto-inject. This hook complements MCP by ensuring memory is always present.

## Future Enhancements

- [ ] Cache memory output for multiple rapid sessions
- [ ] Support custom memory budget via config
- [ ] Inject condensed summary for token-limited sessions
- [ ] Integrate with compaction to refresh memory mid-session
- [ ] Add `kernle-refresh` command to reload memory without restart

## License

Same as parent project.
