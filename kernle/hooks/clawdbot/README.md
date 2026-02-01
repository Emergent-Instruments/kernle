# Kernle Load Hook

Automatically loads Kernle persistent memory into every agent session.

## Problem Statement

AI agents need persistent memory across sessions, but requiring them to manually run `kernle load` is unreliable:
- AIs forget to run the command
- Instructions in AGENT.md can be skipped
- Inconsistent behavior across sessions
- Memory loss at checkpoints

## Solution

This hook **automatically** injects Kernle memory into the session's system prompt, making it as natural as native memory.

## How It Works

1. **Session starts** → `agent:bootstrap` event fires
2. **Hook executes** → Runs `kernle -a {agentId} load`
3. **Output injected** → Creates virtual `KERNLE.md` in `bootstrapFiles`
4. **System prompt** → Memory context automatically included
5. **AI sees memory** → Values, beliefs, goals, episodes all available

## Agent ID Detection

The hook intelligently detects the agent ID:

```
Session Key: "agent:claire:main" → Agent ID: "claire"
Session Key: "agent:bob:work"   → Agent ID: "bob"
Workspace: /Users/claire/clawd  → Agent ID: "clawd"
Fallback: No detection          → Agent ID: "main"
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

### 3. Initialize Kernle for Your Agent

```bash
cd ~/workspace
kernle -a yourname init
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

### Disable for Specific Agents

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
const { stdout } = await execAsync(`kernle -a ${agentId} load`, {
  timeout: 10000, // 10 seconds instead of 5
});
```

## Graceful Degradation

The hook fails silently if:
- Kernle is not installed
- No agent exists with the detected ID
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
kernle -a claire load
```

### View Hook Logs

Hook errors are logged to stderr but don't block sessions.

## Comparison to Manual Loading

| Approach | Consistency | Setup | Maintenance |
|----------|-------------|-------|-------------|
| **Manual** (`kernle load` in AGENTS.md) | ❌ Unreliable | Simple | High (AI must follow instructions) |
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
            "command": "kernle -a claire load"
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

Same as parent project (moltbot/clawdbot).
