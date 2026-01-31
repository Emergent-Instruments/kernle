# Kernle Load Hook for Clawdbot

Automatically loads Kernle persistent memory into every Clawdbot session.

## Quick Start

```bash
# Install the hook
kernle setup clawdbot

# Enable in config
# (kernle setup will prompt you to do this)
```

## Manual Installation

If `kernle setup clawdbot` doesn't work, install manually:

### 1. Copy Hook Files

```bash
# For bundled hooks (requires moltbot repo access)
cp -r hooks/clawdbot/ /path/to/moltbot/src/hooks/bundled/kernle-load/

# For user hooks (recommended)
mkdir -p ~/.config/moltbot/hooks/kernle-load/
cp hooks/clawdbot/* ~/.config/moltbot/hooks/kernle-load/
```

### 2. Enable in Config

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

### 3. Restart Clawdbot

```bash
# Restart the gateway
clawdbot doctor --restart
# or
pkill -f clawdbot && clawdbot
```

## How It Works

1. **Session starts** → `agent:bootstrap` event fires
2. **Hook executes** → Runs `kernle -a {agentId} load`
3. **Output injected** → Creates virtual `KERNLE.md` in bootstrap files
4. **System prompt** → Memory context automatically included
5. **AI sees memory** → Values, beliefs, goals, episodes available

## Agent ID Detection

The hook auto-detects agent ID from:
- Session key: `agent:claire:main` → `claire`
- Workspace dir: `/Users/claire/workspace` → `workspace`
- Fallback: `main`

## Verification

Start a new Clawdbot session and ask:

```
> Do you see KERNLE.md in your context files?
> What are my current values and goals?
```

The AI should respond with your Kernle memory without having to run any commands.

## Troubleshooting

### Hook not running

1. Check hook is enabled:
   ```bash
   cat ~/.clawdbot/clawdbot.json | grep -A 5 kernle-load
   ```

2. Check hook files exist:
   ```bash
   ls ~/.config/moltbot/hooks/kernle-load/
   ```

3. Restart gateway

### Kernle command not found

```bash
# Install kernle
pip install kernle
# or
pipx install kernle

# Verify
kernle --version
```

### Memory not loading

1. Test manually:
   ```bash
   kernle -a yourname load
   ```

2. Check if agent exists:
   ```bash
   kernle -a yourname status
   ```

3. Initialize if needed:
   ```bash
   kernle -a yourname init
   ```

## Performance

- **Execution time**: ~100-500ms per session start
- **Memory overhead**: Minimal (only loads once)
- **Network**: None (local-first)

## See Also

- [Kernle Documentation](../../README.md)
- [Claude Code Hook](../claude-code/README.md)
- [Clawdbot Hooks](https://github.com/anthropics/moltbot/docs/hooks.md)
