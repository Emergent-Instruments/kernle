# Kernle Load Hook for Claude Code / Cowork

Automatically loads Kernle persistent memory into every Claude Code or Cowork session.

## Quick Start

```bash
# Install the hook
kernle setup claude-code

# Or manually:
cd ~/your-project
cp $(kernle setup claude-code --print-path)/settings.json .claude/settings.json
# Edit .claude/settings.json to set YOUR_STACK_NAME
```

## Manual Installation

### Option 1: Project-Level (Recommended)

For a specific project:

```bash
# Create .claude directory
mkdir -p .claude

# Copy template
cp hooks/claude-code/settings.json .claude/settings.json

# Edit and replace YOUR_STACK_NAME with your stack ID
vim .claude/settings.json
```

Example:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kernle -s claire load"
          }
        ]
      }
    ]
  }
}
```

### Option 2: User-Level (Global)

For all Claude Code sessions:

```bash
# Copy to user settings
mkdir -p ~/.claude
cp hooks/claude-code/settings.json ~/.claude/settings.json

# Edit and replace YOUR_STACK_NAME
vim ~/.claude/settings.json
```

### Option 3: Dynamic Stack Detection

Use environment variables:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kernle -s ${USER} load"
          }
        ]
      }
    ]
  }
}
```

## How It Works

1. **Session starts** → `SessionStart` hook fires
2. **Command executes** → Runs `kernle -s {stackId} load`
3. **Stdout captured** → Output automatically added to context
4. **Memory available** → SI sees values, beliefs, goals, episodes

## Alternative: CLAUDE.md Injection

Instead of hooks, use command injection in `CLAUDE.md`:

```markdown
# Kernle Memory

!`kernle -s claire load`
```

The `!`command`` syntax executes at file load time.

## Verification

Start a new Claude Code session:

```bash
cd ~/your-project
claude
```

Ask:
```
> What are my current values and goals from Kernle?
```

The SI should respond with your memory without running commands.

## Troubleshooting

### Hook not running

1. Check settings file exists:
   ```bash
   cat .claude/settings.json
   # or
   cat ~/.claude/settings.json
   ```

2. Verify hook syntax is valid JSON:
   ```bash
   jq . .claude/settings.json
   ```

3. Check Claude Code version supports hooks:
   ```bash
   claude --version  # Should be >= 2.0
   ```

### Kernle command not found

```bash
pip install kernle
# or
pipx install kernle
```

### Wrong stack ID

Update `settings.json` with correct stack name:

```json
{
  "command": "kernle -s YOUR_NAME load"
}
```

### Memory not appearing

1. Test manually:
   ```bash
   kernle -s yourname load
   ```

2. Check if stack initialized:
   ```bash
   kernle -s yourname status
   ```

3. Initialize if needed:
   ```bash
   kernle -s yourname init
   ```

## Performance

- **Execution time**: ~100-500ms per session
- **Runs**: Once at session start
- **Network**: None (local-first)

## Cowork

Cowork uses the same `.claude/settings.json` format. Follow the same instructions above.

## See Also

- [Kernle Documentation](../../README.md)
- [Clawdbot Hook](../clawdbot/README.md)
- [Claude Code Hooks](https://code.claude.com/docs/en/hooks.md)
