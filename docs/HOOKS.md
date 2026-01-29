# Hook-Based Auto-Capture

Kernle supports automatic memory capture through hook integrations with AI coding assistants. This enables hands-free memory capture at natural breakpoints like session end, after significant tool uses, or at regular intervals.

## Overview

Instead of manually calling `kernle episode` or `kernle note` after every session, you can configure hooks to automatically capture context to Kernle's raw memory layer. These captures can later be reviewed and promoted to structured memory types.

## Supported Environments

| Environment | Hook Mechanism | Status |
|-------------|---------------|--------|
| Claude Code | hooks.json | Supported |
| Cursor | Custom rules | Manual |
| Cline | Custom rules | Manual |

## Claude Code Integration

### Setup

1. Create or edit `~/.claude/hooks.json`:

```json
{
  "hooks": [
    {
      "event": "Stop",
      "command": "kernle -a $KERNLE_AGENT_ID raw capture --stdin --source hook-session-end --quiet",
      "description": "Auto-capture session context to Kernle on stop"
    }
  ]
}
```

2. Set the `KERNLE_AGENT_ID` environment variable in your shell profile:

```bash
# ~/.bashrc or ~/.zshrc
export KERNLE_AGENT_ID="my-agent"
```

Or configure it in Claude Code's settings.

### Available Events

| Event | Description | Use Case |
|-------|-------------|----------|
| `Stop` | Triggered when session ends | Capture session summary |
| `PostToolUse` | After any tool is executed | Capture significant actions |

### Example Configurations

#### Session-End Capture

Capture a summary when each Claude Code session ends:

```json
{
  "hooks": [
    {
      "event": "Stop",
      "command": "echo \"Session ended. Last task: $(cat ~/.claude/last_task 2>/dev/null || echo 'unknown')\" | kernle -a $KERNLE_AGENT_ID raw capture --stdin --source hook-session-end --quiet",
      "description": "Capture session summary"
    }
  ]
}
```

#### Periodic Heartbeat

Capture context at regular intervals during long sessions:

```json
{
  "hooks": [
    {
      "event": "PostToolUse",
      "command": "kernle -a $KERNLE_AGENT_ID raw capture \"Tool execution completed\" --source hook-heartbeat --quiet",
      "filter": {
        "tool": ["bash", "write", "edit"]
      }
    }
  ]
}
```

## CLI Flags for Hooks

The `kernle raw capture` command includes flags designed for hook usage:

```bash
# Quiet mode - minimal output, just prints capture ID
kernle raw capture "content" --quiet

# Read from stdin - for piped content
echo "session summary" | kernle raw capture --stdin --source hook-session-end

# Source tracking - identify where captures came from
kernle raw capture "content" --source hook-post-tool

# Combined for hooks
echo "content" | kernle raw capture --stdin --source hook-session-end --quiet
```

### Flag Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--quiet` | `-q` | Suppress output except capture ID |
| `--stdin` | - | Read content from stdin instead of argument |
| `--source` | `-s` | Source identifier for tracking (e.g., 'hook-session-end') |
| `--tags` | `-t` | Comma-separated tags |

## MCP Integration

The `memory_auto_capture` MCP tool also supports hook-based capture:

```json
{
  "name": "memory_auto_capture",
  "arguments": {
    "text": "Session completed: implemented user authentication",
    "source": "hook-session-end",
    "extract_suggestions": true
  }
}
```

With `extract_suggestions: true`, the tool analyzes the text and returns promotion suggestions:

```json
{
  "captured": true,
  "id": "abc12345",
  "source": "hook-session-end",
  "suggestions": ["episode"],
  "promote_command": "kernle raw promote abc12345 --type episode"
}
```

## Processing Auto-Captures

After hooks capture content, review and promote meaningful entries:

```bash
# List unprocessed captures
kernle raw list --unprocessed

# Review with suggestions
kernle raw review

# Guided triage
kernle raw triage

# Promote to structured memory
kernle raw promote abc12345 --type episode
kernle raw promote def67890 --type note

# Clean up junk captures
kernle raw clean --junk --confirm
```

## Best Practices

1. **Use descriptive sources**: Name sources like `hook-session-end`, `hook-post-commit`, `hook-heartbeat` to track where captures originate.

2. **Review regularly**: Auto-captures accumulate in the raw layer. Schedule regular review sessions to promote meaningful content.

3. **Clean up junk**: Use `kernle raw clean --junk` to remove test entries and noise.

4. **Combine with manual capture**: Hooks catch routine events; still use manual `kernle episode` for significant learnings.

5. **Test hooks quietly**: Use `--quiet` mode when testing to avoid noisy output.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KERNLE_AGENT_ID` | Agent identifier | `default` or auto-generated |
| `KERNLE_AUTO_SYNC` | Enable/disable auto-sync | `true` if cloud configured |

## Troubleshooting

### Hook not triggering

1. Check hooks.json syntax is valid JSON
2. Verify KERNLE_AGENT_ID is set
3. Test the command manually in terminal

### Captures not appearing

```bash
# Check if captures exist
kernle raw list --limit 5

# Check source filtering
kernle raw list | grep "hook-"
```

### Too many junk captures

```bash
# See what would be cleaned
kernle raw clean --junk

# Actually clean
kernle raw clean --junk --confirm
```
