# kernle-memory (Claude Code Plugin)

Claude Code plugin for [Kernle](https://kernle.ai) — automatic memory loading, checkpointing, and native write interception.

## What It Does

- **SessionStart**: Loads Kernle memory and injects it as `additionalContext` at session start (and after compaction)
- **PreToolUse**: Blocks writes to `memory/` and `MEMORY.md`, captures content into Kernle instead
- **PreCompact**: Auto-saves checkpoint before context compaction
- **SessionEnd**: Auto-saves final checkpoint when session terminates

## Install

```bash
# Prerequisites
pip install kernle
kernle -s my-project init

# Use the plugin
claude --plugin-dir ./integrations/claude-code
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `KERNLE_STACK_ID` | auto-detected | Kernle stack ID |
| `KERNLE_BIN` | `kernle` | Path to kernle binary |
| `KERNLE_TIMEOUT` | `5` | CLI timeout in seconds |
| `KERNLE_TOKEN_BUDGET` | `8000` | Token budget for memory loading |

### Stack ID Resolution

1. `KERNLE_STACK_ID` environment variable
2. Project directory name from working directory
3. Kernle auto-resolve (hostname + project path hash)

## Development

```bash
cd integrations/claude-code
python -m pytest tests/ -q
```

No build step required — pure Python, stdlib only.

## License

MIT — same as parent project.
