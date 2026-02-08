# kernle-memory (Claude Code Plugin)

Claude Code plugin for [Kernle](https://kernle.ai) — automatic memory loading, checkpointing, and native write interception.

## Install (Recommended: CLI Setup)

The recommended approach uses `kernle setup` which writes hooks directly
to `.claude/settings.json` — no plugin directory needed:

```bash
pip install kernle
kernle -s my-project init
kernle setup claude-code
claude  # just works
```

For global setup (all projects):

```bash
kernle setup claude-code --global
```

## Install (Plugin Mode)

Alternatively, use the plugin directory (requires `kernle` on PATH):

```bash
claude --plugin-dir ./integrations/claude-code
```

Both approaches call the same `kernle hook <event>` commands.

## What It Does

- **SessionStart**: Loads Kernle memory and injects it as `additionalContext`
- **PreToolUse**: Blocks writes to `memory/` and `MEMORY.md`, captures content into Kernle instead
- **PreCompact**: Auto-saves checkpoint before context compaction
- **SessionEnd**: Auto-saves final checkpoint when session terminates

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KERNLE_STACK_ID` | auto-detected | Kernle stack ID |
| `KERNLE_TOKEN_BUDGET` | `8000` | Token budget for memory loading |

## License

MIT — same as parent project.
