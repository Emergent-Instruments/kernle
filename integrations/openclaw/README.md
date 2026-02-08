# kernle-openclaw

OpenClaw Plugin SDK plugin for [Kernle](https://kernle.ai) — automatic memory loading, checkpointing, and native write interception.

## What It Does

- **`before_agent_start`**: Loads Kernle memory and injects it as `prependContext` at session start
- **`agent_end`**: Auto-saves a checkpoint from conversation context at session end
- **`before_tool_call`**: Blocks writes to `memory/` and `MEMORY.md`, captures content into Kernle instead
- **`tool_result_persist`**: Trims large kernle output in the transcript

## Install

```bash
# Prerequisites
pip install kernle
kernle -s my-project init

# Build and install
cd integrations/openclaw
npm install && npm run build
openclaw plugins install .
```

## Configuration

Add to your OpenClaw config (optional — stack ID is auto-detected from session key):

```json
{
  "plugins": {
    "entries": {
      "kernle": {
        "enabled": true,
        "config": {
          "stackId": "my-project",
          "tokenBudget": 8000,
          "timeout": 5000,
          "kernleBin": "kernle"
        }
      }
    }
  }
}
```

### Stack ID Resolution

1. Explicit `stackId` in config
2. `KERNLE_STACK_ID` environment variable
3. Agent ID from session key (e.g., `agent:ash:main` → `ash`)
4. Workspace directory name
5. Fallback: `"main"`

## Development

```bash
npm install
npm test          # Run tests
npm run build     # Compile TypeScript
```

## License

MIT — same as parent project.
