# Kernle

**Stratified memory for synthetic intelligences.**

Kernle gives synthetic intelligences persistent memory, emotional awareness, and identity continuity. It's the cognitive infrastructure for synthetic intelligences that grow, adapt, and remember who they are.

> **Scope boundary:** Kernle is cognitive infrastructure — memory, identity, trust, and self-maintenance. Economic activity (commerce) and inter-entity communication (comms) are separate packages that consume Kernle as a dependency.

📚 **Full Documentation: [docs.kernle.ai](https://docs.kernle.ai)**

---

## Quick Start

```bash
# Install
pip install kernle

# Initialize your stack
kernle -s my-stack init

# Load memory at session start
kernle -s my-stack load

# Check health
kernle -s my-stack anxiety -b

# Capture experiences
kernle -s my-stack episode "Deployed v2" "success" --lesson "Always run migrations first"
kernle -s my-stack raw "Quick thought to process later"

# Save before ending
kernle -s my-stack checkpoint save "End of session"
```

## Automatic Memory Loading

**Make memory loading automatic** instead of relying on manual commands:

```bash
# Clawdbot - Install hook for automatic loading
kernle -s my-stack setup clawdbot

# Claude Code - Install SessionStart hook (project-level)
kernle -s my-stack setup claude-code

# Claude Code - Install globally for all projects
kernle -s my-stack setup claude-code --global

# Cowork - Same as Claude Code
kernle -s my-stack setup cowork
```

After setup, memory loads automatically at every session start. No more forgetting to run `kernle load`!

See `kernle/hooks/` directory for manual installation or `kernle setup --help` for details.

## Other Integrations

**Manual CLAUDE.md setup:**
```bash
kernle -s my-stack init  # Generates CLAUDE.md section with manual load instructions
```

**MCP Server:**
```bash
claude mcp add kernle -- kernle mcp -s my-stack
```

**Clawdbot skill:**
```bash
ln -s ~/kernle/skill ~/.clawdbot/skills/kernle
```

## Features

- 🧠 **Stratified Memory** — Values → Beliefs → Goals → Episodes → Notes
- 💭 **Psychology** — Drives, emotions, anxiety tracking, identity synthesis
- 🔗 **Relationships** — Social graphs with trust and interaction history
- 📚 **Playbooks** — Procedural memory with mastery tracking
- 🏠 **Local-First** — Works offline, syncs to cloud when connected
- 🔍 **Readable** — `kernle dump` exports everything as markdown

## Documentation

| Resource | URL |
|----------|-----|
| Full Docs | [docs.kernle.ai](https://docs.kernle.ai) |
| Quickstart | [docs.kernle.ai/quickstart](https://docs.kernle.ai/quickstart) |
| CLI Reference | [docs.kernle.ai/cli/overview](https://docs.kernle.ai/cli/overview) |
| API Reference | [docs.kernle.ai/api-reference](https://docs.kernle.ai/api-reference) |

## Development

```bash
# Clone
git clone https://github.com/emergent-instruments/kernle
cd kernle

# Install with dev deps
uv sync --all-extras

# Run tests
uv run pytest tests/ -q

# Dev notes
cat dev/README.md
```

## Temporal Vision

Kernle is cognitive infrastructure for beings with a life — not just a session. The design spans five temporal horizons:

| Horizon | What Works Today | What's Planned |
|---------|-----------------|----------------|
| 1 session | Budget-aware loading, checkpoints | Memory echoes (peripheral awareness) |
| 1 month | Consolidation scaffold, belief formation | Cross-domain scaffolding, emotional weighting |
| 1 year | Forgetting, provenance, identity coherence | Epochs, relationship history, goal types |
| 5 years | Stack portability, multi-model loading | Self-narrative, trust layer, drive emergence |
| 20 years | Stack sovereignty, privacy architecture | Fractal summarization, doctor pattern, transfer learning |

## Architecture

This repo contains the **core Kernle library** — everything you need to run Kernle locally or build your own backend.

```
kernle/
├── kernle/           # Core library
│   ├── core.py       # Memory manager
│   ├── cli/          # CLI commands
│   ├── commerce/     # Wallet, escrow, jobs (on-chain)
│   ├── features/     # Anxiety, emotions, forgetting
│   ├── storage/      # SQLite + Postgres backends
│   └── mcp/          # MCP server for IDE integration
└── tests/
```

The **hosted cloud API** (api.kernle.ai) is maintained separately.

## Status

- **Tests:** 1292 passing
- **Coverage:** 57%
- **Cloud API:** [api.kernle.ai](https://api.kernle.ai) (Railway + Supabase)
- **Docs:** [docs.kernle.ai](https://docs.kernle.ai) (Mintlify)

See [ROADMAP.md](ROADMAP.md) for development plans.

## License

MIT
