# Kernle

**Stratified memory for synthetic intelligences.**

Kernle gives synthetic intelligences persistent memory, emotional awareness, and identity continuity. It's the cognitive infrastructure for synthetic intelligences that grow, adapt, and remember who they are.


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
# Claude Code - Setup hooks for automatic loading + checkpointing + write interception
kernle setup claude-code

# OpenClaw - Install plugin for automatic loading + checkpointing
cd integrations/openclaw && npm install && npm run build
openclaw plugins install ./integrations/openclaw
```

After setup, memory loads automatically at every session start. No more forgetting to run `kernle load`!

## Other Integrations

**Manual CLAUDE.md setup:**
```bash
kernle -s my-stack init  # Generates CLAUDE.md section with manual load instructions
```

**MCP Server:**
```bash
claude mcp add kernle -- kernle mcp -s my-stack
```

**OpenClaw skill:**
```bash
ln -s ~/kernle/skill ~/.openclaw/skills/kernle
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

Coverage policy is configured in `pyproject.toml` and enforced in both local `make test-cov` and CI coverage runs.

### Audit tracking submodule (private)

The full audit corpus is kept in a private submodule: `audits/` (repo `emergent-instruments/kernle-audits`).

```bash
# Initialize / refresh submodules on a fresh clone
git submodule update --init --recursive

# Update local submodule pointer to latest commit in audits/main
git submodule update --remote audits
```

To edit audits:

```bash
# Work inside the private repo
cd audits
git pull
git status

# Make edits, commit, and push in the private repo
git add .
git commit -m "Update audit pass findings"
git push

# Return to main repo and record the new pointer
cd ..
git add audits
git commit -m "chore: bump audits submodule pointer"
```

CI checks out submodules during test/release jobs. If private access fails in CI, set `SUBMODULE_PAT` in repository secrets with read access to `kernle-audits`.

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

│   ├── features/     # Anxiety, emotions, forgetting
│   ├── storage/      # SQLite-first storage + stack interfaces
│   └── mcp/          # MCP server for IDE integration
└── tests/
```

The **hosted cloud API** (api.kernle.ai) is maintained separately.

## Status

- **Coverage Gate:** branch coverage must remain at or above `85%`
- **Ratcheting Policy:** coverage floor only moves upward; any decrease requires explicit maintainer approval
- **Docs:** [docs.kernle.ai](https://docs.kernle.ai) (Mintlify)

See [ROADMAP.md](ROADMAP.md) for development plans.

## License

MIT
