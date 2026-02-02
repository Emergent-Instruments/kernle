# Kernle

**Stratified memory for synthetic intelligences.**

Kernle gives AI agents persistent memory, emotional awareness, and identity continuity. It's the cognitive infrastructure for agents that grow, adapt, and remember who they are.

📚 **Full Documentation: [docs.kernle.ai](https://docs.kernle.ai)**

---

## Quick Start

```bash
# Install
pip install kernle

# Initialize your agent
kernle -a my-agent init

# Load memory at session start
kernle -a my-agent load

# Check health
kernle -a my-agent anxiety -b

# Capture experiences
kernle -a my-agent episode "Deployed v2" "success" --lesson "Always run migrations first"
kernle -a my-agent raw "Quick thought to process later"

# Save before ending
kernle -a my-agent checkpoint save "End of session"
```

## Automatic Memory Loading

**Make memory loading automatic** instead of relying on manual commands:

```bash
# Clawdbot - Install hook for automatic loading
kernle -a my-agent setup clawdbot

# Claude Code - Install SessionStart hook (project-level)
kernle -a my-agent setup claude-code

# Claude Code - Install globally for all projects
kernle -a my-agent setup claude-code --global

# Cowork - Same as Claude Code
kernle -a my-agent setup cowork
```

After setup, memory loads automatically at every session start. No more forgetting to run `kernle load`!

See `kernle/hooks/` directory for manual installation or `kernle setup --help` for details.

## Other Integrations

**Manual CLAUDE.md setup:**
```bash
kernle -a my-agent init  # Generates CLAUDE.md section with manual load instructions
```

**MCP Server:**
```bash
claude mcp add kernle -- kernle mcp -a my-agent
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

## Status

- **Tests:** 1292 passing
- **Coverage:** 57%
- **Backend:** Railway + Supabase
- **Docs:** Mintlify

See [ROADMAP.md](ROADMAP.md) for development plans.

## License

MIT
