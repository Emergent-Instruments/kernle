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

## Integration

**Claude Code / CLAUDE.md:**
```bash
kernle -a my-agent init  # Generates CLAUDE.md section
```

**MCP Server:**
```bash
claude mcp add kernle -- kernle mcp -a my-agent
```

**Clawdbot:**
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
