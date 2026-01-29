# Kernle Development Notes

Internal documentation for contributors and development reference.

**For user-facing documentation, see: https://docs.kernle.ai**

---

## Structure

```
dev/
├── audits/          # Historical audit reports
├── design/          # Design documents for new features
├── features/        # Feature implementation notes
├── ARCHITECTURE.md  # System architecture reference
├── CLI.md           # CLI implementation details
├── MEMORY_MODEL.md  # Memory layer design
├── SCHEMA.md        # Database schema reference
└── ...              # Other dev references
```

## Key Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | System design, component relationships |
| `CLI.md` | CLI command implementation reference |
| `MEMORY_MODEL.md` | Memory layers and consolidation design |
| `SCHEMA.md` | SQLite/Postgres schema details |
| `SETUP.md` | Development environment setup |
| `PYTHON_API.md` | Python API internals |

## Audits

The `audits/` folder contains historical code reviews:
- Architecture audits
- Security reviews
- Test coverage analysis
- MCP server audit

These are kept for reference but may be outdated.

## Adding Documentation

- **User-facing**: Add to `docs-site/` (Mintlify)
- **Dev reference**: Add here in `dev/`
- **Design proposals**: Add to `dev/design/`
