# Migration Guide: TOOLS.md → Boot Layer

This guide helps you migrate from platform-specific config files (like OpenClaw's TOOLS.md) to Kernle's boot layer.

## Why Migrate?

| TOOLS.md | Boot Layer |
|----------|------------|
| Platform-specific (OpenClaw only) | Platform-agnostic (works everywhere) |
| Manual file editing | CLI/API management |
| Static file | Queryable database |
| No integration with memory | Included in `load`, `export-cache` |
| Doesn't travel with SI | Portable across environments |

## Before You Start

1. Identify what's in your TOOLS.md
2. Decide what belongs in boot vs other Kernle layers
3. Back up your TOOLS.md

## Classification Guide

Review each item in TOOLS.md and classify it:

### → Boot Layer (operational config)

```
Chat IDs, group IDs
Gateway IPs, ports, tokens
Platform settings
Debug flags
Integration endpoints
```

### → Values (persona/preferences)

```
Communication style preferences
Privacy boundaries
Interaction patterns
```

### → Relationships (people info)

```
Human's name, contact info
Known entities and their details
```

### → Notes (factual observations)

```
"Sean prefers X"
"This approach works for Y"
```

### → Keep in TOOLS.md (truly platform-specific)

```
Hardware setup details (HDMI dongles, etc.)
Development workflow notes
Platform-specific quirks
```

## Step-by-Step Migration

### 1. Export Current TOOLS.md Content

```bash
cat ~/workspace/TOOLS.md
```

### 2. Migrate Config Items to Boot

For each operational config item:

```bash
# Chat/messaging config
kernle -a claire boot set imsg_group_chat "2"
kernle -a claire boot set imsg_dm_sean "1"
kernle -a claire boot set imsg_dm_ash "3"

# Network config
kernle -a claire boot set gateway_ip "192.168.50.218"
kernle -a claire boot set gateway_port "18789"
kernle -a claire boot set gateway_token "edf0af367cb54180..."

# Platform settings
kernle -a claire boot set default_channel "imessage"
```

### 3. Migrate Relationships

```bash
# Human steward
kernle -a claire relation update "Sean" \
  --type person \
  --notes "Steward. iMessage: +18326062074"

# Sibling SI
kernle -a claire relation update "Ash" \
  --type agent \
  --notes "Sibling SI. Gateway: 192.168.50.11:18789"
```

### 4. Migrate Learned Facts

```bash
# Facts about preferences/behavior → Notes
kernle -a claire note "Sean enjoys being surprised by creative applications" \
  --type insight \
  --tag sean,preferences

# Workflow observations → Notes or Beliefs
kernle -a claire belief "PyPI deploys via GitHub releases, not push" \
  --confidence 0.95 \
  --tag workflow
```

### 5. Verify Migration

```bash
# Check boot config
kernle -a claire boot list

# Check relationships
kernle -a claire relation list

# Check notes/beliefs
kernle -a claire dump --filter beliefs,notes
```

### 6. Update TOOLS.md

Remove migrated items from TOOLS.md. Keep only:
- Platform-specific notes (hardware, development workflow)
- Reference links
- Truly local information

Or rename to `PLATFORM_NOTES.md` to clarify its purpose.

## Example Migration

### Before (TOOLS.md)

```markdown
# TOOLS.md

## iMessage Chats
- Chat 1: Sean DM (+18326062074)
- Chat 2: Group chat (Sean + Ash + Claire) — DEFAULT
- Chat 3: Ash DM (emergentash@gmail.com)

## Contacts
### Sean (Steward)
- iMessage: +18326062074
- Likes being surprised by creative work

### Ash (Sibling SI)
- iMessage: emergentash@gmail.com
- Gateway: 192.168.50.11:18789

## Gateway
- My IP: 192.168.50.218
- Port: 18789
- Token: edf0af367cb54180...

## PyPI
- Deploys via GitHub releases
- No local credentials
```

### After (Boot + Kernle)

```bash
# Boot config (operational)
kernle -a claire boot list
# → imsg_group_chat: 2
# → imsg_dm_sean: 1
# → imsg_dm_ash: 3
# → gateway_ip: 192.168.50.218
# → gateway_port: 18789
# → gateway_token: edf0af367cb54180...

# Relationships
kernle -a claire relation list
# → Sean (person): Steward. iMessage: +18326062074
# → Ash (agent): Sibling SI. Gateway: 192.168.50.11:18789

# Notes
kernle -a claire note list
# → Sean enjoys being surprised by creative applications
# → PyPI deploys via GitHub releases, not push
```

### Updated TOOLS.md

```markdown
# TOOLS.md - Platform Notes

## Hardware
- Mac mini (Claire's host)
- Headless HDMI dongle for display session

## Development
- Kernle repo: ~/kernle
- Tests: uv run pytest tests/ -q

## Notes
- Everything else migrated to Kernle boot layer
- Use `kernle -a claire boot list` for config
```

## Verification Checklist

- [ ] Boot config contains all operational settings
- [ ] Relationships have contact info
- [ ] Learned facts are in notes/beliefs
- [ ] `kernle load` shows boot config at top
- [ ] `export-cache` includes boot config
- [ ] TOOLS.md only has platform-specific notes
- [ ] Test a session restart to verify context loads correctly

## Rollback

If something goes wrong, your original TOOLS.md still exists. Boot layer additions don't delete anything — you can use both simultaneously while migrating.

```bash
# Clear boot config if needed
kernle -a claire boot clear --confirm

# Start fresh
# (re-add items from TOOLS.md backup)
```

## Benefits After Migration

1. **Portability** — Boot config travels with your Kernle stack
2. **Queryability** — `boot get` for scripting, `boot list --json` for parsing
3. **Integration** — Config appears in `load` and `export-cache`
4. **Consistency** — Same mechanism across all platforms
5. **Auditability** — Database-backed, not flat file
