# Memory Integrity Architecture

## Overview

This document maps every enforcement point in kernle's memory system —
what is validated, where, and what gaps exist.

## Enforcement Layers

### Entity Layer (kernle/entity.py)

**What it enforces:**
- Active stack required for all writes (`_require_active_stack()` → `NoActiveStackError`)
- Timestamps auto-generated (`datetime.now(timezone.utc)`)
- Source attribution: `source_entity` set to `core:{id}` or `plugin:{name}`
- Source type: `source_type = "direct_experience"` on all direct writes

**What it passes through (no validation):**
- `derived_from` — accepted as-is, no hierarchy check (prior to v0.9.0)
- `confidence` — accepted as-is, no range validation
- `intensity` — accepted as-is, no range validation
- `priority` — accepted as-is, no range validation

### Stack Layer (kernle/stack/sqlite_stack.py)

**What it enforces (v0.9.0+):**
- Stack lifecycle state (INITIALIZING → ACTIVE → MAINTENANCE)
- Provenance validation on save (when ACTIVE):
  - `derived_from` non-empty for types that require it
  - Referenced type from allowed source layer
  - Referenced memory exists in the stack
- Component hooks (on_save, on_search, on_load, on_maintenance)

**What it does NOT enforce:**
- Content validation (accepts any string)
- Range validation on numeric fields

### Storage Layer (kernle/storage/sqlite.py)

**What it enforces:**
- `derived_from` cycle detection (via lineage.py, max depth 10)
- `source_type` is write-once on updates
- `derived_from` is append-only on updates (union merge)
- Schema integrity (SQLite constraints, NOT NULL, UNIQUE)
- Table name allowlisting (prevents SQL injection)

**What it does NOT enforce:**
- Provenance completeness
- Hierarchy rules

### Type Layer (kernle/types.py)

**What it enforces:**
- Nothing at present — no `__post_init__` validation, no range checks
- Types are pure data containers

## Memory Hierarchy

```
Raw (entry point — no provenance required)
  ↓
Episode / Note (must cite ≥1 raw)
  ↓
Belief (must cite ≥1 episode or note)
  ↓
Value (must cite ≥1 belief)

Episode → Goal (must cite ≥1 episode or belief)
Episode → Relationship (must cite ≥1 episode)
Episode/Belief → Drive (must cite ≥1 episode or belief)
```

## Strength Model (v0.9.0)

Replaces binary `is_forgotten` with continuous `strength: float`:

| Range     | Status   | Behavior                                |
|-----------|----------|-----------------------------------------|
| 0.8–1.0   | Strong   | Full weight in search/load              |
| 0.5–0.8   | Fading   | Included but reduced priority           |
| 0.2–0.5   | Weak     | Excluded from load(), still searchable  |
| 0.0–0.2   | Dormant  | Only via explicit `include_dormant=True` |
| 0.0       | Forgotten| Tombstoned, only via `recover()`        |

## Stack Lifecycle States

```
INITIALIZING → ACTIVE → (temporarily) MAINTENANCE → ACTIVE
```

- **INITIALIZING**: Seed writes allowed without provenance. Transitions to ACTIVE
  on first `on_attach()`.
- **ACTIVE**: All writes must have valid provenance (hierarchy enforced).
- **MAINTENANCE**: Only controlled admin operations (migrate, repair).
  Requires explicit call, returns to ACTIVE on exit.

## Controlled Access Operations (v0.9.0)

| Operation   | Method                              | Access        |
|-------------|-------------------------------------|---------------|
| Weaken      | `entity.weaken(id, amount, reason)` | SI via Entity |
| Forget      | `entity.forget(id, reason)`         | SI via Entity |
| Recover     | `entity.recover(id)`                | SI via Entity |
| Consolidate | `entity.consolidate(ids, summary)`  | SI via Entity |
| Repair      | `entity.repair(id, field, value)`   | Admin only    |
| Migrate     | `stack.migrate(memories, source)`   | Admin only    |
| Verify      | `entity.verify(id)`                 | SI via Entity |
| Protect     | `entity.protect(id)`                | SI via Entity |

## Audit Trail

All mutation operations create entries in `memory_audit`:

```sql
CREATE TABLE memory_audit (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

## Gap Table

| Field / Behavior | Should Be | Currently | Fixed In |
|------------------|-----------|-----------|----------|
| `derived_from` required | Yes (for non-raw) | Optional (None default) | v0.9.0 PR 2-3 |
| Hierarchy enforcement | Episode needs Raw, Belief needs Episode | None | v0.9.0 PR 2-3 |
| `confidence` range | 0.0–1.0 | Unclamped | Future |
| `intensity` range | 0.0–1.0 | Unclamped | Future |
| `priority` range | 0–100 | Unclamped | Future |
| `source_entity` on Goal/Drive/Relationship | Populated | Missing | v0.9.0 PR 3 |
| `repeat`/`avoid` on Episode | Wired | Silently dropped | v0.9.0 PR 3 |
| Plugin value/goal/drive attribution | source=plugin:name | Missing | v0.9.0 PR 3 |
| Stack lifecycle states | Enforced | None | v0.9.0 PR 2 |
| Controlled access model | Named operations | Direct stack calls | v0.9.0 PR 5 |
| Memory processing | Automated sessions | Manual | v0.9.0 PR 6 |
