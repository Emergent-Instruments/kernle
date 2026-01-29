# Schema Recommendation: Multi-Tenant Agent Namespacing

**Date:** 2025-01-28  
**Author:** Claire (Schema Architect Subagent)  
**Status:** Recommendation

---

## Executive Summary

**Recommended Approach: Option 2 — Change FKs to reference `agents.id` (UUID)**

This approach provides clean multi-tenant separation with minimal schema changes, leveraging the existing UUID primary key that's already globally unique.

---

## Current State Analysis

### Existing Schema

```sql
-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- ← Already unique!
    agent_id TEXT UNIQUE NOT NULL,  -- Currently globally unique
    user_id TEXT NOT NULL,          -- Added in migration 005
    secret_hash TEXT NOT NULL,
    ...
);

-- Memory tables (example: episodes)
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    ...
);
```

### The Problem

Migration 008 attempted to change `agent_id` from globally unique to unique-per-user:

```sql
-- Drops global uniqueness
ALTER TABLE agents DROP CONSTRAINT agents_agent_id_key;
-- Adds composite uniqueness
ALTER TABLE agents ADD CONSTRAINT agents_user_agent_unique UNIQUE (user_id, agent_id);
-- Re-adds FK... but this FAILS!
ALTER TABLE episodes ADD CONSTRAINT episodes_agent_id_fkey 
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE;
```

**Critical Flaw:** PostgreSQL requires FK target columns to have a unique constraint covering them alone (or be a primary key). Once `agent_id` is only unique within `(user_id, agent_id)`, the FK reference becomes invalid.

### Desired Outcome

Multiple users can have agents with the same name:
- User A has `claire` → `usr_abc123/claire`
- User B has `claire` → `usr_xyz789/claire`

---

## Options Analysis

### Option 1: Add user_id to All Memory Tables

**Approach:** Add `user_id` column to every memory table, create composite FK `(user_id, agent_id)`.

```sql
-- Memory table changes
ALTER TABLE episodes ADD COLUMN user_id TEXT NOT NULL;
ALTER TABLE episodes ADD CONSTRAINT episodes_agent_fkey 
    FOREIGN KEY (user_id, agent_id) REFERENCES agents(user_id, agent_id) ON DELETE CASCADE;
```

**Pros:**
- Clear multi-tenant data separation
- FK constraint properly enforced
- Easy tenant-level queries (`WHERE user_id = ?`)

**Cons:**
- ❌ **Massive migration**: 11 memory tables × schema change + data backfill
- ❌ **Storage overhead**: 12 bytes per row × millions of rows
- ❌ **Query complexity**: Every query needs `user_id` parameter
- ❌ **API changes**: All endpoints need user_id in payload
- ❌ **Index bloat**: Every table needs composite indexes on `(user_id, agent_id)`
- ❌ **Sync protocol changes**: Bi-directional sync needs user_id in all records

**Verdict:** Over-engineered. Solves the problem but creates many new ones.

---

### Option 2: Change FKs to Reference `agents.id` (UUID) ⭐ RECOMMENDED

**Approach:** Keep memory tables as-is, but change FK from `agent_id` (TEXT) to `agent_ref` (UUID) referencing `agents.id`.

```sql
-- Memory table changes (minimal)
ALTER TABLE episodes ADD COLUMN agent_ref UUID;
UPDATE episodes e SET agent_ref = (SELECT id FROM agents WHERE agent_id = e.agent_id);
ALTER TABLE episodes ALTER COLUMN agent_ref SET NOT NULL;
ALTER TABLE episodes DROP CONSTRAINT episodes_agent_id_fkey;
ALTER TABLE episodes ADD CONSTRAINT episodes_agent_ref_fkey 
    FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;
-- Keep agent_id for queries (indexed), but it's no longer the FK
```

**Pros:**
- ✅ **Minimal schema change**: One column addition per table, no storage overhead after migration
- ✅ **UUID is already unique**: No constraint changes needed on agents table
- ✅ **Clean separation**: `agent_id` = human-readable name, `agent_ref` = FK integrity
- ✅ **Standard pattern**: UUID FKs are industry standard for multi-tenant SaaS
- ✅ **Query patterns preserved**: Can still filter by `agent_id` (indexed)
- ✅ **Future-proof**: Easy to add agent aliases, renames, etc.
- ✅ **Sync protocol unchanged**: Records still have `agent_id` for display

**Cons:**
- ⚠️ Need to update queries that JOIN on agent_id (use agent_ref instead)
- ⚠️ 16 bytes per FK (UUID) vs 0 bytes (shared column) — negligible

**Query Pattern After Migration:**

```sql
-- Before: FK was on agent_id
SELECT * FROM episodes WHERE agent_id = 'claire';

-- After: Still works! agent_id is preserved for filtering
SELECT * FROM episodes WHERE agent_id = 'claire';

-- JOINs use the FK column
SELECT e.*, a.display_name 
FROM episodes e 
JOIN agents a ON e.agent_ref = a.id
WHERE e.agent_id = 'claire';
```

**Verdict:** Clean, minimal, industry-standard solution.

---

### Option 3: Compound agent_id (`{user_id}_{name}`)

**Approach:** Keep `agent_id` globally unique but make it compound: `usr_abc123_claire`.

```sql
-- Registration creates compound agent_id
INSERT INTO agents (agent_id, user_id, ...) 
VALUES ('usr_abc123_claire', 'usr_abc123', ...);
```

**Pros:**
- ✅ No schema changes to memory tables
- ✅ agent_id remains globally unique
- ✅ FK constraints work as-is

**Cons:**
- ❌ **Ugly**: `usr_abc123_claire` vs `claire`
- ❌ **Parsing required**: Need to extract user_id and name from compound
- ❌ **Breaking change**: Existing agents need migration
- ❌ **Format coupling**: If user_id format changes, agent_id parsing breaks
- ❌ **Confusing UX**: Users see ugly compound names in CLI output
- ❌ **Namespace display**: `kernle -a usr_abc123_claire` is painful

**Verdict:** Technically works, but poor UX and maintainability.

---

### Option 4: Keep agent_id Globally Unique (Status Quo)

**Approach:** Accept that agent names must be globally unique.

**Pros:**
- ✅ No migration needed
- ✅ Simple mental model

**Cons:**
- ❌ **Namespace collision**: First user to register "claire" owns it forever
- ❌ **Poor multi-tenant UX**: Users can't use intuitive names
- ❌ **Doesn't scale**: Early users exhaust good names

**Verdict:** Unacceptable for a multi-tenant system.

---

## Recommendation: Option 2 (UUID FK)

### Rationale

1. **Minimal disruption**: Adds one column per table, preserves all existing query patterns
2. **Industry standard**: UUID FKs are the norm for multi-tenant SaaS databases
3. **Clean separation of concerns**:
   - `agent_id` = human-readable project name (for display, filtering)
   - `agent_ref` = stable FK for data integrity (never changes)
   - `user_id` = tenant identifier (on agents table only)
4. **Future flexibility**: Easy to add agent renaming, aliases, or sharing later

### Schema After Migration

```sql
-- Agents table (unchanged except unique constraint)
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,  -- No longer globally unique!
    user_id TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    ...
    UNIQUE(user_id, agent_id)  -- Unique per user
);

-- Memory tables (example: episodes)
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,        -- Kept for filtering/display
    agent_ref UUID NOT NULL        -- FK for integrity
        REFERENCES agents(id) ON DELETE CASCADE,
    ...
);

CREATE INDEX idx_episodes_agent_id ON episodes(agent_id);
CREATE INDEX idx_episodes_agent_ref ON episodes(agent_ref);
```

---

## Migration Strategy

### Phase 1: Prepare (Non-Breaking)

```sql
-- 1. Add agent_ref column (nullable initially)
ALTER TABLE episodes ADD COLUMN agent_ref UUID;
ALTER TABLE beliefs ADD COLUMN agent_ref UUID;
-- ... all 11 tables

-- 2. Create indexes
CREATE INDEX idx_episodes_agent_ref ON episodes(agent_ref);
-- ... all 11 tables
```

### Phase 2: Backfill

```sql
-- 3. Populate agent_ref from existing agent_id
UPDATE episodes e 
SET agent_ref = (SELECT id FROM agents a WHERE a.agent_id = e.agent_id);

-- Verify no nulls
SELECT COUNT(*) FROM episodes WHERE agent_ref IS NULL;  -- Should be 0
```

### Phase 3: Cutover

```sql
-- 4. Make agent_ref NOT NULL
ALTER TABLE episodes ALTER COLUMN agent_ref SET NOT NULL;

-- 5. Drop old FK
ALTER TABLE episodes DROP CONSTRAINT IF EXISTS episodes_agent_id_fkey;

-- 6. Add new FK
ALTER TABLE episodes ADD CONSTRAINT episodes_agent_ref_fkey 
    FOREIGN KEY (agent_ref) REFERENCES agents(id) ON DELETE CASCADE;

-- 7. Change agents unique constraint
ALTER TABLE agents DROP CONSTRAINT IF EXISTS agents_agent_id_key;
ALTER TABLE agents ADD CONSTRAINT agents_user_agent_unique UNIQUE (user_id, agent_id);
```

### Phase 4: Cleanup

```sql
-- 8. Create index for unique constraint
CREATE INDEX IF NOT EXISTS idx_agents_user_agent ON agents(user_id, agent_id);
```

### Rollback Plan

If issues arise:
1. `agent_id` columns are preserved — can restore old FK
2. `agent_ref` can be dropped if needed
3. Unique constraint change is reversible

---

## API/CLI Impact

### Registration

**Before:**
```bash
kernle auth register claire
# Creates agent with agent_id='claire' (globally unique)
```

**After:**
```bash
kernle auth register claire
# Creates agent with agent_id='claire', user_id='usr_xxx'
# Unique only within that user's namespace
```

No CLI change needed — same command works.

### Authentication

**Before:**
```python
# JWT contains agent_id
{"sub": "claire", "user_id": "usr_abc123"}
```

**After:**
```python
# Same! agent_id is still the identifier
{"sub": "claire", "user_id": "usr_abc123"}
```

Backend uses `(user_id, agent_id)` for lookups internally.

### Sync Protocol

**Before:**
```json
{"agent_id": "claire", "data": {...}}
```

**After:**
```json
{"agent_id": "claire", "data": {...}}
```

No change. Backend resolves `agent_id` using authenticated `user_id` from token.

### Database Queries

**Before:**
```python
db.table("episodes").select("*").eq("agent_id", "claire").execute()
```

**After:**
```python
# Same! agent_id column preserved for filtering
db.table("episodes").select("*").eq("agent_id", "claire").execute()
```

JOINs that need agent metadata use `agent_ref`:
```python
db.table("episodes").select("*, agents(*)").eq("episodes.agent_ref", agent_uuid).execute()
```

---

## Implementation Checklist

- [ ] Write migration SQL (Phase 1-4)
- [ ] Test migration on staging with production data copy
- [ ] Update `database.py` to set `agent_ref` on insert
- [ ] Update sync routes to handle `agent_ref`
- [ ] Update RLS policies if needed
- [ ] Test registration of duplicate agent names for different users
- [ ] Test sync with existing agents (backward compatibility)
- [ ] Deploy migration during low-traffic window
- [ ] Monitor for FK constraint violations

---

## Appendix: Tables Requiring Migration

| Table | Current FK | New FK |
|-------|-----------|--------|
| episodes | agent_id → agents(agent_id) | agent_ref → agents(id) |
| beliefs | agent_id → agents(agent_id) | agent_ref → agents(id) |
| values | agent_id → agents(agent_id) | agent_ref → agents(id) |
| goals | agent_id → agents(agent_id) | agent_ref → agents(id) |
| notes | agent_id → agents(agent_id) | agent_ref → agents(id) |
| drives | agent_id → agents(agent_id) | agent_ref → agents(id) |
| relationships | agent_id → agents(agent_id) | agent_ref → agents(id) |
| checkpoints | agent_id → agents(agent_id) | agent_ref → agents(id) |
| raw_captures | agent_id → agents(agent_id) | agent_ref → agents(id) |
| playbooks | agent_id → agents(agent_id) | agent_ref → agents(id) |
| emotional_memories | agent_id → agents(agent_id) | agent_ref → agents(id) |

---

## Summary

| Criteria | Option 1 (Composite) | Option 2 (UUID FK) | Option 3 (Compound) | Option 4 (Status Quo) |
|----------|---------------------|-------------------|--------------------|-----------------------|
| Schema changes | Heavy | Light | None | None |
| Query complexity | Higher | Same | Same | Same |
| UX impact | High | None | High | High |
| Industry standard | ✓ | ✓✓ | ✗ | N/A |
| Future flexibility | Medium | High | Low | None |
| Migration risk | High | Low | Medium | None |

**Recommendation: Option 2 (UUID FK)** provides the cleanest path to proper multi-tenant namespacing with minimal disruption to the existing codebase and API contracts.
