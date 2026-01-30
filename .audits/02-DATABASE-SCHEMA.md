# Kernle Database Schema

> **Date**: 2026-01-29
> **Version**: Current implementation + proposed changes

## Overview

Kernle uses a dual-database architecture:
- **SQLite**: Local-first storage with sqlite-vec for semantic search
- **PostgreSQL/Supabase**: Cloud storage with pgvector for semantic search

Both implement the same logical schema with minor variations.

## Schema Version

Current: **v12** (SQLite), migrations up to **003** (PostgreSQL)

---

## Memory Tables

### episodes
**Purpose**: Autobiographical experiences with lessons learned

```sql
CREATE TABLE episodes (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    objective TEXT NOT NULL,           -- What was attempted
    outcome TEXT NOT NULL,             -- What happened
    outcome_type TEXT,                 -- 'success' | 'failure' | 'partial'
    lessons TEXT,                      -- JSON array of lessons learned
    tags TEXT,                         -- JSON array for categorization

    -- Emotional Memory
    emotional_valence REAL DEFAULT 0.0,  -- -1.0 (negative) to 1.0 (positive)
    emotional_arousal REAL DEFAULT 0.0,  -- 0.0 (calm) to 1.0 (intense)
    emotional_tags TEXT,                 -- JSON array: ["joy", "frustration"]

    -- Meta-Memory
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,              -- JSON array of supporting episode IDs
    derived_from TEXT,                 -- JSON array of memory refs (type:id)
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,           -- JSON array of changes

    -- Forgetting/Retention
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,                      -- e.g., "project:api-service"
    context_tags TEXT,                 -- JSON array

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_episodes_agent ON episodes(agent_id);
CREATE INDEX idx_episodes_created ON episodes(created_at);
CREATE INDEX idx_episodes_confidence ON episodes(confidence);
CREATE INDEX idx_episodes_is_forgotten ON episodes(is_forgotten);
CREATE INDEX idx_episodes_is_protected ON episodes(is_protected);
```

---

### beliefs
**Purpose**: What the agent holds true with confidence tracking

```sql
CREATE TABLE beliefs (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    statement TEXT NOT NULL,
    belief_type TEXT DEFAULT 'fact',   -- 'fact' | 'preference' | 'observation'

    -- Meta-Memory
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,              -- JSON array
    derived_from TEXT,                 -- JSON array
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,           -- JSON array

    -- Belief Revision Chain
    supersedes TEXT,                   -- ID of belief this replaced
    superseded_by TEXT,                -- ID of belief that replaced this
    times_reinforced INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 1,       -- False if superseded

    -- Forgetting/Retention
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_beliefs_agent ON beliefs(agent_id);
CREATE INDEX idx_beliefs_confidence ON beliefs(confidence);
CREATE INDEX idx_beliefs_is_active ON beliefs(is_active);
CREATE INDEX idx_beliefs_supersedes ON beliefs(supersedes);
CREATE INDEX idx_beliefs_is_forgotten ON beliefs(is_forgotten);
```

---

### notes
**Purpose**: Quick captures (decisions, insights, quotes)

```sql
CREATE TABLE notes (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    content TEXT NOT NULL,
    note_type TEXT DEFAULT 'note',     -- 'note' | 'decision' | 'insight' | 'quote'
    speaker TEXT,                      -- For quotes: who said it
    reason TEXT,                       -- For decisions: rationale
    tags TEXT,                         -- JSON array

    -- Meta-Memory
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,

    -- Forgetting/Retention
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

---

### agent_values
**Purpose**: Core identity values (protected by default)

```sql
CREATE TABLE agent_values (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    name TEXT NOT NULL,
    statement TEXT NOT NULL,
    priority INTEGER DEFAULT 50,       -- 0-100, higher = more important

    -- Meta-Memory
    confidence REAL DEFAULT 0.9,       -- Higher default for values
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,

    -- Forgetting/Retention (PROTECTED BY DEFAULT)
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,    -- Values protected by default
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

---

### goals
**Purpose**: What the agent is working toward

```sql
CREATE TABLE goals (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT DEFAULT 'medium',    -- 'low' | 'medium' | 'high' | 'critical'
    status TEXT DEFAULT 'active',      -- 'active' | 'completed' | 'paused'

    -- Meta-Memory (standard fields)
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,

    -- Forgetting/Retention (standard fields)
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

---

### drives
**Purpose**: Intrinsic motivation system (protected by default)

```sql
CREATE TABLE drives (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    drive_type TEXT NOT NULL,          -- 'existence' | 'growth' | 'curiosity' | 'connection' | 'reproduction'
    intensity REAL DEFAULT 0.5,        -- 0.0 to 1.0
    focus_areas TEXT,                  -- JSON array

    -- Meta-Memory
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,

    -- Forgetting/Retention (PROTECTED BY DEFAULT)
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 1,    -- Drives protected by default
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,

    UNIQUE(agent_id, drive_type)
);
```

---

### relationships
**Purpose**: Models of other agents/entities

```sql
CREATE TABLE relationships (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,         -- 'agent' | 'person' | 'organization'
    relationship_type TEXT NOT NULL,   -- 'peer' | 'mentor' | 'collaborator'
    notes TEXT,
    sentiment REAL DEFAULT 0.0,        -- -1.0 to 1.0
    interaction_count INTEGER DEFAULT 0,
    last_interaction TEXT,

    -- Meta-Memory (standard fields)
    confidence REAL DEFAULT 0.8,
    source_type TEXT DEFAULT 'direct_experience',
    source_episodes TEXT,
    derived_from TEXT,
    last_verified TEXT,
    verification_count INTEGER DEFAULT 0,
    confidence_history TEXT,

    -- Forgetting/Retention (standard fields)
    times_accessed INTEGER DEFAULT 0,
    last_accessed TEXT,
    is_protected INTEGER DEFAULT 0,
    is_forgotten INTEGER DEFAULT 0,
    forgotten_at TEXT,
    forgotten_reason TEXT,

    -- Context/Scope
    context TEXT,
    context_tags TEXT,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0,

    UNIQUE(agent_id, entity_name)
);
```

---

### playbooks
**Purpose**: Procedural memory ("how I do things")

```sql
CREATE TABLE playbooks (
    -- Identity
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core Content
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger_conditions TEXT NOT NULL,  -- JSON array
    steps TEXT NOT NULL,               -- JSON array of {action, details, adaptations}
    failure_modes TEXT NOT NULL,       -- JSON array
    recovery_steps TEXT,               -- JSON array (optional)

    -- Mastery Tracking
    mastery_level TEXT DEFAULT 'novice', -- 'novice' | 'competent' | 'proficient' | 'expert'
    times_used INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    last_used TEXT,

    -- Meta-Memory
    source_episodes TEXT,              -- JSON array
    tags TEXT,                         -- JSON array
    confidence REAL DEFAULT 0.8,

    -- Timestamps & Sync
    created_at TEXT NOT NULL,
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

---

### raw_entries (CURRENT - see proposed changes)
**Purpose**: Unstructured capture for later processing

```sql
-- CURRENT SCHEMA (to be simplified)
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT DEFAULT 'manual',
    processed INTEGER DEFAULT 0,
    processed_into TEXT,               -- JSON array of type:id refs
    tags TEXT,                         -- JSON array

    -- Meta-Memory (TO BE REMOVED)
    confidence REAL DEFAULT 1.0,
    source_type TEXT DEFAULT 'direct_experience',

    -- Sync
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

-- PROPOSED SCHEMA (simplified blob-based)
-- See 01-PROPOSED-RAW-LAYER.md
```

---

## Supporting Tables

### memory_suggestions
**Purpose**: Auto-extracted suggestions awaiting agent review

```sql
CREATE TABLE memory_suggestions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,         -- 'episode' | 'belief' | 'note'
    content TEXT NOT NULL,             -- JSON object with structured data
    confidence REAL DEFAULT 0.5,
    source_raw_ids TEXT NOT NULL,      -- JSON array of raw entry IDs
    status TEXT DEFAULT 'pending',     -- 'pending' | 'promoted' | 'modified' | 'rejected'
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution_reason TEXT,
    promoted_to TEXT,                  -- Format: type:id

    -- Sync
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

---

### health_check_events
**Purpose**: Compliance and anxiety tracking

```sql
CREATE TABLE health_check_events (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    anxiety_score INTEGER,
    source TEXT DEFAULT 'cli',         -- 'cli' | 'mcp'
    triggered_by TEXT DEFAULT 'manual' -- 'boot' | 'heartbeat' | 'manual'
);
```

---

## Sync Infrastructure

### sync_queue
**Purpose**: Offline changes queue with deduplication

```sql
CREATE TABLE sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    operation TEXT NOT NULL,           -- 'insert' | 'update' | 'delete'
    data TEXT,                         -- JSON payload
    local_updated_at TEXT NOT NULL,
    synced INTEGER DEFAULT 0           -- 0=pending, 1=synced
);

-- Partial unique index for atomic UPSERT
CREATE UNIQUE INDEX idx_sync_queue_unsynced_unique
    ON sync_queue(table_name, record_id) WHERE synced = 0;
```

### sync_meta
**Purpose**: Global sync state

```sql
CREATE TABLE sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### sync_conflicts
**Purpose**: Resolved conflict history for visibility

```sql
CREATE TABLE sync_conflicts (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    local_version TEXT NOT NULL,       -- JSON snapshot
    cloud_version TEXT NOT NULL,       -- JSON snapshot
    resolution TEXT NOT NULL,          -- 'local_wins' | 'cloud_wins'
    resolved_at TEXT NOT NULL,
    local_summary TEXT,
    cloud_summary TEXT
);
```

---

## Vector Search (SQLite)

### vec_embeddings (virtual table via sqlite-vec)
**Purpose**: Semantic search embeddings

```sql
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[384]               -- 384-dim hash embeddings
);
```

### embedding_meta
**Purpose**: Track what's been embedded

```sql
CREATE TABLE embedding_meta (
    id TEXT PRIMARY KEY,
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

---

## PostgreSQL/Supabase Variations

### Key Differences

| Aspect | SQLite | PostgreSQL |
|--------|--------|------------|
| IDs | TEXT (UUID strings) | UUID or TEXT |
| Timestamps | TEXT (ISO strings) | TIMESTAMPTZ |
| Arrays | JSON text | JSONB or native arrays |
| Vectors | FLOAT[384] via sqlite-vec | vector(384) via pgvector |
| Booleans | INTEGER (0/1) | BOOLEAN |
| Embedding dim | 384 (hash) | 1536 (OpenAI) |

### Table Name Mapping

| SQLite | PostgreSQL |
|--------|-----------|
| notes | memories |
| drives | agent_drives |
| relationships | agent_relationships |
| goals | agent_goals (with extra fields) |
| - | agents (registry) |
| - | users (auth) |
| - | api_keys |
| - | api_key_usage |

### Additional PostgreSQL Tables

- `agents`: Agent registry with user association
- `users`: User accounts (separate from agents)
- `api_keys`: API key management
- `api_key_usage`: Rate limiting tracking
- `sync_logs`: Sync operation debugging
- `sync_metadata`: Per-agent sync state
- `emotional_memories`: Dedicated emotional memory table

---

## Common Field Patterns

### All Memory Tables Include:

**Meta-Memory Fields:**
```sql
confidence REAL DEFAULT 0.8,
source_type TEXT DEFAULT 'direct_experience',
source_episodes TEXT,              -- JSON array
derived_from TEXT,                 -- JSON array
last_verified TEXT,
verification_count INTEGER DEFAULT 0,
confidence_history TEXT            -- JSON array
```

**Forgetting Fields:**
```sql
times_accessed INTEGER DEFAULT 0,
last_accessed TEXT,
is_protected INTEGER DEFAULT 0,    -- 1 for values/drives
is_forgotten INTEGER DEFAULT 0,
forgotten_at TEXT,
forgotten_reason TEXT
```

**Sync Fields:**
```sql
local_updated_at TEXT NOT NULL,
cloud_synced_at TEXT,
version INTEGER DEFAULT 1,
deleted INTEGER DEFAULT 0
```

**Context Fields (optional):**
```sql
context TEXT,                      -- e.g., "project:api-service"
context_tags TEXT                  -- JSON array
```

---

## Index Strategy

### Query Patterns Optimized:

1. **By agent**: All tables indexed on `agent_id`
2. **By timestamp**: `created_at`, `local_updated_at` for recency queries
3. **By confidence**: For priority loading and decay calculations
4. **By forgotten/protected**: For retention queries
5. **By sync status**: `cloud_synced_at` for sync operations
6. **By active status**: `is_active` on beliefs, `status` on goals
