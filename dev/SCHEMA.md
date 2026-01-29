# Kernle Database Schema

This document describes the database schemas used by Kernle's storage backends.

## Overview

Kernle supports two storage backends:
- **SQLite** (`SQLiteStorage`): Local-first storage with sqlite-vec for semantic search
- **PostgreSQL/Supabase** (`SupabaseStorage`): Cloud storage with pgvector

Both backends implement the `Storage` protocol defined in `kernle/storage/base.py`.

---

## SQLite Schema (Version 10)

The SQLite schema is defined in `kernle/storage/sqlite.py`.

### Core Tables

#### `episodes`
Stores episodic memories (experiences/work logs).

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `objective` | TEXT NOT NULL | What was attempted |
| `outcome` | TEXT NOT NULL | What happened |
| `outcome_type` | TEXT | success/failure/partial |
| `lessons` | TEXT | JSON array of lessons learned |
| `tags` | TEXT | JSON array of tags |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| `emotional_valence` | REAL DEFAULT 0.0 | -1.0 to 1.0 (negative to positive) |
| `emotional_arousal` | REAL DEFAULT 0.0 | 0.0 to 1.0 (calm to intense) |
| `emotional_tags` | TEXT | JSON array of emotion tags |
| `confidence` | REAL DEFAULT 0.8 | Memory confidence score |
| `source_type` | TEXT DEFAULT 'direct_experience' | How memory was acquired |
| `source_episodes` | TEXT | JSON array of source episode IDs |
| `derived_from` | TEXT | JSON array of memory refs (type:id) |
| `last_verified` | TEXT | ISO timestamp of last verification |
| `verification_count` | INTEGER DEFAULT 0 | Times verified |
| `confidence_history` | TEXT | JSON array of confidence changes |
| `times_accessed` | INTEGER DEFAULT 0 | Access count for forgetting |
| `last_accessed` | TEXT | Last access timestamp |
| `is_protected` | INTEGER DEFAULT 0 | Protected from forgetting |
| `is_forgotten` | INTEGER DEFAULT 0 | Tombstone flag |
| `forgotten_at` | TEXT | When forgotten |
| `forgotten_reason` | TEXT | Why forgotten |
| `local_updated_at` | TEXT NOT NULL | Local modification time |
| `cloud_synced_at` | TEXT | Last sync to cloud |
| `version` | INTEGER DEFAULT 1 | Optimistic concurrency |
| `deleted` | INTEGER DEFAULT 0 | Soft delete flag |

#### `beliefs`
Stores semantic beliefs/facts.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `statement` | TEXT NOT NULL | The belief statement |
| `belief_type` | TEXT DEFAULT 'fact' | fact/preference/observation |
| `confidence` | REAL DEFAULT 0.8 | Belief confidence |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| `source_type` | TEXT | How acquired |
| `source_episodes` | TEXT | Supporting episodes |
| `derived_from` | TEXT | Memory refs this derived from |
| `last_verified` | TEXT | Last verification time |
| `verification_count` | INTEGER DEFAULT 0 | Times verified |
| `confidence_history` | TEXT | JSON confidence changes |
| `supersedes` | TEXT | ID of belief this replaced |
| `superseded_by` | TEXT | ID of belief that replaced this |
| `times_reinforced` | INTEGER DEFAULT 0 | Confirmation count |
| `is_active` | INTEGER DEFAULT 1 | Active vs archived |
| `times_accessed` | INTEGER DEFAULT 0 | Forgetting metric |
| `last_accessed` | TEXT | Last access time |
| `is_protected` | INTEGER DEFAULT 0 | Protected flag |
| `is_forgotten` | INTEGER DEFAULT 0 | Tombstone |
| `forgotten_at` | TEXT | When forgotten |
| `forgotten_reason` | TEXT | Why forgotten |
| `local_updated_at` | TEXT NOT NULL | Local modification |
| `cloud_synced_at` | TEXT | Cloud sync time |
| `version` | INTEGER DEFAULT 1 | Concurrency version |
| `deleted` | INTEGER DEFAULT 0 | Soft delete |

#### `agent_values`
Stores normative values (highest authority in memory hierarchy).

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `name` | TEXT NOT NULL | Value name |
| `statement` | TEXT NOT NULL | Value description |
| `priority` | INTEGER DEFAULT 50 | Priority ranking |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| `confidence` | REAL DEFAULT 0.9 | Value confidence |
| `source_type` | TEXT | How acquired |
| `source_episodes` | TEXT | Supporting episodes |
| `derived_from` | TEXT | Derived from refs |
| `last_verified` | TEXT | Verification time |
| `verification_count` | INTEGER DEFAULT 0 | Times verified |
| `confidence_history` | TEXT | Confidence changes |
| `times_accessed` | INTEGER DEFAULT 0 | Access count |
| `last_accessed` | TEXT | Last access |
| `is_protected` | INTEGER DEFAULT 1 | **Protected by default** |
| `is_forgotten` | INTEGER DEFAULT 0 | Tombstone |
| `forgotten_at` | TEXT | Forgotten time |
| `forgotten_reason` | TEXT | Reason |
| `local_updated_at` | TEXT NOT NULL | Local mod time |
| `cloud_synced_at` | TEXT | Cloud sync |
| `version` | INTEGER DEFAULT 1 | Version |
| `deleted` | INTEGER DEFAULT 0 | Soft delete |

#### `goals`
Stores agent goals.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `title` | TEXT NOT NULL | Goal title |
| `description` | TEXT | Goal description |
| `priority` | TEXT DEFAULT 'medium' | low/medium/high |
| `status` | TEXT DEFAULT 'active' | active/completed/paused |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| `confidence` | REAL DEFAULT 0.8 | Goal confidence |
| (meta-memory fields) | ... | Same as episodes |
| (forgetting fields) | ... | Same as episodes |
| (sync fields) | ... | Same as episodes |

#### `notes`
Stores quick notes, decisions, insights, quotes.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `content` | TEXT NOT NULL | Note content |
| `note_type` | TEXT DEFAULT 'note' | note/decision/insight/quote |
| `speaker` | TEXT | For quotes |
| `reason` | TEXT | For decisions |
| `tags` | TEXT | JSON array |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| (meta-memory fields) | ... | Same as episodes |
| (forgetting fields) | ... | Same as episodes |
| (sync fields) | ... | Same as episodes |

#### `drives`
Stores motivational drives.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `drive_type` | TEXT NOT NULL | curiosity/autonomy/competence/connection |
| `intensity` | REAL DEFAULT 0.5 | 0.0 to 1.0 |
| `focus_areas` | TEXT | JSON array |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| `updated_at` | TEXT NOT NULL | Last update |
| `is_protected` | INTEGER DEFAULT 1 | **Protected by default** |
| (other fields) | ... | Same pattern as above |
| UNIQUE(agent_id, drive_type) | | One drive per type per agent |

#### `relationships`
Stores models of other agents/entities.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `entity_name` | TEXT NOT NULL | Name of other entity |
| `entity_type` | TEXT NOT NULL | agent/person/organization |
| `relationship_type` | TEXT NOT NULL | peer/mentor/collaborator |
| `notes` | TEXT | Relationship notes |
| `sentiment` | REAL DEFAULT 0.0 | -1.0 to 1.0 |
| `interaction_count` | INTEGER DEFAULT 0 | Interactions |
| `last_interaction` | TEXT | Last interaction time |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| (other fields) | ... | Same pattern |
| UNIQUE(agent_id, entity_name) | | One relationship per entity |

#### `playbooks`
Stores procedural memory ("how I do things").

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `name` | TEXT NOT NULL | Playbook name |
| `description` | TEXT NOT NULL | What it does |
| `trigger_conditions` | TEXT NOT NULL | JSON array of when to use |
| `steps` | TEXT NOT NULL | JSON array of {action, details} |
| `failure_modes` | TEXT NOT NULL | JSON array of what can go wrong |
| `recovery_steps` | TEXT | JSON array of recovery actions |
| `mastery_level` | TEXT DEFAULT 'novice' | novice/competent/proficient/expert |
| `times_used` | INTEGER DEFAULT 0 | Usage count |
| `success_rate` | REAL DEFAULT 0.0 | Success percentage |
| `source_episodes` | TEXT | JSON array of source episodes |
| `tags` | TEXT | JSON array |
| `confidence` | REAL DEFAULT 0.8 | Confidence |
| `last_used` | TEXT | Last use time |
| `created_at` | TEXT NOT NULL | ISO timestamp |
| (sync fields) | ... | Same as episodes |

#### `raw_entries`
Stores unstructured captures for later processing.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | UUID |
| `agent_id` | TEXT NOT NULL | Agent identifier |
| `content` | TEXT NOT NULL | Raw content |
| `timestamp` | TEXT NOT NULL | Capture time |
| `source` | TEXT DEFAULT 'manual' | Source of entry |
| `processed` | INTEGER DEFAULT 0 | Processing flag |
| `processed_into` | TEXT | JSON array of memory refs |
| `tags` | TEXT | JSON array |
| `confidence` | REAL DEFAULT 1.0 | Confidence |
| `source_type` | TEXT | Source type |
| (sync fields) | ... | Same as episodes |

### Sync Tables

#### `sync_queue`
Queue for offline changes pending sync. Enhanced in v10 with better deduplication support.

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER PRIMARY KEY | Auto-increment |
| `table_name` | TEXT NOT NULL | Target table |
| `record_id` | TEXT NOT NULL | Record UUID |
| `operation` | TEXT NOT NULL | insert/update/delete |
| `data` | TEXT | JSON payload of the record data (v10+) |
| `local_updated_at` | TEXT NOT NULL | When change was queued (v10+) |
| `synced` | INTEGER DEFAULT 0 | 0=pending, 1=synced (v10+) |
| `payload` | TEXT | Legacy JSON payload (backward compat) |
| `queued_at` | TEXT | Legacy timestamp (backward compat) |

**Note**: The sync queue deduplicates by `(table_name, record_id)`, keeping only the latest operation for each record.

#### `sync_meta`
Sync state metadata.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT PRIMARY KEY | Metadata key |
| `value` | TEXT NOT NULL | Metadata value |
| `updated_at` | TEXT NOT NULL | Last update |

#### `embedding_meta`
Tracks embedded content for change detection.

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PRIMARY KEY | table:record_id |
| `table_name` | TEXT NOT NULL | Source table |
| `record_id` | TEXT NOT NULL | Record UUID |
| `content_hash` | TEXT NOT NULL | Content hash |
| `created_at` | TEXT NOT NULL | Embedding time |

### Vector Table (sqlite-vec)

When sqlite-vec is available:

```sql
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    id TEXT PRIMARY KEY,  -- format: "table:record_id"
    embedding FLOAT[{dim}]  -- dimension from embedder
);
```

---

## Supabase/PostgreSQL Schema

The Supabase schema maps to these table names (see `kernle/storage/postgres.py`):

| SQLite Table | Supabase Table | Notes |
|--------------|----------------|-------|
| `episodes` | `agent_episodes` | Different column names |
| `beliefs` | `agent_beliefs` | |
| `agent_values` | `agent_values` | Same name |
| `goals` | `agent_goals` | |
| `notes` | `agent_notes` | |
| `drives` | `agent_drives` | |
| `relationships` | `agent_relationships` | |
| `playbooks` | — | SQLite only (procedural memory) |
| `raw_entries` | — | SQLite only (raw capture layer) |

### Feature Support Comparison

| Feature | SQLite | Supabase/Postgres |
|---------|--------|-------------------|
| **Core Memory Types** | ✅ Full | ✅ Full |
| **Playbooks** | ✅ | ❌ Not implemented |
| **Raw Entries** | ✅ | ❌ Not implemented |
| **Emotional Memory** | ✅ | ✅ |
| **Meta-Memory** | ✅ Full | ⚠️ Partial |
| **Belief Revision** | ✅ | ⚠️ Partial |
| **Forgetting** | ✅ | ❌ Not implemented |
| **Access Tracking** | ✅ | ❌ Not implemented |
| **Vector Search** | sqlite-vec | pgvector |
| **Offline Support** | ✅ | Requires connection |
| **Sync Queue** | ✅ | N/A (cloud native) |

**Recommendation**: Use SQLite for full functionality. Supabase is suitable for cloud sync of core memories (episodes, beliefs, values, goals, notes, drives, relationships).

### Column Name Differences

**Episodes:**
- SQLite `outcome` → Supabase `outcome_description`
- SQLite `lessons` → Supabase `lessons_learned`

### Features Not Yet in Supabase

1. **Playbooks** - Procedural memory only in SQLite
2. **Raw entries** - Capture queue only in SQLite
3. **Forgetting fields** - Only in SQLite
4. **Belief revision fields** - `supersedes`, `superseded_by`, `times_reinforced`, `is_active`
5. **Full meta-memory** - Partial support in Supabase

---

## Migration Path for New Features

When adding new features:

1. **Add to SQLite first** - Update `SCHEMA` and `SCHEMA_VERSION` in `sqlite.py`
2. **Add migration** - Update `_migrate_schema()` with `ALTER TABLE` statements
3. **Update dataclasses** - Add fields to `base.py` dataclasses
4. **Update row converters** - Use `_safe_get()` for backwards compatibility
5. **Add to Supabase** - Create migration SQL for PostgreSQL
6. **Document here** - Update this file

### Schema Version History

| Version | Changes |
|---------|---------|
| 10 | Enhanced sync_queue with `data`, `local_updated_at`, `synced` columns |
| 9 | Added forgetting fields (times_accessed, is_protected, is_forgotten, etc.) |
| 8 | Added belief revision fields (supersedes, superseded_by, times_reinforced) |
| 7 | Added meta-memory fields (confidence_history, source_type, etc.) |
| 6 | Added emotional memory fields |
| 5 | Added playbooks table |
| 4 | Added raw_entries table |
| 3 | Added sync infrastructure |
| 2 | Added relationships and drives |
| 1 | Initial schema (episodes, beliefs, values, goals, notes) |

---

## Indexes

All tables have indexes on:
- `agent_id` - Primary filter
- `created_at` - Time-based queries
- `cloud_synced_at` - Sync queries
- `confidence` - Confidence filtering
- `source_type` - Source filtering
- `is_forgotten` / `is_protected` - Forgetting queries

---

## Notes

- All timestamps are ISO 8601 format with timezone
- JSON arrays stored as TEXT in SQLite
- Soft deletes use `deleted` flag (not physical deletion)
- Sync uses last-write-wins conflict resolution
- Values and Drives are protected from forgetting by default
