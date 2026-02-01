# Storage Layer Integrity & Performance Audit

**Auditor**: Storage Integrity Subagent  
**Date**: 2026-02-01  
**Scope**: `kernle/storage/` — SQLite, Postgres/Supabase, embeddings, base models  
**Files Reviewed**: `base.py`, `sqlite.py` (~6100 lines), `postgres.py` (~2100 lines), `embeddings.py`

---

## Executive Summary

The storage layer is well-architected for a local-first memory system with cloud sync. SQLite is the primary, feature-complete backend; Postgres/Supabase is a secondary cloud backend with significant feature gaps. The codebase shows good security awareness (table name allowlists, LIKE escaping, path validation) and solid schema design. However, there are **critical data integrity risks** in the Postgres backend, **missing indexes for JSON fields**, and **race conditions** in confidence updates on Postgres. Below are 28 findings rated P0–P3.

---

## Findings

### 1. Schema Integrity

#### F1: No Foreign Keys Enforced (SQLite) — P3 (Low)
**Location**: `sqlite.py` SCHEMA definition  
**Issue**: No `FOREIGN KEY` constraints exist between tables. For example, `source_episodes` references episode IDs stored as a JSON array TEXT column, not via FK. `derived_from` similarly has no referential integrity.  
**Impact**: Orphaned references are possible (e.g., a belief's `source_episodes` could reference a deleted episode). However, given the soft-delete pattern (`deleted=0/1`) and JSON array storage, FK enforcement would be impractical here.  
**Recommendation**: Accept as design trade-off. Consider periodic referential integrity checks as a CLI command (e.g., `kernle doctor --check-refs`).

#### F2: `source_entity` Column Missing from SQLite Schema — P2 (Medium)
**Location**: `base.py` Episode/Belief/Note dataclasses define `source_entity`, but `sqlite.py` SCHEMA and `_row_to_*` converters never reference it.  
**Impact**: `source_entity` data is silently dropped on save and always `None` on read. Any code setting `episode.source_entity = "user@example.com"` will lose that data.  
**Recommendation**: Add `source_entity TEXT` column to episodes, beliefs, notes, and relationships tables. Add migration in `_migrate_schema()`.

#### F3: `note_type` Index Missing — P3 (Low)
**Location**: `sqlite.py` SCHEMA — `notes` table  
**Issue**: Notes are filtered by `note_type` in `get_notes()`, but there's no index on `(agent_id, note_type)`.  
**Impact**: Minor at typical scale (< 10K notes), but will cause full scans at scale.  
**Recommendation**: Add `CREATE INDEX IF NOT EXISTS idx_notes_type ON notes(agent_id, note_type);`

#### F4: `tags` Column Not Indexed — P2 (Medium)
**Location**: All tables with `tags TEXT` (JSON array)  
**Issue**: Tag filtering happens in Python after fetching all rows: `episodes = [e for e in episodes if e.tags and any(t in e.tags for t in tags)]`. No index can help with JSON array contains in SQLite.  
**Impact**: Tag-based queries scan all rows then filter in Python. With O(N) episodes and O(T) tags, this is O(N×T). At 10K+ episodes, this becomes slow.  
**Recommendation**: (a) Add an FTS5 index on tags JSON for keyword matching, or (b) create a separate `episode_tags` junction table for efficient lookups.

---

### 2. JSONB / JSON Fields

#### F5: No GIN Index on `derived_from` / `source_episodes` (Postgres) — P1 (High)
**Location**: `postgres.py` — Supabase schema (server-side)  
**Issue**: Previous audit flagged this. `derived_from` and `source_episodes` are stored as JSONB arrays in Postgres but likely lack GIN indexes. Any query like "find all memories derived from episode X" requires full table scans with `@>` operator.  
**Impact**: Lineage traversal and provenance queries will be O(N) per table at scale.  
**Recommendation**: Add `CREATE INDEX idx_episodes_derived_from ON agent_episodes USING GIN (derived_from);` and similarly for all tables with these fields.

#### F6: JSON Serialization/Deserialization is Consistent — P3 (Informational)
**Location**: `sqlite.py` `_to_json()` / `_from_json()`  
**Assessment**: The implementation correctly uses `json.dumps()` / `json.loads()` with proper null handling. `_from_json` catches `JSONDecodeError` and returns `None`. The Postgres backend relies on Supabase client to handle JSON natively (JSONB columns accept Python lists/dicts directly). **No JSON injection vector found** — all JSON fields are serialized through `json.dumps()`, not string concatenation.

#### F7: `confidence_history` Unbounded Growth — P2 (Medium)
**Location**: `base.py` — `confidence_history: Optional[List[Dict[str, Any]]]`  
**Issue**: Every confidence change appends to the array. There is no max length or rotation policy. Over time, a frequently-updated belief could accumulate thousands of history entries.  
**Impact**: (a) Increasingly large JSON blobs serialized/deserialized on every read. (b) Potential SQLite row size issues (though SQLite supports up to 1GB per row by default). (c) Sync payload bloat.  
**Recommendation**: Cap `confidence_history` at ~100 entries with FIFO eviction. Apply on write in `update_memory_meta()`.

#### F8: `steps` Field in Playbooks — No Validation — P2 (Medium)
**Location**: `base.py` `Playbook.steps: List[Dict[str, Any]]`  
**Issue**: The `steps` field accepts arbitrary dict structures with no schema validation. Malformed steps could cause runtime errors in playbook execution code.  
**Impact**: Data quality issue. Steps with missing `action` keys or wrong types will fail at runtime, not at save time.  
**Recommendation**: Add validation in `save_playbook()` to ensure each step has required keys.

---

### 3. Query Performance

#### F9: Text Search Uses Unescaped LIKE Wildcards — P3 (Low)
**Location**: `sqlite.py` `_text_search()` — `search_term = f"%{query}%"`  
**Issue**: The search query is embedded in a LIKE pattern without escaping `%` and `_` characters. A search for `"100% correct"` would match unexpected rows.  
**Contrast**: `search_raw_fts()` correctly uses `_escape_like_pattern()`, but `_text_search()` does not.  
**Impact**: Incorrect search results. Not a security issue (parameterized queries prevent SQL injection), but a correctness issue.  
**Recommendation**: Apply `_escape_like_pattern()` in `_text_search()` as well.

#### F10: `get_forgetting_candidates()` Does N Table Scans — P2 (Medium)
**Location**: `sqlite.py` line ~5100  
**Issue**: Queries each of 7 tables with `LIMIT limit*2`, fetches all rows, computes salience in Python, then sorts. For 7 tables × 200 rows = 1,400 rows processed in Python per call.  
**Impact**: Acceptable at current scale. At 100K+ memories across tables, this becomes expensive. The `ORDER BY created_at ASC` doesn't leverage an ideal index for this use case.  
**Recommendation**: Add a composite index `(agent_id, is_protected, is_forgotten, created_at)` and consider pre-computing salience scores as a materialized column.

#### F11: `load_all()` Opens Multiple Connections — P2 (Medium)
**Location**: `sqlite.py` `load_all()` calls individual `get_*` methods, each opening a separate connection via `_connect()`.  
**Issue**: 7+ separate connection open/close cycles per `load_all()` call.  
**Impact**: Measurable overhead (~5-10ms per connection on macOS). The method exists specifically to batch queries but doesn't achieve batching.  
**Recommendation**: Implement `load_all()` with a single `_connect()` context and inline queries, similar to how `get_stats()` uses a single connection.

#### F12: Postgres Search is Brute-Force — P1 (High)
**Location**: `postgres.py` `search()` method (line ~950)  
**Issue**: Search fetches `limit * 5` rows per table type, iterates in Python, does case-insensitive string matching. This is O(N) across all records with no semantic understanding.  
**Impact**: At scale (10K+ records), this will be very slow and return poor-quality results. The comment says "TODO: Use pgvector for semantic search."  
**Recommendation**: Implement pgvector-based search. This is a significant feature gap that degrades the value of the Postgres backend.

#### F13: Vector Search Doesn't Filter by `is_forgotten` — P2 (Medium)
**Location**: `sqlite.py` `_vector_search()` and `_text_search()`  
**Issue**: Queries filter `deleted = 0` but NOT `is_forgotten = 0`. Forgotten memories will appear in search results.  
**Impact**: Users see tombstoned memories in search, contradicting the forgetting feature's intent.  
**Recommendation**: Add `AND COALESCE(is_forgotten, 0) = 0` to all search queries.

---

### 4. Concurrency

#### F14: Postgres `record_access()` Has Read-Then-Write Race — P1 (High)
**Location**: `postgres.py` `record_access()` (line ~1640)  
**Issue**: The method reads `times_accessed`, increments in Python, then writes back. Two concurrent calls can read the same value and both write `value + 1`, losing an increment.  
**Impact**: The code even documents this: "Under high concurrency, some increments may be lost." For access tracking, approximate counts are acceptable — but the same pattern could be applied elsewhere.  
**Recommendation**: Use Supabase RPC function with `SET times_accessed = times_accessed + 1` (atomic SQL increment). The relationship code already does this via `increment_interaction_count` RPC.

#### F15: SQLite Concurrency — Connection-Per-Operation Pattern — P2 (Medium)
**Location**: `sqlite.py` `_connect()` context manager  
**Issue**: Each operation opens a new `sqlite3.connect()`. SQLite uses file-level locking — concurrent writes from multiple processes (e.g., CLI + MCP server) can cause `SQLITE_BUSY` errors. No WAL mode is explicitly set.  
**Impact**: Under concurrent access (likely when both CLI and MCP server are running), writes may fail with "database is locked." Default journal mode is DELETE, which blocks readers during writes.  
**Recommendation**: (a) Enable WAL mode: `PRAGMA journal_mode=WAL;` in `_init_db()`. (b) Set `PRAGMA busy_timeout=5000;` to auto-retry on lock contention. (c) Consider a connection pool for long-running processes.

#### F16: `update_memory_meta()` Lacks Optimistic Locking — P2 (Medium)
**Location**: `sqlite.py` `update_memory_meta()` (line ~4615)  
**Issue**: Uses `version = version + 1` but doesn't check expected version. Unlike `update_episode_atomic()` and `update_belief_atomic()` which properly implement OCC, `update_memory_meta()` will silently overwrite concurrent changes.  
**Impact**: Concurrent confidence updates could lose data. The version increment prevents sync conflicts but doesn't prevent local race conditions.  
**Recommendation**: Add optional `expected_version` parameter matching the pattern in `update_episode_atomic()`.

---

### 5. Data Migration Safety

#### F17: Schema Migrations Are Additive-Only — P3 (Low, Positive)
**Location**: `sqlite.py` `_migrate_schema()`  
**Assessment**: All migrations use `ALTER TABLE ... ADD COLUMN`. No destructive migrations (DROP COLUMN, type changes). This is safe — migrations cannot lose data. The raw blob migration (`UPDATE raw_entries SET blob = content`) preserves original data in both columns.

#### F18: No Migration Rollback Mechanism — P2 (Medium)
**Location**: `sqlite.py` `_migrate_schema()`  
**Issue**: Failed migrations are caught and logged but the schema version is still bumped to `SCHEMA_VERSION`. There's no way to roll back a partially-applied migration.  
**Impact**: A partial migration failure (e.g., disk full during ALTER TABLE) could leave the schema in an inconsistent state.  
**Recommendation**: (a) Run migrations in a transaction. (b) Only bump schema version after all migrations succeed. (c) Consider backing up the DB before migration (`cp memories.db memories.db.pre-v12`).

#### F19: `schema_version` Table Doesn't Track Individual Migrations — P3 (Low)
**Location**: `sqlite.py` — single `version INTEGER PRIMARY KEY`  
**Issue**: Only stores one version number. Can't determine which specific migrations have been applied if they were applied out of order or partially.  
**Recommendation**: Consider a `migrations_applied` table with individual migration IDs and timestamps for more granular tracking.

---

### 6. Embedding Storage

#### F20: Hash Embeddings Are Not Semantic — P2 (Medium)
**Location**: `embeddings.py` `HashEmbedder`  
**Issue**: The default embedder uses character n-gram hashing, not semantic embeddings. It's deterministic and fast but only captures lexical similarity (exact/near-exact matches).  
**Impact**: "Find memories about dogs" won't match "I adopted a puppy" — no semantic understanding. The code comments acknowledge this: "Not semantically meaningful."  
**Recommendation**: Prioritize integration testing with OpenAI embeddings to ensure the upgrade path works. Consider adding a local model option (e.g., sentence-transformers) for offline semantic search.

#### F21: Embedding Dimension Change Requires Full Rebuild — P2 (Medium)
**Location**: `sqlite.py` — `vec_embeddings` virtual table created with `FLOAT[{dim}]`  
**Issue**: If the embedding dimension changes (e.g., switching from HashEmbedder's 384 to OpenAI's 1536), the virtual table schema is incompatible. There's no migration path — the old table must be dropped and rebuilt.  
**Impact**: Switching embedding providers will silently fail or error on existing databases. No code handles this dimension mismatch.  
**Recommendation**: Check embedding dimension on startup. If mismatched, rebuild the vector table and re-embed all records.

#### F22: Embedding Not Saved for Drives, Relationships, Playbooks — P3 (Low)
**Location**: `sqlite.py` — `_get_searchable_content()` and search methods  
**Issue**: Only episodes, notes, beliefs, values, and goals generate embeddings. Drives, relationships, and playbooks are not embedded or searchable via vector search.  
**Impact**: These record types won't appear in semantic search results.  
**Recommendation**: Add searchable content extraction for drives (`drive_type: focus_areas`), relationships (`entity_name: notes`), and playbooks (`name: description, triggers`).

---

### 7. Memory Limits

#### F23: No Size Limits on Text Fields — P2 (Medium)
**Location**: All tables — `objective TEXT`, `content TEXT`, `blob TEXT`, `statement TEXT`  
**Issue**: No `CHECK` constraints or application-level validation on field lengths. A raw entry's `blob` field is explicitly designed for "no length limits" brain dumps.  
**Impact**: (a) A single massive memory (e.g., pasting an entire codebase) would inflate the DB, slow queries, and generate huge embeddings. (b) Embedding generation for very long texts may exceed token limits on OpenAI. (c) Sync payloads could become enormous.  
**Recommendation**: Add soft limits with warnings: (a) Warn if a single text field exceeds 100KB. (b) Truncate text before embedding (first 8K chars). (c) Add a max blob size for raw entries in config.

#### F24: `MAX_SYNC_ARRAY_SIZE` Only Enforced During Merge — P3 (Low)
**Location**: `sqlite.py` `_merge_array_fields()` — `MAX_SYNC_ARRAY_SIZE = 500`  
**Issue**: Array size is capped during sync merge, but not during normal writes. A single-backend operation could accumulate >500 items in `source_episodes` or `tags`.  
**Impact**: Low — arrays this large are unlikely in normal use. But the inconsistency means sync could silently truncate user data.  
**Recommendation**: Enforce consistent limits on array fields at write time too.

---

### 8. Postgres vs SQLite Parity

#### F25: Postgres Missing Features — P1 (High)
**Location**: `postgres.py` — documented in file header  

| Feature | SQLite | Postgres |
|---------|--------|----------|
| Raw entries | ✅ Full | ❌ `NotImplementedError` |
| Memory suggestions | ✅ Full | ❌ Not implemented |
| FTS5 keyword search | ✅ Full | ❌ Not implemented |
| Semantic vector search | ✅ sqlite-vec | ❌ TODO comment only |
| Batch operations | ✅ Single txn | ❌ N+1 individual calls |
| Sync queue | ✅ Full | ❌ No-op |
| Health check tracking | ✅ Full | ❌ Not implemented |
| Flat file sync | ✅ Full | ❌ Not implemented |
| Optimistic locking | ✅ `update_*_atomic()` | ❌ Not implemented |

**Impact**: Users switching from SQLite to Postgres will lose significant functionality. The Postgres backend is not a drop-in replacement.  
**Recommendation**: Either (a) clearly document Postgres as a sync-only backend, not a primary backend, or (b) implement the missing features.

#### F26: Column Name Mismatches Between Backends — P1 (High)
**Location**: `postgres.py` header documentation  
**Issue**: Fundamental schema differences between backends:
- Episodes: `outcome` (SQLite) vs `outcome_description` (Postgres)
- Notes: `agent_id` (SQLite) vs `owner_id` (Postgres), stored in `notes` vs `memories` table
- Relationships: `entity_name` (SQLite) vs `other_agent_id` (Postgres), `sentiment` vs `trust_level`
- Drives: `updated_at` (SQLite) vs `last_satisfied_at` (Postgres)

**Impact**: Sync between backends requires field-level mapping that's fragile and error-prone. Adding new fields requires updating mapping code in two places. Data that exists in one schema but not the other (e.g., `source_entity` in SQLite, if added) may be lost during sync.  
**Recommendation**: Create a schema migration plan to align column names, or formalize the mapping layer as a tested abstraction.

#### F27: Postgres `save_belief()` Drops Fields — P1 (High)
**Location**: `postgres.py` `save_belief()` (line ~380)  
**Issue**: Only saves `id, agent_id, statement, belief_type, confidence, is_active, is_foundational, local_updated_at, cloud_synced_at, version`. Missing: `source_type`, `source_episodes`, `derived_from`, `last_verified`, `verification_count`, `confidence_history`, `supersedes`, `superseded_by`, `times_reinforced`, `context`, `context_tags`, ALL forgetting fields.  
**Impact**: **Data loss.** Saving a belief to Postgres then reading it back will have `None` for all meta-memory and forgetting fields. A round-trip through sync will strip this metadata.  
**Recommendation**: Add ALL fields to Postgres save methods. This same pattern likely affects `save_value()`, `save_goal()`, and `save_drive()` — each only saves a subset of fields.

#### F28: Postgres `save_episode()` Hardcodes `confidence: 0.8` — P2 (Medium)
**Location**: `postgres.py` `save_episode()` (line ~170)  
**Issue**: `"confidence": 0.8` is hardcoded instead of using `episode.confidence`. Also missing: `source_entity`, `source_type`, `source_episodes`, `derived_from`, `last_verified`, `verification_count`, `confidence_history`, all forgetting fields, context fields.  
**Impact**: Same as F27 — data loss on save.  
**Recommendation**: Use `episode.confidence` and add all missing fields.

---

## Summary by Priority

| Priority | Count | Findings |
|----------|-------|----------|
| **P0** (Critical) | 0 | — |
| **P1** (High) | 5 | F5 (GIN indexes), F12 (Postgres search), F14 (race condition), F25 (feature parity), F26 (column mismatches), F27 (data loss on save) |
| **P2** (Medium) | 12 | F2 (source_entity), F4 (tags index), F7 (unbounded history), F8 (steps validation), F10 (forgetting scans), F11 (load_all connections), F13 (forgotten in search), F15 (SQLite WAL), F16 (meta update locking), F18 (migration rollback), F20 (hash embeddings), F21 (dimension change), F23 (text size limits), F28 (hardcoded confidence) |
| **P3** (Low) | 6 | F1 (no FKs), F3 (note_type index), F6 (JSON OK), F9 (LIKE escaping), F17 (safe migrations), F19 (migration tracking), F22 (missing embeddings), F24 (array limits) |

---

## Recommended Priority Actions

### Immediate (P1)
1. **F27/F28**: Fix Postgres save methods to include ALL fields from dataclasses. This is actively losing data.
2. **F15**: Add `PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;` to SQLite init — simple 2-line fix that prevents lock errors.
3. **F5**: Add GIN indexes on `derived_from` and `source_episodes` JSONB columns in Postgres.
4. **F14**: Replace read-then-write with atomic SQL increment in Postgres `record_access()`.

### Near-Term (P2)
5. **F2**: Add `source_entity` column to SQLite schema + migration.
6. **F13**: Add `is_forgotten = 0` filter to search queries.
7. **F7**: Cap `confidence_history` at 100 entries.
8. **F21**: Add dimension check + rebuild logic for embedding table.
9. **F11**: Implement single-connection `load_all()`.

### Backlog (P3)
10. Add `note_type` composite index.
11. Consider junction tables for tags.
12. Add migration tracking table.
