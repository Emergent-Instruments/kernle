# Kernle Core Memory System — Security Audit Report

**Auditor:** Automated Security Review  
**Date:** 2026-02-01  
**Scope:** Core memory storage layer (sqlite.py, postgres.py, base.py, core.py, metamemory.py)  
**Commit:** HEAD at time of audit  

---

## Executive Summary

The Kernle core memory system demonstrates **strong security fundamentals**. SQL injection is well-mitigated through consistent use of parameterized queries, agent_id isolation is enforced across all WHERE clauses, and path traversal protections are in place. The codebase shows evidence of deliberate security engineering (table name allowlists, input validation, HTTPS enforcement, file permission hardening).

**Critical (P0):** 0 findings  
**High (P1):** 2 findings  
**Medium (P2):** 4 findings  
**Low (P3):** 5 findings  

---

## 1. SQL Injection Analysis

### 1.1 Parameterized Queries — ✅ PASS

All user-supplied data flows through parameterized queries (`?` placeholders) in sqlite.py. Every `save_*`, `get_*`, `update_*`, `delete_*`, and `search` method uses parameter binding for:
- `agent_id`
- `id` / `record_id`
- All content fields (objective, outcome, statement, content, blob, etc.)
- Filter values (status, note_type, since timestamps, etc.)

**Evidence:** Checked all ~60+ SQL operations in sqlite.py. Zero instances of string formatting for user data.

### 1.2 Dynamic Table Names — ⚠️ FINDING

**[P2-SQL-01] Some f-string table interpolations lack validate_table_name() calls**

The codebase uses f-string interpolation for table names in several methods. An `ALLOWED_TABLES` frozenset and `validate_table_name()` function exist (line 100-141 of sqlite.py), and they are called in several critical paths. However, not all paths that use `table_map.get()` also call `validate_table_name()`.

**Affected methods (table comes from hardcoded `table_map` dict — low actual risk):**
- `update_memory_meta()` (line 4662) — builds `f"UPDATE {table} SET ..."` without `validate_table_name()`
- `record_access()` (line 4846) — uses `f"UPDATE {table} ..."` without validation
- `forget_memory()` (line 4953) — **does** call `validate_table_name()` ✓
- `recover_memory()` (line 5009) — does NOT call `validate_table_name()`
- `protect_memory()` (line 5058) — does NOT call `validate_table_name()`
- `get_forgetting_candidates()` — does NOT call `validate_table_name()`
- `get_forgotten_memories()` — does NOT call `validate_table_name()`
- `get_stats()` (line 4367) — iterates hardcoded table list without validation

**Mitigating factors:**
- All table names come from hardcoded `table_map` dicts (not user input)
- The `table_map.get()` pattern returns `None` for unknown keys, causing early return
- No external caller can inject a table name directly

**Risk:** LOW in practice because values come from hardcoded maps. However, defense-in-depth dictates adding `validate_table_name()` consistently.

**Recommendation:** Add `validate_table_name(table)` call after every `table_map.get()` for consistency and defense-in-depth.

### 1.3 LIKE Pattern Injection — ✅ MITIGATED

The `search_raw_fts()` method properly escapes LIKE wildcards using `_escape_like_pattern()` (line 3886). The `_text_search()` method (line 4304) does NOT escape LIKE patterns, but the search_term is wrapped in `%...%` and the risk is limited to unexpected matching behavior (no data exfiltration).

**[P3-SQL-02] _text_search() LIKE patterns not escaped**

`search_term = f"%{query}%"` at line 4304 does not escape `%` or `_` characters in the user's query. This could cause unexpected matching but cannot cause injection since parameterized queries are used.

**Recommendation:** Apply `_escape_like_pattern()` to user queries in `_text_search()` for consistency.

### 1.4 mark_synced() Integer Array — ✅ ACCEPTABLE

`mark_synced()` (line 5316) builds `f"UPDATE sync_queue SET synced = 1 WHERE id IN ({placeholders})"` where `placeholders` is built from `",".join("?" * len(ids))`. The IDs are passed as parameters. This is safe.

### 1.5 Supabase/PostgreSQL (postgres.py) — ✅ PASS

The Supabase client uses its builder pattern (`.eq()`, `.gte()`, `.lte()`, `.upsert()`) which handles parameterization internally. No raw SQL strings are constructed. All queries include `.eq("agent_id", self.agent_id)` for isolation.

---

## 2. Input Validation

### 2.1 core.py String Validation — ✅ GOOD

The `Kernle` class in core.py implements `_validate_string_input()` (line ~380) which:
- Checks type is `str`
- Enforces max_length (configurable per field)
- Strips null bytes (`\x00`)
- Normalizes line endings (`\r\n` → `\n`)

This is called for:
- Episode: objective (1000), outcome (1000), lessons (500 each), tags (100 each)
- Note: content (2000), speaker (200), reason (1000), tags (100 each)
- Belief: statement (1000)
- Value: name (200), statement (1000)
- Goal: title (500), description (1000)
- Checkpoint: task (500), context (1000)
- Agent ID: separate sanitization with alphanumeric-only filter

### 2.2 Agent ID Validation — ✅ GOOD

`_validate_agent_id()` (line ~340):
- Strips whitespace
- Allows only `[a-zA-Z0-9\-_.]` characters
- Enforces max 100 characters
- Rejects empty strings

### 2.3 Note Type Validation — ✅ GOOD

The `note()` method validates `type` against an explicit allowlist: `("note", "decision", "insight", "quote")`.

### 2.4 derived_from Validation — ✅ GOOD

`_validate_derived_from()` validates reference format (`type:id`) and checks type against a known allowlist. This prevents garbage data in provenance chains.

### 2.5 Raw Entry Size Limits — ✅ GOOD

`save_raw()` rejects blobs > 50MB and logs warnings at 1MB and 10MB thresholds. This prevents storage exhaustion.

### 2.6 Budget Parameter Validation — ✅ GOOD

`load()` clamps budget to `[MIN_TOKEN_BUDGET, MAX_TOKEN_BUDGET]` range (100–50000).

### 2.7 Missing Validation — ⚠️ FINDINGS

**[P2-INPUT-01] belief_type not validated against allowlist in storage layer**

While `note_type` is validated in `core.py`, `belief_type` is not validated against an allowlist anywhere. The `belief()` method in core.py accepts arbitrary strings for `type` parameter (mapped to `belief_type`). A user could set `belief_type` to any string.

Looking at the code pattern, valid types appear to be: `fact`, `opinion`, `principle`, `strategy`, `model`, `observation`. But this is not enforced.

**Recommendation:** Add explicit validation similar to note_type:
```python
if belief_type not in ("fact", "opinion", "principle", "strategy", "model", "observation"):
    raise ValueError("Invalid belief type")
```

**[P2-INPUT-02] Confidence values not clamped in all paths**

While `update_episode_emotion()` properly clamps valence/arousal values, confidence values passed to `save_episode()`, `save_belief()`, etc. are not clamped to [0.0, 1.0] at the storage layer. The `verify_memory()` method in metamemory.py does use `min(1.0, ...)` but doesn't enforce `max(0.0, ...)`.

**Recommendation:** Add `confidence = max(0.0, min(1.0, confidence))` clamping in save methods.

**[P3-INPUT-03] source_type not validated against SourceType enum**

`source_type` field accepts arbitrary strings. The `SourceType` enum exists in base.py but is not enforced at the storage layer.

---

## 3. Access Control (Agent Isolation)

### 3.1 Agent ID Filtering — ✅ EXCELLENT

**Every single query** that reads or modifies memory records includes `agent_id = ?` with `self.agent_id`. This was verified across all methods:

| Method Category | agent_id in WHERE | Count Verified |
|---|---|---|
| `get_*` (single record) | ✅ `AND agent_id = ?` | 12/12 |
| `get_*` (list) | ✅ `WHERE agent_id = ?` | 10/10 |
| `save_*` (INSERT) | ✅ Uses `self.agent_id` | 9/9 |
| `update_*` | ✅ `AND agent_id = ?` in WHERE | 8/8 |
| `delete/forget/recover` | ✅ `AND agent_id = ?` | 3/3 |
| `search` (all variants) | ✅ `WHERE agent_id = ?` | 5/5 |
| `record_access` | ✅ `AND agent_id = ?` | 1/1 |
| `load_all` (batch) | ✅ All 7 queries filtered | 7/7 |

### 3.2 Supabase (postgres.py) Agent Isolation — ✅ PASS

All Supabase queries use `.eq("agent_id", self.agent_id)` for reads and include `"agent_id": self.agent_id` in writes.

### 3.3 Save Methods Override agent_id — ✅ GOOD

In `save_episode()`, `save_belief()`, etc., the INSERT always uses `self.agent_id` (not `episode.agent_id`). This means even if a caller passes a manipulated record with a different agent_id, the storage layer forces the correct agent_id.

### 3.4 Sync Pull Agent Isolation — ⚠️ FINDING

**[P1-ACCESS-01] _mark_synced() missing agent_id filter**

`_mark_synced()` (line 5542) executes:
```python
conn.execute(f"UPDATE {table} SET cloud_synced_at = ? WHERE id = ?", (now, record_id))
```

This updates by `id` only, without `AND agent_id = ?`. While the `id` is a UUID and the method is only called internally during sync operations, this is a defense-in-depth gap. If a UUID collision occurred or a cloud service returned a manipulated record_id, it could theoretically mark another agent's record as synced.

**Risk:** LOW — requires UUID collision AND compromised cloud backend. But violates the otherwise-perfect agent isolation pattern.

**Recommendation:** Add `AND agent_id = ?` to the WHERE clause.

### 3.5 Embedding/Vector Search Agent Isolation — ⚠️ FINDING

**[P1-ACCESS-02] Vector search queries vec_embeddings without agent_id filter**

The `_vector_search()` method (line ~4201) queries `vec_embeddings` table which contains embeddings from ALL agents in the same database. The vector search itself does not filter by agent_id:

```python
rows = conn.execute(
    """SELECT id, distance
       FROM vec_embeddings
       WHERE embedding MATCH ?
       ORDER BY distance
       LIMIT ?""",
    (query_packed, limit * 2),
).fetchall()
```

However, after getting vector results, `_fetch_record()` is called which DOES include `AND agent_id = ?`. So cross-agent results are fetched but then filtered out.

**Risk:** MEDIUM — No data is leaked to the caller because `_fetch_record()` filters by agent_id. But:
1. Performance: Agent A's queries process Agent B's embeddings unnecessarily
2. Timing side-channel: An attacker could potentially infer the existence of other agents' records by observing which vector IDs return `None` from `_fetch_record()`

**Recommendation:** Prefix vector IDs with agent_id (e.g., `agent_id:table:record_id`) and add a WHERE filter, OR use a separate vec table per agent.

---

## 4. Data Integrity (Provenance Write-Once Enforcement)

### 4.1 Provenance Fields Are Mutable — ⚠️ FINDING

**[P2-INTEGRITY-01] Provenance fields (source_type, derived_from, source_episodes) can be overwritten after creation**

The `update_memory_meta()` method allows updating:
- `source_type`
- `source_episodes`
- `derived_from`
- `confidence_history`

There is no write-once enforcement. An agent (or malicious caller) could rewrite the provenance of any memory, changing its source_type from "inference" to "direct_experience" and deleting the derived_from chain.

Similarly, `update_episode_atomic()` and `update_belief_atomic()` can overwrite all fields including provenance fields.

**Mitigating factors:**
- `confidence_history` is append-only by convention (verify_memory appends)
- The provenance chain is informational, not used for access control decisions
- In single-agent mode, the agent is both author and consumer of provenance

**Risk:** MEDIUM in multi-agent scenarios; LOW in single-agent mode.

**Recommendation:** Consider:
1. Making `source_type` and initial `derived_from` write-once (only settable when NULL/empty)
2. Making `confidence_history` append-only at the storage layer
3. Adding a `provenance_locked` flag that prevents provenance mutation after initial creation

### 4.2 Optimistic Concurrency Control — ✅ GOOD

`update_episode_atomic()` and `update_belief_atomic()` implement proper optimistic concurrency control with version checking. The `WHERE version = ?` clause prevents lost updates. `VersionConflictError` is raised on conflict.

### 4.3 Soft Delete Integrity — ✅ GOOD

All delete operations use soft delete (`deleted = 1`), and all read queries filter `AND deleted = 0`. Records can be recovered.

### 4.4 Sync Conflict Resolution — ✅ GOOD

Sync conflicts are recorded in `sync_conflicts` table with full snapshots of both versions. Array fields (tags, lessons, derived_from) are merged using set union rather than overwritten, preserving data from both sides. The `MAX_SYNC_ARRAY_SIZE = 500` limit prevents resource exhaustion from merge amplification.

---

## 5. Secrets and Credentials

### 5.1 No Hardcoded Credentials — ✅ PASS

No hardcoded API keys, tokens, passwords, or secrets were found in any of the audited files. All credentials are loaded from:
1. Environment variables (`KERNLE_SUPABASE_URL`, `KERNLE_SUPABASE_KEY`, `KERNLE_AUTH_TOKEN`, etc.)
2. Config files (`~/.kernle/credentials.json`, `~/.kernle/config.json`)

### 5.2 Credential Loading — ✅ GOOD

`_load_cloud_credentials()` (line 793) follows a proper priority chain:
1. `~/.kernle/credentials.json`
2. Environment variables (override file values)
3. `~/.kernle/config.json` (fallback)

### 5.3 Supabase Key Validation — ✅ GOOD

postgres.py validates:
- URL must be HTTPS (line ~139)
- Key must be non-empty and ≥100 characters (basic JWT-like validation)
- URL hostname is validated against known Supabase domains with fallback to reasonable hostname patterns
- SSRF-like URL manipulation is partially mitigated by hostname validation

### 5.4 Raw Sync Default OFF — ✅ GOOD SECURITY DESIGN

`_should_sync_raw()` defaults to OFF, noting that "raw blobs often contain accidental secrets." This is a thoughtful security decision.

### 5.5 File Permissions — ✅ GOOD

`_init_db()` sets `0o600` on the database file and `0o700` on directories. `_sync_beliefs_to_file()` sets `0o600` on flat files.

**[P3-PERMS-01] Not all flat file writes set 0o600 permissions**

`_sync_values_to_file()`, `_sync_goals_to_file()`, `_sync_relationships_to_file()`, and `_append_raw_to_file()` do NOT set restrictive permissions on their output files. Only `_sync_beliefs_to_file()` does.

**Recommendation:** Apply `os.chmod(file, 0o600)` consistently to all flat file writes.

---

## 6. Path Traversal

### 6.1 Database Path Validation — ✅ EXCELLENT

`_validate_db_path()` (line ~675):
- Resolves to absolute path via `.resolve()`
- Checks `is_relative_to()` against safe directories (home, /tmp, tempdir)
- Handles macOS `/var/folders` and `/private/var/folders`
- Rejects paths outside safe boundaries

### 6.2 Checkpoint Directory Validation — ✅ EXCELLENT

`_validate_checkpoint_dir()` in core.py uses the same robust pattern with `is_relative_to()`.

### 6.3 Agent ID Used in Paths — ✅ MITIGATED

Agent ID is used to construct directory paths (`self._agent_dir = self.db_path.parent / agent_id`). The `_validate_agent_id()` method strips all non-alphanumeric characters except `-_.`, preventing `../` traversal.

### 6.4 Raw File Flat File Writes — ✅ MITIGATED

`_append_raw_to_file()` constructs filenames from timestamps (`YYYY-MM-DD.md`), not user input. The directory is within the validated agent directory.

**[P3-PATH-01] Daily raw file name from datetime is safe but should be validated**

The date string is derived from `datetime.fromisoformat()` which produces safe filenames. No issue, but worth noting.

---

## 7. Additional Findings

**[P3-MISC-01] FTS5 query passed directly to MATCH**

In `search_raw_fts()` (line ~3854), the user's query is passed directly to FTS5 MATCH:
```python
AND raw_fts MATCH ?
```
This is parameterized, so no SQL injection. However, FTS5 MATCH has its own query syntax (AND, OR, NOT, NEAR, prefix*). A malformed FTS5 query will raise `sqlite3.OperationalError`, which is caught. Malicious FTS5 syntax could cause performance issues but not data leakage.

**[P3-MISC-02] Checkpoint file size check is good but MAX_CHECKPOINT_SIZE could be configurable**

The 10MB limit on checkpoint files (line ~978) is hardcoded. This is fine but could benefit from being configurable for agents with large working states.

---

## Summary Table

| ID | Severity | Category | Description | Status |
|---|---|---|---|---|
| P1-ACCESS-01 | **P1** | Access Control | `_mark_synced()` missing agent_id filter | Open |
| P1-ACCESS-02 | **P1** | Access Control | Vector search processes all agents' embeddings | Open |
| P2-SQL-01 | **P2** | SQL Injection | Some f-string table names lack validate_table_name() | Open |
| P2-INPUT-01 | **P2** | Input Validation | belief_type not validated against allowlist | Open |
| P2-INPUT-02 | **P2** | Input Validation | Confidence values not clamped in all paths | Open |
| P2-INTEGRITY-01 | **P2** | Data Integrity | Provenance fields are mutable after creation | Open |
| P3-SQL-02 | **P3** | SQL Injection | _text_search() LIKE patterns not escaped | Open |
| P3-INPUT-03 | **P3** | Input Validation | source_type not validated against SourceType enum | Open |
| P3-PERMS-01 | **P3** | Secrets/Perms | Not all flat file writes set 0o600 | Open |
| P3-MISC-01 | **P3** | Misc | FTS5 query syntax can cause perf issues | Open |
| P3-MISC-02 | **P3** | Misc | MAX_CHECKPOINT_SIZE not configurable | Open |

---

## Positive Security Patterns Observed

1. **`ALLOWED_TABLES` frozenset + `validate_table_name()`** — Table allowlist is a strong pattern
2. **Consistent parameterized queries** — Zero string formatting for user data in SQL
3. **`self.agent_id` forced in save methods** — Can't inject a different agent_id via records
4. **`is_relative_to()` for path validation** — Resistant to `../` and symlink attacks
5. **Raw sync OFF by default** — Prevents accidental secret leakage to cloud
6. **Optimistic concurrency control** — Version-based conflict detection prevents lost updates
7. **HTTPS enforcement for Supabase URLs** — Prevents MITM
8. **Secure file permissions** (0o600/0o700) on database and directories
9. **`MAX_SYNC_ARRAY_SIZE`** — Prevents sync merge amplification DoS
10. **`_escape_like_pattern()`** — LIKE wildcard escaping exists (though inconsistently applied)

---

## Recommended Priority Actions

1. **P1:** Add `AND agent_id = ?` to `_mark_synced()` query
2. **P1:** Add agent_id prefix to vector embeddings or filter vec_embeddings by agent scope
3. **P2:** Add `validate_table_name()` to all methods using `table_map.get()` for defense-in-depth
4. **P2:** Add belief_type and source_type allowlist validation
5. **P2:** Clamp confidence to [0.0, 1.0] in all save paths
6. **P3:** Apply `os.chmod(0o600)` to all flat file writes consistently
