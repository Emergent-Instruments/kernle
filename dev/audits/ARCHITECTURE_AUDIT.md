# Kernle Architecture Audit Report

**Date**: January 27, 2025
**Auditor**: Architecture Review Subagent
**Version**: 0.1.0
**Scope**: Storage abstraction, sync engine, MCP server, memory model, extensibility, performance

---

## Executive Summary

Kernle demonstrates a **well-thought-out stratified memory architecture** that closely follows the spec in `docs/architecture.md`. The codebase shows significant maturity with good separation of concerns, comprehensive input validation, and a sensible offline-first sync strategy. However, there are several architectural concerns that should be addressed to ensure long-term maintainability and scalability.

**Overall Assessment**: 🟡 **GOOD with caveats** - Solid foundation, needs refinement

---

## 1. Storage Abstraction Analysis

### 1.1 Protocol Design: **B+**

**Strengths:**
- Clean `@runtime_checkable` Protocol pattern using ABC
- Comprehensive dataclasses for all memory types with sensible defaults
- Good separation between interface (`base.py`) and implementations
- Factory function `get_storage()` with auto-detection is elegant

**Concerns:**

#### 🔶 Leaky Abstraction: Inconsistent API between backends

The `SupabaseStorage` and `SQLiteStorage` have **different column names** and table structures:

```python
# SQLiteStorage uses:
"agent_values" table, "statement" column

# SupabaseStorage maps to:
"agent_values" table but with "value_type", "is_foundational" extras
"memories" table for notes (not "notes" table)
"agent_episodes" with different column names
```

**Recommendation**: Create a migration layer or ensure both backends use identical schemas. The SQLite schema should be the canonical source.

#### 🔶 Protocol has 40+ abstract methods

The `Storage` protocol is monolithic. Consider splitting into composable traits:
- `EpisodeStorage`
- `BeliefStorage`
- `MetaMemoryStorage`
- `SyncableStorage`
- `ForgettingStorage`

This would allow lighter implementations for specific use cases.

#### 🔶 Missing Protocol methods in implementations

The `SupabaseStorage` doesn't implement all `Storage` protocol methods:
- Missing: `save_raw`, `get_raw`, `list_raw`, `mark_raw_processed`
- Missing: `save_playbook`, `get_playbook`, `list_playbooks`, `search_playbooks`, `update_playbook_usage`
- Missing: All forgetting methods (`forget_memory`, `recover_memory`, `protect_memory`, etc.)
- Missing: Meta-memory methods (`update_memory_meta`, `get_memories_by_confidence`, etc.)

**Impact**: Runtime errors when using Supabase with features requiring these methods.

### 1.2 Data Model Design: **A-**

**Strengths:**
- Dataclasses are well-structured with comprehensive fields
- Meta-memory fields (confidence, source_type, verification_count) on all types
- Forgetting fields (times_accessed, is_protected, is_forgotten) properly integrated
- Emotional memory fields on Episodes (valence, arousal, tags)

**Concerns:**

#### 🔶 Duplicate field definitions

`Belief` dataclass has duplicate field definitions:
```python
supersedes: Optional[str] = None  # Appears twice
superseded_by: Optional[str] = None  # Appears twice
times_reinforced: int = 0  # Appears twice
is_active: bool = True  # Appears twice
```

This is a bug that should be fixed immediately.

#### 🔶 Optional vs Required semantics unclear

Many fields use `Optional[...]` with non-None defaults which is semantically confusing:
```python
lessons: Optional[List[str]] = None  # vs default []
```

**Recommendation**: Use `field(default_factory=list)` for collection fields.

---

## 2. Sync Engine Evaluation

### 2.1 Offline-First Design: **B**

**Strengths:**
- Queue-based change tracking (`sync_queue` table)
- Connectivity caching to avoid repeated checks
- Last-write-wins conflict resolution is simple and predictable
- Atomic deduplication in `_queue_sync` using `INSERT ON CONFLICT DO UPDATE`

**Concerns:**

#### ✅ ~~Race Condition: Queue deduplication~~ (FIXED)

**Original Issue:**
```python
def _queue_sync(self, conn, table, record_id, operation, payload=None):
    # DELETE then INSERT is not atomic
    conn.execute("DELETE FROM sync_queue WHERE table_name = ? AND record_id = ?", ...)
    conn.execute("INSERT INTO sync_queue ...", ...)
```

**Fix Applied:** The `_queue_sync` method now uses atomic UPSERT with `INSERT ... ON CONFLICT DO UPDATE`:
```python
conn.execute(
    """INSERT INTO sync_queue ...
       ON CONFLICT(table_name, record_id) WHERE synced = 0
       DO UPDATE SET operation = excluded.operation, ...""",
    ...
)
```

This ensures deduplication is atomic and prevents race conditions between concurrent writes.

#### 🟠 Race Condition: Merge during sync

```python
def _merge_generic(self, table, cloud_record, local_record, save_fn):
    # Time comparison is not atomic with the save
    if cloud_time > local_time:
        save_fn()
        # Another operation could happen here before mark_synced
        self._mark_synced(conn, table, cloud_record.id)
```

A local write between `save_fn()` and `_mark_synced()` would be lost.

**Recommendation**: Use database transactions to ensure atomicity.

#### 🔶 Sync doesn't handle deletes properly

```python
if change.operation == "delete":
    # TODO: Handle soft delete in cloud
    self._clear_queued_change(conn, change.id)
```

Deletes are cleared from queue but not propagated to cloud.

#### 🔶 Pull filtering is inconsistent

```python
if table == "episodes":
    cloud_records = getter(limit=1000, since=since)
elif table == "goals":
    cloud_records = getter(status=None, limit=1000)  # No since filter!
else:
    cloud_records = getter(limit=1000)  # No since filter!
```

Most tables don't support `since` filtering, causing full pulls on every sync.

### 2.2 Conflict Resolution: **C+**

**Concerns:**

#### 🟠 Last-write-wins loses data

When local and cloud both modified the same record, one version is completely overwritten. For memory systems, this is particularly problematic—you don't want to lose lessons learned just because another device edited first.

**Recommendation**: Consider field-level merging for certain types:
- Merge `lessons` arrays instead of overwriting
- Combine `tags` sets
- Keep highest `times_accessed`

---

## 3. MCP Server Evaluation

### 3.1 Tool Design: **A-**

**Strengths:**
- Comprehensive input validation with `validate_tool_input()`
- Secure error handling that doesn't leak internal details
- Good tool naming convention (`memory_*`)
- Sensible defaults for all parameters

**Concerns:**

#### 🔶 Tool proliferation

26 tools may be overwhelming for MCP clients with limited tool budgets. Consider:
- Combining list/search tools (`memory_belief_list` + `memory_search`)
- Using a single CRUD tool with `action` parameter

#### 🔶 Missing tools from spec

Per `docs/architecture.md`, these capabilities aren't exposed via MCP:
- **Forgetting**: No tools for `forget_memory`, `recover_memory`, `protect_memory`, `get_forgetting_candidates`
- **Meta-memory**: No tools for `update_memory_meta`, confidence management
- **Playbooks**: No tools for procedural memory CRUD
- **Raw entries**: No tools for raw capture/processing
- **Relationships**: Limited—can create via `memory_note` but no dedicated tools
- **Emotional memory**: No tools to set/query emotional associations

**Recommendation**: Add MCP tools for:
```
memory_forget          - Tombstone a memory
memory_protect         - Mark memory as protected
memory_raw             - Quick capture for later processing
memory_playbook        - Create/use procedural memory
memory_relationship    - Manage relationship memories
memory_confidence      - Update memory confidence
```

### 3.2 Consistency: **B+**

Most tools follow consistent patterns, but:

#### 🔶 Return format inconsistency

Some tools return IDs truncated (`{id[:8]}...`), others return full responses. Standardize on:
- Success: `{"id": "...", "type": "...", ...}`
- Error: Clear error message

#### 🔶 Validation duplication

Input validation happens in both MCP server AND core.py:
```python
# MCP server
sanitized["content"] = sanitize_string(arguments.get("content"), "content", 2000)

# core.py
content = self._validate_string_input(content, "content", 2000)
```

This is good defense-in-depth but adds maintenance burden. Consider centralizing in core.py only.

---

## 4. Memory Model Conformance

### 4.1 Spec Alignment: **B+**

| Layer | Spec | Implementation | Status |
|-------|------|----------------|--------|
| L0: Sensory Buffer | Raw input filtering | Not implemented | ❌ |
| L1: Working Memory | Active context | Checkpoint system | ✅ |
| L2: Episodic | Autobiographical | Episodes table | ✅ |
| L3: Semantic | Facts/concepts | Beliefs table | ✅ |
| L4: Procedural | Skills/habits | Playbooks table | ✅ |
| L5: Values & Beliefs | Core identity | Values table | ✅ |
| L6: Drives | Motivations | Drives table | ✅ |
| L7: Relational | Agent models | Relationships table | ✅ |

**Notable gaps:**
- **Sensory Buffer** (L0) not implemented—could be useful for streaming/real-time capture
- **Counterfactual Memory** mentioned in spec but not implemented
- **Temporal Memory** (circadian awareness) not implemented
- **Memory Triggers** (conditional activation) not implemented

### 4.2 Authority Hierarchy: **Not Implemented**

The spec defines an authority hierarchy (Values > Drives > Procedural > Semantic > Episodic) but the implementation doesn't enforce this. All memory types are treated equally.

**Recommendation**: Add a `resolve_conflict()` method that uses authority levels when memories contradict.

### 4.3 Forgetting System: **A-**

Excellent implementation of forgetting with:
- `is_protected` flag for core memories
- `is_forgotten` tombstoning (not hard delete)
- Salience calculation: `(confidence × log(access+1)) / (age_factor + 1)`
- `get_forgetting_candidates()` with proper exclusions

**Minor concern**: Forgetting happens manually—consider adding automatic decay in background.

---

## 5. Extensibility Assessment

### 5.1 Adding New Storage Backends: **B**

**Process:**
1. Implement all 40+ methods from `Storage` protocol
2. Add to `get_storage()` factory
3. Register in `__all__`

**Pain points:**
- Protocol is large (40+ methods)
- No base class with common logic to inherit
- Schema differences between SQLite/Postgres make it unclear what's canonical

**Recommendation**: Create `BaseStorage` ABC with default implementations for:
- JSON serialization helpers
- Timestamp formatting
- Common query patterns

### 5.2 Adding New Memory Types: **C+**

**Process:**
1. Add dataclass to `base.py`
2. Add table to SQLite schema
3. Add table to Postgres schema (different structure)
4. Implement save/get/list/search in both backends
5. Add to row converters
6. Add to meta-memory methods
7. Add to sync methods
8. Add to forgetting methods
9. Add MCP tools
10. Add CLI commands
11. Add to Kernle core

**This is too many touchpoints!**

**Recommendation**:
- Generate schema from dataclass definitions
- Use generic CRUD for new types
- Consider event-driven architecture for cross-cutting concerns

---

## 6. Performance Analysis

### 6.1 Obvious Bottlenecks

#### 🔴 N+1 Query in `load()` method

```python
def load(self, budget: int = 6000) -> Dict[str, Any]:
    return {
        "checkpoint": self.load_checkpoint(),  # File I/O
        "values": self.load_values(),          # DB query
        "beliefs": self.load_beliefs(),        # DB query
        "goals": self.load_goals(),            # DB query
        "drives": self.load_drives(),          # DB query
        "lessons": self.load_lessons(),        # DB query → then iterates
        "recent_work": self.load_recent_work(),  # DB query
        "recent_notes": self.load_recent_notes(), # DB query
        "relationships": self.load_relationships(), # DB query
    }
```

**9 sequential queries** on session start. Should batch into 1-2 queries with JOINs or a single query with UNION.

#### 🔴 N+1 in search fallback

```python
def _search_fallback(self, query, limit, types):
    for type in types:
        # Each type = separate query
        rows = conn.execute(f"SELECT * FROM {table} WHERE ... LIKE ?", ...)
```

**Recommendation**: Use FTS5 virtual table for text search or UNION ALL.

#### 🟠 Embedding on every save

```python
def save_episode(self, episode):
    # ...
    self._save_embedding(conn, "episodes", episode.id, content)
```

Embedding is synchronous on save. For hash embeddings this is fast, but OpenAI embeddings would block.

**Recommendation**: Queue embeddings for background processing.

#### 🟠 Full table scans in sync pull

```python
cloud_records = getter(limit=1000)  # Pulls ALL records
```

Sync pulls up to 1000 records per table without filtering by `since`. For large datasets, this is O(n) per sync.

### 6.2 Index Coverage: **A-**

Good index coverage on:
- `agent_id` (all tables)
- `created_at`, `confidence`, `source_type`
- `is_forgotten`, `is_protected`
- Sync fields

**Missing useful indexes:**
- `(agent_id, created_at)` composite for time-range queries
- `(agent_id, is_forgotten, is_protected)` for forgetting candidates
- FTS index for text search

---

## 7. Architectural Recommendations

### Immediate (P0 - This Sprint)

1. **Fix duplicate fields in Belief dataclass**
   ```python
   # Remove duplicate declarations of supersedes, superseded_by, times_reinforced, is_active
   ```

2. **Implement missing SupabaseStorage methods**
   - At minimum: throw `NotImplementedError` with clear message
   - Ideally: implement raw entries, playbooks, forgetting

3. **Fix sync race conditions**
   - Wrap queue operations in transactions
   - Use `INSERT OR REPLACE` for queue deduplication

### Short-term (P1 - Next 2 Sprints)

4. **Add batch loading**
   ```python
   def load_all(self) -> Dict[str, Any]:
       # Single query with UNION ALL
       with self._get_conn() as conn:
           rows = conn.execute("""
               SELECT 'value' as type, * FROM agent_values WHERE agent_id = ?
               UNION ALL
               SELECT 'belief' as type, * FROM beliefs WHERE agent_id = ?
               ...
           """)
   ```

5. **Add missing MCP tools**
   - `memory_forget`, `memory_protect`, `memory_raw`, `memory_playbook`

6. **Split Storage protocol into traits**
   ```python
   class CoreStorage(Protocol): ...
   class SyncableStorage(CoreStorage, Protocol): ...
   class ForgettableStorage(CoreStorage, Protocol): ...
   ```

### Medium-term (P2 - Next Quarter)

7. **Unify schema between backends**
   - SQLite schema is more complete; use it as source of truth
   - Generate Postgres migrations from SQLite schema

8. **Add FTS5 for text search**
   ```sql
   CREATE VIRTUAL TABLE memory_fts USING fts5(
       content, type, agent_id, tokenize='porter'
   );
   ```

9. **Implement automatic decay**
   ```python
   def run_decay(self, threshold: float = 0.1):
       candidates = self.get_forgetting_candidates(limit=100)
       for c in candidates:
           if c.score < threshold:
               self.forget_memory(c.record_type, c.record.id, reason="automatic_decay")
   ```

10. **Add field-level merge for sync**
    ```python
    def merge_episode(self, local, cloud):
        merged = Episode(...)
        merged.lessons = list(set(local.lessons or []) | set(cloud.lessons or []))
        merged.tags = list(set(local.tags or []) | set(cloud.tags or []))
        merged.times_accessed = max(local.times_accessed, cloud.times_accessed)
        # ... etc
    ```

---

## 8. Summary Scorecard

| Area | Grade | Notes |
|------|-------|-------|
| Storage Abstraction | B+ | Clean protocol but leaky between backends |
| Sync Engine | B- | Good offline-first but has race conditions |
| MCP Server | A- | Comprehensive but missing some spec tools |
| Memory Model Conformance | B+ | Most layers implemented, some gaps |
| Extensibility | C+ | Too many touchpoints for new types |
| Performance | B | N+1 queries, no batching, but good indexes |

**Overall: B**

Kernle has a solid foundation with thoughtful design choices. The main risks are:
1. Sync race conditions could cause data loss
2. Backend divergence will cause maintenance headaches
3. N+1 queries will hurt at scale

Address the P0 items immediately, then work through P1/P2 systematically.

---

*Audit completed by Architecture Review Subagent*
