# Proposed Raw Layer Architecture

> **Status**: PROPOSED - Revised after peer review (Claude, Gemini, GPT 5.2)
> **Date**: 2026-01-30
> **Context**: Simplifying raw layer to blob-based storage with safety nets

## Executive Summary

The raw layer should be simplified from a structured data model to a **blob-based capture** system. The agent dumps whatever they want into a blob field; the system only tracks housekeeping metadata.

**Key changes from original proposal** (based on peer review):
- Keep `source` field (auto-populated enum for provenance)
- Add FTS5 for keyword search (safety net)
- Add partial index for unprocessed entries (performance)
- Make raw sync **opt-in** (security)
- Adjust size thresholds to warn, not reject

---

## Current vs Proposed

### Current Architecture (DEPRECATED)

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,           -- Structured: validated, length-limited
    timestamp TEXT NOT NULL,
    source TEXT DEFAULT 'manual',    -- Structured: freeform
    processed INTEGER DEFAULT 0,
    processed_into TEXT,             -- JSON array
    tags TEXT,                       -- JSON array - STRUCTURED
    confidence REAL DEFAULT 1.0,     -- Meta-memory field
    source_type TEXT,                -- Meta-memory field
    local_updated_at TEXT,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);
```

**Problems with Current Design:**
1. `content` validation (5000 char limit) creates friction
2. `tags` field implies processing/categorization at capture time
3. `confidence` and `source_type` are meta-memory concepts that don't belong in raw
4. Embeddings generated on raw entries (unnecessary complexity)
5. Separate flat files on disk duplicate storage
6. Raw syncs to cloud by default (security risk)

### Proposed Architecture

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- The actual raw capture (THE BLOB)
    blob TEXT NOT NULL,              -- Unstructured, unvalidated, high limit

    -- Housekeeping metadata (NOT part of the memory)
    captured_at TEXT NOT NULL,       -- When captured (auto)
    source TEXT DEFAULT 'unknown',   -- cli|mcp|sdk|import|unknown (auto-populated)
    processed INTEGER DEFAULT 0,     -- Has been promoted?
    processed_into TEXT,             -- What it became (JSON array of type:id)

    -- Sync metadata
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

-- Partial index for anxiety system queries (HIGH PRIORITY)
CREATE INDEX idx_raw_unprocessed
    ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- FTS5 for keyword search safety net (HIGH PRIORITY)
CREATE VIRTUAL TABLE raw_fts USING fts5(
    blob,
    content=raw_entries,
    content_rowid=rowid
);
```

**Key Changes:**
- `content` + `tags` + `confidence` + `source_type` → single `blob` field
- `timestamp` → `captured_at` (clearer name)
- `source` kept but simplified to auto-populated enum
- Added: FTS5 index for keyword search
- Added: Partial index for unprocessed queries
- Removed: embeddings on raw entries
- Removed: separate flat files on disk (blob IS the flat file)
- Removed: validation/length limits on content
- Changed: sync defaults to OFF for raw

---

## Design Philosophy

### What Raw Captures

The blob captures **whatever the agent dumps**:
- Freeform text
- Markdown formatting
- Context embedded naturally ("Working on kernle auth - just realized...")
- Emotions/reactions ("Frustrated - can't figure out...")
- Partial thoughts
- Code snippets
- Anything

The system **does not parse, validate, or structure** the blob.

### What the System Tracks (Housekeeping Only)

| Field | Purpose | Who Sets It |
|-------|---------|-------------|
| `captured_at` | When captured | System (auto) |
| `source` | How it entered (cli/mcp/sdk/import) | System (auto) |
| `processed` | Has been promoted? | System (on promotion) |
| `processed_into` | Audit trail of what it became | System (on promotion) |

### What Raw Does NOT Have

- ❌ Embeddings (FTS5 keyword search is sufficient)
- ❌ Tags (agent embeds in blob if they want)
- ❌ Confidence (meaningless for raw dumps)
- ❌ Source type (meta-memory concept, belongs in promoted memories)
- ❌ Length validation (agent can dump whatever they need)

### Why Keep `source`?

Peer review (GPT 5.2) raised a valid point: even with blob-based capture, knowing *where* data came from is useful for:
- **Debugging**: "Why is this entry malformed?" → "Ah, it came from import"
- **Metrics**: "How much comes from automated import vs brain dump?"
- **Triage**: "Import entries may need different processing than agent thoughts"

This is **operational metadata**, not semantic metadata. It's auto-populated (zero friction) and enum-constrained (not freeform).

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  AGENT BRAIN DUMP                                           │
│  "Working on kernle sync - just realized the queue needs    │
│   deduplication. Also feeling anxious about context limits" │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  raw_entries TABLE                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ blob: "Working on kernle sync - just realized..."   │   │
│  │ captured_at: "2026-01-29T14:32:15Z"                 │   │
│  │ source: "mcp"  (auto-detected)                      │   │
│  │ processed: 0                                         │   │
│  │ processed_into: NULL                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Also indexed in raw_fts for keyword search                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (later, when agent reviews)
┌─────────────────────────────────────────────────────────────┐
│  PROMOTION (Agent's Decision)                               │
│                                                             │
│  Agent reads blob, decides:                                 │
│  - "deduplication" insight → Episode                        │
│  - "anxious" feeling → Note with emotion tag                │
│                                                             │
│  raw_entries.processed = 1                                  │
│  raw_entries.processed_into = ["episode:abc", "note:xyz"]   │
└─────────────────────────────────────────────────────────────┘
```

---

## Search: FTS5 Safety Net

**Original proposal**: Keyword/grep only, no index.

**Revised proposal**: Add FTS5 for better keyword search.

**Rationale** (from peer review): The design assumes raw is processed quickly, but backlogs happen. Once backlog happens, bare `LIKE` queries become useless. FTS5 is trivial to add and makes keyword search actually useful without adding the complexity of embeddings.

```sql
-- Create FTS index
CREATE VIRTUAL TABLE raw_fts USING fts5(
    blob,
    content=raw_entries,
    content_rowid=rowid
);

-- Search example
SELECT r.* FROM raw_entries r
JOIN raw_fts f ON r.rowid = f.rowid
WHERE raw_fts MATCH 'deduplication'
ORDER BY r.captured_at DESC;
```

**What we're NOT adding**: Semantic search / embeddings. If entries sit long enough to need semantic search, the anxiety system should be screaming—that's the right fix, not better search.

---

## Size Limits and Warnings

**Original proposal**: Reject at 10MB.

**Revised proposal**: Warn at all levels, only reject at extreme sizes.

| Size | Action |
|------|--------|
| < 100 KB | Normal operation |
| 100 KB - 1 MB | Info log: "Large raw entry" |
| 1 MB - 10 MB | Warning: "Very large raw entry - consider processing" |
| 10 MB - 50 MB | Warning: "Extremely large entry" |
| > 50 MB | Reject with helpful message |

**Rationale** (from peer review): Legitimate large entries (code reviews, architecture dumps) should be allowed. The anxiety system handles the "you have too many large entries" case via the raw aging dimension.

---

## Cloud Sync: Opt-In for Raw

**Original proposal**: Sync raw to cloud (like other memory types).

**Revised proposal**: Raw sync is **OFF by default**.

**Rationale** (from peer review - GPT 5.2): Raw blobs are exactly where secrets and accidental sensitive data end up. The "oops I dumped my API key" scenario is far more likely in raw capture than in structured memories.

### Configuration

```yaml
# ~/.kernle/config.yaml
sync:
  episodes: true
  beliefs: true
  notes: true
  values: true
  goals: true
  drives: true
  relationships: true
  playbooks: true
  raw: false        # OFF by default - opt-in only
```

### Code Changes

```python
# In sync logic
def should_sync_table(table_name: str) -> bool:
    if table_name == "raw_entries":
        return config.get("sync.raw", False)  # Default OFF
    return config.get(f"sync.{table_name}", True)  # Default ON
```

---

## Anxiety Integration

The anxiety system continues to track raw entry health:

### Existing: Raw Aging Dimension

```python
def _get_aging_raw_entries(self, age_hours: int = 24):
    """Get raw entries older than age_hours and unprocessed."""
    # Uses the new partial index for performance
    # Query: WHERE processed = 0 AND deleted = 0 AND age > threshold
```

### New: Blob Size Check

```python
def _get_large_raw_entries(self, size_threshold_kb: int = 100):
    """Get raw entries with blobs larger than threshold."""
    # Counts entries where length(blob) > threshold * 1024
    # Returns list for anxiety reporting
```

---

## Migration Path

### Phase 1: Schema Migration

```sql
-- Step 1: Add blob column
ALTER TABLE raw_entries ADD COLUMN blob TEXT;

-- Step 2: Migrate with natural language format (readable by agents)
UPDATE raw_entries SET blob =
    content ||
    CASE WHEN source IS NOT NULL AND source != 'manual' AND source != ''
         THEN ' (from ' || source || ')' ELSE '' END ||
    CASE WHEN tags IS NOT NULL AND tags != '[]' AND tags != 'null' AND tags != ''
         THEN ' [tags: ' ||
              REPLACE(REPLACE(REPLACE(tags, '["', ''), '"]', ''), '","', ', ') ||
              ']'
         ELSE '' END;

-- Step 3: Normalize source to enum values
UPDATE raw_entries SET source =
    CASE
        WHEN source IN ('cli', 'mcp', 'sdk', 'import') THEN source
        WHEN source = 'manual' THEN 'cli'
        WHEN source LIKE '%auto%' THEN 'sdk'
        ELSE 'unknown'
    END;

-- Step 4: Rename timestamp to captured_at (if not already)
-- SQLite doesn't support RENAME COLUMN in older versions
-- May need to recreate table

-- Step 5: Create FTS index
CREATE VIRTUAL TABLE IF NOT EXISTS raw_fts USING fts5(
    blob,
    content=raw_entries,
    content_rowid=rowid
);

-- Step 6: Create partial index for unprocessed queries
CREATE INDEX IF NOT EXISTS idx_raw_unprocessed
    ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- Step 7: Populate FTS index
INSERT INTO raw_fts(raw_fts) VALUES('rebuild');
```

### Phase 2: Code Updates

1. Update `RawEntry` dataclass in `kernle/storage/base.py`
2. Update `save_raw()` in `kernle/storage/sqlite.py`
   - Remove validation on content
   - Auto-detect and set `source` enum
   - Insert into FTS index
3. Update `raw()` method in `kernle/core.py`
   - Remove `tags` parameter
   - Remove `source` parameter (auto-detected)
   - Remove validation
4. Add FTS search method `search_raw_fts()`
5. Update sync logic to respect `sync.raw` config
6. Remove embedding generation for raw entries
7. Remove flat file dual-write mechanism
8. Update CLI commands in `kernle/cli/commands/raw.py`
9. Update MCP tools

### Phase 3: Remove Deprecated Code

1. Remove `_append_raw_to_file()` method
2. Remove `sync_raw_from_files()` method
3. Remove `get_raw_dir()` and `get_raw_files()` methods
4. Remove flat file parsing logic
5. Drop deprecated columns (after verification period):
   - `content` (migrated to `blob`)
   - `tags` (migrated into `blob`)
   - `confidence`
   - `source_type`

---

## API Changes

### Python SDK

**Before:**
```python
raw_id = k.raw(content, tags=["tag1"], source="cli")
```

**After:**
```python
raw_id = k.raw(blob)  # source auto-detected
```

### CLI

**Before:**
```bash
kernle raw capture "content" --tags "tag1,tag2" --source cli
```

**After:**
```bash
kernle raw "content"  # Simple
kernle raw --stdin    # Pipe in content
```

### MCP

**Before:**
```json
{
  "tool": "memory_auto_capture",
  "content": "...",
  "tags": ["..."],
  "source": "..."
}
```

**After:**
```json
{
  "tool": "memory_raw",
  "blob": "..."
}
```

---

## Deprecation Plan

### Timeline

| Phase | Duration | Actions |
|-------|----------|---------|
| 1. Dual support | 4 weeks | Old params accepted with deprecation warning |
| 2. Warning escalation | 2 weeks | Old params log errors but still work |
| 3. Removal | After 6 weeks | Old params raise exceptions |

### Deprecation Warnings

```python
def raw(self, blob: str,
        content: str = None,  # DEPRECATED
        tags: list = None,    # DEPRECATED
        source: str = None    # DEPRECATED
       ) -> str:
    if content is not None:
        warnings.warn(
            "The 'content' parameter is deprecated. Use 'blob' instead.",
            DeprecationWarning
        )
        blob = content
    if tags is not None:
        warnings.warn(
            "The 'tags' parameter is deprecated. Include tags in blob text.",
            DeprecationWarning
        )
        # Optionally append to blob
    if source is not None:
        warnings.warn(
            "The 'source' parameter is deprecated and auto-detected.",
            DeprecationWarning
        )
    # ... rest of implementation
```

---

## Threat Model: Raw Blob Security

### What Should Never Be in Raw

- API keys and secrets
- Passwords and credentials
- PII (names, emails, addresses) unless intentional
- Authentication tokens

### Mitigations

1. **Sync off by default**: Raw doesn't leave local machine unless explicitly enabled
2. **No validation = no false sense of security**: System doesn't claim to sanitize
3. **Anxiety system**: Large blobs get flagged, encouraging review
4. **Agent responsibility**: Agent is the source of all data; they control what's captured

### If Secrets Are Accidentally Captured

```bash
# Find and delete raw entries containing potential secrets
kernle raw list --search "api_key\|password\|secret" --json | \
  jq -r '.[].id' | \
  xargs -I {} kernle raw delete {}

# Or via SDK
for entry in k.list_raw():
    if 'api_key' in entry['blob'].lower():
        k.delete_raw(entry['id'])
```

---

## Summary of Changes from Original Proposal

| Aspect | Original | Revised | Rationale |
|--------|----------|---------|-----------|
| `source` field | Remove | Keep (auto-enum) | Debugging value, zero friction |
| Search | grep/LIKE only | FTS5 index | Safety net for backlogs |
| Unprocessed index | None | Partial index | Performance for anxiety queries |
| Size limits | Reject at 10MB | Warn only, reject at 50MB | Allow legitimate large entries |
| Cloud sync | On by default | **Off by default** | Security (secrets in raw) |
| Migration format | Concatenate | Natural language | More readable |

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Keep `processed_into`? | **Yes** - audit trail value is high |
| Rename table? | **No** - not worth migration complexity |
| Timestamp field name? | **`captured_at`** - clearest |
| Cloud sync for raw? | **Off by default** - security concern |
| Keep `source`? | **Yes** - auto-populated enum for provenance |
