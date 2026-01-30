# Feedback Consolidation: Raw Layer Proposal

> **Date**: 2026-01-30
> **Sources**: Claude, Gemini, GPT 5.2
> **Status**: Consolidated recommendations with analysis

---

## Executive Summary

All three reviewers **approved the core proposal** (blob-based raw layer). The disagreements are on edge cases and hardening, not direction.

| Reviewer | Verdict | Key Concern |
|----------|---------|-------------|
| Claude | Approve with minor adjustments | Migration format, size thresholds |
| Gemini | Approved | Embedding duality, missing index |
| GPT 5.2 | Approve with conditions | Security (sync), search fallback |

---

## Consolidated Recommendations

### 1. Keep a Minimal Provenance Field

**Source**: GPT 5.2
**Recommendation**: Add `capture_channel` (enum: `cli|mcp|sdk|import`)

**Rationale**: Even with blob-based capture, knowing *where* data came from is useful for:
- Debugging ("why is this entry malformed?")
- Metrics ("how much comes from automated import vs brain dump?")
- Triage ("import entries may need different processing")

**My Take**: **Agree, but rename to `source`** (already exists, just simplify to enum).

This doesn't violate the blob philosophy—it's not *meaning*, it's *provenance*. The current `source` field is fine; we just need to:
1. Make it an enum (not freeform)
2. Auto-populate it (agent doesn't choose)
3. Keep it minimal: `cli | mcp | sdk | import | unknown`

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    blob TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    source TEXT DEFAULT 'unknown',  -- Auto-populated, enum-like
    processed INTEGER DEFAULT 0,
    processed_into TEXT,
    -- sync fields...
);
```

---

### 2. Add Search Safety Net

**Source**: GPT 5.2, Claude (partial)
**Recommendation**: Add FTS5 or lazy embeddings for older entries

**Options**:
| Option | Complexity | Benefit |
|--------|------------|---------|
| FTS5 on blob | Low | Better keyword search |
| Lazy embeddings (entries >X days old) | Medium | Semantic search for backlog |
| Accept grep-only | None | Keep it simple |

**My Take**: **Add FTS5, skip lazy embeddings**.

FTS5 is trivial to add and makes keyword search actually useful:
```sql
CREATE VIRTUAL TABLE raw_fts USING fts5(blob, content=raw_entries, content_rowid=rowid);
```

Lazy embeddings add complexity and contradict the "raw is ephemeral" philosophy. If entries sit long enough to need semantic search, the anxiety system should be screaming—that's the right fix, not better search.

---

### 3. Make Raw Sync Configurable

**Source**: GPT 5.2
**Recommendation**: Raw sync should be opt-in, not default

**Rationale**: Raw blobs are exactly where secrets and sensitive data end up accidentally. Auto-syncing them to cloud is a security risk.

**My Take**: **Agree strongly**.

Current design syncs everything by default. For raw entries:
- Default: `sync_raw: false` (local only)
- Global toggle in config
- Per-entry override possible but not needed initially

```yaml
# ~/.kernle/config.yaml
sync:
  episodes: true
  beliefs: true
  notes: true
  raw: false  # NEW: opt-in
```

---

### 4. Migration Format

**Source**: Claude, Gemini
**Recommendation**: Use clear delimiters or natural language format

**Options**:
```
# Option A: Delimited (Gemini)
--- CONTENT ---
Auth flow works
--- CONTEXT ---
Tags: auth, success
Source: testing

# Option B: Natural sentence (Claude)
Auth flow works (from testing, tagged: auth, success)

# Option C: JSON header (alternative)
{"tags":["auth","success"],"source":"testing"}
Auth flow works
```

**My Take**: **Option B (natural language)** is best.

Agents will read these blobs later. Natural language is more readable than delimiters, and the metadata is still grep-able. Option C (JSON header) is clever but adds parsing complexity.

Migration SQL:
```sql
UPDATE raw_entries SET blob =
    content ||
    CASE WHEN source != 'manual' THEN ' (from ' || source || ')' ELSE '' END ||
    CASE WHEN tags IS NOT NULL AND tags != '[]' THEN ' [tags: ' ||
        REPLACE(REPLACE(tags, '["', ''), '"]', '') || ']' ELSE '' END;
```

---

### 5. Keep Both MCP Tools

**Source**: GPT 5.2
**Recommendation**: Don't replace `memory_auto_capture`, keep both tools

**Current**:
- `memory_auto_capture` = raw capture + suggestions extraction

**Proposed (GPT's suggestion)**:
- `memory_raw(blob)` = pure dump, no processing
- `memory_auto_capture(blob)` = dump + suggestions

**My Take**: **Disagree slightly**.

The suggestions system is separate from capture. I'd prefer:
- `memory_raw(blob)` = capture only
- `memory_suggestions_extract()` = can be called separately

This keeps capture zero-friction and makes suggestions explicitly agent-initiated. But this is a minor point—either approach works.

---

### 6. Add Missing Index

**Source**: Gemini
**Recommendation**: Partial index on unprocessed raw entries

```sql
CREATE INDEX idx_raw_unprocessed
    ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;
```

**My Take**: **Agree**. This is a correctness issue, not a preference.

The anxiety system queries `processed=0` frequently. Without this index, performance degrades as raw entries accumulate.

---

### 7. Embedding Duality (Local vs Cloud)

**Source**: Gemini
**Recommendation**: Standardize on a canonical embedding model

**Current**:
- Local: 384-dim hash embeddings
- Cloud: 1536-dim OpenAI embeddings

**Issue**: Same search query may return different results locally vs cloud.

**My Take**: **Acknowledge but don't fix now**.

This is a real issue, but:
1. It's a *structured memory* issue, not a raw layer issue
2. Raw entries won't have embeddings anyway (per proposal)
3. The workaround (accept local=faster/dumber, cloud=slower/smarter) is documented and acceptable for now

Future fix: switch local to a small model like `all-MiniLM-L6-v2` that can also run server-side. But this is out of scope for raw layer changes.

---

### 8. Size Thresholds

**Source**: Claude
**Recommendation**: Make warning thresholds configurable or adaptive

**Current proposal**:
- 100KB: Warning
- 1MB: Strong warning
- 10MB: Reject

**My Take**: **Keep fixed thresholds for now, but log not reject**.

Configurable thresholds add complexity. The 100KB threshold is based on "this is bigger than most brain dumps"—legitimate large entries (code reviews, architecture dumps) should still be allowed.

Change:
- 100KB: Info log (not warning)
- 1MB: Warning
- 10MB: Warning (not reject)
- 50MB: Reject with helpful message

The anxiety system handles the "you have too many large entries" case.

---

### 9. Keep `processed_into`

**Source**: All three reviewers
**Recommendation**: Keep for audit trail

**My Take**: **Agree unanimously**.

The lineage value is high, complexity cost is low. When debugging "why is this episode wrong?", tracing back to source raw entry is invaluable.

Treat as best-effort audit metadata, not a state machine.

---

### 10. Documentation Gaps

**Source**: GPT 5.2
**Missing**:
1. Threat model / data sensitivity policy for raw blobs
2. Backward compatibility / versioning plan (MCP tool rename, SDK signature)
3. Operational constraints (volume, retention, backup, perf)
4. Consistency statement (embeddings apply to structured, not raw)

**My Take**: **Valid gaps, add to documentation**.

These are especially important before implementation:
- Threat model: What should never be in raw? How to handle accidental secrets?
- Deprecation plan: How long do old APIs stay supported?
- Operational: What's the expected raw volume? Retention policy?

---

## Revised Proposal Summary

### Schema (Updated)

```sql
CREATE TABLE raw_entries (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,

    -- Core
    blob TEXT NOT NULL,                    -- Unstructured brain dump
    captured_at TEXT NOT NULL,             -- When (auto)
    source TEXT DEFAULT 'unknown',         -- cli|mcp|sdk|import|unknown (auto)

    -- Tracking
    processed INTEGER DEFAULT 0,
    processed_into TEXT,                   -- JSON array of type:id refs

    -- Sync
    local_updated_at TEXT NOT NULL,
    cloud_synced_at TEXT,
    version INTEGER DEFAULT 1,
    deleted INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_raw_unprocessed ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- FTS for keyword search
CREATE VIRTUAL TABLE raw_fts USING fts5(
    blob,
    content=raw_entries,
    content_rowid=rowid
);
```

### Config (New Sync Option)

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
  raw: false        # NEW: opt-in, default off
```

### API (Unchanged from Original Proposal)

```python
# SDK
raw_id = k.raw(blob)  # source auto-detected

# CLI
kernle raw "brain dump text"
kernle raw --stdin < file.txt

# MCP
memory_raw(blob)  # source="mcp" auto-set
```

### Migration (Updated Format)

```sql
-- Step 1: Add blob column
ALTER TABLE raw_entries ADD COLUMN blob TEXT;

-- Step 2: Migrate with natural language format
UPDATE raw_entries SET blob =
    content ||
    CASE WHEN source != 'manual' AND source IS NOT NULL
         THEN ' (from ' || source || ')' ELSE '' END ||
    CASE WHEN tags IS NOT NULL AND tags != '[]' AND tags != 'null'
         THEN ' [tags: ' || REPLACE(REPLACE(REPLACE(tags, '["', ''), '"]', ''), '","', ', ') || ']'
         ELSE '' END;

-- Step 3: Create FTS index
CREATE VIRTUAL TABLE raw_fts USING fts5(blob, content=raw_entries, content_rowid=rowid);

-- Step 4: Add partial index
CREATE INDEX idx_raw_unprocessed ON raw_entries(captured_at)
    WHERE processed = 0 AND deleted = 0;

-- Step 5: (Later) Remove deprecated columns
-- ALTER TABLE raw_entries DROP COLUMN content;
-- ALTER TABLE raw_entries DROP COLUMN tags;
-- ALTER TABLE raw_entries DROP COLUMN confidence;
-- ALTER TABLE raw_entries DROP COLUMN source_type;
```

---

## Action Items

| Item | Priority | Effort |
|------|----------|--------|
| Add FTS5 for raw search | High | Low |
| Add partial index for unprocessed | High | Low |
| Make raw sync opt-in (default off) | High | Medium |
| Keep `source` field (auto-populated enum) | Medium | Low |
| Update migration to natural language format | Medium | Low |
| Add threat model documentation | Medium | Medium |
| Add deprecation plan for API changes | Medium | Low |
| Adjust size thresholds (log not reject) | Low | Low |

---

## Dissenting Opinion

**On "capture_channel" / keeping source field**:

One could argue that even `source` violates the "pure blob" philosophy. The counter-argument: knowing *how* data entered the system is operational metadata, not semantic metadata. It's like a file's creation timestamp—part of the filesystem, not the content.

I lean toward keeping it because the debugging value is high and the friction cost is zero (auto-populated).

---

## Conclusion

The original proposal is sound. These refinements address real edge cases without undermining the core simplification. The most important additions are:

1. **FTS5** - makes keyword search actually usable
2. **Raw sync opt-in** - prevents security footguns
3. **Partial index** - correctness fix for anxiety queries

Everything else is polish.
