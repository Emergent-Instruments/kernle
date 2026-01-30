# Kernle Roadmap

> **Date**: 2026-01-30
> **Purpose**: Track planned improvements and deferred items from architecture review

---

## Immediate (Raw Layer Refactor)

**Status**: Approved, ready for implementation

See [01-PROPOSED-RAW-LAYER.md](./01-PROPOSED-RAW-LAYER.md) for full details.

| Item | Priority | Effort | Status |
|------|----------|--------|--------|
| Add `blob` column, migrate data | High | Medium | Pending |
| Add FTS5 index for keyword search | High | Low | Pending |
| Add partial index for unprocessed queries | High | Low | Pending |
| Make raw sync opt-in (default off) | High | Medium | Pending |
| Keep `source` as auto-populated enum | Medium | Low | Pending |
| Update SDK/CLI/MCP APIs | Medium | Medium | Pending |
| Deprecation warnings for old params | Medium | Low | Pending |
| Remove flat file dual-write | Low | Low | Pending |
| Drop deprecated columns | Low | Low | After verification |

---

## Short-Term (Post Raw Layer)

### 1. Derived-From Linkage

**Source**: Original analysis (before peer review)
**Issue**: When promoting raw → structured memory, `derived_from` field is not set on the new memory.

**Current**:
- Raw entry has `processed_into: ["episode:abc"]` (forward link ✓)
- Episode does NOT have `derived_from: ["raw:xyz"]` (reverse link ✗)

**Fix**: Update `process_raw()` to set `derived_from` on created memories.

```python
def process_raw(self, raw_id, as_type, **kwargs):
    # ... create memory ...

    # Set derived_from on the new memory
    if as_type == "episode":
        episode.derived_from = [f"raw:{raw_id}"]
    # ... etc
```

**Effort**: Low
**Priority**: Medium (completes provenance tracking)

---

### 2. Promotion Helper

**Source**: Gemini review
**Suggestion**: Create a helper that makes promotion easier for LLM agents.

```python
# Current (manual)
raw_id = k.raw("Debugging auth - learned JWKS is better than API keys")
# ... later ...
episode_id = k.process_raw(raw_id, "episode",
    objective="Debug auth flow",
    outcome="Discovered JWKS superiority"
)

# Proposed helper
episode_id = k.promote(raw_id, to="episode",
    extraction_prompt="Extract objective and outcome from this thought"
)
# Internally runs LLM to extract structured fields
```

**Effort**: Medium
**Priority**: Medium (improves agent ergonomics)
**Consideration**: This adds LLM dependency to core library. May be better as an optional extension.

---

### 3. Semantic Anxiety Trigger

**Source**: Gemini review
**Issue**: Anxiety system triggers on volume (time, count) but not on semantic issues.

**Current dimensions**:
1. Context pressure (tokens)
2. Unsaved work (time since checkpoint)
3. Consolidation debt (unreflected episodes)
4. Raw aging (unprocessed entries)
5. Identity coherence (self-model confidence)
6. Memory uncertainty (low-confidence beliefs)

**Proposed addition**: **Contradiction detection**

```python
# Add to anxiety calculation
contradictions = k.find_contradictions()
if len(contradictions) > 0:
    dimensions["semantic_coherence"] = {
        "score": min(100, len(contradictions) * 20),
        "detail": f"{len(contradictions)} potential contradictions detected"
    }
```

**Effort**: Medium
**Priority**: Low (nice-to-have, not blocking)
**Consideration**: `find_contradictions()` already exists but isn't wired into anxiety.

---

## Medium-Term (Deferred from Peer Review)

### 4. Embedding Duality Resolution

**Source**: Gemini review
**Issue**: Local uses 384-dim hash embeddings, cloud uses 1536-dim OpenAI embeddings. Same query may return different results.

**Current state**:
- Local: Fast, offline-capable, less semantic understanding
- Cloud: Slower, requires API, better semantic understanding
- This is documented and "acceptable" for now

**Future options**:

| Option | Pros | Cons |
|--------|------|------|
| Standardize on small local model (e.g., `all-MiniLM-L6-v2`) | Consistent results | Requires model download, slower than hash |
| Keep duality, document clearly | Simple | Semantic schism |
| Use cloud embeddings for local too | Best quality | Requires API, online-only |

**Recommendation**: If this becomes a real user pain point, standardize on a small local model that can also run server-side. Until then, document the tradeoff.

**Effort**: High
**Priority**: Low (not causing reported issues)

---

### 5. Multi-Agent Support

**Source**: Gemini review
**Issue**: Kernle is currently "solipsistic"—one agent, one mind.

**Future need**: Enterprise agents may need shared memories:
- Share `episodes` (how we solved this bug)
- Keep `identity/drives` private (maintain distinct personas)

**Current foundation**: `agent_id` column in every table is the right starting point.

**Future work**:
- `shared_group_id` column or ACL table
- Permission model for shared vs private memories
- Sync logic for shared memories

**Effort**: High
**Priority**: Low (no current users requesting this)

---

### 6. Operational Documentation

**Source**: GPT 5.2 review
**Missing documentation**:

| Doc | Purpose | Priority |
|-----|---------|----------|
| Expected raw volume | How many entries/day is normal? | Medium |
| Retention policy | When should old raw be cleaned? | Medium |
| Backup/restore procedures | How to backup agent memory? | Medium |
| Performance targets | What's acceptable latency? | Low |

**Effort**: Low (documentation only)
**Priority**: Medium (important for production use)

---

## Long-Term (Future Considerations)

### 7. Real-Time Sync

**Current**: Manual or periodic sync triggers.
**Future**: WebSocket-based real-time sync for multi-device agents.

**Effort**: High
**Priority**: Low

---

### 8. Memory Compaction

**Current**: Tombstoned memories stay in database forever.
**Future**: Periodic compaction to archive or truly delete old tombstoned memories.

**Effort**: Medium
**Priority**: Low (storage is cheap)

---

### 9. Memory Export Formats

**Current**: Markdown and JSON export.
**Future**: Additional formats (OPML, Roam JSON, Obsidian markdown).

**Effort**: Low per format
**Priority**: Low (nice-to-have for interop)

---

## Completed Items

| Item | Completed | Notes |
|------|-----------|-------|
| Raw layer architecture review | 2026-01-30 | See peer review consolidation |
| Database schema documentation | 2026-01-29 | See 02-DATABASE-SCHEMA.md |
| Data flow diagrams | 2026-01-29 | See 03-DATA-FLOW-DIAGRAMS.md |
| External integrations documentation | 2026-01-29 | See 04-EXTERNAL-INTEGRATIONS.md |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-30 | Initial roadmap from architecture review |
