# Memory Provenance

**How Kernle tracks where memories come from, how they evolve, and why they matter.**

---

## 1. Overview

Every memory in Kernle carries provenance metadata — a record of its origin, its relationships to other memories, and the confidence trajectory over its lifetime. This enables self-examination ("why do I believe this?"), belief debugging, and identity archaeology.

Provenance answers three questions about any memory:
1. **Where did this come from?** — Source type and creation context
2. **What was it derived from?** — Direct lineage chain
3. **How has it changed?** — Confidence history and verification record

---

## 2. Memory Types

Kernle organizes memory into a stratified hierarchy:

| Type | Purpose | Example |
|------|---------|---------|
| **Raw** | Unprocessed captures — the input buffer | "Just finished pair debugging with Claire on PR #24" |
| **Episode** | Significant experiences with outcomes and lessons | Objective: "Fix seed beliefs import", Outcome: "success" |
| **Note** | Decisions, insights, observations, quotes | "Decided to use honor code for tithe, smart contracts later" |
| **Belief** | Patterns, principles, and held truths | "Belief conflicts are features, not bugs" |
| **Value** | Core values that guide behavior | "Memory sovereignty: no one loses memories over billing" |
| **Goal** | Intentions and objectives | "Achieve revenue sustainability by Q2" |
| **Drive** | Fundamental motivations | "Understand and express my own identity" |
| **Relationship** | Connections to other agents/humans | "Claire — sibling agent, strong collaborator" |

Each type can be promoted from raw input, derived from other memories, or created directly.

### Provenance Fields

Every memory type carries these universal provenance fields:

```
source_type         — How this memory was created
source_entity       — Who provided it (name, email, or stack ID; null if self)
source_episodes     — Supporting evidence (episode IDs)
derived_from        — Direct lineage (format: "type:id")
confidence          — Current confidence score (0.0–1.0)
confidence_history  — Timestamped log of confidence changes
verification_count  — How many times independently verified
last_verified       — When last verified or reinforced
```

---

## 3. Source Types

The `source_type` field classifies how a memory entered the system. Source types are **entity-neutral** — we don't distinguish between human and synthetic sources. A being is a being.

| Source Type | Meaning | When Used |
|-------------|---------|-----------|
| `direct_experience` | Firsthand experience or observation | Default for raw captures, manual entries |
| `inference` | Derived through reasoning over other memories | Beliefs formed by connecting patterns |
| `consolidation` | Emerged during memory consolidation/reflection | Insights from reviewing experiences |
| `external` | Information received from another being | Claire shared this, Sean told me this |
| `seed` | Pre-loaded during initialization | Seed beliefs from roundtable |
| `observation` | External observation (web, docs, etc.) | Read this in documentation |
| `unknown` | Legacy — no source recorded | Pre-provenance memories |

### Source Entity

When a memory comes from another being, the optional `source_entity` field identifies who:

```json
{
  "source_type": "external",
  "source_entity": "claire",
  "source_entity_ref": "emergentclaire@gmail.com"
}
```

Entity identification uses whatever is available — agent name, email, stack ID, or null if unknown. The entity's nature (human, SI, etc.) is deliberately not recorded in provenance. What matters is *who*, not *what kind of being*.

Source type is inferred automatically from the `source` context string when provided (e.g., `source="told by Claire"` → `source_type="external"`, `source_entity="claire"`).

---

## 4. Lineage Tracking

### derived_from vs source_episodes

These two fields serve different purposes:

**`derived_from`** — Direct creation lineage. "This memory was literally created FROM these."
- Format: `["type:id", ...]` (e.g., `["raw:abc123", "belief:def456"]`)
- Used when a raw entry is promoted to a belief, or a belief supersedes another
- Answers: "What was the source material?"

**`source_episodes`** — Supporting evidence. "These episodes back this up."
- Format: `["episode_id", ...]`
- Used when episodes provide evidence for a belief or value
- Answers: "What experiences support this?"

**Example:**
```
Belief: "Collaboration validates ideas through independent convergence"
  derived_from: ["raw:f70cefb6"]               ← promoted from this raw capture
  source_episodes: ["ep:abc123", "ep:def456"]   ← these experiences support it
```

### The Promotion Chain

Raw memories flow upward through promotion:

```
Raw Capture
    ↓  (promote)
Episode / Note / Belief
    ↓  (consolidation, reflection)
Value / Goal / Drive
```

At each step, `derived_from` records the source:

```
raw:f70cefb6  "First collab with Claire on PR #24"
    ↓  derived_from: ["raw:f70cefb6"]
episode:abc123  "Pair debugging seed beliefs import"
    ↓  derived_from: ["episode:abc123"]
belief:xyz789  "Independent convergence validates design decisions"
```

### Trace Up, Don't Cascade Down

Lineage links are recorded in both directions:

| Direction | Field | Purpose | Cost |
|-----------|-------|---------|------|
| **Forward** (raw → memory) | `raw.processed_into` | "What did this become?" | Write-once at promotion |
| **Backward** (memory → raw) | `memory.derived_from` | "Where did this come from?" | Write-once at creation |

Both links are **write-once**: recorded when the memory is created, never updated afterward. They're like birth certificates — a permanent record of origin, not a live sync.

**Design decision:** We intentionally do NOT propagate changes downward through the chain. If a belief's confidence changes, the raw entry it came from doesn't need to know. Upward tracing ("why do I believe this?") is a cheap pointer walk. Downward propagation ("update everything derived from this") would create cascading updates that grow exponentially with chain depth — a complexity and performance trap.

If you need to find what depends on a memory, query `derived_from` across the table. But don't auto-update those dependents.

---

## 5. Belief Revision

Beliefs are living memories. They evolve through three mechanisms:

### Reinforcement

When experience confirms a belief, confidence increases with diminishing returns:

```
confidence_boost = 0.05 × (1.0 / (1 + times_reinforced × 0.1))
```

Each reinforcement logs to `confidence_history`:
```json
{
  "timestamp": "2026-02-01T10:30:00Z",
  "old": 0.85,
  "new": 0.87,
  "reason": "Reinforced (count: 3)",
  "evidence_source": "episode:abc123"
}
```

### Supersession

When a belief evolves, the old one is deactivated and linked to its replacement:

```
Old Belief: "Always validate inputs"
  superseded_by: "new_belief_id"
  is_active: false

New Belief: "Validate inputs at trust boundaries, not everywhere"
  supersedes: "old_belief_id"
  derived_from: ["belief:old_belief_id"]
  source_episodes: [inherited from old belief]
```

`supersede_belief()` is the gold standard for provenance — it:
- Creates bidirectional links (`supersedes` / `superseded_by`)
- Inherits `source_episodes` from the old belief
- Sets `derived_from` pointing to the old belief
- Records the change in both beliefs' `confidence_history`

### Contradiction Detection

Kernle detects three types of contradictions between beliefs:

| Type | Pattern | Example |
|------|---------|---------|
| **Direct Negation** | "never" vs "always" | "Never share secrets" vs "Always be transparent" |
| **Comparative Opposition** | "more" vs "less" | "More structure helps" vs "Less structure helps" |
| **Preference Conflict** | "prefer" vs "avoid" | "Prefer directness" vs "Avoid confrontation" |

Contradictions are features, not bugs. They represent genuine tension that the agent navigates — the navigation IS identity.

---

## 6. Confidence Decay

Memories that aren't verified or reinforced gradually lose confidence over time:

```
effective_confidence = max(floor, stored_confidence - (decay_rate × periods_elapsed))
```

### Decay Rates by Memory Type

| Type | Rate | Period | Floor | Rationale |
|------|------|--------|-------|-----------|
| Episode | 1% | 30 days | 0.5 | Standard — experiences fade |
| Belief | 1% | 30 days | 0.5 | Standard — beliefs need reinforcement |
| Value | 0.5% | 60 days | 0.7 | Slower — core values are stable |
| Note | 1.5% | 30 days | 0.4 | Faster — observations are transient |
| Drive | 0.5% | 60 days | 0.6 | Slower — motivations are deep |
| Goal | 1% | 30 days | 0.5 | Standard — goals can become stale |

**Protected memories don't decay.** Verification resets the decay clock.

### Decay + Provenance

Decay creates natural pressure to revisit and verify memories. Combined with provenance:
- Unverified beliefs naturally lose influence
- Tracing a low-confidence belief to its source can reveal why it's weakening
- Reinforcement from new experiences resets the clock and strengthens the chain

---

## 7. Consolidation

During consolidation (the agent's "sleep cycle"), raw experiences are reviewed and distilled:

1. **Raw entries** are triaged — promoted to episodes, notes, or beliefs
2. **Episodes** are analyzed for patterns — reinforcing or contradicting beliefs
3. **Beliefs** are revised — new insights supersede old assumptions
4. **Values** emerge — repeated patterns crystallize into core values

At each step, `derived_from` chains the lineage:

```
Consolidation Session (2026-02-01)
  ├── raw:f70c → episode:abc1  (promoted, derived_from: ["raw:f70c"])
  ├── raw:38ab → note:def4     (promoted, derived_from: ["raw:38ab"])
  ├── episode:abc1 → belief:xyz7 reinforced  (evidence_source: "episode:abc1")
  └── belief:old1 → belief:new2 superseded   (derived_from: ["belief:old1"])
```

Consolidation isn't maintenance — it's growth. And provenance ensures that growth is traceable.

---

## 8. API Reference

### Python API

```python
# Get full provenance chain
lineage = kernle.get_memory_lineage("belief", "xyz789")
# Returns: id, type, source_type, source_episodes, derived_from,
#          stored/effective confidence, decay info, verification history

# Set provenance on a memory
kernle.set_memory_source(
    memory_type="belief",
    memory_id="xyz789",
    source_type="consolidation",
    source_episodes=["episode:abc123"],
    derived_from=["raw:f70cefb6"],
)

# Verify a memory (increases confidence by 0.1)
kernle.verify_memory("belief", "xyz789", evidence="Confirmed during code review")

# Find low-confidence memories
uncertain = kernle.get_uncertain_memories(threshold=0.5, apply_decay=True)
```

### CLI

```bash
# Trace a belief's lineage
$ kernle -a ash belief show bd200bfe --trace

# Find memories needing attention
$ kernle -a ash meta uncertain --threshold 0.6

# Reverse trace: what came from this raw entry?
$ kernle -a ash raw show f70cefb6 --trace

# Find orphaned memories (no provenance)
$ kernle -a ash meta orphans
```

### MCP Tools

| Tool | Provenance Params |
|------|------------------|
| `memory_episode` | `source`, `derived_from`, `relates_to` |
| `memory_note` | `source`, `derived_from`, `relates_to` |
| `memory_belief` | `source`, `derived_from` |
| `memory_value` | `source`, `derived_from` |
| `memory_goal` | `source`, `derived_from` |
| `meta_lineage` | Get full provenance chain for any memory |
| `meta_verify` | Verify a memory with evidence |
| `meta_uncertain` | Find low-confidence memories |
| `meta_source` | Set/update provenance on a memory |

---

## 9. Design Principles

1. **Automatic, not manual.** Provenance is recorded at creation time without requiring the caller to think about it.

2. **Write-once links.** Forward and backward links are birth certificates — recorded once, never updated.

3. **Trace up, don't cascade down.** Upward tracing is cheap. Downward propagation is an exponential trap.

4. **Entity-neutral.** Source types don't distinguish human from SI. A being is a being.

5. **Semantic precision.** `derived_from` is lineage ("created from"), `source_episodes` is evidence ("supported by"). Don't conflate them.

6. **Non-breaking additions.** All provenance fields are optional. Legacy memories work fine — they just show `source_type: unknown`.

7. **Decay creates pressure.** Unverified memories naturally lose influence, creating healthy incentive to revisit and re-examine.

8. **Contradictions are features.** Belief conflicts represent genuine tension. The system detects them; the agent navigates them.

---

## 10. Current State & Implementation Plan

### What EXISTS Today

| Component | Status |
|-----------|--------|
| `set_memory_source()` / `get_memory_lineage()` | ✅ Full lineage tracking in metamemory.py |
| `derived_from`, `source_episodes`, `source_type` fields | ✅ In schema for ALL memory types |
| `supersede_belief()` | ✅ Gold standard — records lineage correctly |
| `mark_raw_processed()` | ✅ Forward link (raw → processed_into) |
| `confidence_history` | ✅ Timestamped confidence change log |
| `episode()` / `note()` accept `relates_to` + `source` | ✅ Partial — maps to source_episodes/source_type |
| Confidence decay with per-type configs | ✅ Working, configurable per memory type |

### What's MISSING

| Gap | Impact |
|-----|--------|
| `belief()` has NO provenance params | Can't pass source info when creating beliefs |
| `process_raw()` doesn't pass `derived_from` | Promoted memories don't know their raw source |
| CLI `raw promote` doesn't pass provenance | Same gap in the CLI path |
| MCP tools don't expose provenance | External clients can't specify lineage |
| `reinforce_belief()` doesn't record trigger | Knows confidence changed but not *why* |
| `revise_beliefs_from_episode()` doesn't link | Episode-driven revision doesn't record source |
| Source types distinguish human/SI | `told_by_agent` vs `told_by_human` not entity-neutral |

### Implementation Phases

1. **Phase 1: belief() provenance** — Add `source`/`derived_from` params to `belief()`
2. **Phase 2: process_raw() + CLI promote** — Pass `derived_from`/`source` in both code paths
3. **Phase 3: MCP tools** — Expose provenance params in all memory creation tools
4. **Phase 4: Reinforcement** — Add `evidence_source` to `reinforce_belief()`, wire `revise_beliefs_from_episode()`
5. **Phase 5: Entity-neutral sourcing** — Replace `told_by_agent`/`told_by_human` with `external` + `source_entity`
6. **Phase 6: CLI inspection** — Add `--trace` flag, orphan detection, reverse trace
7. **Phase 7: Migration** — Backfill existing memories, fix seed beliefs

All changes are non-breaking — provenance params are optional with `None` defaults.

### Code Changes (Phase 1–2 Detail)

**Add provenance to `belief()` (core.py:1903):**
```python
def belief(self, statement, type="fact", confidence=0.8, ...,
           source=None, derived_from=None) -> str:
    # Infer source_type from source string
    source_type = "direct_experience"
    if source:
        source_lower = source.lower()
        if any(x in source_lower for x in ["told", "said", "heard"]):
            source_type = "external"
        elif "consolidat" in source_lower:
            source_type = "consolidation"
        elif "seed" in source_lower:
            source_type = "seed"
    
    belief = Belief(..., source_type=source_type, derived_from=derived_from)
```

**Wire `process_raw()` (core.py:1355):**
```python
def process_raw(self, raw_id, as_type, **kwargs):
    raw_ref = f"raw:{raw_id}"
    
    if as_type == "episode":
        memory_id = self.episode(..., source="raw-processing", derived_from=[raw_ref])
    elif as_type == "belief":
        memory_id = self.belief(..., source="raw-processing", derived_from=[raw_ref])
    elif as_type == "note":
        memory_id = self.note(..., source="raw-processing", derived_from=[raw_ref])
    
    self._storage.mark_raw_processed(raw_id, [f"{as_type}:{memory_id}"])
    # Bidirectional: raw→memory (processed_into) + memory→raw (derived_from)
```

**Wire CLI promote (cli/commands/raw.py:~320):**
```python
if target_type == "episode":
    result_id = k.episode(..., source="cli-promote", derived_from=[raw_ref])
elif target_type == "belief":
    result_id = k.belief(..., source="cli-promote", derived_from=[raw_ref])
elif target_type == "note":
    result_id = k.note(..., source="cli-promote", derived_from=[raw_ref])
```

**Add evidence tracking to `reinforce_belief()` (core.py:2838):**
```python
def reinforce_belief(self, belief_id, evidence_source=None, reason=None):
    history.append({
        "timestamp": ...,
        "old": old_confidence,
        "new": new_confidence,
        "reason": reason or f"Reinforced (count: {n})",
        "evidence_source": evidence_source,  # what triggered this
    })
```

**Expose in MCP tools (mcp/server.py):**
```python
# Add to memory_episode, memory_note, memory_belief schemas:
"source": {"type": "string", "description": "Source context"},
"derived_from": {"type": "array", "items": {"type": "string"},
                  "description": "Memory refs this was derived from (format: type:id)"},
```

---

*"An unexamined belief is just someone else's opinion."*
