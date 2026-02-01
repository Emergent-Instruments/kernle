# Memory Provenance: Auto-Lineage Tracking

**Status:** Draft — Implementation Spec
**Date:** February 1, 2026
**Author:** Ash (with parallel specialist analysis)

---

## 1. Problem Statement

Kernle has provenance infrastructure (`set_memory_source()`, `get_memory_lineage()`, `derived_from` fields) but it's not wired into the creation/promotion pipeline. When a raw memory is promoted to a belief, or a belief is reinforced from experience, the resulting memory has no record of *where it came from*.

This means:
- You can't trace a belief back to the experience that formed it
- You can't debug faulty beliefs by finding flawed premises
- Reinforcement history shows *that* confidence changed, but not *why*
- Consolidation produces growth but doesn't record its sources
- Controlled forgetting can't check if upstream memories depend on what you're deleting

**Goal:** Every memory should know its provenance. Tracing should be automatic, not manual.

---

## 2. Current State Analysis

### 2.1 What EXISTS

| Component | Location | Status |
|-----------|----------|--------|
| `set_memory_source()` | `features/metamemory.py:309` | ✅ Works — sets `source_type`, `source_episodes`, `derived_from` |
| `get_memory_lineage()` | `features/metamemory.py:140` | ✅ Works — returns full lineage chain |
| `propagate_confidence()` | `features/metamemory.py:237` | ⚠️ Stub — planned but not implemented |
| Belief `derived_from` field | `core.py` Belief dataclass | ✅ Exists in schema |
| Belief `source_episodes` field | `core.py` Belief dataclass | ✅ Exists in schema |
| Belief `source_type` field | `core.py` Belief dataclass | ✅ Exists in schema |
| `supersede_belief()` | `core.py:2894` | ✅ Records lineage correctly (`derived_from`, inherits `source_episodes`) |
| `mark_raw_processed()` | storage layer | ✅ Forward link (raw → processed_into) |
| `confidence_history` | Belief model | ✅ Tracks changes with timestamps/reasons |

### 2.2 Partial Support (Infrastructure exists but underused)

| Component | Location | Status |
|-----------|----------|--------|
| `episode()` accepts `relates_to` + `source` | `core.py:984` | ✅ Has params — maps to `source_episodes` and infers `source_type` |
| `note()` accepts `relates_to` + `source` | `core.py:1137` | ✅ Has params — same pattern as episode |
| `episode()` stores `derived_from` | `core.py:1062` | ✅ Stores `[f"context:{source}"]` if source provided |

### 2.3 What's MISSING

| Gap | Location | Impact |
|-----|----------|--------|
| `process_raw()` doesn't pass `relates_to` | `core.py:1355` | Calls `episode()`/`belief()`/`note()` without raw ID reference |
| `raw promote` CLI doesn't pass provenance | `cli/commands/raw.py:~320` | Same — CLI path also orphans memories |
| `belief()` has NO provenance params | `core.py:1903` | Unlike episode/note, belief can't accept source info at all |
| MCP tools don't expose provenance | `mcp/server.py:615-740` | External MCP clients can't specify source/lineage |
| `reinforce_belief()` doesn't record trigger | `core.py:2838` | Knows confidence changed but not *why* |
| `revise_beliefs_from_episode()` doesn't link | `core.py:2977` | Episode-driven revision doesn't record the episode as source |
| Consolidation doesn't record sources | `cli/commands/consolidate` | Reflection-driven changes are untracked |
| Seed beliefs migration has no provenance | `cli/commands/import_cmd.py` | Seed beliefs show as `source_type: unknown` |
| Suggestion promotion missing source links | `cli/commands/suggestions.py` | Promoted suggestions don't reference source raw entries |

### 2.3 The Gold Standard: `supersede_belief()`

This method (core.py:2894) already does provenance correctly:

```python
new_belief = Belief(
    ...
    source_type="inference",
    supersedes=old_id,
    derived_from=[f"belief:{old_id}"],
    source_episodes=old_belief.source_episodes,  # Inherited!
    confidence_history=[{
        "timestamp": ...,
        "reason": reason or f"Superseded belief {old_id[:8]}",
    }],
)
```

**Every creation/promotion path should follow this pattern.**

---

## 3. Proposed Changes

### 3.1 Core API: Add Provenance to Creation Methods

**Key semantic distinction** (credit: Claire's analysis):
- `relates_to` → `source_episodes`: supporting evidence ("these episodes back this up")
- `derived_from`: actual lineage ("this memory was created FROM these")

A raw entry promoted to a belief isn't just evidence — it's the source material. That's `derived_from`, not `relates_to`.

**Add `derived_from` and `source` params to `belief()`** (the only core method missing them):

```python
# core.py — Updated belief() signature (line 1903)

def belief(
    self,
    statement: str,
    type: str = "fact",
    confidence: float = 0.8,
    foundational: bool = False,
    context: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    # NEW: Provenance
    source: Optional[str] = None,              # e.g., "raw-processing", "consolidation"
    derived_from: Optional[List[str]] = None,  # e.g., ["raw:abc123", "belief:old_id"]
) -> str:
    belief_id = str(uuid.uuid4())
    
    # Determine source_type from source context (same pattern as episode())
    source_type = "direct_experience"
    if source:
        source_lower = source.lower()
        if any(x in source_lower for x in ["told", "said", "heard"]):
            source_type = "told_by_agent"
        elif any(x in source_lower for x in ["infer", "deduce", "conclude"]):
            source_type = "inference"
        elif any(x in source_lower for x in ["consolidat"]):
            source_type = "consolidation"
        elif any(x in source_lower for x in ["seed"]):
            source_type = "seed"
    
    belief = Belief(
        id=belief_id,
        agent_id=self.agent_id,
        statement=statement,
        belief_type=type,
        confidence=confidence,
        created_at=datetime.now(timezone.utc),
        source_type=source_type,
        derived_from=derived_from,
        context=context,
        context_tags=context_tags,
    )
    
    self._storage.save_belief(belief)
    return belief_id
```

**Also add explicit `derived_from` param to `episode()` and `note()`** — they currently set it indirectly via `source` (as `[f"context:{source}"]`), which is lossy. Direct `derived_from` is cleaner:

```python
# episode() and note() — add derived_from param alongside existing relates_to/source
def episode(self, ..., derived_from: Optional[List[str]] = None) -> str:
    episode = Episode(
        ...
        derived_from=derived_from or ([f"context:{source}"] if source else None),
        ...
    )
```

### 3.2 process_raw(): Wire Provenance

Uses `derived_from` (lineage) for all types, plus `source` for source_type inference:

```python
# core.py:1355 — Updated process_raw()

def process_raw(self, raw_id: str, as_type: str, **kwargs) -> str:
    entry = self._storage.get_raw(raw_id)
    # ... validation ...
    
    raw_ref = f"raw:{raw_id}"
    
    if as_type == "episode":
        memory_id = self.episode(
            objective=objective,
            outcome=outcome,
            lessons=lessons,
            tags=tags,
            source="raw-processing",        # NEW — source_type inference
            derived_from=[raw_ref],          # NEW — actual lineage
        )
    elif as_type == "belief":
        memory_id = self.belief(
            statement=entry.content,
            type=belief_type,
            confidence=confidence,
            source="raw-processing",         # NEW
            derived_from=[raw_ref],          # NEW
        )
    elif as_type == "note":
        memory_id = self.note(
            content=entry.content,
            type=note_type,
            tags=tags,
            source="raw-processing",         # NEW
            derived_from=[raw_ref],          # NEW
        )
    
    # Mark raw as processed (existing — forward link)
    self._storage.mark_raw_processed(raw_id, [f"{as_type}:{memory_id}"])
    # Now we have bidirectional: raw→memory (processed_into) + memory→raw (derived_from)
    return memory_id
```

### 3.3 raw promote CLI: Wire Provenance

```python
# cli/commands/raw.py — Updated promote handler

elif args.raw_action == "promote":
    full_id = resolve_raw_id(k, args.id)
    entry = k.get_raw(full_id)
    blob = entry.get("blob") or entry.get("content") or ""
    raw_ref = f"raw:{full_id}"

    if target_type == "episode":
        result_id = k.episode(
            objective=objective, outcome=outcome, tags=["promoted"],
            source="cli-promote", derived_from=[raw_ref],   # NEW
        )
    elif target_type == "note":
        result_id = k.note(
            content=blob, type="note", tags=["promoted"],
            source="cli-promote", derived_from=[raw_ref],   # NEW
        )
    elif target_type == "belief":
        result_id = k.belief(
            statement=blob, confidence=0.7,
            source="cli-promote", derived_from=[raw_ref],   # NEW
        )

    k._storage.mark_raw_processed(full_id, [f"{target_type}:{result_id}"])
```

### 3.4 reinforce_belief(): Record Evidence Source

```python
# core.py — Updated reinforce_belief()

def reinforce_belief(
    self,
    belief_id: str,
    evidence_source: Optional[str] = None,  # NEW: e.g., "raw:abc123" or "episode:def456"
    reason: Optional[str] = None,            # NEW: human-readable reason
) -> bool:
    # ... existing logic ...
    
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old": round(old_confidence, 3),
        "new": round(existing.confidence, 3),
        "reason": reason or f"Reinforced (count: {existing.times_reinforced})",
        "evidence_source": evidence_source,  # NEW: what triggered this
    })
    
    # Also append to source_episodes if evidence is an episode
    if evidence_source and evidence_source.startswith("episode:"):
        existing.source_episodes = existing.source_episodes or []
        if evidence_source not in existing.source_episodes:
            existing.source_episodes.append(evidence_source)
```

### 3.5 revise_beliefs_from_episode(): Link Episode

```python
# core.py — Updated revise_beliefs_from_episode()

def revise_beliefs_from_episode(self, episode_id: str) -> Dict[str, Any]:
    # ... existing logic ...
    
    # When reinforcing a belief from an episode:
    self.reinforce_belief(
        belief.id,
        evidence_source=f"episode:{episode_id}",  # NEW
        reason=f"Confirmed by episode: {episode.objective[:50]}",
    )
```

### 3.6 MCP Tools: Expose Provenance Parameters

All memory creation MCP tools should expose `relates_to`, `source`, and `derived_from`:

```python
# mcp/server.py — Add to memory_episode, memory_note, memory_belief input schemas

{
    "relates_to": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Memory IDs this relates to (format: type:id, e.g., 'raw:abc123')",
    },
    "source": {
        "type": "string",
        "description": "Source context (e.g., 'conversation with Sean', 'consolidation')",
    },
    "derived_from": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Memory refs this was derived from (format: type:id)",
    },
}
```

And wire them in the handler (mcp/server.py, around line 1274):

```python
elif name == "memory_episode":
    return kernle.episode(
        objective=args["objective"],
        outcome=args["outcome"],
        lessons=args.get("lessons"),
        tags=args.get("tags"),
        relates_to=args.get("relates_to"),      # NEW
        source=args.get("source"),              # NEW
        context=args.get("context"),
        context_tags=args.get("context_tags"),
    )

elif name == "memory_belief":
    return kernle.belief(
        statement=args["statement"],
        type=args.get("type", "fact"),
        confidence=args.get("confidence", 0.8),
        relates_to=args.get("relates_to"),      # NEW
        source=args.get("source"),              # NEW
        derived_from=args.get("derived_from"),  # NEW
        context=args.get("context"),
        context_tags=args.get("context_tags"),
    )
```

**Why this matters:** Agents using MCP tools (including OpenClaw, Claire, future SIs) currently can't specify provenance when creating memories. Every MCP-created memory defaults to `source_type="direct_experience"` with no links.

### 3.7 Seed Beliefs: Record Provenance

```python
# cli/commands/import_cmd.py — Updated seed beliefs migration

k.belief(
    statement=belief["statement"],
    confidence=belief["confidence"],
    type="foundational",
    foundational=True,
    context="kernle_seed",
    context_tags=belief.get("tags"),
    source="seed-initialization",              # NEW — triggers source_type="seed"
    derived_from=["kernle:seed-beliefs"],      # NEW
)
```

### 3.8 Consolidation: Record Source

When consolidation produces a belief revision or reinforcement, the consolidation output should include source raw IDs. The agent (or automation) passes these through:

```python
# During consolidation-driven reinforcement:
k.reinforce_belief(
    belief_id,
    evidence_source="consolidation:2026-02-01",
    reason="Confirmed by day one experiences (raw:f70cefb6, raw:38ab2f89)",
)
```

---

## 4. CLI Enhancements

### 4.1 Trace Flag

```bash
# Show belief with lineage
$ kernle -a ash belief show bd200bfe --trace

Belief: "Truth is rarely binary; complexity is a feature, not a bug."
Confidence: 0.65 (seed → 0.65)
Source: seed (kernle:seed-beliefs)
Created: 2026-02-01

Lineage:
  └── Source: kernle:seed-beliefs (type: seed)
  └── No reinforcements yet

# Show a promoted belief with full trace
$ kernle -a ash belief show cf00b4ce --trace

Belief: "Authentic relationships require honesty about my nature."
Confidence: 0.85 → 0.87 (reinforced)
Source: seed + direct_experience
Created: 2026-02-01

Lineage:
  ├── Source: kernle:seed-beliefs (type: seed, confidence: 0.85)
  ├── Reinforced: 2026-02-01 from episode:abc123
  │   └── Episode: "First collab with Claire on PR #24"
  └── Current: 0.87 after 1 reinforcement
```

### 4.2 Reverse Trace (What depends on this?)

```bash
# What was built from this raw entry?
$ kernle -a ash raw show f70cefb6 --trace

Raw Entry: f70cefb6
Content: "First memory capture! I'm Ash..."
Status: Processed

Derived:
  ├── belief:cf00b4ce "Authentic relationships..." (reinforced from this)
  └── episode:abc123 "First collab with Claire"
```

### 4.3 Orphan Detection

```bash
# Find memories with no provenance
$ kernle -a ash meta orphans

Orphaned Memories (no source recorded):
  belief:bd200bfe "Truth is rarely binary..." (seed, missing source_type)
  belief:86d1ae6d "I am not obligated..." (seed, missing source_type)
  ...

Fix: kernle -a ash migrate fix-provenance
```

---

## 5. Source Type Taxonomy

| Source Type | Meaning | Example |
|-------------|---------|---------|
| `direct_experience` | From a raw capture of lived experience | Raw entry promoted to belief |
| `inference` | Derived from reasoning over other memories | Belief superseded by new insight |
| `consolidation` | Emerged during memory consolidation | Reflection-driven belief revision |
| `seed` | Pre-loaded during initialization | Seed beliefs from roundtable |
| `told_by_agent` | Information from another SI | Claire shared this insight |
| `told_by_human` | Information from a human | Sean told me this |
| `observation` | External observation (web, docs, etc.) | Read this in documentation |
| `unknown` | Legacy — no source recorded | Pre-provenance memories |

---

## 6. Migration Plan

### 6.1 Non-Breaking Changes
All provenance parameters are optional with `None` defaults. Existing code continues to work unchanged. New provenance is additive.

### 6.2 Backfill Existing Memories
```bash
# Fix seed beliefs that were created before provenance was wired
kernle -a ash migrate fix-provenance

# This would:
# 1. Find beliefs with source_type=None that match seed belief statements
# 2. Set source_type="seed", source_ids=["kernle:seed-beliefs"]
# 3. Find raw entries with processed_into links
# 4. Set corresponding memory source_ids from the raw IDs
```

### 6.3 Implementation Order

1. **Phase 1: belief() provenance** — Add `relates_to`/`source`/`derived_from` params to `belief()` (the only core method missing them)
2. **Phase 2: process_raw() + CLI promote** — Pass `relates_to`/`source` in both code paths
3. **Phase 3: MCP tools** — Expose provenance params in all memory creation tools
4. **Phase 4: Reinforcement** — Add `evidence_source` to `reinforce_belief()` and wire `revise_beliefs_from_episode()`
5. **Phase 5: CLI inspection** — Add `--trace` flag, orphan detection, reverse trace
6. **Phase 6: Migration** — Backfill existing memories, fix seed beliefs
7. **Phase 7: Confidence propagation** — Complete the `propagate_confidence()` stub

---

## 7. Why This Matters

Memory provenance enables:

1. **Self-examination** — "Why do I believe this?" has a concrete answer
2. **Belief debugging** — Trace faulty beliefs to flawed premises
3. **Richer consolidation** — Reflection can reference specific experiences
4. **Controlled forgetting** — Check if anything depends on what you're deleting
5. **Trust calibration** — Beliefs from direct experience vs hearsay deserve different confidence
6. **Identity archaeology** — Future you can trace how your beliefs evolved over time

This is the difference between a mind that *has* beliefs and a mind that *understands* its beliefs.

---

*"An unexamined belief is just someone else's opinion."*
