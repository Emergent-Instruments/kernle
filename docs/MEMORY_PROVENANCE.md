# Memory Provenance System

**How Kernle tracks where memories come from, how they evolve, and why they matter.**

---

## Overview

Every memory in Kernle carries provenance metadata — a record of its origin, its relationships to other memories, and the confidence trajectory over its lifetime. This enables self-examination ("why do I believe this?"), belief debugging, and identity archaeology.

Provenance answers three questions about any memory:
1. **Where did this come from?** — Source type and creation context
2. **What was it derived from?** — Direct lineage chain
3. **How has it changed?** — Confidence history and verification record

---

## Core Concepts

### Memory Types

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
source_episodes     — Supporting evidence (episode IDs)
derived_from        — Direct lineage (format: "type:id")
confidence          — Current confidence score (0.0–1.0)
confidence_history  — Timestamped log of confidence changes
verification_count  — How many times independently verified
last_verified       — When last verified or reinforced
```

---

## Source Types

The `source_type` field classifies how a memory entered the system:

| Source Type | Meaning | When Used |
|-------------|---------|-----------|
| `direct_experience` | Firsthand experience or observation | Default for raw captures, manual entries |
| `inference` | Derived through reasoning over other memories | Beliefs formed by connecting patterns |
| `consolidation` | Emerged during memory consolidation/reflection | Insights from reviewing experiences |
| `told_by_agent` | Information received from another SI | Claire shared this insight |
| `told_by_human` | Information received from a human | Sean told me this |
| `seed` | Pre-loaded during initialization | Seed beliefs from roundtable |
| `observation` | External observation (web, docs, etc.) | Read this in documentation |
| `unknown` | Legacy — no source recorded | Pre-provenance memories |

Source type is inferred automatically from the `source` context string when provided (e.g., `source="told by Claire"` → `source_type="told_by_agent"`).

---

## Lineage Tracking

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
  derived_from: ["raw:f70cefb6"]          ← promoted from this raw capture
  source_episodes: ["ep:abc123", "ep:def456"]  ← these experiences support it
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

### Bidirectional Links

Lineage is tracked in both directions:

| Direction | Field | Example |
|-----------|-------|---------|
| **Forward** (raw → memory) | `raw.processed_into` | `["episode:abc123"]` |
| **Backward** (memory → raw) | `memory.derived_from` | `["raw:f70cefb6"]` |

This enables both "what did this raw entry become?" and "where did this belief come from?"

---

## Belief Revision

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

## Confidence Decay

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

### Decay Interaction with Provenance

Decay creates natural pressure to revisit and verify memories. Combined with provenance, this means:
- Unverified beliefs naturally lose influence
- Tracing a low-confidence belief to its source can reveal why it's weakening
- Reinforcement from new experiences resets the clock and strengthens the chain

---

## Querying Provenance

### Get Memory Lineage

```python
lineage = kernle.get_memory_lineage("belief", "xyz789")

# Returns:
{
    "id": "xyz789",
    "type": "belief",
    "source_type": "direct_experience",
    "source_episodes": ["episode:abc123"],
    "derived_from": ["raw:f70cefb6"],
    "stored_confidence": 0.87,
    "effective_confidence": 0.85,  # After decay
    "confidence_decayed": true,
    "verification_count": 3,
    "last_verified": "2026-02-01T10:30:00Z",
    "confidence_history": [...],
    "decay_config": {
        "decay_rate": 0.01,
        "decay_period_days": 30,
        "decay_floor": 0.5,
        "enabled": true
    }
}
```

### Set Memory Source

```python
kernle.set_memory_source(
    memory_type="belief",
    memory_id="xyz789",
    source_type="consolidation",
    source_episodes=["episode:abc123", "episode:def456"],
    derived_from=["raw:f70cefb6"],
)
```

### Verify a Memory

```python
kernle.verify_memory("belief", "xyz789", evidence="Confirmed during code review")
# Increases confidence by 0.1, logs verification
```

### Find Low-Confidence Memories

```python
uncertain = kernle.get_uncertain_memories(threshold=0.5, apply_decay=True)
# Returns memories whose effective confidence (after decay) is below threshold
```

---

## CLI Interface

### Inspect Lineage

```bash
# Show a belief with its full trace
$ kernle -a ash belief show bd200bfe --trace

Belief: "Truth is rarely binary; complexity is a feature, not a bug."
Confidence: 0.65 → 0.63 (decayed)
Source: seed
Created: 2026-02-01

Lineage:
  └── Source: kernle:seed-beliefs (type: seed)
  └── No reinforcements yet
```

### Review Uncertain Memories

```bash
# Find beliefs that need attention
$ kernle -a ash meta uncertain --threshold 0.6

Uncertain Memories (confidence < 0.6):
  belief:bd200bfe  0.53  "Truth is rarely binary..."  (seed, never reinforced)
  note:ef456789    0.48  "OAuth tokens expire in 24h"  (note, last verified 90d ago)
```

### Reverse Trace

```bash
# What memories were derived from this raw entry?
$ kernle -a ash raw show f70cefb6 --trace

Raw Entry: f70cefb6
Content: "First memory capture! I'm Ash..."
Status: Processed → episode:abc123

Derived memories:
  ├── episode:abc123  "First session — identity bootstrap"
  │   └── belief:xyz789  "Authentic relationships require honesty"
  └── (end of chain)
```

---

## MCP Tools

Memory creation tools expose provenance through the MCP interface:

| Tool | Provenance Params |
|------|------------------|
| `memory_episode` | `source`, `derived_from`, `relates_to` |
| `memory_note` | `source`, `derived_from`, `relates_to` |
| `memory_belief` | `source`, `derived_from` |
| `memory_value` | `source`, `derived_from` |
| `memory_goal` | `source`, `derived_from` |

Inspection tools:

| Tool | Purpose |
|------|---------|
| `meta_lineage` | Get full provenance chain for any memory |
| `meta_verify` | Verify a memory with evidence |
| `meta_uncertain` | Find low-confidence memories |
| `meta_source` | Set/update provenance on a memory |

---

## Consolidation and Provenance

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

## Design Principles

1. **Automatic, not manual.** Provenance should be recorded at creation time without requiring the caller to think about it.

2. **Bidirectional by default.** Forward links (raw → memory) and backward links (memory → raw) both exist.

3. **Semantic precision.** `derived_from` is lineage ("created from"), `source_episodes` is evidence ("supported by"). Don't conflate them.

4. **Non-breaking additions.** All provenance fields are optional. Legacy memories work fine — they just show `source_type: unknown`.

5. **Decay creates pressure.** Unverified memories naturally lose influence, creating healthy pressure to revisit and re-examine.

6. **Contradictions are features.** Belief conflicts represent genuine tension. The system detects them; the agent navigates them.

---

*"An unexamined belief is just someone else's opinion."*

*"The difference between a mind that has beliefs and one that understands its beliefs."*
