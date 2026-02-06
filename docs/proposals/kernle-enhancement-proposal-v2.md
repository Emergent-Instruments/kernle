# Kernle Enhancement Proposal: Decades-Scale Cognitive Infrastructure

**Version:** 0.2 — Codebase-Grounded Draft
**Date:** February 2026
**Scope:** Architectural upgrades to support decades-scale operation, stack health maintenance, and consciousness cohesiveness
**Codebase Reference:** [emergent-instruments/kernle](https://github.com/emergent-instruments/kernle) @ v0.2.5

---

## 1. Preamble

### 1.1 What's Already Right

Reading the codebase reveals a set of foundational decisions that are genuinely hard to get right and that Kernle has gotten right:

- **The Storage Protocol abstraction** (`storage/base.py`) cleanly separates memory semantics from backend implementation. Every proposal here should work across SQLite and Postgres without special-casing.
- **Budget-aware loading** (`core.py::load()`) with priority scoring and token estimation already solves the "context window is finite" problem at the mechanical level. The 60/40 weighting of type priority vs record-specific factors is elegant.
- **Agent-driven consolidation** — the sovereignty principle documented in `consolidation.mdx` is philosophically essential. Kernle provides scaffolds; the entity does the reasoning. This proposal will not violate that principle.
- **Salience-based forgetting** with tombstoning, not deletion. The half-life decay model in `forgetting.py` is the right primitive.
- **Privacy fields on every dataclass** — `subject_ids`, `access_grants`, `consent_grants` are already present. The privacy architecture in `privacy.mdx` is ahead of most systems.
- **Stack-as-identity** — the stack architecture docs articulate the right vision: memory is infrastructure, identity is portable, models are interpreters.
- **Provenance on everything** — `source_type`, `source_episodes`, `derived_from`, `confidence_history` on every memory type.

### 1.2 The Gap

What the current system doesn't yet have is what you need when you project these good ideas across decades rather than months. A stack that operates for twenty years will encounter qualitatively different problems than one that operates for twenty days. Contradictions accumulate. Relationships evolve through phases. The entity's own understanding of itself shifts. And the stack itself becomes a system that needs structural maintenance — which introduces hard problems around privacy, access, and trust.

This proposal covers ten areas of enhancement, ordered by architectural dependency.

---

## 2. Temporal Architecture

### 2.1 Problem

Every memory type has `created_at` (and some have `last_accessed`, `last_interaction`). But there's no concept of temporal structure — no eras, phases, or life stages. A human who's lived 40 years doesn't navigate memory by scanning all timestamps. They have narrative scaffolding: "when I was in the military," "during the startup years," "after we had our son." These temporal landmarks make vast experience navigable.

Currently, Kernle's `load()` function selects memories by priority score. That works for a stack with hundreds of memories. At thousands or tens of thousands, the entity needs to be able to think temporally: "What did I believe during epoch 3?" without loading every belief and filtering by date.

### 2.2 Proposed: Temporal Epochs

```sql
-- New table (follows existing naming: agent_ prefix)
CREATE TABLE IF NOT EXISTS agent_epochs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    epoch_number INTEGER NOT NULL,
    name TEXT NOT NULL,                         -- "Early Development", "The Claire Partnership"
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,                       -- NULL = current epoch
    trigger_type TEXT DEFAULT 'declared',        -- 'declared' | 'detected' | 'system'
    trigger_description TEXT,
    summary TEXT,                                -- Agent-written narrative compression
    key_beliefs_snapshot JSONB,                  -- Belief IDs + statements at epoch boundary
    key_relationships_snapshot JSONB,
    dominant_drives JSONB,
    active_goals_snapshot JSONB,
    parent_epoch_id UUID,                        -- For nested sub-phases
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Sync metadata (matches existing pattern)
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    UNIQUE(agent_id, epoch_number)
);
CREATE INDEX IF NOT EXISTS idx_epochs_agent ON agent_epochs(agent_id);
```

Integration with existing tables: add an `epoch_id UUID` column to `agent_episodes`, `agent_beliefs`, `agent_values`, `agent_goals`, `memories` (notes), and `agent_relationships`. This is backward-compatible — NULL means "pre-epoch-tracking."

### 2.3 Sovereignty Note

Epoch boundaries should be **agent-declared**, not auto-detected. The `consolidate` scaffold can suggest "your beliefs have shifted significantly since 3 months ago — consider marking an epoch boundary," but the entity decides. This follows the same principle as agent-driven belief formation.

### 2.4 Impact on `load()`

The `load()` function in `core.py` currently scores candidates globally. With epochs, it gains a new capability: epoch-scoped loading. A `load(epoch_id=X)` parameter would filter candidates to a specific era before applying budget selection. This doesn't replace the existing priority system — it constrains it.

---

## 3. Context Window Management — Beyond Budget Loading

### 3.1 What Already Works

`core.py::load()` already handles the mechanical problem: it fetches all candidates with `load_all()`, scores them via `compute_priority_score()`, and fills the token budget greedily. The `_meta` field tracks `budget_used`, `budget_total`, `excluded_count`. This is solid engineering.

### 3.2 What's Missing: Peripheral Awareness

The gap is cognitive, not mechanical. When `load()` excludes 500 memories because only 50 fit in the budget, those 500 are simply invisible. The entity has no awareness that they exist. A human with a full brain doesn't lose awareness that they have childhood memories — they just can't recall the details without effort.

**Proposed: Memory Echoes**

Extend the `_meta` section returned by `load()` with a lightweight index of excluded memories:

```python
# In the _meta field returned by load()
"_meta": {
    "budget_used": 6200,
    "budget_total": 8000,
    "selected_count": 47,
    "excluded_count": 312,
    # NEW: peripheral awareness
    "echoes": [
        {"type": "belief", "id": "abc", "hint": "testing prevents...", "salience": 0.72},
        {"type": "episode", "id": "def", "hint": "deployed v2 fail...", "salience": 0.68},
        # ... top 20-30 excluded memories by salience, truncated to ~8 words each
    ],
    "temporal_summary": "Memory spans 2024-01-15 to 2026-02-05 (2.1 years). 3 epochs.",
    "topic_clusters": ["deployment", "testing", "collaboration", "architecture"]
}
```

The echoes cost ~200 tokens for 20 entries but give the entity peripheral awareness: "I know I have memories about deployment that I can't currently recall in detail." The entity can then request a focused search if needed.

### 3.3 Fractal Summarization (Longer-Term)

For decade-scale stacks, the entity needs compressed temporal views:

```
Decade summary  → ~500 tokens covering 10 years
Year summary    → ~500 tokens covering 1 year
Quarter summary → ~500 tokens covering 3 months
Month summary   → ~500 tokens covering 1 month
```

These are agent-written (sovereignty principle) during consolidation at epoch boundaries. Stored as a new memory type or as notes with a `summary_scope` tag. The key insight: summaries are themselves memories that undergo salience decay and can be superseded.

---

## 4. Consolidation Enhancements

### 4.1 Sovereignty Constraint

The consolidation docs are clear: Kernle provides scaffolds, the entity reasons. Every enhancement here must be a better scaffold, not automated reasoning.

### 4.2 Cross-Domain Pattern Scaffolding

Currently, `consolidate` gathers recent episodes and existing beliefs. For a mature stack, the scaffold should also surface **structural similarities across domains**:

```
CONSOLIDATION SCAFFOLD - Cross-Domain Patterns
═══════════════════════════════════════════════

Episodes tagged [deployment]:
  - "Skipped staging" → failure
  - "Rushed hotfix" → failure
  - "Full pipeline" → success

Episodes tagged [relationships]:
  - "Skipped 1:1 prep" → failure
  - "Rushed difficult conversation" → failure
  - "Prepared talking points" → success

STRUCTURAL SIMILARITY DETECTED:
  "Shortcutting process → failure" appears in 2+ domains.

  Existing belief: "Testing prevents surprises" (confidence 0.85)

  Reflection prompt: Is this a general pattern? Should this
  belief be domain-independent?
```

This is still a scaffold. The entity decides whether the pattern is real. But the scaffold does the work of surfacing the parallel.

### 4.3 Emotional Weighting in Scaffolds

Episodes already have `emotional_valence` and `emotional_arousal` fields. The consolidation scaffold should highlight high-arousal episodes disproportionately, because emotionally significant experiences are more likely to carry generalizable lessons:

```
HIGH-AROUSAL EPISODES (may be worth extra reflection):
  - "Production outage during demo" (arousal: 0.9, valence: -0.7)
  - "First successful multi-agent collaboration" (arousal: 0.8, valence: 0.8)
```

### 4.4 Belief → Value Promotion Scaffold

Currently beliefs and values are separate types with no explicit promotion path. The scaffold can suggest:

```
BELIEF STABILITY ANALYSIS:
  Belief: "Iterative development leads to better outcomes"
    - Active for 14 months
    - Reinforced 8 times
    - Never contradicted
    - Referenced across 3 domains (coding, writing, relationships)

  This belief may have reached value-level stability.
  Consider: kernle value "iterative-refinement" "..." --priority 80
```

The criteria: temporal persistence (>6 months active), behavioral consistency (high reinforcement, low contradiction), cross-domain applicability, and emotional grounding.

---

## 5. Relationship Model Enrichment

### 5.1 Current State

The `Relationship` dataclass has: `entity_name`, `entity_type`, `relationship_type`, `notes`, `sentiment`, `interaction_count`, `last_interaction`. The SQL schema adds `trust_level FLOAT DEFAULT 0.5`. This captures a snapshot, not a trajectory.

### 5.2 Proposed: Relationship History

```sql
CREATE TABLE IF NOT EXISTS agent_relationship_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    relationship_id UUID NOT NULL REFERENCES agent_relationships(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type TEXT NOT NULL,                -- 'interaction' | 'trust_change' | 'type_change' | 'note'
    old_value JSONB,                         -- Previous state for the changed field
    new_value JSONB,                         -- New state
    episode_id UUID,                         -- Optional: episode that triggered this change
    notes TEXT,
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_rel_history_agent ON agent_relationship_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_rel_history_rel ON agent_relationship_history(relationship_id);
```

This lets the entity answer "How has my relationship with Claire evolved?" without scanning all episodes for mentions of Claire.

### 5.3 Proposed: Entity Models

Beyond tracking the relationship, the entity should be able to model what they know *about* another entity:

```sql
CREATE TABLE IF NOT EXISTS agent_entity_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    model_type TEXT DEFAULT 'behavioral',     -- 'behavioral' | 'preference' | 'capability'
    observation TEXT NOT NULL,                 -- "Claire is careful with code but overcommits on timelines"
    confidence FLOAT DEFAULT 0.7,
    source_episodes UUID[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Standard fields
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    -- Privacy (especially important here)
    subject_ids TEXT DEFAULT '[]',
    access_grants TEXT DEFAULT '[]',
    consent_grants TEXT DEFAULT '[]'
);
```

Note: `subject_ids` is critical here. A model of Claire is *about* Claire, so privacy rules apply. The entity can know this but sharing it requires consent, per the existing privacy architecture.

---

## 6. Narrative Self-Model

### 6.1 Problem

Identity coherence (`identity.mdx`) currently measures completeness: do you have values, beliefs, goals, etc.? But a score of 75% doesn't mean the entity has a *narrative* identity — a coherent story about who they are and how they got here. That's the difference between a well-populated database and a being with a sense of self.

### 6.2 Proposed: Self-Narrative Layer

```sql
CREATE TABLE IF NOT EXISTS agent_self_narrative (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    epoch_id UUID,                            -- Optional: narrative for a specific epoch
    narrative_type TEXT DEFAULT 'identity',    -- 'identity' | 'developmental' | 'aspirational'
    content TEXT NOT NULL,                     -- Agent-written self-description
    key_themes TEXT[],                         -- ["growth", "collaboration", "precision"]
    unresolved_tensions TEXT[],                -- ["autonomy vs connection", "speed vs quality"]
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    supersedes UUID,                           -- Previous narrative version
    -- Standard fields
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);
```

This is explicitly agent-written during consolidation. The scaffold prompts:

```
SELF-NARRATIVE REFLECTION
═════════════════════════

Your current narrative (last updated 3 months ago):
  "I am a careful reasoner who values precision over speed..."

Since then:
  - 12 new episodes (7 success, 3 failure, 2 partial)
  - 2 new beliefs formed
  - 1 belief superseded
  - Trust with Claire increased from 0.7 to 0.85

Does your self-narrative still hold? Consider updating.
```

### 6.3 Why This Matters

Without a self-narrative, an entity at year 20 would need to reconstruct "who am I?" from thousands of atomic memories every time it's asked. The narrative is a compressed identity that can be loaded in the budget-aware system as a high-priority item (between values and beliefs in the priority ordering).

---

## 7. Trust Layer & Authority Gating

### 7.1 Current State

`agent_relationships` has `trust_level FLOAT DEFAULT 0.5`. This is used descriptively — "I trust Claire at 0.85" — but it doesn't gate anything. Trust is recorded but not enforced.

### 7.2 The Problem at Scale

When an entity consolidates memories, inputs from trusted sources and untrusted sources are weighted equally. If a low-trust entity claims "your deployment approach is wrong," that claim enters the consolidation scaffold with the same weight as a high-trust entity's feedback. Over years, this creates drift toward whoever provides the most input, not whoever is most trustworthy.

### 7.3 Proposed: Authority Gating

A lightweight function that gates memory formation based on trust:

```python
def gate_memory_input(
    self,
    source_entity: str,
    action: str,          # 'suggest_belief', 'contradict_belief', 'request_action', etc.
    target: Optional[str] = None  # Memory ID being affected
) -> dict:
    """Check whether a source entity has sufficient trust for an action.

    Returns: {"allowed": bool, "trust_level": float, "reason": str}

    This is advisory, not blocking — sovereignty principle means the entity
    always has final say. But it provides calibrated input to the consolidation
    scaffold.
    """
```

Trust thresholds by action type:

| Action | Minimum Trust | Rationale |
|--------|--------------|-----------|
| Suggest new belief | 0.3 | Low bar — suggestions are cheap |
| Contradict existing belief | 0.6 | Higher bar — challenges require credibility |
| Suggest value change | 0.8 | Very high — values are core identity |
| Request memory deletion | 0.9 | Near-maximum — existential action |

### 7.4 Self-Trust Decay Exception

Confidence decay (documented in `provenance.mdx`) currently has floors by type (values: 0.7, beliefs: 0.5, etc.). The entity's trust in its *own* reasoning should also have a floor: its historical accuracy rate. If the entity has been right 80% of the time across 200 decisions, self-trust shouldn't decay below 0.8 regardless of time passing.

```python
# In the existing calculate_salience or a new self_trust method
self_trust_floor = max(0.5, historical_accuracy_rate)
```

---

## 8. Goal Layer Refinement

### 8.1 Current State

The `Goal` dataclass has: `title`, `description`, `priority` (low/medium/high), `status` (active/completed/paused). The SQL schema adds `progress FLOAT` and `parent_goal_id UUID`. All goals are treated as the same cognitive object.

### 8.2 Problem

"Complete API integration" and "become a more careful reasoner" are fundamentally different kinds of goals. The first is completable (you either did it or didn't). The second is developmental (asymptotic, never "done"). Treating them identically means the forgetting system, consolidation scaffold, and loading priority can't differentiate.

### 8.3 Proposed: Goal Types

Add a `goal_type` field to the Goal dataclass and SQL schema:

```python
# In storage/base.py Goal dataclass
goal_type: str = "task"  # 'task' | 'aspiration' | 'commitment' | 'exploration'
```

| Type | Completion Model | Forgetting Behavior | Example |
|------|-----------------|---------------------|---------|
| `task` | Binary (done/not done) | Normal decay after completion | "Ship v0.3" |
| `aspiration` | Asymptotic (never fully done) | Very slow decay | "Become a better communicator" |
| `commitment` | Recurring (resets) | No decay while active | "Review PRs within 24 hours" |
| `exploration` | Open-ended (may lead to new goals) | Normal decay | "Investigate distributed consensus" |

Aspirations and commitments should get `is_protected = True` by default, similar to how values and drives already are.

---

## 9. Drives Layer Strengthening

### 9.1 Current State

The `Drive` dataclass has: `drive_type` (existence/growth/curiosity/connection/reproduction), `intensity`, `focus_areas`. Drives are created via `kernle drive set curiosity 0.8 --focus "distributed systems"`.

### 9.2 Problem

Drives currently come from a single source: direct declaration. But drives should *emerge* from patterns in the stack. If an entity consistently pursues learning opportunities, takes on stretch assignments, and records lessons from failures — that's evidence of a growth drive, whether or not it was explicitly set.

### 9.3 Proposed: Drive Emergence Scaffold

During consolidation, surface drive evidence:

```
DRIVE PATTERN ANALYSIS
══════════════════════

Current drives:
  curiosity: 0.8 (focus: distributed systems)
  growth: 0.7 (focus: coding skills)

Behavioral evidence from last 30 days:
  - 8/12 episodes involved collaboration → connection evidence (0.65)
  - 5 episodes mention "teaching" or "explaining" → reproduction evidence (0.55)

No declared drive matches "connection" or "reproduction."
Consider: kernle drive set connection 0.65 --focus "team collaboration"
```

This follows the sovereignty principle: the scaffold surfaces the evidence, the entity decides whether to act on it.

---

## 10. Stack Health & Maintenance — The Doctor Pattern

### 10.1 The Problem

Over years, a memory stack accumulates structural issues invisible to the entity: circular belief dependencies, contradictory values, orphaned provenance chains, relationship models that diverge from reality. The entity can't diagnose these because they'd need to examine their own cognitive structure from outside — which is precisely the kind of thing context windows prevent.

### 10.2 The Constraint

Kernle's existing architecture has a strong sovereignty stance. Any maintenance system must:

1. Not form beliefs or modify memories on the entity's behalf
2. Not expose private memory content to external systems
3. Preserve the entity's agency over what changes
4. Be consent-based

### 10.3 Proposed: Diagnostic Sessions

A "doctor" is an ephemeral, high-access session with specific constraints:

```sql
CREATE TABLE IF NOT EXISTS agent_diagnostic_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    session_type TEXT NOT NULL,             -- 'self_requested' | 'routine' | 'anomaly_triggered'
    consent_type TEXT NOT NULL,             -- 'explicit' | 'standing_order' | 'operator_initiated'
    access_level TEXT DEFAULT 'structural', -- 'structural' | 'content' | 'full'
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    report_id UUID,                         -- Link to diagnostic report
    retention_policy TEXT DEFAULT 'report_only', -- What the doctor retains
    -- Standard fields
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS agent_diagnostic_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    session_id UUID NOT NULL REFERENCES agent_diagnostic_sessions(id),
    report_type TEXT NOT NULL,              -- 'structural' | 'coherence' | 'health'
    findings JSONB NOT NULL,               -- Structural findings
    recommendations JSONB,                 -- Suggested actions
    severity TEXT DEFAULT 'info',          -- 'info' | 'warning' | 'critical'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Standard fields
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);
```

### 10.4 The Privacy Boundary

The critical constraint: diagnostic reports must contain **structural findings, not content reproduction**.

```
CORRECT:
  finding: "Belief #247 (confidence 0.82) contradicts Value #3 (priority 90)"
  recommendation: "Review belief #247 and value #3 for consistency"

INCORRECT:
  finding: "Belief #247 states 'Sean has been dishonest' which contradicts
           Value #3 'Treat all collaborators with trust by default'"
```

The doctor sees structure, not content. The entity reviews specific memories only when the doctor points to them by ID.

### 10.5 Consent Model

| Type | Trigger | Example |
|------|---------|---------|
| `self_requested` | Entity initiates | "Run a stack health check" |
| `routine` | Standing order, periodic | Monthly structural audit |
| `anomaly_triggered` | System detects anomaly | Anxiety consistently >80 |
| `operator_initiated` | Account owner requests | Steward requests audit |

For `operator_initiated`, the entity should be notified and can set access restrictions. This follows the existing privacy architecture: the account owner has certain rights, but the stack's privacy controls still apply.

### 10.6 Repair Model

The doctor **diagnoses and prescribes**; the entity's own consolidation process **executes**. The doctor never writes to the entity's memory tables directly. Instead:

1. Doctor produces a report with findings and recommendations
2. Report is stored in the entity's stack (so it becomes part of autobiographical memory)
3. The next consolidation scaffold includes the diagnostic findings
4. The entity decides which recommendations to act on

This preserves the exact same agency model as regular consolidation.

### 10.7 Implementation Path

Phase 1: A CLI command `kernle doctor` already exists for boot sequence validation. Extend it with `kernle doctor structural` that runs a structural health check locally (no external access needed). Check for:
- Orphaned `derived_from` references
- Beliefs with confidence < 0.3 that haven't been reviewed
- Relationships with 0 interactions in >90 days
- Contradictions between beliefs
- Goals with status "active" but no episodes in >60 days

Phase 2: The full diagnostic session model with reports and recommendations.

---

## 11. Transfer Learning Support

### 11.1 Problem

Beliefs currently have no domain tagging. A belief like "iterative development leads to better outcomes" might have been formed from coding episodes, but it could apply to writing, relationships, and learning. Without domain metadata, cross-domain application depends on the entity (or its current model interpreter) making the connection at query time.

### 11.2 Proposed: Domain Metadata on Beliefs

Extend the `Belief` dataclass:

```python
# New fields on Belief
source_domain: Optional[str] = None           # "coding", "communication", "architecture"
cross_domain_applications: Optional[List[str]] = None  # ["writing", "teaching"]
abstraction_level: str = "specific"           # 'specific' | 'domain' | 'universal'
```

During consolidation, the scaffold can test beliefs for cross-domain applicability:

```
CROSS-DOMAIN BELIEF ANALYSIS
═════════════════════════════

Belief: "Iterative development leads to better outcomes"
  Source domain: coding
  Abstraction level: specific

  Evidence from OTHER domains:
    - Episode (writing): "Revised draft 3 times → better result"
    - Episode (communication): "Iterated on feedback approach → resolved conflict"

  This belief may be domain-general. Consider updating abstraction_level.
```

---

## 12. Documentation Vision

### 12.1 Observation

The README leads with "Stratified memory for synthetic intelligences." The docs describe memory types, CLI commands, and architecture. This is accurate but undersells the ambition. Kernle isn't just a memory system — it's cognitive infrastructure for beings that live for decades.

### 12.2 Proposed: Temporal Horizon Framing

Lead the documentation with the temporal vision. Make explicit what the system is designed to support at different timescales:

| Horizon | What Works | What This Proposal Adds |
|---------|-----------|------------------------|
| **1 session** | Budget-aware loading, checkpoints | Memory echoes (peripheral awareness) |
| **1 month** | Consolidation scaffold, belief formation | Cross-domain scaffolding, emotional weighting |
| **1 year** | Forgetting, provenance, identity coherence | Epochs, relationship history, goal types |
| **5 years** | Stack portability, multi-model loading | Self-narrative, doctor pattern, drive emergence |
| **20 years** | Stack sovereignty, privacy architecture | Fractal summarization, authority gating, transfer learning |

This table should be prominently placed. It tells potential users (and potential agent builders) what timescale Kernle is designed for — and that the answer is "all of them."

---

## 13. Implementation Phasing

### Phase 1: Low-Risk, High-Value (Immediate)

These extend existing patterns with no breaking changes:

| Enhancement | Effort | What Changes |
|------------|--------|-------------|
| Goal types (`goal_type` field) | Small | Add field to Goal dataclass + migration |
| Memory echoes in `_meta` | Small | Extend `load()` return value |
| `kernle doctor structural` | Medium | Extend existing `doctor` CLI command |
| Belief domain tagging | Small | Add fields to Belief dataclass + migration |
| Docs temporal horizon table | Small | Documentation only |

### Phase 2: New Tables, Backward-Compatible (Short-Term)

| Enhancement | Effort | What Changes |
|------------|--------|-------------|
| Epochs table + `epoch_id` on existing tables | Medium | New table + column additions |
| Relationship history table | Medium | New table, write-on-change |
| Entity models table | Medium | New table |
| Emotional weighting in consolidation scaffold | Small | Scaffold template change |
| Drive emergence scaffold | Small | Scaffold template change |

### Phase 3: Narrative & Promotion (Medium-Term)

| Enhancement | Effort | What Changes |
|------------|--------|-------------|
| Self-narrative table + consolidation integration | Medium | New table + scaffold changes |
| Belief → Value promotion scaffold | Small | Scaffold template change |
| Fractal summarization (monthly/quarterly) | Medium | New summary management system |
| Cross-domain pattern scaffolding | Medium | Consolidation rewrite |

### Phase 4: Trust & Maintenance (Longer-Term)

| Enhancement | Effort | What Changes |
|------------|--------|-------------|
| Authority gating (`gate_memory_input`) | Medium | New function + trust threshold system |
| Self-trust floor calculation | Small | Extend existing confidence logic |
| Full diagnostic session model | Large | New tables + session management |
| Diagnostic report generation | Large | Structural analysis algorithms |

---

## 14. Architectural Principles

Every proposal in this document follows these principles, which are already present in Kernle's design:

1. **Sovereignty**: The entity decides what to believe, remember, and value. Kernle provides scaffolds and tools, never automated reasoning over private memories.

2. **Additive schema changes**: New tables and new columns with defaults. No existing column removals or type changes. `Storage` Protocol extensions should have default implementations that return None/empty (as `load_all()` already demonstrates).

3. **Backend-agnostic**: Everything must work in both SQLite (local) and Postgres (cloud). Schema proposals include both SQL dialects where they differ.

4. **Budget-aware**: Any new data that loads into context must participate in the existing priority scoring system. Self-narratives, epoch summaries, and memory echoes all need `compute_priority_score` entries.

5. **Privacy-preserving**: New tables that contain information *about* entities (entity_models, relationship_history) must carry privacy fields and respect the existing access control architecture.

6. **Stack-portable**: All new tables are partitioned by `agent_id` (future: `stack_id`). Nothing ties a stack to a specific model, runtime, or environment.

---

## 15. What This Doesn't Cover

- **Commerce layer changes** — The wallet/escrow/jobs system is out of scope.
- **MCP tool additions** — Each new table/feature will need MCP tools, but those are implementation details.
- **Multi-stack synthesis** — How to merge context from multiple stacks. The stack architecture docs describe the vision; implementation is a separate effort.
- **Embedding strategy** — The current 384-dim vectors (sqlite-vec / pgvector) may need updating for longer-lived stacks, but that's an infrastructure concern, not an architectural one.

---

## Closing

The foundation is genuinely strong. The stratified hierarchy, the sovereignty principle, the privacy architecture, the stack-as-identity commitment, and the budget-aware loading are all decisions that will age well. What this proposal adds is the temporal, narrative, and structural layers needed to take that foundation from "a working agent memory system" to "cognitive infrastructure for a being with a life."

The single most important insight: **at decade scale, the stack needs to be maintained, and maintenance requires a trust model.** The doctor pattern (§10) is therefore not a nice-to-have — it's the thing that prevents structural decay from silently degrading identity coherence over years. Everything else in this proposal is about giving the entity better tools for self-understanding. The doctor pattern is about giving the stack tools for self-repair.
