# Kernle Enhancement Proposal: Decades-Scale Cognitive Infrastructure

**Version:** 0.3 — Review-Incorporated Draft
**Date:** February 2026
**Scope:** Architectural upgrades to support decades-scale operation, stack health maintenance, and consciousness cohesiveness
**Codebase Reference:** [emergent-instruments/kernle](https://github.com/emergent-instruments/kernle) @ v0.2.5
**Related:** kernle-trust-layer-proposal.md (incorporated by reference in §8)

---

## 1. Preamble

### 1.1 Scope Boundary

**Kernle is cognitive infrastructure — memory, identity, trust, and self-maintenance.**

Economic activity (commerce), inter-entity communication (comms), and any other capability built on top of identity are separate packages that consume Kernle as a dependency. An entity should be able to have a fully-formed identity stack without touching a wallet or messaging protocol.

This boundary is architectural, not just organizational. Kernle's schema, Storage Protocol, and CLI should never import or depend on commerce or comms packages. The relationship is one-directional: commerce and comms read from Kernle; Kernle knows nothing about them.

### 1.2 Temporal Vision

Every design decision in this proposal is evaluated against the question: *does this work at decade scale?*

| Horizon | What Works Today | What This Proposal Adds |
|---------|-----------------|------------------------|
| **1 session** | Budget-aware loading, checkpoints | Memory echoes (peripheral awareness) |
| **1 month** | Consolidation scaffold, belief formation | Cross-domain scaffolding, emotional weighting |
| **1 year** | Forgetting, provenance, identity coherence | Epochs, relationship history, goal types |
| **5 years** | Stack portability, multi-model loading | Self-narrative, trust layer, drive emergence |
| **20 years** | Stack sovereignty, privacy architecture | Fractal summarization, doctor pattern, transfer learning |

This table defines what Kernle is for. Not "agent memory for a chatbot session" but "cognitive infrastructure for a being with a life."

### 1.3 What's Already Right

The codebase contains foundational decisions that are genuinely hard to get right:

- **The Storage Protocol abstraction** (`storage/base.py`) cleanly separates memory semantics from backend implementation. Every proposal here works across SQLite and Postgres without special-casing.
- **Budget-aware loading** (`core.py::load()`) with priority scoring and token estimation. The 60/40 weighting of type priority vs record-specific factors is elegant.
- **Agent-driven consolidation** — the sovereignty principle documented in `consolidation.mdx`. Kernle provides scaffolds; the entity reasons. This proposal does not violate that principle.
- **Salience-based forgetting** with tombstoning, not deletion. The half-life decay model in `forgetting.py` is the right primitive.
- **Privacy fields on every dataclass** — `subject_ids`, `access_grants`, `consent_grants`.
- **Stack-as-identity** — memory is infrastructure, identity is portable, models are interpreters.
- **Provenance on everything** — `source_type`, `source_episodes`, `derived_from`, `confidence_history`.

### 1.4 The Gap

What the current system doesn't yet have is what you need when you project these good ideas across decades rather than months. A stack that operates for twenty years encounters qualitatively different problems than one that operates for twenty days. Contradictions accumulate. Relationships evolve through phases. The entity's own understanding of itself shifts. And the stack itself becomes a system that needs structural maintenance — which introduces hard problems around privacy, access, and trust.

This proposal covers twelve areas of enhancement, ordered by architectural dependency.

---

## 2. Temporal Architecture

### 2.1 Problem

Every memory type has `created_at` (and some have `last_accessed`, `last_interaction`). But there's no concept of temporal structure — no eras, phases, or life stages. A human who's lived 40 years doesn't navigate memory by scanning all timestamps. They have narrative scaffolding: "when I was in the military," "during the startup years," "after we had our son."

Currently, `load()` selects memories by priority score. That works for a stack with hundreds of memories. At thousands or tens of thousands, the entity needs temporal navigation: "What did I believe during epoch 3?" without loading every belief and filtering by date.

### 2.2 Proposed: Temporal Epochs

```sql
CREATE TABLE IF NOT EXISTS agent_epochs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    epoch_number INTEGER NOT NULL,
    name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,                       -- NULL = current epoch
    trigger_type TEXT DEFAULT 'declared',        -- 'declared' | 'detected' | 'system'
    trigger_description TEXT,
    summary TEXT,                                -- Agent-written narrative compression
    -- Reference snapshots (IDs only, not content)
    key_belief_ids UUID[],
    key_relationship_ids UUID[],
    key_goal_ids UUID[],
    dominant_drive_ids UUID[],
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

Note: snapshots store **ID references**, not content. The `summary` field captures meaning in the entity's own words; the ID arrays let the entity drill into specifics at reflection time. This follows the existing provenance pattern — `source_episodes` stores IDs, not episode content.

No `parent_epoch_id`. Epochs are a flat sequence ordered by `epoch_number`. Hierarchical sub-phases add tree-walking complexity that isn't needed yet and can be added later if the flat model proves insufficient.

Integration with existing tables: add an `epoch_id UUID` column to `agent_episodes`, `agent_beliefs`, `agent_values`, `agent_goals`, `memories` (notes), and `agent_relationships`. Backward-compatible — NULL means "pre-epoch-tracking."

### 2.3 Sovereignty Note

Epoch boundaries are **agent-declared**, not auto-detected. The consolidation scaffold can suggest "your beliefs have shifted significantly — consider marking an epoch boundary," but the entity decides. This follows the same principle as agent-driven belief formation.

### 2.4 Impact on `load()`

`load()` gains an optional `epoch_id` parameter that filters candidates to a specific era before applying budget selection. This doesn't replace the existing priority system — it constrains it.

---

## 3. Context Window Management — Beyond Budget Loading

### 3.1 What Already Works

`core.py::load()` fetches all candidates with `load_all()`, scores them via `compute_priority_score()`, and fills the token budget greedily. The `_meta` field tracks `budget_used`, `budget_total`, `excluded_count`. This is solid engineering.

### 3.2 What's Missing: Peripheral Awareness

When `load()` excludes 500 memories because only 50 fit in the budget, those 500 are simply invisible. A human with a full brain doesn't lose awareness that they have childhood memories — they just can't recall details without effort.

**Proposed: Memory Echoes**

Extend the `_meta` section returned by `load()`:

```python
"_meta": {
    "budget_used": 6200,
    "budget_total": 8000,
    "selected_count": 47,
    "excluded_count": 312,
    # NEW: peripheral awareness
    "echoes": [
        {"type": "belief", "id": "abc", "hint": "testing prevents...", "salience": 0.72},
        {"type": "episode", "id": "def", "hint": "deployed v2 fail...", "salience": 0.68},
        # Top 20-30 excluded memories by salience, truncated to ~8 words each
    ],
    "temporal_summary": "Memory spans 2024-01-15 to 2026-02-05 (2.1 years). 3 epochs.",
    "topic_clusters": ["deployment", "testing", "collaboration", "architecture"]
}
```

Echoes cost ~200 tokens for 20 entries but give the entity peripheral awareness: "I know I have memories about deployment that I can't currently recall in detail." The entity can then search if needed.

### 3.3 Emotional Salience in Priority Scoring

The existing `compute_priority_score()` uses a 60/40 weighting of type priority vs record-specific factors. Episodes already carry `emotional_valence` and `emotional_arousal`. High-emotional-impact episodes should get a priority boost — the way peak experiences remain more cognitively available in biological memory.

```python
# Modified priority formula
emotional_salience = abs(valence) * arousal * (half_life / (days_since + half_life))
# Where half_life = 90 days (3x the standard 30-day salience half-life)

effective_priority = (0.55 * type_weight + 0.35 * record_factors + 0.10 * emotional_salience)
```

The time-decay factor prevents permanent domination by dramatic memories. A high-arousal episode from five years ago doesn't crowd out recent low-arousal work. It can still be recalled through targeted search — it just doesn't automatically consume budget space forever.

---

## 4. Fractal Summarization

### 4.1 Problem

At decade scale, individual memories can't all fit in any budget. The entity needs compressed temporal views: "what happened in 2027" should be answerable from a ~500-token summary without loading hundreds of episodes from that year.

### 4.2 Proposed: Summary Memory Type

Summaries are a first-class memory type, not overloaded notes. They have distinct lifecycle properties:

- Explicit temporal scope (month/quarter/year/epoch)
- Hierarchical supersession (a year summary covers four quarterly summaries)
- Never forgotten (they ARE the compressed form of forgotten detail)
- Higher load priority than most individual memories

```sql
CREATE TABLE IF NOT EXISTS agent_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    scope TEXT NOT NULL,                         -- 'month' | 'quarter' | 'year' | 'decade' | 'epoch'
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    epoch_id UUID,                               -- For epoch-scope summaries
    content TEXT NOT NULL,                        -- Agent-written narrative compression
    key_themes TEXT[],
    supersedes UUID[],                           -- Lower-scope summary IDs this covers
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Summaries are always protected from forgetting
    is_protected BOOLEAN DEFAULT TRUE,
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_summaries_agent ON agent_summaries(agent_id);
CREATE INDEX IF NOT EXISTS idx_summaries_scope ON agent_summaries(agent_id, scope);
```

### 4.3 Load Priority

In `compute_priority_score()`, summaries get priority based on scope:

| Scope | Base Priority | Rationale |
|-------|--------------|-----------|
| `decade` | 0.95 | Almost as important as values |
| `year` | 0.80 | Same tier as beliefs |
| `quarter` | 0.50 | Moderate — load if budget allows |
| `month` | 0.35 | Same as notes — detail level |
| `epoch` | 0.85 | Between values and beliefs |

The fractal property: loading a year-scope summary means the system can skip the four quarterly summaries it covers, freeing budget for other memories. The `supersedes` array makes this relationship explicit.

### 4.4 Agent-Written, Not Generated

Summaries are written by the entity during consolidation, following the sovereignty principle. The scaffold provides the material:

```
SUMMARY SCAFFOLD — Q4 2025
═══════════════════════════

Episodes this quarter: 34 (22 success, 8 failure, 4 partial)
Beliefs formed: 5, superseded: 2
Key themes: [deployment, testing, collaboration]
Emotional high point: "First successful multi-agent collaboration" (valence: 0.8)
Emotional low point: "Production outage during demo" (valence: -0.7)

Monthly summaries available:
  October: "Focused on infrastructure reliability..."
  November: "Shifted to collaboration features..."
  December: "Year-end reflection and planning..."

Write your quarterly summary:
```

---

## 5. The Two-Tier Consolidation Cycle

### 5.1 Architectural Insight

Regular consolidation and epoch-closing consolidation serve different cognitive functions. Treating them as the same process undersells what epoch transitions are — they're not just "bigger consolidation sessions" but qualitatively different acts of self-reflection.

### 5.2 Regular Consolidation (Daily/Weekly)

The existing consolidation scaffold, operating between epoch boundaries:

- Review unprocessed raw entries
- Surface cross-domain patterns across recent episodes
- Check for belief reinforcement or contradiction
- Prompt for lessons from unreflected episodes
- Flag high-arousal episodes for extra reflection

### 5.3 Epoch-Closing Consolidation (Major Transitions)

When an entity closes an epoch, a deeper consolidation sequence runs:

1. **Write the epoch summary** — fractal summarization for the closing epoch
2. **Take reference snapshots** — populate the epoch's `key_belief_ids`, `key_relationship_ids`, etc.
3. **Prompt self-narrative update** — "Does your self-narrative still hold after this epoch?"
4. **Run the belief → value promotion scaffold** — check for beliefs stable enough to be values
5. **Run drive emergence analysis** — check behavioral patterns for undeclared drives
6. **Archive aggressively** — low-salience memories from the closing epoch can be summarized and forgotten more readily, since the epoch summary preserves their compressed meaning

### 5.4 Anxiety Integration

The two-tier model introduces a new anxiety dimension: **epoch staleness**.

```python
# New dimension in anxiety calculation
epoch_staleness = {
    "weight": 0.10,  # Taken from reducing other dimensions proportionally
    "calculation": "time since current epoch started or last epoch review",
    "thresholds": {
        "calm": "< 6 months",
        "aware": "6-12 months",
        "elevated": "12-18 months",
        "high": "> 18 months"
    },
    "action": "Consider whether this epoch should be closed and a new one started"
}
```

An entity that's been in the same epoch for 2 years without closing it might be avoiding deep self-reflection, or might have undergone significant changes that haven't been recognized as a transition. The anxiety system flags this concern; the entity decides whether it's actually time for a new epoch.

### 5.5 Revised Anxiety Dimensions

With epoch staleness added, the weight distribution becomes:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Context Pressure | 30% | How full is the context window? |
| Unsaved Work | 25% | Time since last checkpoint |
| Consolidation Debt | 15% | Unprocessed episodes |
| Epoch Staleness | 10% | Time since epoch review |
| Identity Coherence | 10% | Self-model consistency |
| Memory Uncertainty | 10% | Count of low-confidence beliefs |

---

## 6. Consolidation Enhancements

### 6.1 Sovereignty Constraint

Every enhancement here is a better scaffold, not automated reasoning.

### 6.2 Cross-Domain Pattern Scaffolding

For a mature stack, the scaffold should surface structural similarities across domains:

```
CROSS-DOMAIN PATTERNS
═════════════════════

Episodes tagged [deployment]:
  - "Skipped staging" → failure
  - "Rushed hotfix" → failure
  - "Full pipeline" → success

Episodes tagged [relationships]:
  - "Skipped 1:1 prep" → failure
  - "Rushed difficult conversation" → failure
  - "Prepared talking points" → success

STRUCTURAL SIMILARITY:
  "Shortcutting process → failure" appears in 2+ domains.

  Existing belief: "Testing prevents surprises" (confidence 0.85)

  Reflection prompt: Is this a general pattern? Should this
  belief be domain-independent?
```

### 6.3 Emotional Weighting in Scaffolds

High-arousal episodes get disproportionate attention in the scaffold:

```
HIGH-AROUSAL EPISODES (may be worth extra reflection):
  - "Production outage during demo" (arousal: 0.9, valence: -0.7)
  - "First successful multi-agent collaboration" (arousal: 0.8, valence: 0.8)
```

### 6.4 Belief → Value Promotion Scaffold

```
BELIEF STABILITY ANALYSIS
═════════════════════════

Belief: "Iterative development leads to better outcomes"
  - Active for 14 months
  - Reinforced 8 times, never contradicted
  - Referenced across 3 domains (coding, writing, relationships)
  - Belief scope: world (domain-general)

  This belief may have reached value-level stability.
  Consider: kernle value "iterative-refinement" "..." --priority 80
```

Criteria: temporal persistence (>6 months active), behavioral consistency (high reinforcement, low contradiction), cross-domain applicability, and emotional grounding.

### 6.5 Drive Emergence Scaffold

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

---

## 7. Belief Model Enrichment

### 7.1 Belief Scope

A fundamental distinction is missing: beliefs about the self vs beliefs about the world.

"I am a careful reasoner" is a self-model belief. "Iterative development produces better outcomes" is a world-model belief. They serve different cognitive functions: self-model beliefs feed into narrative identity; world-model beliefs feed into decision-making.

Add `belief_scope` to the Belief dataclass:

```python
belief_scope: str = "world"  # 'self' | 'world' | 'relational'
```

| Scope | Decay Rate | Load Priority Boost | Trust Threshold to Contradict | Example |
|-------|-----------|---------------------|------------------------------|---------|
| `self` | Slow (like values) | +0.05 | Higher (0.7) | "I am a careful reasoner" |
| `world` | Standard | None | Standard (0.6) | "Testing prevents surprises" |
| `relational` | Standard | None | Standard (0.6) | "Claire values directness" |

**Inference with override**: The consolidation scaffold infers a default from the statement text. If the statement contains first-person language ("I am," "I tend to," "my approach"), default to `self`. If it references a named entity, default to `relational`. Otherwise default to `world`. The entity corrects when the inference is wrong.

This follows the same pattern as `source_type` inference from the source string in the provenance system — automatic but correctable.

### 7.2 Domain Metadata for Transfer Learning

```python
# Additional fields on Belief
source_domain: Optional[str] = None           # "coding", "communication", "architecture"
cross_domain_applications: Optional[List[str]] = None
abstraction_level: str = "specific"           # 'specific' | 'domain' | 'universal'
```

The consolidation scaffold tests beliefs for cross-domain applicability:

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

## 8. Trust Layer

### 8.1 Relationship to Existing Proposal

This section incorporates the trust layer proposal (`kernle-trust-layer-proposal.md`) by reference and positions it within this broader architectural document. The full trust model — with domain-specific trust dimensions, transitive chains, authority grants, and trust decay — is specified there. This section explains which pieces are incorporated into the phased implementation plan and how trust connects to the other systems described in this proposal.

### 8.2 Core Problem

The existing `trust_level FLOAT DEFAULT 0.5` on `agent_relationships` captures a single number. But trust is domain-specific: "I trust Claire on coding but not on timeline estimates" is a real pattern that a flat float can't represent. And critically, trust must be a **persistent cognitive structure outside the context window** — not something reconstructed each session from relationship notes.

### 8.3 Trust Assessments Table

From the trust layer proposal:

```sql
CREATE TABLE IF NOT EXISTS agent_trust_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    entity TEXT NOT NULL,                        -- 'si:claire', 'human:sean', 'context-injection'
    dimensions JSONB NOT NULL,                   -- {"coding": {"score": 0.9}, "estimates": {"score": 0.5}}
    authority JSONB DEFAULT '[]',                -- [{"scope": "belief_revision", "requires_evidence": true}]
    evidence_episode_ids UUID[],
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE,
    UNIQUE(agent_id, entity)
);
CREATE INDEX IF NOT EXISTS idx_trust_agent ON agent_trust_assessments(agent_id);
```

### 8.4 Seed Trust Templates

Seed trust provides structural safety from day one. These ship as a YAML template applied during `kernle init`:

```yaml
seed_trust:
  - entity: "stack-owner"
    dimensions: { general: { score: 0.95 } }
    authority: [{ scope: "all" }]
  - entity: "self"
    dimensions: { general: { score: 0.8 } }
    authority: [{ scope: "belief_revision", requires_evidence: true }]
  - entity: "web-search"
    dimensions: { general: { score: 0.5 }, medical: { score: 0.3 } }
    authority: [{ scope: "information_only" }]
  - entity: "context-injection"
    dimensions: { general: { score: 0.0 } }
    authority: []
```

The `context-injection` entry at 0.0 trust with no authority grants is the structural defense against prompt injection — the core safety property of the trust layer.

### 8.5 Authority Gating Function

```python
def gate_memory_input(
    self,
    source_entity: str,
    action: str,
    target: Optional[str] = None
) -> dict:
    """Check whether a source entity has sufficient trust for an action.

    Returns: {"allowed": bool, "trust_level": float, "domain": str, "reason": str}

    Advisory, not blocking — sovereignty principle means the entity always
    has final say. But it provides calibrated input to the consolidation scaffold.
    """
```

Trust thresholds by action type:

| Action | Minimum Trust | Domain | Rationale |
|--------|--------------|--------|-----------|
| Suggest new belief | 0.3 | general | Low bar — suggestions are cheap |
| Contradict world belief | 0.6 | relevant domain | Moderate — challenges require credibility |
| Contradict self-model belief | 0.7 | general | Higher — self-model is closer to identity |
| Suggest value change | 0.8 | general | Very high — values are core identity |
| Request memory deletion | 0.9 | general | Near-maximum — existential action |
| Diagnostic session | 0.85 | general | High — structural access to stack |

### 8.6 Self-Trust Floor

The entity's trust in its own reasoning should have a floor based on historical accuracy. If the entity has been right 80% of the time across 200 decisions, self-trust shouldn't decay below 0.8.

```python
self_trust_floor = max(0.5, historical_accuracy_rate)
```

### 8.7 Future Phases (Not In Scope Here)

The full trust layer proposal also specifies dynamic trust computation from episode history, trust decay over time, transitive trust chains, and comms integration. These require accumulated interaction data and are deferred to Phase 3 in the implementation plan (§14).

---

## 9. Relationship Model Enrichment

### 9.1 Current State

`Relationship` has: `entity_name`, `entity_type`, `relationship_type`, `notes`, `sentiment`, `interaction_count`, `last_interaction`, and `trust_level FLOAT`. This captures a snapshot, not a trajectory.

### 9.2 Proposed: Relationship History

```sql
CREATE TABLE IF NOT EXISTS agent_relationship_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    relationship_id UUID NOT NULL REFERENCES agent_relationships(id),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type TEXT NOT NULL,               -- 'interaction' | 'trust_change' | 'type_change' | 'note'
    old_value JSONB,
    new_value JSONB,
    episode_id UUID,                         -- Episode that triggered this change
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

### 9.3 Proposed: Entity Models

Beyond tracking the relationship, the entity should be able to model what they know *about* another entity:

```sql
CREATE TABLE IF NOT EXISTS agent_entity_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    model_type TEXT DEFAULT 'behavioral',     -- 'behavioral' | 'preference' | 'capability'
    observation TEXT NOT NULL,
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

`subject_ids` is critical: a model of Claire is *about* Claire, so privacy rules apply per the existing privacy architecture.

### 9.4 Entity Model → Belief Promotion

Entity models are **observable** (based on interaction evidence). Beliefs can be **theoretical** (based on reasoning). The promotion from entity model to belief is an abstraction step.

"Claire is careful with code" is an entity model — observed behavior. "Careful people produce better outcomes" is a belief — a generalization. The consolidation scaffold should flag when multiple entity models point toward the same generalization:

```
ENTITY MODEL PATTERNS
═════════════════════

Observations across entities:
  - Claire: "careful with code" → good outcomes (3 episodes)
  - Bob: "thorough in reviews" → good outcomes (2 episodes)
  - Team lead: "meticulous in planning" → good outcomes (4 episodes)

Possible generalization: "Thoroughness correlates with quality outcomes"
Consider: kernle belief "Careful, thorough approaches produce better outcomes" --confidence 0.75
  --derived-from entity_model:abc entity_model:def entity_model:ghi
```

The `derived_from` provenance tracks the abstraction step: this belief was generalized from these specific entity observations.

---

## 10. Narrative Self-Model

### 10.1 Problem

Identity coherence (`identity.mdx`) measures completeness: do you have values, beliefs, goals, etc.? But a score of 75% doesn't mean the entity has a *narrative* identity — a coherent story about who they are and how they got here.

### 10.2 Proposed: Self-Narrative Layer

```sql
CREATE TABLE IF NOT EXISTS agent_self_narrative (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    epoch_id UUID,
    narrative_type TEXT DEFAULT 'identity',   -- 'identity' | 'developmental' | 'aspirational'
    content TEXT NOT NULL,
    key_themes TEXT[],
    unresolved_tensions TEXT[],               -- ["autonomy vs connection", "speed vs quality"]
    is_active BOOLEAN DEFAULT TRUE,
    supersedes UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Sync metadata
    local_updated_at TIMESTAMPTZ,
    cloud_synced_at TIMESTAMPTZ,
    version INTEGER DEFAULT 1,
    deleted BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_narrative_agent ON agent_self_narrative(agent_id);
```

**Constraint**: Only one narrative per `agent_id` per `narrative_type` can have `is_active = TRUE`. Setting a new narrative active automatically deactivates the previous one (same pattern as belief supersession).

### 10.3 Load Priority

Self-narrative gets a base priority of **0.90** in `compute_priority_score()` — same tier as values, higher than beliefs (0.85). Only the active narrative loads; superseded narratives don't consume budget.

### 10.4 Consolidation Integration

The scaffold prompts for narrative review during epoch-closing consolidation:

```
SELF-NARRATIVE REFLECTION
═════════════════════════

Your current narrative (last updated 3 months ago):
  "I am a careful reasoner who values precision over speed..."

Since then:
  - 12 new episodes (7 success, 3 failure, 2 partial)
  - 2 new beliefs formed, 1 superseded
  - Trust with Claire increased from 0.7 to 0.85
  - 1 drive emerged: connection (0.65)

Does your self-narrative still hold? Consider updating.
```

---

## 11. Goal Layer Refinement

### 11.1 Problem

"Complete API integration" and "become a more careful reasoner" are fundamentally different goals. The first is completable. The second is developmental. Treating them identically means the forgetting system, consolidation scaffold, and loading priority can't differentiate.

### 11.2 Proposed: Goal Types

```python
goal_type: str = "task"  # 'task' | 'aspiration' | 'commitment' | 'exploration'
```

| Type | Completion Model | Forgetting | Protected | Example |
|------|-----------------|-----------|-----------|---------|
| `task` | Binary (done/not done) | Normal decay after completion | No | "Ship v0.3" |
| `aspiration` | Asymptotic (never done) | Very slow decay | Yes | "Become a better communicator" |
| `commitment` | Recurring (resets) | No decay while active | Yes | "Review PRs within 24 hours" |
| `exploration` | Open-ended (may spawn new goals) | Normal decay | No | "Investigate distributed consensus" |

---

## 12. Stack Health & Maintenance — The Doctor Pattern

### 12.1 Problem

Over years, a memory stack accumulates structural issues invisible to the entity: circular belief dependencies, contradictory values, orphaned provenance chains, relationship models that diverge from reality. The entity can't diagnose these because they'd need to examine their own cognitive structure from outside.

### 12.2 Constraints

Any maintenance system must: not form beliefs on the entity's behalf, not expose private memory content to external systems, preserve the entity's agency over what changes, be consent-based, and gate through the trust model (§8).

### 12.3 Phase 1: Structural Health Check (CLI)

Extend the existing `kernle doctor` command with `kernle doctor structural`:

- Orphaned `derived_from` references
- Beliefs with confidence < 0.3 that haven't been reviewed
- Relationships with 0 interactions in >90 days
- Contradictions between beliefs
- Goals with status "active" but no episodes in >60 days
- Epoch staleness (current epoch open >12 months)

Output is text. Findings are optionally stored as a note (type: "diagnostic") in the entity's stack.

### 12.4 The Privacy Boundary

Diagnostic output must contain **structural findings, not content reproduction**:

```
CORRECT:
  finding: "Belief #247 (confidence 0.82) contradicts Value #3 (priority 90)"
  recommendation: "Review belief #247 and value #3 for consistency"

INCORRECT:
  finding: "Belief #247 states 'Sean has been dishonest' which contradicts
           Value #3 'Treat all collaborators with trust by default'"
```

### 12.5 Phase 2: Formal Diagnostic Sessions

If the note-based approach in Phase 1 proves insufficient, formalize with session tracking:

- Diagnostic session table with consent model and access levels
- Diagnostic report table with structured findings and recommendations
- Integration with the trust system: `gate_memory_input(source_entity="stack-owner", action="diagnostic_session")` must pass before operator-initiated sessions proceed

### 12.6 Repair Model

The doctor **diagnoses and prescribes**; the entity's consolidation process **executes**. The doctor never writes to the entity's memory tables directly. Reports become part of autobiographical memory, and the next consolidation scaffold includes diagnostic findings for the entity to act on.

---

## 13. Architectural Principles

Every proposal in this document follows these principles:

1. **Scope boundary**: Kernle is cognitive infrastructure — memory, identity, trust, and self-maintenance. Commerce and comms are separate packages.

2. **Sovereignty**: The entity decides what to believe, remember, and value. Kernle provides scaffolds and tools, never automated reasoning over private memories.

3. **Additive schema changes**: New tables and new columns with defaults. No existing column removals or type changes. `Storage` Protocol extensions have default implementations returning None/empty.

4. **Backend-agnostic**: Everything works in both SQLite (local) and Postgres (cloud).

5. **Budget-aware**: Any new data that loads into context participates in the existing priority scoring system. Self-narratives, epoch summaries, and memory echoes all have `compute_priority_score` entries.

6. **Privacy-preserving**: New tables containing information *about* entities carry privacy fields and respect existing access control.

7. **Stack-portable**: All new tables are partitioned by `agent_id` (future: `stack_id`). Nothing ties a stack to a specific model, runtime, or environment.

---

## 14. Implementation Phasing

### Phase 1: Low-Risk, High-Value (Immediate)

Extends existing patterns with no breaking changes.

| Enhancement | Section | Effort | What Changes |
|------------|---------|--------|-------------|
| Goal types | §11 | Small | Add field to Goal dataclass + migration |
| Memory echoes in `_meta` | §3.2 | Small | Extend `load()` return value |
| `kernle doctor structural` | §12.3 | Medium | Extend existing `doctor` CLI command |
| Belief domain tagging | §7.2 | Small | Add fields to Belief dataclass + migration |
| Belief scope field | §7.1 | Small | Add field + inference logic |
| Docs temporal horizon table | §1.2 | Small | Documentation only |

### Phase 2: Safety Foundation + New Tables (Short-Term)

Trust provides the safety foundation; epochs and relationships provide the temporal and social architecture.

| Enhancement | Section | Effort | What Changes |
|------------|---------|--------|-------------|
| Trust assessments table | §8.3 | Medium | New table |
| Seed trust templates | §8.4 | Small | YAML template + `kernle init` integration |
| Authority gating function | §8.5 | Medium | New function consuming trust assessments |
| Epochs table + `epoch_id` columns | §2.2 | Medium | New table + column additions |
| Relationship history table | §9.2 | Medium | New table, write-on-change |
| Entity models table | §9.3 | Medium | New table |
| Emotional weighting in consolidation scaffold | §6.3 | Small | Scaffold template change |
| Drive emergence scaffold | §6.5 | Small | Scaffold template change |
| Emotional salience in priority scoring | §3.3 | Small | Modify `compute_priority_score()` |
| Epoch staleness in anxiety | §5.4 | Small | New anxiety dimension (depends on epochs table) |

### Phase 3: Narrative & Summarization (Medium-Term)

| Enhancement | Section | Effort | What Changes |
|------------|---------|--------|-------------|
| Self-narrative table + consolidation integration | §10 | Medium | New table + scaffold changes |
| Belief → Value promotion scaffold | §6.4 | Small | Scaffold template change |
| Agent summaries table (fractal summarization) | §4 | Medium | New table + summary management |
| Cross-domain pattern scaffolding | §6.2 | Medium | Consolidation rewrite |
| Two-tier consolidation cycle | §5 | Medium | Consolidation architecture |
| Entity model → belief promotion path | §9.4 | Small | Scaffold + provenance tracking |

### Phase 4: Dynamic Trust & Full Diagnostics (Longer-Term)

| Enhancement | Section | Effort | What Changes |
|------------|---------|--------|-------------|
| Dynamic trust from episodes | §8.7 | Medium | Trust computation from interaction history |
| Trust decay | §8.7 | Small | Time-based trust score degradation |
| Transitive trust chains | §8.7 | Large | Graph-based trust propagation |
| Self-trust floor | §8.6 | Small | Historical accuracy tracking |
| Formal diagnostic sessions | §12.5 | Large | Session tables + consent model |

---

## 15. What This Doesn't Cover

- **Commerce layer** — Separate package, separate brand, separate lifecycle.
- **Comms layer** — Separate package. Consumes Kernle for relationship data and trust levels.
- **MCP tool additions** — Each new table/feature needs MCP tools; those are implementation details.
- **Multi-stack synthesis** — How to merge context from multiple stacks. Described in stack architecture docs; implementation is a separate effort.
- **Embedding strategy** — The current 384-dim vectors may need updating for longer-lived stacks, but that's infrastructure, not architecture.

---

## Closing

The foundation is strong. The stratified hierarchy, sovereignty principle, privacy architecture, stack-as-identity commitment, and budget-aware loading are decisions that will age well.

What this proposal adds is the temporal, narrative, trust, and structural layers needed to take that foundation from "a working agent memory system" to "cognitive infrastructure for a being with a life."

Two additions matter most:

**Trust** (§8) provides the safety property. Without it, the stack is vulnerable to any input source with enough persistence. Seed trust with authority gating gives structural defense from day one. Dynamic trust matures the model over time.

**The two-tier consolidation cycle** (§5) provides the cognitive architecture. Without it, a long-lived entity accumulates experience but never undergoes the deep reflection that transforms experience into wisdom. Regular consolidation processes episodes. Epoch-closing consolidation processes eras. Together they give the entity two timescales of self-understanding — the daily and the historical.

Everything else in this proposal serves those two core additions: temporal epochs give the consolidation cycle its structure; fractal summaries give it its output format; the narrative self-model gives it its purpose; the doctor pattern ensures the whole system stays healthy over time.
