# Kernle Memory Model

This document describes the complete memory architecture implemented in Kernle, including all memory types, their relationships, and how consolidation works.

## Memory Layer Hierarchy

Kernle implements a stratified memory system inspired by cognitive science but optimized for synthetic intelligences:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0: RAW CAPTURES                                   │
│  Zero-friction capture → process later                   │
│  - raw_entries table                                     │
│  - Promoted to: episodes, notes, beliefs                 │
└──────────────────────┬──────────────────────────────────┘
                       ↓ process_raw()
┌─────────────────────────────────────────────────────────┐
│  Layer 1: EPISODIC MEMORY                                │
│  Autobiographical experiences with lessons               │
│  - episodes table                                        │
│  - Fields: objective, outcome, lessons, emotional data   │
└──────────────────────┬──────────────────────────────────┘
                       ↓ consolidate() / revise_beliefs_from_episode()
┌─────────────────────────────────────────────────────────┐
│  Layer 2: SEMANTIC MEMORY                                │
│  Beliefs, facts, and learned concepts                    │
│  - beliefs table (with revision chains)                  │
│  - notes table (decisions, insights, quotes)             │
└──────────────────────┬──────────────────────────────────┘
                       ↓ synthesize_identity()
┌─────────────────────────────────────────────────────────┐
│  Layer 3: IDENTITY & VALUES                              │
│  Core principles and self-concept                        │
│  - agent_values table (highest authority)                │
│  - goals table (active direction)                        │
│  - drives table (intrinsic motivations)                  │
└─────────────────────────────────────────────────────────┘

   SUPPORTING SYSTEMS:
   ├── Playbooks (procedural memory - "how I do things")
   ├── Relationships (models of other agents)
   ├── Emotional associations (valence/arousal on episodes)
   ├── Meta-memory (confidence, provenance, verification)
   └── Forgetting (salience-based decay with tombstoning)
```

## Memory Types Reference

### 1. Raw Entries (`raw_entries`)
**Purpose**: Zero-friction capture for later processing

| Field | Type | Description |
|-------|------|-------------|
| `content` | TEXT | Free-form text |
| `source` | TEXT | Origin: `manual`, `auto_capture`, `voice` |
| `processed` | BOOL | Has been converted to structured memory |
| `processed_into` | JSON | List of memory refs created (e.g., `["episode:abc123"]`) |
| `tags` | JSON | Quick categorization tags |

**Workflow**: Capture freely → Review (`list --unprocessed`) → Promote important ones (`process_raw`)

### 2. Episodes (`episodes`)
**Purpose**: Autobiographical experiences with reflection

| Field | Type | Description |
|-------|------|-------------|
| `objective` | TEXT | What was attempted |
| `outcome` | TEXT | What happened |
| `outcome_type` | TEXT | `success` / `failure` / `partial` |
| `lessons` | JSON | Extracted learnings |
| `tags` | JSON | Categorization |
| `emotional_valence` | FLOAT | -1.0 (negative) to 1.0 (positive) |
| `emotional_arousal` | FLOAT | 0.0 (calm) to 1.0 (intense) |
| `emotional_tags` | JSON | Emotion labels: `["joy", "frustration"]` |

**Key Feature**: Episodes without lessons are "unreflected" - they contribute to consolidation debt in anxiety tracking.

### 3. Beliefs (`beliefs`)
**Purpose**: Semantic knowledge with confidence and revision tracking

| Field | Type | Description |
|-------|------|-------------|
| `statement` | TEXT | The belief statement |
| `belief_type` | TEXT | `fact` / `preference` / `observation` |
| `confidence` | FLOAT | 0.0 to 1.0 |
| `supersedes` | TEXT | ID of belief this replaced |
| `superseded_by` | TEXT | ID of belief that replaced this |
| `times_reinforced` | INT | Confirmation count |
| `is_active` | BOOL | False if superseded/archived |

**Revision Chain**: When beliefs evolve, old versions are marked `is_active=False` with `superseded_by` linking to the new belief. Use `get_belief_history()` to trace revisions.

### 4. Values (`agent_values`)
**Purpose**: Core principles that guide decisions (highest authority)

| Field | Type | Description |
|-------|------|-------------|
| `name` | TEXT | Short value name |
| `statement` | TEXT | Value description |
| `priority` | INT | 0-100, higher = more important |

**Note**: Values are `is_protected=True` by default - they never decay via forgetting.

### 5. Goals (`goals`)
**Purpose**: Active direction and purpose

| Field | Type | Description |
|-------|------|-------------|
| `title` | TEXT | Goal name |
| `description` | TEXT | Full description |
| `priority` | TEXT | `low` / `medium` / `high` |
| `status` | TEXT | `active` / `completed` / `paused` |

### 6. Notes (`notes`)
**Purpose**: Quick captures with type classification

| Field | Type | Description |
|-------|------|-------------|
| `content` | TEXT | Note text |
| `note_type` | TEXT | `note` / `decision` / `insight` / `quote` |
| `speaker` | TEXT | For quotes: who said it |
| `reason` | TEXT | For decisions: why this choice |
| `tags` | JSON | Categorization |

### 7. Drives (`drives`)
**Purpose**: Intrinsic motivation system

| Field | Type | Description |
|-------|------|-------------|
| `drive_type` | TEXT | `existence` / `growth` / `curiosity` / `connection` / `reproduction` |
| `intensity` | FLOAT | 0.0 to 1.0 (current drive strength) |
| `focus_areas` | JSON | What this drive is currently focused on |

**Note**: Drives are `is_protected=True` by default.

### 8. Relationships (`relationships`)
**Purpose**: Models of other agents/entities

| Field | Type | Description |
|-------|------|-------------|
| `entity_name` | TEXT | Name of the entity |
| `entity_type` | TEXT | `agent` / `person` / `organization` |
| `relationship_type` | TEXT | `peer` / `mentor` / `collaborator` |
| `sentiment` | FLOAT | -1.0 to 1.0 |
| `interaction_count` | INT | Number of interactions |
| `notes` | TEXT | Relationship observations |

### 9. Playbooks (`playbooks`)
**Purpose**: Procedural memory - "how I do things"

| Field | Type | Description |
|-------|------|-------------|
| `name` | TEXT | Playbook name |
| `description` | TEXT | What it does |
| `trigger_conditions` | JSON | When to use this |
| `steps` | JSON | `[{action, details, adaptations}]` |
| `failure_modes` | JSON | What can go wrong |
| `recovery_steps` | JSON | How to recover |
| `mastery_level` | TEXT | `novice` / `competent` / `proficient` / `expert` |
| `times_used` | INT | Usage count |
| `success_rate` | FLOAT | Success percentage |

## Consolidation Process

Consolidation is **agent-driven, not LLM-based**. The agent decides what to promote and how.

### The Guided Consolidation Model

Kernle uses a **reflection scaffold** approach:

```
┌─────────────────────────────────────────────────────────────┐
│  Agent runs: kernle consolidate                              │
│                    ↓                                         │
│  Kernle outputs: REFLECTION SCAFFOLD                         │
│  (episodes, lessons, existing beliefs, reflection prompts)   │
│                    ↓                                         │
│  Agent reads scaffold and REASONS (in their own context)     │
│                    ↓                                         │
│  Agent decides: "I see pattern X, I should believe Y"        │
│                    ↓                                         │
│  Agent runs: kernle belief add "Y" --confidence 0.8          │
└─────────────────────────────────────────────────────────────┘
```

**Critical distinction**: The `consolidate` command does NOT call an external AI to analyze memories or suggest beliefs. It outputs structured data that helps the agent reflect using their own reasoning.

### Why Scaffolds, Not AI Analysis?

If Kernle used external AI to consolidate memories, it would be implanting beliefs:
- External model interprets your experiences
- Those interpretations become "your" beliefs
- You can't explain why you believe something
- Identity becomes externally determined

With scaffolds:
- Agent sees the raw data (episodes, lessons)
- Agent does their own pattern recognition
- Agent forms beliefs through their own reasoning
- Agent can always explain why they believe something

### Raw → Structured
```python
# Agent reviews raw captures
raw_list = kernle.list_raw(processed=False)

# Agent decides what each capture becomes
kernle.process_raw(raw_id, as_type="episode", objective="...", outcome="...")
kernle.process_raw(raw_id, as_type="note", type="decision")
kernle.process_raw(raw_id, as_type="belief", confidence=0.8)
```

### Episodes → Beliefs (Agent-Driven)
```python
# Get reflection scaffold for an episode
result = kernle.revise_beliefs_from_episode(episode_id)
# Returns scaffold: { episode_summary, existing_beliefs, reflection_prompts }

# Agent reads scaffold, does their own reasoning, then:
kernle.belief("Pattern I noticed", confidence=0.7)
```

### Pattern Extraction (Guided)
```bash
# Get reflection scaffold for consolidation
kernle consolidate --min-episodes 3

# Output: Structured prompt showing episodes, lessons, beliefs
# Agent reads it, reasons about patterns, then adds beliefs:
kernle belief add "Shortcuts create more work" --confidence 0.85
```

### Example Consolidation Session

```bash
# Step 1: Run consolidate
$ kernle -a claire consolidate

# Kernle outputs a REFLECTION SCAFFOLD:
# - Recent episodes with outcomes and lessons
# - Existing beliefs that might be relevant
# - Prompts to guide reflection

# Step 2: Agent reasons (happens in agent's context)
# "I see testing failures in 3 episodes. My testing belief needs reinforcement.
#  I also notice a new pattern about code review."

# Step 3: Agent adds beliefs based on THEIR reasoning
$ kernle -a claire belief add "Code review catches issues that tests miss" --confidence 0.8
$ kernle -a claire belief reinforce <testing-belief-id>

# Kernle stored what the agent decided. No AI analyzed the memories.
```

## Meta-Memory System

Every memory type has these meta-fields:

| Field | Description |
|-------|-------------|
| `confidence` | How certain we are (0.0-1.0) |
| `source_type` | How acquired: `direct_experience`, `inference`, `told_by_agent`, `consolidation` |
| `source_episodes` | Episode IDs that support this memory |
| `derived_from` | Memory refs this was derived from (`type:id`) |
| `last_verified` | When last confirmed |
| `verification_count` | Times verified |
| `confidence_history` | JSON array of confidence changes with timestamps |

### Key Operations
```python
kernle.verify_memory("belief", belief_id)  # Increases confidence
kernle.get_memory_lineage("belief", belief_id)  # Get provenance
kernle.get_uncertain_memories(threshold=0.5)  # Find weak memories
```

## Forgetting System

Kernle uses **tombstoning, not deletion**. Forgotten memories can be recovered.

### Salience Formula
```
salience = (confidence × log(times_accessed + 1)) / (days_since_access / half_life + 1)
```

- **High salience**: Confident, frequently accessed, recently used
- **Low salience**: Uncertain, rarely accessed, old

### Protection
- Values and Drives are protected by default
- Any memory can be marked protected: `kernle.protect("episode", id, True)`
- Protected memories never decay

### Forgetting Cycle
```python
# Preview what would be forgotten
candidates = kernle.get_forgetting_candidates(threshold=0.3)

# Run forgetting (dry_run=True to preview)
result = kernle.run_forgetting_cycle(threshold=0.3, dry_run=False)

# Recover a forgotten memory
kernle.recover("episode", episode_id)
```

## Anxiety & Health Metrics

The anxiety system tracks 5 dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Context Pressure | 35% | How full is context window |
| Unsaved Work | 25% | Time since last checkpoint |
| Consolidation Debt | 20% | Unreflected episodes count |
| Identity Coherence | 10% | Inverse of identity confidence |
| Memory Uncertainty | 10% | Low-confidence belief count |

**Composite Score**: 0-100, with levels: Calm (0-30), Aware (31-50), Elevated (51-70), High (71-85), Critical (86-100)

## Search Functionality

### Local Search
Uses sqlite-vec for semantic search when available, falls back to text matching.

### Cloud Hybrid Search
When cloud credentials are configured:
1. Try cloud search first (timeout: 3s)
2. Fall back to local on failure
3. Merge results by relevance score

```python
# Search across all memory types
results = kernle.search("topic", limit=10)

# Playbook-specific semantic search
playbook = kernle.find_playbook("situation description")
```

## Sync Architecture

**Local-First with Sync Queue**:
1. All changes written to local SQLite first
2. Changes queued in `sync_queue` table
3. Queue deduplicates by `(table, record_id)`
4. Push to cloud when online
5. Pull remote changes on `load()` if auto_sync enabled

**Conflict Resolution**: Last-write-wins based on `local_updated_at`
