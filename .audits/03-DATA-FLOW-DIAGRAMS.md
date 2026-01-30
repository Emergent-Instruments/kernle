# Kernle Data Flow Diagrams

> **Date**: 2026-01-29
> **Purpose**: Visual documentation of data movement through the system

---

## 1. Memory Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LAYER 0: RAW CAPTURES                                                  │
│  ════════════════════                                                   │
│  Purpose: Zero-friction brain dump                                      │
│  Storage: raw_entries table (proposed: blob field only)                 │
│  Lifespan: Short (days/weeks) - should be processed quickly             │
│  Search: Keyword/grep only (no embeddings)                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "Working on sync - realized queue needs deduplication..."      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ process_raw()                            │
│                              ▼                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1: EPISODIC MEMORY                                               │
│  ════════════════════════                                               │
│  Purpose: Autobiographical experiences with lessons                     │
│  Storage: episodes table                                                │
│  Lifespan: Long (months/years)                                          │
│  Search: Semantic + keyword                                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  objective: "Implement sync queue"                              │   │
│  │  outcome: "Discovered deduplication needed"                     │   │
│  │  lessons: ["Queue requires UPSERT pattern"]                     │   │
│  │  emotional_valence: 0.3 (slightly positive)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ consolidate()                            │
│                              ▼                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 2: SEMANTIC MEMORY                                               │
│  ════════════════════════                                               │
│  Purpose: Beliefs, facts, learned concepts                              │
│  Storage: beliefs, notes tables                                         │
│  Lifespan: Long with revision chains                                    │
│  Search: Semantic + keyword                                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BELIEFS:                                                        │   │
│  │  "Sync queues need deduplication to prevent race conditions"    │   │
│  │  confidence: 0.85, type: "fact"                                 │   │
│  │                                                                  │   │
│  │  NOTES:                                                          │   │
│  │  Decision: "Use UPSERT with partial unique index"               │   │
│  │  Reason: "Atomic operation, no SELECT-then-INSERT races"        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              │ synthesize_identity()                    │
│                              ▼                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 3: IDENTITY & VALUES                                             │
│  ══════════════════════════                                             │
│  Purpose: Core principles, self-concept, direction                      │
│  Storage: agent_values, goals, drives tables                            │
│  Lifespan: Permanent (protected by default)                             │
│  Authority: Highest - guides all decisions                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  VALUES:                                                         │   │
│  │  "Correctness over speed" (priority: 90)                        │   │
│  │  "Transparency in design decisions" (priority: 85)              │   │
│  │                                                                  │   │
│  │  GOALS:                                                          │   │
│  │  "Build reliable sync system" (status: active)                  │   │
│  │                                                                  │   │
│  │  DRIVES:                                                         │   │
│  │  growth: 0.8, curiosity: 0.7, existence: 0.6                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

SUPPORTING SYSTEMS:
├── playbooks (procedural memory - "how I do things")
├── relationships (models of other agents/people)
├── emotional_memory (valence/arousal on episodes)
├── meta_memory (confidence, provenance, verification)
└── forgetting (salience-based decay with tombstoning)
```

---

## 2. Memory Capture Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         AGENT (Source of All Data)                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Agent decides what to capture
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  ZERO FRICTION   │    │  STRUCTURED      │    │  IDENTITY        │
│  (Raw Capture)   │    │  (Direct Entry)  │    │  (Core)          │
│                  │    │                  │    │                  │
│  k.raw(blob)     │    │  k.episode(...)  │    │  k.value(...)    │
│                  │    │  k.note(...)     │    │  k.goal(...)     │
│                  │    │  k.belief(...)   │    │  k.drive(...)    │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  raw_entries     │    │  episodes        │    │  agent_values    │
│  ─────────────── │    │  notes           │    │  goals           │
│  • blob          │    │  beliefs         │    │  drives          │
│  • captured_at   │    │  ─────────────── │    │  ─────────────── │
│  • processed     │    │  • full schema   │    │  • is_protected=1│
│                  │    │  • embeddings    │    │  • embeddings    │
│  NO EMBEDDINGS   │    │  • meta-memory   │    │  • meta-memory   │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │     SYNC QUEUE       │
                    │  ────────────────    │
                    │  • table_name        │
                    │  • record_id         │
                    │  • operation         │
                    │  • data (JSON)       │
                    │  • synced = 0        │
                    └──────────┬───────────┘
                               │
                               │ When online
                               ▼
                    ┌──────────────────────┐
                    │    CLOUD SYNC        │
                    │  (PostgreSQL)        │
                    └──────────────────────┘
```

---

## 3. Raw Entry Processing (Promotion)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           RAW ENTRY                                      │
│  ────────────────────────────────────────────────────────────────────    │
│  blob: "Debugging auth flow - realized JWKS is more reliable than       │
│         API key verification. Supabase disabled legacy keys. Frustrating │
│         but learned a lot about JWT verification patterns."              │
│  captured_at: 2026-01-29T14:30:00Z                                       │
│  processed: 0                                                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Agent reviews and decides
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  AS EPISODE      │    │  AS NOTE         │    │  AS BELIEF       │
│  ────────────────│    │  ────────────────│    │  ────────────────│
│                  │    │                  │    │                  │
│  objective:      │    │  type: decision  │    │  statement:      │
│  "Debug auth"    │    │  content:        │    │  "JWKS is more   │
│                  │    │  "Use JWKS not   │    │   reliable than  │
│  outcome:        │    │   API keys"      │    │   API key auth"  │
│  "JWKS works"    │    │                  │    │                  │
│                  │    │  reason:         │    │  confidence: 0.8 │
│  lessons:        │    │  "API keys can   │    │  type: fact      │
│  ["JWKS > API"]  │    │   be disabled"   │    │                  │
│                  │    │                  │    │                  │
│  emotional:      │    │  tags: [auth,    │    │                  │
│  valence: 0.2    │    │   security]      │    │                  │
│  arousal: 0.6    │    │                  │    │                  │
│  tags: [frustrat │    │                  │    │                  │
│         ion]     │    │                  │    │                  │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RAW ENTRY (updated)                                                     │
│  ────────────────────────────────────────────────────────────────────    │
│  processed: 1                                                            │
│  processed_into: ["episode:abc123", "note:def456", "belief:ghi789"]      │
└──────────────────────────────────────────────────────────────────────────┘

NOTE: One raw entry can produce multiple structured memories.
      The raw blob is preserved for audit trail.
```

---

## 4. Sync Flow (Local ↔ Cloud)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LOCAL (SQLite)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│  │   episodes    │     │    beliefs    │     │    notes      │         │
│  │   (384-dim    │     │   (384-dim    │     │   (384-dim    │         │
│  │    hash emb)  │     │    hash emb)  │     │    hash emb)  │         │
│  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘         │
│          │                     │                     │                  │
│          │ On save: _queue_sync()                    │                  │
│          ▼                     ▼                     ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        SYNC QUEUE                               │   │
│  │  ───────────────────────────────────────────────────────────    │   │
│  │  id │ table_name │ record_id │ operation │ data    │ synced    │   │
│  │  1  │ episodes   │ abc123    │ upsert    │ {...}   │ 0         │   │
│  │  2  │ beliefs    │ def456    │ upsert    │ {...}   │ 0         │   │
│  │  ───────────────────────────────────────────────────────────    │   │
│  │  UPSERT deduplicates: only latest change per (table, record_id) │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
                                     │ sync() called (manual or auto)
                                     │ is_online() check (cached 30s)
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │         HTTPS / REST           │
                    │    POST /sync/push             │
                    │    POST /sync/pull             │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLOUD (PostgreSQL)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Validate agent_id from JWT                                          │
│  2. Strip server-controlled fields (prevents mass assignment)           │
│  3. Generate OpenAI embedding (1536-dim)                                │
│  4. Upsert to PostgreSQL                                                │
│  5. Index in pgvector                                                   │
│                                                                         │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│  │   episodes    │     │    beliefs    │     │   memories    │         │
│  │   (1536-dim   │     │   (1536-dim   │     │   (1536-dim   │         │
│  │    OpenAI)    │     │    OpenAI)    │     │    OpenAI)    │         │
│  └───────────────┘     └───────────────┘     └───────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

CONFLICT RESOLUTION: Last-Write-Wins
────────────────────────────────────
• Compare: cloud_synced_at vs local_updated_at
• Cloud newer → Overwrite local (record conflict)
• Local newer → Keep local (will push on next sync)
• Conflicts logged to sync_conflicts table for visibility
```

---

## 5. Forgetting / Retention Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ALL MEMORIES                                   │
│                     (episodes, beliefs, notes, etc.)                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     SALIENCE CALCULATION     │
                    │  ────────────────────────    │
                    │                              │
                    │  salience = (confidence ×    │
                    │              reinforcement)  │
                    │             / (age + 1)      │
                    │                              │
                    │  reinforcement = log(access+1)│
                    │  age = days / half_life(30) │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                    ▼                              ▼
        ┌───────────────────┐          ┌───────────────────┐
        │  salience > 0.3   │          │  salience ≤ 0.3   │
        │  ───────────────  │          │  ───────────────  │
        │                   │          │                   │
        │  RETAIN           │          │  CANDIDATE FOR    │
        │  (no action)      │          │  FORGETTING       │
        │                   │          │                   │
        └───────────────────┘          └─────────┬─────────┘
                                                 │
                                                 ▼
                                   ┌──────────────────────────┐
                                   │  PROTECTION CHECK        │
                                   │  ──────────────────      │
                                   │  is_protected = 1?       │
                                   └──────────────┬───────────┘
                                                  │
                                   ┌──────────────┴───────────┐
                                   │                          │
                                   ▼                          ▼
                       ┌───────────────────┐      ┌───────────────────┐
                       │  PROTECTED        │      │  NOT PROTECTED    │
                       │  ───────────      │      │  ───────────────  │
                       │                   │      │                   │
                       │  NEVER FORGET     │      │  TOMBSTONE        │
                       │  (values, drives) │      │  ────────────     │
                       │                   │      │  is_forgotten = 1 │
                       └───────────────────┘      │  forgotten_at = now│
                                                  │  forgotten_reason │
                                                  │                   │
                                                  │  (RECOVERABLE)    │
                                                  └───────────────────┘

CONFIDENCE DECAY (separate from forgetting):
────────────────────────────────────────────
Type-specific decay rates applied over time:
• episodes: -1% per 30 days, floor 0.5
• beliefs:  -1% per 30 days, floor 0.5
• values:   -0.5% per 60 days, floor 0.7 (slow decay)
• notes:    -1.5% per 30 days, floor 0.4 (faster decay)
• drives:   -0.5% per 60 days, floor 0.6 (slow decay)

Protected memories don't decay.
Verification resets decay timer.
```

---

## 6. Anxiety / Health Monitoring Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ANXIETY SYSTEM                                 │
│                   (Memory Health Monitoring)                            │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  SIX DIMENSIONS (weighted composite score)                            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 1. CONTEXT PRESSURE (30%)                                       │ │
│  │    ──────────────────────                                       │ │
│  │    Measures: Estimated token usage vs limit                     │ │
│  │    Source: Session tracking (tokens per minute estimate)        │ │
│  │    Critical at: >80% of context limit                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 2. UNSAVED WORK (20%)                                           │ │
│  │    ─────────────────                                            │ │
│  │    Measures: Minutes since last checkpoint                      │ │
│  │    Source: checkpoint.timestamp vs now                          │ │
│  │    Critical at: >60 minutes                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 3. CONSOLIDATION DEBT (15%)                                     │ │
│  │    ───────────────────────                                      │ │
│  │    Measures: Count of unreflected episodes                      │ │
│  │    Source: Episodes without lessons                             │ │
│  │    Urgent at: >15 episodes                                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 4. RAW AGING (15%)                                              │ │
│  │    ────────────────                                             │ │
│  │    Measures: Unprocessed raw entries >24 hours old              │ │
│  │    Source: raw_entries WHERE processed=0 AND age>24h            │ │
│  │    Critical at: 7+ aging entries                                │ │
│  │                                                                 │ │
│  │    PROPOSED ADDITION: Blob size check                           │ │
│  │    Alert if any raw entry >100KB                                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 5. IDENTITY COHERENCE (10%)                                     │ │
│  │    ───────────────────────                                      │ │
│  │    Measures: Self-model confidence (inverted)                   │ │
│  │    Source: synthesize_identity() confidence score               │ │
│  │    Weak at: <50% confidence                                     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ 6. MEMORY UNCERTAINTY (10%)                                     │ │
│  │    ───────────────────────                                      │ │
│  │    Measures: Count of low-confidence beliefs (<0.5)             │ │
│  │    Source: beliefs WHERE confidence < 0.5                       │ │
│  │    High at: >5 uncertain beliefs                                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│  COMPOSITE SCORE (0-100)                                              │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   0-30   │ CALM      │ Continue normally                              │
│  31-50   │ AWARE     │ Checkpoint soon                                │
│  51-70   │ ELEVATED  │ Full checkpoint + consolidate                  │
│  71-85   │ HIGH      │ Priority memory work                           │
│  86-100  │ CRITICAL  │ Emergency save protocol                        │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 7. Entry Points Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      KERNLE ENTRY POINTS                                │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  1. CLI (Command Line Interface)                                        │
│     ─────────────────────────────                                       │
│     kernle -a <agent_id> <command> [options]                            │
│                                                                         │
│     Primary commands:                                                   │
│     • load          - Load working memory                               │
│     • checkpoint    - Save/load/clear working state                     │
│     • episode       - Record experiences                                │
│     • note          - Capture decisions/insights                        │
│     • belief        - Add/manage beliefs                                │
│     • value/goal    - Identity management                               │
│     • raw           - Raw capture layer                                 │
│     • sync          - Cloud synchronization                             │
│     • anxiety       - Health monitoring                                 │
│     • consolidate   - Pattern recognition scaffold                      │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ All use same core
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. MCP (Model Context Protocol)                                        │
│     ────────────────────────────                                        │
│     Configured via .mcp.json                                            │
│     28 tools exposed to Claude/other MCP clients                        │
│                                                                         │
│     Tool naming: memory_<operation>                                     │
│     • memory_load, memory_checkpoint_save/load                          │
│     • memory_episode, memory_note, memory_belief                        │
│     • memory_search, memory_when                                        │
│     • memory_auto_capture (raw + suggestions)                           │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ All use same core
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. Python SDK                                                          │
│     ──────────                                                          │
│     from kernle import Kernle                                           │
│     k = Kernle(agent_id="my_agent")                                     │
│                                                                         │
│     Direct method calls:                                                │
│     • k.load(), k.checkpoint(), k.episode(), k.note()                   │
│     • k.belief(), k.value(), k.goal(), k.drive()                        │
│     • k.raw(), k.process_raw(), k.search()                              │
│     • k.sync(), k.get_anxiety_report()                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         KERNLE CORE                                     │
│                    (kernle/core.py)                                     │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                         │
│  Mixins:                                                                │
│  • AnxietyMixin     - Health monitoring                                 │
│  • EmotionsMixin    - Emotional memory                                  │
│  • ForgettingMixin  - Salience-based retention                          │
│  • KnowledgeMixin   - Knowledge mapping                                 │
│  • MetaMemoryMixin  - Confidence/provenance                             │
│  • SuggestionsMixin - Raw promotion suggestions                         │
│                                                                         │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      STORAGE ABSTRACTION                                │
│                   (kernle/storage/base.py)                              │
│  ────────────────────────────────────────────────────────────────────   │
│                                                                         │
│  Implementations:                                                       │
│  • SQLiteStorage   - Local-first (kernle/storage/sqlite.py)             │
│  • PostgresStorage - Cloud (kernle/storage/postgres.py)                 │
│                                                                         │
│  Auto-detection via get_storage() factory                               │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Session Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       SESSION LIFECYCLE                                 │
└─────────────────────────────────────────────────────────────────────────┘

SESSION START
═════════════
     │
     ▼
┌─────────────────────────────────────────┐
│  k.load(budget=8000)                    │
│  ─────────────────────                  │
│  1. Load checkpoint (if exists)         │
│  2. Load memories by priority:          │
│     values > beliefs > goals > ...      │
│  3. Budget-aware truncation             │
│  4. Optional: pull from cloud           │
└─────────────────────────────────────────┘
     │
     ▼
DURING SESSION
══════════════
     │
     ▼
┌─────────────────────────────────────────┐
│  Agent works, captures memories         │
│  ─────────────────────────────          │
│  • k.raw("quick thought")               │
│  • k.episode("task", "outcome")         │
│  • k.note("decision: X", type="dec")    │
│  • k.belief("learned: Y")               │
│                                         │
│  Periodic anxiety checks                │
│  Auto-save if configured                │
└─────────────────────────────────────────┘
     │
     ▼
SESSION END
═══════════
     │
     ▼
┌─────────────────────────────────────────┐
│  k.checkpoint(task, pending, context)   │
│  ─────────────────────────────────────  │
│  1. Save working state                  │
│  2. Optional: k.sync() to cloud         │
│  3. State preserved for next session    │
└─────────────────────────────────────────┘
     │
     ▼
BETWEEN SESSIONS
════════════════
     │
     ▼
┌─────────────────────────────────────────┐
│  Background processes (optional)        │
│  ─────────────────────────────          │
│  • Confidence decay over time           │
│  • Forgetting cycle (if enabled)        │
│  • Cloud sync (if online)               │
└─────────────────────────────────────────┘
```

---

## Summary: What Flows Where

| Data Type | Entry | Storage | Sync | Search | Retention |
|-----------|-------|---------|------|--------|-----------|
| Raw | k.raw() | SQLite blob | Yes | Keyword | Until processed |
| Episode | k.episode() | SQLite + embeddings | Yes | Semantic | Salience-based |
| Note | k.note() | SQLite + embeddings | Yes | Semantic | Salience-based |
| Belief | k.belief() | SQLite + embeddings | Yes | Semantic | Salience-based |
| Value | k.value() | SQLite + embeddings | Yes | Semantic | Protected (permanent) |
| Goal | k.goal() | SQLite + embeddings | Yes | Semantic | Salience-based |
| Drive | k.drive() | SQLite + embeddings | Yes | Semantic | Protected (permanent) |
| Relationship | k.relationship() | SQLite + embeddings | Yes | Semantic | Salience-based |
| Playbook | k.playbook() | SQLite + embeddings | Yes | Semantic | Salience-based |
