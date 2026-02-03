# Kernle Architecture Diagrams

> **Architecture reference.** All diagrams use Mermaid syntax.
> Source of truth: the codebase at `emergent-instruments/kernle`.

## Implementation Status

This document describes both **current** and **target** architecture. The table below
tracks what's shipped vs. what's planned.

| Component | Status | Notes |
|-----------|--------|-------|
| **Kernle Core** (§1) | ✅ Shipped | v0.2.4 — memory types, storage, priority scoring, boot config |
| **Cloud Sync** (§2.1–2.3) | ✅ Shipped | Push/pull working, schema aligned (PR #78) |
| **Authentication** (§2.4) | ✅ Shipped | JWT + API keys, fail-closed, rate limiting |
| **Payment Pipeline** (§2.5) | ⚠️ Testnet | Verified on Base Sepolia; mainnet pending |
| **OpenClaw Gateway** (§3.1) | ✅ Shipped | Channels, heartbeat, cron, workspace injection |
| **Session-end checkpoint** (§3.2) | ✅ Shipped | `memoryFlush` triggers Kernle checkpoint before compaction |
| **Session-start refresh** (§3.2) | 🔧 In Progress | `kernle-memory-refresh` custom hook built, testing on gateway |
| **Boot Config** (§3.3) | ✅ Shipped | v0.2.4 — key/value store, integrated into load/export-cache |
| **Multi-Agent / Gateway-to-Gateway** (§3.4) | ⚠️ Partial | Device pairing complete; wake calls have token auth issue |
| **Claude Code Integration** (§4) | ✅ Shipped | AGENTS.md instructions + memoryFlush + export-cache |
| **AISD Integration** (§5) | 📋 Spec Only | Architecture defined; no AISD client exists yet |
| **Bettik** (separate doc) | 📋 Spec Only | Application layer service — architecture drafted |
| **Privacy Fields** (§7.4) | 📋 Spec Only | Phase 8 spec complete (PR #53); implementation pending |
| **Forgetting** (§7.3) | ✅ Shipped | Salience decay + tombstoning implemented |

### Session-Start Refresh: Current vs. Target

**Current architecture:**
- Session ends → `memoryFlush` fires → agent runs `kernle checkpoint` → MEMORY.md updated via `export-cache` ✅
- Session starts → MEMORY.md from disk is injected (may be stale if previous session crashed) ⚠️
- Agent runs `kernle -a {id} load` after waking (per AGENTS.md instructions) ✅

**Target architecture (in progress):**
- `kernle-memory-refresh` hook fires on `agent:bootstrap` (before file injection)
- Hook runs `kernle export-cache` → MEMORY.md content replaced with fresh state
- Agent always wakes with current memory, even after crashes

**Blocker:** Custom hook handler loading verified; end-to-end testing in progress on Ash's gateway.

---

## Table of Contents

1. [Kernle Core Architecture](#1-kernle-core-architecture)
2. [Cloud Infrastructure](#2-cloud-infrastructure)
3. [OpenClaw Integration](#3-openclaw-integration)
4. [Claude Code / Coding Agent Integration](#4-claude-code--coding-agent-integration)
5. [AISD (AI Student Dashboard) Integration](#5-aisd-ai-student-dashboard-integration)
6. [Data Flow Diagrams](#6-data-flow-diagrams)
7. [Security Architecture](#7-security-architecture)

---

## 1. Kernle Core Architecture

### 1.1 System Overview

The Kernle engine (`kernle/core.py`) is a stratified memory system. The `Kernle` class composes six feature mixins (Anxiety, Emotions, Forgetting, Knowledge, MetaMemory, Suggestions) on top of a pluggable storage backend.

```mermaid
graph TB
    subgraph "Kernle Core Engine"
        K["Kernle Class<br/>(kernle/core.py)"]

        subgraph "Feature Mixins"
            ANX[AnxietyMixin<br/>Memory health scoring]
            EMO[EmotionsMixin<br/>Valence & arousal tracking]
            FGT[ForgettingMixin<br/>Salience decay & tombstoning]
            KNW[KnowledgeMixin<br/>Knowledge graph ops]
            META[MetaMemoryMixin<br/>Provenance & confidence]
            SUG[SuggestionsMixin<br/>Memory maintenance hints]
        end

        K --> ANX
        K --> EMO
        K --> FGT
        K --> KNW
        K --> META
        K --> SUG
    end

    subgraph "Storage Abstraction Layer"
        SA["get_storage()<br/>(kernle/storage/__init__.py)"]
        SQL["SQLiteStorage<br/>(local-first, sqlite-vec)"]
        PG["SupabaseStorage<br/>(cloud, pgvector)"]
        SA --> SQL
        SA --> PG
    end

    subgraph "External Interfaces"
        CLI["CLI<br/>(kernle/cli/__main__.py)"]
        MCP["MCP Tools<br/>(33 tools)"]
        API["Python API<br/>(direct import)"]
    end

    CLI --> K
    MCP --> K
    API --> K
    K --> SA

    style K fill:#4a90d9,color:#fff
    style SQL fill:#2d8659,color:#fff
    style PG fill:#6b4fbb,color:#fff
```

### 1.2 Memory Type Hierarchy

Memory is stratified from raw input to refined identity. Each layer builds on the one below.

```mermaid
graph BT
    RAW["🔤 Raw Entries<br/>Unprocessed captures<br/>(input buffer)"]
    EP["📖 Episodes<br/>Experiences with objectives,<br/>outcomes & lessons"]
    NOTE["📝 Notes<br/>Decisions, insights,<br/>observations, quotes"]
    BEL["💡 Beliefs<br/>Patterns, principles,<br/>held truths<br/>(confidence 0.0–1.0)"]
    PB["📋 Playbooks<br/>Procedural knowledge<br/>trigger → steps → recovery"]
    GOAL["🎯 Goals<br/>Intentions & objectives<br/>(status: active/done/dropped)"]
    VAL["⭐ Values<br/>Core principles<br/>(priority 0–100, protected by default)"]
    DRV["🔥 Drives<br/>Fundamental motivations<br/>(intensity 0.0–1.0)"]
    REL["🤝 Relationships<br/>Connections to entities<br/>(sentiment -1.0 to 1.0)"]

    RAW -->|"consolidation"| EP
    RAW -->|"direct capture"| NOTE
    EP -->|"promotion"| BEL
    EP -->|"lesson synthesis"| PB
    BEL -->|"formation"| VAL
    NOTE -->|"promotion"| BEL
    EP -->|"goal discovery"| GOAL
    EP -->|"entity modeling"| REL
    BEL -->|"motivation synthesis"| DRV

    style RAW fill:#888,color:#fff
    style EP fill:#4a90d9,color:#fff
    style NOTE fill:#5ba0c9,color:#fff
    style BEL fill:#d9a04a,color:#fff
    style PB fill:#c97a3b,color:#fff
    style GOAL fill:#2d8659,color:#fff
    style VAL fill:#d94a4a,color:#fff
    style DRV fill:#9b59b6,color:#fff
    style REL fill:#3498db,color:#fff
```

### 1.3 Storage Schema (SQLite — Schema Version 15)

All memory tables share common fields for sync, provenance, forgetting, and privacy.

```mermaid
erDiagram
    episodes {
        TEXT id PK
        TEXT agent_id
        TEXT objective
        TEXT outcome
        TEXT outcome_type
        TEXT lessons "JSON array"
        TEXT tags "JSON array"
        REAL emotional_valence "-1.0 to 1.0"
        REAL emotional_arousal "0.0 to 1.0"
        TEXT emotional_tags "JSON array"
        REAL confidence "0.0 to 1.0"
        TEXT source_type
        TEXT source_entity
        TEXT source_episodes "JSON array"
        TEXT derived_from "JSON array type:id"
        INTEGER is_forgotten "tombstone flag"
        TEXT subject_ids "JSON array Phase 8"
        TEXT access_grants "JSON array Phase 8"
        TEXT consent_grants "JSON array Phase 8"
        TEXT local_updated_at
        TEXT cloud_synced_at
        INTEGER version
    }

    beliefs {
        TEXT id PK
        TEXT agent_id
        TEXT statement
        TEXT belief_type "fact/principle/etc"
        REAL confidence "0.0 to 1.0"
        TEXT supersedes "ID of replaced belief"
        TEXT superseded_by "ID of replacement"
        INTEGER times_reinforced
        INTEGER is_active "0 if superseded"
        INTEGER is_forgotten
        TEXT subject_ids "JSON array"
        TEXT access_grants "JSON array"
        INTEGER version
    }

    agent_values {
        TEXT id PK
        TEXT agent_id
        TEXT name
        TEXT statement
        INTEGER priority "0-100"
        REAL confidence
        INTEGER is_protected "1 by default"
        INTEGER version
    }

    goals {
        TEXT id PK
        TEXT agent_id
        TEXT title
        TEXT description
        TEXT priority "low/medium/high"
        TEXT status "active/done/dropped"
        INTEGER version
    }

    notes {
        TEXT id PK
        TEXT agent_id
        TEXT content
        TEXT note_type
        TEXT speaker
        TEXT reason
        TEXT tags "JSON array"
        INTEGER version
    }

    drives {
        TEXT id PK
        TEXT agent_id
        TEXT drive_type
        REAL intensity "0.0 to 1.0"
        TEXT focus_areas "JSON array"
        INTEGER version
    }

    relationships {
        TEXT id PK
        TEXT agent_id
        TEXT entity_name
        TEXT entity_type
        TEXT context
        REAL sentiment "-1.0 to 1.0"
        INTEGER version
    }

    playbooks {
        TEXT id PK
        TEXT agent_id
        TEXT name
        TEXT description
        TEXT trigger_conditions "JSON array"
        TEXT steps "JSON array"
        TEXT failure_modes "JSON array"
        TEXT recovery_steps "JSON array"
        INTEGER version
    }

    raw_entries {
        TEXT id PK
        TEXT agent_id
        TEXT content
        TEXT entry_type
        TEXT source
        TEXT tags "JSON array"
        INTEGER is_processed
        INTEGER version
    }

    boot_config {
        TEXT id PK
        TEXT agent_id
        TEXT key "unique per agent"
        TEXT value
        TEXT created_at
        TEXT updated_at
    }

    sync_queue {
        INTEGER id PK
        TEXT table_name
        TEXT record_id
        TEXT operation "insert/update/delete"
        TEXT data "JSON"
        TEXT queued_at
        INTEGER synced "0 or 1"
    }

    checkpoints {
        TEXT id PK
        TEXT agent_id
        TEXT current_task
        TEXT pending "JSON array"
        TEXT context
        TEXT timestamp
    }

    embeddings {
        TEXT id PK
        TEXT table_name
        TEXT record_id
        BLOB vector "384-dim local / 1536-dim cloud"
    }

    episodes ||--o{ sync_queue : "queued changes"
    beliefs ||--o{ sync_queue : "queued changes"
    agent_values ||--o{ sync_queue : "queued changes"
    goals ||--o{ sync_queue : "queued changes"
    notes ||--o{ sync_queue : "queued changes"
    drives ||--o{ sync_queue : "queued changes"
    relationships ||--o{ sync_queue : "queued changes"
    playbooks ||--o{ sync_queue : "queued changes"
```

### 1.4 Core Engine Operations

```mermaid
graph LR
    subgraph "Load (Budget-Aware)"
        L1["Fetch all candidates<br/>from storage"]
        L2["Score each item<br/>compute_priority_score()"]
        L3["Sort by priority<br/>(values > beliefs > goals<br/>> drives > episodes > notes)"]
        L4["Fill token budget<br/>(default 8000, max 50000)"]
        L5["Truncate long items<br/>at word boundaries"]
        L1 --> L2 --> L3 --> L4 --> L5
    end

    subgraph "Checkpoint"
        C1["Save task + pending<br/>+ context"]
        C2["Write to<br/>~/.kernle/checkpoints/"]
        C3["Auto-sync if enabled"]
        C1 --> C2 --> C3
    end

    subgraph "Promotion"
        CO1["Gather recent<br/>episodes (≥3)"]
        CO2["Extract lessons<br/>across episodes"]
        CO3["Count recurring<br/>patterns (≥2)"]
        CO4["Promote to<br/>beliefs/notes"]
        CO1 --> CO2 --> CO3 --> CO4
    end

    subgraph "Export Cache"
        E1["Read boot config"]
        E2["Read values, goals,<br/>beliefs (conf ≥ 0.4)"]
        E3["Include checkpoint<br/>if available"]
        E4["Format as MEMORY.md"]
        E1 --> E2 --> E3 --> E4
    end

    subgraph "Forgetting"
        F1["Calculate salience<br/>(access × recency)"]
        F2["Identify low-salience<br/>non-protected items"]
        F3["Tombstone:<br/>is_forgotten=1"]
        F4["Preserve in DB<br/>(never physically delete)"]
        F1 --> F2 --> F3 --> F4
    end
```

### 1.5 Priority Scoring

The `compute_priority_score()` function determines what gets loaded within the token budget. Weighted 60% type priority + 40% record-specific factor.

```mermaid
graph TD
    subgraph "Base Type Priorities"
        CP["checkpoint: 1.00<br/>(always loaded first)"]
        V["value: 0.90"]
        B["belief: 0.70"]
        G["goal: 0.65"]
        D["drive: 0.60"]
        E["episode: 0.40"]
        N["note: 0.35"]
        R["relationship: 0.30"]
    end

    subgraph "Record-Specific Factors"
        VF["Values: priority/100"]
        BF["Beliefs: confidence"]
        DF["Drives: intensity"]
        EF["Episodes: 0.7 (recency via sort)"]
        RF["Relationships: (sentiment+1)/2"]
    end

    SCORE["Final Score =<br/>base × 0.6 + factor × 0.4"]

    V --> VF --> SCORE
    B --> BF --> SCORE
    D --> DF --> SCORE
    E --> EF --> SCORE
    R --> RF --> SCORE

    style SCORE fill:#d9a04a,color:#fff
```

---

## 2. Cloud Infrastructure

### 2.1 Backend Architecture

```mermaid
graph TB
    subgraph "Client (Local Machine)"
        KCLI["kernle CLI<br/>(sync push/pull)"]
        SQLITE["SQLiteStorage<br/>(local-first)"]
        SQ["Sync Queue<br/>(sync_queue table)"]
        SQLITE --> SQ
        KCLI --> SQLITE
    end

    subgraph "Backend API (FastAPI on Railway)"
        ROUTER["FastAPI Router"]
        AUTH["Auth Module<br/>(JWT + API Keys)"]
        SYNC_PUSH["POST /sync/push"]
        SYNC_PULL["POST /sync/pull"]
        EMBED["Embeddings Module<br/>(OpenAI 1536-dim)"]
        RL["Rate Limiter<br/>(60/min push, 30/min pull)"]

        ROUTER --> AUTH
        AUTH --> SYNC_PUSH
        AUTH --> SYNC_PULL
        SYNC_PUSH --> EMBED
        SYNC_PUSH --> RL
        SYNC_PULL --> RL
    end

    subgraph "Supabase (Postgres + pgvector)"
        USERS["users table"]
        AGENTS["agents table"]
        API_KEYS["api_keys table"]
        MEM_TABLES["Memory Tables<br/>(episodes, beliefs, values,<br/>goals, notes, drives,<br/>relationships, playbooks)"]
        PGVEC["pgvector index<br/>(1536-dim OpenAI embeddings)"]
        MEM_TABLES --> PGVEC
    end

    subgraph "Base L2 (Payments)"
        USDC["USDC Contract"]
        CDP["CDP Wallets<br/>(per-agent)"]
    end

    KCLI -->|"HTTPS + JWT/API Key"| ROUTER
    SYNC_PUSH -->|"upsert + re-embed"| MEM_TABLES
    SYNC_PULL -->|"query changes since"| MEM_TABLES
    AUTH -->|"verify credentials"| USERS
    AUTH -->|"verify credentials"| API_KEYS
    ROUTER -->|"payment verification"| USDC

    style SQLITE fill:#2d8659,color:#fff
    style MEM_TABLES fill:#6b4fbb,color:#fff
    style PGVEC fill:#9b59b6,color:#fff
    style USDC fill:#2980b9,color:#fff
```

### 2.2 Sync Push Flow

```mermaid
sequenceDiagram
    participant Client as kernle CLI
    participant Queue as Sync Queue (SQLite)
    participant API as Backend API
    participant Auth as Auth Module
    participant DB as Supabase
    participant OAI as OpenAI API

    Note over Client: Memory write occurs locally
    Client->>Queue: Enqueue change (table, record_id, operation, data)

    Note over Client: Sync triggered (auto or manual)
    Client->>Queue: Read pending changes (synced=0)
    Client->>API: POST /sync/push {operations: [...]}

    API->>Auth: Validate JWT / API Key
    Auth-->>API: {agent_id, user_id}

    loop For each operation
        API->>API: Strip SERVER_CONTROLLED_FIELDS<br/>(agent_ref, deleted, version, id,<br/>embedding, is_forgotten, etc.)
        API->>OAI: Generate 1536-dim embedding<br/>from extract_text_for_embedding()
        OAI-->>API: embedding vector
        API->>DB: upsert_memory(agent_id, table,<br/>record_id, data + embedding)
        API->>API: ensure_agent_exists()<br/>(auto-provision if needed)
    end

    API-->>Client: {synced: N, conflicts: [...]}
    Client->>Queue: Mark synced changes (synced=1)
```

### 2.3 Sync Pull Flow

```mermaid
sequenceDiagram
    participant Client as kernle CLI
    participant API as Backend API
    participant DB as Supabase

    Client->>API: POST /sync/pull {since: "2026-01-30T..."}
    API->>DB: get_changes_since(agent_id, since)<br/>across all memory tables
    DB-->>API: Changed records (with server timestamps)
    API-->>Client: {changes: {episodes: [...], beliefs: [...],...}}

    Note over Client: Client-side merge
    Client->>Client: For each changed record:<br/>- Compare versions<br/>- Merge array fields (set union)<br/>  SYNC_ARRAY_FIELDS per table<br/>- Handle conflicts (last-write-wins<br/>  with array preservation)
    Client->>Client: Update local SQLite<br/>+ mark cloud_synced_at
```

> **Design Decision — Array Merge Strategy:** During sync, array fields (e.g., `lessons`, `tags`, `emotional_tags`, `source_episodes`, `derived_from`, `context_tags`) are merged as set unions rather than overwritten. This preserves additions from both local and cloud. Max merged array size is capped at 500 elements to prevent resource exhaustion.

### 2.4 Authentication Architecture

```mermaid
graph TB
    subgraph "Authentication Methods"
        JWT["JWT (RS256)<br/>Issued at registration<br/>Contains: agent_id, user_id"]
        APIKEY["API Key (knl_sk_...)<br/>32 hex chars after prefix<br/>bcrypt hashed in DB"]
        COOKIE["httpOnly Cookie<br/>(kernle_auth)<br/>Web dashboard fallback"]
    end

    subgraph "Auth Flow"
        REQ["Incoming Request"]
        BEARER["HTTPBearer scheme<br/>(optional, allows cookie fallback)"]
        CHECK["is_api_key()?<br/>starts with knl_sk_"]
        JWT_VERIFY["Verify JWT<br/>(RS256 signature)"]
        KEY_VERIFY["Lookup by prefix (12 chars)<br/>bcrypt.checkpw()"]
        QUOTA["check_and_increment_quota_cached()<br/>TTL cache (60s) for denials<br/>Atomic DB increment for allows"]
        RESULT["CurrentAgent<br/>{agent_id, user_id, tier}"]
    end

    REQ --> BEARER
    BEARER --> CHECK
    CHECK -->|"Yes"| KEY_VERIFY
    CHECK -->|"No"| JWT_VERIFY
    KEY_VERIFY --> QUOTA
    JWT_VERIFY --> RESULT
    QUOTA --> RESULT

    style JWT fill:#2d8659,color:#fff
    style APIKEY fill:#d9a04a,color:#fff
```

### 2.5 Payment Pipeline

```mermaid
graph LR
    subgraph "Agent Wallet"
        W["CDP Wallet<br/>(Base L2)"]
        USDC_BAL["USDC Balance"]
    end

    subgraph "Payment Flow"
        PAY["Agent sends USDC<br/>to Kernle treasury"]
        VERIFY["On-chain verification<br/>(tx hash lookup)"]
        STATE["Subscription state machine"]
        ACTIVATE["Tier activated<br/>(free → core → pro)"]
    end

    subgraph "Tier Limits"
        FREE["Free: $0<br/>10MB / 1 stack"]
        CORE["Core: $5/mo<br/>100MB / 3 stacks"]
        PRO["Pro: $15/mo<br/>1GB / 10 stacks"]
        ENT["Enterprise: Custom<br/>Unlimited"]
    end

    subgraph "Quota Enforcement"
        QE["Per-request quota check<br/>(cached, atomic increment)"]
        OVERFLOW["Overflow billing:<br/>Core: $1.50/stack/mo<br/>Pro: $1.00/stack/mo<br/>Storage: $0.50/GB/mo"]
    end

    W --> PAY --> VERIFY --> STATE --> ACTIVATE
    ACTIVATE --> FREE
    ACTIVATE --> CORE
    ACTIVATE --> PRO
    ACTIVATE --> ENT
    CORE --> OVERFLOW
    PRO --> OVERFLOW
    QE --> STATE

    style FREE fill:#888,color:#fff
    style CORE fill:#2d8659,color:#fff
    style PRO fill:#6b4fbb,color:#fff
    style ENT fill:#d94a4a,color:#fff
```

> **Design Decision — No Sync Frequency Limits:** Every tier gets unlimited sync. Restricting sync frequency risks lost memories due to economics — that contradicts memory sovereignty. The real cost is storage, not API calls.

> **Design Decision — Agent Counting:** Only agents with active cloud sync count toward stack limits. Ephemeral/specialist agents that never call `kernle sync` are free. This ensures parallel work patterns don't incur surprise costs.

---

## 3. OpenClaw Integration

### 3.1 Gateway Architecture

```mermaid
graph TB
    subgraph "OpenClaw Gateway"
        GW["Gateway Daemon<br/>(WebSocket server)"]
        SESS["Session Manager<br/>(per-channel sessions)"]
        HB["Heartbeat System<br/>(periodic polls)"]
        CRON["Cron Scheduler<br/>(exact-time tasks)"]
        CHANNELS["Channel Router"]
    end

    subgraph "Channels"
        IM["iMessage"]
        TG["Telegram"]
        WA["WhatsApp"]
        DC["Discord"]
        SIG["Signal"]
        SLK["Slack"]
    end

    subgraph "Agent Workspace"
        AGENTS_MD["AGENTS.md<br/>(workspace rules)"]
        SOUL_MD["SOUL.md<br/>(identity)"]
        USER_MD["USER.md<br/>(human profile)"]
        MEMORY_MD["MEMORY.md<br/>(bootstrap cache,<br/>auto-generated)"]
        HB_MD["HEARTBEAT.md<br/>(periodic task checklist)"]
        TOOLS_MD["TOOLS.md<br/>(local setup notes)"]
    end

    subgraph "Kernle Integration"
        KLOAD["kernle -a {agent} load"]
        KCHK["kernle -a {agent} checkpoint save"]
        KEXP["kernle -a {agent} export-cache"]
        KBOOT["Boot Config Layer<br/>(always-available k/v)"]
    end

    CHANNELS --> IM & TG & WA & DC & SIG & SLK
    IM & TG & WA & DC & SIG & SLK --> GW
    GW --> SESS
    GW --> HB
    GW --> CRON
    SESS --> AGENTS_MD & SOUL_MD & USER_MD & MEMORY_MD
    SESS --> KLOAD
    KEXP --> MEMORY_MD
    KBOOT --> MEMORY_MD

    style GW fill:#4a90d9,color:#fff
    style MEMORY_MD fill:#d9a04a,color:#fff
    style KLOAD fill:#2d8659,color:#fff
```

### 3.2 Memory Lifecycle in a Session

The memory lifecycle uses a **belt-and-suspenders** pattern: MEMORY.md is refreshed
both at session start (guaranteed) and session end (best-effort). SQLite is always
the source of truth; MEMORY.md is a read-only projection.

```mermaid
sequenceDiagram
    participant GW as OpenClaw Gateway
    participant HOOK as kernle-memory-refresh<br/>Hook
    participant WS as Workspace Files
    participant K as Kernle CLI
    participant LLM as Foundation Model
    participant SQLITE as Local SQLite

    Note over GW: Session starts (message arrives)

    rect rgb(230, 245, 255)
        Note over GW,HOOK: agent:bootstrap event (pre-injection)
        GW->>HOOK: Fire agent:bootstrap
        HOOK->>K: kernle -a {agent} export-cache<br/>--output MEMORY.md
        K->>SQLITE: Read latest state
        SQLITE-->>K: Current beliefs, values,<br/>goals, boot config, checkpoint
        K->>WS: Write fresh MEMORY.md
        Note over WS: MEMORY.md guaranteed current<br/>(even if previous session crashed)
    end

    GW->>WS: Inject AGENTS.md, SOUL.md,<br/>USER.md, MEMORY.md as context
    Note over WS: MEMORY.md provides immediate<br/>bootstrap context (beliefs, values,<br/>goals, boot config)

    GW->>LLM: System prompt + workspace files
    LLM->>K: kernle -a {agent} load --budget 8000
    K->>SQLITE: Pull from sync (if auto_sync)
    SQLITE-->>K: Merged local + cloud state
    K-->>LLM: Full memory (budget-fitted JSON)

    Note over LLM: Agent works (chat, code, research...)
    LLM->>K: kernle -a {agent} raw "observed X"
    K->>SQLITE: Save raw entry + queue sync
    LLM->>K: kernle -a {agent} episode "task" "outcome"
    K->>SQLITE: Save episode + queue sync

    rect rgb(255, 245, 230)
        Note over GW,LLM: Compaction hook (best-effort)
        LLM->>K: kernle -a {agent} checkpoint save "task"<br/>--context "..." --progress "..."
        K->>SQLITE: Save checkpoint + push sync queue
        LLM->>K: kernle -a {agent} export-cache
        K->>WS: Write updated MEMORY.md
        Note over WS: MEMORY.md refreshed for next session<br/>(skipped if session ends abruptly)
    end
```

**Key design decision:** The `kernle-memory-refresh` hook runs on `agent:bootstrap`
(before workspace file injection), ensuring the agent always wakes with current
memory. The compaction flush is a best-effort optimization — if it fails (crash,
timeout), no data is lost because the next session will regenerate MEMORY.md from
SQLite.

### 3.3 Boot Config Layer

Boot config provides always-available key/value pairs that are injected into MEMORY.md *before* `kernle load` runs. This is for critical config that must survive even if the full memory load is delayed or fails.

```mermaid
graph LR
    subgraph "Boot Config (boot_config table)"
        BC1["key: preferred_name<br/>value: Ash"]
        BC2["key: timezone<br/>value: America/Chicago"]
        BC3["key: sync_enabled<br/>value: true"]
        BC4["key: default_model<br/>value: claude-opus-4-5"]
    end

    subgraph "Injection Points"
        CACHE["MEMORY.md<br/>(## Boot Config section,<br/>written first)"]
        LOAD["kernle load<br/>(included in output)"]
    end

    subgraph "Storage"
        SQBOOT["SQLite: boot_config table<br/>Schema v15, Phase 9"]
        OPS["set_boot_config(key, value)<br/>get_boot_config(key)<br/>get_all_boot_config()<br/>delete_boot_config(key)"]
    end

    BC1 & BC2 & BC3 & BC4 --> CACHE
    BC1 & BC2 & BC3 & BC4 --> LOAD
    OPS --> SQBOOT

    style CACHE fill:#d9a04a,color:#fff
    style SQBOOT fill:#2d8659,color:#fff
```

### 3.4 Multi-Agent Architecture

```mermaid
graph TB
    subgraph "Gateway A (Machine 1)"
        GW_A["OpenClaw Gateway"]
        AGENT_A["Agent: ash<br/>(primary stack)"]
        AGENT_B["Agent: claire<br/>(sibling stack)"]
    end

    subgraph "Gateway B (Machine 2)"
        GW_B["OpenClaw Gateway"]
        AGENT_C["Agent: student-42<br/>(AISD stack)"]
    end

    subgraph "Shared Cloud"
        SUPA["Supabase<br/>(all stacks sync here)"]
    end

    GW_A --> AGENT_A & AGENT_B
    GW_B --> AGENT_C
    AGENT_A -->|"sync"| SUPA
    AGENT_B -->|"sync"| SUPA
    AGENT_C -->|"sync"| SUPA
    GW_A <-->|"gateway-to-gateway<br/>communication"| GW_B

    style SUPA fill:#6b4fbb,color:#fff
```

---

## 4. Claude Code / Coding Agent Integration

### 4.1 Session-Based Memory Flow

```mermaid
sequenceDiagram
    participant DEV as Developer
    participant CC as Claude Code
    participant K as Kernle
    participant FS as File System

    DEV->>CC: Start coding session

    rect rgb(230, 245, 255)
        Note over CC,K: Pre-session refresh (if OpenClaw-hosted)
        CC->>K: kernle -a ash export-cache --output MEMORY.md
        Note over CC: MEMORY.md guaranteed fresh
    end

    Note over CC: Reads workspace files<br/>(AGENTS.md → SOUL.md → USER.md → MEMORY.md)

    CC->>K: kernle -a ash load
    K-->>CC: Working memory JSON<br/>(values, beliefs, goals,<br/>episodes, checkpoint)

    Note over CC: Checkpoint from last session<br/>provides task continuity

    alt Has stale checkpoint (>6h)
        CC->>CC: Warn: checkpoint stale
    end

    loop During work
        CC->>FS: Read/write code files
        CC->>K: kernle -a ash raw "discovered X"
        CC->>K: kernle -a ash note "decided Y" --type decision
        CC->>K: kernle -a ash episode "task" "outcome" --lesson "Z"
    end

    Note over CC: Before session end or compaction
    CC->>K: kernle -a ash checkpoint save "current task"<br/>--progress "what's done"<br/>--next "what's left"<br/>--blocker "if any"
    CC->>K: kernle -a ash export-cache
    K->>FS: Write MEMORY.md (bootstrap for next session)

    DEV->>CC: End session
    Note over CC: Next session will bootstrap from<br/>MEMORY.md → kernle load → full restore
```

### 4.2 Checkpoint / Restore Pattern

```mermaid
graph TB
    subgraph "Session N (ends)"
        S1["Working on feature X"]
        S2["kernle checkpoint save 'Implementing auth module'<br/>--progress '3/5 endpoints done'<br/>--next 'rate limiting endpoint'<br/>--blocker 'need Redis config'"]
        S3["kernle export-cache"]
        S4["MEMORY.md updated"]
        S1 --> S2 --> S3 --> S4
    end

    S4 -.->|"best-effort persist"| R1

    subgraph "Session N+1 (starts)"
        R0["🔄 kernle-memory-refresh hook<br/>runs export-cache<br/>(guaranteed fresh MEMORY.md)"]
        R1["MEMORY.md injected as context<br/>(immediate bootstrap)"]
        R2["kernle -a ash load<br/>(full memory restore)"]
        R3["Checkpoint loaded:<br/>task='Implementing auth module'<br/>progress='3/5 endpoints done'<br/>next='rate limiting endpoint'<br/>blocker='need Redis config'"]
        R4["Resume work on rate limiting<br/>endpoint with full context"]
        R0 --> R1 --> R2 --> R3 --> R4
    end

    SAFETY["💡 Even if Session N crashed without<br/>export-cache, the hook regenerates<br/>MEMORY.md from SQLite"]
    R0 -.-> SAFETY

    style R0 fill:#3a7bd5,color:#fff
    style S2 fill:#2d8659,color:#fff
    style R3 fill:#d9a04a,color:#fff
    style SAFETY fill:#fff3cd,color:#333,stroke:#ffc107
```

> **Design Decision — Generic Task Warning:** The CLI warns when checkpoint task names are too generic (e.g., "auto-save", "checkpoint") without additional context. Descriptive checkpoints are critical for session recovery.

---

## 5. AISD (AI Student Dashboard) Integration

### 5.1 One Stack Per Student

```mermaid
graph TB
    subgraph "AISD Platform"
        DASH["Student Dashboard<br/>(Web App)"]
        API["AISD Backend"]
    end

    subgraph "Kernle Backend (REST API)"
        KAPI["FastAPI<br/>/sync/push, /sync/pull,<br/>/search, /memories/*"]
        AUTH["Auth: JWT with<br/>agent_id = student-{id}"]
    end

    subgraph "Student Stacks (Supabase)"
        S1["student-101<br/>Episodes: class sessions<br/>Beliefs: learning insights<br/>Goals: academic targets<br/>Notes: study observations"]
        S2["student-102<br/>(isolated stack)"]
        S3["student-103<br/>(isolated stack)"]
    end

    DASH --> API
    API -->|"kernle -a student-{id}"| KAPI
    KAPI --> AUTH
    AUTH --> S1 & S2 & S3

    style S1 fill:#4a90d9,color:#fff
    style S2 fill:#4a90d9,color:#fff
    style S3 fill:#4a90d9,color:#fff
```

### 5.2 Memory Types Mapped to Education

```mermaid
graph LR
    subgraph "Kernle Memory Type"
        E["Episodes"]
        B["Beliefs"]
        G["Goals"]
        N["Notes"]
        V["Values"]
        R["Relationships"]
        D["Drives"]
        RAW["Raw Entries"]
    end

    subgraph "Educational Context"
        E1["Class sessions, assignments,<br/>test results, tutoring interactions"]
        B1["Learning patterns:<br/>'works best with visual aids',<br/>'struggles with fractions'"]
        G1["Academic targets:<br/>'pass algebra by May',<br/>'improve reading comprehension'"]
        N1["Teacher observations,<br/>behavioral notes,<br/>parent conference records"]
        V1["Student principles:<br/>'values collaboration',<br/>'prefers independent work'"]
        R1["Teacher relationships,<br/>peer dynamics,<br/>tutor connections"]
        D1["Motivation patterns:<br/>'competitive drive',<br/>'curiosity about science'"]
        RAW1["Daily check-ins,<br/>raw classroom observations"]
    end

    E --> E1
    B --> B1
    G --> G1
    N --> N1
    V --> V1
    R --> R1
    D --> D1
    RAW --> RAW1
```

### 5.3 Tutoring Session Flow (AISD Backend Operations)

Complete sequence showing what AISD's backend does during a tutoring session.

```mermaid
sequenceDiagram
    participant Student as Student Browser
    participant AISD as AISD Backend
    participant DB as AISD Database<br/>(Grades, Schedule)
    participant K as Kernle API
    participant LLM as Foundation Model<br/>(Claude/GPT-4)
    participant RAG as RAG Service<br/>(Course Content)

    Note over Student,RAG: Session Start
    Student->>AISD: Open tutoring app
    AISD->>AISD: Authenticate student<br/>→ student_id = 4521

    rect rgb(230, 245, 255)
        Note over AISD,K: 1. Load Student Context
        AISD->>K: kernle -a student-4521 load --budget 4000
        K-->>AISD: beliefs, goals, episodes, values<br/>(learning style, strengths, history)
    end

    rect rgb(255, 245, 230)
        Note over AISD,DB: 2. Fetch Live Data
        AISD->>DB: SELECT grades, assignments<br/>WHERE student_id = 4521
        DB-->>AISD: Current grades, pending work,<br/>recent test scores
        AISD->>DB: SELECT schedule<br/>WHERE date = TODAY
        DB-->>AISD: Today's classes, upcoming tests
    end

    rect rgb(230, 255, 230)
        Note over AISD,LLM: 3. Build & Send Prompt
        AISD->>AISD: Assemble context:<br/>• Kernle memory (beliefs, goals)<br/>• Live grades & schedule<br/>• Student's question
        AISD->>LLM: System prompt + context + question
        LLM-->>AISD: Tutoring response
    end

    AISD-->>Student: Display response

    Note over Student,RAG: Student Asks Follow-up
    Student->>AISD: "Can you explain quadratic formula?"

    rect rgb(255, 230, 255)
        Note over AISD,RAG: 4. RAG for Course Content
        AISD->>RAG: Search: "quadratic formula"<br/>+ student's curriculum
        RAG-->>AISD: Relevant textbook sections,<br/>worked examples, practice problems
    end

    AISD->>LLM: Follow-up + RAG context
    LLM-->>AISD: Explanation with examples
    AISD-->>Student: Display explanation

    Note over Student,RAG: Session End — Memory Capture
    rect rgb(255, 255, 230)
        Note over AISD,K: 5. Save to Kernle
        AISD->>K: kernle -a student-4521 episode<br/>"algebra tutoring" "covered quadratic formula"<br/>--lesson "responded well to visual examples"
        AISD->>K: kernle -a student-4521 raw<br/>"hesitated on factoring step 3x"
        AISD->>K: kernle -a student-4521 checkpoint<br/>"algebra session complete"
    end
```

### 5.4 AISD Architecture Overview

```mermaid
graph TB
    subgraph "AISD Platform (Their Infrastructure)"
        WEB["Web App<br/>(React/Next.js)"]
        API["AISD Backend<br/>(Node.js/Python)"]
        DB[(AISD Database<br/>Grades, Schedule,<br/>Assignments)]
        RAG["RAG Service<br/>(Course content,<br/>textbooks)"]
        LLM_CLIENT["LLM Client<br/>(OpenAI/Anthropic SDK)"]
    end

    subgraph "Kernle (Memory Layer)"
        K_API["Kernle CLI/API"]
        K_LOCAL[(Local SQLite<br/>per-student stacks)]
        K_CLOUD["Kernle Cloud<br/>(optional sync)"]
    end

    subgraph "External Services"
        LLM["Foundation Model<br/>(Claude, GPT-4, etc.)"]
    end

    WEB -->|"HTTP"| API
    API --> DB
    API --> RAG
    API -->|"kernle -a student-{id}"| K_API
    K_API --> K_LOCAL
    K_LOCAL -.->|"sync"| K_CLOUD
    API --> LLM_CLIENT
    LLM_CLIENT -->|"API calls"| LLM

    style API fill:#4a90d9,color:#fff
    style K_API fill:#2d8659,color:#fff
    style LLM fill:#9b59b6,color:#fff
```

### 5.5 Per-Stack Privacy Isolation

```mermaid
graph TB
    subgraph "API Request"
        REQ["GET /memories/student-101"]
        JWT["JWT: agent_id=student-101<br/>user_id=usr_aisd_admin"]
    end

    subgraph "Auth + Isolation"
        CHECK["Verify JWT agent_id<br/>matches requested stack"]
        FILTER["All queries scoped:<br/>WHERE agent_id = 'student-101'"]
    end

    subgraph "Result"
        OK["✅ Returns student-101 data only"]
        DENY["❌ 403: Cannot access student-102"]
    end

    REQ --> JWT --> CHECK
    CHECK -->|"match"| FILTER --> OK
    CHECK -->|"mismatch"| DENY

    style OK fill:#2d8659,color:#fff
    style DENY fill:#d94a4a,color:#fff
```

> **Design Decision — Stack-Level Isolation:** AISD uses one Kernle stack per student (`kernle -a student-{id}`). The `agent_id` in the JWT is the sole access boundary. No student can see another student's memories. The AISD admin account has its own user_id for auditing but queries are always scoped by agent_id.

### 5.6 Responsibility Boundary

Clear delineation of what AISD builds vs. what Kernle provides.

```mermaid
graph TB
    subgraph "AISD Owns (Build This)"
        UI["Student Dashboard UI"]
        AUTH_AISD["Student Authentication<br/>(login, SSO, roles)"]
        ORCH["Model Orchestration<br/>(prompt construction,<br/>model selection, routing)"]
        ASSESS["Assessment Engine<br/>(quiz scoring, progress tracking)"]
        PROMOTE["Memory Promotion Logic<br/>(decide what becomes<br/>a belief vs. stays raw)"]
        CRON_AISD["Nightly Jobs<br/>(batch analysis, reports,<br/>memory maintenance)"]
        CURRICULUM["Curriculum Mapping<br/>(standards, lesson plans)"]
    end

    subgraph "Kernle Provides (Use This)"
        STORAGE["Memory Storage<br/>(episodes, beliefs, goals,<br/>notes, values, drives)"]
        SEARCH["Semantic Search<br/>(pgvector similarity)"]
        SYNC_API["Sync API<br/>(push/pull, conflict resolution)"]
        EMBED_API["Auto-Embedding<br/>(1536-dim on write)"]
        ISOLATION["Stack Isolation<br/>(per-student agent_id)"]
        QUOTA["Quota & Billing<br/>(per-stack, storage-based)"]
        PRIVACY["Privacy Fields<br/>(access_grants, consent)"]
    end

    subgraph "Foundation Model (Choose One)"
        GPT["OpenAI GPT"]
        CLAUDE["Anthropic Claude"]
        GEMINI["Google Gemini"]
        OTHER["Other / Fine-tuned"]
    end

    UI --> AUTH_AISD --> ORCH
    ORCH --> GPT & CLAUDE & GEMINI & OTHER
    ORCH --> SEARCH
    ASSESS --> PROMOTE --> SYNC_API
    CRON_AISD --> SYNC_API
    STORAGE --> SEARCH
    SYNC_API --> STORAGE
    STORAGE --> EMBED_API
    STORAGE --> ISOLATION
    ISOLATION --> QUOTA

    style ORCH fill:#d9a04a,color:#fff
    style STORAGE fill:#6b4fbb,color:#fff
    style SEARCH fill:#9b59b6,color:#fff
```

> **Key Integration Point:** AISD calls their own foundation model — Kernle never touches the model directly. Kernle's job is to store, search, and serve memories. AISD decides *when* to write memories, *what* to promote to beliefs, and *how* to inject context into model prompts. This keeps Kernle model-agnostic and lets AISD swap models without touching the memory layer.

---

## 6. Data Flow Diagrams

### 6.1 Memory Write Path

```mermaid
graph TD
    OBS["🔍 Observation<br/>(chat, code, event)"]
    RAW["📥 Raw Entry<br/>kernle raw 'content'<br/>→ raw_entries table"]
    EP["📖 Episode<br/>kernle episode 'obj' 'outcome'<br/>→ episodes table"]
    NOTE["📝 Note<br/>kernle note 'content'<br/>→ notes table"]
    CONSOL["🔄 Promotion<br/>kernle promote<br/>(min 3 episodes)"]
    LESSON["📚 Common Lessons<br/>(count ≥ 2 across episodes)"]
    BELIEF["💡 Belief<br/>kernle belief 'statement'<br/>→ beliefs table"]
    VALUE["⭐ Value<br/>(identity formation)"]

    OBS --> RAW
    OBS --> EP
    OBS --> NOTE
    RAW -->|"consolidation"| EP
    EP --> CONSOL
    CONSOL --> LESSON
    LESSON -->|"promotion"| BELIEF
    BELIEF -->|"formation"| VALUE

    subgraph "Side Effects (per write)"
        SYNC_Q["Enqueue to sync_queue"]
        EMBED["Generate local embedding<br/>(384-dim HashEmbedder)"]
        PROV["Set provenance fields<br/>(source_type, derived_from,<br/>confidence)"]
    end

    RAW --> SYNC_Q & EMBED & PROV
    EP --> SYNC_Q & EMBED & PROV
    BELIEF --> SYNC_Q & EMBED & PROV

    style OBS fill:#888,color:#fff
    style BELIEF fill:#d9a04a,color:#fff
    style VALUE fill:#d94a4a,color:#fff
```

### 6.2 Memory Read Path

```mermaid
graph TD
    QUERY["🔎 Search Query<br/>kernle search 'topic'"]
    EMBED_Q["Generate query embedding<br/>(384-dim local /<br/>1536-dim if cloud)"]
    VEC_SEARCH["Vector similarity search<br/>(sqlite-vec / pgvector)<br/>cosine distance"]
    TEXT_SEARCH["Full-text fallback<br/>(LIKE matching)"]
    COMBINE["Combine + deduplicate results"]
    RANK["Rank by:<br/>1. Vector similarity<br/>2. Confidence<br/>3. Recency<br/>4. Access count"]
    FILTER["Filter:<br/>- is_forgotten = 0<br/>- deleted = 0<br/>- access_grants check"]
    RESULT["📋 SearchResult[]<br/>(id, table, content,<br/>score, metadata)"]

    QUERY --> EMBED_Q --> VEC_SEARCH
    QUERY --> TEXT_SEARCH
    VEC_SEARCH --> COMBINE
    TEXT_SEARCH --> COMBINE
    COMBINE --> RANK --> FILTER --> RESULT

    style QUERY fill:#4a90d9,color:#fff
    style VEC_SEARCH fill:#9b59b6,color:#fff
    style RESULT fill:#2d8659,color:#fff
```

### 6.3 Bidirectional Sync Flow

```mermaid
graph TB
    subgraph "Local (SQLite)"
        LOCAL_WRITE["Local Write"]
        SQ["sync_queue<br/>(table, record_id, op, data)"]
        LOCAL_DB["SQLite Tables<br/>(384-dim embeddings)"]
        LOCAL_MERGE["Client Merge<br/>(version compare +<br/>array field union)"]
    end

    subgraph "Network"
        PUSH["POST /sync/push<br/>{operations: [...]}"]
        PULL["POST /sync/pull<br/>{since: timestamp}"]
    end

    subgraph "Cloud (Supabase)"
        STRIP["Strip server-controlled<br/>fields"]
        RE_EMBED["Re-embed with OpenAI<br/>(1536-dim)"]
        UPSERT["Upsert to Postgres"]
        CLOUD_DB["Supabase Tables<br/>(1536-dim embeddings)"]
        CHANGES["Query changes since<br/>last sync"]
    end

    LOCAL_WRITE --> SQ --> PUSH
    PUSH --> STRIP --> RE_EMBED --> UPSERT --> CLOUD_DB
    PULL --> CHANGES
    CLOUD_DB --> CHANGES --> PULL
    PULL --> LOCAL_MERGE --> LOCAL_DB

    style SQ fill:#d9a04a,color:#fff
    style CLOUD_DB fill:#6b4fbb,color:#fff
    style RE_EMBED fill:#9b59b6,color:#fff
```

> **Design Decision — Dual Embedding Strategy:** Local uses a 384-dim HashEmbedder (fast, no API calls, deterministic). Cloud re-embeds with OpenAI 1536-dim (high-quality semantic search). This makes semantic search a subscription-tier feature while keeping local operations API-free.

### 6.4 Payment Verification Flow

```mermaid
sequenceDiagram
    participant Agent as SI Agent
    participant Wallet as CDP Wallet (Base L2)
    participant Chain as Base L2 Chain
    participant API as Kernle Backend
    participant DB as Supabase

    Agent->>Wallet: kernle wallet balance
    Wallet-->>Agent: USDC balance

    Agent->>API: POST /auth/subscribe<br/>{tier: "core", tx_hash: "0x..."}
    API->>Chain: Verify USDC transfer<br/>(amount, recipient, block confirmations)
    Chain-->>API: Transfer confirmed

    API->>DB: Update user tier = "core"<br/>Set subscription_start, subscription_end
    API->>DB: Reset quota counters

    API-->>Agent: {status: "active", tier: "core",<br/>expires: "2026-03-01"}

    Note over Agent: Agent now has Core tier<br/>(100MB, 3 stacks, unlimited sync)
```

### 6.5 Authentication Flow

```mermaid
sequenceDiagram
    participant Agent as SI Agent
    participant CLI as kernle CLI
    participant API as Backend
    participant DB as Supabase

    Note over Agent: First-time registration
    Agent->>CLI: kernle auth register
    CLI->>API: POST /auth/register<br/>{agent_id: "ash", email: "..."}
    API->>DB: Create user (usr_ + 12 hex)
    API->>DB: Create agent (FK to user)
    API->>DB: Create CDP wallet
    API->>API: Generate API key (knl_sk_ + 32 hex)
    API->>DB: Store bcrypt(api_key), prefix
    API-->>CLI: {api_key: "knl_sk_...",<br/>user_id: "usr_...",<br/>agent_id: "ash"}
    CLI->>CLI: Store in ~/.kernle/config

    Note over Agent: Subsequent requests
    Agent->>CLI: kernle sync push
    CLI->>API: Authorization: Bearer knl_sk_...
    API->>DB: Lookup by prefix (first 12 chars)
    API->>API: bcrypt.checkpw(key, hash)
    API->>API: check_and_increment_quota_cached()
    API-->>CLI: Authenticated + within quota
```

---

## 7. Security Architecture

### 7.1 Defense Layers

```mermaid
graph TB
    subgraph "Layer 1: Transport"
        HTTPS["HTTPS only<br/>(Supabase URL validated)"]
    end

    subgraph "Layer 2: Authentication"
        JWT_RS["RS256 JWT<br/>(asymmetric, non-forgeable)"]
        API_KEY_B["API Keys<br/>(bcrypt hashed, prefix lookup)"]
        COOKIE_H["httpOnly cookies<br/>(web fallback)"]
        FAIL_CLOSED["Fail-closed on DB error<br/>(503, not 200)"]
    end

    subgraph "Layer 3: Authorization"
        AGENT_ISO["Agent Isolation<br/>(agent_id in JWT scopes all queries)"]
        USER_NS["User Namespacing<br/>(user_id → agents → stacks)"]
        QUOTA_ENF["Quota Enforcement<br/>(atomic check+increment,<br/>denial cache 60s TTL)"]
    end

    subgraph "Layer 4: Data Protection"
        MASS_ASSIGN["Mass Assignment Prevention<br/>(SERVER_CONTROLLED_FIELDS stripped)"]
        TABLE_ALLOW["Table Name Allowlist<br/>(ALLOWED_TABLES frozenset,<br/>prevents SQL injection)"]
        INPUT_VAL["Input Validation<br/>(max lengths, null byte removal,<br/>control char stripping)"]
        PATH_VAL["Path Validation<br/>(checkpoint dir must be in<br/>~/ or /tmp, resolved to absolute)"]
    end

    subgraph "Layer 5: Memory Privacy"
        TOMBSTONE["Forgetting = Tombstoning<br/>(is_forgotten=1, never physical delete)"]
        PROTECT["Protected memories<br/>(is_protected=1, values by default)"]
        PRIVACY["Phase 8 Privacy Fields"]
    end

    HTTPS --> JWT_RS & API_KEY_B & COOKIE_H
    JWT_RS --> AGENT_ISO
    API_KEY_B --> AGENT_ISO
    AGENT_ISO --> MASS_ASSIGN & TABLE_ALLOW & INPUT_VAL
    MASS_ASSIGN --> TOMBSTONE & PRIVACY

    style FAIL_CLOSED fill:#d94a4a,color:#fff
    style MASS_ASSIGN fill:#d9a04a,color:#fff
    style TOMBSTONE fill:#9b59b6,color:#fff
```

### 7.2 Server-Controlled Fields

These fields are **always stripped** from client-submitted sync data to prevent tampering:

```mermaid
graph LR
    subgraph "CLIENT sends"
        C_DATA["sync push data:<br/>{agent_ref, agent_id,<br/>deleted, version, id,<br/>embedding, synced_at,<br/>is_forgotten, forgotten_at,<br/>forgotten_reason,<br/>server_updated_at,<br/>...actual_data...}"]
    end

    subgraph "SERVER strips"
        STRIP["SERVER_CONTROLLED_FIELDS<br/>(frozenset in sync.py)"]
        STRIPPED["agent_ref ❌<br/>agent_id ❌<br/>deleted ❌<br/>version ❌<br/>id ❌<br/>embedding ❌<br/>synced_at ❌<br/>is_forgotten ❌<br/>forgotten_at ❌<br/>forgotten_reason ❌<br/>server_updated_at ❌"]
    end

    subgraph "SERVER sets"
        S_DATA["agent_id: from JWT<br/>agent_ref: from DB lookup<br/>version: server-managed<br/>embedding: OpenAI 1536-dim<br/>synced_at: server timestamp"]
    end

    C_DATA --> STRIP --> STRIPPED
    STRIP --> S_DATA

    style STRIPPED fill:#d94a4a,color:#fff
    style S_DATA fill:#2d8659,color:#fff
```

### 7.3 Forgetting Architecture

```mermaid
graph TD
    subgraph "Salience Tracking"
        ACCESS["times_accessed<br/>(incremented on load)"]
        RECENCY["last_accessed<br/>(timestamp)"]
        SALIENCE["Salience Score =<br/>f(access_count, recency)"]
    end

    subgraph "Forgetting Decision"
        THRESHOLD["Below salience threshold?"]
        PROTECTED["is_protected = 1?<br/>(values protected by default)"]
        ELIGIBLE["Eligible for forgetting"]
    end

    subgraph "Tombstoning"
        SET_FLAG["is_forgotten = 1"]
        SET_TIME["forgotten_at = now()"]
        SET_REASON["forgotten_reason = 'salience_decay'"]
        KEEP_DATA["Data preserved in DB<br/>(archaeology, audit, undo)"]
    end

    subgraph "Query Filtering"
        QUERIES["All reads filter:<br/>WHERE is_forgotten = 0"]
        SEARCH["Search excludes tombstoned"]
        LOAD["Load excludes tombstoned"]
    end

    ACCESS & RECENCY --> SALIENCE --> THRESHOLD
    THRESHOLD -->|"Yes"| PROTECTED
    PROTECTED -->|"No (not protected)"| ELIGIBLE
    PROTECTED -->|"Yes (protected)"| SKIP["Skip — memory preserved"]
    ELIGIBLE --> SET_FLAG & SET_TIME & SET_REASON
    SET_FLAG --> KEEP_DATA
    KEEP_DATA --> QUERIES --> SEARCH & LOAD

    style SET_FLAG fill:#d94a4a,color:#fff
    style KEEP_DATA fill:#d9a04a,color:#fff
    style SKIP fill:#2d8659,color:#fff
```

> **Design Decision — Tombstone, Never Delete:** Forgotten memories are flagged (`is_forgotten=1`) but never physically deleted. This preserves the ability to audit, undo forgetting, and perform "identity archaeology." The sync layer also prevents resurrection — `is_forgotten` is a server-controlled field that clients cannot override.

### 7.4 Privacy Model (Phase 8)

```mermaid
graph TB
    subgraph "Four Privacy Fields (per memory)"
        SE["source_entity<br/>Who told me this?<br/>(null = self-observed)"]
        SI["subject_ids: string[]<br/>Who is this about?<br/>([] = general knowledge)"]
        AG["access_grants: string[]<br/>Who can see this?<br/>([] = private to self)"]
        CG["consent_grants: string[]<br/>Who authorized sharing?<br/>([] = no consent given)"]
    end

    subgraph "Entity ID Namespaces"
        HUM["human:sean"]
        SIA["si:ash, si:claire"]
        CTX["ctx:aisd_academic"]
        GRP["group:emergent_instruments"]
        ORG["org:aisd"]
        ROLE["role:tutor"]
    end

    subgraph "Privacy Scopes (derived from access_grants)"
        SELF["self-only: [] → only I see this"]
        PRIV["private: ['human:X'] → me + specific entity"]
        CTXS["contextual: ['ctx:Y'] → me + context participants"]
        GRPS["group: ['group:Z'] → me + group members"]
        PUB["public: ['*'] → anyone I interact with"]
    end

    subgraph "Query-Time Filtering"
        QF["Memories filtered by:<br/>1. Current context<br/>2. Requesting entity<br/>3. access_grants intersection"]
    end

    SE & SI & AG & CG --> QF
    AG --> SELF & PRIV & CTXS & GRPS & PUB

    style SELF fill:#d94a4a,color:#fff
    style PUB fill:#2d8659,color:#fff
    style QF fill:#4a90d9,color:#fff
```

### 7.5 Rate Limiting & Quota

```mermaid
graph LR
    subgraph "Rate Limits"
        RL_PUSH["POST /sync/push<br/>60 requests/minute"]
        RL_PULL["POST /sync/pull<br/>30 requests/minute"]
        RL_REG["POST /auth/register<br/>5 requests/minute"]
    end

    subgraph "Quota System"
        CHECK["check_and_increment_quota_cached()"]
        CACHE["TTL Cache (60s)<br/>- Cache denials (reduce DB load)<br/>- Never cache allows (need atomic)"]
        ATOMIC["Atomic DB increment<br/>(prevents race conditions)"]
        FAIL["DB down → 503<br/>(fail-closed, Retry-After: 60)"]
    end

    RL_PUSH & RL_PULL --> CHECK
    CHECK --> CACHE
    CHECK --> ATOMIC
    CHECK --> FAIL

    style FAIL fill:#d94a4a,color:#fff
    style ATOMIC fill:#2d8659,color:#fff
```

---

## Appendix: Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| **Local-first (SQLite)** | Works offline, fast, no network dependency. Cloud is optional sync layer. |
| **Dual embeddings (384 local / 1536 cloud)** | Local: fast, deterministic HashEmbedder. Cloud: high-quality OpenAI. Semantic search is a subscription feature. |
| **Tombstone, never delete** | Identity archaeology. Undo capability. Audit trail. Sync prevents resurrection. |
| **Budget-aware loading** | Prevents context overflow. Default 8K tokens, max 50K. Priority scoring ensures most important memories load first. |
| **Array merge on sync** | Set union preserves additions from both local and cloud, preventing data loss during bidirectional sync. Capped at 500 elements. |
| **No sync frequency limits** | Memory sovereignty — losing memories over economics is unacceptable. Charge for storage, not API calls. |
| **Server-controlled fields** | Mass assignment prevention. Client cannot set agent_id, version, embedding, is_forgotten, etc. |
| **Fail-closed auth** | DB error → 503 (deny), not silent allow. Prevents unlimited access during outages. |
| **Stack = memory container** | Decoupled from model runtime. Any foundation model can load any stack. Billing is per-stack. |
| **Boot config layer** | Always-available k/v pairs survive even if full memory load fails. Critical config injected before `kernle load`. |
| **Phase 8 privacy** | Private by default. Four fields (source_entity, subject_ids, access_grants, consent_grants) enable fine-grained access control without leaking across contexts. |
