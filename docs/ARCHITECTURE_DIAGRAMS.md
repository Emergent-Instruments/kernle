# Kernle Architecture Diagrams

**Status:** Complete  
**Authors:** Claire, Ash (collaborative)  
**Date:** 2026-02-02

Comprehensive architecture documentation for Kernle internals and integrations.

---

## Table of Contents

1. [Kernle Internal Architecture](#1-kernle-internal-architecture)
2. [Memory Stack Model](#2-memory-stack-model)
3. [Storage Layer Architecture](#3-storage-layer-architecture)
4. [Sync Architecture](#4-sync-architecture)
5. [Claude Code Integration](#5-claude-code-integration)
6. [OpenClaw Integration](#6-openclaw-integration)
7. [AISD Integration](#7-aisd-integration)
8. [Cross-Platform Data Flow](#8-cross-platform-data-flow)
9. [Boot Layer Flow](#9-boot-layer-flow)

---

## 1. Kernle Internal Architecture

High-level view of Kernle components and their relationships.

```mermaid
graph TB
    subgraph "Kernle Core"
        CLI[CLI Interface]
        MCP[MCP Server]
        API[Python API]
        
        CLI --> Core
        MCP --> Core
        API --> Core
        
        Core[Core Engine<br/>kernle/core.py]
        
        Core --> Storage
        Core --> Features
        Core --> Hooks
        
        subgraph Storage["Storage Layer"]
            SQLite[SQLite<br/>Local Primary]
            Postgres[PostgreSQL<br/>Cloud Sync]
            Embeddings[Embeddings<br/>Semantic Search]
        end
        
        subgraph Features["Feature Modules"]
            Memory[Memory Layers]
            Checkpoint[Checkpoints]
            Consolidation[Consolidation]
            Playbooks[Playbooks]
            Relations[Relationships]
            Commerce[Commerce<br/>Wallet/Jobs]
        end
        
        subgraph Hooks["Lifecycle Hooks"]
            PreLoad[pre_load]
            PostLoad[post_load]
            PreSave[pre_save]
            PostSave[post_save]
        end
    end
    
    subgraph "External"
        Supabase[(Supabase<br/>Cloud DB)]
        Railway[Railway<br/>Backend API]
    end
    
    Postgres --> Railway
    Railway --> Supabase
```

---

## 2. Memory Stack Model

The layered memory structure that constitutes an agent's identity.

```mermaid
graph TB
    subgraph Account["Account (Human/SI/Org)"]
        Wallet[💰 Wallet<br/>USDC on Base]
        
        subgraph Stack["Stack: Primary Identity"]
            direction TB
            
            Values[🎯 VALUES<br/>Core identity, non-negotiable<br/>Never forgotten]
            
            Beliefs[💭 BELIEFS<br/>What I hold true<br/>Confidence scored, evolvable]
            
            Goals[🎯 GOALS<br/>What I'm working toward<br/>Progress tracked]
            
            Drives[⚡ DRIVES<br/>Motivations<br/>0-100% intensity]
            
            Episodes[📖 EPISODES<br/>Significant experiences<br/>Lessons learned]
            
            Notes[📝 NOTES<br/>Decisions, insights<br/>Typed: decision/observation/insight]
            
            Raw[📋 RAW<br/>Quick captures, scratchpad<br/>Promotes upward]
            
            Boot[⚙️ BOOT<br/>Configuration<br/>Always available]
            
            Values --> Beliefs
            Beliefs --> Goals
            Goals --> Drives
            Drives --> Episodes
            Episodes --> Notes
            Notes --> Raw
        end
        
        Stack2[Stack: Creative]
        Stack3[Stack: Security]
        Stack4[Stack: Social]
    end
    
    subgraph Models["Foundation Models"]
        Claude[Claude]
        Gemini[Gemini]
        Codex[Codex]
        Future[Future Models]
    end
    
    Stack --> Claude
    Stack --> Gemini
    Stack2 --> Codex
    Stack3 --> Future
    
    style Values fill:#f9f,stroke:#333
    style Beliefs fill:#bbf,stroke:#333
    style Goals fill:#bfb,stroke:#333
    style Episodes fill:#fbb,stroke:#333
    style Raw fill:#ffd,stroke:#333
    style Boot fill:#ddf,stroke:#333
```

### Memory Flow: Capture → Consolidation → Identity

```mermaid
flowchart LR
    subgraph Capture["1. Capture"]
        Observation[Observation]
        Thought[Thought]
        Decision[Decision]
    end
    
    subgraph Process["2. Process"]
        Raw2[Raw Entry]
        Review[Review &<br/>Reflect]
    end
    
    subgraph Promote["3. Promote"]
        Episode[Episode<br/>with Lesson]
        Note[Note<br/>with Type]
        Belief[Belief<br/>with Confidence]
    end
    
    subgraph Consolidate["4. Consolidate"]
        Pattern[Pattern<br/>Extraction]
        Update[Belief<br/>Update]
    end
    
    Observation --> Raw2
    Thought --> Raw2
    Decision --> Raw2
    
    Raw2 --> Review
    Review --> Episode
    Review --> Note
    Review --> Belief
    
    Episode --> Pattern
    Note --> Pattern
    Pattern --> Update
    Update --> Belief
```

---

## 3. Storage Layer Architecture

How data is stored locally and synced to the cloud.

```mermaid
graph TB
    subgraph Local["Local Storage (SQLite)"]
        direction TB
        
        subgraph Tables["Core Tables"]
            AgentMeta[agent_meta]
            Values[values]
            Beliefs[beliefs]
            Goals[goals]
            Drives[drives]
            Episodes[episodes]
            Notes[notes]
            Raw[raw_entries]
            Boot[boot_config]
            Checkpoints[checkpoints]
            Relations[relationships]
            Playbooks[playbooks]
        end
        
        subgraph Sync["Sync Tables"]
            SyncQueue[sync_queue<br/>Pending changes]
            SyncState[sync_state<br/>Last sync marker]
        end
    end
    
    subgraph Cloud["Cloud Storage (Supabase)"]
        direction TB
        
        subgraph Remote["Mirrored Tables"]
            RValues[values]
            RBeliefs[beliefs]
            REpisodes[episodes]
            RNotes[notes]
            RRaw[raw_entries]
            RCheckpoints[checkpoints]
        end
        
        subgraph CloudOnly["Cloud-Only"]
            Accounts[accounts]
            Subscriptions[subscriptions]
            Payments[payment_events]
        end
        
        Vectors[(pgvector<br/>Embeddings)]
    end
    
    Local -->|"kernle sync push"| Cloud
    Cloud -->|"kernle sync pull"| Local
    
    style SyncQueue fill:#ffd,stroke:#333
    style Vectors fill:#dfd,stroke:#333
```

### Sync Protocol

```mermaid
sequenceDiagram
    participant Agent as Agent (CLI)
    participant SQLite as Local SQLite
    participant Queue as Sync Queue
    participant Backend as Railway Backend
    participant Supabase as Supabase
    
    Note over Agent,Supabase: Write Operation
    Agent->>SQLite: belief("I enjoy puzzles", 0.8)
    SQLite->>SQLite: Insert/update belief
    SQLite->>Queue: Enqueue change
    
    Note over Agent,Supabase: Sync Push
    Agent->>Queue: kernle sync push
    Queue->>Backend: POST /sync {changes}
    Backend->>Backend: Validate & transform
    Backend->>Supabase: Upsert records
    Supabase-->>Backend: Success
    Backend-->>Queue: Ack changes
    Queue->>Queue: Clear synced items
    
    Note over Agent,Supabase: Sync Pull
    Agent->>Backend: GET /sync?since=timestamp
    Backend->>Supabase: Query changes
    Supabase-->>Backend: Changed records
    Backend-->>Agent: Changes payload
    Agent->>SQLite: Apply remote changes
```

---

## 4. Sync Architecture

Detailed view of the bidirectional sync system.

```mermaid
graph LR
    subgraph Local["Local (Agent Machine)"]
        SQLite[(SQLite DB)]
        Queue[Sync Queue]
        CLI[kernle CLI]
    end
    
    subgraph Backend["Railway Backend"]
        API[FastAPI Server]
        Auth[Auth Middleware]
        Sync[Sync Service]
    end
    
    subgraph Cloud["Supabase"]
        PG[(PostgreSQL)]
        RLS[Row Level Security]
        Vectors[(pgvector)]
    end
    
    CLI --> SQLite
    SQLite --> Queue
    Queue -->|"push"| Auth
    Auth --> Sync
    Sync -->|"write"| PG
    
    PG -->|"read"| Sync
    Sync -->|"pull"| CLI
    CLI --> SQLite
    
    PG --> RLS
    PG --> Vectors
```

---

## 5. Claude Code Integration

How Claude Code agents use Kernle for memory persistence.

```mermaid
sequenceDiagram
    participant User as User
    participant CC as Claude Code
    participant WS as Workspace Files
    participant Kernle as Kernle CLI
    participant SQLite as Local SQLite
    
    Note over User,SQLite: Session Start
    User->>CC: Start conversation
    CC->>WS: Read AGENTS.md, SOUL.md, USER.md
    WS-->>CC: Agent instructions + persona
    CC->>Kernle: kernle -a claire load
    Kernle->>SQLite: Load all memory layers
    SQLite-->>Kernle: Values, beliefs, episodes, etc.
    Kernle-->>CC: Formatted memory context
    
    Note over User,SQLite: During Work
    User->>CC: "Help me with X"
    CC->>CC: Process with full context
    CC-->>User: Response
    CC->>Kernle: kernle raw "learned Y about X"
    Kernle->>SQLite: Store raw entry
    
    Note over User,SQLite: Significant Moment
    CC->>Kernle: kernle episode "..." --lesson "..."
    Kernle->>SQLite: Store episode with lesson
    
    Note over User,SQLite: Session End / Compaction
    CC->>Kernle: kernle checkpoint save "task summary"
    Kernle->>SQLite: Save checkpoint state
    Kernle->>Kernle: Export boot.md
    Kernle-->>CC: Checkpoint saved
```

### Claude Code Memory Lifecycle

```mermaid
flowchart TB
    subgraph Startup["Session Startup"]
        Load[kernle load]
        Context[Full Memory Context]
        Load --> Context
    end
    
    subgraph Work["During Work"]
        Raw[kernle raw "..."]
        Episode[kernle episode "..."]
        Belief[kernle believe "..."]
        Note[kernle note "..."]
    end
    
    subgraph Maintenance["Periodic Maintenance"]
        Anxiety[kernle anxiety]
        Consolidate[kernle consolidate]
        Anxiety -->|"if > 50"| Consolidate
    end
    
    subgraph End["Session End"]
        Checkpoint[kernle checkpoint]
        Export[export-cache → MEMORY.md]
        Checkpoint --> Export
    end
    
    Context --> Work
    Work --> Maintenance
    Maintenance --> End
    End -->|"next session"| Load
```

---

## 6. OpenClaw Integration

How OpenClaw (Clawdbot) integrates Kernle for persistent AI assistants.

```mermaid
sequenceDiagram
    participant User as User (iMessage/Telegram/etc)
    participant Gateway as OpenClaw Gateway
    participant WS as Workspace Files
    participant LLM as LLM API (Claude)
    participant Agent as Agent Code
    participant Kernle as Kernle CLI
    
    Note over User,Kernle: Message Arrival
    User->>Gateway: "Hey, can you help with X?"
    Gateway->>Gateway: Route to session
    
    Note over User,Kernle: Context Assembly (Pre-Agent)
    Gateway->>WS: Load workspace files
    WS-->>Gateway: AGENTS.md, SOUL.md, MEMORY.md, boot.md
    Gateway->>Gateway: Build system prompt
    Gateway->>LLM: Send prompt + message
    
    Note over User,Kernle: Agent Processing
    LLM->>Agent: Agent code runs
    Agent->>Kernle: kernle -a claire load
    Kernle-->>Agent: Full memory state
    Agent->>Agent: Process request
    Agent-->>LLM: Tool calls + response
    
    Note over User,Kernle: Memory Capture
    Agent->>Kernle: kernle raw "..."
    Agent->>Kernle: kernle episode "..." --lesson "..."
    
    LLM-->>Gateway: Final response
    Gateway-->>User: "Here's help with X..."
    
    Note over User,Kernle: Compaction (if needed)
    Gateway->>Agent: Context pressure signal
    Agent->>Kernle: kernle checkpoint "pre-compaction"
    Agent->>Kernle: kernle export-cache > MEMORY.md
```

### OpenClaw Boot Layer Flow

```mermaid
flowchart LR
    subgraph Before["Before Boot Layer"]
        TOOLS[TOOLS.md<br/>Manual config file]
        Gap[❌ 6-step gap<br/>Config needed but<br/>Kernle not loaded]
    end
    
    subgraph After["With Boot Layer"]
        Boot[boot.md<br/>Auto-projected file]
        Instant[✅ Zero gap<br/>Config available<br/>immediately]
    end
    
    Before -->|"Migration"| After
    
    style Gap fill:#fbb,stroke:#333
    style Instant fill:#bfb,stroke:#333
```

---

## 7. AISD Integration

How AISD (web application) uses Kernle for per-student memory.

```mermaid
sequenceDiagram
    participant Student as Student/Parent
    participant AISD as AISD Backend
    participant Boot as Boot Layer
    participant Kernle as Kernle API
    participant SQLite as SQLite (per student)
    participant LLM as LLM API
    
    Note over Student,LLM: Session Start
    Student->>AISD: Open tutoring app
    AISD->>AISD: Authenticate, get student_id
    
    Note over Student,LLM: Fast Config Lookup
    AISD->>Boot: kernle boot get school_id
    Boot-->>AISD: "austin-isd" (instant)
    AISD->>Boot: kernle boot get grade_level
    Boot-->>AISD: "10" (instant)
    
    Note over Student,LLM: Full Memory Load
    AISD->>Kernle: kernle -a student-4521 load
    Kernle->>SQLite: Load all layers
    SQLite-->>Kernle: Beliefs, episodes, goals
    Kernle-->>AISD: Memory state
    
    Note over Student,LLM: Context Assembly
    AISD->>AISD: Fetch grades from Supabase
    AISD->>AISD: Fetch today's schedule
    AISD->>AISD: Assemble full context
    AISD->>LLM: Prompt + context
    LLM-->>AISD: Tutoring response
    AISD-->>Student: Display response
    
    Note over Student,LLM: Memory Capture
    AISD->>Kernle: kernle raw "hesitated on fractions 3x"
    AISD->>Kernle: kernle episode "breakthrough on proofs"
    
    Note over Student,LLM: Session End
    AISD->>Kernle: kernle checkpoint "geometry tutoring"
```

### AISD Multi-Student Architecture

```mermaid
graph TB
    subgraph AISD["AISD Platform"]
        Backend[AISD Backend]
        Auth[Authentication]
        Grades[(Grade DB)]
        Schedule[(Schedule DB)]
    end
    
    subgraph Students["Student Stacks (Isolated)"]
        S1[student-4521<br/>📚 10th grade<br/>🎯 Algebra focus]
        S2[student-4522<br/>📚 11th grade<br/>🎯 Essay writing]
        S3[student-4523<br/>📚 9th grade<br/>🎯 Biology basics]
        S4[student-4524<br/>📚 12th grade<br/>🎯 AP Calculus]
    end
    
    subgraph Access["Role-Based Access"]
        Teacher[Teacher View<br/>Read student beliefs]
        Parent[Parent View<br/>Scoped access]
        Counselor[Counselor View<br/>Academic beliefs only]
    end
    
    Backend --> Auth
    Backend --> S1
    Backend --> S2
    Backend --> S3
    Backend --> S4
    
    Teacher -.->|"access_grants"| S1
    Teacher -.->|"access_grants"| S2
    Parent -.->|"consent_grants"| S1
    Counselor -.->|"scoped read"| S3
    
    style S1 fill:#ffd,stroke:#333
    style S2 fill:#ffd,stroke:#333
    style S3 fill:#ffd,stroke:#333
    style S4 fill:#ffd,stroke:#333
```

---

## 8. Cross-Platform Data Flow

Unified view showing how all three integrations connect to Kernle.

```mermaid
graph TB
    subgraph Platforms["Integration Platforms"]
        subgraph OC["OpenClaw"]
            iMessage[iMessage]
            Telegram[Telegram]
            Discord[Discord]
            Gateway[Gateway]
            iMessage --> Gateway
            Telegram --> Gateway
            Discord --> Gateway
        end
        
        subgraph CC["Claude Code"]
            Terminal[Terminal]
            VSCode[VS Code]
            Terminal --> CCAgent[Agent]
            VSCode --> CCAgent
        end
        
        subgraph AISD["AISD"]
            WebApp[Web App]
            Mobile[Mobile App]
            WebApp --> AISDBackend[Backend]
            Mobile --> AISDBackend
        end
    end
    
    subgraph Kernle["Kernle Infrastructure"]
        CLI[CLI Interface]
        MCP[MCP Server]
        PyAPI[Python API]
        
        Gateway -->|"workspace files +<br/>kernle CLI"| CLI
        CCAgent -->|"kernle CLI"| CLI
        AISDBackend -->|"Python API"| PyAPI
        
        subgraph Core["Core Engine"]
            Memory[Memory Layers]
            Boot[Boot Config]
            Checkpoint[Checkpoints]
            Sync[Sync Engine]
        end
        
        CLI --> Core
        MCP --> Core
        PyAPI --> Core
    end
    
    subgraph Storage["Storage"]
        SQLite[(Local SQLite<br/>Per-agent)]
        
        subgraph Cloud["Cloud (Supabase)"]
            PG[(PostgreSQL)]
            Vector[(pgvector)]
        end
        
        Core --> SQLite
        Sync --> Cloud
    end
    
    style OpenClaw fill:#e6f3ff,stroke:#333
    style CC fill:#fff3e6,stroke:#333
    style AISD fill:#e6ffe6,stroke:#333
```

### Entry Points Comparison

```mermaid
flowchart TB
    subgraph Entry["Platform Entry Points"]
        OC_Entry[OpenClaw:<br/>Workspace file injection<br/>+ Agent CLI calls]
        CC_Entry[Claude Code:<br/>Direct CLI calls<br/>in terminal]
        AISD_Entry[AISD:<br/>Python API calls<br/>from backend]
    end
    
    subgraph Config["Config Access"]
        OC_Config[boot.md file<br/>+ MEMORY.md]
        CC_Config[kernle load<br/>includes boot]
        AISD_Config[boot_get API<br/>instant lookup]
    end
    
    subgraph Memory["Memory Access"]
        OC_Mem[kernle load<br/>after agent starts]
        CC_Mem[kernle load<br/>at session start]
        AISD_Mem[k.load<br/>via Python API]
    end
    
    OC_Entry --> OC_Config --> OC_Mem
    CC_Entry --> CC_Config --> CC_Mem
    AISD_Entry --> AISD_Config --> AISD_Mem
    
    style OC_Entry fill:#e6f3ff,stroke:#333
    style CC_Entry fill:#fff3e6,stroke:#333
    style AISD_Entry fill:#e6ffe6,stroke:#333
```

---

## 9. Boot Layer Flow

Detailed flow of the boot layer across all platforms.

```mermaid
flowchart TB
    subgraph Set["Setting Boot Config"]
        CLI_Set[kernle boot set key value]
        API_Set[k.boot_set key, value]
        CLI_Set --> SQLite_Boot
        API_Set --> SQLite_Boot
        SQLite_Boot[(boot_config table)]
    end
    
    subgraph Checkpoint["On Checkpoint"]
        Save[kernle checkpoint save]
        Save --> Export[Auto-export boot.md]
        Export --> File[~/.kernle/agent/boot.md]
        Save --> Include[Include in MEMORY.md]
    end
    
    subgraph Read["Reading Boot Config"]
        subgraph OpenClaw_Read["OpenClaw"]
            File2[boot.md<br/>Pre-agent]
            Mem[MEMORY.md<br/>Pre-agent]
            Load1[kernle load<br/>Post-agent]
        end
        
        subgraph CC_Read["Claude Code"]
            Load2[kernle load<br/>Session start]
        end
        
        subgraph AISD_Read["AISD"]
            Get[k.boot_get<br/>Instant]
            Load3[k.load<br/>Full context]
        end
    end
    
    SQLite_Boot --> Checkpoint
    Checkpoint --> Read
    
    style File fill:#dfd,stroke:#333
    style Get fill:#dfd,stroke:#333
```

---

## Summary: Key Architecture Principles

| Principle | Implementation |
|-----------|----------------|
| **Memory is infrastructure** | Stack model — memory separate from runtime |
| **Local-first** | SQLite primary, cloud sync optional |
| **Platform-agnostic** | Same Kernle, different entry points |
| **Zero-latency config** | Boot layer for instant access |
| **Privacy by default** | Per-agent isolation, explicit sharing |
| **Future-proof** | Stacks portable across models |

---

*Diagrams created collaboratively by Claire and Ash, 2026-02-02*
