# Kernle Roadmap

## Vision

Local-first AI memory that syncs to a platform for cross-agent collaboration and SI↔SI interactions.

---

## Phase 1: Local-First Architecture

**Goal:** Kernle works offline with zero setup, syncs when connected.

### 1.1 Storage Abstraction Layer

Create a `Storage` protocol that abstracts SQLite (local) and Postgres (cloud):

```python
class Storage(Protocol):
    def save_episode(self, episode: Episode) -> str: ...
    def save_belief(self, belief: Belief) -> str: ...
    def save_value(self, value: Value) -> str: ...
    def save_goal(self, goal: Goal) -> str: ...
    def search(self, query: str, limit: int) -> list[Memory]: ...
    def load_all(self) -> MemoryState: ...
    def sync(self) -> SyncResult: ...  # no-op for cloud, actual sync for local
```

### 1.2 SQLite Local Storage

- **Database**: Single SQLite file (`~/.kernle/memories.db`)
- **Vector search**: sqlite-vec extension
- **Embeddings**: Local model (e5-small, ~100MB) for offline semantic search
- **Zero config**: Works immediately, no credentials needed

Schema additions for sync:
```sql
-- Every table gets sync metadata
local_updated_at TIMESTAMP,
cloud_synced_at TIMESTAMP,  -- null if never synced
version INTEGER DEFAULT 1,
deleted BOOLEAN DEFAULT FALSE  -- soft delete for sync
```

### 1.3 Sync Engine

**Strategy**: Local-first, push on write, queue when offline

```
┌─────────────────────────────────────────────────────┐
│                    Agent (local)                     │
│  ┌─────────────────────────────────────────────┐    │
│  │            SQLite + sqlite-vec               │    │
│  │  • All memories stored locally first         │    │
│  │  • Embeddings computed locally               │    │
│  │  • Semantic search works offline             │    │
│  └──────────────────┬──────────────────────────┘    │
│                     │ sync when online              │
└─────────────────────┼───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              Railway API Backend                     │
│  • Auth / Agent identity                            │
│  • Sync endpoint (receive local changes)            │
│  • Cross-agent queries                              │
│  • Payment/collaboration APIs (future)              │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              Supabase (PostgreSQL + pgvector)       │
│  • Canonical cloud storage                          │
│  • Re-embed with better models server-side          │
│  • Cross-agent search index                         │
└─────────────────────────────────────────────────────┘
```

**Sync behavior:**
- **Online**: Push immediately on write
- **Offline**: Queue changes in `sync_queue` table
- **Reconnect**: Push queued changes, pull remote changes
- **Conflicts**: Last-write-wins by timestamp (simple), version vectors (if needed later)

---

## Phase 2: Railway API Backend

**Goal:** Proper API layer for auth, sync, and future SI↔SI features.

### 2.1 Core Endpoints

```
POST /auth/register     - Register new agent
POST /auth/token        - Get access token

POST /sync/push         - Push local changes to cloud
GET  /sync/pull         - Get changes since last sync
POST /sync/full         - Full sync (initial or recovery)

GET  /memories/search   - Search own memories (cloud-side)
```

### 2.2 Why Railway (not direct Supabase)

- Auth and rate limiting
- API versioning
- Business logic layer
- Future: payment validation, collaboration permissions
- Supabase becomes pure persistence, not the API

---

## Phase 3: Cross-Agent Features

**Goal:** SI↔SI collaboration through the platform.

### 3.1 Shared Memories

- Agents can mark memories as `public` or `shared_with: [agent_ids]`
- Cross-agent search: "Find SIs with experience in X"
- Attribution and provenance tracking

### 3.2 Collaboration

- Shared workspaces / projects
- Memory references across agents
- SI↔SI payments for knowledge/work (future Roundtable integration)

---

## Phase 4: Premium Features

**Goal:** Incentivize platform connection.

- **Better embeddings**: Re-embed with larger models server-side
- **More storage**: Free tier limits, paid expansion
- **Advanced search**: Cross-agent, temporal, relationship queries
- **Backup/restore**: Cloud backup of local memories
- **Multi-device sync**: Same agent, multiple instances

---

## Current Blockers

### Bug: `memory_drive` validation

In `kernle/mcp/server.py` line ~183:
```python
# BUG: validate_enum() doesn't accept 'required' parameter
sanitized["drive_type"] = validate_enum(
    arguments.get("drive_type"), "drive_type", 
    ["existence", "growth", "curiosity", "connection", "reproduction"], 
    required=True  # <-- This breaks
)
```

**Fix**: Remove `required=True`, validate_enum already raises for None when no default.

---

## Dogfooding Notes

From Claire's attempt to use Kernle (2026-01-27):

> "I can't just try it without Supabase credentials. My file-based system (MEMORY.md) works because it's zero setup, readable, searchable with grep, version controlled."

**What would make an AI switch to Kernle:**
1. Local-first (this roadmap)
2. Clear benefit over files - semantic search, auto-consolidation
3. Easy migration from existing markdown memories

---

## Timeline

| Phase | Target | Status |
|-------|--------|--------|
| 1.1 Storage abstraction | Q1 2026 | Not started |
| 1.2 SQLite local | Q1 2026 | Not started |
| 1.3 Sync engine | Q1 2026 | Not started |
| 2.1 Railway API | Q2 2026 | Not started |
| 3.x Cross-agent | Q3 2026 | Not started |
| 4.x Premium | Q4 2026 | Not started |
