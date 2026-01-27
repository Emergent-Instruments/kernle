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
| 1.1 Storage abstraction | Q1 2026 | ✅ Complete |
| 1.2 SQLite local | Q1 2026 | ✅ Complete |
| 1.3 Sync engine | Q1 2026 | ✅ Complete |
| **P0 MCP Server** | Q1 2026 | ✅ Complete (23 tools) |
| **P1 Identity Synthesis** | Q1 2026 | ✅ Complete |
| **P1 Emotional Memory** | Q1 2026 | ✅ Complete |
| **P1 Meta-Memory** | Q1 2026 | ✅ Complete |
| **Anxiety Tracking** | Q1 2026 | ✅ Complete (5-dimension model) |
| **Raw Layer + Dump** | Q1 2026 | ✅ Complete |
| **Playbooks** | Q1 2026 | ✅ Complete |
| **Controlled Forgetting** | Q1 2026 | ✅ Complete |
| **Belief Revision** | Q1 2026 | ✅ Complete |
| **Meta-cognition** | Q1 2026 | ✅ Complete |
| 2.1 Railway API | Q2 2026 | Not started |
| 3.x Cross-agent | Q3 2026 | Not started |
| 4.x Premium | Q4 2026 | Not started |

**Test count: 497 passing** (as of January 27, 2026)

## Implementation Notes

### Phase 1.1 & 1.2 (Completed 2026-01-27)

**Storage Protocol** (`kernle/storage/base.py`):
- `Storage` protocol with full CRUD for all memory types
- Data classes: Episode, Belief, Value, Goal, Note, Drive, Relationship
- Sync metadata: `local_updated_at`, `cloud_synced_at`, `version`, `deleted`
- `SyncResult` for tracking sync operations

**SQLite Storage** (`kernle/storage/sqlite.py`):
- Zero-config local storage at `~/.kernle/memories.db`
- Vector search using `sqlite-vec` extension
- Hash-based embeddings for offline semantic search (no ML deps)
- Optional OpenAI embeddings when API key available
- Sync queue for offline-first operation

**Embeddings** (`kernle/storage/embeddings.py`):
- `HashEmbedder`: Deterministic, fast, zero-dependency (default)
- `OpenAIEmbedder`: Cloud embeddings when available
- 384-dimension embeddings (matches e5-small for future upgrade)

### Phase 1.3 (Completed 2026-01-27)

**Sync Engine** (`kernle/storage/sqlite.py`):
- **Sync Queue**: Enhanced `sync_queue` table with payload and deduplication
  - Auto-queues changes on every save operation
  - Deduplicates by (table, record_id) - keeps only latest operation
- **Connectivity Detection**: `is_online()` method with caching
  - Pings cloud storage to check reachability
  - Results cached for 30 seconds to avoid excessive checks
- **Push Sync**: `sync()` method pushes queued changes to cloud
  - Processes queue in order
  - Marks records as synced (`cloud_synced_at`)
  - Clears queue entries on success
  - Continues on partial failure
- **Pull Sync**: `pull_changes()` fetches remote updates
  - Pulls from all tables since last sync time
  - Merges with local records using conflict resolution
- **Conflict Resolution**: Last-write-wins by timestamp
  - Compares `cloud_synced_at` vs `local_updated_at`
  - Newer record wins
  - Conflicts counted but automatically resolved
- **Sync Metadata**: `sync_meta` table tracks sync state
  - `last_sync_time` persisted across sessions

**Protocol Updates** (`kernle/storage/base.py`):
- Added `QueuedChange` dataclass for queue entries
- Added `pull_changes(since)` method to protocol
- Added `is_online()` method to protocol

**Tests** (`tests/test_sync_engine.py`):
- 31 comprehensive tests covering:
  - Queue basics (auto-queue, deduplication, persistence)
  - Connectivity detection (online/offline, caching)
  - Push sync (all record types, queue clearing, error handling)
  - Pull sync (new records, conflict detection)
  - Conflict resolution (cloud-wins, local-wins scenarios)
  - Sync metadata (last sync time tracking, persistence)
  - Edge cases (deleted records, empty queue, partial failures)
