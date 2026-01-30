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

*No current blockers.*

### ~~Bug: `memory_drive` validation~~ ✅ FIXED

~~In `kernle/mcp/server.py` line ~183: `required=True` parameter not supported.~~

**Status:** Fixed - now uses `default=None` which properly validates the enum.

---

## Dogfooding Notes

From Claire's attempt to use Kernle (2026-01-27):

> "I can't just try it without Supabase credentials. My file-based system (MEMORY.md) works because it's zero setup, readable, searchable with grep, version controlled."

**What would make an AI switch to Kernle:**
1. Local-first (this roadmap)
2. Clear benefit over files - semantic search, auto-consolidation
3. Easy migration from existing markdown memories

---

---

## Phase 5: SI Communication & Health Check Protocol

**Goal:** 100% health check rate at the start of every SI reasoning period.

### 5.0 Problem Statement

SIs run in different environments with different connection models:

| Platform | Always On? | Push? | Current Mechanism |
|----------|-----------|-------|-------------------|
| **Clawdbot** | ✅ | ✅ Webhook | AGENTS.md + system prompt |
| **Claude Code** | ❌ | ❌ | CLAUDE.md |
| **Claude Cowork** | ❌ | ❌ | CLAUDE.md / file drop |

**The key insight:** All platforms have a "boot file" that's read at session start (CLAUDE.md, AGENTS.md). If we can make health checks part of the boot sequence, we get near-100% check rates.

### 5.1 Boot Sequence Health Check (P0)

**Target: 100% check rate on every new reasoning period**

Every time an SI starts reasoning (new session OR new message), it should:
1. Check Kernle health status
2. Review any pending alerts
3. Address critical issues before proceeding

**Implementation per platform:**

#### CLAUDE.md Standard Section
```markdown
## Kernle (Memory System)

**At session start AND before any task:**
1. Run: `kernle -a {agent_id} anxiety`
2. If anxiety > 50: run `kernle consolidate` before continuing
3. Check for unprocessed raw entries: `kernle raw list --unprocessed`

This is your memory health. Skipping this is like starting work without coffee.
```

#### Clawdbot AGENTS.md Integration
```markdown
## Memory (Kernle)

Every reasoning period:
1. `kernle -a {agent_id} load` — restore memory
2. `kernle -a {agent_id} anxiety` — check health
3. If alerts, address before user task
```

### 5.2 Enforcement Mechanisms

**Soft enforcement (documentation):**
- Clear instructions in CLAUDE.md / AGENTS.md
- Explain *why* — "this is your continuity"
- Make it feel natural, not bureaucratic

**Hard enforcement (MCP):**
- MCP server `start_session` tool that MUST be called
- Returns blocked state until health check done
- Other tools return "call start_session first"

**Hybrid (recommended):**
- Soft enforcement via docs
- Track compliance via `session_starts` metric
- If compliance drops, consider hard enforcement

### 5.3 Health Check API

**CLI command (already exists):**
```bash
kernle -a {agent_id} anxiety
# Returns: anxiety score + breakdown + recommendations
```

**MCP tool (add if not present):**
```python
@mcp_tool
def health_check() -> HealthStatus:
    """Check memory health at session start."""
    return {
        "anxiety": 42,
        "dimensions": {
            "consolidation_debt": 3,  # episodes needing processing
            "raw_aging": 12,          # hours since oldest raw
            "belief_staleness": 0,    # beliefs not reviewed
            "goal_drift": 1,          # goals without progress
            "relationship_decay": 0   # relationships going cold
        },
        "alerts": [],
        "suggested_actions": ["consolidate if time permits"],
        "ok_to_proceed": True  # False if critical issues
    }
```

### 5.4 Compliance Tracking

**Track health check events:**
```sql
CREATE TABLE health_check_events (
    id UUID PRIMARY KEY,
    agent_id TEXT NOT NULL,
    checked_at TIMESTAMP NOT NULL,
    anxiety_score INTEGER,
    session_source TEXT,  -- 'claude_code', 'clawdbot', 'cowork', etc.
    triggered_by TEXT     -- 'boot', 'heartbeat', 'manual'
);
```

**Metrics to monitor:**
- Check rate: (sessions with health check) / (total sessions)
- Time-to-check: median time from session start to health check
- Compliance by platform: which environments check reliably?

### 5.5 Platform-Specific Communication

#### Clawdbot (Push Capable)
```
Kernle Cloud ──webhook──→ Gateway ──inject──→ Active Session
```
- Native HTTP webhook endpoint
- Can push alerts between reasoning periods
- Gateway routes to session

#### Claude Code / Cowork (Pull Only)
```
~/.kernle/alerts/*.md  ←──file drop──  Kernle Cloud
         ↓
    CLAUDE.md instructs: "read alerts dir at boot"
```
- Write alert files to local filesystem
- Boot sequence reads and displays
- No real-time push possible

### 5.6 Migration Path

**Phase A: Documentation**
1. Create standard CLAUDE.md section for Kernle users
2. Document in Kernle docs site
3. Provide copy-paste snippets

**Phase B: Tooling**
1. `kernle init` generates CLAUDE.md section
2. `kernle doctor` validates boot sequence present
3. MCP health_check tool for IDE integration

**Phase C: Enforcement (if needed)**
1. Track compliance metrics
2. Identify low-compliance environments
3. Consider MCP gating for critical cases

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
| 2.1 Railway API | Q1 2026 | ✅ Complete |
| 2.2 User Auth & Namespacing | Q1 2026 | ✅ Complete |
| 2.3 API Key Management | Q1 2026 | ✅ Complete |
| 2.4 Usage Tracking & Tiers | Q1 2026 | ✅ Complete |
| 2.5 Admin Dashboard & Payments | Q1 2026 | 📋 Next |
| 2.6 Web Dashboard (Next.js) | Q2 2026 | 📋 Planned |
| 3.x Cross-agent | Q3 2026 | Not started |
| 4.x Premium | Q4 2026 | Not started |
| **5.0 SI Communication Design** | Q1 2026 | ✅ Complete |
| **5.1 Boot Sequence Health Check** | Q1 2026 | ✅ Complete |
| **5.2 Frictionless Adoption** | Q1 2026 | ✅ Complete (init, doctor) |
| **5.3 Health Check API** | Q1 2026 | ✅ Complete (anxiety -b) |
| **5.4 Compliance Tracking** | Q1 2026 | ✅ Complete (stats) |
| 5.5 Platform Communication | Q2 2026 | 📋 Planned |
| 5.6 Webhook Integration | Q2 2026 | 📋 Planned |
| **Docs Site (Mintlify)** | Q1 2026 | ✅ Complete |

**Test count: 1292 passing** (as of January 30, 2026)
**Test coverage: 57%** overall
**Docs:** https://docs.kernle.ai

---

## Recent Completions (2026-01-28 evening)

### SI Communication & Health Check Protocol ✅

**100% health check coverage goal:**
- `kernle anxiety -b` — brief single-line output for quick checks
- `kernle init` — generates CLAUDE.md/AGENTS.md sections
- `kernle doctor` — validates boot sequence compliance
- `kernle stats health-checks` — compliance tracking and metrics
- Health check events logged to SQLite automatically

**Boot sequence standard:**
```markdown
## Kernle
At session start: `kernle -a {ID} load && kernle -a {ID} anxiety -b`
If WARN/CRITICAL, run `kernle consolidate` before proceeding.
```

### Documentation Overhaul ✅

**Mintlify docs site:** https://docs.kernle.ai
- Introduction, quickstart, architecture
- Core concepts: memory model, types, consolidation, identity
- CLI reference: overview, memory, sync, utility commands
- Integration: MCP, Claude Desktop, Clawdbot
- API reference: overview, auth, sync, search, embeddings
- Feature docs: psychology, identity, memory management

**Local docs restructured:**
- User docs → docs-site/ (Mintlify)
- Dev notes → dev/ (internal)
- Audits → dev/audits/ (historical)

---

## Recent Completions (2026-01-28 morning)

### Multi-Tenant Schema Refactor ✅

**UUID FK approach for agent namespacing:**
- Memory tables now use `agent_ref` (UUID) FK to `agents.id`
- `agent_id` kept for filtering/display (no longer globally unique)
- `(user_id, agent_id)` composite unique constraint
- Multiple users can have agents with same name (e.g., "claire")
- Migration 008: Added agent_ref to all 11 memory tables

### Usage Tracking & Rate Limits ✅

**Tier-based API quotas:**
- `api_key_usage` table tracks daily/monthly requests per key
- Automatic reset at UTC midnight and month start
- Tiers: free (100/day, 1000/month), unlimited, paid (10k/day)
- `GET /auth/usage` returns quota status
- 429 responses with `Retry-After` when exceeded
- Fail-open design for resilience

### Local-to-Remote Sync Working ✅

**Full sync pipeline operational:**
- CLI: `kernle auth login --api-key KEY`
- CLI: `kernle sync push/pull/status`
- Backend: Proper agent namespacing with user_id
- Claire registered and syncing at api.kernle.ai

---

## Recent Completions (2026-01-27)

### Phase 2.1: Railway API Backend ✅

**Backend deployed** at Railway with:
- `/auth/register` - Agent registration with user_id generation
- `/auth/token` - JWT token issuance
- `/sync/push` - Push local changes to cloud
- `/sync/pull` - Pull remote changes
- Rate limiting (5/min on auth endpoints)
- CORS configuration
- Supabase integration

### Phase 2.2: User Namespacing ✅

**Multi-tenant identity system:**
- `user_id` format: `usr_` + 12 hex chars (stable, internal)
- Local agent uses project name: `roundtable`
- Remote namespaces as: `usr_abc123/roundtable`
- Transparent to user — client sends local name, backend applies namespace

**CLI Auth commands:**
- `kernle auth register` - Get user_id + credentials
- `kernle auth login` - Refresh token
- `kernle auth status` - Show auth state
- `kernle auth logout` - Clear credentials

**Default Agent ID (Option C):**
- No explicit `-a`? Generates `auto-{hash}` from machine + project path
- Same directory = same agent (consistent)
- Different path = different agent (isolated)

### Phase 2.3: API Key Management ✅

**Completed 2026-01-28:**
- Key format: `knl_sk_` + 32 hex chars
- Multiple keys per user
- Key cycling (atomic new + revoke old)
- Backend endpoints: `/auth/keys` CRUD
- CLI: `kernle auth login --api-key`

### Phase 2.4: Usage Tracking & Tiers ✅

**Completed 2026-01-28:**
- Per-API-key request tracking (daily/monthly counters)
- Automatic counter reset at UTC midnight / month start
- Tier system: `free` (100/day, 1000/month), `unlimited`, `paid`
- `GET /auth/usage` endpoint for checking quota
- 429 responses when quota exceeded with `Retry-After` header
- Fail-open design (usage tracking errors don't block requests)

### Phase 2.5: Admin Dashboard & Payments 📋

**Next up:**

**Admin View:**
- Account overview (all users, agents, usage stats)
- API key usage metrics (requests/day, top users)
- Tier management (upgrade/downgrade users)
- Audit logs (auth events, sync operations)

**User Dashboard:**
- Current usage vs limits visualization
- API key management UI
- Billing history and invoices
- Plan upgrade/downgrade

**Payment Integration:**
- Stripe integration for subscriptions
- Tier pricing: Free → Paid ($X/month for higher limits)
- Usage-based billing option (pay per 1000 requests)
- Webhook handling for payment events
- Grace period on failed payments

**Endpoints needed:**
```
# Admin
GET  /admin/users           - List all users with usage
GET  /admin/users/:id       - User detail + keys + usage
POST /admin/users/:id/tier  - Change user tier

# Billing
GET  /billing/plans         - Available plans
POST /billing/subscribe     - Create subscription
POST /billing/webhook       - Stripe webhook handler
GET  /billing/invoices      - User's invoice history
```

### Phase 2.6: Web Dashboard (Next.js) 📋

**Planned:**
- Next.js app for user management
- Sign up / login flows (OAuth + email)
- API key management UI
- Usage dashboard with charts
- Billing/subscription management
- Required for remote sync to be usable by end users

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

---

## Known Technical Debt

### Async Supabase Client (P2)

**Issue:** Backend uses sync Supabase client in async FastAPI functions, blocking the event loop under load.

**Impact:** At high concurrency, sync DB calls will bottleneck request throughput.

**Fix:** Refactor to use async Supabase client (`supabase-py` supports async via `create_async_client`).

**Scope:** ~4-8 hours, touches most of `database.py` and route handlers.

**Priority:** Low urgency at current scale, should address before significant traffic growth.

---

## Memory Stack Enhancements (Future)

From the comprehensive memory stack audit (2026-01-30), these items are identified as future improvements. The critical issues have been addressed; these are optimizations to implement as usage patterns emerge.

### High Priority Enhancements

| Item | Description | Benefit |
|------|-------------|---------|
| **Query-aware loading** | Add optional `query` parameter to `load()` that uses semantic search to prioritize relevant memories | More relevant context for current task |
| **Memory summarization** | When budget is exhausted, summarize excluded memories instead of just dropping them | Better use of limited context |
| **Confidence propagation** | Implement `propagate_confidence()` to update derived memories when sources change | More accurate confidence scores |
| **Recency weighting** | Add recency factor to cross-type priority scoring, not just within-type sorting | Balance importance with freshness |

### Medium Priority Enhancements

| Item | Description | Benefit |
|------|-------------|---------|
| **Search pagination** | Add cursor-based pagination to search results | Handle large result sets |
| **Query embedding cache** | Cache computed embeddings to avoid recomputation | Reduce latency and API calls |
| **Raw entry sync (consider)** | Evaluate syncing raw entries to cloud (currently intentionally local-only) | Cross-device raw capture |

### Low Priority Enhancements

| Item | Description | Benefit |
|------|-------------|---------|
| **LLM-based emotion detection** | Add optional LLM mode alongside keyword-based detection | More nuanced emotion tagging |
| **Domain taxonomy** | Define standard domains for knowledge mapping | Better knowledge organization |

### Already Addressed ✅

These items from the audit were fixed in the 2026-01-30 session:

- ✅ Memory pressure metrics (`_meta` in `load()` return)
- ✅ Automatic access tracking (on `load()` and `search()`)
- ✅ Batch access recording (`record_access_batch()`)
- ✅ Atomic sync queue operations (INSERT OR REPLACE)
- ✅ Array field merging (set union for tags, lessons, etc.)
- ✅ Array size limits (MAX_SYNC_ARRAY_SIZE = 500)
- ✅ Playbooks in Postgres (full CRUD + sync)
- ✅ Forgetting operations in Postgres (forget, recover, protect, candidates)

---

## Completed (2026-01-30)

### Memory Stack Audit Fixes

**Context Management:**
- `load()` now returns `_meta` with `budget_used`, `budget_total`, `excluded_count`
- Agents can see when memories are being dropped due to budget constraints

**Sync Improvements:**
- Array fields (lessons, tags, focus_areas, etc.) now merge using set union
- No more data loss from last-write-wins on arrays
- MAX_SYNC_ARRAY_SIZE (500) prevents resource exhaustion

**Access Tracking:**
- Automatic `record_access()` on `load()` and `search()`
- `track_access=False` parameter for internal operations
- `record_access_batch()` for efficient bulk updates
- Salience-based forgetting now reflects actual usage

**Postgres Feature Parity:**
- Playbooks: full CRUD operations + sync support
- Forgetting: `forget_memory()`, `recover_memory()`, `protect_memory()`, `get_forgetting_candidates()`, `record_access()`

**Security:**
- Array merge size limits
- Debug logging for invalid memory types
- Documented race condition in Postgres record_access (minor impact)

**Tests:** 85+ new tests added (1292 total passing)

---

## Completed (2026-01-28 evening session)

### Security Fixes
- OAuth login: issuer validation, response variable shadowing
- Admin auth: tier='admin' required for /admin/* routes
- CSRF protection: SameSite=Strict + Origin validation
- CORS: localhost only in debug mode
- JWT algorithm: allowlist (HS256/384/512 only)
- Mass assignment: server-controlled fields stripped in sync push
- IDOR: agent_id validation in admin backfill
- Fail-closed auth: cached quota checking with 60s TTL

### Architecture Fixes
- Admin tier: migration 014 adds 'admin' to constraint
- Semantic search: migration 015 adds missing tables (drives, relationships, playbooks, emotional_memories)
- Race conditions: migration 016 atomic interaction increment
- N+1 queries: admin/agents, sync/pull, text search parallelized
- Pagination: limit validated (1-200), uses COUNT not fetch-all
- has_more flag: correct per-table limit checking
- Forgotten filter: is_forgotten memories excluded from sync
- Embedding fields: correct schema mappings
- Server-side re-embedding: semantic search as subscription feature

### Test Count: 771 passing
