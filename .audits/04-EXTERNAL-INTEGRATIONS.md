# Kernle External Integrations & APIs

> **Date**: 2026-01-29
> **Purpose**: Document how Kernle integrates with external systems

---

## 1. CLI Interface

### Installation & Invocation

```bash
# Install
pip install kernle

# Invoke (all commands require agent_id)
kernle -a <agent_id> <command> [options]
kernle --agent <agent_id> <command> [options]
```

### Command Categories

#### Memory Operations
```bash
# Load working memory with budget
kernle -a myagent load --budget 8000

# Checkpoints
kernle -a myagent checkpoint save --task "Working on X" --pending "item1,item2"
kernle -a myagent checkpoint load
kernle -a myagent checkpoint clear

# Quick status
kernle -a myagent status
kernle -a myagent resume
```

#### Content Capture
```bash
# Episodes (experiences)
kernle -a myagent episode "Implemented auth" "Success - JWKS working" \
    --lessons "JWKS is reliable" --tags "auth,security"

# Notes
kernle -a myagent note "Decision: Use UPSERT for sync queue" --type decision \
    --reason "Atomic, no race conditions"

# Beliefs
kernle -a myagent belief "Sync queues need deduplication" \
    --type fact --confidence 0.85

# Raw capture (proposed: just blob)
kernle -a myagent raw "Quick thought about auth..."
kernle -a myagent raw --stdin < notes.txt
```

#### Identity & Goals
```bash
# Values (protected by default)
kernle -a myagent value "Correctness" "Prefer correct over fast" --priority 90

# Goals
kernle -a myagent goal "Ship sync feature" --priority high

# Drives
kernle -a myagent drive set growth --intensity 0.8 --focus "learning Rust"
```

#### Search & Query
```bash
# Semantic search
kernle -a myagent search "authentication patterns"

# Temporal queries
kernle -a myagent when today
kernle -a myagent when "last week"
kernle -a myagent temporal --since "2026-01-01" --until "2026-01-15"
```

#### Sync & Auth
```bash
# Sync operations
kernle -a myagent sync status
kernle -a myagent sync push
kernle -a myagent sync pull
kernle -a myagent sync full

# Authentication
kernle -a myagent auth register
kernle -a myagent auth login
kernle -a myagent auth status
kernle -a myagent auth keys list
```

#### Advanced Operations
```bash
# Consolidation
kernle -a myagent consolidate --min-episodes 5

# Forgetting
kernle -a myagent forget candidates --threshold 0.3
kernle -a myagent forget run --limit 10 --dry-run

# Anxiety monitoring
kernle -a myagent anxiety

# Raw entry management
kernle -a myagent raw list --unprocessed
kernle -a myagent raw process <id> --type episode
kernle -a myagent raw triage
```

---

## 2. MCP (Model Context Protocol) Interface

### Configuration

**.mcp.json** (in project root):
```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "--agent", "claude-code"]
    }
  }
}
```

### Available Tools (28 total)

#### Core Memory
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_load` | Load working memory | `budget` (100-50000) |
| `memory_checkpoint_save` | Save checkpoint | `task`, `pending_items`, `context` |
| `memory_checkpoint_load` | Load checkpoint | - |
| `memory_status` | Get memory statistics | - |

#### Content Capture
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_episode` | Record experience | `objective`, `outcome`, `lessons`, `tags` |
| `memory_note` | Capture note | `content`, `type`, `speaker`, `reason` |
| `memory_belief` | Add belief | `statement`, `type`, `confidence` |
| `memory_value` | Add value | `name`, `statement`, `priority` |
| `memory_goal` | Create goal | `title`, `description`, `priority` |
| `memory_drive` | Set drive | `drive_type`, `intensity`, `focus_areas` |
| `memory_auto_capture` | Raw capture + suggestions | `content`, `tags`, `source` |

#### Search & Query
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_search` | Semantic search | `query`, `limit`, `min_score` |
| `memory_note_search` | Search notes | `query`, `limit` |
| `memory_when` | Temporal query | `period` (today/yesterday/this week/last hour) |
| `memory_consolidate` | Reflection scaffold | `min_episodes` |

#### List Operations
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_belief_list` | List beliefs | `limit` |
| `memory_value_list` | List values | `limit` |
| `memory_goal_list` | List goals | `status`, `limit` |
| `memory_drive_list` | List drives | - |

#### Update Operations
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_episode_update` | Update episode | `episode_id`, `outcome`, `lessons`, `tags` |
| `memory_goal_update` | Update goal | `goal_id`, `status`, `priority`, `description` |
| `memory_belief_update` | Update belief | `belief_id`, `statement`, `confidence` |

#### Suggestions
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_suggestions_list` | List suggestions | `status`, `limit` |
| `memory_suggestions_promote` | Promote raw entry | `suggestion_id`, `type` |
| `memory_suggestions_reject` | Reject suggestion | `suggestion_id`, `reason` |

#### Sync
| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `memory_sync` | Trigger sync | `direction` (push/pull/full) |

### MCP Tool Response Format

All tools return structured JSON:
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

Or on error:
```json
{
  "success": false,
  "data": null,
  "error": "Error message"
}
```

---

## 3. Python SDK

### Installation
```bash
pip install kernle
```

### Basic Usage
```python
from kernle import Kernle

# Initialize (auto-detects storage)
k = Kernle(agent_id="my_agent")

# Or explicit storage
from kernle.storage import SQLiteStorage
storage = SQLiteStorage(agent_id="my_agent", db_path="~/.kernle/my_agent.db")
k = Kernle(agent_id="my_agent", storage=storage)
```

### Session Workflow
```python
# Start session
memory = k.load(budget=8000, sync=True)

# Work
k.raw("Quick thought...")
k.episode("Did X", "Result Y", lessons=["Learned Z"])
k.note("Decision: Use approach A", type="decision", reason="Because B")

# End session
k.checkpoint("Current task", ["Next step 1", "Next step 2"])
k.sync()
```

### Full API Reference

#### Memory Loading
```python
# Load with budget-aware selection
memory = k.load(
    budget=8000,           # Token budget
    truncate=True,         # Truncate long items
    max_item_chars=500,    # Max chars per item
    sync=None              # None=use default, True/False=override
)

# Checkpoint operations
k.checkpoint(task, pending_items, context, sync=None)
checkpoint = k.load_checkpoint()
k.clear_checkpoint()
```

#### Content Capture
```python
# Episode
episode_id = k.episode(
    objective="What was attempted",
    outcome="What happened",
    lessons=["Lesson 1", "Lesson 2"],
    tags=["tag1", "tag2"],
    context="project:myproject"
)

# Note
note_id = k.note(
    content="The content",
    type="note",           # note | decision | insight | quote
    speaker="Person",      # For quotes
    reason="Why",          # For decisions
    tags=["tag"],
    protect=False
)

# Belief
belief_id = k.belief(
    statement="The belief",
    type="fact",           # fact | preference | observation
    confidence=0.8
)

# Raw capture (proposed: simplified)
raw_id = k.raw("Brain dump text")

# Process raw to structured
memory_id = k.process_raw(raw_id, as_type="episode", objective="...")
```

#### Identity & Goals
```python
# Value
k.value(name="Integrity", statement="Always be honest", priority=90)

# Goal
goal_id = k.goal(title="Ship feature", description="...", priority="high")
k.update_goal(goal_id, status="completed")

# Drive
k.drive(drive_type="growth", intensity=0.8, focus_areas=["Rust", "systems"])
k.satisfy_drive("growth", amount=0.2)

# Relationship
rel_id = k.relationship(
    name="Alice",
    trust_level=0.8,
    notes="Collaborator on X",
    entity_type="person"
)
```

#### Search & Query
```python
# Semantic search
results = k.search("authentication patterns", limit=10)

# Temporal query
memories = k.what_happened(when="today")
memories = k.what_happened(when="last week")

# Load specific types
beliefs = k.load_beliefs(limit=20)
episodes = k.load_recent_work(limit=5)
lessons = k.load_lessons(limit=20)
```

#### Meta-Memory
```python
# Confidence operations
k.verify_memory("belief", belief_id, evidence="Confirmed by testing")
k.reinforce_belief(belief_id)

# Lineage
lineage = k.get_memory_lineage("belief", belief_id)

# Belief revision
new_id = k.supersede_belief(old_id, "Updated statement", reason="New evidence")
```

#### Forgetting
```python
# Get candidates
candidates = k.get_forgetting_candidates(threshold=0.3, limit=20)

# Run cycle
report = k.run_forgetting_cycle(threshold=0.3, limit=10, dry_run=True)

# Protect/recover
k.protect("belief", belief_id, protected=True)
k.recover("belief", belief_id)
```

#### Consolidation & Analysis
```python
# Get reflection scaffold
scaffold = k.consolidate(min_episodes=5)

# Identity synthesis
identity = k.synthesize_identity()
confidence = k.get_identity_confidence()
drift = k.detect_identity_drift(days=30)

# Find contradictions
contradictions = k.find_contradictions()
semantic = k.find_semantic_contradictions()
```

#### Sync & Export
```python
# Sync operations
result = k.sync()
status = k.get_sync_status()

# Export
content = k.dump(include_raw=True, format="markdown")
k.export("backup.md", include_raw=True, format="markdown")
```

---

## 4. Backend REST API

### Base URL
```
https://api.kernle.dev  (or self-hosted)
```

### Authentication
```
Authorization: Bearer <jwt_token>
# or
X-API-Key: knl_sk_<key>
```

### Endpoints

#### Sync Operations

**POST /sync/push** - Push local changes to cloud
```json
// Request
{
  "operations": [
    {
      "operation": "upsert",
      "table": "episodes",
      "record_id": "abc123",
      "data": { ... },
      "local_updated_at": "2026-01-29T14:30:00Z",
      "version": 1
    }
  ],
  "last_sync_at": "2026-01-29T14:00:00Z"
}

// Response
{
  "synced": 5,
  "conflicts": [],
  "server_time": "2026-01-29T14:30:05Z"
}
```

**POST /sync/pull** - Pull changes from cloud
```json
// Request
{
  "since": "2026-01-29T14:00:00Z"
}

// Response
{
  "operations": [...],
  "server_time": "2026-01-29T14:30:05Z",
  "has_more": false
}
```

**POST /sync/full** - Full sync (all records)
```json
// Response
{
  "operations": [...],
  "server_time": "2026-01-29T14:30:05Z",
  "has_more": false
}
```

#### Memory Search

**POST /memories/search** - Semantic search
```json
// Request
{
  "query": "authentication patterns",
  "limit": 10,
  "memory_types": ["episodes", "beliefs", "notes"]
}

// Response
{
  "results": [
    {
      "type": "episode",
      "id": "abc123",
      "content": { ... },
      "score": 0.87
    }
  ],
  "query": "authentication patterns",
  "total": 5
}
```

#### Authentication

**POST /auth/register** - Register new agent
**POST /auth/login** - Login
**GET /auth/me** - Get current user
**POST /auth/logout** - Logout

#### API Keys

**GET /auth/keys** - List API keys
**POST /auth/keys** - Create API key
**DELETE /auth/keys/{key_id}** - Revoke key

---

## 5. Cloud Storage (PostgreSQL/Supabase)

### Connection

Via environment variables:
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=xxx
```

Or in code:
```python
k = Kernle(
    agent_id="my_agent",
    supabase_url="https://xxx.supabase.co",
    supabase_key="xxx"
)
```

### Features

- **Semantic search**: pgvector with 1536-dim OpenAI embeddings
- **Multi-tenant**: Agent isolation via `agent_id`
- **Row-level security**: Enforced at database level
- **Conflict resolution**: Last-write-wins with conflict history

### Embedding Differences

| Storage | Dimension | Algorithm | API Required |
|---------|-----------|-----------|--------------|
| SQLite (local) | 384 | Hash-based | No |
| PostgreSQL (cloud) | 1536 | OpenAI | Yes |

Embeddings are **regenerated server-side** on sync push.

---

## 6. Configuration Files

### ~/.kernle/config.yaml (user config)
```yaml
default_agent: my-agent
auto_sync: true
sync_interval: 300  # seconds
token_budget: 8000
```

### .mcp.json (project MCP config)
```json
{
  "mcpServers": {
    "kernle": {
      "command": "kernle",
      "args": ["mcp", "--agent", "project-agent"]
    }
  }
}
```

### Environment Variables
```bash
# Storage
KERNLE_DB_PATH=~/.kernle/agents.db
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=xxx

# OpenAI (for cloud embeddings)
OPENAI_API_KEY=xxx

# Logging
KERNLE_LOG_LEVEL=INFO
```

---

## 7. Data Privacy & Security

### Local-First Architecture
- All data stored locally in SQLite by default
- Cloud sync is optional and explicit
- Credentials stored securely (bcrypt hashing)

### Cloud Security
- JWT-based authentication
- API key hashing (bcrypt)
- Row-level security on PostgreSQL
- Agent isolation (no cross-tenant access)
- Server-controlled fields prevent mass assignment

### Data Sensitivity
- No automatic capture of sensitive data
- Agent controls all memory content
- Soft delete preserves audit trail
- Sync conflicts logged for visibility
