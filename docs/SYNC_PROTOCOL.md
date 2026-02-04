# Sync Protocol

This document describes the sync protocol between local SQLite storage and the Kernle Cloud backend.

## Overview

Kernle uses a local-first architecture:
1. All operations happen on local SQLite first
2. Changes are queued for sync
3. When online, changes push to cloud and pull remote updates
4. Conflict resolution uses last-write-wins

## Operations

### Operation Types

| Operation | Meaning | Backend Behavior |
|-----------|---------|------------------|
| `update` | Upsert (insert or update) | **Recommended** - Works for both new and existing records |
| `insert` | Create new record | May fail if record exists; use `update` instead |
| `delete` | Soft delete | Sets `deleted=true` |

**Important:** The client SDK always uses `operation: "update"` internally, regardless of whether the record is new. This provides upsert semantics. External API consumers should do the same.

### Queue Entry Format

```python
{
    "table_name": "episodes",     # Target table
    "record_id": "uuid-string",   # Record's unique ID
    "operation": "update",        # Always use "update" for upsert
    "data": {...},                # Record data (JSON)
    "local_updated_at": "ISO8601" # Timestamp for conflict resolution
}
```

## Push Protocol

### Endpoint
```
POST /sync/push
Authorization: Bearer <token>
Content-Type: application/json
```

### Request Body
```json
{
    "agent_id": "your-agent-id",
    "operations": [
        {
            "operation": "update",
            "table": "episodes",
            "record_id": "550e8400-e29b-41d4-a716-446655440000",
            "local_updated_at": "2026-02-04T19:00:00Z",
            "data": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_id": "your-agent-id",
                "objective": "Test episode",
                "outcome": "Success",
                ...
            }
        }
    ]
}
```

### Response
```json
{
    "results": [
        {
            "record_id": "550e8400-e29b-41d4-a716-446655440000",
            "synced": 1,
            "conflicts": []
        }
    ],
    "pushed": 1,
    "conflicts": 0,
    "errors": []
}
```

## Field Requirements

### Required Fields

All records must include:
- `id` - UUID string
- `agent_id` - Must match the authenticated agent

### Known Fields

Only send fields that exist in the database schema. Unknown fields will cause silent failures with `synced=0`.

**Episode Fields:**
- id, agent_id, objective, outcome, lesson, timestamp
- emotional_valence, emotional_arousal, emotional_tags
- confidence, source_type, derived_from, source_episodes
- context, context_tags, subject_ids, access_grants
- (forgetting fields): times_accessed, last_accessed, is_protected, is_forgotten

**Belief Fields:**
- id, agent_id, statement, belief_type, confidence
- context, context_tags, source_type, derived_from
- is_active, supersedes, superseded_by, times_reinforced

*See schema files for complete field lists.*

## Retry Behavior (v0.2.5+)

The local client implements resilient sync:
- Failed records increment `retry_count`
- Records with >= 5 retries are moved to "dead letter" queue
- Sync continues processing other records after failures
- Dead letter records can be inspected and cleared

```python
# Get failed records
failed = storage.get_failed_sync_records(min_retries=5)

# Clear old failures (> 7 days)
cleared = storage.clear_failed_sync_records(older_than_days=7)
```

## Conflict Resolution

When the same record is modified both locally and remotely:

1. **Cloud wins** if `cloud_synced_at > local_updated_at`
2. **Local wins** if `local_updated_at > cloud_synced_at`
3. Array fields (tags, lessons) are **merged** using set union

### Array Merging

For fields like `tags`, `lessons`, `focus_areas`:
```
local:  ["a", "b", "c"]
cloud:  ["b", "c", "d"]
result: ["a", "b", "c", "d"]
```

## Common Issues

### "Database error: operation failed"

**Cause:** Unknown field in data payload.

**Fix:** Only include known schema fields. Check the table schema before pushing.

### Records stuck in queue

**Cause:** Persistent failures (e.g., invalid agent_id).

**Fix:** Check `get_failed_sync_records()` for error details. Clear or fix the problematic records.

### Insert vs Update confusion

**Cause:** Using `operation: "insert"` for records that may already exist.

**Fix:** Always use `operation: "update"` for client sync. The backend handles upsert semantics.

## CLI Commands

```bash
# Check sync status
kernle sync status

# Force push
kernle sync push

# Force pull
kernle sync pull

# View pending changes
kernle sync queue
```

## See Also

- [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md) - System overview
- [Python API](PYTHON_API.md) - Client SDK reference
