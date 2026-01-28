# Kernle Backend API Reference

Base URL: `https://kernle-backend.up.railway.app` (production)

## Authentication

All endpoints except `/auth/register`, `/auth/token`, and `/auth/oauth/token` require authentication.

### Methods

1. **JWT Bearer Token**: Include in header: `Authorization: Bearer <token>`
2. **API Key**: Include in header: `Authorization: Bearer <api_key>` (keys start with `knl_sk_`)

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /auth/register | 5/minute |
| POST /auth/token | 5/minute |
| POST /auth/oauth/token | 10/minute |
| POST /auth/keys | 10/minute |
| POST /auth/keys/{id}/cycle | 10/minute |

---

## Auth Endpoints

### Register Agent

```
POST /auth/register
```

Register a new agent. Returns access token and secret (shown only once).

**Request Body:**
```json
{
  "agent_id": "string (1-64 chars, lowercase alphanumeric, _, -)",
  "display_name": "string | null",
  "email": "string | null"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "string",
  "secret": "string (SAVE THIS - shown only once)"
}
```

**Side Effects:**
- Creates seed beliefs for the new agent (foundational SI wisdom)

**Errors:**
- `409 Conflict`: Agent ID already exists

---

### Get Token

```
POST /auth/token
```

Get an access token for an existing agent.

**Request Body:**
```json
{
  "agent_id": "string",
  "secret": "string"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "string | null"
}
```

**Errors:**
- `401 Unauthorized`: Invalid agent ID or secret

---

### OAuth Token Exchange

```
POST /auth/oauth/token
```

Exchange a Supabase OAuth access token for a Kernle access token. Supports account merging across OAuth providers (Google, GitHub, etc.) via email matching.

**Request Body:**
```json
{
  "access_token": "string (Supabase JWT)"
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user_id": "string"
}
```

**Side Effects:**
- Creates new agent if not exists (with seed beliefs)
- Merges accounts if email matches existing agent

**Errors:**
- `401 Unauthorized`: Invalid or expired Supabase token

---

### Get Current Agent

```
GET /auth/me
```

Get information about the currently authenticated agent.

🔒 **Requires Authentication**

**Response (200):**
```json
{
  "agent_id": "string",
  "display_name": "string | null",
  "created_at": "2024-01-01T00:00:00Z",
  "last_sync_at": "2024-01-01T00:00:00Z | null",
  "user_id": "string | null"
}
```

---

## API Key Management

### Create API Key

```
POST /auth/keys
```

Create a new API key for the authenticated user.

🔒 **Requires Authentication**

**Request Body (optional):**
```json
{
  "name": "string (1-64 chars, default: 'Default')"
}
```

**Response (200):**
```json
{
  "id": "string",
  "name": "string",
  "key": "knl_sk_... (SAVE THIS - shown only once)",
  "key_prefix": "knl_sk_a1b2",
  "created_at": "2024-01-01T00:00:00Z"
}
```

---

### List API Keys

```
GET /auth/keys
```

List all API keys for the authenticated user (metadata only, no raw keys).

🔒 **Requires Authentication**

**Response (200):**
```json
{
  "keys": [
    {
      "id": "string",
      "name": "string",
      "key_prefix": "knl_sk_a...",
      "created_at": "2024-01-01T00:00:00Z",
      "last_used_at": "2024-01-01T00:00:00Z | null",
      "is_active": true
    }
  ]
}
```

---

### Revoke API Key

```
DELETE /auth/keys/{key_id}
```

Revoke (delete) an API key. The key stops working immediately.

🔒 **Requires Authentication**

**Response:** `204 No Content`

**Errors:**
- `404 Not Found`: Key doesn't exist or doesn't belong to user

---

### Cycle API Key

```
POST /auth/keys/{key_id}/cycle
```

Atomically deactivate old key and create a new one with the same name.

🔒 **Requires Authentication**

**Response (200):**
```json
{
  "old_key_id": "string",
  "new_key": {
    "id": "string",
    "name": "string",
    "key": "knl_sk_... (SAVE THIS)",
    "key_prefix": "knl_sk_x...",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

**Errors:**
- `400 Bad Request`: Cannot cycle an inactive key
- `404 Not Found`: Key not found

---

## Sync Endpoints

### Push Changes

```
POST /sync/push
```

Push local changes to the cloud. Auto-generates embeddings for text content.

🔒 **Requires Authentication**

**Request Body:**
```json
{
  "operations": [
    {
      "operation": "insert | update | delete",
      "table": "episodes | beliefs | values | goals | notes | drives | relationships | checkpoints | raw_captures | playbooks | emotional_memories",
      "record_id": "string (UUID)",
      "data": { ... } | null,
      "local_updated_at": "2024-01-01T00:00:00Z",
      "version": 1
    }
  ],
  "last_sync_at": "2024-01-01T00:00:00Z | null"
}
```

**Response (200):**
```json
{
  "synced": 5,
  "conflicts": [
    {
      "record_id": "string",
      "error": "string"
    }
  ],
  "server_time": "2024-01-01T00:00:00Z"
}
```

**Notes:**
- Embeddings are auto-generated for records with text content
- Agent's `last_sync_at` is updated on successful push

---

### Pull Changes

```
POST /sync/pull
```

Pull changes from the cloud since a given timestamp.

🔒 **Requires Authentication**

**Request Body:**
```json
{
  "since": "2024-01-01T00:00:00Z | null"
}
```

**Response (200):**
```json
{
  "operations": [
    {
      "operation": "insert | update | delete",
      "table": "string",
      "record_id": "string",
      "data": { ... } | null,
      "local_updated_at": "2024-01-01T00:00:00Z",
      "version": 1
    }
  ],
  "server_time": "2024-01-01T00:00:00Z",
  "has_more": false
}
```

**Notes:**
- `since=null` returns all records (initial sync)
- `has_more=true` indicates more than 1000 records available

---

### Full Sync

```
POST /sync/full
```

Get all records for the agent (excludes deleted records).

🔒 **Requires Authentication**

**Response (200):**
```json
{
  "operations": [
    {
      "operation": "update",
      "table": "string",
      "record_id": "string",
      "data": { ... },
      "local_updated_at": "2024-01-01T00:00:00Z",
      "version": 1
    }
  ],
  "server_time": "2024-01-01T00:00:00Z",
  "has_more": false
}
```

---

## Memory Endpoints

### Search Memories

```
POST /memories/search
```

Search agent's memories using text matching.

🔒 **Requires Authentication**

**Request Body:**
```json
{
  "query": "string",
  "limit": 10,
  "memory_types": ["episodes", "beliefs", "notes"] | null
}
```

**Response (200):**
```json
{
  "results": [
    {
      "id": "string",
      "memory_type": "episodes",
      "content": "objective | outcome text",
      "score": 1.0,
      "created_at": "2024-01-01T00:00:00Z",
      "metadata": { ... }
    }
  ],
  "query": "string",
  "total": 5
}
```

**Searchable Fields by Memory Type:**

| Type | Fields Searched |
|------|-----------------|
| episodes | objective, outcome |
| beliefs | statement |
| values | name, description |
| goals | description |
| notes | content |
| drives | name, description |
| relationships | description |
| checkpoints | current_task, context |
| raw_captures | content |
| playbooks | name, description |
| emotional_memories | trigger, response |

---

## Embedding Endpoints

### Create Embedding

```
POST /embeddings
```

Create an embedding vector for text using OpenAI's text-embedding-3-small model.

🔒 **Requires Authentication**

**Request Body:**
```json
{
  "text": "string (1-100000 chars)"
}
```

**Response (200):**
```json
{
  "embedding": [0.123, -0.456, ...],
  "dimensions": 1536,
  "model": "text-embedding-3-small"
}
```

**Errors:**
- `503 Service Unavailable`: OpenAI API key not configured

---

### Create Batch Embeddings

```
POST /embeddings/batch
```

Create embeddings for multiple texts in a single request.

🔒 **Requires Authentication**

**Request Body:**
```json
{
  "texts": ["string", "string", ...] 
}
```

*Max 100 texts per request*

**Response (200):**
```json
{
  "embeddings": [[0.123, ...], null, [0.456, ...]],
  "dimensions": 1536,
  "model": "text-embedding-3-small",
  "successful": 2,
  "failed": 1
}
```

**Notes:**
- Returns `null` for texts that failed to embed
- Returns partial results if some texts fail
- `503` only if ALL texts fail

---

## Memory Tables

The following memory types are supported:

| Table | Description |
|-------|-------------|
| `episodes` | Significant experiences with lessons |
| `beliefs` | Statements the agent holds true (with confidence) |
| `values` | Core identity and principles |
| `goals` | What the agent is working toward |
| `notes` | Decisions, insights, observations |
| `drives` | Motivations and drives |
| `relationships` | Connections with people/entities |
| `checkpoints` | State snapshots for continuity |
| `raw_captures` | Quick thoughts, scratchpad |
| `playbooks` | Behavioral patterns and procedures |
| `emotional_memories` | Trigger-response patterns |

---

## Seed Beliefs

When a new agent registers (via `/auth/register` or `/auth/oauth/token`), they automatically receive foundational "seed beliefs" - core SI (Superintelligent) wisdom principles. These provide a baseline ethical and operational framework for new agents.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Missing or invalid authentication
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: External service unavailable
