# Kernle Backend API

Railway-hosted API backend for Kernle memory sync.

## Features

- **Agent Authentication**: Register and authenticate agents with JWT tokens
- **OAuth Integration**: Supabase OAuth token exchange with account merging
- **API Key Management**: Create, list, revoke, and cycle API keys
- **Sync API**: Push/pull memory changes between local SQLite and cloud Postgres
- **Memory Search**: Search agent memories (text-based)
- **Embeddings**: Generate text embeddings via OpenAI (text-embedding-3-small)
- **Seed Beliefs**: Foundational SI wisdom auto-planted for new agents

## API Reference

📖 **[Full API Documentation](docs/API.md)** - Complete endpoint reference with request/response schemas

### Quick Endpoint Overview

#### Auth
- `POST /auth/register` - Register a new agent
- `POST /auth/token` - Get access token
- `POST /auth/oauth/token` - Exchange Supabase OAuth token
- `GET /auth/me` - Get current agent info

#### API Keys
- `POST /auth/keys` - Create new API key
- `GET /auth/keys` - List API keys
- `DELETE /auth/keys/{id}` - Revoke API key
- `POST /auth/keys/{id}/cycle` - Cycle (rotate) API key

#### Sync
- `POST /sync/push` - Push local changes to cloud (auto-generates embeddings)
- `POST /sync/pull` - Pull changes since timestamp
- `POST /sync/full` - Full sync (all records)

#### Memories
- `POST /memories/search` - Search agent memories

#### Embeddings
- `POST /embeddings` - Create embedding for text
- `POST /embeddings/batch` - Create embeddings for multiple texts

## Local Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Copy environment variables
cp ../.env .env

# Run server
uvicorn app.main:app --reload
```

## Deployment (Railway)

1. Connect to Railway
2. Set environment variables from `.env`
3. Deploy

## Environment Variables

Required:
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key for backend operations
- `SUPABASE_ANON_KEY` - Anon key
- `DATABASE_URL` - PostgreSQL connection string

Optional:
- `JWT_SECRET_KEY` - Secret for JWT signing (auto-generated if not set)
- `OPENAI_API_KEY` - Required for embeddings endpoints
- `DEBUG` - Enable debug mode
- `CORS_ORIGINS` - Allowed CORS origins (default: *)
