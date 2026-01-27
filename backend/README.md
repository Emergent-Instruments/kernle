# Kernle Backend API

Railway-hosted API backend for Kernle memory sync.

## Features

- **Agent Authentication**: Register and authenticate agents with JWT tokens
- **Sync API**: Push/pull memory changes between local SQLite and cloud Postgres
- **Memory Search**: Search agent memories (text-based, pgvector coming soon)

## Endpoints

### Auth
- `POST /auth/register` - Register a new agent
- `POST /auth/token` - Get access token
- `GET /auth/me` - Get current agent info

### Sync
- `POST /sync/push` - Push local changes to cloud
- `POST /sync/pull` - Pull changes since timestamp
- `POST /sync/full` - Full sync (all records)

### Memories
- `POST /memories/search` - Search agent memories

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
- `DEBUG` - Enable debug mode
- `CORS_ORIGINS` - Allowed CORS origins (default: *)
