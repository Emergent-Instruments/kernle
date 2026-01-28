"""Database utilities for Supabase integration."""

from typing import Annotated

from fastapi import Depends

from supabase import Client, create_client

from .config import Settings, get_settings

_supabase_client: Client | None = None

def get_supabase_client(settings: Settings | None = None) -> Client:
    """Get cached Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        if settings is None:
            settings = get_settings()
        # Prefer new secret key, fall back to legacy service_role_key
        api_key = settings.supabase_secret_key or settings.supabase_service_role_key
        if not api_key:
            raise ValueError("Either SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY must be set")
        _supabase_client = create_client(settings.supabase_url, api_key)
    return _supabase_client


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Client:
    """FastAPI dependency for Supabase client."""
    return get_supabase_client(settings)


# Type alias for dependency injection
Database = Annotated[Client, Depends(get_db)]


# =============================================================================
# Table Names
# =============================================================================

AGENTS_TABLE = "agents"
EPISODES_TABLE = "episodes"
BELIEFS_TABLE = "beliefs"
VALUES_TABLE = "values"
GOALS_TABLE = "goals"
NOTES_TABLE = "notes"
DRIVES_TABLE = "drives"
RELATIONSHIPS_TABLE = "relationships"
CHECKPOINTS_TABLE = "checkpoints"
RAW_CAPTURES_TABLE = "raw_captures"
PLAYBOOKS_TABLE = "playbooks"
EMOTIONAL_MEMORIES_TABLE = "emotional_memories"


# =============================================================================
# Agent Operations
# =============================================================================

async def create_agent(
    db: Client,
    agent_id: str,
    secret_hash: str,
    user_id: str,
    display_name: str | None = None,
    email: str | None = None,
) -> dict:
    """Create a new agent in the database."""
    data = {
        "agent_id": agent_id,
        "secret_hash": secret_hash,
        "user_id": user_id,
        "display_name": display_name,
        "email": email,
    }
    result = db.table(AGENTS_TABLE).insert(data).execute()
    return result.data[0] if result.data else None


async def get_agent(db: Client, agent_id: str) -> dict | None:
    """Get an agent by ID."""
    result = db.table(AGENTS_TABLE).select("*").eq("agent_id", agent_id).execute()
    return result.data[0] if result.data else None


async def update_agent_last_sync(db: Client, agent_id: str) -> None:
    """Update the agent's last sync timestamp."""
    db.table(AGENTS_TABLE).update({"last_sync_at": "now()"}).eq("agent_id", agent_id).execute()


# =============================================================================
# Memory Operations
# =============================================================================

MEMORY_TABLES = {
    "episodes": EPISODES_TABLE,
    "beliefs": BELIEFS_TABLE,
    "values": VALUES_TABLE,
    "goals": GOALS_TABLE,
    "notes": NOTES_TABLE,
    "drives": DRIVES_TABLE,
    "relationships": RELATIONSHIPS_TABLE,
    "checkpoints": CHECKPOINTS_TABLE,
    "raw_captures": RAW_CAPTURES_TABLE,
    "playbooks": PLAYBOOKS_TABLE,
    "emotional_memories": EMOTIONAL_MEMORIES_TABLE,
}


async def upsert_memory(
    db: Client,
    agent_id: str,
    table: str,
    record_id: str,
    data: dict,
) -> dict:
    """Insert or update a memory record."""
    if table not in MEMORY_TABLES:
        raise ValueError(f"Unknown table: {table}")

    table_name = MEMORY_TABLES[table]
    record = {
        **data,
        "id": record_id,
        "agent_id": agent_id,
        "cloud_synced_at": "now()",
    }
    result = db.table(table_name).upsert(record).execute()
    return result.data[0] if result.data else None


async def delete_memory(
    db: Client,
    agent_id: str,
    table: str,
    record_id: str,
) -> bool:
    """Soft-delete a memory record."""
    if table not in MEMORY_TABLES:
        raise ValueError(f"Unknown table: {table}")

    table_name = MEMORY_TABLES[table]
    result = db.table(table_name).update({"deleted": True}).eq("id", record_id).eq("agent_id", agent_id).execute()
    return len(result.data) > 0


async def get_changes_since(
    db: Client,
    agent_id: str,
    since: str | None,
    limit: int = 1000,
) -> list[dict]:
    """Get all changes for an agent since a given timestamp."""
    changes = []

    for table_key, table_name in MEMORY_TABLES.items():
        query = db.table(table_name).select("*").eq("agent_id", agent_id)
        if since:
            query = query.gt("cloud_synced_at", since)
        query = query.limit(limit)
        result = query.execute()

        for record in result.data:
            changes.append({
                "table": table_key,
                "record_id": record["id"],
                "data": record,
                "operation": "delete" if record.get("deleted") else "update",
            })

    return changes
