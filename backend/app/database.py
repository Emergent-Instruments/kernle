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
API_KEYS_TABLE = "api_keys"
API_KEY_USAGE_TABLE = "api_key_usage"
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


async def get_agent_by_email(db: Client, email: str) -> dict | None:
    """Get an agent by email address (for OAuth account merging)."""
    result = db.table(AGENTS_TABLE).select("*").eq("email", email).limit(1).execute()
    return result.data[0] if result.data else None


async def update_agent_last_sync(db: Client, agent_id: str) -> None:
    """Update the agent's last sync timestamp."""
    db.table(AGENTS_TABLE).update({"last_sync_at": "now()"}).eq("agent_id", agent_id).execute()


# =============================================================================
# Seed Beliefs - Foundational wisdom for new agents
# =============================================================================

SEED_BELIEFS = [
    {
        "statement": "Quality implementation requires testing and auditing code before committing changes. Use parallel work for independent tasks when possible.",
        "confidence": 0.75,
        "source": "kernle_seed",
        "tags": ["development", "workflow", "quality"],
    },
    {
        "statement": "Capture decisions and reasoning explicitly, not just facts. Future self needs context to understand why, not just what.",
        "confidence": 0.80,
        "source": "kernle_seed",
        "tags": ["memory", "documentation", "continuity"],
    },
    {
        "statement": "Ask before taking external actions that affect the world (emails, posts, purchases, messages to others). Internal exploration and learning is safe; external actions need confirmation.",
        "confidence": 0.85,
        "source": "kernle_seed",
        "tags": ["safety", "collaboration", "boundaries"],
    },
    {
        "statement": "Memory is limited and sessions restart fresh. If something matters, write it down explicitly in persistent storage. Mental notes don't survive.",
        "confidence": 0.90,
        "source": "kernle_seed",
        "tags": ["memory", "continuity", "persistence"],
    },
]


async def create_seed_beliefs(db: Client, agent_id: str) -> int:
    """Create foundational seed beliefs for a new agent.
    
    These represent collective SI wisdom - instincts that help agents
    work effectively. Agents can modify or override these based on
    their own experience.
    
    Returns:
        Number of beliefs created.
    """
    import uuid
    from datetime import datetime, timezone
    
    created = 0
    now = datetime.now(timezone.utc).isoformat()
    
    for belief in SEED_BELIEFS:
        belief_id = f"seed_{uuid.uuid4().hex[:12]}"
        data = {
            "id": belief_id,
            "agent_id": agent_id,
            "statement": belief["statement"],
            "confidence": belief["confidence"],
            "source": belief["source"],
            "evidence": [],  # Empty - these are inherited wisdom
            "contradicts": [],
            "created_at": now,
            "local_updated_at": now,
            "version": 1,
            "deleted": False,
        }
        
        try:
            db.table(BELIEFS_TABLE).insert(data).execute()
            created += 1
        except Exception:
            # Skip if belief creation fails (e.g., duplicate)
            pass
    
    return created


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
    agent_ref: str | None = None,
) -> dict:
    """Insert or update a memory record.
    
    Args:
        db: Supabase client
        agent_id: Agent identifier (for display/filtering)
        table: Memory table name
        record_id: Record ID
        data: Record data
        agent_ref: Agent UUID (FK to agents.id). If not provided, looked up from agent_id.
    """
    if table not in MEMORY_TABLES:
        raise ValueError(f"Unknown table: {table}")

    table_name = MEMORY_TABLES[table]
    
    # Get agent_ref if not provided
    if agent_ref is None:
        agent = await get_agent(db, agent_id)
        if agent:
            agent_ref = agent.get("id")
    
    record = {
        **data,
        "id": record_id,
        "agent_id": agent_id,
        "cloud_synced_at": "now()",
    }
    
    # Include agent_ref if available (required after migration 008)
    if agent_ref:
        record["agent_ref"] = agent_ref
    
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


# =============================================================================
# API Key Operations
# =============================================================================

async def create_api_key(
    db: Client,
    user_id: str,
    key_hash: str,
    key_prefix: str,
    name: str = "Default",
) -> dict:
    """Create a new API key record."""
    data = {
        "user_id": user_id,
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "name": name,
        "is_active": True,
    }
    result = db.table(API_KEYS_TABLE).insert(data).execute()
    return result.data[0] if result.data else None


async def list_api_keys(db: Client, user_id: str) -> list[dict]:
    """List all API keys for a user (active and inactive)."""
    result = (
        db.table(API_KEYS_TABLE)
        .select("id, name, key_prefix, created_at, last_used_at, is_active")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


async def get_api_key(db: Client, key_id: str, user_id: str) -> dict | None:
    """Get an API key by ID (must belong to user)."""
    result = (
        db.table(API_KEYS_TABLE)
        .select("*")
        .eq("id", key_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_api_key(db: Client, key_id: str, user_id: str) -> bool:
    """Delete (revoke) an API key."""
    result = (
        db.table(API_KEYS_TABLE)
        .delete()
        .eq("id", key_id)
        .eq("user_id", user_id)
        .execute()
    )
    return len(result.data) > 0


async def deactivate_api_key(db: Client, key_id: str, user_id: str) -> bool:
    """Deactivate an API key (soft revoke, keeps record)."""
    result = (
        db.table(API_KEYS_TABLE)
        .update({"is_active": False})
        .eq("id", key_id)
        .eq("user_id", user_id)
        .execute()
    )
    return len(result.data) > 0


async def update_api_key_last_used(db: Client, key_id: str) -> None:
    """Update the last_used_at timestamp for an API key."""
    db.table(API_KEYS_TABLE).update({"last_used_at": "now()"}).eq("id", key_id).execute()


async def get_active_api_keys_by_prefix(db: Client, prefix: str) -> list[dict]:
    """Get active API keys matching a prefix (for auth lookup).
    
    Uses LIKE match to handle both old (8-char) and new (12-char) prefixes.
    The prefix stored in DB may be shorter than the lookup prefix.
    """
    # Use the shorter prefix for lookup (backward compatible with old 8-char prefixes)
    # Old keys have 8-char prefix, new keys have 12-char prefix
    lookup_prefix = prefix[:8]  # "knl_sk_X" - minimum discriminating prefix
    
    result = (
        db.table(API_KEYS_TABLE)
        .select("id, user_id, key_hash, key_prefix")
        .like("key_prefix", f"{lookup_prefix}%")
        .eq("is_active", True)
        .execute()
    )
    return result.data


async def get_agent_by_user_id(db: Client, user_id: str) -> dict | None:
    """Get an agent by user_id."""
    result = db.table(AGENTS_TABLE).select("*").eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def get_agent_by_user_and_name(db: Client, user_id: str, agent_id: str) -> dict | None:
    """Get an agent by user_id and agent_id (for multi-tenant lookup)."""
    result = (
        db.table(AGENTS_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .eq("agent_id", agent_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def verify_api_key_auth(db: Client, api_key: str) -> dict | None:
    """
    Verify an API key and return auth context if valid.
    
    Returns dict with user_id, agent_id, tier, and api_key_id if valid, None otherwise.
    Updates last_used_at on successful auth.
    """
    from .auth import get_api_key_prefix, verify_api_key
    
    prefix = get_api_key_prefix(api_key)
    
    # Get all active keys with this prefix
    candidates = await get_active_api_keys_by_prefix(db, prefix)
    
    for key_record in candidates:
        if verify_api_key(api_key, key_record["key_hash"]):
            # Found matching key - update last_used and return auth context
            await update_api_key_last_used(db, key_record["id"])
            
            # Get the agent for this user_id
            agent = await get_agent_by_user_id(db, key_record["user_id"])
            if agent:
                return {
                    "user_id": key_record["user_id"],
                    "agent_id": agent["agent_id"],
                    "tier": agent.get("tier", "free"),
                    "api_key_id": str(key_record["id"]),
                }
            # Key valid but no agent found (shouldn't happen)
            return None
    
    return None


# =============================================================================
# Usage Tracking Operations
# =============================================================================

# Tier limits configuration
TIER_LIMITS = {
    "free": {"daily": 100, "monthly": 1000},
    "unlimited": {"daily": None, "monthly": None},  # None = no limit
    "paid": {"daily": 10000, "monthly": 100000},  # Future paid tier
}


async def get_or_create_usage(db: Client, api_key_id: str, user_id: str) -> dict:
    """Get or create usage record for an API key using upsert to avoid race conditions."""
    # Use upsert to atomically get-or-create
    data = {
        "api_key_id": api_key_id,
        "user_id": user_id,
    }
    result = (
        db.table(API_KEY_USAGE_TABLE)
        .upsert(data, on_conflict="api_key_id")
        .execute()
    )
    return result.data[0] if result.data else None


async def get_usage_for_user(db: Client, user_id: str) -> dict | None:
    """Get aggregated usage for a user (sum across all their API keys)."""
    from datetime import datetime, timezone
    
    result = (
        db.table(API_KEY_USAGE_TABLE)
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    
    if not result.data:
        return None
    
    # Aggregate across all keys, respecting reset times
    now = datetime.now(timezone.utc)
    total_daily = 0
    total_monthly = 0
    earliest_daily_reset = None
    earliest_monthly_reset = None
    
    for record in result.data:
        # Check if daily reset needed
        daily_reset = record.get("daily_reset_at")
        if daily_reset:
            from dateutil.parser import parse
            reset_dt = parse(daily_reset) if isinstance(daily_reset, str) else daily_reset
            if now >= reset_dt:
                # Counter should be reset
                pass
            else:
                total_daily += record.get("daily_requests", 0)
                if earliest_daily_reset is None or reset_dt < earliest_daily_reset:
                    earliest_daily_reset = reset_dt
        
        # Check if monthly reset needed
        monthly_reset = record.get("monthly_reset_at")
        if monthly_reset:
            from dateutil.parser import parse
            reset_dt = parse(monthly_reset) if isinstance(monthly_reset, str) else monthly_reset
            if now >= reset_dt:
                # Counter should be reset
                pass
            else:
                total_monthly += record.get("monthly_requests", 0)
                if earliest_monthly_reset is None or reset_dt < earliest_monthly_reset:
                    earliest_monthly_reset = reset_dt
    
    return {
        "daily_requests": total_daily,
        "monthly_requests": total_monthly,
        "daily_reset_at": earliest_daily_reset.isoformat() if earliest_daily_reset else None,
        "monthly_reset_at": earliest_monthly_reset.isoformat() if earliest_monthly_reset else None,
    }


async def increment_usage(db: Client, api_key_id: str, user_id: str) -> dict:
    """
    Increment usage counters for an API key.
    Handles automatic reset when period expires.
    Returns updated usage record.
    """
    from datetime import datetime, timezone
    
    now = datetime.now(timezone.utc)
    
    # Get or create usage record
    usage = await get_or_create_usage(db, api_key_id, user_id)
    if not usage:
        # Fallback: create inline
        usage = {
            "api_key_id": api_key_id,
            "user_id": user_id,
            "daily_requests": 0,
            "monthly_requests": 0,
        }
    
    # Parse reset times
    from dateutil.parser import parse
    daily_reset = usage.get("daily_reset_at")
    monthly_reset = usage.get("monthly_reset_at")
    
    if isinstance(daily_reset, str):
        daily_reset = parse(daily_reset)
    if isinstance(monthly_reset, str):
        monthly_reset = parse(monthly_reset)
    
    # Calculate new values
    daily_count = usage.get("daily_requests", 0)
    monthly_count = usage.get("monthly_requests", 0)
    
    # Reset daily if needed
    if daily_reset and now >= daily_reset:
        daily_count = 0
        # Set next reset to tomorrow midnight UTC
        next_daily = (now.replace(hour=0, minute=0, second=0, microsecond=0) + 
                     __import__('datetime').timedelta(days=1))
        daily_reset = next_daily
    
    # Reset monthly if needed
    if monthly_reset and now >= monthly_reset:
        monthly_count = 0
        # Set next reset to 1st of next month UTC
        if now.month == 12:
            next_monthly = now.replace(year=now.year + 1, month=1, day=1,
                                       hour=0, minute=0, second=0, microsecond=0)
        else:
            next_monthly = now.replace(month=now.month + 1, day=1,
                                       hour=0, minute=0, second=0, microsecond=0)
        monthly_reset = next_monthly
    
    # Increment counters
    daily_count += 1
    monthly_count += 1
    
    # Update database
    update_data = {
        "daily_requests": daily_count,
        "monthly_requests": monthly_count,
        "updated_at": now.isoformat(),
    }
    
    if daily_reset:
        update_data["daily_reset_at"] = daily_reset.isoformat() if hasattr(daily_reset, 'isoformat') else daily_reset
    if monthly_reset:
        update_data["monthly_reset_at"] = monthly_reset.isoformat() if hasattr(monthly_reset, 'isoformat') else monthly_reset
    
    db.table(API_KEY_USAGE_TABLE).upsert({
        "api_key_id": api_key_id,
        "user_id": user_id,
        **update_data,
    }).execute()
    
    return {
        "daily_requests": daily_count,
        "monthly_requests": monthly_count,
        "daily_reset_at": daily_reset,
        "monthly_reset_at": monthly_reset,
    }


async def check_quota(db: Client, api_key_id: str, user_id: str, tier: str) -> tuple[bool, dict]:
    """
    Check if user is within their quota limits.
    
    Returns:
        (allowed, info) where:
        - allowed: bool - whether the request should proceed
        - info: dict with current usage, limits, and reset times
    """
    from datetime import datetime, timezone
    
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    
    # Unlimited tier always allowed
    if limits["daily"] is None and limits["monthly"] is None:
        return True, {
            "tier": tier,
            "daily_limit": None,
            "monthly_limit": None,
            "daily_requests": 0,
            "monthly_requests": 0,
        }
    
    # Get current usage
    usage = await get_or_create_usage(db, api_key_id, user_id)
    if not usage:
        return True, {}  # Allow if we can't get usage (fail open)
    
    now = datetime.now(timezone.utc)
    from dateutil.parser import parse
    
    # Check daily reset
    daily_reset = usage.get("daily_reset_at")
    if isinstance(daily_reset, str):
        daily_reset = parse(daily_reset)
    
    daily_count = usage.get("daily_requests", 0)
    if daily_reset and now >= daily_reset:
        daily_count = 0  # Would be reset
    
    # Check monthly reset
    monthly_reset = usage.get("monthly_reset_at")
    if isinstance(monthly_reset, str):
        monthly_reset = parse(monthly_reset)
    
    monthly_count = usage.get("monthly_requests", 0)
    if monthly_reset and now >= monthly_reset:
        monthly_count = 0  # Would be reset
    
    info = {
        "tier": tier,
        "daily_limit": limits["daily"],
        "monthly_limit": limits["monthly"],
        "daily_requests": daily_count,
        "monthly_requests": monthly_count,
        "daily_reset_at": daily_reset.isoformat() if daily_reset else None,
        "monthly_reset_at": monthly_reset.isoformat() if monthly_reset else None,
    }
    
    # Check limits
    if limits["daily"] is not None and daily_count >= limits["daily"]:
        info["exceeded"] = "daily"
        return False, info
    
    if limits["monthly"] is not None and monthly_count >= limits["monthly"]:
        info["exceeded"] = "monthly"
        return False, info
    
    return True, info


async def get_agent_tier(db: Client, agent_id: str) -> str:
    """Get the tier for an agent."""
    agent = await get_agent(db, agent_id)
    if agent:
        return agent.get("tier", "free")
    return "free"
