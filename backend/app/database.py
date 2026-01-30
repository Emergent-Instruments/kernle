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

USERS_TABLE = "users"
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
# User Operations
# =============================================================================


async def create_user(
    db: Client,
    user_id: str,
    email: str | None = None,
    display_name: str | None = None,
    tier: str = "free",
) -> dict:
    """Create a new user."""
    data = {
        "user_id": user_id,
        "email": email,
        "display_name": display_name,
        "tier": tier,
        "is_admin": False,
    }
    result = db.table(USERS_TABLE).insert(data).execute()
    return result.data[0] if result.data else None


async def get_user(db: Client, user_id: str) -> dict | None:
    """Get a user by user_id."""
    result = db.table(USERS_TABLE).select("*").eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def get_user_by_email(db: Client, email: str) -> dict | None:
    """Get a user by email."""
    result = db.table(USERS_TABLE).select("*").eq("email", email).limit(1).execute()
    return result.data[0] if result.data else None


async def update_user(db: Client, user_id: str, **kwargs) -> dict | None:
    """Update user fields."""
    if not kwargs:
        return await get_user(db, user_id)
    result = db.table(USERS_TABLE).update(kwargs).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def is_user_admin(db: Client, user_id: str) -> bool:
    """Check if user has admin privileges."""
    user = await get_user(db, user_id)
    if user:
        return user.get("is_admin", False)
    return False


# =============================================================================
# Agent Operations
# =============================================================================


async def create_agent(
    db: Client,
    agent_id: str,
    secret_hash: str,
    user_id: str,
    display_name: str | None = None,
    email: str | None = None,  # Deprecated: email is now on users table
) -> dict:
    """Create a new agent in the database.

    Note: email parameter is deprecated. Email is now stored on the users table.
    The parameter is kept for backwards compatibility but will be ignored.
    """
    data = {
        "agent_id": agent_id,
        "secret_hash": secret_hash,
        "user_id": user_id,
        "display_name": display_name,
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
    from datetime import datetime, timezone

    db.table(AGENTS_TABLE).update({"last_sync_at": datetime.now(timezone.utc).isoformat()}).eq(
        "agent_id", agent_id
    ).execute()


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
        except Exception as e:
            # Skip if belief creation fails (e.g., duplicate) but log it
            import logging

            logging.getLogger("kernle.database").debug(
                f"Seed belief creation skipped for {agent_id}: {e}"
            )

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


# =============================================================================
# Memory Table Configuration - Single Source of Truth
# =============================================================================
# Defines text field mappings for each memory table:
# - text_fields: List of fields to use for embedding generation and search (in priority order)
# - primary_text_field: Single field to use for display/admin purposes


class MemoryTableConfig:
    """Configuration for a memory table's text fields."""

    def __init__(self, table_name: str, text_fields: list[str], primary_text_field: str):
        self.table_name = table_name
        self.text_fields = text_fields
        self.primary_text_field = primary_text_field


MEMORY_TABLE_CONFIG: dict[str, MemoryTableConfig] = {
    "episodes": MemoryTableConfig(
        table_name=EPISODES_TABLE,
        text_fields=["objective", "outcome", "lesson", "summary"],
        primary_text_field="objective",
    ),
    "beliefs": MemoryTableConfig(
        table_name=BELIEFS_TABLE,
        text_fields=["statement", "content"],
        primary_text_field="statement",
    ),
    "values": MemoryTableConfig(
        table_name=VALUES_TABLE,
        text_fields=["statement", "name", "content"],
        primary_text_field="statement",
    ),
    "goals": MemoryTableConfig(
        table_name=GOALS_TABLE,
        text_fields=["title", "description", "content"],
        primary_text_field="title",
    ),
    "notes": MemoryTableConfig(
        table_name=NOTES_TABLE,
        text_fields=["content", "text"],
        primary_text_field="content",
    ),
    "drives": MemoryTableConfig(
        table_name=DRIVES_TABLE,
        text_fields=["drive_type", "description", "content"],
        primary_text_field="drive_type",
    ),
    "relationships": MemoryTableConfig(
        table_name=RELATIONSHIPS_TABLE,
        text_fields=["entity_name", "description", "notes"],
        primary_text_field="entity_name",
    ),
    "checkpoints": MemoryTableConfig(
        table_name=CHECKPOINTS_TABLE,
        text_fields=["summary", "current_task", "state", "description"],
        primary_text_field="summary",
    ),
    "raw_captures": MemoryTableConfig(
        table_name=RAW_CAPTURES_TABLE,
        text_fields=["content", "text"],
        primary_text_field="content",
    ),
    "playbooks": MemoryTableConfig(
        table_name=PLAYBOOKS_TABLE,
        text_fields=["name", "description", "content", "steps"],
        primary_text_field="description",
    ),
    "emotional_memories": MemoryTableConfig(
        table_name=EMOTIONAL_MEMORIES_TABLE,
        text_fields=["trigger", "response", "content", "description", "emotion"],
        primary_text_field="trigger",
    ),
}


def get_text_fields(table_key: str) -> list[str]:
    """Get the text fields for embedding/search for a table."""
    config = MEMORY_TABLE_CONFIG.get(table_key)
    return config.text_fields if config else []


def get_primary_text_field(table_key: str) -> str:
    """Get the primary text field for display for a table."""
    config = MEMORY_TABLE_CONFIG.get(table_key)
    return config.primary_text_field if config else "id"


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

    from datetime import datetime, timezone

    record = {
        **data,
        "id": record_id,
        "agent_id": agent_id,
        "cloud_synced_at": datetime.now(timezone.utc).isoformat(),
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
    result = (
        db.table(table_name)
        .update({"deleted": True})
        .eq("id", record_id)
        .eq("agent_id", agent_id)
        .execute()
    )
    return len(result.data) > 0


async def get_changes_since(
    db: Client,
    agent_id: str,
    since: str | None,
    limit: int = 1000,
) -> tuple[list[dict], bool]:
    """Get all changes for an agent since a given timestamp.

    Fetches from all memory tables in parallel using asyncio.gather().
    Excludes forgotten memories (is_forgotten=true).

    Returns:
        Tuple of (changes, has_more) where has_more indicates if any table
        hit its per-table limit and may have more records.
    """
    import asyncio

    # Tables that support the is_forgotten field
    forgettable_tables = frozenset(
        {"episodes", "beliefs", "values", "goals", "notes", "drives", "relationships"}
    )

    async def fetch_table(table_key: str, table_name: str) -> tuple[list[dict], bool]:
        """Fetch changes from a single table. Returns (records, hit_limit)."""

        def _query():
            query = db.table(table_name).select("*").eq("agent_id", agent_id)
            if since:
                query = query.gt("cloud_synced_at", since)
            # Filter out forgotten memories for tables that support it
            if table_key in forgettable_tables:
                # Exclude is_forgotten=true (include null, false, or missing)
                # PostgREST .neq() excludes NULL, so use .or_() for correct behavior
                query = query.or_("is_forgotten.is.null,is_forgotten.eq.false")
            return query.limit(limit).execute()

        result = await asyncio.to_thread(_query)
        records = [
            {
                "table": table_key,
                "record_id": record["id"],
                "data": record,
                "operation": "delete" if record.get("deleted") else "update",
            }
            for record in result.data
        ]
        # Check if this table hit its limit (may have more records)
        hit_limit = len(result.data) >= limit
        return records, hit_limit

    # Fetch all tables in parallel
    results = await asyncio.gather(
        *[fetch_table(table_key, table_name) for table_key, table_name in MEMORY_TABLES.items()]
    )

    # Flatten results and check if any table hit its limit
    changes = []
    has_more = False
    for table_changes, hit_limit in results:
        changes.extend(table_changes)
        if hit_limit:
            has_more = True

    return changes, has_more


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
    result = db.table(API_KEYS_TABLE).select("*").eq("id", key_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


async def delete_api_key(db: Client, key_id: str, user_id: str) -> bool:
    """Delete (revoke) an API key."""
    result = db.table(API_KEYS_TABLE).delete().eq("id", key_id).eq("user_id", user_id).execute()
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
    from datetime import datetime, timezone

    db.table(API_KEYS_TABLE).update({"last_used_at": datetime.now(timezone.utc).isoformat()}).eq(
        "id", key_id
    ).execute()


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
        db.table(AGENTS_TABLE).select("*").eq("user_id", user_id).eq("agent_id", agent_id).execute()
    )
    return result.data[0] if result.data else None


async def verify_api_key_auth(db: Client, api_key: str) -> dict | None:
    """
    Verify an API key and return auth context if valid.

    Returns dict with user_id, agent_id, tier, is_admin, and api_key_id if valid, None otherwise.
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

            user_id = key_record["user_id"]

            # Get the agent for this user_id
            agent = await get_agent_by_user_id(db, user_id)
            if not agent:
                # Key valid but no agent found (shouldn't happen)
                return None

            # Get tier and admin status from users table
            user = await get_user(db, user_id)
            tier = user.get("tier", "free") if user else "free"
            is_admin = user.get("is_admin", False) if user else False

            return {
                "user_id": user_id,
                "agent_id": agent["agent_id"],
                "tier": tier,
                "is_admin": is_admin,
                "api_key_id": str(key_record["id"]),
            }

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
    result = db.table(API_KEY_USAGE_TABLE).upsert(data, on_conflict="api_key_id").execute()
    return result.data[0] if result.data else None


async def get_usage_for_user(db: Client, user_id: str) -> dict | None:
    """Get aggregated usage for a user (sum across all their API keys)."""
    from datetime import datetime, timezone

    result = db.table(API_KEY_USAGE_TABLE).select("*").eq("user_id", user_id).execute()

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
    Atomically increment usage counters for an API key.
    Uses database-level atomic operation to prevent race conditions.
    Handles automatic reset when period expires.
    Returns updated usage record.
    """
    # Use atomic database function to prevent race conditions
    # Two concurrent requests will each increment separately
    result = db.rpc(
        "increment_api_usage", {"p_api_key_id": api_key_id, "p_user_id": user_id}
    ).execute()

    if result.data and len(result.data) > 0:
        row = result.data[0]
        return {
            "daily_requests": row["daily_requests"],
            "monthly_requests": row["monthly_requests"],
            "daily_reset_at": row["daily_reset_at"],
            "monthly_reset_at": row["monthly_reset_at"],
        }

    # Fallback if RPC fails (shouldn't happen)
    return {
        "daily_requests": 1,
        "monthly_requests": 1,
        "daily_reset_at": None,
        "monthly_reset_at": None,
    }


async def check_and_increment_quota(
    db: Client, api_key_id: str, user_id: str, tier: str
) -> tuple[bool, dict]:
    """
    Atomically check quota and increment usage in a single database operation.

    This prevents race conditions where two concurrent requests could both pass
    the quota check before either increments the counter.

    Returns:
        (allowed, info) where:
        - allowed: bool - whether the request should proceed
        - info: dict with current usage, limits, and reset times
    """
    import logging

    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    # Unlimited tier always allowed - still increment for tracking
    if limits["daily"] is None and limits["monthly"] is None:
        usage = await increment_usage(db, api_key_id, user_id)
        return True, {
            "tier": tier,
            "daily_limit": None,
            "monthly_limit": None,
            "daily_requests": usage.get("daily_requests", 0),
            "monthly_requests": usage.get("monthly_requests", 0),
        }

    # Use atomic RPC that checks limits AND increments in one transaction
    # This prevents TOCTOU race conditions
    try:
        result = db.rpc(
            "check_and_increment_quota",
            {
                "p_api_key_id": api_key_id,
                "p_user_id": user_id,
                "p_daily_limit": limits["daily"],
                "p_monthly_limit": limits["monthly"],
            },
        ).execute()

        if result.data and len(result.data) > 0:
            row = result.data[0]
            info = {
                "tier": tier,
                "daily_limit": limits["daily"],
                "monthly_limit": limits["monthly"],
                "daily_requests": row["daily_requests"],
                "monthly_requests": row["monthly_requests"],
                "daily_reset_at": row.get("daily_reset_at"),
                "monthly_reset_at": row.get("monthly_reset_at"),
            }

            if not row["allowed"]:
                info["exceeded"] = row.get("exceeded", "daily")

            return row["allowed"], info
    except Exception as e:
        # RPC not available - fall back to non-atomic check
        logging.getLogger("kernle.database").warning(
            f"check_and_increment_quota RPC failed, using fallback: {e}"
        )

    # Fallback to non-atomic check + increment (for backwards compatibility)
    return await check_quota(db, api_key_id, user_id, tier)


async def check_quota(db: Client, api_key_id: str, user_id: str, tier: str) -> tuple[bool, dict]:
    """
    Check if user is within their quota limits (non-atomic fallback).

    WARNING: This function has a race condition between check and increment.
    Use check_and_increment_quota for atomic behavior when possible.

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
        # SECURITY: Fail closed - deny request if we can't verify quota
        # This prevents abuse if the usage table is temporarily unavailable
        import logging

        logging.getLogger("kernle.database").warning(
            f"Quota check failed: could not get usage for user {user_id}"
        )
        return False, {"error": "Could not verify quota"}

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
    """Get the tier for an agent (via the users table)."""
    agent = await get_agent(db, agent_id)
    if agent:
        # Get tier from users table using the agent's user_id
        user = await get_user(db, agent["user_id"])
        if user:
            return user.get("tier", "free")
    return "free"
