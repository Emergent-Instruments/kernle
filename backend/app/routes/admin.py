"""Admin routes for system management.

These routes require admin authentication (special admin API key or role).
"""

import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import BaseModel

from ..auth import AdminAgent
from ..database import MEMORY_TABLES, Database, get_primary_text_field, get_user
from ..embeddings import create_embedding
from ..logging_config import get_logger

logger = get_logger("kernle.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# =============================================================================
# Caching Infrastructure
# =============================================================================


class TTLCache:
    """Simple in-memory cache with TTL expiration."""

    def __init__(self, ttl_seconds: int = 30):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        """Get value if exists and not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            # Expired, remove it
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value with current timestamp."""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()


# Cache for health stats (30 second TTL)
_health_stats_cache = TTLCache(ttl_seconds=30)


# =============================================================================
# Models
# =============================================================================


class AgentSummary(BaseModel):
    """Public agent summary (no private data)."""

    agent_id: str
    user_id: str
    tier: str
    created_at: datetime | None
    last_sync_at: datetime | None
    memory_counts: dict[str, int]
    embedding_coverage: dict[str, dict]  # table -> {total, with_embedding, percent}


class AgentListResponse(BaseModel):
    """List of agent summaries."""

    agents: list[AgentSummary]
    total: int


class EmbeddingBackfillRequest(BaseModel):
    """Request to backfill embeddings for an agent."""

    agent_id: str
    tables: list[str] | None = None  # None = all tables
    limit: int = 100  # Process in batches


class EmbeddingBackfillResponse(BaseModel):
    """Response from embedding backfill."""

    agent_id: str
    processed: int
    failed: int
    tables_updated: dict[str, int]


class SystemStats(BaseModel):
    """System-wide statistics."""

    total_agents: int
    total_memories: int
    memories_with_embeddings: int
    embedding_coverage_percent: float
    by_table: dict[str, dict]


class HealthStats(BaseModel):
    """System health and detailed statistics."""

    database_status: str  # "connected" | "degraded" | "error"
    api_status: str  # "healthy" | "degraded"
    memory_distribution: dict[str, int]  # memory type -> count
    pending_syncs: int  # count where local_updated_at > cloud_synced_at
    avg_sync_lag_seconds: float  # average sync delay
    confidence_distribution: dict[str, int]  # bucketed confidence scores
    total_memories: int
    active_memories: int
    forgotten_memories: int
    protected_memories: int


# =============================================================================
# Helper Functions
# =============================================================================

# MEMORY_TABLES and get_primary_text_field imported from database.py (single source of truth)


async def _get_agent_memory_stats(db, agent_id: str) -> tuple[dict, dict]:
    """Get memory counts and embedding coverage for an agent."""
    counts = {}
    coverage = {}

    for table in MEMORY_TABLES:
        try:
            # Total count
            total_result = (
                db.table(table)
                .select("id", count="exact")
                .eq("agent_id", agent_id)
                .eq("deleted", False)
                .execute()
            )
            total = total_result.count or 0
            counts[table] = total

            # With embedding count
            if total > 0:
                with_emb_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("agent_id", agent_id)
                    .eq("deleted", False)
                    .not_.is_("embedding", "null")
                    .execute()
                )
                with_emb = with_emb_result.count or 0
            else:
                with_emb = 0

            coverage[table] = {
                "total": total,
                "with_embedding": with_emb,
                "percent": round(with_emb / total * 100, 1) if total > 0 else 100.0,
            }
        except Exception as e:
            logger.warning(f"Error getting stats for {table}: {type(e).__name__}")
            counts[table] = 0
            coverage[table] = {"total": 0, "with_embedding": 0, "percent": 100.0}

    return counts, coverage


async def _get_bulk_memory_stats(db, agent_ids: list[str]) -> dict[str, tuple[dict, dict]]:
    """Get memory counts and embedding coverage for multiple agents in batch.

    Returns dict mapping agent_id -> (counts, coverage)
    Uses count="exact" to get counts without fetching row data.
    """
    if not agent_ids:
        return {}

    # Initialize results for all agents
    results: dict[str, tuple[dict, dict]] = {}
    for agent_id in agent_ids:
        counts = {table: 0 for table in MEMORY_TABLES}
        coverage = {
            table: {"total": 0, "with_embedding": 0, "percent": 100.0} for table in MEMORY_TABLES
        }
        results[agent_id] = (counts, coverage)

    # Query each table for each agent using count="exact"
    # This makes more queries but avoids fetching all row data
    for table in MEMORY_TABLES:
        for agent_id in agent_ids:
            try:
                # Get total count using count="exact" (no data transfer)
                total_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("agent_id", agent_id)
                    .eq("deleted", False)
                    .execute()
                )
                total = total_result.count or 0

                # Get embedding count using count="exact"
                emb_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("agent_id", agent_id)
                    .eq("deleted", False)
                    .not_.is_("embedding", "null")
                    .execute()
                )
                with_emb = emb_result.count or 0

                # Update results
                counts, coverage = results[agent_id]
                counts[table] = total
                coverage[table] = {
                    "total": total,
                    "with_embedding": with_emb,
                    "percent": round(with_emb / total * 100, 1) if total > 0 else 100.0,
                }

            except Exception as e:
                logger.warning(f"Error getting stats for {table}/{agent_id}: {type(e).__name__}")
                # Results already initialized with zeros

    return results


async def _get_bulk_last_sync(db, agent_ids: list[str]) -> dict[str, str | None]:
    """Get last sync time for multiple agents in batch.

    Returns dict mapping agent_id -> last_sync_at (or None)
    """
    if not agent_ids:
        return {}

    result: dict[str, str | None] = {aid: None for aid in agent_ids}

    try:
        # Fetch all sync logs for these agents, ordered by time
        sync_result = (
            db.table("sync_log")
            .select("agent_id, synced_at")
            .in_("agent_id", agent_ids)
            .order("synced_at", desc=True)
            .execute()
        )

        # Keep only the most recent per agent
        for row in sync_result.data:
            aid = row["agent_id"]
            if result.get(aid) is None:  # Only take first (most recent)
                result[aid] = row["synced_at"]

    except Exception as e:
        logger.warning(f"Error getting bulk sync times: {type(e).__name__}")

    return result


# =============================================================================
# Routes
# =============================================================================


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    admin: AdminAgent,
    db: Database,
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
):
    """
    List all agents with summary stats (admin only).

    Returns public info only - no private memory contents.
    """
    # Validate pagination bounds
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    # Get agents
    result = (
        db.table("agents")
        .select("agent_id, user_id, created_at")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )

    if not result.data:
        return AgentListResponse(agents=[], total=0)

    # Extract agent IDs and user IDs for batch queries
    agent_ids = [row["agent_id"] for row in result.data]
    user_ids = list({row["user_id"] for row in result.data if row.get("user_id")})

    # Batch fetch user tiers from users table (authoritative source)
    user_tiers: dict[str, str] = {}
    if user_ids:
        try:
            users_result = (
                db.table("users").select("user_id, tier").in_("user_id", user_ids).execute()
            )
            for user_row in users_result.data:
                user_tiers[user_row["user_id"]] = user_row.get("tier", "free")
        except Exception as e:
            logger.warning(f"Error fetching user tiers: {type(e).__name__}")

    # Batch fetch all stats and sync times (2 queries per table + 1 for sync = ~19 total)
    # Instead of 18 queries per agent (could be 900+ for 50 agents)
    memory_stats = await _get_bulk_memory_stats(db, agent_ids)
    sync_times = await _get_bulk_last_sync(db, agent_ids)

    # Build response
    agents = []
    for row in result.data:
        agent_id = row["agent_id"]
        user_id = row.get("user_id") or "unknown"
        counts, coverage = memory_stats.get(agent_id, ({}, {}))
        last_sync = sync_times.get(agent_id)

        # Get tier from users table (not agents table)
        tier = user_tiers.get(user_id, "free") if user_id != "unknown" else "free"

        agents.append(
            AgentSummary(
                agent_id=agent_id,
                user_id=user_id,
                tier=tier,
                created_at=row.get("created_at"),
                last_sync_at=last_sync,
                memory_counts=counts,
                embedding_coverage=coverage,
            )
        )

    # Get total count
    count_result = db.table("agents").select("id", count="exact").execute()
    total = count_result.count or len(agents)

    return AgentListResponse(agents=agents, total=total)


@router.get("/stats", response_model=SystemStats)
async def system_stats(
    admin: AdminAgent,
    db: Database,
):
    """Get system-wide statistics."""
    total_agents_result = db.table("agents").select("id", count="exact").execute()
    total_agents = total_agents_result.count or 0

    total_memories = 0
    with_embeddings = 0
    by_table = {}

    for table in MEMORY_TABLES:
        try:
            total_result = (
                db.table(table).select("id", count="exact").eq("deleted", False).execute()
            )
            table_total = total_result.count or 0

            emb_result = (
                db.table(table)
                .select("id", count="exact")
                .eq("deleted", False)
                .not_.is_("embedding", "null")
                .execute()
            )
            table_emb = emb_result.count or 0

            total_memories += table_total
            with_embeddings += table_emb

            by_table[table] = {
                "total": table_total,
                "with_embedding": table_emb,
                "percent": round(table_emb / table_total * 100, 1) if table_total > 0 else 100.0,
            }
        except Exception as e:
            logger.warning(f"Error getting stats for {table}: {type(e).__name__}")
            by_table[table] = {"total": 0, "with_embedding": 0, "percent": 100.0}

    return SystemStats(
        total_agents=total_agents,
        total_memories=total_memories,
        memories_with_embeddings=with_embeddings,
        embedding_coverage_percent=(
            round(with_embeddings / total_memories * 100, 1) if total_memories > 0 else 100.0
        ),
        by_table=by_table,
    )


@router.get("/health-stats", response_model=HealthStats)
async def health_stats(
    admin: AdminAgent,
    db: Database,
):
    """
    Get system health and detailed memory statistics.

    Returns aggregate counts only - no private memory content exposed.
    Results are cached for 30 seconds to prevent DoS via repeated expensive queries.
    """
    # Check cache first
    cache_key = "health_stats"
    cached = _health_stats_cache.get(cache_key)
    if cached is not None:
        return cached

    # Check database connectivity
    database_status = "connected"
    try:
        db.table("agents").select("agent_id", count="exact").limit(1).execute()
    except Exception as e:
        logger.error(f"Database health check failed: {type(e).__name__}")
        database_status = "error"

    # API status (if we got here, API is healthy)
    api_status = "healthy"

    # Memory distribution by type
    memory_distribution: dict[str, int] = {}
    total_memories = 0
    active_memories = 0
    forgotten_memories = 0
    protected_memories = 0

    # Tables that support is_forgotten/is_protected columns
    tables_with_forgetting = ["episodes", "beliefs", "values", "goals"]

    for table in MEMORY_TABLES:
        try:
            # Total count for this table
            total_result = (
                db.table(table).select("id", count="exact").eq("deleted", False).execute()
            )
            table_total = total_result.count or 0
            memory_distribution[table] = table_total
            total_memories += table_total

            # For tables with forgetting support, count forgotten/protected
            if table in tables_with_forgetting:
                # Active (not forgotten)
                active_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("deleted", False)
                    .eq("is_forgotten", False)
                    .execute()
                )
                active_memories += active_result.count or 0

                # Forgotten
                forgotten_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("deleted", False)
                    .eq("is_forgotten", True)
                    .execute()
                )
                forgotten_memories += forgotten_result.count or 0

                # Protected
                protected_result = (
                    db.table(table)
                    .select("id", count="exact")
                    .eq("deleted", False)
                    .eq("is_protected", True)
                    .execute()
                )
                protected_memories += protected_result.count or 0
            else:
                # Tables without forgetting are counted as active
                active_memories += table_total

        except Exception as e:
            logger.warning(f"Error getting memory distribution for {table}: {type(e).__name__}")
            memory_distribution[table] = 0

    # Pending syncs: count where local_updated_at > cloud_synced_at
    pending_syncs = 0
    total_sync_lag_seconds = 0.0
    sync_lag_count = 0

    for table in MEMORY_TABLES:
        try:
            # Get records where local_updated_at > cloud_synced_at
            # Use a raw query approach by selecting records and comparing in Python
            # since Supabase doesn't support column-to-column comparison directly
            pending_result = (
                db.table(table)
                .select("local_updated_at, cloud_synced_at")
                .eq("deleted", False)
                .not_.is_("local_updated_at", "null")
                .not_.is_("cloud_synced_at", "null")
                .limit(1000)  # Cap for performance
                .execute()
            )
            for row in pending_result.data:
                local = row.get("local_updated_at")
                cloud = row.get("cloud_synced_at")
                if local and cloud:
                    # Parse timestamps and compare
                    from datetime import datetime as dt

                    try:
                        local_dt = dt.fromisoformat(local.replace("Z", "+00:00"))
                        cloud_dt = dt.fromisoformat(cloud.replace("Z", "+00:00"))
                        if local_dt > cloud_dt:
                            pending_syncs += 1
                            lag = (local_dt - cloud_dt).total_seconds()
                            total_sync_lag_seconds += lag
                            sync_lag_count += 1
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.warning(f"Error checking sync status for {table}: {type(e).__name__}")

    avg_sync_lag_seconds = total_sync_lag_seconds / sync_lag_count if sync_lag_count > 0 else 0.0

    # Confidence distribution (for tables with confidence column)
    # Buckets: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
    confidence_distribution: dict[str, int] = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }

    tables_with_confidence = ["beliefs", "episodes", "goals", "values"]

    for table in tables_with_confidence:
        try:
            # Fetch confidence values
            conf_result = (
                db.table(table)
                .select("confidence")
                .eq("deleted", False)
                .not_.is_("confidence", "null")
                .limit(5000)  # Cap for performance
                .execute()
            )
            for row in conf_result.data:
                conf = row.get("confidence")
                if conf is not None:
                    if conf < 0.2:
                        confidence_distribution["0.0-0.2"] += 1
                    elif conf < 0.4:
                        confidence_distribution["0.2-0.4"] += 1
                    elif conf < 0.6:
                        confidence_distribution["0.4-0.6"] += 1
                    elif conf < 0.8:
                        confidence_distribution["0.6-0.8"] += 1
                    else:
                        confidence_distribution["0.8-1.0"] += 1
        except Exception as e:
            logger.warning(f"Error getting confidence distribution for {table}: {type(e).__name__}")

    result = HealthStats(
        database_status=database_status,
        api_status=api_status,
        memory_distribution=memory_distribution,
        pending_syncs=pending_syncs,
        avg_sync_lag_seconds=round(avg_sync_lag_seconds, 2),
        confidence_distribution=confidence_distribution,
        total_memories=total_memories,
        active_memories=active_memories,
        forgotten_memories=forgotten_memories,
        protected_memories=protected_memories,
    )

    # Cache the result for 30 seconds
    _health_stats_cache.set(cache_key, result)

    return result


@router.post("/embeddings/backfill", response_model=EmbeddingBackfillResponse)
async def backfill_embeddings(
    request: EmbeddingBackfillRequest,
    admin: AdminAgent,
    db: Database,
):
    """
    Backfill embeddings for memories that don't have them.

    Processes up to `limit` memories across specified tables.
    Call multiple times to process large backlogs.
    """
    # Verify agent exists before processing
    agent_result = (
        db.table("agents").select("agent_id").eq("agent_id", request.agent_id).limit(1).execute()
    )
    if not agent_result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {request.agent_id} not found"
        )

    tables = request.tables or MEMORY_TABLES
    processed = 0
    failed = 0
    tables_updated = {}

    for table in tables:
        if table not in MEMORY_TABLES:
            continue

        text_field = get_primary_text_field(table)

        # Get memories without embeddings
        try:
            result = (
                db.table(table)
                .select(f"id, {text_field}")
                .eq("agent_id", request.agent_id)
                .eq("deleted", False)
                .is_("embedding", "null")
                .limit(request.limit - processed)
                .execute()
            )
        except Exception as e:
            logger.error(f"Error fetching {table} for backfill: {type(e).__name__}")
            continue

        table_updated = 0

        for row in result.data:
            if processed >= request.limit:
                break

            text = row.get(text_field, "")
            if not text:
                continue

            # Generate embedding
            embedding = await create_embedding(text)

            if embedding:
                try:
                    db.table(table).update({"embedding": embedding}).eq("id", row["id"]).execute()

                    table_updated += 1
                    processed += 1
                except Exception as e:
                    logger.error(
                        f"Error updating embedding for {table}/{row['id']}: {type(e).__name__}"
                    )
                    failed += 1
            else:
                failed += 1

        if table_updated > 0:
            tables_updated[table] = table_updated

    logger.info(
        f"Embedding backfill for {request.agent_id}: {processed} processed, {failed} failed"
    )

    return EmbeddingBackfillResponse(
        agent_id=request.agent_id,
        processed=processed,
        failed=failed,
        tables_updated=tables_updated,
    )


@router.get("/agents/{agent_id}", response_model=AgentSummary)
async def get_agent(
    agent_id: str = Path(..., min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$"),
    admin: AdminAgent = None,
    db: Database = None,
    user_id: str | None = Query(
        default=None, max_length=128, description="Filter by user_id (for multi-tenant)"
    ),
):
    """Get detailed stats for a specific agent.

    If multiple users have agents with the same agent_id, use user_id parameter
    to specify which one. Without user_id, returns the first match.
    """
    query = db.table("agents").select("agent_id, user_id, created_at").eq("agent_id", agent_id)

    if user_id:
        query = query.eq("user_id", user_id)

    result = query.limit(1).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent {agent_id} not found"
        )

    row = result.data[0]
    agent_user_id = row.get("user_id") or "unknown"
    counts, coverage = await _get_agent_memory_stats(db, agent_id)

    # Get tier from users table (authoritative source)
    tier = "free"
    if agent_user_id != "unknown":
        user = await get_user(db, agent_user_id)
        if user:
            tier = user.get("tier", "free")

    return AgentSummary(
        agent_id=row["agent_id"],
        user_id=agent_user_id,
        tier=tier,
        created_at=row.get("created_at"),
        last_sync_at=None,
        memory_counts=counts,
        embedding_coverage=coverage,
    )
