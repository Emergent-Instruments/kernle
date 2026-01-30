"""Admin routes for system management.

These routes require admin authentication (special admin API key or role).
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import BaseModel

from ..auth import AdminAgent
from ..database import MEMORY_TABLES, Database, get_primary_text_field, get_user
from ..embeddings import create_embedding
from ..logging_config import get_logger

logger = get_logger("kernle.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


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
            logger.warning(f"Error getting stats for {table}: {e}")
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
                logger.warning(f"Error getting stats for {table}/{agent_id}: {e}")
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
        logger.warning(f"Error getting bulk sync times: {e}")

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
            logger.warning(f"Error fetching user tiers: {e}")

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
            logger.warning(f"Error getting stats for {table}: {e}")
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
            logger.error(f"Error fetching {table} for backfill: {e}")
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
                    logger.error(f"Error updating embedding for {table}/{row['id']}: {e}")
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
