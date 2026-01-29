"""Memory search routes with semantic search support."""

import asyncio
import re

from fastapi import APIRouter

from ..auth import CurrentAgent
from ..database import MEMORY_TABLES, Database, get_text_fields
from ..embeddings import create_embedding
from ..logging_config import get_logger
from ..models import MemorySearchRequest, MemorySearchResponse, MemorySearchResult

logger = get_logger("kernle.memories")


def escape_like(query: str) -> str:
    """Escape SQL LIKE special characters to prevent injection."""
    return re.sub(r"([%_\\])", r"\\\1", query)


router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Search agent's memories using semantic similarity (pgvector) with text fallback.

    Semantic search creates an embedding of the query and finds memories with
    similar embeddings using cosine similarity. Falls back to text matching
    if embeddings are unavailable.
    """
    # Try semantic search first
    query_embedding = await create_embedding(request.query)

    if query_embedding:
        results = await _semantic_search(
            db, auth.agent_id, query_embedding, request.limit, request.memory_types
        )
        if results:
            return MemorySearchResponse(
                results=results,
                query=request.query,
                total=len(results),
            )

    # Fall back to text search
    results = await _text_search(
        db, auth.agent_id, request.query, request.limit, request.memory_types
    )

    return MemorySearchResponse(
        results=results,
        query=request.query,
        total=len(results),
    )


async def _semantic_search(
    db: Database,
    agent_id: str,
    query_embedding: list[float],
    limit: int,
    memory_types: list[str] | None,
) -> list[MemorySearchResult]:
    """Search using pgvector semantic similarity."""
    import logging

    logger = logging.getLogger("kernle.memories")

    try:
        logger.info(
            f"Semantic search: agent={agent_id}, embedding_dims={len(query_embedding)}, limit={limit}"
        )

        # Convert embedding list to pgvector string format
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Call the RPC function
        result = db.rpc(
            "search_memories_semantic",
            {
                "query_embedding": embedding_str,
                "p_agent_id": agent_id,
                "p_limit": limit,
                "p_memory_types": memory_types,
            },
        ).execute()

        logger.info(f"Semantic search returned {len(result.data)} results")

        return [
            MemorySearchResult(
                id=row["id"],
                memory_type=row["memory_type"],
                content=row["content"],
                score=row["score"],
                created_at=row["created_at"],
                metadata=row["metadata"],
            )
            for row in result.data
        ]
    except Exception as e:
        # RPC not available or failed - fall back to text search
        logger.error(f"Semantic search failed, falling back to text: {e}")
        return []


async def _text_search(
    db: Database,
    agent_id: str,
    query: str,
    limit: int,
    memory_types: list[str] | None,
) -> list[MemorySearchResult]:
    """Fall back to text-based search.

    Searches all tables in parallel using asyncio.gather().
    """

    tables_to_search = (
        [t for t in memory_types if t in MEMORY_TABLES]
        if memory_types
        else list(MEMORY_TABLES.keys())
    )

    safe_query = escape_like(query)

    async def search_table(table_key: str) -> list[MemorySearchResult]:
        """Search a single table for matching records."""
        table_name = MEMORY_TABLES[table_key]
        content_fields = get_text_fields(table_key)

        def _query():
            db_query = (
                db.table(table_name)
                .select("*")
                .eq("agent_id", agent_id)
                .eq("deleted", False)
                .limit(limit)
            )
            if content_fields:
                db_query = db_query.ilike(content_fields[0], f"%{safe_query}%")
            return db_query.execute()

        try:
            result = await asyncio.to_thread(_query)
            return [
                MemorySearchResult(
                    id=record["id"],
                    memory_type=table_key,
                    content=_extract_content(record, content_fields),
                    score=0.5,  # Text search has no similarity score
                    created_at=record.get("created_at"),
                    metadata={
                        k: v
                        for k, v in record.items()
                        if k not in ["id", "agent_id", "created_at", "deleted", "embedding"]
                    },
                )
                for record in result.data
            ]
        except Exception as e:
            # Log but don't fail - partial results are better than none
            logger.warning(f"Text search failed for table {table_key}: {e}")
            return []

    # Search all tables in parallel
    table_results = await asyncio.gather(
        *[search_table(table_key) for table_key in tables_to_search]
    )

    # Flatten and sort by created_at
    results = [r for table_list in table_results for r in table_list]
    results.sort(key=lambda x: x.created_at or "", reverse=True)
    return results[:limit]


def _extract_content(record: dict, content_fields: list[str]) -> str:
    """Extract content from record based on available fields."""
    parts = []
    for field in content_fields:
        if field in record and record[field]:
            parts.append(str(record[field]))
    return " | ".join(parts) if parts else str(record.get("id", ""))
