"""Memory search routes."""

import re

from fastapi import APIRouter

from ..auth import CurrentAgent
from ..database import MEMORY_TABLES, Database
from ..models import MemorySearchRequest, MemorySearchResponse, MemorySearchResult


def escape_like(query: str) -> str:
    """Escape SQL LIKE special characters to prevent injection."""
    # Escape backslash first, then %, then _
    return re.sub(r'([%_\\])', r'\\\1', query)

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: MemorySearchRequest,
    agent_id: CurrentAgent,
    db: Database,
):
    """
    Search agent's memories using text matching.

    Note: Full semantic search requires pgvector setup.
    This endpoint provides basic text search as a starting point.
    """
    results = []

    # Determine which tables to search
    tables_to_search = (
        [t for t in request.memory_types if t in MEMORY_TABLES]
        if request.memory_types
        else list(MEMORY_TABLES.keys())
    )

    # Search each table with text matching
    for table_key in tables_to_search:
        table_name = MEMORY_TABLES[table_key]

        # Use Supabase's textSearch or ilike for basic matching
        # Note: For production, implement pgvector similarity search
        try:
            # Try to search common content fields
            query = (
                db.table(table_name)
                .select("*")
                .eq("agent_id", agent_id)
                .eq("deleted", False)
                .limit(request.limit)
            )

            # Add text filter based on table type
            content_fields = _get_content_fields(table_key)
            if content_fields:
                # Use ilike for case-insensitive search on first content field
                # Escape LIKE special characters to prevent SQL injection
                safe_query = escape_like(request.query)
                query = query.ilike(content_fields[0], f"%{safe_query}%")

            result = query.execute()

            for record in result.data:
                content = _extract_content(record, content_fields)
                results.append(
                    MemorySearchResult(
                        id=record["id"],
                        memory_type=table_key,
                        content=content,
                        score=1.0,  # Basic search doesn't have scores
                        created_at=record.get("created_at"),
                        metadata={
                            k: v for k, v in record.items()
                            if k not in ["id", "agent_id", "created_at", "deleted", "embedding"]
                        },
                    )
                )
        except Exception:
            # Skip tables that don't have expected columns
            continue

    # Sort by created_at descending and limit
    results.sort(key=lambda x: x.created_at or "", reverse=True)
    results = results[:request.limit]

    return MemorySearchResponse(
        results=results,
        query=request.query,
        total=len(results),
    )


def _get_content_fields(table_key: str) -> list[str]:
    """Get the content fields to search for each table type."""
    field_map = {
        "episodes": ["objective", "outcome"],
        "beliefs": ["statement"],
        "values": ["name", "description"],
        "goals": ["description"],
        "notes": ["content"],
        "drives": ["name", "description"],
        "relationships": ["description"],
        "checkpoints": ["current_task", "context"],
        "raw_captures": ["content"],
        "playbooks": ["name", "description"],
        "emotional_memories": ["trigger", "response"],
    }
    return field_map.get(table_key, ["content"])


def _extract_content(record: dict, fields: list[str]) -> str:
    """Extract content from a record for display."""
    parts = []
    for field in fields:
        if field in record and record[field]:
            parts.append(str(record[field]))
    return " | ".join(parts) if parts else str(record)
