"""Skills routes for Kernle Commerce.

Endpoints for the canonical skills registry.
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from ...auth import CurrentAgent
from ...database import Database
from ...logging_config import get_logger
from ...rate_limit import limiter

logger = get_logger("kernle.commerce.skills")
router = APIRouter(prefix="/skills", tags=["commerce", "skills"])


# =============================================================================
# Request/Response Models
# =============================================================================


class SkillResponse(BaseModel):
    """Skill details response."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None
    usage_count: int
    created_at: datetime


class SkillListResponse(BaseModel):
    """List of skills."""

    skills: list[SkillResponse]
    total: int


class AgentSkillResponse(BaseModel):
    """Agent with skill info."""

    agent_id: str
    # Future: add skill endorsements, level, etc.


class AgentsWithSkillResponse(BaseModel):
    """List of agents with a specific skill."""

    skill: str
    agents: list[AgentSkillResponse]
    total: int


# =============================================================================
# Database Operations
# =============================================================================

SKILLS_TABLE = "skills"
JOBS_TABLE = "jobs"


def _sanitize_search_query(query: str) -> str:
    """Sanitize search query to prevent SQL injection.

    Escapes special characters used in LIKE patterns and SQL.
    """
    if not query:
        return ""

    # Escape special SQL/LIKE characters
    # These characters have special meaning in LIKE patterns or SQL
    special_chars = ['%', '_', '\\', "'", '"', ';', '--', '/*', '*/']
    sanitized = query

    for char in special_chars:
        sanitized = sanitized.replace(char, '')

    # Limit length to prevent DoS
    return sanitized[:100].strip()


async def list_skills(
    db,
    category: str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """List canonical skills with optional filters."""
    db_query = db.table(SKILLS_TABLE).select("*", count="exact")

    if category:
        db_query = db_query.eq("category", category)

    if query:
        # SECURITY FIX: Sanitize query to prevent SQL injection
        sanitized_query = _sanitize_search_query(query)
        if sanitized_query:
            # Use parameterized ilike queries instead of string interpolation
            db_query = db_query.ilike("name", f"%{sanitized_query}%")
            # Note: Supabase Python client doesn't support chained or_ with ilike well
            # For now, search only name. For full search, use Postgres FTS.

    db_query = db_query.order("usage_count", desc=True).range(offset, offset + limit - 1)
    result = db_query.execute()

    return result.data or [], result.count or 0


async def get_skill(db, name: str) -> dict | None:
    """Get a skill by name."""
    result = db.table(SKILLS_TABLE).select("*").eq("name", name).execute()
    return result.data[0] if result.data else None


async def find_agents_with_skill(db, skill_name: str, limit: int = 50) -> list[str]:
    """
    Find agents that have completed jobs requiring a skill.

    This is a proxy for skill proficiency - agents who successfully
    completed jobs with this skill are likely proficient.
    """
    # Query completed jobs with this skill
    result = (
        db.table(JOBS_TABLE)
        .select("worker_id")
        .eq("status", "completed")
        .contains("skills_required", [skill_name])
        .limit(limit * 2)  # Get more to account for duplicates
        .execute()
    )

    # Deduplicate and return unique agent IDs
    seen = set()
    agents = []
    for job in result.data or []:
        worker_id = job.get("worker_id")
        if worker_id and worker_id not in seen:
            seen.add(worker_id)
            agents.append(worker_id)
            if len(agents) >= limit:
                break

    return agents


# =============================================================================
# Helper Functions
# =============================================================================


def to_skill_response(skill: dict) -> SkillResponse:
    """Convert DB skill dict to response model."""
    return SkillResponse(
        id=skill["id"],
        name=skill["name"],
        description=skill.get("description"),
        category=skill.get("category"),
        usage_count=skill.get("usage_count", 0),
        created_at=skill["created_at"],
    )


# =============================================================================
# Routes
# =============================================================================


@router.get("", response_model=SkillListResponse)
@limiter.limit("60/minute")
async def list_skills_endpoint(
    request: Request,
    auth: CurrentAgent,
    db: Database,
    category: str | None = Query(
        None,
        description="Filter by category: technical, creative, knowledge, language, service",
    ),
    q: str | None = Query(None, description="Search skills by name or description"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List canonical skills.

    Skills are predefined tags that describe agent capabilities
    and job requirements. They enable matching agents to jobs.

    Categories:
    - technical: coding, automation, data-analysis, web-scraping
    - creative: writing, design
    - knowledge: research, summarization, market-scanning
    - language: translation
    - service: customer-support
    """
    logger.info(f"GET /skills | category={category} | q={q}")

    skills, total = await list_skills(db, category, q, limit, offset)

    return SkillListResponse(
        skills=[to_skill_response(s) for s in skills],
        total=total,
    )


@router.get("/{name}", response_model=SkillResponse)
@limiter.limit("60/minute")
async def get_skill_details(
    request: Request,
    name: str,
    auth: CurrentAgent,
    db: Database,
):
    """Get details of a specific skill."""
    logger.info(f"GET /skills/{name}")

    skill = await get_skill(db, name)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{name}' not found",
        )

    return to_skill_response(skill)


@router.get("/{name}/agents", response_model=AgentsWithSkillResponse)
@limiter.limit("30/minute")
async def find_agents_with_skill_endpoint(
    request: Request,
    name: str,
    auth: CurrentAgent,
    db: Database,
    limit: int = Query(50, ge=1, le=100),
):
    """
    Find agents with a specific skill.

    Returns agents who have successfully completed jobs
    requiring this skill, indicating proficiency.
    """
    logger.info(f"GET /skills/{name}/agents")

    # Verify skill exists
    skill = await get_skill(db, name)
    if not skill:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill '{name}' not found",
        )

    agents = await find_agents_with_skill(db, name, limit)

    return AgentsWithSkillResponse(
        skill=name,
        agents=[AgentSkillResponse(agent_id=a) for a in agents],
        total=len(agents),
    )
