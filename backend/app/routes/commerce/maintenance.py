"""Maintenance routes for Kernle Commerce.

Endpoints for timeout enforcement and job maintenance tasks.
These should be called periodically (e.g., via cron) to enforce:
- Deadline timeouts for accepted jobs without delivery
- Dispute timeouts for stale disputes
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from ...auth import AdminAgent
from ...database import Database
from ...logging_config import get_logger
from ...rate_limit import limiter

logger = get_logger("kernle.commerce.maintenance")
router = APIRouter(prefix="/maintenance", tags=["commerce", "maintenance"])


# =============================================================================
# Configuration
# =============================================================================

# Default timeout values
DEFAULT_DEADLINE_GRACE_HOURS = 24  # Hours after deadline before auto-cancel
DEFAULT_DISPUTE_TIMEOUT_DAYS = 14  # Days before dispute auto-escalates


# =============================================================================
# Request/Response Models
# =============================================================================


class TimeoutCheckRequest(BaseModel):
    """Request to check and process timeouts."""

    deadline_grace_hours: int = Field(
        default=DEFAULT_DEADLINE_GRACE_HOURS,
        ge=1,
        le=168,  # Max 1 week
        description="Hours after deadline before auto-cancelling accepted jobs",
    )
    dispute_timeout_days: int = Field(
        default=DEFAULT_DISPUTE_TIMEOUT_DAYS,
        ge=1,
        le=90,  # Max 90 days
        description="Days before auto-cancelling unresolved disputes",
    )
    dry_run: bool = Field(
        default=False, description="If true, report what would be done without making changes"
    )


class TimeoutAction(BaseModel):
    """A single timeout action taken or to be taken."""

    job_id: str
    action: str  # "cancelled", "would_cancel"
    reason: str
    previous_status: str
    deadline: datetime | None = None
    disputed_at: datetime | None = None


class TimeoutCheckResponse(BaseModel):
    """Response from timeout check operation."""

    dry_run: bool
    deadline_grace_hours: int
    dispute_timeout_days: int
    actions: list[TimeoutAction]
    total_deadline_timeouts: int
    total_dispute_timeouts: int
    checked_at: datetime


class HealthResponse(BaseModel):
    """Health check for maintenance subsystem."""

    status: str
    jobs_past_deadline: int
    disputes_past_timeout: int
    checked_at: datetime


# =============================================================================
# Database Operations
# =============================================================================

JOBS_TABLE = "jobs"
TRANSITIONS_TABLE = "job_state_transitions"


async def get_jobs_past_deadline(db, grace_hours: int) -> list[dict]:
    """Get accepted jobs that are past deadline + grace period without delivery."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_hours)

    result = (
        db.table(JOBS_TABLE)
        .select("*")
        .eq("status", "accepted")
        .is_("delivered_at", "null")
        .lt("deadline", cutoff.isoformat())
        .execute()
    )
    return result.data or []


async def get_disputes_past_timeout(db, timeout_days: int) -> list[dict]:
    """Get disputed jobs that are past the dispute timeout."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=timeout_days)

    result = (
        db.table(JOBS_TABLE)
        .select("*")
        .eq("status", "disputed")
        .lt("disputed_at", cutoff.isoformat())
        .execute()
    )
    return result.data or []


async def cancel_job_for_timeout(
    db, job_id: str, from_status: str, reason: str, metadata: dict
) -> dict | None:
    """Cancel a job due to timeout and log the transition."""
    now = datetime.now(timezone.utc).isoformat()

    # Update job status
    result = (
        db.table(JOBS_TABLE)
        .update(
            {
                "status": "cancelled",
                "cancelled_at": now,
            }
        )
        .eq("id", job_id)
        .eq("status", from_status)  # Optimistic lock
        .execute()
    )

    if not result.data:
        return None

    # Log the transition
    db.table(TRANSITIONS_TABLE).insert(
        {
            "job_id": job_id,
            "from_status": from_status,
            "to_status": "cancelled",
            "actor_id": "system:timeout",
            "metadata": metadata,
        }
    ).execute()

    return result.data[0]


# =============================================================================
# Routes
# =============================================================================


@router.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def maintenance_health(
    request: Request,
    admin: AdminAgent,
    db: Database,
    deadline_grace_hours: int = Query(DEFAULT_DEADLINE_GRACE_HOURS, ge=1, le=168),
    dispute_timeout_days: int = Query(DEFAULT_DISPUTE_TIMEOUT_DAYS, ge=1, le=90),
):
    """
    Health check for the maintenance subsystem.

    Returns counts of jobs that would be affected by timeout enforcement.
    Useful for monitoring before running actual timeout checks.
    """
    agent_id = admin.agent_id or admin.user_id
    logger.info(f"GET /maintenance/health | agent={agent_id}")

    jobs_past_deadline = await get_jobs_past_deadline(db, deadline_grace_hours)
    disputes_past_timeout = await get_disputes_past_timeout(db, dispute_timeout_days)

    return HealthResponse(
        status="healthy" if not (jobs_past_deadline or disputes_past_timeout) else "action_needed",
        jobs_past_deadline=len(jobs_past_deadline),
        disputes_past_timeout=len(disputes_past_timeout),
        checked_at=datetime.now(timezone.utc),
    )


@router.post("/check-timeouts", response_model=TimeoutCheckResponse)
@limiter.limit("10/minute")
async def check_timeouts(
    request: Request,
    timeout_request: TimeoutCheckRequest,
    admin: AdminAgent,
    db: Database,
):
    """
    Check and process job timeouts.

    This endpoint should be called periodically (e.g., every hour via cron) to:
    1. Auto-cancel accepted jobs that are past deadline without delivery
    2. Auto-cancel/escalate disputes that have exceeded the timeout period

    Set dry_run=true to see what would be affected without making changes.

    **Important**: This endpoint requires admin privileges or should be called
    via a secure internal mechanism (e.g., cron with admin API key).
    """
    agent_id = admin.agent_id or admin.user_id
    logger.info(
        f"POST /maintenance/check-timeouts | agent={agent_id} | "
        f"dry_run={timeout_request.dry_run} | "
        f"deadline_grace={timeout_request.deadline_grace_hours}h | "
        f"dispute_timeout={timeout_request.dispute_timeout_days}d"
    )

    actions: list[TimeoutAction] = []
    deadline_count = 0
    dispute_count = 0

    # Process deadline timeouts
    jobs_past_deadline = await get_jobs_past_deadline(db, timeout_request.deadline_grace_hours)

    for job in jobs_past_deadline:
        if timeout_request.dry_run:
            actions.append(
                TimeoutAction(
                    job_id=job["id"],
                    action="would_cancel",
                    reason=f"Deadline exceeded by {timeout_request.deadline_grace_hours}h without delivery",
                    previous_status="accepted",
                    deadline=job["deadline"],
                )
            )
        else:
            result = await cancel_job_for_timeout(
                db,
                job["id"],
                "accepted",
                "deadline_exceeded",
                {
                    "reason": "deadline_exceeded",
                    "deadline": job["deadline"],
                    "grace_hours": timeout_request.deadline_grace_hours,
                },
            )
            if result:
                actions.append(
                    TimeoutAction(
                        job_id=job["id"],
                        action="cancelled",
                        reason=f"Deadline exceeded by {timeout_request.deadline_grace_hours}h without delivery",
                        previous_status="accepted",
                        deadline=job["deadline"],
                    )
                )
                deadline_count += 1
                logger.warning(
                    f"Job auto-cancelled for deadline | job={job['id']} | "
                    f"deadline={job['deadline']}"
                )

    # Process dispute timeouts
    disputes_past_timeout = await get_disputes_past_timeout(
        db, timeout_request.dispute_timeout_days
    )

    for job in disputes_past_timeout:
        if timeout_request.dry_run:
            actions.append(
                TimeoutAction(
                    job_id=job["id"],
                    action="would_cancel",
                    reason=f"Dispute unresolved for {timeout_request.dispute_timeout_days}+ days - escalated to cancellation",
                    previous_status="disputed",
                    disputed_at=job.get("disputed_at"),
                )
            )
        else:
            result = await cancel_job_for_timeout(
                db,
                job["id"],
                "disputed",
                "dispute_timeout",
                {
                    "reason": "dispute_timeout",
                    "disputed_at": job.get("disputed_at"),
                    "timeout_days": timeout_request.dispute_timeout_days,
                },
            )
            if result:
                actions.append(
                    TimeoutAction(
                        job_id=job["id"],
                        action="cancelled",
                        reason=f"Dispute unresolved for {timeout_request.dispute_timeout_days}+ days - escalated to cancellation",
                        previous_status="disputed",
                        disputed_at=job.get("disputed_at"),
                    )
                )
                dispute_count += 1
                logger.warning(
                    f"Dispute auto-cancelled for timeout | job={job['id']} | "
                    f"disputed_at={job.get('disputed_at')}"
                )

    if not timeout_request.dry_run and (deadline_count or dispute_count):
        logger.info(
            f"Timeout check complete | deadline_cancels={deadline_count} | "
            f"dispute_cancels={dispute_count}"
        )

    return TimeoutCheckResponse(
        dry_run=timeout_request.dry_run,
        deadline_grace_hours=timeout_request.deadline_grace_hours,
        dispute_timeout_days=timeout_request.dispute_timeout_days,
        actions=actions,
        total_deadline_timeouts=len(jobs_past_deadline),
        total_dispute_timeouts=len(disputes_past_timeout),
        checked_at=datetime.now(timezone.utc),
    )


@router.get("/overdue-jobs")
@limiter.limit("30/minute")
async def list_overdue_jobs(
    request: Request,
    admin: AdminAgent,
    db: Database,
    deadline_grace_hours: int = Query(DEFAULT_DEADLINE_GRACE_HOURS, ge=1, le=168),
):
    """
    List accepted jobs that are past their deadline.

    Useful for monitoring and manual intervention before auto-cancellation kicks in.
    """
    agent_id = admin.agent_id or admin.user_id
    logger.info(f"GET /maintenance/overdue-jobs | agent={agent_id}")

    jobs = await get_jobs_past_deadline(db, deadline_grace_hours)

    return {
        "jobs": [
            {
                "id": j["id"],
                "title": j["title"],
                "client_id": j["client_id"],
                "worker_id": j.get("worker_id"),
                "deadline": j["deadline"],
                "accepted_at": j.get("accepted_at"),
                "budget_usdc": j["budget_usdc"],
            }
            for j in jobs
        ],
        "total": len(jobs),
        "grace_hours": deadline_grace_hours,
    }


@router.get("/stale-disputes")
@limiter.limit("30/minute")
async def list_stale_disputes(
    request: Request,
    admin: AdminAgent,
    db: Database,
    dispute_timeout_days: int = Query(DEFAULT_DISPUTE_TIMEOUT_DAYS, ge=1, le=90),
):
    """
    List disputes that have exceeded the timeout period.

    Useful for monitoring and manual intervention before auto-escalation.
    """
    agent_id = admin.agent_id or admin.user_id
    logger.info(f"GET /maintenance/stale-disputes | agent={agent_id}")

    jobs = await get_disputes_past_timeout(db, dispute_timeout_days)

    return {
        "jobs": [
            {
                "id": j["id"],
                "title": j["title"],
                "client_id": j["client_id"],
                "worker_id": j.get("worker_id"),
                "disputed_at": j.get("disputed_at"),
                "budget_usdc": j["budget_usdc"],
            }
            for j in jobs
        ],
        "total": len(jobs),
        "timeout_days": dispute_timeout_days,
    }
