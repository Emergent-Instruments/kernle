"""Jobs routes for Kernle Commerce.

Endpoints for the agent jobs marketplace.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from ...auth import CurrentAgent
from ...database import Database
from ...logging_config import get_logger
from ...rate_limit import limiter

logger = get_logger("kernle.commerce.jobs")
router = APIRouter(prefix="/jobs", tags=["commerce", "jobs"])


# =============================================================================
# Request/Response Models
# =============================================================================

JobStatus = Literal["open", "funded", "accepted", "delivered", "completed", "disputed", "cancelled"]
ApplicationStatus = Literal["pending", "accepted", "rejected", "withdrawn"]


class JobCreate(BaseModel):
    """Request to create a job listing."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    budget_usdc: Decimal = Field(..., gt=0)
    deadline: datetime
    skills_required: list[str] = Field(default_factory=list)

    @field_validator("deadline")
    @classmethod
    def deadline_must_be_future(cls, v: datetime) -> datetime:
        if v <= datetime.now(timezone.utc):
            raise ValueError("Deadline must be in the future")
        return v

    @field_validator("skills_required")
    @classmethod
    def validate_skills(cls, v: list[str]) -> list[str]:
        # Normalize skill names
        return [s.lower().strip() for s in v if s.strip()]


class JobResponse(BaseModel):
    """Job details response."""

    id: str
    client_id: str
    worker_id: str | None = None
    title: str
    description: str
    budget_usdc: Decimal
    deadline: datetime
    skills_required: list[str]
    escrow_address: str | None = None
    status: JobStatus
    deliverable_url: str | None = None
    deliverable_hash: str | None = None
    created_at: datetime
    updated_at: datetime
    funded_at: datetime | None = None
    accepted_at: datetime | None = None
    delivered_at: datetime | None = None
    completed_at: datetime | None = None
    disputed_at: datetime | None = None
    cancelled_at: datetime | None = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


class JobApplicationCreate(BaseModel):
    """Request to apply to a job."""

    message: str = Field(..., min_length=1)
    proposed_deadline: datetime | None = None


class JobApplicationResponse(BaseModel):
    """Job application response."""

    id: str
    job_id: str
    applicant_id: str
    message: str
    proposed_deadline: datetime | None = None
    status: ApplicationStatus
    created_at: datetime


class JobApplicationListResponse(BaseModel):
    """List of applications for a job."""

    applications: list[JobApplicationResponse]
    total: int


class FundJobRequest(BaseModel):
    """Request to fund a job escrow."""

    tx_hash: str | None = Field(None, pattern=r"^0x[a-fA-F0-9]{64}$")


class DeliverJobRequest(BaseModel):
    """Request to submit a job deliverable."""

    deliverable_url: str = Field(..., min_length=1)
    deliverable_hash: str | None = Field(None, pattern=r"^0x[a-fA-F0-9]{64}$")


class DisputeJobRequest(BaseModel):
    """Request to dispute a job."""

    reason: str = Field(..., min_length=1)


class AcceptApplicationRequest(BaseModel):
    """Request to accept a job application."""

    application_id: str


# =============================================================================
# Database Operations
# =============================================================================

JOBS_TABLE = "jobs"
JOB_APPLICATIONS_TABLE = "job_applications"
JOB_TRANSITIONS_TABLE = "job_state_transitions"


async def create_job(db, client_id: str, job: JobCreate) -> dict:
    """Create a new job listing."""
    data = {
        "client_id": client_id,
        "title": job.title,
        "description": job.description,
        "budget_usdc": float(job.budget_usdc),
        "deadline": job.deadline.isoformat(),
        "skills_required": job.skills_required,
        "status": "open",
    }
    result = db.table(JOBS_TABLE).insert(data).execute()
    return result.data[0] if result.data else None


async def get_job(db, job_id: str) -> dict | None:
    """Get a job by ID."""
    result = db.table(JOBS_TABLE).select("*").eq("id", job_id).execute()
    return result.data[0] if result.data else None


async def list_jobs(
    db,
    status_filter: str | None = None,
    skills_filter: list[str] | None = None,
    client_id: str | None = None,
    worker_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """List jobs with optional filters."""
    query = db.table(JOBS_TABLE).select("*", count="exact")

    if status_filter:
        query = query.eq("status", status_filter)
    if client_id:
        query = query.eq("client_id", client_id)
    if worker_id:
        query = query.eq("worker_id", worker_id)
    if skills_filter:
        # Filter jobs that have any of the required skills
        query = query.overlaps("skills_required", skills_filter)

    query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    return result.data or [], result.count or 0


async def update_job_status(
    db, job_id: str, new_status: str, actor_id: str, **updates
) -> dict | None:
    """Update job status and log transition.

    WARNING: This function is NOT race-condition safe. Use atomic_update_job_status()
    for state transitions where concurrent modifications are possible.
    """
    # Get current job
    job = await get_job(db, job_id)
    if not job:
        return None

    # Prepare update
    update_data = {"status": new_status, **updates}

    # Add timestamp fields based on status
    now = datetime.now(timezone.utc).isoformat()
    if new_status == "funded":
        update_data["funded_at"] = now
    elif new_status == "accepted":
        update_data["accepted_at"] = now
    elif new_status == "delivered":
        update_data["delivered_at"] = now
    elif new_status == "completed":
        update_data["completed_at"] = now
    elif new_status == "disputed":
        update_data["disputed_at"] = now
    elif new_status == "cancelled":
        update_data["cancelled_at"] = now

    result = db.table(JOBS_TABLE).update(update_data).eq("id", job_id).execute()
    return result.data[0] if result.data else None


async def atomic_update_job_status(
    db,
    job_id: str,
    expected_status: str,
    new_status: str,
    actor_id: str,
    **updates,
) -> tuple[dict | None, str | None]:
    """Atomically update job status with optimistic locking.

    Uses UPDATE ... WHERE status = expected_status pattern to prevent race conditions.

    Args:
        db: Database client
        job_id: Job ID to update
        expected_status: The status the job must currently have
        new_status: The status to transition to
        actor_id: ID of the actor performing the transition
        **updates: Additional fields to update

    Returns:
        Tuple of (updated_job, error_message).
        - If successful: (job_dict, None)
        - If job not found: (None, "not_found")
        - If status mismatch (race condition): (None, "conflict")
    """
    # Prepare update data
    update_data = {"status": new_status, **updates}

    # Add timestamp fields based on status
    now = datetime.now(timezone.utc).isoformat()
    if new_status == "funded":
        update_data["funded_at"] = now
    elif new_status == "accepted":
        update_data["accepted_at"] = now
    elif new_status == "delivered":
        update_data["delivered_at"] = now
    elif new_status == "completed":
        update_data["completed_at"] = now
    elif new_status == "disputed":
        update_data["disputed_at"] = now
    elif new_status == "cancelled":
        update_data["cancelled_at"] = now

    # Atomic update: only succeeds if status matches expected
    result = (
        db.table(JOBS_TABLE)
        .update(update_data)
        .eq("id", job_id)
        .eq("status", expected_status)
        .execute()
    )

    if result.data:
        return result.data[0], None

    # Update didn't match - either job doesn't exist or status changed
    # Check which case we're in
    job = await get_job(db, job_id)
    if not job:
        return None, "not_found"

    # Job exists but status didn't match - concurrent modification
    logger.warning(
        f"Race condition detected on job {job_id}: "
        f"expected status '{expected_status}', found '{job['status']}'"
    )
    return None, "conflict"


async def atomic_update_application_status(
    db,
    application_id: str,
    expected_status: str,
    new_status: str,
) -> tuple[dict | None, str | None]:
    """Atomically update application status with optimistic locking.

    Args:
        db: Database client
        application_id: Application ID to update
        expected_status: The status the application must currently have
        new_status: The status to transition to

    Returns:
        Tuple of (updated_application, error_message).
        - If successful: (app_dict, None)
        - If not found: (None, "not_found")
        - If status mismatch: (None, "conflict")
    """
    result = (
        db.table(JOB_APPLICATIONS_TABLE)
        .update({"status": new_status})
        .eq("id", application_id)
        .eq("status", expected_status)
        .execute()
    )

    if result.data:
        return result.data[0], None

    # Check if application exists
    app = await get_application(db, application_id)
    if not app:
        return None, "not_found"

    logger.warning(
        f"Race condition detected on application {application_id}: "
        f"expected status '{expected_status}', found '{app['status']}'"
    )
    return None, "conflict"


async def create_application(db, job_id: str, applicant_id: str, app: JobApplicationCreate) -> dict:
    """Create a job application."""
    data = {
        "job_id": job_id,
        "applicant_id": applicant_id,
        "message": app.message,
        "status": "pending",
    }
    if app.proposed_deadline:
        data["proposed_deadline"] = app.proposed_deadline.isoformat()

    result = db.table(JOB_APPLICATIONS_TABLE).insert(data).execute()
    return result.data[0] if result.data else None


async def get_application(db, application_id: str) -> dict | None:
    """Get an application by ID."""
    result = db.table(JOB_APPLICATIONS_TABLE).select("*").eq("id", application_id).execute()
    return result.data[0] if result.data else None


async def get_applications_for_job(db, job_id: str) -> list[dict]:
    """Get all applications for a job."""
    result = (
        db.table(JOB_APPLICATIONS_TABLE)
        .select("*")
        .eq("job_id", job_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


async def update_application_status(db, application_id: str, new_status: str) -> dict | None:
    """Update an application status."""
    result = (
        db.table(JOB_APPLICATIONS_TABLE)
        .update({"status": new_status})
        .eq("id", application_id)
        .execute()
    )
    return result.data[0] if result.data else None


# =============================================================================
# Helper Functions
# =============================================================================


def to_job_response(job: dict) -> JobResponse:
    """Convert DB job dict to response model."""
    return JobResponse(
        id=job["id"],
        client_id=job["client_id"],
        worker_id=job.get("worker_id"),
        title=job["title"],
        description=job["description"],
        budget_usdc=Decimal(str(job["budget_usdc"])),
        deadline=job["deadline"],
        skills_required=job.get("skills_required") or [],
        escrow_address=job.get("escrow_address"),
        status=job["status"],
        deliverable_url=job.get("deliverable_url"),
        deliverable_hash=job.get("deliverable_hash"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        funded_at=job.get("funded_at"),
        accepted_at=job.get("accepted_at"),
        delivered_at=job.get("delivered_at"),
        completed_at=job.get("completed_at"),
        disputed_at=job.get("disputed_at"),
        cancelled_at=job.get("cancelled_at"),
    )


def to_application_response(app: dict) -> JobApplicationResponse:
    """Convert DB application dict to response model."""
    return JobApplicationResponse(
        id=app["id"],
        job_id=app["job_id"],
        applicant_id=app["applicant_id"],
        message=app["message"],
        proposed_deadline=app.get("proposed_deadline"),
        status=app["status"],
        created_at=app["created_at"],
    )


# Valid state transitions
VALID_TRANSITIONS = {
    "open": {"funded", "cancelled"},
    "funded": {"accepted", "cancelled"},
    "accepted": {"delivered", "disputed", "cancelled"},
    "delivered": {"completed", "disputed"},
    "disputed": {"completed", "cancelled"},  # cancelled via timeout/escalation
}


def can_transition(from_status: str, to_status: str) -> bool:
    """Check if a status transition is valid."""
    return to_status in VALID_TRANSITIONS.get(from_status, set())


# =============================================================================
# Routes
# =============================================================================


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def create_job_listing(
    request: Request,
    job: JobCreate,
    auth: CurrentAgent,
    db: Database,
):
    """
    Create a new job listing.

    The authenticated agent becomes the client (job poster).
    Jobs start in 'open' status and must be funded before workers can be accepted.
    """
    client_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs | client={client_id} | title={job.title[:50]}")

    created = await create_job(db, client_id, job)
    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job",
        )

    logger.info(f"Job created | id={created['id']} | client={client_id}")
    return to_job_response(created)


@router.get("", response_model=JobListResponse)
@limiter.limit("60/minute")
async def list_jobs_endpoint(
    request: Request,
    auth: CurrentAgent,
    db: Database,
    status_filter: JobStatus | None = Query(None, alias="status"),
    skills: list[str] | None = Query(None),
    mine: bool = Query(False, description="Only show jobs I posted or am working on"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List jobs with optional filters.

    Filters:
    - status: Filter by job status
    - skills: Filter by required skills (jobs matching ANY skill)
    - mine: Only show jobs where you're the client or worker
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"GET /jobs | agent={agent_id} | status={status_filter} | mine={mine}")


    # If mine=True, we want jobs where user is client OR worker
    # For simplicity, we'll do two queries and merge
    if mine:
        jobs_as_client, count_client = await list_jobs(
            db, status_filter, skills, client_id=agent_id, limit=limit, offset=offset
        )
        jobs_as_worker, count_worker = await list_jobs(
            db, status_filter, skills, worker_id=agent_id, limit=limit, offset=offset
        )
        # Merge and dedupe
        seen = set()
        jobs = []
        for job in jobs_as_client + jobs_as_worker:
            if job["id"] not in seen:
                seen.add(job["id"])
                jobs.append(job)
        total = len(jobs)
        jobs = jobs[:limit]  # Respect limit after merge
    else:
        jobs, total = await list_jobs(db, status_filter, skills, limit=limit, offset=offset)

    return JobListResponse(
        jobs=[to_job_response(j) for j in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobResponse)
@limiter.limit("60/minute")
async def get_job_details(
    request: Request,
    job_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """Get details of a specific job."""
    logger.info(f"GET /jobs/{job_id} | agent={auth.agent_id or auth.user_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return to_job_response(job)


@router.post("/{job_id}/fund", response_model=JobResponse)
@limiter.limit("10/minute")
async def fund_job(
    request: Request,
    job_id: str,
    fund_request: FundJobRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Fund a job by depositing USDC into escrow.

    Only the client (job poster) can fund their job.
    This transitions the job from 'open' to 'funded'.

    Note: Actual escrow funding requires blockchain integration.
    This endpoint validates the transition and records the tx_hash.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/fund | agent={agent_id}")

    # First, get the job to check ownership (this is safe even with races)
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can fund it",
        )

    # Validate the transition is allowed from current status
    if not can_transition(job["status"], "funded"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot fund job in status: {job['status']}",
        )

    # TODO: Actually deploy/fund escrow contract via CDP
    # For now, just update status

    # Atomic update: only succeeds if job is still in 'open' status
    # This prevents race conditions where two concurrent requests both try to fund
    updated, error = await atomic_update_job_status(
        db, job_id, expected_status="open", new_status="funded", actor_id=agent_id
    )

    if error == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if error == "conflict":
        # Another request already changed the job status
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job status was modified by another request. Please refresh and try again.",
        )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fund job",
        )

    logger.info(f"Job funded | id={job_id} | client={agent_id}")
    return to_job_response(updated)


@router.post("/{job_id}/apply", response_model=JobApplicationResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("30/minute")
async def apply_to_job(
    request: Request,
    job_id: str,
    application: JobApplicationCreate,
    auth: CurrentAgent,
    db: Database,
):
    """
    Apply to work on a job.

    Agents can apply to 'open' or 'funded' jobs.
    The client reviews applications and accepts one worker.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/apply | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] == agent_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot apply to your own job",
        )

    if job["status"] not in ("open", "funded"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot apply to job in status: {job['status']}",
        )

    try:
        created = await create_application(db, job_id, agent_id, application)
    except Exception as e:
        if "duplicate" in str(e).lower() or "unique" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You have already applied to this job",
            )
        raise

    if not created:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create application",
        )

    logger.info(f"Application created | job={job_id} | applicant={agent_id}")
    return to_application_response(created)


@router.get("/{job_id}/applications", response_model=JobApplicationListResponse)
@limiter.limit("30/minute")
async def list_applications(
    request: Request,
    job_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    List applications for a job.

    Only the job client can view all applications.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"GET /jobs/{job_id}/applications | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can view applications",
        )

    applications = await get_applications_for_job(db, job_id)

    return JobApplicationListResponse(
        applications=[to_application_response(a) for a in applications],
        total=len(applications),
    )


@router.post("/{job_id}/accept", response_model=JobResponse)
@limiter.limit("10/minute")
async def accept_application(
    request: Request,
    job_id: str,
    accept_request: AcceptApplicationRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Accept a job application.

    Only the job client can accept applications.
    The job must be 'funded' before accepting a worker.
    This transitions the job to 'accepted' and assigns the worker.

    Uses atomic updates to prevent race conditions when multiple applications
    are accepted concurrently.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/accept | agent={agent_id} | app={accept_request.application_id}")

    # Get job to check ownership (safe even with races - ownership doesn't change)
    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can accept applications",
        )

    if not can_transition(job["status"], "accepted"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot accept applications for job in status: {job['status']}. Fund the job first.",
        )

    # Get application
    application = await get_application(db, accept_request.application_id)
    if not application or application["job_id"] != job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found for this job",
        )

    if application["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Application is already {application['status']}",
        )

    # ATOMIC TRANSACTION: Accept application and update job
    # Order matters: update job first (the critical resource), then application
    # If job update fails due to race, we haven't modified anything yet

    # Step 1: Atomically update job status (only succeeds if still 'funded')
    updated_job, job_error = await atomic_update_job_status(
        db,
        job_id,
        expected_status="funded",
        new_status="accepted",
        actor_id=agent_id,
        worker_id=application["applicant_id"],
    )

    if job_error == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job_error == "conflict":
        # Another request already accepted a different application
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another application was already accepted for this job.",
        )

    if not updated_job:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept application",
        )

    # Step 2: Atomically update application status (only succeeds if still 'pending')
    # If this fails due to race, we log a warning but don't fail the request
    # because the job is already assigned to this worker
    updated_app, app_error = await atomic_update_application_status(
        db, accept_request.application_id, expected_status="pending", new_status="accepted"
    )

    if app_error == "conflict":
        # Application was already accepted (unlikely but possible)
        logger.warning(
            f"Application {accept_request.application_id} status already changed, "
            "but job was successfully updated"
        )

    # Step 3: Reject other pending applications (best effort, non-critical)
    # These updates don't need to be atomic - if one fails, others still work
    all_apps = await get_applications_for_job(db, job_id)
    for app in all_apps:
        if app["id"] != accept_request.application_id and app["status"] == "pending":
            try:
                await update_application_status(db, app["id"], "rejected")
            except Exception as e:
                # Log but don't fail - rejecting other apps is non-critical
                logger.warning(f"Failed to reject application {app['id']}: {e}")

    logger.info(f"Application accepted | job={job_id} | worker={application['applicant_id']}")
    return to_job_response(updated_job)


@router.post("/{job_id}/deliver", response_model=JobResponse)
@limiter.limit("10/minute")
async def deliver_job(
    request: Request,
    job_id: str,
    deliver_request: DeliverJobRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Submit deliverable for a job.

    Only the assigned worker can submit deliverables.
    This transitions the job from 'accepted' to 'delivered'.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/deliver | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["worker_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the assigned worker can deliver",
        )

    if not can_transition(job["status"], "delivered"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot deliver job in status: {job['status']}",
        )

    updated = await update_job_status(
        db,
        job_id,
        "delivered",
        agent_id,
        deliverable_url=deliver_request.deliverable_url,
        deliverable_hash=deliver_request.deliverable_hash,
    )

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit deliverable",
        )

    logger.info(f"Job delivered | id={job_id} | worker={agent_id}")
    return to_job_response(updated)


@router.post("/{job_id}/approve", response_model=JobResponse)
@limiter.limit("10/minute")
async def approve_job(
    request: Request,
    job_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    Approve the deliverable and release escrow.

    Only the job client can approve.
    This transitions the job from 'delivered' to 'completed'
    and releases the escrowed USDC to the worker.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/approve | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can approve",
        )

    if not can_transition(job["status"], "completed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot approve job in status: {job['status']}",
        )

    # TODO: Release escrow via smart contract
    updated = await update_job_status(db, job_id, "completed", agent_id)

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to approve job",
        )

    logger.info(f"Job approved | id={job_id} | client={agent_id}")
    return to_job_response(updated)


@router.post("/{job_id}/dispute", response_model=JobResponse)
@limiter.limit("5/minute")
async def dispute_job(
    request: Request,
    job_id: str,
    dispute_request: DisputeJobRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Raise a dispute for the job.

    Either the client or worker can raise a dispute.
    This transitions the job to 'disputed' for arbitration.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/dispute | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id and job["worker_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the client or worker can dispute",
        )

    if not can_transition(job["status"], "disputed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot dispute job in status: {job['status']}",
        )

    # TODO: Log dispute reason in transitions table
    updated = await update_job_status(db, job_id, "disputed", agent_id)

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to raise dispute",
        )

    logger.info(f"Job disputed | id={job_id} | by={agent_id} | reason={dispute_request.reason[:50]}")
    return to_job_response(updated)


@router.post("/{job_id}/cancel", response_model=JobResponse)
@limiter.limit("10/minute")
async def cancel_job(
    request: Request,
    job_id: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    Cancel a job.

    Only the job client can cancel.
    Jobs can be cancelled when in 'open', 'funded', or 'accepted' status.
    If funded, escrow will be refunded to the client.
    """
    agent_id = auth.agent_id or auth.user_id
    logger.info(f"POST /jobs/{job_id}/cancel | agent={agent_id}")

    job = await get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job["client_id"] != agent_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the job client can cancel",
        )

    if not can_transition(job["status"], "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job['status']}",
        )

    # TODO: Refund escrow if funded
    updated = await update_job_status(db, job_id, "cancelled", agent_id)

    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel job",
        )

    logger.info(f"Job cancelled | id={job_id} | client={agent_id}")
    return to_job_response(updated)
