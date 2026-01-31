"""Escrow routes for Kernle Commerce.

Endpoints for escrow contract information.
Most escrow operations are performed through job endpoints.
"""

from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from ...auth import CurrentAgent
from ...database import Database
from ...logging_config import get_logger
from ...rate_limit import limiter

logger = get_logger("kernle.commerce.escrow")
router = APIRouter(prefix="/escrow", tags=["commerce", "escrow"])


# =============================================================================
# Request/Response Models
# =============================================================================


class EscrowResponse(BaseModel):
    """Escrow contract details response."""

    address: str
    job_id: str
    client_id: str
    worker_id: str | None
    amount_usdc: Decimal
    status: str  # funded, released, refunded, disputed
    chain: str
    funded_at: datetime | None = None
    released_at: datetime | None = None


# =============================================================================
# Database Operations
# =============================================================================

JOBS_TABLE = "jobs"


async def get_job_by_escrow(db, escrow_address: str) -> dict | None:
    """Get a job by its escrow contract address."""
    result = db.table(JOBS_TABLE).select("*").eq("escrow_address", escrow_address).execute()
    return result.data[0] if result.data else None


# =============================================================================
# Routes
# =============================================================================


@router.get("/{address}", response_model=EscrowResponse)
@limiter.limit("30/minute")
async def get_escrow_details(
    request: Request,
    address: str,
    auth: CurrentAgent,
    db: Database,
):
    """
    Get escrow contract details by address.

    Returns information about the escrow contract including
    the associated job, amounts, and current status.
    """
    logger.info(f"GET /escrow/{address}")

    # Validate address format
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid escrow address format",
        )

    job = await get_job_by_escrow(db, address)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Escrow not found",
        )

    # Map job status to escrow status
    job_status = job["status"]
    if job_status == "completed":
        escrow_status = "released"
    elif job_status == "cancelled":
        escrow_status = "refunded"
    elif job_status == "disputed":
        escrow_status = "disputed"
    elif job_status in ("funded", "accepted", "delivered"):
        escrow_status = "funded"
    else:
        escrow_status = "pending"

    return EscrowResponse(
        address=address,
        job_id=job["id"],
        client_id=job["client_id"],
        worker_id=job.get("worker_id"),
        amount_usdc=Decimal(str(job["budget_usdc"])),
        status=escrow_status,
        chain="base",  # TODO: Get from wallet/job
        funded_at=job.get("funded_at"),
        released_at=job.get("completed_at") if job_status == "completed" else None,
    )
