"""Subscription management routes."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from ..auth import CurrentAgent
from ..database import Database
from ..logging_config import get_logger
from ..rate_limit import limiter

logger = get_logger("kernle.subscriptions")

router = APIRouter(tags=["subscriptions"])

# ---------------------------------------------------------------------------
# Pydantic models — request / response
# ---------------------------------------------------------------------------

VALID_TIERS = {"free", "core", "pro", "enterprise"}
TIER_PRICES = {"free": "0", "core": "5.000000", "pro": "15.000000"}


# -- Shared / nested models -------------------------------------------------

class TierInfo(BaseModel):
    """Current subscription tier details."""
    tier: str
    status: str  # active | grace_period | cancelled | expired
    starts_at: datetime | None = None
    renews_at: datetime | None = None
    cancelled_at: datetime | None = None
    auto_renew: bool = True
    renewal_amount: str | None = None


class UsageInfo(BaseModel):
    """Current-period usage snapshot."""
    period: str  # YYYY-MM
    storage_used: int = 0
    storage_limit: int = 0
    sync_count: int = 0
    last_sync: datetime | None = None
    agents_used: int = 0
    agents_limit: int = 0


class PaymentInfo(BaseModel):
    """Single payment record."""
    id: str
    amount: str
    currency: str = "USDC"
    tx_hash: str | None = None
    from_address: str | None = None
    to_address: str | None = None
    chain: str = "base"
    status: str  # pending | confirmed | failed | refunded
    period_start: datetime | None = None
    period_end: datetime | None = None
    created_at: datetime | None = None
    confirmed_at: datetime | None = None


# -- Request models ----------------------------------------------------------

class UpgradeRequest(BaseModel):
    """Request to upgrade subscription tier."""
    tier: str = Field(..., description="Target tier: core, pro, or enterprise")


class DowngradeRequest(BaseModel):
    """Request to downgrade subscription tier."""
    tier: str = Field(..., description="Target tier to downgrade to")


class PaymentConfirmRequest(BaseModel):
    """Confirm an on-chain payment."""
    payment_id: str = Field(..., description="Payment ID returned by /upgrade")
    tx_hash: str = Field(
        ...,
        description="On-chain transaction hash (0x + 64 hex chars)",
        pattern=r"^0x[0-9a-fA-F]{64}$",
    )


# -- Response models ---------------------------------------------------------

class SubscriptionStatusResponse(BaseModel):
    """Full subscription status with usage."""
    subscription: TierInfo
    usage: UsageInfo


class UpgradeResponse(BaseModel):
    """Payment instructions returned after requesting an upgrade."""
    payment_id: str
    amount: str
    currency: str = "USDC"
    treasury_address: str
    chain: str = "base"
    tier: str
    message: str


class DowngradeResponse(BaseModel):
    """Confirmation of a scheduled downgrade."""
    current_tier: str
    new_tier: str
    effective_at: datetime
    message: str


class CancelResponse(BaseModel):
    """Confirmation of cancellation."""
    tier: str
    status: str
    cancelled_at: datetime
    active_until: datetime
    message: str


class ReactivateResponse(BaseModel):
    """Confirmation of reactivation."""
    tier: str
    status: str
    auto_renew: bool
    renews_at: datetime | None = None
    message: str


class PaymentHistoryResponse(BaseModel):
    """Paginated payment history."""
    payments: list[PaymentInfo]
    total: int
    page: int
    per_page: int


class PaymentConfirmResponse(BaseModel):
    """Result of payment confirmation."""
    payment_id: str
    status: str  # confirmed | pending_verification | failed
    tier: str | None = None
    activated_at: datetime | None = None
    message: str


class UsageResponse(BaseModel):
    """Current-period usage with quota info."""
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Service imports (lazy — the companion task creates these)
# ---------------------------------------------------------------------------

# These are imported inside each handler to allow the module to load even if
# the service module is not yet present (companion task builds it).  In
# production both will exist.

def _svc():
    """Lazy-import subscription service functions."""
    from ..subscriptions import service as svc
    return svc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# ── GET /api/v1/subscriptions/me ──────────────────────────────────────────

@router.get("/subscriptions/me", response_model=SubscriptionStatusResponse)
async def get_subscription_status(
    auth: CurrentAgent,
    db: Database,
):
    """
    Get current subscription status and usage for the authenticated user.

    Returns tier info, renewal dates, and current-period usage.
    """
    svc = _svc()

    subscription = await svc.get_subscription(db, auth.user_id)
    if not subscription:
        # Every user implicitly has a free tier
        subscription = {
            "tier": "free",
            "status": "active",
            "starts_at": None,
            "renews_at": None,
            "cancelled_at": None,
            "auto_renew": False,
            "renewal_amount": None,
        }

    usage = await svc.get_current_usage(db, auth.user_id)
    if not usage:
        usage = {
            "period": datetime.utcnow().strftime("%Y-%m"),
            "storage_used": 0,
            "storage_limit": 0,
            "sync_count": 0,
            "last_sync": None,
            "agents_used": 0,
            "agents_limit": 0,
        }

    return SubscriptionStatusResponse(
        subscription=TierInfo(**subscription),
        usage=UsageInfo(**usage),
    )


# ── POST /api/v1/subscriptions/upgrade ────────────────────────────────────

@router.post("/subscriptions/upgrade", response_model=UpgradeResponse)
@limiter.limit("5/minute")
async def upgrade_subscription(
    request: Request,
    body: UpgradeRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Request a subscription upgrade.

    Returns payment instructions (amount, treasury address, payment_id).
    The client should make an on-chain USDC transfer and then call
    POST /subscriptions/payments/confirm with the tx_hash.
    """
    target_tier = body.tier.lower()

    if target_tier not in VALID_TIERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier '{body.tier}'. Valid tiers: {', '.join(sorted(VALID_TIERS))}",
        )

    if target_tier == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot upgrade to free tier. Use /downgrade instead.",
        )

    if target_tier == "enterprise":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enterprise tier requires custom arrangement. Contact support.",
        )

    svc = _svc()

    # Validate upgrade is allowed (e.g. not already on this tier or higher)
    current_sub = await svc.get_subscription(db, auth.user_id)
    current_tier = current_sub.get("tier", "free") if current_sub else "free"

    tier_order = {"free": 0, "core": 1, "pro": 2, "enterprise": 3}
    if tier_order.get(target_tier, 0) <= tier_order.get(current_tier, 0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upgrade from '{current_tier}' to '{target_tier}'. "
                   f"Current tier is equal or higher.",
        )

    # Create a pending payment record and return instructions
    payment = await svc.create_upgrade_payment(db, auth.user_id, target_tier)
    if not payment:
        logger.error(f"Failed to create upgrade payment for user={auth.user_id} tier={target_tier}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate upgrade. Please try again.",
        )

    logger.info(
        f"Upgrade initiated: user={auth.user_id} "
        f"{current_tier} -> {target_tier} payment_id={payment['payment_id']}"
    )

    return UpgradeResponse(
        payment_id=payment["payment_id"],
        amount=payment["amount"],
        currency=payment.get("currency", "USDC"),
        treasury_address=payment["treasury_address"],
        chain=payment.get("chain", "base"),
        tier=target_tier,
        message=f"Send {payment['amount']} USDC to {payment['treasury_address']} on Base. "
                f"Then confirm with POST /subscriptions/payments/confirm.",
    )


# ── POST /api/v1/subscriptions/downgrade ──────────────────────────────────

@router.post("/subscriptions/downgrade", response_model=DowngradeResponse)
@limiter.limit("5/minute")
async def downgrade_subscription(
    request: Request,
    body: DowngradeRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Schedule a subscription downgrade (effective at end of current period).

    The user keeps their current tier until the period ends.
    """
    target_tier = body.tier.lower()

    if target_tier not in VALID_TIERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier '{body.tier}'. Valid tiers: {', '.join(sorted(VALID_TIERS))}",
        )

    svc = _svc()

    current_sub = await svc.get_subscription(db, auth.user_id)
    current_tier = current_sub.get("tier", "free") if current_sub else "free"

    tier_order = {"free": 0, "core": 1, "pro": 2, "enterprise": 3}
    if tier_order.get(target_tier, 0) >= tier_order.get(current_tier, 0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot downgrade from '{current_tier}' to '{target_tier}'. "
                   f"Target must be a lower tier.",
        )

    result = await svc.schedule_downgrade(db, auth.user_id, target_tier)
    if not result:
        logger.error(f"Failed to schedule downgrade for user={auth.user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule downgrade.",
        )

    logger.info(
        f"Downgrade scheduled: user={auth.user_id} "
        f"{current_tier} -> {target_tier} effective_at={result['effective_at']}"
    )

    return DowngradeResponse(
        current_tier=current_tier,
        new_tier=target_tier,
        effective_at=result["effective_at"],
        message=f"Downgrade to '{target_tier}' scheduled. "
                f"Your current '{current_tier}' tier remains active until {result['effective_at']}.",
    )


# ── POST /api/v1/subscriptions/cancel ─────────────────────────────────────

@router.post("/subscriptions/cancel", response_model=CancelResponse)
@limiter.limit("5/minute")
async def cancel_subscription(
    request: Request,
    auth: CurrentAgent,
    db: Database,
):
    """
    Cancel auto-renewal for the current subscription.

    The subscription remains active until the end of the current billing period.
    """
    svc = _svc()

    current_sub = await svc.get_subscription(db, auth.user_id)
    if not current_sub or current_sub.get("tier") == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active paid subscription to cancel.",
        )

    if current_sub.get("status") == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription is already cancelled.",
        )

    result = await svc.cancel_subscription(db, auth.user_id)
    if not result:
        logger.error(f"Failed to cancel subscription for user={auth.user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription.",
        )

    logger.info(f"Subscription cancelled: user={auth.user_id} tier={current_sub['tier']}")

    return CancelResponse(
        tier=current_sub["tier"],
        status="cancelled",
        cancelled_at=result["cancelled_at"],
        active_until=result["active_until"],
        message=f"Auto-renewal cancelled. Your '{current_sub['tier']}' tier remains active "
                f"until {result['active_until']}.",
    )


# ── POST /api/v1/subscriptions/reactivate ─────────────────────────────────

@router.post("/subscriptions/reactivate", response_model=ReactivateResponse)
@limiter.limit("5/minute")
async def reactivate_subscription(
    request: Request,
    auth: CurrentAgent,
    db: Database,
):
    """
    Re-enable auto-renewal for a cancelled subscription.

    Only works if the subscription has not yet expired.
    """
    svc = _svc()

    current_sub = await svc.get_subscription(db, auth.user_id)
    if not current_sub or current_sub.get("tier") == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No paid subscription to reactivate.",
        )

    if current_sub.get("status") not in ("cancelled",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Subscription status is '{current_sub.get('status')}' — "
                   f"only cancelled subscriptions can be reactivated.",
        )

    result = await svc.reactivate_subscription(db, auth.user_id)
    if not result:
        logger.error(f"Failed to reactivate subscription for user={auth.user_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reactivate subscription.",
        )

    logger.info(f"Subscription reactivated: user={auth.user_id} tier={current_sub['tier']}")

    return ReactivateResponse(
        tier=current_sub["tier"],
        status="active",
        auto_renew=True,
        renews_at=result.get("renews_at"),
        message=f"Auto-renewal re-enabled for '{current_sub['tier']}' tier.",
    )


# ── GET /api/v1/subscriptions/payments ─────────────────────────────────────

@router.get("/subscriptions/payments", response_model=PaymentHistoryResponse)
async def list_payments(
    auth: CurrentAgent,
    db: Database,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
):
    """
    List payment history for the authenticated user.

    Supports pagination via page/per_page query params.
    """
    svc = _svc()

    result = await svc.list_payments(db, auth.user_id, page=page, per_page=per_page)

    payments = [PaymentInfo(**p) for p in result.get("payments", [])]

    return PaymentHistoryResponse(
        payments=payments,
        total=result.get("total", 0),
        page=page,
        per_page=per_page,
    )


# ── POST /api/v1/subscriptions/payments/confirm ───────────────────────────

@router.post("/subscriptions/payments/confirm", response_model=PaymentConfirmResponse)
@limiter.limit("10/minute")
async def confirm_payment(
    request: Request,
    body: PaymentConfirmRequest,
    auth: CurrentAgent,
    db: Database,
):
    """
    Confirm an on-chain USDC payment by providing the transaction hash.

    The server verifies the transaction on-chain (or marks it for async
    verification) and activates the tier upgrade upon confirmation.
    """
    svc = _svc()

    # Verify the payment belongs to this user
    payment = await svc.get_payment(db, body.payment_id, auth.user_id)
    if not payment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payment not found or does not belong to this account.",
        )

    if payment.get("status") == "confirmed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment has already been confirmed.",
        )

    # Attempt on-chain verification and tier activation
    result = await svc.confirm_payment(db, body.payment_id, body.tx_hash, auth.user_id)
    if not result:
        logger.error(
            f"Payment confirmation failed: user={auth.user_id} "
            f"payment_id={body.payment_id} tx_hash={body.tx_hash}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to confirm payment. Please try again or contact support.",
        )

    logger.info(
        f"Payment confirmed: user={auth.user_id} payment_id={body.payment_id} "
        f"tx_hash={body.tx_hash} status={result['status']}"
    )

    return PaymentConfirmResponse(
        payment_id=body.payment_id,
        status=result["status"],
        tier=result.get("tier"),
        activated_at=result.get("activated_at"),
        message=result.get("message", "Payment processed."),
    )


# ── GET /api/v1/usage/me ──────────────────────────────────────────────────

@router.get("/usage/me", response_model=UsageResponse)
async def get_usage(
    auth: CurrentAgent,
    db: Database,
):
    """
    Get current-period usage with quota info for the authenticated user.

    Returns storage used/limit, sync count, active agents count/limit.
    """
    svc = _svc()

    usage = await svc.get_current_usage(db, auth.user_id)
    if not usage:
        usage = {
            "period": datetime.utcnow().strftime("%Y-%m"),
            "storage_used": 0,
            "storage_limit": 0,
            "sync_count": 0,
            "last_sync": None,
            "agents_used": 0,
            "agents_limit": 0,
        }

    return UsageResponse(usage=UsageInfo(**usage))
