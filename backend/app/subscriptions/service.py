"""Subscription service — business logic for tier management, payments, and quotas.

All monetary values use Decimal. Sync frequency is never limited.
On-chain payment verification uses the payments.verification module
(USDC transfer verification on Base / Base Sepolia / Ethereum).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from supabase import Client

from .models import (
    TIER_CONFIGS,
    OverflowPricing,
    PaymentInfo,
    PaymentStatus,
    QuotaCheckResult,
    Subscription,
    SubscriptionPayment,
    SubscriptionStatus,
    SubscriptionTier,
    TierConfig,
    UsageRecord,
    get_tier_config,
    is_downgrade,
    is_upgrade,
)

logger = logging.getLogger("kernle.subscriptions")

# =============================================================================
# Table names (keep in sync with SQL migrations)
# =============================================================================

SUBSCRIPTIONS_TABLE = "subscriptions"
SUBSCRIPTION_PAYMENTS_TABLE = "subscription_payments"
USAGE_RECORDS_TABLE = "usage_records"

# Kernle treasury address on Base (receives subscription payments)
# Loaded from environment — MUST be set before production deployment
import os

TREASURY_ADDRESS = os.environ.get(
    "KERNLE_TREASURY_ADDRESS",
    "0x0000000000000000000000000000000000000000",  # Default: zero address (testnet only)
)

def _validate_treasury_address():
    """Fail loudly if treasury is zero address in production."""
    env = os.environ.get("KERNLE_ENV", "development")
    if env == "production" and TREASURY_ADDRESS == "0x0000000000000000000000000000000000000000":
        raise RuntimeError(
            "FATAL: KERNLE_TREASURY_ADDRESS is zero address in production! "
            "All payments would be burned. Set a real multisig address."
        )

_validate_treasury_address()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _current_period() -> str:
    """Return the current billing period as 'YYYY-MM'."""
    return _now().strftime("%Y-%m")


def _next_month(dt: datetime) -> datetime:
    """Return the same day next month (clamped to valid day)."""
    import calendar

    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    max_day = calendar.monthrange(year, month)[1]
    return dt.replace(year=year, month=month, day=min(dt.day, max_day))


# =============================================================================
# Service
# =============================================================================


class SubscriptionService:
    """Stateless service — every method receives a Supabase `Client`."""

    # ------------------------------------------------------------------
    # Subscription CRUD
    # ------------------------------------------------------------------

    @staticmethod
    async def get_or_create_subscription(
        db: Client,
        user_id: str,
    ) -> Subscription:
        """Return the user's active subscription, creating a free-tier one if none exists."""
        import asyncio

        def _query():
            return (
                db.table(SUBSCRIPTIONS_TABLE)
                .select("*")
                .eq("user_id", user_id)
                .in_("status", ["active", "grace_period"])
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

        result = await asyncio.to_thread(_query)

        if result.data:
            return Subscription(**result.data[0])

        # No active subscription — create free tier
        now = _now()
        data = {
            "user_id": user_id,
            "tier": SubscriptionTier.free.value,
            "status": SubscriptionStatus.active.value,
            "starts_at": now.isoformat(),
            "renews_at": None,  # free tier doesn't renew
            "auto_renew": False,
        }

        def _insert():
            return db.table(SUBSCRIPTIONS_TABLE).insert(data).execute()

        result = await asyncio.to_thread(_insert)

        if not result.data:
            raise RuntimeError(f"Failed to create free subscription for user {user_id}")

        logger.info("Created free-tier subscription for user %s", user_id)
        return Subscription(**result.data[0])

    @staticmethod
    async def get_subscription(
        db: Client,
        subscription_id: str,
    ) -> Subscription | None:
        """Fetch a subscription by its ID."""
        import asyncio

        def _query():
            return (
                db.table(SUBSCRIPTIONS_TABLE)
                .select("*")
                .eq("id", subscription_id)
                .limit(1)
                .execute()
            )

        result = await asyncio.to_thread(_query)
        return Subscription(**result.data[0]) if result.data else None

    # ------------------------------------------------------------------
    # Upgrade
    # ------------------------------------------------------------------

    @staticmethod
    async def upgrade_tier(
        db: Client,
        user_id: str,
        new_tier: SubscriptionTier,
    ) -> tuple[Subscription, PaymentInfo]:
        """Validate and initiate an upgrade.

        Returns the updated subscription and payment info the caller
        should use to submit on-chain payment.

        Raises:
            ValueError: if the tier change is invalid.
        """
        import asyncio

        sub = await SubscriptionService.get_or_create_subscription(db, user_id)
        current_tier = sub.tier

        if new_tier == current_tier:
            raise ValueError(f"Already on {current_tier.value} tier")

        if not is_upgrade(current_tier, new_tier):
            raise ValueError(
                f"Cannot upgrade from {current_tier.value} to {new_tier.value} — "
                "use downgrade instead"
            )

        if new_tier == SubscriptionTier.enterprise:
            raise ValueError("Enterprise tier requires custom arrangement — contact support")

        config = get_tier_config(new_tier)
        now = _now()
        renews_at = _next_month(now)

        update = {
            "tier": new_tier.value,
            "status": SubscriptionStatus.active.value,
            "starts_at": now.isoformat(),
            "renews_at": renews_at.isoformat(),
            "auto_renew": True,
            "updated_at": now.isoformat(),
        }

        def _update():
            return (
                db.table(SUBSCRIPTIONS_TABLE)
                .update(update)
                .eq("id", sub.id)
                .execute()
            )

        result = await asyncio.to_thread(_update)

        if not result.data:
            raise RuntimeError("Failed to update subscription")

        updated_sub = Subscription(**result.data[0])

        payment_info = PaymentInfo(
            amount=config.price_usdc,
            currency="USDC",
            to_address=TREASURY_ADDRESS,
            chain="base",
            status=PaymentStatus.pending,
        )

        logger.info(
            "User %s upgraded %s → %s (amount=%s USDC)",
            user_id,
            current_tier.value,
            new_tier.value,
            config.price_usdc,
        )
        return updated_sub, payment_info

    # ------------------------------------------------------------------
    # Downgrade
    # ------------------------------------------------------------------

    @staticmethod
    async def downgrade_tier(
        db: Client,
        user_id: str,
        new_tier: SubscriptionTier,
    ) -> tuple[Subscription, datetime]:
        """Schedule a downgrade effective at the next billing period.

        Returns the subscription (unchanged tier until period ends) and the
        effective date.

        Raises:
            ValueError: if the tier change is invalid.
        """
        import asyncio

        sub = await SubscriptionService.get_or_create_subscription(db, user_id)

        if new_tier == sub.tier:
            raise ValueError(f"Already on {sub.tier.value} tier")

        if not is_downgrade(sub.tier, new_tier):
            raise ValueError(
                f"Cannot downgrade from {sub.tier.value} to {new_tier.value} — "
                "use upgrade instead"
            )

        # Downgrade takes effect at the end of the current period
        effective_at = sub.renews_at or _next_month(_now())

        # Store the pending downgrade — the renewal job will apply it
        # We record the target tier in a metadata-friendly way:
        # set auto_renew to false so the current tier expires, then
        # the renewal checker will create the new-tier subscription.
        now = _now()
        update = {
            "auto_renew": False,
            "updated_at": now.isoformat(),
            # Store pending downgrade tier for the renewal job
            # Using cancelled_at as a signal that a change is pending
        }

        def _update():
            return (
                db.table(SUBSCRIPTIONS_TABLE)
                .update(update)
                .eq("id", sub.id)
                .execute()
            )

        result = await asyncio.to_thread(_update)

        if not result.data:
            raise RuntimeError("Failed to schedule downgrade")

        updated_sub = Subscription(**result.data[0])

        logger.info(
            "User %s scheduled downgrade %s → %s effective %s",
            user_id,
            sub.tier.value,
            new_tier.value,
            effective_at.isoformat(),
        )
        return updated_sub, effective_at

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    @staticmethod
    async def cancel_subscription(
        db: Client,
        user_id: str,
    ) -> Subscription:
        """Cancel auto-renewal. The subscription remains active until renews_at."""
        import asyncio

        sub = await SubscriptionService.get_or_create_subscription(db, user_id)

        if sub.tier == SubscriptionTier.free:
            raise ValueError("Free tier cannot be cancelled")

        if sub.status == SubscriptionStatus.cancelled:
            raise ValueError("Subscription is already cancelled")

        now = _now()
        update = {
            "auto_renew": False,
            "cancelled_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        def _update():
            return (
                db.table(SUBSCRIPTIONS_TABLE)
                .update(update)
                .eq("id", sub.id)
                .execute()
            )

        result = await asyncio.to_thread(_update)

        if not result.data:
            raise RuntimeError("Failed to cancel subscription")

        logger.info("User %s cancelled subscription (expires %s)", user_id, sub.renews_at)
        return Subscription(**result.data[0])

    # ------------------------------------------------------------------
    # Renewal
    # ------------------------------------------------------------------

    @staticmethod
    async def check_renewal(
        db: Client,
        subscription_id: str,
    ) -> Subscription:
        """Check whether a subscription should renew and handle accordingly.

        Called by a periodic job. Logic:
        1. If auto_renew is True and renews_at has passed → attempt renewal
           (actual payment initiation is the caller's responsibility).
        2. If auto_renew is False and renews_at has passed → expire or
           downgrade to free.
        3. If still within period → no-op.

        Returns the (possibly updated) subscription.
        """
        import asyncio

        sub = await SubscriptionService.get_subscription(db, subscription_id)
        if sub is None:
            raise ValueError(f"Subscription {subscription_id} not found")

        # Free tier never renews
        if sub.tier == SubscriptionTier.free:
            return sub

        # Not yet due
        now = _now()
        if sub.renews_at and now < sub.renews_at:
            return sub

        # --- Renewal is due ---

        if sub.auto_renew:
            # Extend by one month — caller must verify payment succeeded
            new_renews = _next_month(now)
            update = {
                "renews_at": new_renews.isoformat(),
                "status": SubscriptionStatus.active.value,
                "updated_at": now.isoformat(),
            }

            def _renew():
                return (
                    db.table(SUBSCRIPTIONS_TABLE)
                    .update(update)
                    .eq("id", sub.id)
                    .execute()
                )

            result = await asyncio.to_thread(_renew)
            logger.info("Subscription %s renewed until %s", subscription_id, new_renews)
            return Subscription(**result.data[0]) if result.data else sub

        # auto_renew is off — check grace period
        if sub.status == SubscriptionStatus.active:
            # Enter 7-day grace period
            update = {
                "status": SubscriptionStatus.grace_period.value,
                "updated_at": now.isoformat(),
            }

            def _grace():
                return (
                    db.table(SUBSCRIPTIONS_TABLE)
                    .update(update)
                    .eq("id", sub.id)
                    .execute()
                )

            result = await asyncio.to_thread(_grace)
            logger.info("Subscription %s entered grace period", subscription_id)
            return Subscription(**result.data[0]) if result.data else sub

        if sub.status == SubscriptionStatus.grace_period:
            # Grace period check: 7 days after renews_at
            grace_end = sub.renews_at
            if grace_end:
                from datetime import timedelta

                grace_end = grace_end + timedelta(days=7)

            if grace_end is None or now >= grace_end:
                # Grace period expired — downgrade to free
                update = {
                    "tier": SubscriptionTier.free.value,
                    "status": SubscriptionStatus.expired.value,
                    "auto_renew": False,
                    "renews_at": None,
                    "updated_at": now.isoformat(),
                }

                def _expire():
                    return (
                        db.table(SUBSCRIPTIONS_TABLE)
                        .update(update)
                        .eq("id", sub.id)
                        .execute()
                    )

                result = await asyncio.to_thread(_expire)
                logger.warning("Subscription %s expired → free tier", subscription_id)
                return Subscription(**result.data[0]) if result.data else sub

        return sub

    # ------------------------------------------------------------------
    # Payments
    # ------------------------------------------------------------------

    @staticmethod
    async def record_payment(
        db: Client,
        subscription_id: str,
        tx_hash: str,
        amount: Decimal,
        from_address: str,
        to_address: str = TREASURY_ADDRESS,
        chain: str = "base",
        currency: str = "USDC",
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        user_id: str | None = None,
    ) -> SubscriptionPayment:
        """Store a payment record.

        On-chain verification happens elsewhere — this just persists
        the record with status=pending until confirmed.
        """
        import asyncio

        now = _now()
        if period_start is None:
            period_start = now
        if period_end is None:
            period_end = _next_month(now)

        data = {
            "subscription_id": subscription_id,
            "amount": str(amount),  # Supabase stores DECIMAL as string
            "currency": currency,
            "tx_hash": tx_hash,
            "from_address": from_address,
            "to_address": to_address,
            "chain": chain,
            "status": PaymentStatus.pending.value,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "created_at": now.isoformat(),
        }
        if user_id:
            data["user_id"] = user_id

        def _insert():
            return db.table(SUBSCRIPTION_PAYMENTS_TABLE).insert(data).execute()

        result = await asyncio.to_thread(_insert)

        if not result.data:
            raise RuntimeError(f"Failed to record payment for subscription {subscription_id}")

        logger.info(
            "Recorded payment %s for subscription %s (amount=%s %s)",
            tx_hash,
            subscription_id,
            amount,
            currency,
        )
        return SubscriptionPayment(**result.data[0])

    @staticmethod
    async def confirm_payment(
        db: Client,
        tx_hash: str,
    ) -> SubscriptionPayment | None:
        """Mark a payment as confirmed (called after on-chain verification)."""
        import asyncio

        now = _now()

        def _update():
            return (
                db.table(SUBSCRIPTION_PAYMENTS_TABLE)
                .update({
                    "status": PaymentStatus.confirmed.value,
                    "confirmed_at": now.isoformat(),
                })
                .eq("tx_hash", tx_hash)
                .execute()
            )

        result = await asyncio.to_thread(_update)
        if not result.data:
            return None

        logger.info("Payment %s confirmed", tx_hash)
        return SubscriptionPayment(**result.data[0])

    @staticmethod
    async def fail_payment(
        db: Client,
        tx_hash: str,
    ) -> SubscriptionPayment | None:
        """Mark a payment as failed."""
        import asyncio

        def _update():
            return (
                db.table(SUBSCRIPTION_PAYMENTS_TABLE)
                .update({"status": PaymentStatus.failed.value})
                .eq("tx_hash", tx_hash)
                .execute()
            )

        result = await asyncio.to_thread(_update)
        if not result.data:
            return None

        logger.warning("Payment %s failed", tx_hash)
        return SubscriptionPayment(**result.data[0])

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    @staticmethod
    async def get_usage(
        db: Client,
        user_id: str,
    ) -> UsageRecord:
        """Get current-period usage, creating a record if needed."""
        import asyncio

        period = _current_period()

        def _upsert():
            return (
                db.table(USAGE_RECORDS_TABLE)
                .upsert(
                    {
                        "user_id": user_id,
                        "period": period,
                    },
                    on_conflict="user_id,period",
                )
                .execute()
            )

        result = await asyncio.to_thread(_upsert)

        if result.data:
            return UsageRecord(**result.data[0])

        # Fallback: try select
        def _select():
            return (
                db.table(USAGE_RECORDS_TABLE)
                .select("*")
                .eq("user_id", user_id)
                .eq("period", period)
                .limit(1)
                .execute()
            )

        result = await asyncio.to_thread(_select)
        if result.data:
            return UsageRecord(**result.data[0])

        # Return empty usage
        return UsageRecord(user_id=user_id, period=period)

    @staticmethod
    async def increment_usage(
        db: Client,
        user_id: str,
        storage_delta: int = 0,
        sync_count_delta: int = 0,
    ) -> UsageRecord:
        """Atomically increment usage counters for the current period.

        storage_delta can be negative (e.g. when data is deleted).
        sync_count_delta is for analytics only — no limits enforced.
        """
        import asyncio

        period = _current_period()
        now = _now()

        # Ensure record exists
        await SubscriptionService.get_usage(db, user_id)

        # Use RPC for atomic increment if available, otherwise fallback
        try:

            def _rpc():
                return db.rpc(
                    "increment_subscription_usage",
                    {
                        "p_user_id": user_id,
                        "p_period": period,
                        "p_storage_delta": storage_delta,
                        "p_sync_delta": sync_count_delta,
                    },
                ).execute()

            result = await asyncio.to_thread(_rpc)
            if result.data and len(result.data) > 0:
                return UsageRecord(**result.data[0])
        except Exception:
            logger.debug(
                "increment_subscription_usage RPC unavailable, using read-modify-write"
            )

        # Fallback: read-modify-write (has race window, acceptable for now)
        usage = await SubscriptionService.get_usage(db, user_id)
        new_storage = max(0, usage.storage_bytes + storage_delta)
        new_sync = usage.sync_count + sync_count_delta

        def _update():
            return (
                db.table(USAGE_RECORDS_TABLE)
                .update({
                    "storage_bytes": new_storage,
                    "sync_count": new_sync,
                    "updated_at": now.isoformat(),
                })
                .eq("user_id", user_id)
                .eq("period", period)
                .execute()
            )

        result = await asyncio.to_thread(_update)
        if result.data:
            return UsageRecord(**result.data[0])

        return UsageRecord(
            user_id=user_id,
            period=period,
            storage_bytes=new_storage,
            sync_count=new_sync,
        )

    @staticmethod
    async def update_stacks_syncing(
        db: Client,
        user_id: str,
        stacks_syncing: int,
    ) -> UsageRecord:
        """Set the current number of stacks actively syncing.

        Called when agents start/stop syncing to keep the count accurate.
        """
        import asyncio

        period = _current_period()
        now = _now()

        # Ensure record exists
        await SubscriptionService.get_usage(db, user_id)

        def _update():
            return (
                db.table(USAGE_RECORDS_TABLE)
                .update({
                    "stacks_syncing": stacks_syncing,
                    "updated_at": now.isoformat(),
                })
                .eq("user_id", user_id)
                .eq("period", period)
                .execute()
            )

        result = await asyncio.to_thread(_update)
        if result.data:
            return UsageRecord(**result.data[0])

        return UsageRecord(
            user_id=user_id,
            period=period,
            stacks_syncing=stacks_syncing,
        )

    # ------------------------------------------------------------------
    # Quota enforcement
    # ------------------------------------------------------------------

    @staticmethod
    async def check_quota(
        db: Client,
        user_id: str,
        operation: Literal["push", "create_stack"] = "push",
    ) -> QuotaCheckResult:
        """Check whether the user's current usage allows the operation.

        Quotas enforced:
        - storage_bytes vs tier storage_limit_bytes
        - stacks_syncing vs tier stack_limit

        NOT enforced (by design):
        - sync frequency — unlimited on all tiers
        """
        sub = await SubscriptionService.get_or_create_subscription(db, user_id)
        config = get_tier_config(sub.tier)
        usage = await SubscriptionService.get_usage(db, user_id)

        # Enterprise: unlimited
        if config.is_custom:
            return QuotaCheckResult(
                allowed=True,
                tier=sub.tier,
                storage_used=usage.storage_bytes,
                storage_limit=0,
                stacks_used=usage.stacks_syncing,
                stacks_limit=0,
            )

        result = QuotaCheckResult(
            allowed=True,
            tier=sub.tier,
            storage_used=usage.storage_bytes,
            storage_limit=config.storage_limit_bytes,
            stacks_used=usage.stacks_syncing,
            stacks_limit=config.stack_limit,
        )

        if operation == "push":
            # Check storage
            if usage.storage_bytes >= config.storage_limit_bytes:
                # On tiers with overflow pricing, allow but log
                if config.overflow:
                    logger.info(
                        "User %s exceeded storage (%d / %d) — overflow billing applies",
                        user_id,
                        usage.storage_bytes,
                        config.storage_limit_bytes,
                    )
                else:
                    result.allowed = False
                    result.reason = (
                        f"Storage limit reached ({usage.storage_bytes} / "
                        f"{config.storage_limit_bytes} bytes). "
                        "Upgrade tier or remove old data."
                    )
                    return result

        if operation == "create_stack":
            # Check stack count
            if usage.stacks_syncing >= config.stack_limit:
                if config.overflow:
                    logger.info(
                        "User %s exceeded stacks (%d / %d) — overflow billing applies",
                        user_id,
                        usage.stacks_syncing,
                        config.stack_limit,
                    )
                else:
                    result.allowed = False
                    result.reason = (
                        f"Stack limit reached ({usage.stacks_syncing} / "
                        f"{config.stack_limit}). "
                        "Upgrade tier for more stacks."
                    )
                    return result

        return result


# =============================================================================
# Module-level bridge functions (called by routes)
#
# These translate between the route layer's dict-based interface and the
# SubscriptionService's model-based static methods, and wire in on-chain
# payment verification via the payments module.
# =============================================================================

PAYMENT_INTENTS_TABLE = "payment_intents"


async def _check_tx_hash_used(db: Client, tx_hash: str, exclude_intent_id: str) -> bool:
    """Check if a tx_hash has already been used in payment_intents or subscription_payments."""
    import asyncio

    # Check payment_intents (confirmed or processing)
    def _check_intents():
        return (
            db.table(PAYMENT_INTENTS_TABLE)
            .select("id")
            .eq("tx_hash", tx_hash)
            .neq("id", exclude_intent_id)
            .in_("status", ["confirmed", "processing"])
            .limit(1)
            .execute()
        )

    result = await asyncio.to_thread(_check_intents)
    if result.data:
        return True

    # Check subscription_payments
    def _check_payments():
        return (
            db.table(SUBSCRIPTION_PAYMENTS_TABLE)
            .select("id")
            .eq("tx_hash", tx_hash)
            .limit(1)
            .execute()
        )

    result = await asyncio.to_thread(_check_payments)
    return bool(result.data)


async def get_subscription(db: Client, user_id: str) -> dict | None:
    """Get subscription as a dict for the route layer."""
    sub = await SubscriptionService.get_or_create_subscription(db, user_id)
    config = get_tier_config(sub.tier)
    return {
        "tier": sub.tier.value,
        "status": sub.status.value,
        "starts_at": sub.starts_at,
        "renews_at": sub.renews_at,
        "cancelled_at": sub.cancelled_at,
        "auto_renew": sub.auto_renew,
        "renewal_amount": str(config.price_usdc) if sub.auto_renew else None,
    }


async def get_current_usage(db: Client, user_id: str) -> dict | None:
    """Get current-period usage as a dict for the route layer."""
    sub = await SubscriptionService.get_or_create_subscription(db, user_id)
    config = get_tier_config(sub.tier)
    usage = await SubscriptionService.get_usage(db, user_id)
    return {
        "period": usage.period,
        "storage_used": usage.storage_bytes,
        "storage_limit": config.storage_limit_bytes,
        "sync_count": usage.sync_count,
        "last_sync": usage.updated_at,
        "agents_used": usage.stacks_syncing,
        "agents_limit": config.stack_limit,
    }


async def create_upgrade_payment(
    db: Client,
    user_id: str,
    target_tier: str,
) -> dict | None:
    """Create a pending payment intent for a tier upgrade.

    Returns payment instructions (payment_id, amount, treasury_address).
    The actual upgrade is applied when the payment is confirmed.
    """
    import asyncio
    import uuid

    tier = SubscriptionTier(target_tier)
    config = get_tier_config(tier)

    sub = await SubscriptionService.get_or_create_subscription(db, user_id)
    payment_id = str(uuid.uuid4())
    now = _now()

    # Store a payment intent — links the pending payment to the tier upgrade
    from datetime import timedelta
    expires_at = now + timedelta(hours=24)  # 24h to complete payment

    data = {
        "id": payment_id,
        "user_id": user_id,
        "subscription_id": sub.id,
        "tier": target_tier,
        "amount": str(config.price_usdc),
        "currency": "USDC",
        "treasury_address": TREASURY_ADDRESS,
        "chain": "base",
        "status": "pending",
        "expires_at": expires_at.isoformat(),
        "created_at": now.isoformat(),
    }

    def _insert():
        return db.table(PAYMENT_INTENTS_TABLE).insert(data).execute()

    try:
        result = await asyncio.to_thread(_insert)
        if not result.data:
            return None
    except Exception:
        logger.exception("Failed to create payment intent for user %s", user_id)
        return None

    return {
        "payment_id": payment_id,
        "amount": str(config.price_usdc),
        "currency": "USDC",
        "treasury_address": TREASURY_ADDRESS,
        "chain": "base",
    }


async def get_payment(
    db: Client,
    payment_id: str,
    user_id: str,
) -> dict | None:
    """Fetch a payment intent by ID, scoped to the user."""
    import asyncio

    def _query():
        return (
            db.table(PAYMENT_INTENTS_TABLE)
            .select("*")
            .eq("id", payment_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

    result = await asyncio.to_thread(_query)
    return result.data[0] if result.data else None


async def confirm_payment(
    db: Client,
    payment_id: str,
    tx_hash: str,
    user_id: str,
) -> dict | None:
    """Verify an on-chain USDC payment and activate the tier upgrade.

    Flow:
    1. Look up payment intent
    2. Verify the transfer on-chain via verify_usdc_transfer
    3. If valid: mark confirmed, apply tier upgrade, record payment
    4. If invalid: mark failed with error details
    """
    import asyncio

    from ..payments import verify_usdc_transfer

    # 1. Fetch payment intent
    intent = await get_payment(db, payment_id, user_id)
    if not intent:
        return None

    if intent.get("status") == "confirmed":
        return {
            "status": "confirmed",
            "tier": intent.get("tier"),
            "message": "Payment was already confirmed.",
        }

    # Reject non-pending intents (expired, cancelled, failed)
    if intent.get("status") != "pending":
        return {
            "status": "failed",
            "message": f"Payment intent is '{intent.get('status')}', expected 'pending'.",
        }

    # Check expiry (P0 fix: enforce expires_at)
    expires_at = intent.get("expires_at")
    if expires_at:
        from datetime import datetime as dt
        if isinstance(expires_at, str):
            # Parse ISO format, handle timezone
            expiry = dt.fromisoformat(expires_at.replace("Z", "+00:00"))
        else:
            expiry = expires_at
        if _now() > expiry:
            # Mark as expired
            def _expire():
                return (
                    db.table(PAYMENT_INTENTS_TABLE)
                    .update({"status": "expired", "updated_at": _now().isoformat()})
                    .eq("id", payment_id)
                    .execute()
                )
            await asyncio.to_thread(_expire)
            return {
                "status": "failed",
                "message": "Payment intent has expired. Please create a new upgrade request.",
            }

    # Validate tx_hash format (P0 fix: prevent injection)
    import re
    if not re.match(r'^0x[0-9a-fA-F]{64}$', tx_hash):
        return {
            "status": "failed",
            "message": "Invalid transaction hash format. Expected 0x + 64 hex characters.",
        }

    # Check tx_hash hasn't been used for another payment (P0 fix: prevent replay)
    existing_tx = await _check_tx_hash_used(db, tx_hash, payment_id)
    if existing_tx:
        return {
            "status": "failed",
            "message": "This transaction hash has already been used for another payment.",
        }

    expected_amount = Decimal(intent["amount"])
    chain = intent.get("chain", "base")
    now = _now()

    # Atomic claim: set status to 'processing' to prevent double-confirm race
    # Only one concurrent request can claim a pending intent
    def _claim():
        return (
            db.table(PAYMENT_INTENTS_TABLE)
            .update({"status": "processing", "tx_hash": tx_hash, "updated_at": now.isoformat()})
            .eq("id", payment_id)
            .eq("status", "pending")  # atomic: only succeeds if still pending
            .execute()
        )

    claim_result = await asyncio.to_thread(_claim)
    if not claim_result.data:
        # Another request already claimed this intent
        return {
            "status": "failed",
            "message": "Payment is already being processed by another request.",
        }

    # 2. On-chain verification
    try:
        result = await verify_usdc_transfer(
            tx_hash=tx_hash,
            expected_amount=expected_amount,
            expected_to=TREASURY_ADDRESS,
            chain=chain,
            tolerance=Decimal("0"),  # P0 fix: exact amount required (no underpayment)
        )
    except Exception as e:
        logger.exception("On-chain verification error for payment %s", payment_id)
        # Mark intent as needing retry, don't fail permanently on network errors
        def _mark_error():
            return (
                db.table(PAYMENT_INTENTS_TABLE)
                .update({
                    "status": "verification_error",
                    "tx_hash": tx_hash,
                    "error": str(e),
                    "updated_at": now.isoformat(),
                })
                .eq("id", payment_id)
                .execute()
            )

        await asyncio.to_thread(_mark_error)
        return {
            "status": "pending_verification",
            "message": f"Verification temporarily unavailable: {e}. Will retry.",
        }

    if not result.success:
        # 3a. Verification failed — mark payment as failed
        def _fail():
            return (
                db.table(PAYMENT_INTENTS_TABLE)
                .update({
                    "status": "failed",
                    "tx_hash": tx_hash,
                    "error": result.error,
                    "error_code": result.error_code,
                    "updated_at": now.isoformat(),
                })
                .eq("id", payment_id)
                .execute()
            )

        await asyncio.to_thread(_fail)
        logger.warning(
            "Payment verification failed for %s: %s (%s)",
            payment_id,
            result.error,
            result.error_code,
        )
        return {
            "status": "failed",
            "message": f"Payment verification failed: {result.error}",
        }

    # 3b. Verification succeeded — confirm and activate
    target_tier = intent.get("tier")

    # IMPORTANT: Record payment FIRST, then upgrade tier (P0 fix: order matters)
    # If record_payment fails (e.g. duplicate tx_hash), we must not upgrade the tier.
    sub = await SubscriptionService.get_or_create_subscription(db, user_id)

    try:
        # Record payment — has UNIQUE constraint on tx_hash, prevents replay
        await SubscriptionService.record_payment(
            db=db,
            subscription_id=sub.id,
            tx_hash=tx_hash,
            amount=expected_amount,
            from_address=result.from_address or "",
            to_address=TREASURY_ADDRESS,
            chain=chain,
            user_id=user_id,  # P0 fix: include user_id for audit trail
        )
        # Mark payment as confirmed
        await SubscriptionService.confirm_payment(db, tx_hash)
    except Exception as e:
        # Payment recording failed (likely duplicate tx_hash) — roll back intent
        logger.warning("Payment recording failed for %s: %s", payment_id, e)
        def _rollback():
            return (
                db.table(PAYMENT_INTENTS_TABLE)
                .update({"status": "failed", "error": str(e), "updated_at": now.isoformat()})
                .eq("id", payment_id)
                .execute()
            )
        await asyncio.to_thread(_rollback)
        return {
            "status": "failed",
            "message": f"Payment recording failed: {e}. Tier not upgraded.",
        }

    # Payment recorded successfully — NOW upgrade the tier
    try:
        upgraded_sub, _ = await SubscriptionService.upgrade_tier(
            db, user_id, SubscriptionTier(target_tier)
        )
    except ValueError as e:
        # Edge case: tier already upgraded (e.g. duplicate confirm)
        logger.warning("Tier upgrade skipped for %s: %s", user_id, e)
        upgraded_sub = sub

    # Update payment intent to confirmed
    def _confirm_intent():
        return (
            db.table(PAYMENT_INTENTS_TABLE)
            .update({
                "status": "confirmed",
                "confirmed_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "from_address": result.from_address,
                "block_number": result.block_number,
                "confirmations": result.confirmations,
            })
            .eq("id", payment_id)
            .execute()
        )

    await asyncio.to_thread(_confirm_intent)

    logger.info(
        "Payment %s confirmed — user %s upgraded to %s (tx=%s, block=%s, %d confirmations)",
        payment_id,
        user_id,
        target_tier,
        tx_hash,
        result.block_number,
        result.confirmations or 0,
    )

    return {
        "status": "confirmed",
        "tier": target_tier,
        "activated_at": now.isoformat(),
        "message": f"Payment confirmed! Your subscription has been upgraded to {target_tier}.",
    }


async def schedule_downgrade(
    db: Client,
    user_id: str,
    target_tier: str,
) -> dict | None:
    """Schedule a downgrade effective at end of billing period."""
    try:
        sub, effective_at = await SubscriptionService.downgrade_tier(
            db, user_id, SubscriptionTier(target_tier)
        )
        return {"effective_at": effective_at}
    except ValueError:
        return None


async def cancel_subscription(db: Client, user_id: str) -> dict | None:
    """Cancel auto-renewal."""
    try:
        sub = await SubscriptionService.cancel_subscription(db, user_id)
        return {
            "cancelled_at": sub.cancelled_at or _now(),
            "active_until": sub.renews_at or _now(),
        }
    except ValueError:
        return None


async def reactivate_subscription(db: Client, user_id: str) -> dict | None:
    """Re-enable auto-renewal for a cancelled subscription."""
    import asyncio

    sub = await SubscriptionService.get_or_create_subscription(db, user_id)

    if sub.status != SubscriptionStatus.cancelled and sub.cancelled_at is None:
        return None

    now = _now()
    update = {
        "auto_renew": True,
        "cancelled_at": None,
        "status": SubscriptionStatus.active.value,
        "updated_at": now.isoformat(),
    }

    def _update():
        return (
            db.table(SUBSCRIPTIONS_TABLE)
            .update(update)
            .eq("id", sub.id)
            .execute()
        )

    result = await asyncio.to_thread(_update)
    if not result.data:
        return None

    logger.info("Subscription reactivated for user %s", user_id)
    return {
        "renews_at": sub.renews_at,
    }


async def list_payments(
    db: Client,
    user_id: str,
    page: int = 1,
    per_page: int = 20,
) -> dict:
    """List payment history with pagination."""
    import asyncio

    # Get user's subscription(s) to find their payment records
    sub = await SubscriptionService.get_or_create_subscription(db, user_id)
    offset = (page - 1) * per_page

    def _query():
        return (
            db.table(SUBSCRIPTION_PAYMENTS_TABLE)
            .select("*", count="exact")
            .eq("subscription_id", sub.id)
            .order("created_at", desc=True)
            .range(offset, offset + per_page - 1)
            .execute()
        )

    result = await asyncio.to_thread(_query)

    payments = []
    for row in result.data or []:
        payments.append({
            "id": row.get("id", ""),
            "amount": str(row.get("amount", "0")),
            "currency": row.get("currency", "USDC"),
            "tx_hash": row.get("tx_hash"),
            "from_address": row.get("from_address"),
            "to_address": row.get("to_address"),
            "chain": row.get("chain", "base"),
            "status": row.get("status", "pending"),
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
            "created_at": row.get("created_at"),
            "confirmed_at": row.get("confirmed_at"),
        })

    return {
        "payments": payments,
        "total": result.count or len(payments),
    }
